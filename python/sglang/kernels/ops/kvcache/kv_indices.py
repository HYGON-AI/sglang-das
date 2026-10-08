import triton
import triton.language as tl

_FLASHMLA_CREATE_KV_BLOCK_SIZE = 4096
FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON = tl.constexpr(_FLASHMLA_CREATE_KV_BLOCK_SIZE)


@triton.jit
def create_draft_extend_kv_metadata_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,  # [bs]
    seq_lens_ptr,  # [bs], post draft-extend lengths
    extend_seq_lens_ptr,  # [bs]
    kv_indptr_ptr,  # [bs + 1]
    qo_indptr_ptr,  # [bs + 1]
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
    BS_BLOCK: tl.constexpr,
):
    """Build all Triton draft-extend ragged metadata in one launch.

    Draft extend has a fixed query width for a captured graph bucket.  The
    prefix length is ``seq_len - extend_len``.  Computing its small exclusive
    scan in each request program lets the same launch also materialize the KV
    indices, replacing the former arange + cumsum + KV-index launch chain.
    """
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    batch_offs = tl.arange(0, BS_BLOCK)
    batch_mask = batch_offs < tl.num_programs(axis=0)
    seq_lens = tl.load(seq_lens_ptr + batch_offs, mask=batch_mask, other=0).to(tl.int32)
    extend_lens = tl.load(
        extend_seq_lens_ptr + batch_offs, mask=batch_mask, other=0
    ).to(tl.int32)
    kv_lens = tl.maximum(seq_lens - extend_lens, 0)

    kv_start = tl.sum(tl.where(batch_offs < pid, kv_lens, 0), axis=0)
    kv_len = tl.sum(tl.where(batch_offs == pid, kv_lens, 0), axis=0)
    tl.store(kv_indptr_ptr + pid, kv_start)
    tl.store(qo_indptr_ptr + pid, pid * NUM_TOKENS_PER_REQ)
    if pid == tl.num_programs(axis=0) - 1:
        tl.store(kv_indptr_ptr + pid + 1, kv_start + kv_len)
        tl.store(qo_indptr_ptr + pid + 1, (pid + 1) * NUM_TOKENS_PER_REQ)

    req_pool_index = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
    num_loop = tl.cdiv(kv_len, BLOCK_SIZE)
    for i in range(num_loop):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_len
        values = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offsets.to(tl.int64),
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_start + offsets, values, mask=mask)


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        # index into req_to_token_ptr needs to be int64
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


@triton.jit
def create_chunked_prefix_cache_kv_indices(
    req_to_token_ptr,  # (max_batch, max_context_len,)
    req_pool_indices_ptr,  # (batch_size,)
    chunk_start_idx_ptr,  # (batch_size,)
    chunk_seq_lens_ptr,  # (batch_size,)
    chunk_cu_seq_lens_ptr,  # (batch_size + 1,)
    chunk_kv_indices_ptr,  # (num_chunk_tokens,)
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    chunk_kv_indices_offset = tl.load(chunk_cu_seq_lens_ptr + pid)

    # get the token positions of current chunk
    chunk_start_pos = tl.load(chunk_start_idx_ptr + pid).to(tl.int32)
    chunk_seq_len = tl.load(chunk_seq_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(chunk_seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < chunk_seq_len
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + chunk_start_pos
            + offset,
            mask=mask,
        )
        tl.store(
            chunk_kv_indices_ptr + chunk_kv_indices_offset + offset, data, mask=mask
        )


def get_num_page_per_block_flashmla(page_size: int = 64) -> int:
    num_page_per_block = _FLASHMLA_CREATE_KV_BLOCK_SIZE // page_size
    return num_page_per_block


def get_num_kv_index_blocks_flashmla(kv_indices_width: int, page_size: int) -> int:
    """Grid axis-1 size for create_flashmla_kv_indices_triton: the number of
    page-blocks spanning the widest sequence (one CTA per block). kv_indices_width
    is the per-row width of the kv_indices buffer (the kernel's kv_indices_ptr_stride).
    """
    npb = get_num_page_per_block_flashmla(page_size)
    return (kv_indices_width + npb - 1) // npb


@triton.jit
def create_flashmla_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    kv_indices_ptr_stride: tl.constexpr,
    PAGED_SIZE: tl.constexpr = 64,
    # Unified-memory dense-view path (page-major envelope shared with the mamba
    # sub-pool). req_to_token holds VIRTUAL token ids; the block table the MLA
    # kernel consumes must hold DENSE page ids. When v2p_ptr is given, map each
    # virtual page through it to the physical page, then scale by PAGE_MULT
    # (= num MLA layers) so the entry addresses the layer's dense per-page block
    # in the (num_pages*L, page_size, kv_cache_dim) reshaped view. Both default
    # to the identity (v2p_ptr None, PAGE_MULT 1) for the static pool.
    v2p_ptr=None,
    PAGE_MULT: tl.constexpr = 1,
):
    NUM_PAGE_PER_BLOCK: tl.constexpr = (
        FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON // PAGED_SIZE
    )
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start

    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)
    num_pages_loop = tl.cdiv(kv_end - kv_start, FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON)

    # One CTA per page-block (grid axis 1) rather than one CTA looping all blocks;
    # CTAs beyond this sequence's block count are guarded out.
    i = tl.program_id(axis=1)
    if i < num_pages_loop:
        # index into req_to_token_ptr needs to be int64
        paged_offset = (
            tl.arange(0, NUM_PAGE_PER_BLOCK).to(tl.int64) + i * NUM_PAGE_PER_BLOCK
        ) * PAGED_SIZE
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK

        mask = paged_offset < num_paged * PAGED_SIZE
        mask_out = paged_offset_out < num_paged

        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + paged_offset,
            mask=mask,
        )
        page = data // PAGED_SIZE
        if v2p_ptr is not None:
            # virtual page -> physical page (page-level v2p); masked so padded
            # lanes never index the table out of bounds.
            page = tl.load(v2p_ptr + page, mask=mask_out, other=0)
        tl.store(
            kv_indices_ptr + pid * kv_indices_ptr_stride + paged_offset_out,
            page * PAGE_MULT,
            mask=mask_out,
        )
