# Official Main Catch-up 20260629 — Code Conflict Review

> Scope: only the six Python files that produced textual merge conflicts. The conflict ledger and all automatically merged files are intentionally excluded.
> View in VS Code with **Markdown: Open Preview** (`Ctrl+Shift+V`). The `diff` blocks render removed conflict state in red and the final resolved code in green.

## Comparison

- DCU parent (`ours`): `b97ca20827da8b9eed8db0cbe0b128f33ccc7aee`
- Common official base: `eeee3abbbf8196e54c227faecfd5faba7b1dfc4b`
- Official endpoint (`theirs`): `f920a37da46e1cbb6ba27b76365a622eba593811`
- Resolved merge: `18f04cd62152c0ee92a501e7338632d53903be7b`
- Reconstructed textual conflicts: 6 files, 26 hunks

Each section reconstructs Git’s three-way auto-conflict text from the two merge parents and common base, then compares it with the committed resolution. Lines beginning with `-` belong to the unresolved auto-conflict state; lines beginning with `+` are the final resolution.

## Conflict files

<details>
<summary><code>python/sglang/srt/layers/attention/dsa/dsa_indexer.py</code> — 13 conflict hunk(s)</summary>

**Resolution intent:** Adopt official CUDA fusion and graph split APIs; preserve the DCU-first LightOp, BF16/FP8 index-cache, page-size-64, and logits paths.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
+++ RESOLVED/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
@@ -589,30 +589,20 @@
         self,
         q_lora: torch.Tensor,
         x: torch.Tensor,
         positions: torch.Tensor,
         enable_dual_stream: bool,
         forward_batch: ForwardBatch,
         apply_hadamard_scale: bool = True,
     ):
-<<<<<<< DCU main@b97ca2082
+        weights_raw = None
         if _is_dcu:
             query, _ = self.wq_b(q_lora)
             query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
-||||||| official previous@eeee3abbbf81
-        if enable_dual_stream:
-            current_stream = torch.cuda.current_stream()
-            self.alt_stream.wait_stream(current_stream)
-=======
-        weights_raw = None
-        if enable_dual_stream:
-            current_stream = torch.cuda.current_stream()
-            self.alt_stream.wait_stream(current_stream)
->>>>>>> official target@f920a37da46e
 
             key, _ = self.wk(x)
 
             if key.ndim == 2:
                 key = key.view(key.shape[0], -1, self.head_dim)
 
             op.fuse_layernorm_rotary_embedding(
                 positions,
@@ -640,17 +630,21 @@
                     query, _ = self.wq_b(q_lora)
                     query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                     q_rope, _ = torch.split(
                         query,
                         [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                         dim=-1,
                     )
                 with torch.cuda.stream(self.alt_stream):
-                    key, _ = self.wk(x)
+                    # TODO we should also put DeepGEMM half SM here?
+                    if _use_dsa_indexer_fusion:
+                        key, weights_raw = self._fused_k_weights(x)
+                    else:
+                        key, _ = self.wk(x)
                     key = self.k_norm(key)
 
                     k_rope, _ = torch.split(
                         key,
                         [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                         dim=-1,
                     )
 
@@ -658,140 +652,82 @@
             else:
                 query, _ = self.wq_b(q_lora)
                 query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                 q_rope, _ = torch.split(
                     query,
                     [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                     dim=-1,
                 )
-<<<<<<< DCU main@b97ca2082
-                key, _ = self.wk(x)
-||||||| official previous@eeee3abbbf81
-            with torch.cuda.stream(self.alt_stream):
-                # TODO we should also put DeepGEMM half SM here?
-                key, _ = self.wk(x)
-=======
-            with torch.cuda.stream(self.alt_stream):
-                # TODO we should also put DeepGEMM half SM here?
                 if _use_dsa_indexer_fusion:
                     key, weights_raw = self._fused_k_weights(x)
                 else:
                     key, _ = self.wk(x)
->>>>>>> official target@f920a37da46e
                 key = self.k_norm(key)
                 k_rope, _ = torch.split(
                     key,
                     [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                     dim=-1,
                 )
 
-<<<<<<< DCU main@b97ca2082
             q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
-||||||| official previous@eeee3abbbf81
-            current_stream.wait_stream(self.alt_stream)
-        else:
-            query, _ = self.wq_b(q_lora)
-            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
-            q_rope, _ = torch.split(
-                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
-            )
-            key, _ = self.wk(x)
-            key = self.k_norm(key)
-            k_rope, _ = torch.split(
-                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
-            )
-
-        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
-=======
-            current_stream.wait_stream(self.alt_stream)
-        else:
-            query, _ = self.wq_b(q_lora)
-            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
-            q_rope, _ = torch.split(
-                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
-            )
-            if _use_dsa_indexer_fusion:
-                key, weights_raw = self._fused_k_weights(x)
-            else:
-                key, _ = self.wk(x)
-            key = self.k_norm(key)
-            k_rope, _ = torch.split(
-                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
-            )
-
-        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
->>>>>>> official target@f920a37da46e
 
             self._update_rope_guarded(query[..., : self.rope_head_dim], q_rope)
             self._update_rope_guarded(key[..., : self.rope_head_dim], k_rope)
 
         if enable_dual_stream:
             current_stream = torch.cuda.current_stream()
             self.alt_stream.wait_stream(current_stream)
-<<<<<<< DCU main@b97ca2082
-
-            query = rotate_activation(query, apply_scale=apply_hadamard_scale)
-||||||| official previous@eeee3abbbf81
-            query = rotate_activation(query)
-=======
-            query = self._maybe_rotate(query)
->>>>>>> official target@f920a37da46e
+            query = (
+                rotate_activation(query, apply_scale=apply_hadamard_scale)
+                if _is_dcu
+                else self._maybe_rotate(query)
+            )
 
             with torch.cuda.stream(self.alt_stream):
-<<<<<<< DCU main@b97ca2082
-                key = rotate_activation(key, apply_scale=apply_hadamard_scale)
-||||||| official previous@eeee3abbbf81
-                key = rotate_activation(key)
-=======
-                key = self._maybe_rotate(key)
->>>>>>> official target@f920a37da46e
+                key = (
+                    rotate_activation(key, apply_scale=apply_hadamard_scale)
+                    if _is_dcu
+                    else self._maybe_rotate(key)
+                )
             current_stream.wait_stream(self.alt_stream)
         elif (
             self.alt_stream is not None
             and forward_batch.attn_cp_metadata is not None
             and self.dsa_enable_prefill_cp
         ):
-<<<<<<< DCU main@b97ca2082
-            key = rotate_activation(key, apply_scale=apply_hadamard_scale)
-||||||| official previous@eeee3abbbf81
-            key = rotate_activation(key)
-=======
-            key = self._maybe_rotate(key)
->>>>>>> official target@f920a37da46e
+            key = (
+                rotate_activation(key, apply_scale=apply_hadamard_scale)
+                if _is_dcu
+                else self._maybe_rotate(key)
+            )
             current_stream = torch.cuda.current_stream()
             self.alt_stream.wait_stream(current_stream)
-<<<<<<< DCU main@b97ca2082
-            query = rotate_activation(query, apply_scale=apply_hadamard_scale)
-||||||| official previous@eeee3abbbf81
-            query = rotate_activation(query)
-=======
-            query = self._maybe_rotate(query)
->>>>>>> official target@f920a37da46e
+            query = (
+                rotate_activation(query, apply_scale=apply_hadamard_scale)
+                if _is_dcu
+                else self._maybe_rotate(query)
+            )
 
             with torch.cuda.stream(self.alt_stream):
                 key = cp_all_gather_rerange_output(
                     key.contiguous(),
                     self.cp_size,
                     forward_batch,
                     torch.cuda.current_stream(),
                 )
             current_stream.wait_stream(self.alt_stream)
             return query, key, weights_raw
         else:
-<<<<<<< DCU main@b97ca2082
-            query = rotate_activation(query, apply_scale=apply_hadamard_scale)
-            key = rotate_activation(key, apply_scale=apply_hadamard_scale)
-||||||| official previous@eeee3abbbf81
-            query = rotate_activation(query)
-            key = rotate_activation(key)
-=======
-            query = self._maybe_rotate(query)
-            key = self._maybe_rotate(key)
->>>>>>> official target@f920a37da46e
+            if _is_dcu:
+                query = rotate_activation(query, apply_scale=apply_hadamard_scale)
+                key = rotate_activation(key, apply_scale=apply_hadamard_scale)
+            else:
+                query = self._maybe_rotate(query)
+                key = self._maybe_rotate(key)
 
         # allgather+rerrange
         if forward_batch.attn_cp_metadata is not None and self.dsa_enable_prefill_cp:
             key = cp_all_gather_rerange_output(
                 key.contiguous(),
                 self.cp_size,
                 forward_batch,
                 torch.cuda.current_stream(),
@@ -1458,52 +1394,25 @@
         # keyword args carry the graph contract and default to the eager behavior:
         #   - num_tokens: slice key/out_cache_loc to the unpadded count (the graph
         #     runs at a static padded shape). None => full (eager) shape.
         #   - topk_result: pre-allocated padded buffer to fill in place (a downstream
         #     captured graph reads it at a fixed address). None => return a fresh,
         #     naturally-sized tensor.
         assert forward_batch.forward_mode.is_extend_without_speculative()
         x_meta = x[0] if isinstance(x, tuple) else x
-<<<<<<< DCU main@b97ca2082
-        # Fast path: only compute and store k cache, skip all q and weights ops
-        key = self._get_k_bf16(x, positions, enable_dual_stream)
-||||||| official previous@eeee3abbbf81
-
-        # Fast path: only compute and store k cache, skip all q and weights ops
-        key = self._get_k_bf16(x, positions, enable_dual_stream)
-=======
 
         # Fast path: only compute and store k cache, skip all q and weights ops.
         # num_tokens (graph contract) slices to the unpadded count.
->>>>>>> official target@f920a37da46e
         out_cache_loc = None
         if num_tokens is not None:
             assert num_tokens <= forward_batch.out_cache_loc.shape[0]
             out_cache_loc = forward_batch.out_cache_loc[:num_tokens]
         elif not forward_batch.out_cache_loc.is_contiguous():
             forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
-<<<<<<< DCU main@b97ca2082
-        self._store_index_k_cache(
-            forward_batch=forward_batch,
-            layer_id=layer_id,
-            key=key,
-            act_quant=act_quant,
-            out_cache_loc=out_cache_loc,
-        )
-||||||| official previous@eeee3abbbf81
-        self._store_index_k_cache(
-            forward_batch=forward_batch,
-            layer_id=layer_id,
-            key=key,
-            act_quant=act_quant,
-            out_cache_loc=out_cache_loc,
-        )
-
-=======
 
         # Write the same K representation the decode path reads back: fused
         # (no-Hadamard) when fusion is on, else the legacy Hadamard path.
         if _use_dsa_indexer_fusion:
             key_raw, _ = self._fused_k_weights(x)
             if num_tokens is not None:
                 assert num_tokens <= key_raw.shape[0]
                 key_raw = key_raw[:num_tokens]
@@ -1524,17 +1433,16 @@
             self._store_index_k_cache(
                 forward_batch=forward_batch,
                 layer_id=layer_id,
                 key=key,
                 act_quant=act_quant,
                 out_cache_loc=out_cache_loc,
             )
 
->>>>>>> official target@f920a37da46e
         # MHA doesn't need topk_indices
         if not return_indices:
             return None
 
         # MLA: use dummy logits with topk kernel's fast path to generate indices
         # When length <= 2048, naive_topk_cuda directly generates [0,1,...,length-1,-1,...]
         seq_lens_expanded = metadata.get_seqlens_expanded()
         dummy_logits = torch.zeros(
@@ -2043,16 +1951,17 @@
         if (
             _use_dsa_indexer_fusion
             and not in_piecewise_or_breakable_cuda_graph
             and forward_batch.attn_cp_metadata is None
         ):
             q_fp8, weights = self._fused_q_prepare_and_store(
                 x, q_lora, positions, forward_batch, layer_id, act_quant
             )
+            q_index = q_fp8
         elif (
             is_graph_dsa_split_op_surface(forward_batch)
             and not self.dsa_enable_prefill_cp
         ):
             # Default path for non-CP prefill under PCG/BCG: run the whole indexer
             # (q/k proj, head gate, k-cache store, topk) as a single eager split op
             # instead of capturing it piecemeal in the graph. The split op is
             # fusion-aware, so this also covers the fused path here.
@@ -2081,274 +1990,190 @@
                 positions=positions,
                 topk_result=topk_result,
             )
             result = _broadcast_indexer_topk_from_rank0(
                 topk_result if return_indices else None
             )
             return maybe_capture_indexer_topk(layer_id, result)
 
-        elif enable_dual_stream and forward_batch.forward_mode.is_decode_or_idle():
-            current_stream = torch.cuda.current_stream()
-            self.alt_stream.wait_stream(current_stream)
-<<<<<<< DCU main@b97ca2082
-            if _is_dcu:
-                if self._use_dcu_bf16_index_cache(forward_batch):
-                    q_index, key = self._get_q_k_bf16(
-                        q_lora, x, positions, False, forward_batch=forward_batch
-                    )
-                    get_token_to_kv_pool().set_index_k_buffer(
-                        layer_id=layer_id,
-                        loc=forward_batch.out_cache_loc,
-                        index_k=key,
-                    )
-                    weights = self._get_bf16_logits_head_gate(x)
-                else:
-                    weights = self._project_and_scale_head_gates(x)
-                    query, key = self._get_q_k_bf16(
-                        q_lora, x, positions, False, forward_batch=forward_batch,
-                        apply_hadamard_scale=False
-                    )
-                    k_buf = get_token_to_kv_pool().get_index_k_with_scale_buffer(layer_id=layer_id)
-                    k_loc = forward_batch.out_cache_loc
-                    page_size = get_token_to_kv_pool().page_size
-                    is_e4m3 = not _is_fp8_fnuz
-
-                    hadamard_scale = self.hidden_size ** -0.5
-                    fused_q_scale = hadamard_scale * self.softmax_scale
-                    fused_k_scale = hadamard_scale
-
-                    q_fp8, q_scale, weights = op.fuse_qk_quant_and_store_index_k_cache(
-                            query,
-                            key,
-                            k_buf,
-                            k_loc,
-                            page_size,
-                            weights,           # weights_in_opt
-                            fused_q_scale,     # q_scale_factor
-                            fused_k_scale,     # k_scale_factor
-                            1e-5,              # eps
-                            False,             # use_ue8m0
-                            is_e4m3            # is_e4m3
-                        )
-                    q_index = q_fp8.view(torch.float8_e4m3fnuz) if _is_fp8_fnuz else q_fp8.view(torch.float8_e4m3fn)
-            else:
-                if weights_proj_lora:
-                    weights = self.weights_proj(x)[0].float() * self.n_heads**-0.5
-                else:
-                    weights = self._project_and_scale_head_gates(x)
-                query, key = self._get_q_k_bf16(
-                    q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
-||||||| official previous@eeee3abbbf81
-            if weights_proj_lora:
-                weights = self.weights_proj(x)[0].float() * self.n_heads**-0.5
-            else:
-                weights = self._project_and_scale_head_gates(x)
-            query, key = self._get_q_k_bf16(
-                q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
+        elif _is_dcu:
+            decode_dual_stream = (
+                enable_dual_stream
+                and forward_batch.forward_mode.is_decode_or_idle()
             )
-            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
-            with torch.cuda.stream(self.alt_stream):
-                self._store_index_k_cache(
+            if decode_dual_stream:
+                current_stream = torch.cuda.current_stream()
+                self.alt_stream.wait_stream(current_stream)
+
+            use_bf16_index_cache = self._use_dcu_bf16_index_cache(forward_batch)
+            if use_bf16_index_cache:
+                q_index, key, _ = self._get_q_k_bf16(
+                    q_lora,
+                    x,
+                    positions,
+                    False,
                     forward_batch=forward_batch,
+                )
+                get_token_to_kv_pool().set_index_k_buffer(
                     layer_id=layer_id,
-                    key=key,
-                    act_quant=act_quant,
-=======
+                    loc=forward_batch.out_cache_loc,
+                    index_k=key,
+                )
+            else:
+                fused_weights_in = (
+                    self._project_and_scale_head_gates(x)
+                    if decode_dual_stream
+                    else None
+                )
+                query, key, _ = self._get_q_k_bf16(
+                    q_lora,
+                    x,
+                    positions,
+                    False,
+                    forward_batch=forward_batch,
+                    apply_hadamard_scale=False,
+                )
+                pool = get_token_to_kv_pool()
+                hadamard_scale = self.hidden_size**-0.5
+                q_fp8, q_scale, fused_weights = (
+                    op.fuse_qk_quant_and_store_index_k_cache(
+                        query,
+                        key,
+                        pool.get_index_k_with_scale_buffer(layer_id=layer_id),
+                        forward_batch.out_cache_loc,
+                        pool.page_size,
+                        fused_weights_in,
+                        hadamard_scale * self.softmax_scale,
+                        hadamard_scale,
+                        1e-5,
+                        False,
+                        not _is_fp8_fnuz,
+                    )
+                )
+                q_index = q_fp8.view(
+                    torch.float8_e4m3fnuz if _is_fp8_fnuz else torch.float8_e4m3fn
+                )
+
+            x_for_gate = self._get_gate_input_tensor(x)
+            if use_bf16_index_cache:
+                if decode_dual_stream:
+                    weights = self._get_bf16_logits_head_gate(x)
+                else:
+                    weights = (
+                        self._project_and_scale_head_gates(x_for_gate).unsqueeze(-1)
+                        * self.softmax_scale
+                    )
+            elif decode_dual_stream:
+                weights = fused_weights
+            elif weights_proj_lora:
+                weights = self.weights_proj(x_for_gate)[0].float() * self.n_heads**-0.5
+                weights = self._apply_q_scale_and_softmax_scale(weights, q_scale)
+            else:
+                weights = self._get_logits_head_gate(x_for_gate, q_scale)
+
+        elif enable_dual_stream and forward_batch.forward_mode.is_decode_or_idle():
+            current_stream = torch.cuda.current_stream()
+            self.alt_stream.wait_stream(current_stream)
             if not _use_dsa_indexer_fusion:
                 if weights_proj_lora:
                     weights = self.weights_proj(x)[0].float() * self.n_heads**-0.5
                 else:
                     weights = self._project_and_scale_head_gates(x)
             query, key, weights_raw = self._get_q_k_bf16(
                 q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
             )
             q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
             with torch.cuda.stream(self.alt_stream):
                 self._store_index_k_cache(
                     forward_batch=forward_batch,
                     layer_id=layer_id,
                     key=key,
                     act_quant=act_quant,
->>>>>>> official target@f920a37da46e
                 )
-<<<<<<< DCU main@b97ca2082
-||||||| official previous@eeee3abbbf81
-            current_stream.wait_stream(self.alt_stream)
-            weights = self._apply_q_scale_and_softmax_scale(weights, q_scale)
-        else:
-            query, key = self._get_q_k_bf16(
-                q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
-            )
-
-            if enable_dual_stream:
-                current_stream = torch.cuda.current_stream()
-                self.alt_stream.wait_stream(current_stream)
-
-=======
             current_stream.wait_stream(self.alt_stream)
             if _use_dsa_indexer_fusion:
                 weights = self._scale_head_gates(weights_raw, q_scale)
             else:
                 weights = self._apply_q_scale_and_softmax_scale(weights, q_scale)
+            q_index = q_fp8
         else:
             query, key, weights_raw = self._get_q_k_bf16(
                 q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
             )
 
             if enable_dual_stream:
                 current_stream = torch.cuda.current_stream()
                 self.alt_stream.wait_stream(current_stream)
 
->>>>>>> official target@f920a37da46e
                 q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                 with torch.cuda.stream(self.alt_stream):
                     self._store_index_k_cache(
                         forward_batch=forward_batch,
                         layer_id=layer_id,
                         key=key,
                         act_quant=act_quant,
                     )
                 current_stream.wait_stream(self.alt_stream)
-                weights = self._apply_q_scale_and_softmax_scale(weights, q_scale)
-                q_index = q_fp8
-        else:
-            if _is_dcu:
-                if self._use_dcu_bf16_index_cache(forward_batch):
-                    q_index, key = self._get_q_k_bf16(
-                        q_lora,
-                        x,
-                        positions,
-                        enable_dual_stream if not _is_dcu else False,
-                        forward_batch=forward_batch,
-                    )
-                    get_token_to_kv_pool().set_index_k_buffer(
-                        layer_id=layer_id,
-                        loc=forward_batch.out_cache_loc,
-                        index_k=key,
-                    )
-                else:
-                    query, key = self._get_q_k_bf16(
-                        q_lora,
-                        x,
-                        positions,
-                        enable_dual_stream if not _is_dcu else False,
-                        forward_batch=forward_batch,
-                        apply_hadamard_scale=False,
-                    )
-                    k_buf = (
-                        get_token_to_kv_pool().get_index_k_with_scale_buffer(
-                            layer_id=layer_id
-                        )
-                    )
-                    k_loc = forward_batch.out_cache_loc
-                    page_size = get_token_to_kv_pool().page_size
-                    is_e4m3 = not _is_fp8_fnuz
-
-                    hadamard_scale = self.hidden_size ** -0.5
-                    fused_q_scale = hadamard_scale * self.softmax_scale
-                    fused_k_scale = hadamard_scale
-
-                    q_fp8, q_scale, _ = op.fuse_qk_quant_and_store_index_k_cache(
-                        query,
-                        key,
-                        k_buf,
-                        k_loc,
-                        page_size,
-                        None,              # weights_in_opt=None
-                        fused_q_scale,     # q_scale_factor
-                        fused_k_scale,     # k_scale_factor
-                        1e-5,              # eps
-                        False,             # use_ue8m0
-                        is_e4m3            # is_e4m3
-                    )
-                    q_index = (
-                        q_fp8.view(torch.float8_e4m3fnuz)
-                        if _is_fp8_fnuz
-                        else q_fp8.view(torch.float8_e4m3fn)
-                    )
-            else:
-                query, key = self._get_q_k_bf16(
-                    q_lora,
-                    x,
-                    positions,
-                    enable_dual_stream if not _is_dcu else False,
+            elif not in_piecewise_or_breakable_cuda_graph:
+                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
+                self._store_index_k_cache(
                     forward_batch=forward_batch,
+                    layer_id=layer_id,
+                    key=key,
+                    act_quant=act_quant,
                 )
-                if enable_dual_stream:
-                    current_stream = torch.cuda.current_stream()
-                    self.alt_stream.wait_stream(current_stream)
-                    q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
-                    with torch.cuda.stream(self.alt_stream):
-                        self._store_index_k_cache(
-                            forward_batch=forward_batch,
-                            layer_id=layer_id,
-                            key=key,
-                            act_quant=act_quant,
-                        )
-                    current_stream.wait_stream(self.alt_stream)
-                elif not in_piecewise_or_breakable_cuda_graph:
-                    q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
-                    self._store_index_k_cache(
-                        forward_batch=forward_batch,
-                        layer_id=layer_id,
-                        key=key,
-                        act_quant=act_quant,
-                    )
-                else:
-                    # Graph paths not handled by the full DSA indexer split op
-                    # still need q_fp8 for paged topk and q_scale for the graph
-                    # head-gate path. K-cache storage is handled by the full
-                    # graph split path when prefill requires it.
-                    q_fp8, q_scale = act_quant(
-                        query, self.block_size, self.scale_fmt
-                    )
-                q_index = q_fp8
+            else:
+                # Graph paths not handled by the full DSA indexer split op
+                # still need q_fp8 for paged topk and q_scale for
+                # logits_head_gate_graph. K-cache storage is handled by the
+                # full graph split path when prefill requires it.
+                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
+            q_index = q_fp8
 
+            # aiter (ROCm gfx95): the 3-tuple (fp8, scale, bf16) from
+            # fused_rms_fp8_group_quant is passed directly to _get_logits_head_gate,
+            # which extracts the bf16 tensor via _weights_proj_bf16_in_fp32_out,
+            # completely skipping the FP8 dequantization path below.
             if (
-                not _is_dcu
-                and _use_aiter
+                _use_aiter
                 and _is_gfx95_supported
                 and isinstance(x, tuple)
                 and len(x) == 3
             ):
                 x_for_gate = x
-            elif _is_dcu:
-                x_for_gate = self._get_gate_input_tensor(x)
+            elif isinstance(x, tuple):
+                assert len(x) in (
+                    2,
+                    3,
+                ), "For tuple input, only (x, x_s) or (x, x_s, y) formats are accepted"
+                x_q, x_s = x[0], x[1]
+                if (
+                    x_s is not None
+                    and x_q.dim() == 2
+                    and x_s.dim() == 2
+                    and x_q.shape[0] == x_s.shape[0]
+                ):
+                    m, n = x_q.shape
+                    ng = x_s.shape[1]
+                    if ng > 0 and n % ng == 0:
+                        group = n // ng
+                        x_for_gate = (
+                            x_q.to(torch.float32)
+                            .view(m, ng, group)
+                            .mul_(x_s.to(torch.float32).unsqueeze(-1))
+                            .view(m, n)
+                            .to(torch.bfloat16)
+                        )
+                    else:
+                        x_for_gate = x_q.to(torch.bfloat16)
+                else:
+                    x_for_gate = x_q.to(torch.bfloat16)
             else:
                 x_for_gate = x
-<<<<<<< DCU main@b97ca2082
-            if _is_dcu and self._use_dcu_bf16_index_cache(forward_batch):
-                weights = (
-                    self._project_and_scale_head_gates(x_for_gate).unsqueeze(-1)
-                    * self.softmax_scale
-                )
-            elif not _is_dcu and in_piecewise_or_breakable_cuda_graph:
-                if weights_proj_lora:
-                    raise RuntimeError(GRAPH_WEIGHTS_PROJ_LORA_ERROR)
-                weights = logits_head_gate_graph(
-                    x_for_gate,
-                    self.weights_proj.weight,
-                    self.n_heads**-0.5,
-                    self.softmax_scale,
-                    q_scale,
-                )
-||||||| official previous@eeee3abbbf81
-
-            if in_piecewise_or_breakable_cuda_graph:
-                if weights_proj_lora:
-                    raise RuntimeError(GRAPH_WEIGHTS_PROJ_LORA_ERROR)
-                weights = logits_head_gate_graph(
-                    x_for_gate,
-                    self.weights_proj.weight,
-                    self.n_heads**-0.5,
-                    self.softmax_scale,
-                    q_scale,
-                )
-=======
 
             if in_piecewise_or_breakable_cuda_graph:
                 if _use_dsa_indexer_fusion:
                     weights = scale_head_gate_graph(
                         weights_raw,
                         self.n_heads**-0.5,
                         self.softmax_scale,
                         q_scale,
@@ -2360,17 +2185,16 @@
                         x_for_gate,
                         self.weights_proj.weight,
                         self.n_heads**-0.5,
                         self.softmax_scale,
                         q_scale,
                     )
             elif _use_dsa_indexer_fusion:
                 weights = self._scale_head_gates(weights_raw, q_scale)
->>>>>>> official target@f920a37da46e
             elif weights_proj_lora:
                 weights = self.weights_proj(x_for_gate)[0].float() * self.n_heads**-0.5
                 weights = self._apply_q_scale_and_softmax_scale(weights, q_scale)
             else:
                 weights = self._get_logits_head_gate(x_for_gate, q_scale)
         # if not forward_batch.out_cache_loc.is_contiguous():
         #     forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
         # get_token_to_kv_pool().set_index_k_scale_buffer(
~~~~

</details>

<details>
<summary><code>python/sglang/srt/layers/attention/flashattention_backend.py</code> — 5 conflict hunk(s)</summary>

**Resolution intent:** Adopt official SWA/cascade/device-side metadata for non-DCU paths; preserve DCU NHD/HND layouts, VLLM decode, and LightOp metadata.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/flashattention_backend.py
+++ RESOLVED/python/sglang/srt/layers/attention/flashattention_backend.py
@@ -25,26 +25,18 @@
 from sglang.srt.layers.utils.cp_utils import (
     cp_allgather_and_save_kv_cache,
     cp_attn_forward_extend,
 )
 from sglang.srt.mem_cache.memory_pool import KVWriteLoc
 from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
 from sglang.srt.server_args import get_global_server_args
-<<<<<<< DCU main@b97ca2082
-from sglang.srt.speculative.spec_info import SpecInput
-from sglang.srt.utils import get_compiler_backend, is_dcu
-||||||| official previous@eeee3abbbf81
-from sglang.srt.speculative.spec_info import SpecInput
-from sglang.srt.utils import get_compiler_backend
-=======
 from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm
-from sglang.srt.utils import get_compiler_backend
->>>>>>> official target@f920a37da46e
+from sglang.srt.utils import get_compiler_backend, is_dcu
 
 if TYPE_CHECKING:
     from sglang.srt.layers.radix_attention import RadixAttention
     from sglang.srt.model_executor.model_runner import ModelRunner
 
 from sgl_kernel import merge_state_v2
 
 from sglang.jit_kernel.flash_attention import (
@@ -1563,21 +1555,23 @@
         causal = True
         if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
             causal = False
 
         kwargs = {}
         if sinks is not None:
             kwargs["sinks"] = sinks
 
-        # _fa_out = (
-        #     forward_batch._attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
-        #     if getattr(forward_batch, "_attn_output", None) is not None
-        #     else None
-        # )
+        _fa_out = (
+            forward_batch._attn_output.view(
+                -1, layer.tp_q_head_num, layer.v_head_dim
+            )
+            if getattr(forward_batch, "_attn_output", None) is not None
+            else None
+        )
         # flash_attn_with_kvcache_base = flash_attn_with_kvcache_fa3
 
         # flash_attn_with_kvcache = (
         #     flash_attn_with_kvcache_fa4
         #     if self.fa_impl_ver == 4
         #     else flash_attn_with_kvcache_base
         # )
 
@@ -1655,253 +1649,181 @@
                         page_table = (
                             self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                 metadata.page_table
                             ).to(torch.int32)
                         )
                 cache_seqlens = metadata.cache_seqlens_int32
                 cu_seqlens_q = metadata.cu_seqlens_q
                 max_seqlen_q = metadata.max_seq_len_q
-<<<<<<< DCU main@b97ca2082
-                max_seqlen_k = metadata.max_seq_len_k
-                # page_table = metadata.page_table
-                cu_seqlens_k = metadata.cu_seqlens_k
-                cache_seqlens = metadata.cache_seqlens_int32
-                if not _kv_layout_dcu_fa:
-                    key_cache = key_cache.view(
-                        -1, self.page_size, layer.tp_k_head_num, layer.head_dim
-                    )
-                    value_cache = value_cache.view(
-                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
-                    )
-                else:
-                    key_cache = key_cache.view(
-                        -1, layer.tp_k_head_num, self.page_size, layer.head_dim
-                    )
-                    value_cache = value_cache.view(
-                        -1, layer.tp_v_head_num, layer.head_dim, self.page_size
+                if _is_dcu:
+                    max_seqlen_k = metadata.max_seq_len_k
+                    cu_seqlens_k = metadata.cu_seqlens_k
+                    if not _kv_layout_dcu_fa:
+                        key_cache = key_cache.view(
+                            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
+                        )
+                        value_cache = value_cache.view(
+                            -1,
+                            self.page_size,
+                            layer.tp_v_head_num,
+                            layer.v_head_dim,
+                        )
+                    else:
+                        key_cache = key_cache.view(
+                            -1, layer.tp_k_head_num, self.page_size, layer.head_dim
+                        )
+                        value_cache = value_cache.view(
+                            -1,
+                            layer.tp_v_head_num,
+                            layer.head_dim,
+                            self.page_size,
+                        )
+
+                    q_reshaped = q.contiguous().view(
+                        -1, layer.tp_q_head_num, layer.head_dim
                     )
-||||||| official previous@eeee3abbbf81
-                q_reshaped = q.contiguous().view(
-                    -1, layer.tp_q_head_num, layer.head_dim
-                )
-=======
-
-                pa_swa_active = False
-                if self.is_prefill_aware_swa and metadata.pa_swa_page_table is not None:
-                    page_table = metadata.pa_swa_page_table
-                    cache_seqlens = metadata.pa_swa_cache_seqlens
-                    window_size = (-1, -1)
-                    pa_swa_active = True
-
-                q_reshaped = q.contiguous().view(
-                    -1, layer.tp_q_head_num, layer.head_dim
-                )
->>>>>>> official target@f920a37da46e
-
-<<<<<<< DCU main@b97ca2082
-                if layer.is_cross_attention:
-                    page_table = metadata.encoder_page_table
-                    cache_seqlens = metadata.encoder_lens_int32
-                    cu_seqlens_k = metadata.encoder_cu_seqlens_k
-                    window_size = (-1, -1)
-                q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
-                if not _kv_layout_dcu_fa:
-                    result = varlen_fwd_unified(
-                        q=q,
-                        k=key_cache,
-                        v=value_cache,
-                        cu_seqlens_q=cu_seqlens_q,
-                        seqused_k=cache_seqlens,
-                        block_table=page_table,
-                        max_seqlen_q=max_seqlen_q,
-                        max_seqlen_k=self.max_context_len,
-                        softmax_scale=layer.scaling,
-                        causal=True,
-                        softcap=layer.logit_cap,
-                        window_size=window_size,
-                        q_descale=k_descale,
-                        k_descale=k_descale,
-                        v_descale=v_descale,
-                        return_softmax_lse=use_cascade_attn,
-                        s_aux=kwargs.get('sinks', None)
-||||||| official previous@eeee3abbbf81
-                # Default: single-token self-attention
-                # Use precomputed scheduler_metadata when available and applicable.
-                # scheduler_metadata is only valid for non-SWA, non-cascade decode.
-                sched_meta = None
-                if (
-                    metadata.scheduler_metadata is not None
-                    and not is_swa_layer
-                    and not use_cascade_attn
-                ):
-                    sched_meta = metadata.scheduler_metadata
-                result = flash_attn_with_kvcache(
-                    q=q_reshaped,
-                    k_cache=key_cache,
-                    v_cache=value_cache,
-                    page_table=page_table,
-                    cache_seqlens=cache_seqlens,
-                    cu_seqlens_q=metadata.cu_seqlens_q,
-                    max_seqlen_q=max_seqlen_q,
-                    softmax_scale=layer.scaling,
-                    causal=False if use_cascade_attn else causal,
-                    window_size=window_size,
-                    softcap=layer.logit_cap,
-                    k_descale=k_descale,
-                    v_descale=v_descale,
-                    return_softmax_lse=use_cascade_attn,
-                    num_splits=self.num_splits,
-                    out=_fa_out,
-                    ver=self.fa_impl_ver,
-                    scheduler_metadata=sched_meta,
-                    **kwargs,
-                )
-                if use_cascade_attn:
-                    o, softmax_lse, *rest = result
-                    o_expand, softmax_lse_expand, *rest_expand = (
-                        flash_attn_with_kvcache(
+                    if not _kv_layout_dcu_fa:
+                        result = varlen_fwd_unified(
                             q=q_reshaped,
-                            k_cache=key_cache,
-                            v_cache=value_cache,
-                            page_table=self.forward_metadata_spec_decode_expand.page_table,
-                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
-                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
-                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
-                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
+                            k=key_cache,
+                            v=value_cache,
+                            cu_seqlens_q=cu_seqlens_q,
+                            seqused_k=cache_seqlens,
+                            block_table=page_table,
+                            max_seqlen_q=max_seqlen_q,
+                            max_seqlen_k=self.max_context_len,
+                            softmax_scale=layer.scaling,
+                            causal=True,
+                            softcap=layer.logit_cap,
+                            window_size=window_size,
+                            q_descale=k_descale,
+                            k_descale=k_descale,
+                            v_descale=v_descale,
+                            return_softmax_lse=use_cascade_attn,
+                            s_aux=kwargs.get("sinks"),
+                        )
+                    elif max_seqlen_q > 1:
+                        result = flash_attn_varlen_func(
+                            q=q_reshaped,
+                            k=key_cache,
+                            v=value_cache,
+                            cu_seqlens_q=cu_seqlens_q,
+                            cu_seqlens_k=cu_seqlens_k,
+                            max_seqlen_q=max_seqlen_q,
+                            max_seqlen_k=max_seqlen_k,
                             softmax_scale=layer.scaling,
-                            causal=False,
+                            causal=True,
                             window_size=window_size,
                             softcap=layer.logit_cap,
                             k_descale=k_descale,
                             v_descale=v_descale,
-                            return_softmax_lse=True,
+                            return_softmax_lse=use_cascade_attn,
                             num_splits=self.num_splits,
-                            ver=self.fa_impl_ver,
                             **kwargs,
                         )
-=======
-                # Default: single-token self-attention
-                # Use precomputed scheduler_metadata when available and applicable.
-                # scheduler_metadata is only valid for non-SWA, non-cascade decode.
-                sched_meta = None
-                if (
-                    metadata.scheduler_metadata is not None
-                    and not is_swa_layer
-                    and not use_cascade_attn
-                    and not pa_swa_active
-                ):
-                    sched_meta = metadata.scheduler_metadata
-                result = flash_attn_with_kvcache(
-                    q=q_reshaped,
-                    k_cache=key_cache,
-                    v_cache=value_cache,
-                    page_table=page_table,
-                    cache_seqlens=cache_seqlens,
-                    cu_seqlens_q=metadata.cu_seqlens_q,
-                    max_seqlen_q=max_seqlen_q,
-                    softmax_scale=layer.scaling,
-                    causal=False if use_cascade_attn else causal,
-                    window_size=window_size,
-                    softcap=layer.logit_cap,
-                    k_descale=k_descale,
-                    v_descale=v_descale,
-                    return_softmax_lse=use_cascade_attn,
-                    num_splits=self.num_splits,
-                    out=_fa_out,
-                    ver=self.fa_impl_ver,
-                    scheduler_metadata=sched_meta,
-                    **kwargs,
-                )
-                if use_cascade_attn:
-                    o, softmax_lse, *rest = result
-                    o_expand, softmax_lse_expand, *rest_expand = (
-                        flash_attn_with_kvcache(
-                            q=q_reshaped,
+                    else:
+                        result = vllm_flash_attn_with_kvcache(
+                            q=q_reshaped.unsqueeze(1),
                             k_cache=key_cache,
                             v_cache=value_cache,
-                            page_table=self.forward_metadata_spec_decode_expand.page_table,
-                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
-                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
-                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
-                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
+                            page_table=page_table,
+                            cache_seqlens=cache_seqlens,
+                            cu_seqlens_q=cu_seqlens_q,
+                            cu_seqlens_k_new=cu_seqlens_k,
+                            max_seqlen_q=max_seqlen_q,
                             softmax_scale=layer.scaling,
-                            causal=False,
+                            causal=True,
                             window_size=window_size,
                             softcap=layer.logit_cap,
                             k_descale=k_descale,
                             v_descale=v_descale,
-                            return_softmax_lse=True,
+                            return_softmax_lse=use_cascade_attn,
                             num_splits=self.num_splits,
-                            ver=self.fa_impl_ver,
                             **kwargs,
                         )
->>>>>>> official target@f920a37da46e
-                    )
-                elif max_seqlen_q > 1:
-                    result = flash_attn_varlen_func(
-                        q=q,
-                        k=key_cache,
-                        v=value_cache,
-                        cu_seqlens_q=cu_seqlens_q,
-                        cu_seqlens_k=cu_seqlens_k,
-                        max_seqlen_q=max_seqlen_q,
-                        max_seqlen_k=max_seqlen_k,
-                        softmax_scale=layer.scaling,
-                        causal=True,
-                        window_size=window_size,
-                        softcap=layer.logit_cap,
-                        k_descale=k_descale,
-                        v_descale=v_descale,
-                        return_softmax_lse=use_cascade_attn,
-                        num_splits=self.num_splits,
-                        **kwargs,
-                    )
-                elif _kv_layout_dcu_fa:
-                    result = vllm_flash_attn_with_kvcache(
-                        q=q.unsqueeze(1),
-                        k_cache=key_cache,
-                        v_cache=value_cache,
-                        page_table=page_table,
-                        cache_seqlens=cache_seqlens,
-                        cu_seqlens_q=cu_seqlens_q,
-                        cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
-                        max_seqlen_q=max_seqlen_q,
-                        # max_seqlen_k=max_seqlen_k,
-                        softmax_scale=layer.scaling,
-                        causal=True,
-                        window_size=window_size,
-                        softcap=layer.logit_cap,
-                        k_descale=k_descale,
-                        v_descale=v_descale,
-                        return_softmax_lse=use_cascade_attn,
-                        num_splits=self.num_splits,
-                        **kwargs,
-                    )
+                    o = result
                 else:
+                    pa_swa_active = False
+                    if (
+                        self.is_prefill_aware_swa
+                        and metadata.pa_swa_page_table is not None
+                    ):
+                        page_table = metadata.pa_swa_page_table
+                        cache_seqlens = metadata.pa_swa_cache_seqlens
+                        window_size = (-1, -1)
+                        pa_swa_active = True
+
+                    q_reshaped = q.contiguous().view(
+                        -1, layer.tp_q_head_num, layer.head_dim
+                    )
+
+                    # scheduler_metadata is only valid for non-SWA,
+                    # non-cascade decode.
+                    sched_meta = None
+                    if (
+                        metadata.scheduler_metadata is not None
+                        and not is_swa_layer
+                        and not use_cascade_attn
+                        and not pa_swa_active
+                    ):
+                        sched_meta = metadata.scheduler_metadata
                     result = flash_attn_with_kvcache(
-                        q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
+                        q=q_reshaped,
                         k_cache=key_cache,
                         v_cache=value_cache,
                         page_table=page_table,
                         cache_seqlens=cache_seqlens,
-                        cu_seqlens_q=cu_seqlens_q,
-                        cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
+                        cu_seqlens_q=metadata.cu_seqlens_q,
                         max_seqlen_q=max_seqlen_q,
                         softmax_scale=layer.scaling,
-                        causal=True,
+                        causal=False if use_cascade_attn else causal,
                         window_size=window_size,
                         softcap=layer.logit_cap,
                         k_descale=k_descale,
                         v_descale=v_descale,
                         return_softmax_lse=use_cascade_attn,
                         num_splits=self.num_splits,
+                        out=_fa_out,
+                        ver=self.fa_impl_ver,
+                        scheduler_metadata=sched_meta,
                         **kwargs,
                     )
-                o = result
+                    if use_cascade_attn:
+                        o, softmax_lse, *rest = result
+                        o_expand, softmax_lse_expand, *rest_expand = (
+                            flash_attn_with_kvcache(
+                                q=q_reshaped,
+                                k_cache=key_cache,
+                                v_cache=value_cache,
+                                page_table=self.forward_metadata_spec_decode_expand.page_table,
+                                cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
+                                cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
+                                cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
+                                max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
+                                softmax_scale=layer.scaling,
+                                causal=False,
+                                window_size=window_size,
+                                softcap=layer.logit_cap,
+                                k_descale=k_descale,
+                                v_descale=v_descale,
+                                return_softmax_lse=True,
+                                num_splits=self.num_splits,
+                                ver=self.fa_impl_ver,
+                                **kwargs,
+                            )
+                        )
+                        o, _ = merge_state_v2(
+                            o,
+                            softmax_lse.T.contiguous(),
+                            o_expand,
+                            softmax_lse_expand.T.contiguous(),
+                        )
+                    else:
+                        o = result
         else:
             # Do absorbed multi-latent attention
             kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(q.dtype)
             k_rope = kv_cache[:, :, layer.v_head_dim :]
             c_kv = kv_cache[:, :, : layer.v_head_dim]
             k_rope_cache = k_rope.view(
                 -1,
                 self.page_size,
@@ -2541,29 +2463,29 @@
             )
 
         if forward_mode.is_decode_or_idle():
             if spec_info is not None:
                 # Draft Decode
                 if self.topk <= 1:
                     # When topk = 1, we use the normal decode metadata
                     metadata = self.decode_cuda_graph_metadata[bs]
-<<<<<<< DCU main@b97ca2082
-                    max_len = seq_lens_cpu.max().item()
-                    metadata.max_seq_len_k = max_len + self.speculative_step_id + 1
-                    max_seq_pages = (
-                        metadata.max_seq_len_k + self.page_size - 1
-                    ) // self.page_size
-
-                    assert_buffer_fits(
-                        max_seq_pages,
-                        metadata.page_table.shape[1],
-                        "FA3 draft-decode page_table",
-                    )
                     if _is_dcu:
+                        max_len = self._host_max_seq_len(seq_lens_cpu, seq_lens)
+                        metadata.max_seq_len_k = (
+                            max_len + self.speculative_step_id + 1
+                        )
+                        max_seq_pages = (
+                            metadata.max_seq_len_k + self.page_size - 1
+                        ) // self.page_size
+                        assert_buffer_fits(
+                            max_seq_pages,
+                            metadata.page_table.shape[1],
+                            "FA3 draft-decode page_table",
+                        )
                         normal_decode_set_metadata_lightop(
                             metadata.cache_seqlens_int32,
                             metadata.cu_seqlens_k,
                             metadata.page_table,
                             self.req_to_token,
                             req_pool_indices,
                             max_seq_pages,
                             seq_lens,
@@ -2572,87 +2494,36 @@
                             metadata.swa_page_table,
                             (
                                 self.token_to_kv_pool
                                 if self.use_sliding_window_kv_pool
                                 else None
                             ),
                         )
                     else:
+                        # Build the page table on device and self-guard using
+                        # cache_seqlens; max_seq_len_k is unread in this path.
                         normal_decode_set_metadata(
                             metadata.cache_seqlens_int32,
                             metadata.cu_seqlens_k,
                             metadata.page_table,
                             self.req_to_token,
                             req_pool_indices,
                             self.decode_cuda_graph_metadata["strided_indices"],
-                            max_seq_pages,
+                            self.max_num_pages,
                             seq_lens,
                             self.speculative_step_id + 1,
                             self.page_size,
                             metadata.swa_page_table,
                             (
                                 self.token_to_kv_pool
                                 if self.use_sliding_window_kv_pool
                                 else None
                             ),
                         )
-||||||| official previous@eeee3abbbf81
-                    max_len = seq_lens_cpu.max().item()
-                    metadata.max_seq_len_k = max_len + self.speculative_step_id + 1
-                    max_seq_pages = (
-                        metadata.max_seq_len_k + self.page_size - 1
-                    ) // self.page_size
-
-                    assert_buffer_fits(
-                        max_seq_pages,
-                        metadata.page_table.shape[1],
-                        "FA3 draft-decode page_table",
-                    )
-                    normal_decode_set_metadata(
-                        metadata.cache_seqlens_int32,
-                        metadata.cu_seqlens_k,
-                        metadata.page_table,
-                        self.req_to_token,
-                        req_pool_indices,
-                        self.decode_cuda_graph_metadata["strided_indices"],
-                        max_seq_pages,
-                        seq_lens,
-                        self.speculative_step_id + 1,
-                        self.page_size,
-                        metadata.swa_page_table,
-                        (
-                            self.token_to_kv_pool
-                            if self.use_sliding_window_kv_pool
-                            else None
-                        ),
-                    )
-=======
-                    # Page table built on-device (self-guards on cache_seqlens);
-                    # max_seq_len_k left unset -- unread here (scheduler_metadata
-                    # is normal-decode-only).
-                    normal_decode_set_metadata(
-                        metadata.cache_seqlens_int32,
-                        metadata.cu_seqlens_k,
-                        metadata.page_table,
-                        self.req_to_token,
-                        req_pool_indices,
-                        self.decode_cuda_graph_metadata["strided_indices"],
-                        self.max_num_pages,
-                        seq_lens,
-                        self.speculative_step_id + 1,
-                        self.page_size,
-                        metadata.swa_page_table,
-                        (
-                            self.token_to_kv_pool
-                            if self.use_sliding_window_kv_pool
-                            else None
-                        ),
-                    )
->>>>>>> official target@f920a37da46e
 
                 else:
                     # When top k > 1, we need two specific draft decode metadata, and then merge states
                     # 1. The first half of metadata for prefix tokens
                     metadata = self.draft_decode_metadata_topk_normal[bs]
                     if self.page_size > 1:
                         # First attention handles seq_lens - last_page_lens if page size > 1.
                         last_page_lens = seq_lens % self.page_size
@@ -2707,27 +2578,27 @@
                         num_seqs = cache_loc.shape[0]
                         metadata_expand.page_table[:num_seqs, :decode_length].copy_(
                             cache_loc[:, :decode_length]
                         )
                 # TODO: Handle local attention metadata for draft decode when llama4 eagle is supported
             else:
                 # Normal Decode
                 metadata = self.decode_cuda_graph_metadata[bs]
-<<<<<<< DCU main@b97ca2082
-                max_len = seq_lens_cpu.max().item()
-                max_seq_pages = (max_len + self.page_size - 1) // self.page_size
-                metadata.max_seq_len_k = max_len
-
-                assert_buffer_fits(
-                    max_seq_pages,
-                    metadata.page_table.shape[1],
-                    "FA3 decode page_table",
-                )
                 if _is_dcu:
+                    max_len = self._host_max_seq_len(seq_lens_cpu, seq_lens)
+                    max_seq_pages = (
+                        max_len + self.page_size - 1
+                    ) // self.page_size
+                    metadata.max_seq_len_k = max_len
+                    assert_buffer_fits(
+                        max_seq_pages,
+                        metadata.page_table.shape[1],
+                        "FA3 decode page_table",
+                    )
                     normal_decode_set_metadata_lightop(
                         metadata.cache_seqlens_int32,
                         metadata.cu_seqlens_k,
                         metadata.page_table,
                         self.req_to_token,
                         req_pool_indices,
                         max_seq_pages,
                         seq_lens,
@@ -2735,61 +2606,17 @@
                         self.page_size,
                         metadata.swa_page_table,
                         (
                             self.token_to_kv_pool
                             if self.use_sliding_window_kv_pool
                             else None
                         ),
                     )
-                else:
-                    normal_decode_set_metadata(
-                        metadata.cache_seqlens_int32,
-                        metadata.cu_seqlens_k,
-                        metadata.page_table,
-                        self.req_to_token,
-                        req_pool_indices,
-                        self.decode_cuda_graph_metadata["strided_indices"],
-                        max_seq_pages,
-                        seq_lens,
-                        0,
-                        self.page_size,
-                        metadata.swa_page_table,
-                        (
-                            self.token_to_kv_pool
-                            if self.use_sliding_window_kv_pool
-                            else None
-                            ),
-                        )
-||||||| official previous@eeee3abbbf81
-                max_len = seq_lens_cpu.max().item()
-                max_seq_pages = (max_len + self.page_size - 1) // self.page_size
-                metadata.max_seq_len_k = max_len
-
-                assert_buffer_fits(
-                    max_seq_pages,
-                    metadata.page_table.shape[1],
-                    "FA3 decode page_table",
-                )
-                normal_decode_set_metadata(
-                    metadata.cache_seqlens_int32,
-                    metadata.cu_seqlens_k,
-                    metadata.page_table,
-                    self.req_to_token,
-                    req_pool_indices,
-                    self.decode_cuda_graph_metadata["strided_indices"],
-                    max_seq_pages,
-                    seq_lens,
-                    0,
-                    self.page_size,
-                    metadata.swa_page_table,
-                    self.token_to_kv_pool if self.use_sliding_window_kv_pool else None,
-                )
-=======
-                if self.is_prefill_aware_swa:
+                elif self.is_prefill_aware_swa:
                     # Prefill-aware SWA still needs a host max to bound the
                     # per-batch page table built below.
                     max_len = self._host_max_seq_len(seq_lens_cpu, seq_lens)
                     metadata.max_seq_len_k = max_len
                     pa_max_len = min(
                         self._pa_swa_max_prefill_len + self.sliding_window_size,
                         max_len,
                     )
@@ -2802,19 +2629,18 @@
                             self.sliding_window_size,
                             bs,
                             pa_max_len,
                             device,
                             dst_page_table=metadata.page_table,
                             dst_kv_lens=metadata.cache_seqlens_int32,
                         )
                 else:
-                    # Page table uses the static max_num_pages bound (no D2H).
-                    # max_seq_len_k only feeds scheduler_metadata below, so use
-                    # the free CPU mirror for a tight split heuristic when present.
+                    # Use the static max_num_pages bound so this hot path avoids
+                    # a D2H sync on devices using the upstream Triton metadata op.
                     metadata.max_seq_len_k = (
                         seq_lens_cpu.max().item()
                         if seq_lens_cpu is not None
                         else self.max_context_len
                     )
                     normal_decode_set_metadata(
                         metadata.cache_seqlens_int32,
                         metadata.cu_seqlens_k,
@@ -2828,17 +2654,17 @@
                         self.page_size,
                         metadata.swa_page_table,
                         (
                             self.token_to_kv_pool
                             if self.use_sliding_window_kv_pool
                             else None
                         ),
                     )
->>>>>>> official target@f920a37da46e
+
 
                 self._maybe_update_local_attn_metadata_for_replay(
                     metadata,
                     bs,
                 )
 
                 # Recompute scheduler_metadata into pre-allocated buffer
                 if (
~~~~

</details>

<details>
<summary><code>python/sglang/srt/layers/attention/linear/gdn_backend.py</code> — 1 conflict hunk(s)</summary>

**Resolution intent:** Preserve causal_conv1d_fn_dcu while using official contiguous state buffers and cache indices.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/linear/gdn_backend.py
+++ RESOLVED/python/sglang/srt/layers/attention/linear/gdn_backend.py
@@ -490,66 +490,40 @@
             if forward_metadata.has_mamba_track_mask:
                 mixed_qkv_to_track = mixed_qkv[
                     :, forward_metadata.track_conv_indices
                 ].transpose(0, 1)
                 conv_states[forward_metadata.conv_states_mask_indices] = (
                     mixed_qkv_to_track
                 )
 
-<<<<<<< DCU main@b97ca2082
             if _is_dcu and _use_causal_conv1d:
                 mixed_qkv = causal_conv1d_fn_dcu(
                     mixed_qkv,
                     layer.conv_weights,
                     layer.bias,
                     activation=layer.activation,
-                    initial_states=conv_states,
+                    initial_states=conv_states_contig,
                     has_initial_state=has_initial_states,
-                    cache_indices=cache_indices,
+                    cache_indices=state_cache_indices,
                     query_start_loc=query_start_loc,
                     seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                 ).transpose(0, 1)[:seq_len]
             else:
                 mixed_qkv = causal_conv1d_fn(
                     mixed_qkv,
                     layer.conv_weights,
                     layer.bias,
                     activation=layer.activation,
-                    conv_states=conv_states,
+                    conv_states=conv_states_contig,
                     has_initial_state=has_initial_states,
-                    cache_indices=cache_indices,
+                    cache_indices=state_cache_indices,
                     query_start_loc=query_start_loc,
                     seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                 ).transpose(0, 1)[:seq_len]
-||||||| official previous@eeee3abbbf81
-            mixed_qkv = causal_conv1d_fn(
-                mixed_qkv,
-                layer.conv_weights,
-                layer.bias,
-                activation=layer.activation,
-                conv_states=conv_states,
-                has_initial_state=has_initial_states,
-                cache_indices=cache_indices,
-                query_start_loc=query_start_loc,
-                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
-            ).transpose(0, 1)[:seq_len]
-=======
-            mixed_qkv = causal_conv1d_fn(
-                mixed_qkv,
-                layer.conv_weights,
-                layer.bias,
-                activation=layer.activation,
-                conv_states=conv_states_contig,
-                has_initial_state=has_initial_states,
-                cache_indices=state_cache_indices,
-                query_start_loc=query_start_loc,
-                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
-            ).transpose(0, 1)[:seq_len]
->>>>>>> official target@f920a37da46e
 
         actual_seq_len = mixed_qkv.shape[0]
         qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
         if (is_cuda() or is_hip()) and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM:
             query, key, value = fused_qkv_split_gdn_prefill(
                 mixed_qkv,
                 layer.num_q_heads,
                 layer.num_k_heads,
~~~~

</details>

<details>
<summary><code>python/sglang/srt/managers/scheduler.py</code> — 2 conflict hunk(s)</summary>

**Resolution intent:** Take the official recurrent hidden-state specification for decode and prefill disaggregation buffers.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/managers/scheduler.py
+++ RESOLVED/python/sglang/srt/managers/scheduler.py
@@ -1144,42 +1144,18 @@
             self.disaggregation_mode == DisaggregationMode.DECODE
         ):  # *2 for the headroom.
             buffer_size = (self.req_to_token_pool.size) * 2
             self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                 buffer_size
             )
             self.disagg_metadata_buffers = MetadataBuffers(
                 buffer_size,
-<<<<<<< DCU main@b97ca2082
-                hidden_size=(
-                    getattr(model_config, "spec_hidden_size", model_config.hidden_size)
-                    if self.spec_algorithm.carries_draft_hidden_states()
-                    else 16  # minimal padding size for RDMA
-                ),
-                hidden_states_dtype=(
-                    model_config.dtype
-                    if self.spec_algorithm.carries_draft_hidden_states()
-                    else torch.float32
-                ),
-||||||| official previous@eeee3abbbf81
-                hidden_size=(
-                    model_config.spec_hidden_size
-                    if self.spec_algorithm.carries_draft_hidden_states()
-                    else 16  # minimal padding size for RDMA
-                ),
-                hidden_states_dtype=(
-                    model_config.dtype
-                    if self.spec_algorithm.carries_draft_hidden_states()
-                    else torch.float32
-                ),
-=======
                 hidden_size=disagg_hidden_size,
                 hidden_states_dtype=disagg_hidden_states_dtype,
->>>>>>> official target@f920a37da46e
                 custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
             )
 
             # The decode requests polling kv cache
             self.disagg_decode_transfer_queue = DecodeTransferQueue(
                 gloo_group=self.attn_tp_cpu_group,
                 req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                 tp_rank=self.ps.tp_rank,
@@ -1213,42 +1189,18 @@
         elif self.disaggregation_mode == DisaggregationMode.PREFILL:
             # *2 for the headroom.
             buffer_size = self.max_running_requests * 2
             self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                 buffer_size
             )
             self.disagg_metadata_buffers = MetadataBuffers(
                 buffer_size,
-<<<<<<< DCU main@b97ca2082
-                hidden_size=(
-                    getattr(model_config, "spec_hidden_size", model_config.hidden_size)
-                    if self.spec_algorithm.carries_draft_hidden_states()
-                    else 16  # minimal padding size for RDMA
-                ),
-                hidden_states_dtype=(
-                    model_config.dtype
-                    if self.spec_algorithm.carries_draft_hidden_states()
-                    else torch.float32
-                ),
-||||||| official previous@eeee3abbbf81
-                hidden_size=(
-                    model_config.spec_hidden_size
-                    if self.spec_algorithm.carries_draft_hidden_states()
-                    else 16  # minimal padding size for RDMA
-                ),
-                hidden_states_dtype=(
-                    model_config.dtype
-                    if self.spec_algorithm.carries_draft_hidden_states()
-                    else torch.float32
-                ),
-=======
                 hidden_size=disagg_hidden_size,
                 hidden_states_dtype=disagg_hidden_states_dtype,
->>>>>>> official target@f920a37da46e
                 custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
             )
 
             self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                 token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                 draft_token_to_kv_pool=draft_token_to_kv_pool,
                 req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                 metadata_buffers=self.disagg_metadata_buffers,
~~~~

</details>

<details>
<summary><code>python/sglang/srt/mem_cache/memory_pool.py</code> — 2 conflict hunk(s)</summary>

**Resolution intent:** Keep the validated DCU FA K/V physical layout before the official generic HND/vectorized/NHD allocation paths.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/mem_cache/memory_pool.py
+++ RESOLVED/python/sglang/srt/mem_cache/memory_pool.py
@@ -1349,27 +1349,19 @@
         self.device_module = torch.get_device_module(self.device)
 
         _use_alt_stream = _is_cuda or current_platform.is_cuda_alike()
         self.alt_stream = (
             self.device_module.Stream()
             if _use_alt_stream and enable_alt_stream
             else None
         )
-<<<<<<< DCU main@b97ca2082
-        if enable_kv_cache_copy:
-||||||| official previous@eeee3abbbf81
-
-        if enable_kv_cache_copy:
-=======
-
         if enable_kv_cache_copy and not self.use_hnd:
             # The tiled byte copy assumes NHD slot-rows; HND uses a (page, off)
             # gather in move_kv_cache instead, so skip the slot-row copy config.
->>>>>>> official target@f920a37da46e
             self._init_kv_copy_and_warmup()
         else:
             self._kv_copy_config = None
 
         self._finalize_allocation_log(size)
 
         # for store_cache JIT kernel
         self.row_dim = self.head_num * self.head_dim
@@ -1429,41 +1421,36 @@
 
     def _create_buffers(self):
         with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
             with (
                 torch.cuda.use_mem_pool(self.custom_mem_pool)
                 if self.enable_custom_mem_pool
                 else nullcontext()
             ):
-<<<<<<< DCU main@b97ca2082
                 if _kv_layout_dcu_fa:
-                    page_num = int((self.size + self.page_size) / self.page_size)
+                    page_num = (self.size + self.page_size) // self.page_size
                     self.k_buffer = [
                         torch.zeros(
                             (page_num, self.head_num, self.page_size, self.head_dim),
                             dtype=self.store_dtype,
                             device=self.device,
                         )
                         for _ in range(self.layer_num)
                     ]
                     self.v_buffer = [
                         torch.zeros(
                             (page_num, self.head_num, self.v_head_dim, self.page_size),
                             dtype=self.store_dtype,
                             device=self.device,
                         )
                         for _ in range(self.layer_num)
                     ]
-                elif self.kv_cache_layout == "vectorized_5d":
-||||||| official previous@eeee3abbbf81
-                if self.kv_cache_layout == "vectorized_5d":
-=======
                 # The padded page (slot 0's page) absorbs dummy padded-token writes.
-                if self.use_hnd:
+                elif self.use_hnd:
                     k_shape = (
                         self.num_pages,
                         self.head_num,
                         self.page_size,
                         self.head_dim,
                     )
                     v_shape = (
                         self.num_pages,
@@ -1475,17 +1462,16 @@
                         torch.zeros(k_shape, dtype=self.store_dtype, device=self.device)
                         for _ in range(self.layer_num)
                     ]
                     self.v_buffer = [
                         torch.zeros(v_shape, dtype=self.store_dtype, device=self.device)
                         for _ in range(self.layer_num)
                     ]
                 elif self.kv_cache_layout == "vectorized_5d":
->>>>>>> official target@f920a37da46e
                     total_slots = self.size + self.page_size
                     num_blocks = total_slots // self.page_size
                     x = self._kv_vector_x
                     # K: (num_blocks, H, D_k // X, page, X)
                     self.k_buffer = [
                         torch.zeros(
                             (
                                 num_blocks,
~~~~

</details>

<details>
<summary><code>python/sglang/srt/models/deepseek_v2.py</code> — 3 conflict hunk(s)</summary>

**Resolution intent:** Adopt official JIT router/fused-A and shared-expert launch order; retain DCU BMM, HIP helpers, and residual-RMS arguments.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/models/deepseek_v2.py
+++ RESOLVED/python/sglang/srt/models/deepseek_v2.py
@@ -287,22 +287,23 @@
         fused_qk_rope_cat_and_cache_mla,
         get_dsv3_gemm_output_zero_allocator_size,
     )
 
 if _use_aiter:
     pass
 
 if _is_cuda:
-<<<<<<< DCU main@b97ca2082
-    from flashinfer.gemm import mm_M1_16_K7168_N256 as _raw_dsv3_router_gemm
     from sgl_kernel import bmm_fp8 as _raw_bmm_fp8
-    from sgl_kernel import dsv3_fused_a_gemm, dsv3_router_gemm
 
     from sglang.jit_kernel.concat_mla import concat_mla_k
+    from sglang.jit_kernel.dsv3_router_gemm import (
+        dsv3_router_gemm as _jit_dsv3_router_gemm,
+    )
+    from sglang.jit_kernel.fused_a_gemm import dsv3_fused_a_gemm
 
     @register_custom_op(mutates_args=["out"])
     def _bmm_fp8_op(
         A: torch.Tensor,
         B: torch.Tensor,
         out: torch.Tensor,
         A_scale: torch.Tensor,
         B_scale: torch.Tensor,
@@ -318,42 +319,27 @@
             )
         _bmm_fp8_op(A, B, out, A_scale, B_scale)
         return out
 elif _is_hip:
     from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
         decode_attention_fwd_grouped_rope,
     )
     from sgl_kernel import merge_state_v2
-||||||| official previous@eeee3abbbf81
-    from flashinfer.gemm import mm_M1_16_K7168_N256 as _raw_dsv3_router_gemm
-    from sgl_kernel import dsv3_fused_a_gemm, dsv3_router_gemm
-=======
-    from sglang.jit_kernel.dsv3_router_gemm import (
-        dsv3_router_gemm as _jit_dsv3_router_gemm,
-    )
-    from sglang.jit_kernel.fused_a_gemm import dsv3_fused_a_gemm
->>>>>>> official target@f920a37da46e
 elif _is_npu:
     from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
         forward_dsa_core_npu,
         forward_dsa_prepare_npu,
         forward_mha_core_npu,
         forward_mha_prepare_npu,
         forward_mla_core_npu,
         forward_mla_prepare_npu,
     )
 elif _is_musa:
-<<<<<<< DCU main@b97ca2082
-    from sgl_kernel import concat_mla_k, dsv3_fused_a_gemm, dsv3_router_gemm
-||||||| official previous@eeee3abbbf81
-    from sgl_kernel import dsv3_fused_a_gemm, dsv3_router_gemm
-=======
     from sgl_kernel import dsv3_fused_a_gemm
->>>>>>> official target@f920a37da46e
 else:
     pass
 
 logger = logging.getLogger(__name__)
 
 # 暂时先放这
 def ds_bmm_wrapper(q: torch.Tensor, w: torch.Tensor, scale: float, dtype: torch.dtype):
     # # scale=1时去掉elementwise数乘
@@ -1120,48 +1106,34 @@
     ) -> torch.Tensor:
         # Note(kpham-sgl): launch the shared expert BEFORE the routed call.
         # The routed deep_gemm pre-permute calls `dispose_tensor` which
         # `set_()`s `hidden_states` to empty (host-side); any later kernel
         # launch consuming `hidden_states` then captures `data_ptr() == 0`
         # into the decode CUDA graph and replays from null.
         current_stream = torch.cuda.current_stream()
         self.alt_stream.wait_stream(current_stream)
-<<<<<<< DCU main@b97ca2082
-        # shared_output = self._forward_shared_experts(
-        #     hidden_states, gemm_output_zero_allocator
-        # )
-        i_q = None
-        i_s = None
-        if _use_fused_rms_quant:
-            shared_output, new_resi, i_q, i_s = self._forward_shared_experts(
-                hidden_states, gemm_output_zero_allocator,
-                rms_weight = rms_weight,
-                residual = residual,
-            )
-        else:
-            shared_output = self._forward_shared_experts(
-                hidden_states, gemm_output_zero_allocator,
-            )
-||||||| official previous@eeee3abbbf81
-        shared_output = self._forward_shared_experts(
-            hidden_states, gemm_output_zero_allocator
-        )
-=======
->>>>>>> official target@f920a37da46e
         server_args = get_global_server_args()
         dispatch_info = (
             ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
             if server_args.enable_eplb
             else None
         )
         with torch.cuda.stream(self.alt_stream):
-            shared_output = self._forward_shared_experts(
-                hidden_states, gemm_output_zero_allocator
-            )
+            if _use_fused_rms_quant and rms_weight is not None and residual is not None:
+                shared_output, _, _, _ = self._forward_shared_experts(
+                    hidden_states,
+                    gemm_output_zero_allocator,
+                    rms_weight=rms_weight,
+                    residual=residual,
+                )
+            else:
+                shared_output = self._forward_shared_experts(
+                    hidden_states, gemm_output_zero_allocator
+                )
         # router_logits: (num_tokens, n_experts)
         router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
         if use_flashinfer_trtllm_bypass:
             topk_output = BypassedTopKOutput(
                 hidden_states=hidden_states,
                 router_logits=router_logits,
                 topk_config=self.topk.topk_config,
             )
~~~~

</details>


