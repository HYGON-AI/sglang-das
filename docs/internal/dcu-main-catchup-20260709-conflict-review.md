# Official Main Catch-up 20260709 — Code Conflict Review

> Scope: only the 11 files that produced textual merge conflicts. The conflict ledger and all automatically merged files are intentionally excluded.
> View in VS Code with **Markdown: Open Preview** (`Ctrl+Shift+V`). The `diff` blocks render removed conflict state in red and the final resolved code in green.

## Comparison

- DCU parent (`ours`): `b654e63e9815446a27eaf883abf7bf9b9e5e24d8`
- Common official base: `9a6f8e599204aa37481f5f37a1b20938aee98d5c`
- Official endpoint (`theirs`): `bd7e54d7379e437cf5f027382d6ca214e046626b`
- Resolved merge: `f4d00bcaae4dd4288fcc206fb70ac27a7211ed3d`
- Reconstructed textual conflicts: 11 files, 20 hunks

Each section reconstructs Git’s three-way auto-conflict text from the two merge parents and common base, then compares it with the committed resolution. Lines beginning with `-` belong to the unresolved auto-conflict state; lines beginning with `+` are the final resolution.

## Conflict files

<details>
<summary><code>python/sglang/srt/layers/attention/dsa/dsa_indexer.py</code> — 4 conflict hunks</summary>

**Resolution intent:** Apply official DeepGEMM head padding to generic FP8 logits while preserving DCU BF16, LightOp, and CP-ragged paths.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
+++ RESOLVED/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
@@ -1386,23 +1386,17 @@
                         scale.view(torch.float32).flatten(),
                         True,
                     )
                 else:
                     q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(
                         q_fp8[:q_offset], weights[:q_offset]
                     )
                     logits = deep_gemm.fp8_mqa_logits(
-<<<<<<< DCU main@b654e63e9815
-                        q[:q_offset],
-||||||| official previous@9a6f8e599204
-                        q_fp8[:q_offset],
-=======
                         q_padded,
->>>>>>> official target@bd7e54d7379e
                         kv_fp8,
                         w_padded,
                         ks,
                         ke,
                         clean_logits=False,
                     )

             assert logits.shape[0] == len(seq_lens_expanded)
@@ -1475,23 +1469,17 @@
                         ke[start:end],
                         scale
                     )
                 else:
                     q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(
                         q_fp8[start:end], weights[start:end]
                     )
                     logits_chunk = deep_gemm.fp8_mqa_logits(
-<<<<<<< DCU main@b654e63e9815
-                        q[start:end],
-||||||| official previous@9a6f8e599204
-                        q_fp8[start:end],
-=======
                         q_padded,
->>>>>>> official target@bd7e54d7379e
                         kv_fp8,
                         w_padded,
                         ks[start:end],
                         ke[start:end],
                         clean_logits=False,
                     )


@@ -1696,17 +1684,16 @@
                 actual_seq_q_list.append(actual_seq_q)
                 batch_idx_list.append(batch_idx)

             ks = torch.cat(ks_list, dim=0)
             ke_offset = torch.cat(ke_offset_list, dim=0)
             ke = ks + ke_offset
             actual_seq_q = torch.cat(actual_seq_q_list, dim=0)
             with self._with_real_sm_count():
-<<<<<<< DCU main@b654e63e9815
                 if use_bf16_index_cache:
                     kv_bf16 = torch.cat(k_bf16_list, dim=0)
                     # CP ragged BF16 path also bypasses fp8 packing: concatenate the
                     # gathered BF16 K chunks, then call mqa_logits with scale=None.
                     logits = op.mqa_logits(
                         q,
                         kv_bf16,
                         weights.to(torch.float32),
@@ -1716,46 +1703,43 @@
                         kv_bf16.shape[0],
                         q.shape[1],
                         q.shape[2],
                         None,
                         True,
                     )
                 else:
                     k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
-                    k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
+                    k_scale = (
+                        torch.cat(k_scale_list, dim=0)
+                        .view(torch.float32)
+                        .squeeze(-1)
+                    )
                     kv_fp8 = (k_fp8, k_scale)
-                    logits = deep_gemm.fp8_mqa_logits(
-                        q,
-                        kv_fp8,
-                        weights,
-                        ks,
-                        ke,
-                        clean_logits=False,
-                    )
-||||||| official previous@9a6f8e599204
-                logits = deep_gemm.fp8_mqa_logits(
-                    q_fp8,
-                    kv_fp8,
-                    weights,
-                    ks,
-                    ke,
-                    clean_logits=False,
-                )
-=======
-                q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(q_fp8, weights)
-                logits = deep_gemm.fp8_mqa_logits(
-                    q_padded,
-                    kv_fp8,
-                    w_padded,
-                    ks,
-                    ke,
-                    clean_logits=False,
-                )
->>>>>>> official target@bd7e54d7379e
+                    if _is_dcu:
+                        logits = deep_gemm.fp8_mqa_logits(
+                            q,
+                            kv_fp8,
+                            weights,
+                            ks,
+                            ke,
+                            clean_logits=False,
+                        )
+                    else:
+                        q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(
+                            q, weights
+                        )
+                        logits = deep_gemm.fp8_mqa_logits(
+                            q_padded,
+                            kv_fp8,
+                            w_padded,
+                            ks,
+                            ke,
+                            clean_logits=False,
+                        )
             topk_result = metadata.topk_transform(
                 logits,
                 self.index_topk,
                 ks=ks,
                 cu_seqlens_q=actual_seq_q,
                 ke_offset=ke_offset,
                 batch_idx_list=batch_idx_list,
             )
@@ -1784,17 +1768,16 @@
                 (kv_len - actual_seq_q) + 1,
                 kv_len + 1,
                 dtype=torch.int32,
                 device="cuda",
             )
             ke = ks + ke_offset

             with self._with_real_sm_count():
-<<<<<<< DCU main@b654e63e9815
                 if use_bf16_index_cache:
                     # Single-chunk CP ragged BF16 path mirrors the multi-chunk case:
                     # direct BF16 K input and no quant scale tensor.
                     logits = op.mqa_logits(
                         q,
                         k_fp8,
                         weights.to(torch.float32),
                         ks,
@@ -1802,68 +1785,36 @@
                         q.shape[0],
                         k_fp8.shape[0],
                         q.shape[1],
                         q.shape[2],
                         None,
                         True,
                     )
                 elif _is_dcu:
-                    k_scale = get_token_to_kv_pool().get_index_k_scale_continuous(
-                        layer_id,
-                        kv_len,
-                        block_tables[0],
-                    )
-                    k_fp8 = k_fp8.view(torch.float8_e4m3fn)
-                    k_scale = k_scale.view(torch.float32).squeeze(-1)
                     logits = lightop.mqa_logits(
                         q,
                         k_fp8,
                         weights,
                         ks,
                         ke,
                         k_scale,
                     )
                 else:
-                    k_scale = get_token_to_kv_pool().get_index_k_scale_continuous(
-                        layer_id,
-                        kv_len,
-                        block_tables[0],
-                    )
-                    k_fp8 = k_fp8.view(torch.float8_e4m3fn)
-                    k_scale = k_scale.view(torch.float32).squeeze(-1)
-                    kv_fp8 = (k_fp8, k_scale)
+                    q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(
+                        q, weights
+                    )
                     logits = deep_gemm.fp8_mqa_logits(
-                        q,
+                        q_padded,
                         kv_fp8,
-                        weights,
+                        w_padded,
                         ks,
                         ke,
                         clean_logits=False,
                     )
-||||||| official previous@9a6f8e599204
-                logits = deep_gemm.fp8_mqa_logits(
-                    q_fp8,
-                    kv_fp8,
-                    weights,
-                    ks,
-                    ke,
-                    clean_logits=False,
-                )
-=======
-                q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(q_fp8, weights)
-                logits = deep_gemm.fp8_mqa_logits(
-                    q_padded,
-                    kv_fp8,
-                    w_padded,
-                    ks,
-                    ke,
-                    clean_logits=False,
-                )
->>>>>>> official target@bd7e54d7379e
             actual_seq_q = torch.tensor([actual_seq_q], dtype=torch.int32).to(
                 device="cuda", non_blocking=True
             )
             topk_result = metadata.topk_transform(
                 logits,
                 self.index_topk,
                 ks=ks,
                 cu_seqlens_q=actual_seq_q,
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/dp_attention.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Add official runtime flags and retain the DCU platform import.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/dp_attention.py
+++ RESOLVED/python/sglang/srt/layers/dp_attention.py
@@ -24,24 +24,18 @@
     get_tensor_model_parallel_rank,
     get_tensor_model_parallel_world_size,
     get_tp_group,
     tensor_model_parallel_all_reduce,
 )
 from sglang.srt.distributed.device_communicators.pynccl_allocator import (
     use_symmetric_memory,
 )
-<<<<<<< DCU main@b654e63e9815
-from sglang.srt.utils import get_bool_env_var, is_hip, is_dcu
-||||||| official previous@9a6f8e599204
-from sglang.srt.utils import get_bool_env_var, is_hip
-=======
 from sglang.srt.runtime_context import get_flags
-from sglang.srt.utils import get_bool_env_var, is_hip
->>>>>>> official target@bd7e54d7379e
+from sglang.srt.utils import get_bool_env_var, is_dcu, is_hip

 if TYPE_CHECKING:
     from sglang.srt.configs.model_config import ModelConfig
     from sglang.srt.server_args import ServerArgs

 logger = logging.getLogger(__name__)

 if TYPE_CHECKING:
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/moe/ep_moe/layer.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Accept official runner deprecation routing without widening CUDA-only paths to HIP/DCU.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/moe/ep_moe/layer.py
+++ RESOLVED/python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -462,38 +462,16 @@
             quant_config, Fp8Config
         ):
             self.deprecate_flag = True
         elif (
             deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
             and envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
         ):
             self.deprecate_flag = True
-<<<<<<< DCU main@b654e63e9815
-||||||| official previous@9a6f8e599204
-        elif (
-            get_moe_runner_backend().is_flashinfer_cutedsl()
-            and quant_config is not None
-            and quant_config.get_name() == "modelopt_fp4"
-        ):
-            self.deprecate_flag = True
-        elif (
-            quant_config is None
-            and self.w13_weight.dtype == torch.bfloat16
-            and get_moe_runner_backend().is_deep_gemm()
-            and get_moe_a2a_backend().is_deepep()
-            and get_deepep_mode().enable_low_latency()
-            and not _is_npu
-            and not _is_hip
-        ):
-            assert (
-                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
-            ), "Unquantized DeepEP low-latency MoE requires DeepGEMM BF16"
-            self.deprecate_flag = True
-=======
         elif (
             get_moe_runner_backend().is_flashinfer_cutedsl()
             and quant_config is not None
             and quant_config.get_name() in ("modelopt_fp4", "modelopt_mixed")
         ):
             self.deprecate_flag = True
         elif (
             quant_config is None
@@ -503,17 +481,16 @@
             and get_deepep_mode().enable_low_latency()
             and not _is_npu
             and not _is_hip
         ):
             assert (
                 deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
             ), "Unquantized DeepEP low-latency MoE requires DeepGEMM BF16"
             self.deprecate_flag = True
->>>>>>> official target@bd7e54d7379e
         else:
             self.deprecate_flag = False

         if self.deprecate_flag:
             return

         if isinstance(quant_config, Fp8Config):
             self.use_block_quant = getattr(self.quant_method, "block_quant", False)
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/moe/token_dispatcher/deepep.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Retain the DCU group-GEMM dispatch ABI and use the official low-latency API elsewhere.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/moe/token_dispatcher/deepep.py
+++ RESOLVED/python/sglang/srt/layers/moe/token_dispatcher/deepep.py
@@ -768,78 +768,58 @@
         return deepep_output

     def _dispatch_core(
         self,
         hidden_states: torch.Tensor,
         topk_ids: torch.Tensor,
         topk_weights: torch.Tensor,
     ):
-        use_nvfp4 = use_fp8 = False
         input_global_scale = self.quant_config.get("input_global_scale", None)
-        bf16_dispatch = self.quant_config.get("bf16_dispatch", False)
-        if input_global_scale is not None:
-            use_nvfp4 = True
-        else:
-            backend = get_moe_runner_backend()
-            # BF16 dispatch is needed when:
-            #   - quant_config requests BF16 dispatch explicitly
-            #   - flashinfer_cutedsl: kernel quantizes to NVFP4 internally
-            #   - NPU with SGLANG_DEEPEP_BF16_DISPATCH: INT8 input + BF16 weight GMM not supported
-            #   - deep_gemm with SGLANG_DEEPEP_BF16_DISPATCH: user requests BF16 dispatch
-            need_bf16_dispatch = (
-                bf16_dispatch
-                or backend.is_flashinfer_cutedsl()
-                or (_is_npu and envs.SGLANG_DEEPEP_BF16_DISPATCH.get())
-                or (backend.is_deep_gemm() and envs.SGLANG_DEEPEP_BF16_DISPATCH.get())
-            )
-            if not need_bf16_dispatch:
-                use_fp8 = True

         # round_scale / use_ue8m0 are FP8-DeepGEMM specific; they cause DeepEP
         # to return int32-packed UE8M0 scales that don't feed the flashinfer
         # cutedsl kernel.
         fp8_deepgemm_scale_opts = (
             dict(
                 round_scale=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                 and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
                 use_ue8m0=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                 and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
             )
-            if use_fp8
+            if self.use_fp8
             else dict()
         )

         buffer = self._get_buffer()
         _deepep_precompile_tp_barrier()
-<<<<<<< DCU main@b654e63e9815
         if use_groupgemm:
             if _use_fp8_w8a8_moe:
                 packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
                     buffer.low_latency_dispatch(
                         hidden_states,
                         topk_ids,
                         topk_weights,
                         self.num_max_dispatch_tokens_per_rank,
                         self.num_experts,
-                        quant_type = 2,
+                        quant_type=2,
                         fp8_round_scale=False,
                         async_finish=not self.return_recv_hook,
                         return_recv_hook=self.return_recv_hook,
                     )
                 )
             elif _use_marlin_w16a16_moe:
                 packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
                     buffer.low_latency_dispatch(
                         hidden_states,
                         topk_ids,
                         topk_weights,
                         self.num_max_dispatch_tokens_per_rank,
                         self.num_experts,
-                        quant_type = 0,
+                        quant_type=0,
                         fp8_round_scale=False,
                         async_finish=not self.return_recv_hook,
                         return_recv_hook=self.return_recv_hook,
                     )
                 )
             else:
                 packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
                     buffer.low_latency_dispatch(
@@ -854,69 +834,30 @@
                         return_recv_hook=self.return_recv_hook,
                     )
                 )
         else:
             packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
                 buffer.low_latency_dispatch(
                     hidden_states,
                     topk_ids,
-                    topk_weights,
                     self.num_max_dispatch_tokens_per_rank,
                     self.num_experts,
-                    quant_type = 0,
-                    **(dict(use_nvfp4=True) if use_nvfp4 else dict()),
+                    use_fp8=self.use_fp8,
+                    **(dict(topk_weights=topk_weights) if _is_npu else dict()),
+                    **(dict(use_nvfp4=True) if self.use_nvfp4 else dict()),
                     **(
                         dict(x_global_scale=input_global_scale)
                         if input_global_scale is not None
                         else dict()
                     ),
                     async_finish=not self.return_recv_hook,
                     return_recv_hook=self.return_recv_hook,
-                    round_scale=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
-                    and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
-                    use_ue8m0=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
-                    and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
-                )
-||||||| official previous@9a6f8e599204
-        packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
-            buffer.low_latency_dispatch(
-                hidden_states,
-                topk_ids,
-                self.num_max_dispatch_tokens_per_rank,
-                self.num_experts,
-                use_fp8=self.use_fp8,
-                **(dict(use_nvfp4=True) if self.use_nvfp4 else dict()),
-                **(
-                    dict(x_global_scale=input_global_scale)
-                    if input_global_scale is not None
-                    else dict()
-                ),
-                async_finish=not self.return_recv_hook,
-                return_recv_hook=self.return_recv_hook,
-                **fp8_deepgemm_scale_opts,
-=======
-        packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
-            buffer.low_latency_dispatch(
-                hidden_states,
-                topk_ids,
-                self.num_max_dispatch_tokens_per_rank,
-                self.num_experts,
-                use_fp8=self.use_fp8,
-                **(dict(topk_weights=topk_weights) if _is_npu else dict()),
-                **(dict(use_nvfp4=True) if self.use_nvfp4 else dict()),
-                **(
-                    dict(x_global_scale=input_global_scale)
-                    if input_global_scale is not None
-                    else dict()
-                ),
-                async_finish=not self.return_recv_hook,
-                return_recv_hook=self.return_recv_hook,
-                **fp8_deepgemm_scale_opts,
->>>>>>> official target@bd7e54d7379e
+                    **fp8_deepgemm_scale_opts,
+                )
             )
         return packed_recv_hidden, self.packed_recv_count, event, hook

     def combine_a(
         self,
         hidden_states: torch.Tensor,
         topk_ids: torch.Tensor,
         topk_weights: torch.Tensor,
~~~~

</details>


<details>
<summary><code>python/sglang/srt/mem_cache/memory_pool.py</code> — 3 conflict hunks</summary>

**Resolution intent:** Adopt official KV descriptors and VMM backing while retaining DCU FA layout and copy strides.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/mem_cache/memory_pool.py
+++ RESOLVED/python/sglang/srt/mem_cache/memory_pool.py
@@ -1506,23 +1506,32 @@
             device=self.device,
         )
         self.v_data_ptrs = torch.tensor(
             [x.data_ptr() for x in self.v_buffer],
             dtype=torch.uint64,
             device=self.device,
         )
         self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
-        self.data_strides = torch.tensor(
-            [
-                np.prod(x.shape[1:]) * x.dtype.itemsize
-                for x in self.k_buffer + self.v_buffer
-            ],
-            device=self.device,
-        )
+        if _kv_layout_dcu_fa:
+            self.data_strides = torch.tensor(
+                [
+                    self.head_num * self.v_head_dim * x.dtype.itemsize
+                    for x in self.k_buffer + self.v_buffer
+                ],
+                device=self.device,
+            )
+        else:
+            self.data_strides = torch.tensor(
+                [
+                    np.prod(x.shape[1:]) * x.dtype.itemsize
+                    for x in self.k_buffer + self.v_buffer
+                ],
+                device=self.device,
+            )

     def _kv_buffer_shapes(self):
         """(k_shape, v_shape)"""
         if self.use_hnd:
             return (
                 (self.num_pages, self.head_num, self.page_size, self.head_dim),
                 (self.num_pages, self.head_num, self.page_size, self.v_head_dim),
             )
@@ -1553,65 +1562,17 @@
                         torch.zeros(
                             (page_num, self.head_num, self.v_head_dim, self.page_size),
                             dtype=self.store_dtype,
                             device=self.device,
                         )
                         for _ in range(self.layer_num)
                     ]
                 # The padded page (slot 0's page) absorbs dummy padded-token writes.
-<<<<<<< DCU main@b654e63e9815
-                elif self.use_hnd:
-                    k_shape = (
-                        self.num_pages,
-                        self.head_num,
-                        self.page_size,
-                        self.head_dim,
-                    )
-                    v_shape = (
-                        self.num_pages,
-                        self.head_num,
-                        self.page_size,
-                        self.v_head_dim,
-                    )
-                    self.k_buffer = [
-                        torch.zeros(k_shape, dtype=self.store_dtype, device=self.device)
-                        for _ in range(self.layer_num)
-                    ]
-                    self.v_buffer = [
-                        torch.zeros(v_shape, dtype=self.store_dtype, device=self.device)
-                        for _ in range(self.layer_num)
-                    ]
                 elif self.kv_cache_layout == "vectorized_5d":
-||||||| official previous@9a6f8e599204
-                if self.use_hnd:
-                    k_shape = (
-                        self.num_pages,
-                        self.head_num,
-                        self.page_size,
-                        self.head_dim,
-                    )
-                    v_shape = (
-                        self.num_pages,
-                        self.head_num,
-                        self.page_size,
-                        self.v_head_dim,
-                    )
-                    self.k_buffer = [
-                        torch.zeros(k_shape, dtype=self.store_dtype, device=self.device)
-                        for _ in range(self.layer_num)
-                    ]
-                    self.v_buffer = [
-                        torch.zeros(v_shape, dtype=self.store_dtype, device=self.device)
-                        for _ in range(self.layer_num)
-                    ]
-                elif self.kv_cache_layout == "vectorized_5d":
-=======
-                if self.kv_cache_layout == "vectorized_5d":
->>>>>>> official target@bd7e54d7379e
                     total_slots = self.size + self.page_size
                     num_blocks = total_slots // self.page_size
                     x = self._kv_vector_x
                     # K: (num_blocks, H, D_k // X, page, X)
                     self.k_buffer = [
                         torch.zeros(
                             (
                                 num_blocks,
@@ -1646,65 +1607,16 @@
                         torch.zeros(k_shape, dtype=self.store_dtype, device=self.device)
                         for _ in range(self.layer_num)
                     ]
                     self.v_buffer = [
                         torch.zeros(v_shape, dtype=self.store_dtype, device=self.device)
                         for _ in range(self.layer_num)
                     ]

-<<<<<<< DCU main@b654e63e9815
-        self.k_data_ptrs = torch.tensor(
-            [x.data_ptr() for x in self.k_buffer],
-            dtype=torch.uint64,
-            device=self.device,
-        )
-        self.v_data_ptrs = torch.tensor(
-            [x.data_ptr() for x in self.v_buffer],
-            dtype=torch.uint64,
-            device=self.device,
-        )
-        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
-
-        if _kv_layout_dcu_fa:
-            self.data_strides = torch.tensor(
-                [
-                    np.prod(self.head_num * self.v_head_dim) * x.dtype.itemsize
-                    for x in self.k_buffer + self.v_buffer
-                ],
-                device=self.device,
-            )
-        else:
-            self.data_strides = torch.tensor(
-                [
-                    np.prod(x.shape[1:]) * x.dtype.itemsize
-                    for x in self.k_buffer + self.v_buffer
-                ],
-                device=self.device,
-            )
-||||||| official previous@9a6f8e599204
-        self.k_data_ptrs = torch.tensor(
-            [x.data_ptr() for x in self.k_buffer],
-            dtype=torch.uint64,
-            device=self.device,
-        )
-        self.v_data_ptrs = torch.tensor(
-            [x.data_ptr() for x in self.v_buffer],
-            dtype=torch.uint64,
-            device=self.device,
-        )
-        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
-        self.data_strides = torch.tensor(
-            [
-                np.prod(x.shape[1:]) * x.dtype.itemsize
-                for x in self.k_buffer + self.v_buffer
-            ],
-            device=self.device,
-        )
-=======
     # -- post-capture VA backing (opt-in; overridable per layout) --------------

     def _build_kv_buffer_descs(self):
         """Per-buffer layout descriptors, k0..k(L-1) then v0..v(L-1). Drives both the
         CUDA-VMM post-capture backing and PD-transfer registration
         (get_contiguous_buf_infos). Override per layout."""
         itemsize = self.store_dtype.itemsize
         # Derive from the real buffers when they exist (covers arbitrary layouts,
@@ -1762,17 +1674,16 @@
     def _finalize_backing_tokens(self, final_num_tokens: int) -> None:
         """Token-count primitive shared by composite pools (e.g. SWA sub-pools)."""
         self._post_capture_owner.finalize(final_num_tokens)
         self.size = int(final_num_tokens)

     @property
     def post_capture_backed_bytes(self) -> int:
         return self._post_capture_owner.backed_bytes if self._post_capture_owner else 0
->>>>>>> official target@bd7e54d7379e

     def _clear_buffers(self):
         del self.k_buffer
         del self.v_buffer
         if self._post_capture_owner is not None:
             self._post_capture_owner.close()
             self._post_capture_owner = None

@@ -1796,85 +1707,23 @@
     def get_contiguous_buf_infos(self):
         """(ptrs, lens, item_lens) for PD KV transfer, derived from the descriptors.
         ``lens`` is the final span at the CURRENT serving size -- for a post-capture
         pool that is the physically-backed span, not the reserved VA upper bound."""
         assert not self.use_hnd, (
             "PD-disaggregation KV transfer assumes NHD slot-row layout; "
             "HND KV cache (SGLANG_USE_HND_KVCACHE) is not supported with disagg yet."
         )
-<<<<<<< DCU main@b654e63e9815
-        # layer_num x [seq_len, head_num, head_dim]
-        # layer_num x [page_num, page_size, head_num, head_dim]
-        kv_data_ptrs = [
-            self._get_key_buffer(i).data_ptr()
-            for i in range(self.start_layer, self.start_layer + self.layer_num)
-        ] + [
-            self._get_value_buffer(i).data_ptr()
-            for i in range(self.start_layer, self.start_layer + self.layer_num)
-        ]
-        kv_data_lens = [
-            self._get_key_buffer(i).nbytes
-            for i in range(self.start_layer, self.start_layer + self.layer_num)
-        ] + [
-            self._get_value_buffer(i).nbytes
-            for i in range(self.start_layer, self.start_layer + self.layer_num)
-        ]
-
-        if _kv_layout_dcu_fa:
-            kv_item_lens = [
-                self._get_key_buffer(i)[0].nbytes
-                for i in range(self.start_layer, self.start_layer + self.layer_num)
-            ] + [
-                self._get_value_buffer(i)[0].nbytes
-                for i in range(self.start_layer, self.start_layer + self.layer_num)
-            ]
-        else:
-            kv_item_lens = [
-                self._get_key_buffer(i)[0].nbytes * self.page_size
-                for i in range(self.start_layer, self.start_layer + self.layer_num)
-            ] + [
-                self._get_value_buffer(i)[0].nbytes * self.page_size
-                for i in range(self.start_layer, self.start_layer + self.layer_num)
-            ]
-        return kv_data_ptrs, kv_data_lens, kv_item_lens
-||||||| official previous@9a6f8e599204
-        # layer_num x [seq_len, head_num, head_dim]
-        # layer_num x [page_num, page_size, head_num, head_dim]
-        kv_data_ptrs = [
-            self._get_key_buffer(i).data_ptr()
-            for i in range(self.start_layer, self.start_layer + self.layer_num)
-        ] + [
-            self._get_value_buffer(i).data_ptr()
-            for i in range(self.start_layer, self.start_layer + self.layer_num)
-        ]
-        kv_data_lens = [
-            self._get_key_buffer(i).nbytes
-            for i in range(self.start_layer, self.start_layer + self.layer_num)
-        ] + [
-            self._get_value_buffer(i).nbytes
-            for i in range(self.start_layer, self.start_layer + self.layer_num)
-        ]
-        kv_item_lens = [
-            self._get_key_buffer(i)[0].nbytes * self.page_size
-            for i in range(self.start_layer, self.start_layer + self.layer_num)
-        ] + [
-            self._get_value_buffer(i)[0].nbytes * self.page_size
-            for i in range(self.start_layer, self.start_layer + self.layer_num)
-        ]
-        return kv_data_ptrs, kv_data_lens, kv_item_lens
-=======
         tensors = self._pd_registerable_tensors()
         ptrs = [t.data_ptr() for t in tensors]
         lens = [
             d.final_span_bytes(self.size, self.page_size) for d in self._kv_buffer_descs
         ]
         item_lens = [d.item_len_bytes(self.page_size) for d in self._kv_buffer_descs]
         return ptrs, lens, item_lens
->>>>>>> official target@bd7e54d7379e

     def get_cpu_copy(self, indices, mamba_indices=None):
         assert not self.use_hnd, (
             "CPU KV offload indexes by slot (NHD); HND KV cache "
             "(SGLANG_USE_HND_KVCACHE) is not supported with CPU offload yet."
         )
         current_platform.synchronize()
         kv_cache_cpu = []
~~~~

</details>


<details>
<summary><code>python/sglang/srt/model_executor/model_runner.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Use the canonical official chunked-prefix backend registry from server_args.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/model_executor/model_runner.py
+++ RESOLVED/python/sglang/srt/model_executor/model_runner.py
@@ -285,43 +285,16 @@
     "trtllm_mla",
     "tokenspeed_mla",
     "ascend",
     "dsa",
     "nsa",  # Deprecated alias for "dsa"
     "intel_xpu",
 ]

-<<<<<<< DCU main@b654e63e9815
-CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = [
-    "flashinfer",
-    "fa3",
-    "dcu_mla",
-    "fa4",
-    "flashmla",
-    "cutedsl_mla",
-    "cutlass_mla",
-    "trtllm_mla",
-    "tokenspeed_mla",
-    "dcu_mla",
-]
-||||||| official previous@9a6f8e599204
-CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = [
-    "flashinfer",
-    "fa3",
-    "fa4",
-    "flashmla",
-    "cutedsl_mla",
-    "cutlass_mla",
-    "trtllm_mla",
-    "tokenspeed_mla",
-]
-=======
->>>>>>> official target@bd7e54d7379e
-
 TORCH_DTYPE_TO_KV_CACHE_STR = {
     torch.float8_e4m3fn: "fp8_e4m3",
     torch.float8_e4m3fnuz: "fp8_e4m3",
     torch.float8_e5m2: "fp8_e5m2",
     torch.bfloat16: "bf16",
 }


~~~~

</details>


<details>
<summary><code>python/sglang/srt/models/bailing_moe.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Adopt named stream-pool leasing and preserve the DCU SBO stream requirement.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/models/bailing_moe.py
+++ RESOLVED/python/sglang/srt/models/bailing_moe.py
@@ -1189,24 +1189,17 @@
         quant_config: Optional[QuantizationConfig] = None,
         prefix: str = "",
     ):
         super().__init__()
         self.pp_group = get_pp_group()

         self.config = config
         self.quant_config = quant_config
-<<<<<<< DCU main@b654e63e9815
-        # config.num_hidden_layers = 10  # debug
-        alt_stream = torch.cuda.Stream() if _is_cuda or is_sbo_enabled() else None
-||||||| official previous@9a6f8e599204
-        alt_stream = torch.cuda.Stream() if _is_cuda else None
-=======
-        alt_stream = get_stream("alt") if _is_cuda else None
->>>>>>> official target@bd7e54d7379e
+        alt_stream = get_stream("alt") if _is_cuda or is_sbo_enabled() else None

         self.model = BailingMoEModel(
             config,
             quant_config,
             alt_stream=alt_stream,
             prefix=add_prefix("model", ""),
         )

~~~~

</details>


<details>
<summary><code>python/sglang/srt/models/deepseek_v4.py</code> — 3 conflict hunks</summary>

**Resolution intent:** Retain the DCU rotary helper and adopt official DSV4 WO-A FP8 quantization operands.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/models/deepseek_v4.py
+++ RESOLVED/python/sglang/srt/models/deepseek_v4.py
@@ -73,23 +73,17 @@
     is_dp_attention_enabled,
     is_dp_gatherv_active,
 )
 from sglang.srt.layers.layernorm import RMSNorm
 from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
 from sglang.srt.layers.logits_processor import LogitsProcessor
 from sglang.srt.layers.moe import get_moe_a2a_backend, should_use_dp_reduce_scatterv
 from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
-<<<<<<< DCU main@b654e63e9815
 from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton
-from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
-||||||| official previous@9a6f8e599204
-from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
-=======
->>>>>>> official target@bd7e54d7379e
 from sglang.srt.layers.rotary_embedding import get_rope_wrapper
 from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
 from sglang.srt.layers.utils.cp_utils import (
     cp_all_gather_rerange_output,
     cp_round_robin_input_ids,
     cp_split_and_rebuild_data,
     cp_split_and_rebuild_position,
     prepare_context_parallel_metadata,
@@ -1138,66 +1132,23 @@

         o = o.view(o.shape[0], self.n_local_groups, -1)

         if _FP8_WO_A_GEMM:
             import deepgemm as deep_gemm

             T, G, D = o.shape
             R = self.o_lora_rank
-<<<<<<< DCU main@b654e63e9815
-
-            o_fp8, o_s = sglang_per_token_group_quant_fp8(
-                o.reshape(T * G, D).contiguous(),
-                group_size=128,
-                scale_ue8m0=True,
-            )
-
-            lhs_fp8 = o_fp8.view(T, G, D)
-            lhs_scale = o_s.view(T, G, -1)
-
-            w = self.wo_a.weight
-            if tuple(w.shape) == (D, G * R):
-                rhs_fp8 = w.t().contiguous().view(G, R, D)
-            elif tuple(w.shape) == (G * R, D):
-                rhs_fp8 = w.contiguous().view(G, R, D)
-            else:
-                raise RuntimeError(
-                    f"unexpected wo_a.weight shape={tuple(w.shape)}, "
-                    f"expected {(D, G * R)} or {(G * R, D)}"
-                )
-
-            weight_scale = getattr(self.wo_a, "weight_scale", None)
-            if weight_scale is None:
-                weight_scale = self.wo_a.weight_scale_inv
-            rhs_scale = weight_scale.data.reshape(G, R).contiguous()
-
-||||||| official previous@9a6f8e599204
-            o_fp8, o_s = sglang_per_token_group_quant_fp8(
-                o.reshape(T * G, D).contiguous(),
-                group_size=128,
-                scale_ue8m0=True,
-            )
-=======
             o_fp8, o_s = sglang_per_token_group_quant_fp8_dsv4_wo_a(o)
->>>>>>> official target@bd7e54d7379e
             output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)

             deep_gemm.fp8_einsum(
                 "bhr,hdr->bhd",
-<<<<<<< DCU main@b654e63e9815
-                (lhs_fp8, lhs_scale),
-                (rhs_fp8, rhs_scale),
-||||||| official previous@9a6f8e599204
-                (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
-                (self.wo_a.weight.view(G, R, D), self.wo_a.weight_scale_inv.data),
-=======
                 (o_fp8, o_s),
                 (self.wo_a.weight.view(G, R, D), self.wo_a.weight_scale_inv.data),
->>>>>>> official target@bd7e54d7379e
                 output,
                 recipe=(1, 1, 128),
             )
             o = output
         else:
             wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
             o = torch.einsum("tgd,grd->tgr", o, wo_a)

~~~~

</details>


<details>
<summary><code>python/sglang/srt/speculative/draft_utils.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Combine official CPU helpers with retained DCU backend selection.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/speculative/draft_utils.py
+++ RESOLVED/python/sglang/srt/speculative/draft_utils.py
@@ -1,25 +1,20 @@
 import logging

 from sglang.srt.server_args import ServerArgs
-<<<<<<< DCU main@b654e63e9815
-from sglang.srt.utils.common import is_blackwell, is_dcu, is_hip, is_musa, is_npu
-||||||| official previous@9a6f8e599204
-from sglang.srt.utils.common import is_blackwell, is_hip, is_musa, is_npu
-=======
 from sglang.srt.utils.common import (
     cpu_has_amx_support,
     is_blackwell,
     is_cpu,
+    is_dcu,
     is_hip,
     is_musa,
     is_npu,
 )
->>>>>>> official target@bd7e54d7379e

 logger = logging.getLogger(__name__)


 class DraftBackendFactory:
     def __init__(
         self,
         server_args: ServerArgs,
~~~~

</details>


<details>
<summary><code>python/sglang/srt/speculative/triton_ops/cache_locs.py</code> — 3 conflict hunks</summary>

**Resolution intent:** Keep DCU cache-location wrappers and add official CPU dispatch.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/speculative/triton_ops/cache_locs.py
+++ RESOLVED/python/sglang/srt/speculative/triton_ops/cache_locs.py
@@ -1,62 +1,43 @@
 from __future__ import annotations

 import torch
 import triton
 import triton.language as tl

-<<<<<<< DCU main@b654e63e9815
 from sglang.srt.utils import (
     get_bool_env_var,
+    is_cpu,
     is_cuda,
     is_dcu,
     is_hip,
     is_musa,
     is_npu,
     is_xpu,
     next_power_of_2,
 )

-||||||| official previous@9a6f8e599204
-from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_xpu, next_power_of_2
-
-=======
-from sglang.srt.utils import (
-    is_cpu,
-    is_cuda,
-    is_hip,
-    is_musa,
-    is_npu,
-    is_xpu,
-    next_power_of_2,
-)
-
 _is_cpu = is_cpu()
->>>>>>> official target@bd7e54d7379e
 _is_cuda = is_cuda()
 _is_hip = is_hip()
 _is_dcu = is_dcu()
 _is_npu = is_npu()
 _is_musa = is_musa()
 _is_xpu = is_xpu()

-<<<<<<< DCU main@b654e63e9815
 if _is_dcu:
     from sgl_kernel.kvcacheio import (
         dcu_assign_extend_cache_locs,
         dcu_assign_req_to_token_pool,
     )

-||||||| official previous@9a6f8e599204
-=======
 if _is_cpu:
     from sgl_kernel import assign_extend_cache_locs_cpu, assign_req_to_token_pool_cpu

->>>>>>> official target@bd7e54d7379e

 @triton.jit
 def assign_req_to_token_pool(
     req_pool_indices,
     req_to_token,
     start_offset,
     end_offset,
     out_cache_loc,
@@ -91,44 +72,40 @@
 def assign_req_to_token_pool_func(
     req_pool_indices: torch.Tensor,
     req_to_token: torch.Tensor,
     start_offset: torch.Tensor,
     end_offset: torch.Tensor,
     out_cache_loc: torch.Tensor,
     batch_size: int,
 ):
-<<<<<<< DCU main@b654e63e9815
     if _is_dcu and get_bool_env_var(
         "SGLANG_ASSIGN_REQ_TO_TOKEN_POOL", default="true"
     ):
         dcu_assign_req_to_token_pool(
             req_pool_indices=req_pool_indices,
             req_to_token=req_to_token,
             allocate_lens=start_offset,
             new_allocate_lens=end_offset,
             out_cache_loc=out_cache_loc,
             shape=req_to_token.shape[1],
             bs=batch_size,
         )
         return

-||||||| official previous@9a6f8e599204
-=======
     if _is_cpu:
         assign_req_to_token_pool_cpu(
             req_pool_indices,
             req_to_token,
             start_offset,
             end_offset,
             out_cache_loc,
             req_to_token.shape[1],
         )
         return
->>>>>>> official target@bd7e54d7379e
     assign_req_to_token_pool[(batch_size,)](
         req_pool_indices,
         req_to_token,
         start_offset,
         end_offset,
         out_cache_loc,
         req_to_token.shape[1],
         next_power_of_2(batch_size),
~~~~

</details>


<details>
<summary><code>sgl-kernel/python/sgl_kernel/kvcacheio.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Keep DCU cache-location bindings and add the official CPU all-layer KV-copy binding.

~~~~diff
--- AUTO-CONFLICT/sgl-kernel/python/sgl_kernel/kvcacheio.py
+++ RESOLVED/sgl-kernel/python/sgl_kernel/kvcacheio.py
@@ -352,17 +352,16 @@
         src_indices,
         dst_indices,
         item_size,
         dst_layout_dim,
         num_layers,
         block_quota,
         num_warps_per_block,
     )
-<<<<<<< DCU main@b654e63e9815

 def dcu_assign_req_to_token_pool(
     req_pool_indices:torch.Tensor,
     req_to_token:torch.Tensor,
     allocate_lens:torch.Tensor,
     new_allocate_lens:torch.Tensor,
     out_cache_loc:torch.Tensor,
     shape:int,
@@ -440,25 +439,22 @@
 ):
     torch.ops.sgl_kernel.dcu_align_evict_mask_to_page_size(
         seq_lens,
         evict_mask,
         page_size,
         num_draft_tokens,
         bs,
     )
-||||||| official previous@9a6f8e599204
-=======


 def copy_all_layer_kv_cache_cpu(
     data_ptrs: torch.Tensor,
     strides: torch.Tensor,
     tgt_loc: torch.Tensor,
     src_loc: torch.Tensor,
 ):
     torch.ops.sgl_kernel.copy_all_layer_kv_cache_cpu(
         data_ptrs,
         strides,
         tgt_loc,
         src_loc,
     )
->>>>>>> official target@bd7e54d7379e
~~~~

</details>
