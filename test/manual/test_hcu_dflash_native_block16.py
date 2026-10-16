"""Run with HIP_VISIBLE_DEVICES=<free GPU> python test_hcu_dflash_native_block16.py."""
import importlib.util
import os
import unittest
from pathlib import Path
import torch

import ast, types
import sys
from unittest.mock import patch
import flash_attn.flash_attn_interface as vendor
root=Path(__file__).resolve().parents[2]
def load_function(path,name,namespace):
    tree=ast.parse(Path(path).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),"exec"),namespace)
    return namespace[name]
native=vendor.flash_attn_varlen_func
def forbidden(**kwargs):raise AssertionError("Triton attention was called")
namespace=dict(torch=torch,os=os,_is_hcu=True,_use_triton_vllm_fa=False,
    get_bool_env_var=lambda name:True,
    get_spec=lambda:types.SimpleNamespace(speculative_algorithm="DFLASH", speculative_num_draft_tokens=16, speculative_dflash_block_size=None),
    envs=types.SimpleNamespace(SGLANG_USE_QWEN_DSPARK=types.SimpleNamespace(get=lambda:False)),
    triton_vllm_flash_attn_varlen_func=forbidden,
    vllm_flash_attn_varlen_func_interface=vendor.vllm_flash_attn_varlen_func)
wrapped=load_function(root/"python/sglang/srt/layers/attention/flashattention_interface.py","vllm_flash_attn_varlen_func",namespace)
module=types.SimpleNamespace(triton_vllm_flash_attn_varlen_func=wrapped)

class PagedSWATest(unittest.TestCase):
    def check_case(
        self,
        lengths,
        qlen,
        window,
        causal,
        layout="legacy_bhsd",
        graph=False,
        target=False,
        kv_dtype=torch.float8_e5m2,
    ):
        torch.manual_seed(7)
        bs, hq, hk, d, page = len(lengths), (8 if target else 16), (1 if target else 4), (256 if target else 128), 64
        pages = (max(lengths) + page - 1) // page
        k = (torch.randn(bs * pages, page, hk, d, device="cuda") * .2).to(kv_dtype)
        v = (torch.randn_like(k, dtype=torch.float32) * .2).to(k.dtype)
        table = torch.randperm(bs * pages, device="cuda").to(torch.int32).view(bs, pages)
        q = torch.randn(bs * qlen, hq, d, device="cuda", dtype=torch.bfloat16) * .2
        seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        cuq = torch.arange(bs + 1, device="cuda", dtype=torch.int32) * qlen
        if layout == "legacy_bhsd":
            kk, vv = k.permute(0, 2, 1, 3).contiguous(), v.permute(0, 2, 3, 1).contiguous()
        elif layout == "bhsd":
            kk, vv = k.permute(0, 2, 1, 3).contiguous(), v.permute(0, 2, 1, 3).contiguous()
        else:
            kk, vv = k, v
        out = torch.empty_like(q)
        def run():
            return module.triton_vllm_flash_attn_varlen_func(q, kk, vv, cuq, qlen, seq, 0, d**-.5, causal, window, table, 2, None, None, None, layout, out)
        def reference():
            result = []
            for b, n in enumerate(seq.tolist()):
                ix = table[b].long()
                key = k[ix].flatten(0, 1)[:n].float().repeat_interleave(hq // hk, 1)
                val = v[ix].flatten(0, 1)[:n].float().repeat_interleave(hq // hk, 1)
                scores = torch.einsum("qhd,khd->hqk", q[b*qlen:(b+1)*qlen].float(), key) * d**-.5
                posq = n - qlen + torch.arange(qlen, device="cuda")[:, None]
                posk = torch.arange(n, device="cuda")[None]
                mask = torch.ones((qlen, n), device="cuda", dtype=torch.bool)
                if window[0] >= 0: mask &= posk >= posq - window[0]
                if window[1] >= 0: mask &= posk <= posq + window[1]
                if causal: mask &= posk <= posq
                scores.masked_fill_(~mask[None], -float("inf"))
                result.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1).nan_to_num(), val))
            return torch.cat(result).to(q.dtype)
        out = run()
        torch.testing.assert_close(out, reference(), atol=6e-4, rtol=.03)
        self.assertTrue(out.isfinite().all().item())
        if graph:
            for _ in range(2): run()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): out = run()
            for new_lengths in ([0, 8192], [64, 8192], [256, 4128], lengths):
                seq.copy_(torch.tensor(new_lengths, device="cuda", dtype=seq.dtype))
                g.replay()
                torch.testing.assert_close(out, reference(), atol=6e-4, rtol=.03)

    @patch("torch.cuda.get_device_properties", return_value=types.SimpleNamespace(gcnArchName="gfx936"))
    def test_dispatch_is_opt_in(self, _props):
        # Replace the old route with a sentinel; do not execute Triton attention.
        sentinel = object()
        saved = {key: namespace[key] for key in (
            "get_bool_env_var", "get_spec", "triton_vllm_flash_attn_varlen_func"
        )}
        kwargs = dict(
            q=torch.empty(16, 16, 128, dtype=torch.bfloat16),
            k=torch.empty(1, 4, 64, 128, dtype=torch.float8_e5m2),
            v=torch.empty(1, 4, 128, 64, dtype=torch.float8_e5m2),
            cu_seqlens_q=torch.tensor([0, 16], dtype=torch.int32),
            max_seqlen_q=16, seqused_k=torch.tensor([16], dtype=torch.int32),
            max_seqlen_k=64, softmax_scale=128**-.5, causal=False,
            window_size=(4095, 4095), block_table=torch.zeros(1, 1, dtype=torch.int32),
            fa_version=2, q_descale=None, k_descale=None, v_descale=None,
            layout="legacy_bhsd",
        )
        try:
            namespace["triton_vllm_flash_attn_varlen_func"] = lambda **kw: sentinel
            for enabled, algorithm in ((False, "DFLASH"), (True, "EAGLE"), (True, "DSPARK")):
                namespace["get_bool_env_var"] = lambda name: enabled
                namespace["get_spec"] = lambda: types.SimpleNamespace(speculative_algorithm=algorithm, speculative_num_draft_tokens=16, speculative_dflash_block_size=None)
                self.assertIs(wrapped(**kwargs), sentinel)
        finally:
            namespace.update(saved)


    def test_native_window_boundaries(self):
        for n in (128,4096,4128,8192,21021):
            for causal in (False,True):
                for window in ((-1,-1),(4095,0 if causal else 4095)):
                    with self.subTest(n=n,causal=causal,window=window):
                        self.check_case([n],16,window,causal)


    def test_target_native_zero_hint(self):
        saved = namespace["get_spec"]
        try:
            for block in (8, 16):
                namespace["get_spec"] = lambda: types.SimpleNamespace(speculative_algorithm="DFLASH", speculative_num_draft_tokens=block, speculative_dflash_block_size=block)
                for n in (128, 4128, 21021):
                    self.check_case([n], block, (-1, -1), True, target=True)
                self.check_case([128, 21021], block, (-1, -1), True, graph=True, target=True)
        finally:
            namespace["get_spec"] = saved

    def test_native_graph_zero_hint(self):
        self.check_case([128,21021],16,(4095,4095),False,graph=True)
        self.check_case([128,21021],16,(-1,-1),True,graph=True)


    def test_unmodified_vendor_signature(self):
        import inspect
        self.assertNotIn("hcu_block16_paged", inspect.signature(vendor.vllm_flash_attn_varlen_func).parameters)

    def test_draft_small_blocks(self):
        saved = namespace["get_spec"]
        try:
            for block in (4, 8):
                namespace["get_spec"] = lambda: types.SimpleNamespace(speculative_algorithm="DFLASH", speculative_num_draft_tokens=block, speculative_dflash_block_size=block)
                for causal in (False, True):
                    self.check_case([128, 21021], block, (4095, 0 if causal else 4095), causal, graph=True)
        finally:
            namespace["get_spec"] = saved

    def test_bf16_kv_native(self):
        saved = namespace["get_spec"]
        try:
            for block, target in ((8, False), (8, True), (16, False), (16, True)):
                namespace["get_spec"] = lambda: types.SimpleNamespace(
                    speculative_algorithm="DFLASH",
                    speculative_num_draft_tokens=block,
                    speculative_dflash_block_size=block,
                )
                self.check_case(
                    [128, 8192],
                    block,
                    (-1, -1) if target else (4095, 4095),
                    target,
                    graph=True,
                    target=target,
                    kv_dtype=torch.bfloat16,
                )
        finally:
            namespace["get_spec"] = saved

    def test_target_dispatch_and_legacy_signature(self):
        saved = namespace.copy()
        try:
            for enabled, algorithm, block, arch, expected in (
                (True, "DFLASH", 8, "gfx936", 2),
                (True, "DFLASH", 16, "gfx936", 4),
                (True, "DFLASH", 4, "gfx936", 0),
                (False, "DFLASH", 16, "gfx936", 0),
                (True, "EAGLE", 16, "gfx936", 0),
                (True, "DSPARK", 16, "gfx936", 0),
                (True, None, 16, "gfx936", 0),
                (True, "DFLASH", 16, "gfx938", 0),
            ):
                namespace["get_bool_env_var"] = lambda name: enabled
                namespace["get_spec"] = lambda: types.SimpleNamespace(speculative_algorithm=algorithm, speculative_num_draft_tokens=None, speculative_dflash_block_size=block)
                legacy_calls = []
                def legacy(q, k, v, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal, window_size, block_table, fa_version, q_descale, k_descale, v_descale):
                    legacy_calls.append(max_seqlen_k)
                    return "legacy"
                namespace["vllm_flash_attn_varlen_func_interface"] = legacy
                with patch("torch.cuda.get_device_properties", return_value=types.SimpleNamespace(gcnArchName=arch)), patch.object(vendor.flash_attn_cuda, "paged_attention") as kernel:
                    wrapped(torch.empty(block,8,256,dtype=torch.bfloat16), torch.empty(1,1,64,256,dtype=torch.float8_e5m2), torch.empty(1,1,256,64,dtype=torch.float8_e5m2), torch.tensor([0,block]), block, torch.tensor([block]), 0, 256**-.5, True, (-1,-1), torch.zeros(1,2,dtype=torch.int32), 3, None,None,None,"legacy_bhsd")
                    self.assertEqual(kernel.call_count, expected)
                    if expected:
                        self.assertFalse(legacy_calls)
                        for call in kernel.call_args_list:
                            self.assertEqual(call.args[12], 128)
                    else:
                        self.assertEqual(legacy_calls, [0])
        finally:
            namespace.clear(); namespace.update(saved)

    def test_alias_selects_native(self):
        saved = namespace["get_spec"]
        try:
            namespace["get_spec"] = lambda: types.SimpleNamespace(speculative_algorithm="DFLASH", speculative_num_draft_tokens=None, speculative_dflash_block_size=16)
            self.check_case([128],16,(-1,-1),True,target=True)
        finally:
            namespace["get_spec"] = saved

if __name__ == "__main__": unittest.main()
