"""Single DSV4.1 INT8 projection entry point for the AOT hipBLASLt backend.

The existing HIP and private Triton computations are unchanged. Native operators
are provided by the matching sglang-kernel common_ops build, not a JIT extension.
"""
import json
import os
import threading
import weakref
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

_OPERATORS = (
    "dsv41_int8_quantize_out",
    "dsv41_int8_gemm_out",
    "dsv41_int8_scale_out",
    "dsv41_int8_prepare",
    "dsv41_int8_gemm_packed",
    "dsv41_int8_scale_packed",
    "dsv41_int8_scale_scheduled",
)

class _NativeOps:
    def __init__(self, ops):
        self._ops = ops

    def quantize_out(self, x, quantized, scales):
        return self._ops.dsv41_int8_quantize_out(x, quantized, scales)


    def gemm_out(self, q, weight, out, groups, tile_n, group_major, algo_rank=0):
        return self._ops.dsv41_int8_gemm_out(
            q, weight, out, groups, tile_n, group_major, algo_rank
        )


    def scale_out(self, raw, x_scale, weight_scale, out, groups, group_major):
        return self._ops.dsv41_int8_scale_out(
            raw, x_scale, weight_scale, out, groups, group_major
        )


    def prepare(self, packed, m, groups, rank, k, batches, lda, stride, pitch, sharded, algo):
        return self._ops.dsv41_int8_prepare(
            packed, m, groups, rank, k, batches, lda, stride, pitch, sharded, algo
        )


    def gemm(self, q, packed, raw, plan):
        return self._ops.dsv41_int8_gemm_packed(q, packed, raw, plan)


    def scale(self, raw, x_scale, weight_scale, out, groups, batches, sharded, mode):
        return self._ops.dsv41_int8_scale_packed(
            raw, x_scale, weight_scale, out, groups, batches, sharded, mode
        )


    def scale_scheduled(self, raw, x_scale, weight_scale, out, mode, tile, threads):
        return self._ops.dsv41_int8_scale_scheduled(
            raw, x_scale, weight_scale, out, mode, tile, threads
        )


@lru_cache(maxsize=1)
def backend():
    try:
        import sgl_kernel  # Loads the installed common_ops registration.
    except ImportError as exc:
        raise RuntimeError("Rebuild the matching HCU sglang-kernel wheel with DSV4.1 INT8 support.") from exc
    ops = torch.ops.sgl_kernel
    missing = [name for name in _OPERATORS if not hasattr(ops, name)]
    if missing:
        raise RuntimeError("Rebuild the matching HCU sglang-kernel wheel; missing INT8 operators: " + ", ".join(missing))
    return _NativeOps(ops)

load_native = backend
_load = backend
_lock = threading.RLock()
_packs = {}
_plans = {}
_scale_seen = set()

_DEFAULT_SERVICE_CONFIG = {'algorithms': {'wo_a:1:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:5:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:128:4160:8:8192:1024:0': '79220000000000000000000000000000',
                'wo_a:256:4160:8:8192:1024:0': 'd4220000000000000000000000000000',
                'wo_a:8:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:16:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:32:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:64:4160:8:8192:1024:0': '79220000000000000000000000000000',
                'wo_a:96:4160:8:8192:1024:0': '79220000000000000000000000000000',
                'lm_head:1:5120:101:1280:6619136:0': 'e7220000000000000000000000000000',
                'lm_head:5:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:8:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:16:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:32:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:64:5136:101:1280:6619136:1': 'cf220000000000000000000000000000',
                'lm_head:96:5248:101:1280:6619136:0': 'd4220000000000000000000000000000',
                'lm_head:128:5152:101:1280:6619136:0': 'cf220000000000000000000000000000',
                'lm_head:256:5376:101:1280:6619136:0': 'df220000000000000000000000000000',
                'markov_tp8:1:256:1:16160:4136960:0': '8b220000000000000000000000000000',
                'markov_tp8:5:256:1:16160:4136960:0': '8d220000000000000000000000000000',
                'markov_tp8:32:256:1:16160:4136960:0': '0d230000000000000000000000000000',
                'markov_tp8:128:256:1:16160:4136960:0': 'cf220000000000000000000000000000',
                'markov_tp8:256:256:4:4040:1034240:0': '83220000000000000000000000000000',
                'markov_full:1:256:101:1280:327680:0': '87220000000000000000000000000000',
                'markov_full:5:256:101:1280:327680:0': '87220000000000000000000000000000',
                'markov_full:32:256:101:1280:327680:0': 'e6220000000000000000000000000000',
                'markov_full:128:256:101:1280:327680:0': 'cf220000000000000000000000000000',
                'markov_full:256:256:101:1280:327680:0': 'cf220000000000000000000000000000',
                'markov_tp8:256:320:4:4040:1034240:0': 'e0220000000000000000000000000000',
                'wo_a:6:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:10:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:12:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:15:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:18:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:20:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:24:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:25:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:30:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:35:4096:8:8192:1024:0': '79220000000000000000000000000000',
                'wo_a:36:4096:8:8192:1024:0': '79220000000000000000000000000000',
                'wo_a:40:4096:8:8192:1024:0': '79220000000000000000000000000000',
                'wo_a:42:4096:8:8192:1024:0': '79220000000000000000000000000000',
                'wo_a:48:4096:8:8192:1024:0': '79220000000000000000000000000000',
                'lm_head:6:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:10:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:12:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:15:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:18:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:20:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:24:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:25:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:30:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:35:5120:101:1280:6619136:1': 'de220000000000000000000000000000',
                'lm_head:36:5120:101:1280:6619136:1': 'de220000000000000000000000000000',
                'lm_head:40:5120:101:1280:6619136:1': 'de220000000000000000000000000000',
                'lm_head:42:5120:101:1280:6619136:1': 'de220000000000000000000000000000',
                'lm_head:48:5120:101:1280:6619136:1': 'de220000000000000000000000000000',
                'markov_full:2:256:101:1280:327680:0': '88220000000000000000000000000000',
                'markov_full:3:256:101:1280:327680:0': '88220000000000000000000000000000',
                'markov_full:4:256:101:1280:327680:0': '87220000000000000000000000000000',
                'markov_full:6:256:101:1280:327680:0': '87220000000000000000000000000000',
                'markov_full:7:256:101:1280:327680:0': '87220000000000000000000000000000',
                'markov_full:8:256:101:1280:327680:0': '87220000000000000000000000000000',
                'wo_a:1:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:5:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:6:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:8:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:10:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:12:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:15:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:16:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:18:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:20:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:24:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:25:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:30:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:32:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:35:4096:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:36:4096:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:40:4096:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:42:4096:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:48:4096:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:64:4160:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:96:4160:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:128:4160:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:256:4160:8:1024:4325376:0': 'd4220000000000000000000000000000',
                'markov_full:1:256:1010:128:32768:0': '88220000000000000000000000000000',
                'markov_full:2:256:1010:128:32768:0': '87220000000000000000000000000000',
                'markov_full:3:256:1010:128:32768:0': '87220000000000000000000000000000',
                'markov_full:4:256:1010:128:32768:0': '87220000000000000000000000000000',
                'markov_full:5:256:1010:128:32768:0': '87220000000000000000000000000000',
                'markov_full:6:256:1010:128:32768:0': '87220000000000000000000000000000',
                'markov_full:7:256:1010:128:32768:0': '87220000000000000000000000000000',
                'markov_full:8:256:1010:128:32768:0': '87220000000000000000000000000000',
                'wo_a:36:4224:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:40:4224:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:42:4224:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:48:4224:8:1024:4325376:0': '79220000000000000000000000000000'},
 'scale_schedules': {'lm_head': {'10': [1, 1, 320],
                                 '18': [1, 1, 320],
                                 '24': [4, 2, 128],
                                 '48': [4, 2, 128]}}}

@lru_cache(maxsize=1)
def _service_config():
    path = os.environ.get("SGLANG_DSV41_INT8_SERVICE_ALGO_CONFIG")
    return json.loads(Path(path).read_text()) if path else _DEFAULT_SERVICE_CONFIG

@lru_cache(maxsize=1)
def _algorithms():
    return _service_config()["algorithms"]

@lru_cache(maxsize=1)
def _scale_schedules():
    return _service_config().get("scale_schedules", {}).get("lm_head", {})

@triton.jit
def _scale_int32(ACC, XS, WS, Y, T:tl.constexpr, G:tl.constexpr,
                 R:tl.constexpr, GROUP_MAJOR:tl.constexpr, B:tl.constexpr):
    j=tl.program_id(0)*B+tl.arange(0,B)
    valid=j<T*G*R
    row=j//R
    group=row%G
    token=row//G
    col=j%R
    if GROUP_MAJOR:
        raw=(group*T+token)*R+col
    else:
        raw=j
    acc=tl.load(ACC+raw,valid,other=0).to(tl.float32)
    xs=tl.load(XS+row,valid,other=0)
    ws=tl.load(WS+group*R+col,valid,other=0)
    tl.store(Y+j,acc*xs*ws,valid)


@triton.jit
def _scale_grid(ACC,XS,WS,Y,M:tl.constexpr,N:tl.constexpr,WIDTH:tl.constexpr,BLOCK:tl.constexpr):
 local=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);row=tl.program_id(1);group=tl.program_id(2);col=group*WIDTH+local;valid=local<WIDTH
 a=tl.load(ACC+(group*M+row)*WIDTH+local,valid,0).to(tl.float32);xs=tl.load(XS+row);ws=tl.load(WS+col,valid,0);tl.store(Y+row*N+col,(a*xs)*ws,valid)


def quantize(x):
    ops = load_native()
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("HIP INT8 quantization requires BF16/FP16/FP32 activations")
    x = x.contiguous()
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1] + (1,), dtype=torch.float32, device=x.device)
    ops.quantize_out(x, q, scales)
    return q, scales


@lru_cache(maxsize=4)
def _algorithm_config(path):
    with open(path) as f:
        return json.load(f)["algorithms"]


def _project_unpacked(x, w, s, groups, rank, *, output_dtype=None, tile_n=None, group_major=False, algo_rank=None):
    if (
        x.ndim != 3 or x.shape[1] != groups or groups <= 0 or rank <= 0
        or w.shape != (groups * rank, x.shape[-1])
        or w.dtype != torch.int8 or not w.is_contiguous()
        or s.shape != (groups * rank, 1)
        or s.dtype != torch.float32 or not s.is_contiguous()
        or x.device != w.device or x.device != s.device
    ):
        raise ValueError("Invalid channelwise INT8 projection tensors")
    output_dtype = output_dtype or x.dtype
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("HIP projection output requires BF16/FP16/FP32")
    if tile_n is None:
        tile_n = int(os.environ.get("SGLANG_DSV41_INT8_BLASLT_TILE_N", "0")) if groups == 1 else 0
    if tile_n < 0:
        raise ValueError("INT8 BLASLt tile size must be non-negative")
    ops = load_native()
    t = x.shape[0]
    if algo_rank is None:
        config = os.environ.get("SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG")
        key = f"{t}:{groups}:{rank}:{x.shape[-1]}:{tile_n}:{int(group_major)}"
        algo_rank = int(_algorithm_config(config).get(key, 0)) if config else 0
    y = torch.empty((t, groups, rank), device=x.device, dtype=output_dtype)
    if not t:
        return y
    q, xs = quantize(x)
    if group_major:
        q = q.transpose(0, 1).contiguous()
    raw = torch.empty((groups, t, rank) if group_major else (t, groups, rank),
                      device=x.device, dtype=torch.int32)
    ops.gemm_out(q, w, raw, groups, tile_n, group_major, algo_rank)
    ops.scale_out(raw, xs, s, y, groups, group_major)
    return y


def _scheduled_scale(raw,xs,ws,out,m,schedule):
 mode,tile,threads=map(int,schedule);key=(raw.device.index,m,mode,tile,threads)
 if key not in _scale_seen:
  with _lock:
   if key not in _scale_seen:
    print('BLASLT_OPT_SCALE',json.dumps(dict(role='lm_head',m=m,mode=mode,tile=tile,threads=threads,output_dtype=str(out.dtype),device=raw.device.index)),flush=True);_scale_seen.add(key)
 _load().scale_scheduled(raw,xs,ws,out,mode,tile,threads)


def _version(w):
 try:return w._version
 except RuntimeError:return -1


def _role(g,n,k):
 if (g,n,k)==(8,1024,4096):return 'wo_a'
 if (g,n,k)==(1,129280,5120):return 'lm_head'
 if g==1 and k==256 and n in (16160,129280):return 'markov_tp8' if n==16160 else 'markov_full'
 return None


def _layout(role,m,g,n,k):
 if role=='wo_a':return g,n,k*n+131072,4224 if m in (36,40,42,48) else (k+64 if m>=64 else k),False
 if role=='lm_head':
  pitch=5120 if m<64 else (5136 if m==64 else (5248 if m<=96 else (5152 if m<=128 else 5376)))
  return 101,1280,6619136,pitch,1<m<=64
 batches=(1010 if m<=8 else 101) if role=='markov_full' else (4 if m>128 else 1)
 return batches,n//batches,k*(n//batches),320 if role=='markov_tp8' and m==256 else k,False


def _pack(w,g,n,k,batches,lda,astride):
 key=(w.device.index,w.data_ptr(),tuple(w.shape),_version(w),g,n,k,batches,lda,astride)
 with _lock:
  found=_packs.get(key)
  if found is not None:return found['packed']
  if torch.cuda.is_current_stream_capturing():raise RuntimeError('INT8 packed weights must be prepared before graph capture')
  width=g*n//batches;span=(batches-1)*astride+(k-1)*lda+width
  storage=torch.zeros(span,device=w.device,dtype=torch.int8)
  if g>1:storage.as_strided((k,g,n),(lda,astride,1)).copy_(w.reshape(g,n,k).permute(2,0,1))
  else:storage.as_strided((batches,k,width),(astride,lda,1)).copy_(w.reshape(batches,width,k).transpose(1,2))
  packed=storage.view(1,1,-1);_packs[key]=dict(owner=w,packed=packed)
  print('BLASLT_OPT_PACK',json.dumps(dict(groups=g,rank=n,k=k,batches=batches,lda=lda,stride=astride,bytes=span)),flush=True)
  return packed


def _prepare_packed_weight(w):
 if os.environ.get('SGLANG_DSV41_INT8_SERVICE_OPT')!='1' or not w.is_cuda:return
 if w.dtype!=torch.int8 or not w.is_contiguous() or w.ndim!=2:return
 n0,k=w.shape;g=8 if (n0,k)==(8192,4096) else 1;n=n0//g;role=_role(g,n,k)
 if role is None:return
 for m in ([1,256] if role in ('markov_tp8','markov_full') else [1]):
  batches,lda,astride,_,_=_layout(role,m,g,n,k);_pack(w,g,n,k,batches,lda,astride)


def release_weight_cache(weight):
 """Drop only this full weight's packed copies before compact VP releases it.

 The native plans own descriptors, not tensors. Other projection weights and
 their packed copies remain resident. Call only before graph capture.
 """
 if weight.is_cuda and torch.cuda.is_current_stream_capturing():
  raise RuntimeError('Cannot release INT8 weight cache during graph capture')
 with _lock:
  keys=[key for key,value in _packs.items() if value['owner'] is weight]
  for key in keys:del _packs[key]
 return len(keys)


def _project_packed(x,w,s,groups,rank,*,output_dtype=None,tile_n=None,group_major=False,algo_rank=None):
 from sglang.kernels.ops.quantization import int8_kernel as quant
 k=x.shape[-1];m=x.shape[0];g=groups;n=rank;role=_role(g,n,k);dt=output_dtype or x.dtype
 if role is None or tile_n not in (None,0) or group_major or algo_rank is not None:
  return _project_unpacked(x,w,s,g,n,output_dtype=dt,tile_n=tile_n,group_major=group_major,algo_rank=algo_rank)
 if x.ndim!=3 or x.shape[1]!=g or w.shape!=(g*n,k) or w.dtype!=torch.int8 or s.shape!=(g*n,1) or s.dtype!=torch.float32 or not w.is_contiguous() or not s.is_contiguous() or x.device!=w.device or x.device!=s.device:raise ValueError('Invalid optimized INT8 projection tensors')
 if x.dtype not in (torch.bfloat16,torch.float16,torch.float32) or dt not in (torch.bfloat16,torch.float16,torch.float32):raise ValueError('Floating activations/output required')
 # Preserve caller-requested output dtype; default follows activation dtype.
 y=torch.empty((m,g,n),device=x.device,dtype=dt)
 if m==0:return y
 x=x.contiguous();batches,lda,astride,pitch,shard=_layout(role,m,g,n,k);packed=_pack(w,g,n,k,batches,lda,astride)
 ops=_load();aid=_algorithms().get('%s:%s:%s:%s:%s:%s:%s'%(role,m,pitch,batches,lda,astride,int(shard)),'')
 key=(w.device.index,m,g,n,k,batches,lda,astride,pitch,shard,aid)
 with _lock:
  plan=_plans.get(key)
  if plan is None:
   plan=ops.prepare(packed,m,g,n,k,batches,lda,astride,pitch,shard,aid);_plans[key]=plan
   print('BLASLT_OPT_PLAN',json.dumps(dict(role=role,m=m,groups=g,rank=n,k=k,batches=batches,lda=lda,stride=astride,q_pitch=pitch,shard=shard,algorithm=aid or 'heuristic',input_dtype=str(x.dtype),output_dtype=str(dt))),flush=True)
 qstorage=torch.empty((m,g,pitch),device=x.device,dtype=torch.int8);q=qstorage[:,:,:k];xs=torch.empty((m,g,1),device=x.device,dtype=torch.float32);raw=torch.empty((m,g,n),device=x.device,dtype=torch.int32)
 block=triton.next_power_of_2(k)
 quant._per_token_quant_int8[(m*g,)](x,q,xs,None,stride_x=k,stride_xq=pitch,N=k,CAL_SUM=False,BLOCK=block,IS_HIP=quant._is_hip,num_warps=min(max(block//256,1),8),num_stages=1)
 ops.gemm(q,packed,raw,plan)
 if shard and dt!=torch.float32:
  _scale_grid[(triton.cdiv(n//batches,2048),m,batches)](raw,xs,s,y,m,n,n//batches,2048,num_warps=4,enable_fp_fusion=False)
 elif shard:
  schedule=_scale_schedules().get(str(m))
  if schedule is not None:_scheduled_scale(raw,xs,s,y,m,schedule)
  elif m<=5:_scale_grid[(triton.cdiv(n//batches,2048),m,batches)](raw,xs,s,y,m,n,n//batches,2048,num_warps=4,enable_fp_fusion=False)
  else:ops.scale(raw,xs,s,y,g,batches,True,1)
 elif role=='wo_a' or dt==torch.float16 or (role=='markov_tp8') or (role=='markov_full' and m<=5) or (role=='lm_head' and m>=256):
  sb=2048 if ((role=='wo_a' and m in (16,32,96,128)) or (role=='lm_head' and m>=256) or (role=='markov_tp8' and m>128)) else 1024
  _scale_int32[(triton.cdiv(m*g*n,sb),)](raw,xs,s,y,m,g,n,False,sb,num_warps=8 if sb==2048 else 4,enable_fp_fusion=False)
 else:ops.scale(raw,xs,s,y,g,batches,False,0)
 return y


def prepare_weight(w):
    if os.environ.get("SGLANG_DSV41_INT8_SERVICE_OPT") == "1":
        _prepare_packed_weight(w)


def project(x, w, s, groups, rank, **kwargs):
    if os.environ.get("SGLANG_DSV41_INT8_SERVICE_OPT") == "1":
        return _project_packed(x, w, s, groups, rank, **kwargs)
    return _project_unpacked(x, w, s, groups, rank, **kwargs)
