/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <memory>
#include <tuple>
#include <climits>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <mutex>
#include <vector>
#include <string>
#include <cstring>

// Keep the validated HIP scale arithmetic precise without changing flags for other kernels.
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif

namespace dsv41_unpacked {
// SPDX-License-Identifier: Apache-2.0
// Only the three DSV4.1 channelwise INT8 projections use this bridge.
namespace {
void check(hipblasStatus_t s, const char* op) {
  TORCH_CHECK(s == HIPBLAS_STATUS_SUCCESS, "DSV41 INT8 hipBLASLt ", op,
              " failed, status=", static_cast<int>(s));
}
void require_warmup(hipStream_t stream) {
  hipStreamCaptureStatus state;
  TORCH_CHECK(hipStreamIsCapturing(stream, &state) == hipSuccess,
              "Cannot query HIP capture state");
  TORCH_CHECK(state == hipStreamCaptureStatusNone,
              "Warm up the DSV41 hipBLASLt INT8 shape before graph capture");
}
struct Handle {
  hipblasLtHandle_t value = nullptr;
  Handle() { check(hipblasLtCreate(&value), "create"); }
  ~Handle() { if(value) hipblasLtDestroy(value); }
};
hipblasLtHandle_t handle_for(int device, hipStream_t stream) {
  thread_local std::map<int, std::unique_ptr<Handle>> handles;
  auto it = handles.find(device);
  if(it == handles.end()) {
    require_warmup(stream);
    it = handles.emplace(device, std::make_unique<Handle>()).first;
  }
  return it->second->value;
}
struct Plan {
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr,d=nullptr;
  hipblasLtMatmulPreference_t pref = nullptr;
  hipblasLtMatmulAlgo_t algo;
  Plan(hipblasLtHandle_t h, int64_t m,int64_t n,int64_t cols,int64_t k,
       int64_t g,bool group_major,int64_t algo_rank) {
    check(hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_32I,HIP_R_32I),"descriptor");
    hipblasOperation_t ta=HIPBLAS_OP_T,tb=HIPBLAS_OP_N;
    check(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)),"transA");
    check(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)),"transB");
    const int64_t ldx=group_major?k:g*k, ldy=group_major?n:g*n;
    check(hipblasLtMatrixLayoutCreate(&a,HIP_R_8I,k,cols,k),"A layout");
    check(hipblasLtMatrixLayoutCreate(&b,HIP_R_8I,k,m,ldx),"B layout");
    check(hipblasLtMatrixLayoutCreate(&c,HIP_R_32I,cols,m,ldy),"C layout");
    check(hipblasLtMatrixLayoutCreate(&d,HIP_R_32I,cols,m,ldy),"D layout");
    if(g>1) {
      int32_t count=static_cast<int32_t>(g);
      int64_t strides[4]={n*k,group_major?m*k:k,group_major?m*n:n,group_major?m*n:n};
      hipblasLtMatrixLayout_t layouts[4]={a,b,c,d};
      for(int i=0;i<4;i++) {
        check(hipblasLtMatrixLayoutSetAttribute(layouts[i],HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&count,sizeof(count)),"batch count");
        check(hipblasLtMatrixLayoutSetAttribute(layouts[i],HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&strides[i],sizeof(int64_t)),"batch stride");
      }
    }
    check(hipblasLtMatmulPreferenceCreate(&pref),"preference");
    uint64_t workspace=0;
    check(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace)),"workspace");
    hipblasLtMatmulHeuristicResult_t results[64]{};
    int count=0;
    const auto status=hipblasLtMatmulAlgoGetHeuristic(h,desc,a,b,c,d,pref,64,results,&count);
    TORCH_CHECK(status==HIPBLAS_STATUS_SUCCESS && count>algo_rank && results[algo_rank].state==HIPBLAS_STATUS_SUCCESS && results[algo_rank].workspaceSize==0,
      "No zero-workspace hipBLASLt INT8->INT32 algorithm: M=",m,", N=",cols,", K=",k,", G=",g,
      ", group_major=",group_major,", status=",static_cast<int>(status),", count=",count,
      ". Forced BLASLt mode never falls back to another GEMM.");
    algo=results[algo_rank].algo;
    const char* log=std::getenv("SGLANG_DSV41_INT8_BLASLT_LOG");
    if(log && log[0]=='1') std::cerr<<"[DSV41_BLASLT_INT8] M="<<m<<" N="<<cols<<" K="<<k<<" G="<<g
      <<" group_major="<<group_major<<" algo_rank="<<algo_rank<<" A/B=INT8 C/D=INT32 compute=INT32 workspace=0\n";
  }
  ~Plan() {
    if(a)hipblasLtMatrixLayoutDestroy(a);if(b)hipblasLtMatrixLayoutDestroy(b);
    if(c)hipblasLtMatrixLayoutDestroy(c);if(d)hipblasLtMatrixLayoutDestroy(d);
    if(pref)hipblasLtMatmulPreferenceDestroy(pref);if(desc)hipblasLtMatmulDescDestroy(desc);
  }
};
Plan& plan_for(int device,hipblasLtHandle_t h,hipStream_t stream,int64_t m,int64_t n,
               int64_t cols,int64_t k,int64_t g,bool group_major,int64_t algo_rank) {
  using Key=std::tuple<int,int64_t,int64_t,int64_t,int64_t,int64_t,bool,int64_t>;
  thread_local std::map<Key,std::unique_ptr<Plan>> plans;
  const Key key{device,m,n,cols,k,g,group_major,algo_rank};
  auto it=plans.find(key);
  if(it==plans.end()) {
    require_warmup(stream);
    it=plans.emplace(key,std::make_unique<Plan>(h,m,n,cols,k,g,group_major,algo_rank)).first;
  }
  return *it->second;
}
void gemm_out(const at::Tensor& q,const at::Tensor& w,at::Tensor out,
              int64_t groups,int64_t tile_n,bool group_major,int64_t algo_rank) {
  TORCH_CHECK(q.is_cuda()&&w.is_cuda()&&out.is_cuda()&&q.device()==w.device()&&q.device()==out.device(),"HCU tensors must share a device");
  TORCH_CHECK(q.scalar_type()==at::kChar&&w.scalar_type()==at::kChar&&out.scalar_type()==at::kInt,"Need INT8 A/B and INT32 output");
  TORCH_CHECK(q.dim()==3&&w.dim()==2&&out.dim()==3&&q.is_contiguous()&&w.is_contiguous()&&out.is_contiguous(),"Need contiguous activation/output and row-major weights");
  TORCH_CHECK(groups>0&&tile_n>=0&&algo_rank>=0&&algo_rank<64,"Invalid grouping/tile");
  const auto m=q.size(group_major?1:0),k=q.size(2);
  TORCH_CHECK(q.size(group_major?0:1)==groups&&w.size(1)==k&&w.size(0)%groups==0,"Invalid grouped input dimensions");
  const auto n=w.size(0)/groups;
  TORCH_CHECK(out.size(group_major?0:1)==groups&&out.size(group_major?1:0)==m&&out.size(2)==n,"Invalid output shape");
  TORCH_CHECK(k>0&&n>0&&k%4==0&&m<=INT_MAX&&n<=INT_MAX&&k<=INT_MAX&&groups<=INT_MAX,"Unsupported INT8 matrix dimensions");
  TORCH_CHECK(groups==1||tile_n==0,"Output-channel tiling is only for dense projections");
  if(m==0)return;
  c10::DeviceGuard guard(q.device());
  auto stream=c10::hip::getCurrentHIPStream(q.get_device()).stream();
  auto h=handle_for(q.get_device(),stream);
  const int32_t alpha=1,beta=0;
  const auto* xp=q.data_ptr<int8_t>();const auto* wp=w.data_ptr<int8_t>();
  auto* yp=out.data_ptr<int32_t>();
  const int64_t step=tile_n?tile_n:n;
  for(int64_t start=0;start<n;start+=step) {
    const auto cols=std::min(step,n-start);
    auto& p=plan_for(q.get_device(),h,stream,m,n,cols,k,groups,group_major,algo_rank);
    check(hipblasLtMatmul(h,p.desc,&alpha,wp+start*k,p.a,xp,p.b,&beta,yp+start,p.c,
                         yp+start,p.d,&p.algo,nullptr,0,stream),"matmul");
  }
}
}

}


namespace dsv41_quant {
// SPDX-License-Identifier: Apache-2.0
// HIP activation quantization and INT32 scale epilogue for DSV4.1 projections.
namespace {
__device__ inline float wave_max(float v) {
  for(int delta=warpSize/2;delta;delta>>=1)v=fmaxf(v,__shfl_xor(v,delta));
  return v;
}
template<typename T>
__global__ void quantize_kernel(const T* x,int8_t* q,float* scales,int64_t k) {
  const auto row=static_cast<int64_t>(blockIdx.x);
  float a=0.0f;
  for(int64_t j=threadIdx.x;j<k;j+=blockDim.x)a=fmaxf(a,fabsf(static_cast<float>(x[row*k+j])));
  a=wave_max(a);
  __shared__ float maxima[32];
  if(threadIdx.x%warpSize==0)maxima[threadIdx.x/warpSize]=a;
  __syncthreads();
  a=threadIdx.x<blockDim.x/warpSize?maxima[threadIdx.x]:0.0f;
  a=wave_max(a);
  if(threadIdx.x==0)maxima[0]=fmaxf(a,1.0e-10f);
  __syncthreads();
  const float absmax=maxima[0];
  const float scale=absmax/127.0f;
  const float inverse=127.0f/absmax;
  if(threadIdx.x==0)scales[row]=scale;
  for(int64_t j=threadIdx.x;j<k;j+=blockDim.x) {
    const float v=static_cast<float>(x[row*k+j])*inverse;
    q[row*k+j]=static_cast<int8_t>(static_cast<int32_t>(roundf(v)));
  }
}
template<typename T,int K,int THREADS>
__global__ __launch_bounds__(THREADS) void quantize_fixed_kernel(const T* x,int8_t* q,float* scales) {
  constexpr int CHUNKS=(K+THREADS-1)/THREADS;
  const auto row=static_cast<int64_t>(blockIdx.x);
  float values[CHUNKS];
  float a=0.0f;
#pragma unroll
  for(int i=0;i<CHUNKS;i++) {
    const int j=threadIdx.x+i*THREADS;
    const float v=j<K?static_cast<float>(x[row*K+j]):0.0f;
    values[i]=v;a=fmaxf(a,fabsf(v));
  }
  a=wave_max(a);
  __shared__ float maxima[32];
  if(threadIdx.x%warpSize==0)maxima[threadIdx.x/warpSize]=a;
  __syncthreads();
  a=threadIdx.x<THREADS/warpSize?maxima[threadIdx.x]:0.0f;
  a=wave_max(a);
  if(threadIdx.x==0)maxima[0]=fmaxf(a,1.0e-10f);
  __syncthreads();
  const float absmax=maxima[0];
  if(threadIdx.x==0)scales[row]=absmax/127.0f;
  const float inverse=127.0f/absmax;
#pragma unroll
  for(int i=0;i<CHUNKS;i++) {
    const int j=threadIdx.x+i*THREADS;
    if(j<K)q[row*K+j]=static_cast<int8_t>(static_cast<int32_t>(roundf(values[i]*inverse)));
  }
}
template<typename T>
void launch_quantize(const T* x,int8_t* q,float* scales,int64_t rows,int64_t k,hipStream_t stream) {
  if(k==256) {
    hipLaunchKernelGGL((quantize_fixed_kernel<T,256,256>),dim3(rows),dim3(256),0,stream,x,q,scales);
  } else if(k==4096) {
    hipLaunchKernelGGL((quantize_fixed_kernel<T,4096,512>),dim3(rows),dim3(512),0,stream,x,q,scales);
  } else if(k==5120) {
    hipLaunchKernelGGL((quantize_fixed_kernel<T,5120,512>),dim3(rows),dim3(512),0,stream,x,q,scales);
  } else {
    hipLaunchKernelGGL((quantize_kernel<T>),dim3(rows),dim3(256),0,stream,x,q,scales,k);
  }
}

template<typename T,bool GROUP_MAJOR>
__global__ void scale_kernel(const int32_t* acc,const float* xs,const float* ws,
                             T* out,int64_t count,int64_t m,int64_t g,int64_t n) {
  const auto col=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(col>=n)return;
  for(int64_t group=blockIdx.z;group<g;group+=gridDim.z) {
    for(int64_t token=blockIdx.y;token<m;token+=gridDim.y) {
      const auto row=token*g+group,j=row*n+col;
      const auto source=GROUP_MAJOR?(group*m+token)*n+col:j;
      const float a=static_cast<float>(acc[source]);
      const float v=(a*xs[row])*ws[group*n+col];
      out[j]=static_cast<T>(v);
    }
  }
}
void launch_check(const char* operation) {
  const auto e=hipGetLastError();
  TORCH_CHECK(e==hipSuccess,"DSV41 HIP ",operation," launch failed: ",hipGetErrorString(e));
}
}
void dsv41_blaslt_quantize_out(const at::Tensor& x,at::Tensor q,at::Tensor scales) {
  TORCH_CHECK(x.is_cuda()&&q.is_cuda()&&scales.is_cuda()&&x.device()==q.device()&&x.device()==scales.device(),"Quantization tensors must share an HCU device");
  TORCH_CHECK(x.dim()>=2&&x.is_contiguous()&&q.is_contiguous()&&scales.is_contiguous(),"Quantization requires contiguous tensors");
  TORCH_CHECK(q.scalar_type()==at::kChar&&scales.scalar_type()==at::kFloat&&q.sizes()==x.sizes(),"Need INT8 output and FP32 scales");
  const auto k=x.size(-1);
  TORCH_CHECK(k>0&&scales.numel()==x.numel()/k,"Invalid quantization scale shape");
  const auto rows=x.numel()/k;
  TORCH_CHECK(rows<=2147483647LL,"Too many quantization rows");
  if(!rows)return;
  c10::DeviceGuard guard(x.device());
  auto stream=c10::hip::getCurrentHIPStream(x.get_device()).stream();
  switch(x.scalar_type()) {
    case at::kBFloat16:launch_quantize(x.data_ptr<c10::BFloat16>(),q.data_ptr<int8_t>(),scales.data_ptr<float>(),rows,k,stream);break;
    case at::kHalf:launch_quantize(x.data_ptr<c10::Half>(),q.data_ptr<int8_t>(),scales.data_ptr<float>(),rows,k,stream);break;
    case at::kFloat:launch_quantize(x.data_ptr<float>(),q.data_ptr<int8_t>(),scales.data_ptr<float>(),rows,k,stream);break;
    default:TORCH_CHECK(false,"INT8 quantization supports BF16/FP16/FP32 input only");
  }
  launch_check("quantize");
}
void dsv41_blaslt_scale_out(const at::Tensor& acc,const at::Tensor& xs,const at::Tensor& ws,
                            at::Tensor out,int64_t groups,bool group_major) {
  TORCH_CHECK(acc.is_cuda()&&xs.is_cuda()&&ws.is_cuda()&&out.is_cuda()&&acc.device()==xs.device()&&acc.device()==ws.device()&&acc.device()==out.device(),"Scale tensors must share an HCU device");
  TORCH_CHECK(acc.scalar_type()==at::kInt&&xs.scalar_type()==at::kFloat&&ws.scalar_type()==at::kFloat,"Need INT32 accumulation and FP32 scales");
  TORCH_CHECK(acc.dim()==3&&out.dim()==3&&groups>0&&out.size(1)==groups,"Invalid grouped scale dimensions");
  TORCH_CHECK(acc.is_contiguous()&&xs.is_contiguous()&&ws.is_contiguous()&&out.is_contiguous(),"Scale tensors must be contiguous");
  const auto m=out.size(0),n=out.size(2),count=out.numel();
  TORCH_CHECK(xs.numel()==m*groups&&ws.numel()==groups*n&&acc.numel()==count&&acc.size(group_major?0:1)==groups&&acc.size(group_major?1:0)==m&&acc.size(2)==n,"Scale shapes do not match accumulation");
  if(!count)return;
  c10::DeviceGuard guard(out.device());
  auto stream=c10::hip::getCurrentHIPStream(out.get_device()).stream();
  const dim3 grid((n+255)/256,static_cast<unsigned>(m<65535?m:65535),static_cast<unsigned>(groups<65535?groups:65535));
#define LAUNCH(T) do { if(group_major) { hipLaunchKernelGGL((scale_kernel<T,true>),grid,dim3(256),0,stream,acc.data_ptr<int32_t>(),xs.data_ptr<float>(),ws.data_ptr<float>(),out.data_ptr<T>(),count,m,groups,n); } else { hipLaunchKernelGGL((scale_kernel<T,false>),grid,dim3(256),0,stream,acc.data_ptr<int32_t>(),xs.data_ptr<float>(),ws.data_ptr<float>(),out.data_ptr<T>(),count,m,groups,n); } } while(0)
  switch(out.scalar_type()) {
    case at::kBFloat16:LAUNCH(c10::BFloat16);break;
    case at::kHalf:LAUNCH(c10::Half);break;
    case at::kFloat:LAUNCH(float);break;
    default:TORCH_CHECK(false,"Scale output supports BF16/FP16/FP32 only");
  }
#undef LAUNCH
  launch_check("scale");
}

}
#undef LAUNCH

namespace dsv41_vector_scale {

struct alignas(8) BF4 {hip_bfloat16 a,b,c,d;};
template<bool FP32,int TILE,int THREADS>
__global__ __launch_bounds__(THREADS) void scale_vec_kernel(const int32_t* __restrict__ acc,const float* __restrict__ xs,const float* __restrict__ ws,void* __restrict__ out,int m,int g,int n) {
 int col=(blockIdx.x*THREADS+threadIdx.x)*4,group=blockIdx.z;
 if(col>=n)return;
 float4 w=*reinterpret_cast<const float4*>(ws+group*n+col);
 #pragma unroll
 for(int i=0;i<TILE;i++) {
  int token=blockIdx.y*TILE+i;if(token>=m)continue;
  int row=token*g+group;int64_t off=(int64_t)row*n+col;
  int4 a=*reinterpret_cast<const int4*>(acc+off);float s=xs[row];
  float4 v;
  v.x=(float(a.x)*s)*w.x;v.y=(float(a.y)*s)*w.y;v.z=(float(a.z)*s)*w.z;v.w=(float(a.w)*s)*w.w;
  if constexpr(FP32) *reinterpret_cast<float4*>(static_cast<float*>(out)+off)=v;
  else {
   BF4 b{hip_bfloat16(v.x),hip_bfloat16(v.y),hip_bfloat16(v.z),hip_bfloat16(v.w)};
   *reinterpret_cast<BF4*>(static_cast<hip_bfloat16*>(out)+off)=b;
  }
 }
}
template<bool F,int T,int TH>
void go(const int32_t* a,const float* x,const float* w,void* y,int m,int g,int n,hipStream_t s) {
 hipLaunchKernelGGL((scale_vec_kernel<F,T,TH>),dim3((n+TH*4-1)/(TH*4),(m+T-1)/T,g),dim3(TH),0,s,a,x,w,y,m,g,n);
}
extern "C" int scale_vec(const int32_t* a,const float* x,const float* w,void* y,int m,int g,int n,int fp32,int tile,int threads,hipStream_t s) {
 if(n%4||m<=0||g<=0)return 1;
 #define RUN(F,T) if(threads==128)go<F,T,128>(a,x,w,y,m,g,n,s);else go<F,T,256>(a,x,w,y,m,g,n,s)
 #define CASE(T) case T:if(fp32){RUN(true,T);}else{RUN(false,T);}break
 switch(tile){CASE(1);CASE(2);CASE(4);CASE(8);CASE(16);default:return 1;}
 return hipGetLastError();
}

}
#undef RUN
#undef CASE

namespace dsv41_sharded_scale {

template<int TILE,int THREADS>
__global__ __launch_bounds__(THREADS) void sharded_scale(const int32_t* acc,const float* xs,const float* ws,float* out,int m,int groups,int width){
 int row=blockIdx.y,group=blockIdx.z;float scale=xs[row];int n=groups*width;
 #pragma unroll
 for(int t=0;t<TILE;t++){
  int col=(blockIdx.x*THREADS*TILE+threadIdx.x+t*THREADS)*4;
  if(col<width){
   int4 a=reinterpret_cast<const int4*>(acc+(group*m+row)*width)[col/4];float4 w=reinterpret_cast<const float4*>(ws+group*width)[col/4];float4 y;
   y.x=(float(a.x)*scale)*w.x;y.y=(float(a.y)*scale)*w.y;y.z=(float(a.z)*scale)*w.z;y.w=(float(a.w)*scale)*w.w;
   reinterpret_cast<float4*>(out+row*n+group*width)[col/4]=y;
  }
 }
}
template<int T> int go(const int32_t*a,const float*x,const float*w,float*y,int m,int g,int width,int tile,hipStream_t stream){
 #define LAUNCH(V) case V:hipLaunchKernelGGL((sharded_scale<V,T>),dim3((width+T*4*V-1)/(T*4*V),m,g),dim3(T),0,stream,a,x,w,y,m,g,width);break
 switch(tile){LAUNCH(1);LAUNCH(2);LAUNCH(4);default:return 1;}
 return hipGetLastError();
}
extern "C" int scale_sharded(const int32_t*a,const float*x,const float*w,float*y,int m,int g,int width,int tile,int threads,hipStream_t stream){
 if(width%4||m<=0||g<=0)return 1;
 switch(threads){case 64:return go<64>(a,x,w,y,m,g,width,tile,stream);case 128:return go<128>(a,x,w,y,m,g,width,tile,stream);case 256:return go<256>(a,x,w,y,m,g,width,tile,stream);case 512:return go<512>(a,x,w,y,m,g,width,tile,stream);default:return 1;}
}

}
#undef LAUNCH

namespace dsv41_scheduled_scale {
template<int T,int TILE,bool GROUP_FAST>
__global__ __launch_bounds__(T) void by_group(const int32_t* acc,const float* xs,const float* ws,float* out,int m){
 int row=GROUP_FAST?blockIdx.z:blockIdx.y,group=GROUP_FAST?blockIdx.y:blockIdx.z;float scale=xs[row];
 #pragma unroll
 for(int t=0;t<TILE;t++){int v=blockIdx.x*T*TILE+threadIdx.x+t*T;if(v<320){int4 a=reinterpret_cast<const int4*>(acc)[(group*m+row)*320+v];float4 w=reinterpret_cast<const float4*>(ws)[group*320+v];float4 y;y.x=(float(a.x)*scale)*w.x;y.y=(float(a.y)*scale)*w.y;y.z=(float(a.z)*scale)*w.z;y.w=(float(a.w)*scale)*w.w;reinterpret_cast<float4*>(out)[row*32320+group*320+v]=y;}}
}
template<int T,int TILE>
__global__ __launch_bounds__(T) void flat_logits(const int32_t* acc,const float* xs,const float* ws,float* out,int m){
 int row=blockIdx.y;float scale=xs[row];
 #pragma unroll
 for(int t=0;t<TILE;t++){int col=blockIdx.x*T*TILE+threadIdx.x+t*T;if(col<32320){int group=col/320,local=col-group*320;int4 a=reinterpret_cast<const int4*>(acc)[(group*m+row)*320+local];float4 w=reinterpret_cast<const float4*>(ws)[col];float4 y;y.x=(float(a.x)*scale)*w.x;y.y=(float(a.y)*scale)*w.y;y.z=(float(a.z)*scale)*w.z;y.w=(float(a.w)*scale)*w.w;reinterpret_cast<float4*>(out)[row*32320+col]=y;}}
}
template<int T,int ROWS,bool GROUP_FAST>
__global__ __launch_bounds__(T) void multi_rows(const int32_t* acc,const float* xs,const float* ws,float* out,int m){
 int first=(GROUP_FAST?blockIdx.z:blockIdx.y)*ROWS,group=GROUP_FAST?blockIdx.y:blockIdx.z;int v=blockIdx.x*T+threadIdx.x;
 if(v<320){float4 w=reinterpret_cast<const float4*>(ws)[group*320+v];
 #pragma unroll
 for(int r=0;r<ROWS;r++){int row=first+r;if(row<m){float scale=xs[row];int4 a=reinterpret_cast<const int4*>(acc)[(group*m+row)*320+v];float4 y;y.x=(float(a.x)*scale)*w.x;y.y=(float(a.y)*scale)*w.y;y.z=(float(a.z)*scale)*w.z;y.w=(float(a.w)*scale)*w.w;reinterpret_cast<float4*>(out)[row*32320+group*320+v]=y;}}}
}
template<int T>int go(const int32_t*a,const float*x,const float*w,float*y,int m,int mode,int tile,hipStream_t stream){
 #define BY(V,F) case V:hipLaunchKernelGGL((by_group<T,V,F>),dim3((320+T*V-1)/(T*V),F?101:m,F?m:101),dim3(T),0,stream,a,x,w,y,m);break
 #define FL(V) case V:hipLaunchKernelGGL((flat_logits<T,V>),dim3((32320+T*V-1)/(T*V),m),dim3(T),0,stream,a,x,w,y,m);break
 #define MR(V,F) case V:hipLaunchKernelGGL((multi_rows<T,V,F>),dim3((320+T-1)/T,F?101:(m+V-1)/V,F?(m+V-1)/V:101),dim3(T),0,stream,a,x,w,y,m);break
 if(mode==0){switch(tile){BY(1,false);BY(2,false);BY(4,false);default:return 1;}}
 else if(mode==1){switch(tile){BY(1,true);BY(2,true);BY(4,true);default:return 1;}}
 else if(mode==2){switch(tile){FL(1);FL(2);FL(4);default:return 1;}}
 else if(mode==3){switch(tile){MR(2,false);MR(4,false);default:return 1;}}
 else if(mode==4){switch(tile){MR(2,true);MR(4,true);default:return 1;}}
 else return 1;
 return hipGetLastError();}
extern "C" int scale_schedule(const int32_t*a,const float*x,const float*w,float*y,int m,int g,int width,int mode,int tile,int threads,hipStream_t stream){
 if(m<=0||g!=101||width!=1280)return 1;
 switch(threads){case 64:return go<64>(a,x,w,y,m,mode,tile,stream);case 128:return go<128>(a,x,w,y,m,mode,tile,stream);case 192:return go<192>(a,x,w,y,m,mode,tile,stream);case 256:return go<256>(a,x,w,y,m,mode,tile,stream);case 320:return go<320>(a,x,w,y,m,mode,tile,stream);case 384:return go<384>(a,x,w,y,m,mode,tile,stream);case 512:return go<512>(a,x,w,y,m,mode,tile,stream);default:return 1;}}

}
#undef BY
#undef FL
#undef MR

namespace dsv41_packed {

extern "C" int scale_vec(const int32_t*,const float*,const float*,void*,int,int,int,int,int,int,hipStream_t);
extern "C" int scale_sharded(const int32_t*,const float*,const float*,float*,int,int,int,int,int,hipStream_t);
extern "C" int scale_schedule(const int32_t*,const float*,const float*,float*,int,int,int,int,int,int,hipStream_t);
namespace {
void ck(hipblasStatus_t s,const char*n){TORCH_CHECK(s==HIPBLAS_STATUS_SUCCESS,"optimized BLASLt ",n," status ",int(s));}
struct Plan {
 int64_t m,g,n,k,batches,lda,astride,pitch;bool shard;int device;
 hipblasLtHandle_t h=nullptr;hipblasLtMatmulDesc_t desc=nullptr;hipblasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr,d=nullptr;hipblasLtMatmulAlgo_t algo{};int32_t alpha=1,beta=0;std::mutex mutex;
 Plan(const at::Tensor&w,int64_t mm,int64_t gg,int64_t nn,int64_t kk,int64_t bb,int64_t ld,int64_t st,int64_t pp,bool sh,const std::string& hex):m(mm),g(gg),n(nn),k(kk),batches(bb),lda(ld),astride(st),pitch(pp),shard(sh),device(w.get_device()){
  TORCH_CHECK(w.is_cuda()&&w.scalar_type()==at::kChar&&w.is_contiguous(),"packed INT8 required");
  TORCH_CHECK(m>0&&g>0&&n>0&&k>0&&batches>0&&(g==1||batches==g)&&g*n%batches==0&&pitch>=k&&pitch%4==0&&lda>=g*n/batches&&astride>0&&(!shard||g==1),"invalid geometry");
  int64_t width=g*n/batches;TORCH_CHECK(w.numel()>=(batches-1)*astride+(k-1)*lda+width,"packed weight span");
  c10::DeviceGuard guard(w.device());auto stream=c10::hip::getCurrentHIPStream(device).stream();hipStreamCaptureStatus capture;TORCH_CHECK(hipStreamIsCapturing(stream,&capture)==hipSuccess&&capture==hipStreamCaptureStatusNone,"warm up each shape before graph capture");
  ck(hipblasLtCreate(&h),"create");ck(hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_32I,HIP_R_32I),"desc");
  ck(hipblasLtMatrixLayoutCreate(&a,HIP_R_8I,width,k,lda),"A");ck(hipblasLtMatrixLayoutCreate(&b,HIP_R_8I,k,m,g*pitch),"B");
  ck(hipblasLtMatrixLayoutCreate(&c,HIP_R_32I,width,m,shard?width:g*n),"C");ck(hipblasLtMatrixLayoutCreate(&d,HIP_R_32I,width,m,shard?width:g*n),"D");
  if(batches>1){int32_t count=batches;int64_t strides[4]={astride,g>1?pitch:0,shard?m*width:width,shard?m*width:width};hipblasLtMatrixLayout_t ls[4]={a,b,c,d};for(int i=0;i<4;i++){ck(hipblasLtMatrixLayoutSetAttribute(ls[i],HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&count,sizeof(count)),"batch");ck(hipblasLtMatrixLayoutSetAttribute(ls[i],HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&strides[i],sizeof(int64_t)),"stride");}}
  if(!hex.empty()){
   TORCH_CHECK(hex.size()==32,"algorithm hex length");unsigned char bytes[16];for(int i=0;i<16;i++)bytes[i]=std::stoul(hex.substr(i*2,2),nullptr,16);std::memcpy(algo.data,bytes,16);
   hipblasLtMatmulHeuristicResult_t result{};ck(hipblasLtMatmulAlgoCheck(h,desc,a,b,c,d,&algo,&result),"selected algorithm check");TORCH_CHECK(result.workspaceSize==0,"selected algorithm workspace");
  }else{
   hipblasLtMatmulPreference_t pref;ck(hipblasLtMatmulPreferenceCreate(&pref),"preference");uint64_t zero=0;ck(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&zero,sizeof(zero)),"workspace preference");hipblasLtMatmulHeuristicResult_t results[32]{};int count=0;auto status=hipblasLtMatmulAlgoGetHeuristic(h,desc,a,b,c,d,pref,32,results,&count);hipblasLtMatmulPreferenceDestroy(pref);ck(status,"heuristic");bool found=false;for(int i=0;i<count;i++)if(results[i].state==HIPBLAS_STATUS_SUCCESS&&results[i].workspaceSize==0){algo=results[i].algo;found=true;break;}TORCH_CHECK(found,"no zero-workspace INT8 algorithm");
  }
 }
 ~Plan(){if(a)hipblasLtMatrixLayoutDestroy(a);if(b)hipblasLtMatrixLayoutDestroy(b);if(c)hipblasLtMatrixLayoutDestroy(c);if(d)hipblasLtMatrixLayoutDestroy(d);if(desc)hipblasLtMatmulDescDestroy(desc);if(h)hipblasLtDestroy(h);}
};
std::mutex registry_mutex;std::vector<std::shared_ptr<Plan>> plans;
std::shared_ptr<Plan> get(int64_t id){std::lock_guard<std::mutex> lock(registry_mutex);TORCH_CHECK(id>0&&id<=int64_t(plans.size())&&plans[id-1],"invalid plan");return plans[id-1];}
int64_t prepare(const at::Tensor&w,int64_t m,int64_t g,int64_t n,int64_t k,int64_t batches,int64_t lda,int64_t astride,int64_t pitch,bool shard,std::string hex){auto p=std::make_shared<Plan>(w,m,g,n,k,batches,lda,astride,pitch,shard,hex);std::lock_guard<std::mutex> lock(registry_mutex);plans.push_back(p);return plans.size();}
void gemm(const at::Tensor&q,const at::Tensor&w,at::Tensor raw,int64_t id){
 auto p=get(id);std::lock_guard<std::mutex> lock(p->mutex);c10::DeviceGuard guard(q.device());
 TORCH_CHECK(q.is_cuda()&&w.is_cuda()&&raw.is_cuda()&&q.get_device()==p->device&&w.device()==q.device()&&raw.device()==q.device(),"GEMM device mismatch");
 TORCH_CHECK(q.scalar_type()==at::kChar&&w.scalar_type()==at::kChar&&raw.scalar_type()==at::kInt&&w.is_contiguous()&&raw.is_contiguous(),"GEMM types/layout");
 TORCH_CHECK(q.dim()==3&&q.size(0)==p->m&&q.size(1)==p->g&&q.size(2)==p->k&&q.stride(2)==1&&q.stride(1)==p->pitch&&q.stride(0)==p->g*p->pitch&&raw.numel()==p->m*p->g*p->n,"GEMM geometry mismatch");
 TORCH_CHECK(w.numel()>=(p->batches-1)*p->astride+(p->k-1)*p->lda+p->g*p->n/p->batches,"weight span mismatch");
 auto stream=c10::hip::getCurrentHIPStream(p->device).stream();ck(hipblasLtMatmul(p->h,p->desc,&p->alpha,w.data_ptr(),p->a,q.data_ptr(),p->b,&p->beta,raw.data_ptr(),p->c,raw.data_ptr(),p->d,&p->algo,nullptr,0,stream),"INT8/INT32 matmul");
}
void scale(const at::Tensor&raw,const at::Tensor&xs,const at::Tensor&ws,at::Tensor out,int64_t groups,int64_t batches,bool shard,int64_t mode){
 TORCH_CHECK(raw.is_cuda()&&xs.device()==raw.device()&&ws.device()==raw.device()&&out.device()==raw.device(),"scale device mismatch");c10::DeviceGuard guard(raw.device());
 TORCH_CHECK(raw.dim()==3&&out.sizes()==raw.sizes()&&groups>0&&batches>0&&raw.size(1)==groups,"scale geometry");int64_t m=raw.size(0),g=groups,n=raw.size(2);
 TORCH_CHECK(raw.scalar_type()==at::kInt&&xs.scalar_type()==at::kFloat&&ws.scalar_type()==at::kFloat&&raw.is_contiguous()&&xs.is_contiguous()&&ws.is_contiguous()&&out.is_contiguous()&&xs.numel()==m*g&&ws.numel()==g*n,"scale types/layout");
 auto stream=c10::hip::getCurrentHIPStream(raw.get_device()).stream();int status;
 if(shard){TORCH_CHECK(g==1&&n%batches==0&&out.scalar_type()==at::kFloat,"shard output must be FP32");int th=mode==1?512:128;int tile=mode==3?4:1;status=scale_sharded(raw.data_ptr<int32_t>(),xs.data_ptr<float>(),ws.data_ptr<float>(),out.data_ptr<float>(),m,batches,n/batches,tile,th,stream);}
 else{TORCH_CHECK(out.scalar_type()==at::kFloat||out.scalar_type()==at::kBFloat16,"vector output type");status=scale_vec(raw.data_ptr<int32_t>(),xs.data_ptr<float>(),ws.data_ptr<float>(),out.data_ptr(),m,g,n,out.scalar_type()==at::kFloat,2,128,stream);}
 TORCH_CHECK(status==0,"HIP scale launch failed ",status);
}
void scale_scheduled(const at::Tensor&raw,const at::Tensor&xs,const at::Tensor&ws,at::Tensor out,int64_t mode,int64_t tile,int64_t threads){
 TORCH_CHECK(raw.is_cuda()&&xs.device()==raw.device()&&ws.device()==raw.device()&&out.device()==raw.device(),"scheduled scale device mismatch");
 TORCH_CHECK(raw.dim()==3&&raw.size(1)==1&&raw.size(2)==129280&&out.sizes()==raw.sizes(),"scheduled scale requires [M,1,129280]");
 TORCH_CHECK(raw.scalar_type()==at::kInt&&xs.scalar_type()==at::kFloat&&ws.scalar_type()==at::kFloat&&out.scalar_type()==at::kFloat,"scheduled scale types");
 TORCH_CHECK(raw.is_contiguous()&&xs.is_contiguous()&&ws.is_contiguous()&&out.is_contiguous()&&xs.numel()==raw.size(0)&&ws.numel()==129280,"scheduled scale layout");
 if(raw.size(0)==0)return;
 c10::DeviceGuard guard(raw.device());auto stream=c10::hip::getCurrentHIPStream(raw.get_device()).stream();
 int status=scale_schedule(raw.data_ptr<int32_t>(),xs.data_ptr<float>(),ws.data_ptr<float>(),out.data_ptr<float>(),raw.size(0),101,1280,mode,tile,threads,stream);
 TORCH_CHECK(status==0,"HIP scale schedule launch failed ",status);
}
}

}

TORCH_LIBRARY_FRAGMENT(sgl_kernel,m) {
  m.def("dsv41_int8_gemm_out(Tensor q, Tensor w, Tensor(a!) out, int groups, int tile_n, bool group_major, int algo_rank=0) -> ()");
}
TORCH_LIBRARY_IMPL(sgl_kernel,CUDA,m) {m.impl("dsv41_int8_gemm_out",dsv41_unpacked::gemm_out);}

TORCH_LIBRARY_FRAGMENT(sgl_kernel,m) {
  m.def("dsv41_int8_quantize_out(Tensor x, Tensor(a!) q, Tensor(b!) scales) -> ()");
  m.def("dsv41_int8_scale_out(Tensor acc, Tensor xs, Tensor ws, Tensor(a!) out, int groups, bool group_major) -> ()");
}
TORCH_LIBRARY_IMPL(sgl_kernel,CUDA,m) {
  m.impl("dsv41_int8_quantize_out",dsv41_quant::dsv41_blaslt_quantize_out);
  m.impl("dsv41_int8_scale_out",dsv41_quant::dsv41_blaslt_scale_out);
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel,m){m.def("dsv41_int8_prepare(Tensor packed, int tokens, int groups, int rank, int k, int batches, int leading_dim, int batch_stride, int q_pitch, bool sharded, str algohex) -> int");m.def("dsv41_int8_gemm_packed(Tensor q, Tensor packed, Tensor(a!) raw, int plan) -> ()");m.def("dsv41_int8_scale_packed(Tensor raw, Tensor xs, Tensor ws, Tensor(a!) out, int groups, int batches, bool sharded, int mode) -> ()");}
TORCH_LIBRARY_IMPL(sgl_kernel,CUDA,m){m.impl("dsv41_int8_prepare",dsv41_packed::prepare);m.impl("dsv41_int8_gemm_packed",dsv41_packed::gemm);m.impl("dsv41_int8_scale_packed",dsv41_packed::scale);}

TORCH_LIBRARY_FRAGMENT(sgl_kernel,m){m.def("dsv41_int8_scale_scheduled(Tensor raw, Tensor xs, Tensor ws, Tensor(a!) out, int mode, int tile, int threads) -> ()");}
TORCH_LIBRARY_IMPL(sgl_kernel,CUDA,m){m.impl("dsv41_int8_scale_scheduled",dsv41_packed::scale_scheduled);}

#else
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/numeric_types.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include "cutlass_extensions/epilogue/epilogue_per_row_per_col_scale.h"
#include "cutlass_extensions/gemm/gemm_universal_base_compat.h"
#include "cutlass_extensions/gemm/gemm_with_epilogue_visitor.h"
#include "utils.h"

using namespace cute;

template <
    typename ElementOutput,
    typename ArchTag,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    int NumStages>
void cutlass_int8_scaled_mm(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

  using DefaultGemmConf = cutlass::gemm::device::
      DefaultGemmConfiguration<OperatorClass, ArchTag, ElementInputA, ElementInputB, ElementOutput, ElementCompute>;
  using EpilogueOutputOp = typename DefaultGemmConf::EpilogueOutputOp;

  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<
      ElementInputA,
      cutlass::layout::RowMajor,
      DefaultGemmConf::kAlignmentA,
      ElementInputB,
      cutlass::layout::ColumnMajor,
      DefaultGemmConf::kAlignmentB,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      ThreadblockSwizzle,
      NumStages,
      true,
      typename DefaultGemmConf::Operator>::GemmKernel;

  using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
      cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
          GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
          GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess,
          cutlass::sizeof_bits<ElementOutput>::value>,
      ElementCompute>;

  using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<
      ThreadblockShape,
      GemmKernel_::kThreadCount,
      AlphaColTileIterator,
      typename GemmKernel_::Epilogue::OutputTileIterator,
      ElementAccumulator,
      ElementCompute,
      EpilogueOutputOp>;

  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor, typename GemmKernel_::Epilogue>::Epilogue;

  using GemmKernel =
      cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;

  using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

  Gemm gemm_op;

  int m = mat_a.size(0);
  int k = mat_a.size(1);
  int n = mat_b.size(1);

  auto a_ptr = static_cast<ElementInputA*>(mat_a.data_ptr());
  auto b_ptr = static_cast<ElementInputB*>(mat_b.data_ptr());
  auto o_ptr = static_cast<ElementOutput*>(out.data_ptr());

  auto a_s_ptr = static_cast<ElementCompute*>(scales_a.data_ptr());
  auto b_s_ptr = static_cast<ElementCompute*>(scales_b.data_ptr());

  int64_t lda = mat_a.stride(0);
  int64_t ldb = mat_b.stride(1);
  int64_t ldd = out.stride(0);

  ElementOutput* bias_ptr = nullptr;
  int64_t ldc = 0;
  if (bias) {
    bias_ptr = static_cast<ElementOutput*>(bias->data_ptr());
  }

  typename EpilogueOutputOp::Params linearScalingParams;
  typename EpilogueVisitor::Arguments visitor_args{linearScalingParams};

  typename Gemm::Arguments args{
      {m, n, k}, {a_ptr, lda}, {b_ptr, ldb}, {b_s_ptr, 0}, {a_s_ptr, 0}, {bias_ptr, ldc}, {o_ptr, ldd}, visitor_args};

  auto workspace = torch::empty(
      gemm_op.get_workspace_size(args), torch::TensorOptions().dtype(torch::kUInt8).device(mat_a.device()));

  auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());

  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(
      can_implement == cutlass::Status::kSuccess,
      "gemm cannot implement, error: ",
      cutlassGetStatusString(can_implement));

  auto status = gemm_op(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "gemm executioin failed, error: ", cutlassGetStatusString(status));
}

template <typename ElementOutput, typename ArchTag, typename InstructionShape>
void sm75_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  int m = mat_a.size(0);
  if (m <= 32) {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<32, 128, 64>,
        cutlass::gemm::GemmShape<32, 64, 64>,
        InstructionShape,
        2>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else if (m <= 64) {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<64, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        2>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else if (m <= 256) {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<128, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        2>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        2>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

template <typename ElementOutput, typename ArchTag, typename InstructionShape>
void sm80_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  int m = mat_a.size(0);
  int n = mat_b.size(1);
  if (m <= 16) {
    if (n <= 4096) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          6>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 32) {
    if (n <= 4096) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          6>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 64) {
    if (n <= 4096) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 128, 128>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 128 && n < 8192) {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<64, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        5>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        5>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

// Dispatch shape for sm89 (L40S, L20, RTX 4090), according to:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/scaled_mm_c2x_sm89_int8_dispatch.cuh
template <typename ElementOutput, typename ArchTag, typename InstructionShape>
void sm89_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  int m = mat_a.size(0);
  int n = mat_b.size(1);
  if (m <= 16) {
    if (n <= 8192) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<16, 128, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          4>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 32) {
    if (n <= 8192) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<32, 128, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          4>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 64) {
    if (n <= 8192) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 128, 128>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 128) {
    if (n <= 8192) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 128, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 256) {
    if (n <= 4096) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 128, 128>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else if (n <= 8192) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<256, 128, 64>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        5>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

template <
    typename ElementOutput,
    typename TileShape,
    typename ClusterShape,
    typename MainloopScheduleType,
    bool WithBias>
void cutlass_int8_scaled_mm_sm90(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  using ArchTag = cutlass::arch::Sm90;

  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using XScale = cutlass::epilogue::fusion::
      Sm90ColBroadcast<0, TileShape, ElementCompute, ElementCompute, Stride<Int<1>, Int<0>, Int<0>>>;

  using WScale = cutlass::epilogue::fusion::
      Sm90RowBroadcast<0, TileShape, ElementCompute, ElementCompute, Stride<Int<0>, Int<1>, Int<0>>>;

  using Bias = cutlass::epilogue::fusion::
      Sm90RowBroadcast<0, TileShape, ElementOutput, ElementOutput, Stride<Int<0>, Int<1>, Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  // Scale
  using Compute0 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

  using Compute1 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

  // With bias
  using ComputeWithBias = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiply_add, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeWithBias = cutlass::epilogue::fusion::Sm90EVT<ComputeWithBias, XScale, EVTCompute0, Bias>;

  using EpilogueEVT = typename cutlass::platform::conditional<WithBias, EVTComputeWithBias, EVTCompute1>::type;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementOutput,
      cutlass::layout::RowMajor,
      AlignmentC,
      ElementOutput,
      cutlass::layout::RowMajor,
      AlignmentOutput,
      EpilogueScheduleType,
      EpilogueEVT>::CollectiveOp;

  using Stages = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementInputA,
      cutlass::layout::RowMajor,
      AlignmentA,
      ElementInputB,
      cutlass::layout::ColumnMajor,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      Stages,
      MainloopScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  Gemm gemm_op;

  int m = mat_a.size(0);
  int k = mat_a.size(1);
  int n = mat_b.size(1);

  auto a_ptr = static_cast<ElementInputA*>(mat_a.data_ptr());
  auto b_ptr = static_cast<ElementInputB*>(mat_b.data_ptr());
  auto o_ptr = static_cast<ElementOutput*>(out.data_ptr());

  auto a_s_ptr = static_cast<ElementCompute*>(scales_a.data_ptr());
  auto b_s_ptr = static_cast<ElementCompute*>(scales_b.data_ptr());

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  StrideC stride_c;
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));

  typename Gemm::Arguments args = {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {a_ptr, stride_a, b_ptr, stride_b},
      {{},  // epilogue.thread
       nullptr,
       stride_c,
       o_ptr,
       stride_d}};

  if constexpr (WithBias) {
    ElementOutput* bias_ptr = static_cast<ElementOutput*>(bias->data_ptr());
    args.epilogue.thread = {
        {a_s_ptr},
        {{b_s_ptr}, {}, {}},
        {bias_ptr},
        {},
    };
  } else {
    args.epilogue.thread = {
        {a_s_ptr},
        {{b_s_ptr}, {}, {}},
        {},
    };
  }

  auto workspace = torch::empty(
      gemm_op.get_workspace_size(args), torch::TensorOptions().dtype(torch::kUInt8).device(mat_a.device()));

  auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());

  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(
      can_implement == cutlass::Status::kSuccess,
      "gemm cannot implement, error: ",
      cutlassGetStatusString(can_implement));

  auto status = gemm_op(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "gemm executioin failed, error: ", cutlassGetStatusString(status));
}

template <typename ElementOutput, typename TileShape, typename ClusterShape, typename MainloopScheduleType>
void sm90_dispatch_bias(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  if (bias) {
    cutlass_int8_scaled_mm_sm90<ElementOutput, TileShape, ClusterShape, MainloopScheduleType, true>(
        out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    cutlass_int8_scaled_mm_sm90<ElementOutput, TileShape, ClusterShape, MainloopScheduleType, false>(
        out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

template <typename ElementOutput>
void sm90_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  int m = mat_a.size(0);
  int n = mat_b.size(1);
  if (m <= 32) {
    if (n < 8192) {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _64, _128>,
          Shape<_1, _8, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _128, _128>,
          Shape<_1, _8, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 64) {
    if (n < 8192) {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _64, _128>,
          Shape<_1, _4, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _64, _256>,
          Shape<_1, _1, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 128) {
    if (n <= 4096) {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _64, _128>,
          Shape<_2, _1, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _128, _128>,
          Shape<_2, _1, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else {
    return sm90_dispatch_bias<
        ElementOutput,
        Shape<_128, _128, _128>,
        Shape<_2, _1, _1>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

torch::Tensor int8_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  TORCH_CHECK(mat_b.stride(0) == 1, "mat_b must be a column major tensor");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");
  TORCH_CHECK(mat_a.size(1) % 16 == 0, "mat_a.size(1) must be multiple of 16 for memory alignment");
  TORCH_CHECK(mat_b.size(0) % 16 == 0, "mat_b.size(0) must be multiple of 16 for memory alignment");
  TORCH_CHECK(mat_b.size(1) % 8 == 0, "mat_b.size(1) must be multiple of 8 for memory alignment");  // out.stride(0)
  TORCH_CHECK(mat_a.scalar_type() == torch::kInt8, "mat_a must be Int8");
  TORCH_CHECK(mat_b.scalar_type() == torch::kInt8, "mat_b must be Int8");
  TORCH_CHECK(out_dtype == torch::kHalf || out_dtype == torch::kBFloat16, "out_dtype must be Half or BFloat16");

  TORCH_CHECK(scales_a.numel() == mat_a.size(0), "size of scales_a is not matched");
  TORCH_CHECK(scales_b.numel() == mat_b.size(1), "size of scales_b is not matched");
  TORCH_CHECK(scales_a.is_contiguous(), "scales_a must be contiguous");
  TORCH_CHECK(scales_b.is_contiguous(), "scales_b msut be contiguous");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be Float32");

  if (bias) {
    TORCH_CHECK(bias->numel() == mat_b.size(1), "size of bias is not matched");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->dtype() == out_dtype, "bias dtype must match output dtype");
  }

  torch::Tensor out = torch::empty({mat_a.size(0), mat_b.size(1)}, mat_a.options().dtype(out_dtype));

  auto sm_version = getSMVersion();

  if (sm_version >= 75 && sm_version < 80) {
    TORCH_CHECK(out_dtype == torch::kHalf, "out_dtype must be Half for SM75");
    sm75_dispatch_shape<cutlass::half_t, cutlass::arch::Sm75, cutlass::gemm::GemmShape<8, 8, 16>>(
        out, mat_a, mat_b, scales_a, scales_b, bias);
  } else if (sm_version >= 80 && sm_version < 90) {
    // sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
    if (sm_version == 86 || sm_version == 89) {
      if (out_dtype == torch::kBFloat16) {
        sm89_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      } else {
        sm89_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      }
    } else {
      if (out_dtype == torch::kBFloat16) {
        sm80_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      } else {
        sm80_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      }
    }
  } else if (sm_version == 90) {
#if defined CUDA_VERSION && CUDA_VERSION >= 12000
    // cutlass 3.x
    if (out_dtype == torch::kBFloat16) {
      sm90_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      sm90_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
#else
    // fallback to cutlass 2.x
    if (out_dtype == torch::kBFloat16) {
      sm80_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
          out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      sm80_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
          out, mat_a, mat_b, scales_a, scales_b, bias);
    }
#endif
  } else {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "No implemented int8_scaled_mm for current compute capability.");
  }

  return out;
}

#endif
