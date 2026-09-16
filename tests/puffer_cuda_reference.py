"""Compile selected, unmodified kernels from the optional local PufferLib checkout.

No PufferLib training build is required. The harness supplies float32 buffers
and a one-head action configuration, then calls the actual CUDA implementations.
"""
import ctypes
import os
import shutil
from pathlib import Path
import subprocess

import numpy as np


def _definition(source, marker):
    start = source.index(marker)
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    if source[start:].startswith(('struct ', 'enum ')):
        end += 1
    return source[start:end]


def build_reference(directory):
    root = Path(__file__).resolve().parents[1] / 'PufferLib/src'
    algo = (root / 'algo.cu').read_text()
    trainer = (root / 'pufferl.cu').read_text()
    preamble = '''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
using precision_t = float;
struct Prec { float* data; int shape[4]; };
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float from_float(float x) { return x; }
#define NUM_ATNS 1
#define ACT_SIZES {3}
constexpr int PPO_THREADS = 256;
constexpr int PPO_MAX_HEAD_A = 3;
constexpr int ADV_VEC_WIDTH = 4;
'''
    markers = [
        '__device__ __forceinline__ float finite_or_clamp(',
        '__device__ __forceinline__ float safe_continuous_mean(',
        '__device__ __forceinline__ float safe_continuous_logstd(',
        'enum LossIdx {', 'struct PPOGraphArgs {', 'struct PPOKernelArgs {',
        '__device__ __forceinline__ float load_logit_masked(',
        '__device__ __forceinline__ float ppo_discrete_logsumexp(',
        '__device__ __forceinline__ void ppo_continuous_head(',
        '__global__ void cache_imp_and_v(', '__global__ void ppo_loss_compute(',
        '__device__ __forceinline__ void adv_ld(',
        '__device__ __forceinline__ void adv_st(', '__global__ void puff_advantage(',
    ]
    source = preamble + _definition(trainer, '__device__ __forceinline__ void block_reduce_sum(')
    source += '\n' + '\n'.join(_definition(algo, marker) for marker in markers)
    source += r'''
struct Allocation {
    float* p = nullptr;
    ~Allocation() { if (p) cudaFree(p); }
};
#define CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) return (int)err; } while (0)
extern "C" int advantage(const float* input, float* output, int S, int T,
        int vtrace, float gamma, float lambda, float rho, float c) {
    int n = S*T;
    Allocation mem;
    CHECK(cudaMalloc(&mem.p, 6*n*sizeof(float)));
    float* p = mem.p;
    CHECK(cudaMemcpy(p, input, 4*n*sizeof(float), cudaMemcpyHostToDevice));
    puff_advantage<<<(S+63)/64,64>>>(p,p+n,p+2*n,vtrace?p+3*n:nullptr,
        p+4*n,p+5*n,gamma,lambda,rho,c,S,T);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(output,p+4*n,2*n*sizeof(float),cudaMemcpyDeviceToHost));
    return 0;
}
extern "C" int loss(const float* input, float* output, int n, int continuous,
        float clip, float vfcoef, float entcoef) {
    // Inputs: fused predictions; actions; behavior LP; advantages; behavior V;
    // targets; logstd. One action head (three categorical classes or one Normal).
    if (n > PPO_THREADS) return -1; // One block makes reduction output explicit.
    int A = continuous ? 1 : 3;
    int pred = n*(A+1);
    int nin = pred+5*n+1;
    int nout = LOSS_N + 2*n*A + 2*n;
    Allocation mem;
    CHECK(cudaMalloc(&mem.p, (nin+nout+n*A+1)*sizeof(float)));
    float* p = mem.p;
    CHECK(cudaMemcpy(p,input,nin*sizeof(float),cudaMemcpyHostToDevice));
    float* out = p+nin;
    CHECK(cudaMemset(out,0,nout*sizeof(float)));
    float* grad = out+LOSS_N;
    float* stdgrad = grad+n*A;
    float* vgrad = stdgrad+n*A;
    float* ratios = vgrad+n;
    float* mask = out+nout;
    float* entropy = mask+n*A;
    CHECK(cudaMemcpy(entropy,&entcoef,sizeof(float),cudaMemcpyHostToDevice));
    // Masks are all legal for cross-library comparisons.
    float hostmask[PPO_THREADS*3];
    for(int i=0;i<n*A;i++) hostmask[i]=1.f;
    CHECK(cudaMemcpy(mask,hostmask,n*A*sizeof(float),cudaMemcpyHostToDevice));
    Prec dec = {p,{1,n,A+1,0}};
    Prec logstd = {continuous ? p+pred+5*n : nullptr,{1,1,0,0}};
    int* act_sizes;
    CHECK(cudaMalloc(&act_sizes,sizeof(int)));
    cudaError_t size_error = cudaMemcpy(act_sizes,&A,sizeof(int),cudaMemcpyHostToDevice);
    if(size_error != cudaSuccess) { cudaFree(act_sizes); return (int)size_error; }
    cache_imp_and_v<<<1,PPO_THREADS>>>(dec,p+pred,p+pred+n,mask,logstd,
        act_sizes,ratios,vgrad,grad,vgrad);
    PPOGraphArgs g = {ratios,p+pred,p+pred+n,p+pred+2*n,p+pred+3*n,p+pred+4*n};
    PPOKernelArgs a = {grad,stdgrad,vgrad,p,logstd.data,p+A,act_sizes,mask,
        1,clip,clip,vfcoef,entropy,n,A,1,(bool)continuous};
    ppo_loss_compute<<<1,PPO_THREADS>>>(out,a,g);
    cudaError_t result = cudaGetLastError();
    if(result == cudaSuccess) result=cudaMemcpy(output,out,nout*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(act_sizes);
    return (int)result;
}
'''
    directory = Path(directory)
    cpp = directory / 'puffer_parity.cu'
    lib = directory / 'puffer_parity.so'
    cpp.write_text(source)
    host_compiler = os.environ.get('CUDAHOSTCXX') or shutil.which('g++-15')
    host_flags = ['-ccbin', host_compiler] if host_compiler else []
    build = subprocess.run(['nvcc', *host_flags,'-std=c++17','-O2','-shared','-Xcompiler','-fPIC',
                    '-arch=native',str(cpp),'-o',str(lib)],check=False,capture_output=True,text=True,
                    env={k:v for k,v in os.environ.items() if not k.startswith('BASH_FUNC_')})
    if build.returncode:
        raise RuntimeError(build.stderr)
    dll = ctypes.CDLL(str(lib))
    ptr = np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    dll.advantage.argtypes = [ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int]+[ctypes.c_float]*4
    dll.loss.argtypes = [ptr,ptr,ctypes.c_int,ctypes.c_int]+[ctypes.c_float]*3
    return dll
