# vLLM Inference Profiling, Performance Analysis with NVIDIA Nsight Systems
#### *findings_raw.txt* includes output from Nsight
#### *inference.py* and *inference_2.py* include the code I used to run the LLM
#### *relevant_topics.md* breaks down topics explored throughout this project and mentioned below 

## Setup
- GPU: NVIDIA L4 (23GB VRAM)
- Model: meta-llama/Llama-3.2-1B
- Framework: vLLM
- Profiler: NVIDIA Nsight Systems
- Workload: 20 prompts, max_tokens=128, temperature=0.7

## Baseline Performance
- Throughput: 1,654.97 output tokens/sec
- Model load time: 1.81 seconds
- KV cache size: 458,880 tokens
- GPU memory used: 2.32 GiB

## CUDA GPU Kernel Analysis

### Top kernels by time:
| Kernel | Time (%) | Notes |
|--------|----------|-------|
| ampere_bf16_s1688gemm (128x128) | 16.3% | Feed-forward matrix multiply |
| cutlass bf16 gemm 256x128 | 11.4% | Large GEMM, prefill phase |
| ampere_bf16_s16816gemm (128x64) | 10.6% | Decode phase matrix multiply |
| cutlass bf16 gemm 256x64 | 8.3% | Attention projection |
| flash_fwd_splitkv_kernel | 3.8% | PagedAttention (FlashAttention) |
| reshape_and_cache_flash_kernel | 0.8% | KV cache population |

GEMM kernels collectively account for ~57% of GPU time, consistent 
with transformer inference being compute-bound during prefill.

## Bottlenecks Identified

### 1. Host-to-Device Memory Transfer Overhead
- 526 Host→Device transfers totaling 2.47 GB
- Single largest transfer: 525 MB
- Avg transfer: 4.7 MB
- These transfers represent model weight loading and input tokenization 
  overhead. Repeated small transfers (median 0.001 MB) suggest 
  per-request data staging that could be batched.

### 2. CPU-GPU Synchronization Overhead
- cudaEventSynchronize accounts for 40.3% of all CUDA API time
- 108 synchronization calls averaging 10.9ms each
- This indicates the CPU scheduler is frequently blocking on GPU 
  completion, limiting pipeline overlap between prefill and decode.

### 3. KV Cache Operations
- reshape_and_cache_flash_kernel called 2,074 times
- Indicates per-token KV cache writes during decode phase
- With larger batch sizes, this could become a memory bandwidth bottleneck

## Observations

- FlashAttention (flash_fwd_splitkv_kernel) is correctly being used 
  as the attention backend, confirmed in vLLM logs
- Triton kernels (triton_poi_fused_mul_silu_slice) handle the SwiGLU 
  activation function in Llama's feed-forward layers
- The L4's BF16 tensor cores are being fully utilized across all major 
  GEMM operations
- DeviceSegmentedRadixSort kernel (1.3%) handles token sampling 
  (top-p/top-k) during decode

## Potential Improvements

1. **Increase batch size** — running 20 prompts concurrently is good, although 
   GPU utilization could improve with larger batches, reducing 
   the per-request synchronization overhead
2. **Reduce Host-to-Device transfers** — pinning input tensors in 
   CPU memory would reduce transfer latency for repeated inference runs
3. **Enable CUDA graphs** — vLLM already captures CUDA graphs 
   (seen in cudaGraphInstantiateWithFlags calls), but tuning graph 
   capture range could reduce kernel launch overhead
4. **Quantization** — applying INT8 or FP8 quantization would reduce 
   memory bandwidth pressure on the KV cache operations

## Optimization: Batch Size Scaling

**5,872 toks/s** — that's a **3.5x improvement** over the 1,682 toks/s baseline.

| Metric | Baseline (20 prompts) | Optimized (100 prompts) | Improvement |
|---|---|---|---|
| Output throughput | 1,682 toks/s | 5,873 toks/s | **+3.5x** |
| Input throughput | 126.5 toks/s | 456.6 toks/s | **+3.6x** |

### What changed
The `cudaEventSynchronize` bottleneck (38.3% of CUDA API time) represents fixed overhead per batch step, i.e. the he CPU stalling while waiting for the GPU to finish. With only 20 prompts, this overhead dominates. Scaling to 100 prompts amortizes it: the GPU does the same number of sync calls but processes 5x more tokens between each one.

The result actually exceeded the theoretical ceiling (~3,173 toks/s estimated from 53% GPU utilization). The reason: larger batches allow vLLM to pack more tokens into each decode step, improving GPU utilization beyond just reducing idle time.

### Summary
1. **Profiling identified** `cudaEventSynchronize` as 38.3% of CUDA API time — the CPU stalling between small batches
2. **Fix:** increase batch size from 20 → 100 prompts so fixed overhead is amortized
3. **Result:** 3.5x throughput improvement

## Tools Used
- NVIDIA Nsight Systems
- vLLM
- CUDA
- PyTorch
