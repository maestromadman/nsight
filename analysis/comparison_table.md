# Experiment Comparison Table

| Metric | BF16 Baseline | FP8 | FP8 + CUDA Graphs |
|---|---|---|---|
| Throughput (tok/s) | 272.53 | 421.74 (+54.7%) | 587.34 (+115.5% vs baseline) |
| Total wall time (s) | 12.696 | 8.545 | 6.330 |
| Avg latency per request (s) | 0.423 | 0.285 | 0.211 |
| KV cache available (GiB) | 3.93 | 10.35 | 9.52 |
| KV cache token capacity | 32,160 | 84,768 | 77,984 |
| Max concurrency @ 4096 tok | 7.85× | 20.70× | 19.04× |
| `cudaMemcpyAsync` % of CUDA API time | [TBD — from nsys stats] | [TBD] | [TBD] |
| `cudaEventSynchronize` % of CUDA API time | [TBD — from nsys stats] | [TBD] | [TBD] |
| Top kernel (% of GPU time) | [TBD — from nsys stats] | [TBD] | [TBD] |
| Requests queued (yes/no) | [TBD] | [TBD] | [TBD] |

**Configuration details**

| Setting | BF16 Baseline | FP8 | FP8 + CUDA Graphs |
|---|---|---|---|
| `dtype` | `bfloat16` | `bfloat16` | `bfloat16` |
| `quantization` | none | `fp8` | `fp8` |
| `enforce_eager` | `True` | `True` | `False` |
| `enable_prefix_caching` | `False` | `False` | `False` |
| `gpu_memory_utilization` | 0.90 | 0.90 | 0.90 |
| Model weight size (approx) | ~16 GB | ~8 GB | ~8 GB |
