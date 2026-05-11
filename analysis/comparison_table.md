# Experiment Comparison Table

| Metric | BF16 Baseline | FP8 | FP8 + CUDA Graphs |
|---|---|---|---|
| Throughput (tok/s) | [TBD] | [TBD] | [TBD] |
| Total wall time (s) | [TBD] | [TBD] | [TBD] |
| KV cache headroom (GB) | [TBD] | [TBD] | [TBD] |
| `cudaMemcpyAsync` % of CUDA API time | [TBD] | [TBD] | [TBD] |
| `cudaEventSynchronize` % of CUDA API time | [TBD] | [TBD] | [TBD] |
| Top kernel (% of GPU time) | [TBD] | [TBD] | [TBD] |
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
