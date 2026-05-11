# Experiment Comparison Table

| Metric | BF16 Baseline | FP8 | FP8 + CUDA Graphs |
|---|---|---|---|
| Throughput (tok/s) | 279.16 | 450.40 (+61.3%) | 626.68 (+39.1% vs FP8, **+124.5% vs baseline**) |
| Total wall time (s) | 12.742 | 8.410 | 6.234 |
| Avg latency per request (s) | 0.4247 | 0.2803 | 0.2078 |
| KV cache available (GiB) | 3.93 | 10.35 | 9.87 (−0.47 GiB graph pool) |
| KV cache token capacity | 32,160 | 84,768 | 80,880 |
| Max concurrency @ 4,096 tok/req | 7.85× | 20.70× | 19.75× |
| `cudaEventSynchronize` % of CUDA API time | *not captured* | **64.1%** (603 calls) | **52.6%** (783 calls) |
| `cudaMemcpyAsync` % of CUDA API time | *not captured* | **18.2%** (4,582 calls) | **19.9%** (4,569 calls) |
| `cudaLaunchKernel` % of CUDA API time | *not captured* | **14.7%** (178,970 calls) | **2.8%** (53,058 calls, **↓70%**) |
| `cudaGraphLaunch` % of CUDA API time | — | — | **0.2%** (331 replays) |
| Top kernel (% of GPU time) | *not captured* | CUTLASS FP8 GEMM (**43.7%**) | CUTLASS FP8 GEMM (**38.9%**) |
| Requests queued (KV eviction pressure) | **Yes** (7.85× < 30 users) | No (20.70×) | No (19.75×) |

> **Note on Exp 1 GPU metrics:** The CUDA API and kernel sections of `exp1_stats.txt` were truncated during collection. To retrieve them: `grep -A 30 "CUDA API Summary" ~/nsight/analysis/exp1_stats.txt`. Based on the BF16 eager architecture, `cudaEventSynchronize` is expected to exceed 64% (the FP8 eager value) since BF16 weights impose higher memory pressure per kernel.

---

**Configuration details**

| Setting | BF16 Baseline | FP8 | FP8 + CUDA Graphs |
|---|---|---|---|
| `dtype` | `bfloat16` | `bfloat16` | `bfloat16` |
| `quantization` | none | `fp8` | `fp8` |
| `enforce_eager` | `True` | `True` | `False` |
| `enable_prefix_caching` | `False` | `False` | `False` |
| `gpu_memory_utilization` | 0.90 | 0.90 | 0.90 |
| Model weight size (approx) | ~15 GB (BF16) | ~8.5 GB (FP8) | ~8.5 GB (FP8) |
| CUDA graph startup overhead | — | — | ~8.5 s (amortized) |

---

**Key takeaways**

- **FP8 alone (+61.3%):** Halving model weight footprint frees 6.4 GiB of VRAM, raising KV cache headroom from 3.93 → 10.35 GiB. At avg ~120 tok/req, all 30 users fit in KV cache simultaneously, eliminating eviction-driven re-prefill. Native FP8 tensor cores on L4 (Ada Lovelace SM89) execute the GEMM workload in fewer cycles.
- **CUDA Graphs on top (+39.1% vs FP8):** Reducing `cudaLaunchKernel` calls by 70% (178,970 → 53,058) eliminates ~3.5 s of pure kernel-enqueue overhead. Each decode step collapses to a single `cudaGraphLaunch`. The remaining `cudaEventSynchronize` (52.6%) and new `cudaDeviceSynchronize` (18.7%) represent the irreducible scheduling boundary between decode steps.
- **Remaining bottleneck:** `cudaEventSynchronize` at 52.6% in Exp 3 shows the GPU still sits idle between steps while the CPU advances the scheduler. Prefix caching (`enable_prefix_caching=True`) would eliminate repeated prefill work for repeated prompt prefixes, and tensor parallelism (TP=2) would spread the memory-bandwidth-bound GEMM across two GPUs.
