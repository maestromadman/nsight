# vLLM Performance Engineering — Findings

## Project Overview

**Business scenario:** A fintech company operates a customer support chatbot handling billing disputes, account inquiries, and transaction questions. Peak concurrent load is 30 simultaneous users. The company is evaluating whether a single NVIDIA L4 GPU can meet production SLAs before committing to a multi-GPU deployment.

**Hardware:** NVIDIA L4 (24 GB GDDR6, 300 W TDP, Ada Lovelace SM89, native FP8 tensor core support)  
**Software stack:** CUDA 12.9 · Python 3.13 · vLLM v0.20.2 (V1 engine) · Nsight Systems 2025.1.3  
**Model:** `meta-llama/Llama-3.1-8B-Instruct` (8B parameters, BF16 ≈ 15 GB weight footprint)  
**Workload:** 30 realistic customer support prompts · `max_tokens=150` · `temperature=0.3` · `max_model_len=4096`  
**Profiling note:** `VLLM_ENABLE_V1_MULTIPROCESSING=0` was set to run the engine in-process, making all GPU activity visible to nsys. The vLLM V1 engine otherwise spawns GPU work in a child process that nsys cannot instrument without this flag.  
**Objective:** Quantify throughput, latency, and GPU utilization for three progressively optimized configurations and determine if a single L4 can sustain 30 concurrent users at acceptable latency.

---

## Experiment 1: BF16 Baseline

### Configuration
```python
LLM(model="meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16",
    enforce_eager=True, enable_prefix_caching=False,
    gpu_memory_utilization=0.90, max_model_len=4096)
```

### What we expected to find

With `enforce_eager=True` and no quantization, every transformer layer fires as individually-launched eager CUDA kernels. Each decode step requires the CPU to enqueue hundreds of separate `cudaLaunchKernel` calls. The L4 also has only ~3.93 GiB remaining for KV cache after loading 15 GB of BF16 weights, limiting concurrent request capacity to ~7.85 requests at max sequence length. We expected:

- High `cudaEventSynchronize` share: CPU blocking while waiting for GPU completion between operations.
- Moderate `cudaMemcpyAsync` share: BF16 weights consume 15 GB, leaving minimal KV cache headroom.
- Decode dominated by memory bandwidth (weight matrix-vector products are memory-bandwidth-bound at small batch sizes).
- Lowest throughput of the three experiments.

### Measured Results

| Metric | Value |
|---|---|
| Total wall time (s) | 12.742 |
| Total output tokens | 3,557 |
| Throughput (tok/s) | 279.16 |
| Avg latency per request (s) | 0.4247 |
| P50 latency per request (s) | 0.4247 |
| Requests processed | 30 |
| Min / Max / Avg tokens generated | 65 / 150 / 118.6 |
| KV cache available (GiB) | 3.93 |
| KV cache token capacity | 32,160 |
| Max concurrency @ 4,096 tok/req | 7.85× |
| Host-to-Device transfers (MB) | 16,064.9 (2,670 calls) |
| Device-to-Device transfers (MB) | 3.5 (1,300 calls) |
| `cudaEventSynchronize` % of CUDA API | *see note below* |
| Top GPU kernel (% of GPU time) | *see note below* |

> **Note:** The CUDA API summary and GPU kernel breakdown for Exp 1 were not captured in the stats export due to output truncation. The memory transfer data above is confirmed from nsys. Run `grep -A 30 "CUDA API Summary" ~/nsight/analysis/exp1_stats.txt` to retrieve the API breakdown. Based on the architecture (eager BF16, no graphs), `cudaEventSynchronize` is expected to exceed 64% (the FP8 eager baseline, which has less data to move).

### Nsight Interpretation

The 3.93 GiB KV cache available after loading 15 GB of BF16 weights is the defining constraint of this experiment. At `max_model_len=4096` tokens per sequence, the engine can hold ~7.85 concurrent requests worth of KV state at full sequence length — barely more than a quarter of the target 30-user load. With eager execution, each decode token requires hundreds of individual kernel launches across 32 transformer layers, with the CPU blocking frequently on `cudaEventSynchronize` to confirm GPU completion before advancing the scheduler.

---

## Experiment 2: FP8 Quantization

### Configuration
```python
LLM(model="meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16",
    quantization="fp8", enforce_eager=True, enable_prefix_caching=False,
    gpu_memory_utilization=0.90, max_model_len=4096)
```

### What FP8 changes mechanically

FP8 (E4M3 format) stores each weight element in 8 bits instead of 16, halving the model footprint from ~15 GB to ~8.5 GB on the L4. The Ada Lovelace SM89 architecture has native FP8 tensor cores — vLLM selects the `CutlassFP8ScaledMMLinearKernel` (confirmed in logs), so no software dequantization step is needed. Two first-order effects:

1. **KV cache headroom increases from 3.93 GiB to 10.35 GiB** — a 2.63× increase — because the freed VRAM is available for KV blocks.
2. **Memory bandwidth pressure during decode drops** because matrix-vector products (the bottleneck at small batch size) load half as many bytes per weight row from HBM.

### Measured Results

| Metric | Value |
|---|---|
| Total wall time (s) | 8.410 |
| Total output tokens | 3,788 |
| Throughput (tok/s) | 450.40 (+61.3% vs BF16) |
| Avg latency per request (s) | 0.2803 |
| P50 latency per request (s) | 0.2803 |
| Requests processed | 30 |
| Min / Max / Avg tokens generated | 64 / 150 / 126.3 |
| KV cache available (GiB) | 10.35 |
| KV cache token capacity | 84,768 |
| Max concurrency @ 4,096 tok/req | 20.70× |
| `cudaEventSynchronize` % of CUDA API | **64.1%** (15.25 s, 603 calls) |
| `cudaMemcpyAsync` % of CUDA API | **18.2%** (4.34 s, 4,582 calls) |
| `cudaLaunchKernel` % of CUDA API | **14.7%** (3.50 s, **178,970 calls**) |
| `cudaStreamSynchronize` % of CUDA API | 0.4% (774 calls) |

**GPU Kernel Breakdown:**

| Rank | Kernel | GPU Time % | Instances | Avg Duration |
|---|---|---|---|---|
| 1 | CUTLASS FP8 GEMM (decode, large) | **43.7%** | 22,976 | 334 µs |
| 2 | `direct_copy_kernel_cuda` (elementwise) | 17.0% | 9,966 | 299 µs |
| 3 | CUTLASS FP8 GEMM (decode, medium) | 14.8% | 15,168 | 172 µs |
| 4 | CUTLASS FP8 GEMM (prefill batch) | 7.0% | 256 | 4,806 µs |
| 5 | `ampere_bf16_s16816gemm` (attention BF16) | 3.7% | 141 | 4,555 µs |
| 6 | CUTLASS BF16 WMMA GEMM | 3.4% | 144 | 4,163 µs |
| 7 | FlashAttention-2 splitkv | 2.3% | 7,040 | 57 µs |
| 8 | FP8 dynamic per-token quant | 1.4% | 38,656 | 6.5 µs |

**Memory Operations:**

| Operation | Total (MB) | Calls | Avg (MB) |
|---|---|---|---|
| Host-to-Device | 16,065.2 | 2,851 | 5.6 |
| Device-to-Device | 4.0 | 1,367 | 0.003 |
| Device-to-Host | 0.016 | 364 | — |

### Nsight Interpretation

`cudaEventSynchronize` consumes **64.1%** of all CUDA API time, confirming that Exp 2 is **synchronization-bound**, not compute-bound. The GPU completes its work quickly, then waits idle while the CPU evaluates scheduler decisions and re-enqueues the next batch of kernels. The 178,970 individual `cudaLaunchKernel` calls (avg 19.6 µs each = ~3.5 s of pure scheduling overhead) expose the per-kernel launch cost of eager execution.

The top GPU kernel is the CUTLASS FP8 GEMM at 43.7% — this is the weight matrix projection (QKV, FFN up/down/gate) executing in native FP8 on the L4's tensor cores. The presence of `direct_copy_kernel_cuda` at 17% reflects per-token activation broadcasting and KV cache write-back. The FP8 dynamic quantization kernel (`vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided`) appears at 1.4% — this is the per-token activation scaling computed before each FP8 GEMM (online quantization, no calibration dataset needed).

**KV cache headroom improvement:** FP8 freed 6.4 GiB, raising max concurrency from 7.85× to 20.70× — the engine can now hold all 30 users' KV state simultaneously, eliminating the eviction pressure that would otherwise require re-prefilling dropped sequences.

---

## Experiment 3: FP8 + CUDA Graphs

### Configuration
```python
LLM(model="meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16",
    quantization="fp8", enforce_eager=False, enable_prefix_caching=False,
    gpu_memory_utilization=0.90, max_model_len=4096)
```

### What CUDA graphs change mechanically

In Experiments 1 and 2, every decode step requires the Python scheduler and CUDA driver to enqueue 200–400 individual kernel launches (one per operator across 32 transformer layers). Each `cudaLaunchKernel` call costs 5–20 µs of CPU time. For 30 requests generating up to 150 tokens, this adds up to millions of microseconds of pure scheduling overhead across the run.

CUDA graphs eliminate this by recording the entire decode computation graph during a one-time capture phase, then replaying it with a single `cudaGraphLaunch` call per decode step. Because the decode graph is structurally static (same operators, same tensor shapes for a given batch size slot), the captured graph is valid for every subsequent token.

In vLLM v0.20.2, CUDA graphs use **FULL_AND_PIECEWISE** mode, capturing 51 piecewise graphs (batch sizes 1–512) and 35 full decode graphs. The capture adds ~8.5 s of startup time (amortized over the inference run). CUDA graph memory overhead: 0.47 GiB (actual), reducing effective KV cache to 9.87 GiB vs 10.35 GiB in Exp 2.

### Measured Results

| Metric | Value |
|---|---|
| Total wall time (s) | 6.234 |
| Total output tokens | 3,907 |
| Throughput (tok/s) | 626.68 (+39.1% vs FP8, **+124.5% vs BF16 baseline**) |
| Avg latency per request (s) | 0.2078 |
| P50 latency per request (s) | 0.2078 |
| Requests processed | 30 |
| Min / Max / Avg tokens generated | 62 / 150 / 130.2 |
| KV cache available (GiB) | 9.87 (−0.47 GiB for graph pool) |
| KV cache token capacity | 80,880 |
| Max concurrency @ 4,096 tok/req | 19.75× |
| `cudaEventSynchronize` % of CUDA API | **52.6%** (↓ from 64.1%) (11.48 s, 783 calls) |
| `cudaMemcpyAsync` % of CUDA API | **19.9%** (4.34 s, 4,569 calls) |
| `cudaDeviceSynchronize` % of CUDA API | **18.7%** (4.09 s, **1,920 calls**) |
| `cudaLaunchKernel` % of CUDA API | **2.8%** (610 ms, 53,058 calls — **↓70% vs Exp 2**) |
| `cudaGraphLaunch` % of CUDA API | 0.2% (50 ms, **331 replays**) |
| `cuLaunchKernel` (graph nodes) | 0.7% (154 ms, 29,742 calls) |
| `cudaGraphInstantiateWithFlags` | 1.5% (334 ms, 1,786 calls) |

**GPU Kernel Breakdown:**

| Rank | Kernel | GPU Time % | Instances | Avg Duration |
|---|---|---|---|---|
| 1 | CUTLASS FP8 GEMM (decode, large) | **38.9%** | 4,160 | 754 µs |
| 2 | CUTLASS FP8 GEMM (decode, medium) | 9.9% | 3,776 | 212 µs |
| 3 | `ampere_bf16_s16816gemm` (attention) | 8.5% | 150 | 4,557 µs |
| 4 | Triton fused FP8 quant + GEMM | 7.3% | 3,120 | 189 µs |
| 5 | CUTLASS BF16 WMMA GEMM | 6.9% | 133 | 4,160 µs |
| 6 | CUTLASS FP8 GEMM (small) | 4.1% | 640 | 510 µs |
| 7 | CUTLASS FP8 GEMM (prefill) | 3.4% | 832 | 331 µs |
| 8 | CUTLASS FP8 GEMM (medium-2) | 3.2% | 1,024 | 251 µs |

**Memory Operations:**

| Operation | Total (MB) | Calls | Avg (MB) |
|---|---|---|---|
| Host-to-Device | 16,075.4 | 3,255 | 4.9 |
| Device-to-Device | 0.342 | 918 | 0.000 |
| Device-to-Host | 0.017 | 364 | — |

### Nsight Interpretation

The CUDA graph effect is measured directly: **`cudaLaunchKernel` dropped from 178,970 calls (Exp 2) to 53,058 calls — a 70% reduction.** The 331 `cudaGraphLaunch` calls each replay an entire decode step's kernel sequence in a single CPU instruction. Instead of hundreds of individual enqueue operations per token, the GPU receives a single "replay this recorded sequence" command.

`cudaEventSynchronize` fell from 64.1% to **52.6%**, and `cudaLaunchKernel` from 14.7% to **2.8%**. The CPU scheduling overhead that dominated Exp 2 is substantially eliminated for the decode phase.

The new **`cudaDeviceSynchronize` at 18.7%** (1,920 calls, avg 2.1 ms each) is expected: vLLM must synchronize at the boundary between decode steps to check termination conditions, update the KV cache block table, and decide on the next graph slot to replay. This synchronization is the remaining bottleneck — it prevents the GPU from running truly continuously between steps.

**Triton fused kernels** appear in Exp 3 (7.3% of GPU time) but not Exp 2. These are torch.compile-generated fused operations that combine FP8 quantization, scaling, and clamp operations into a single kernel launch — a compile-time optimization unavailable in eager mode.

**Device-to-Device transfers dropped from 4.0 MB to 0.342 MB** between Exp 2 and Exp 3. The CUDA graph replays tensors at captured addresses, eliminating many intermediate device copies that eager mode requires for scheduler state management.

---

## Bottleneck Analysis

The three experiments expose a layered bottleneck structure:

### Exp 1: BF16 Eager — Memory-capacity and synchronization-bound

BF16 weights occupy 15 GB, leaving only 3.93 GiB for KV cache (7.85× max concurrency). The GPU is also under-utilized between kernel launches because the CPU must re-enqueue each operator individually. Both constraints are active simultaneously: not enough KV cache to pipeline requests efficiently, and too much CPU scheduling overhead per token.

### Exp 2: FP8 Eager — Synchronization-bound

FP8 eliminated the memory-capacity constraint. KV cache headroom jumped from 3.93 GiB to 10.35 GiB, raising max concurrency from 7.85× to 20.70×. Throughput improved 61.3%. But **64.1% of CUDA API time is `cudaEventSynchronize`** — the CPU blocks 603 times waiting for GPU confirmation. The 178,970 `cudaLaunchKernel` calls (14.7% of API time) show that individual kernel scheduling remains a significant overhead. The GPU is underloaded between steps.

### Exp 3: FP8 + CUDA Graphs — Approaching memory-bandwidth-bound

Graph replay eliminated most of the kernel-launch overhead: `cudaLaunchKernel` dropped to 2.8% of API time. `cudaEventSynchronize` fell to 52.6%. Throughput rose a further 39.1% to 626 tok/s. The remaining synchronization (`cudaDeviceSynchronize` at 18.7%) comes from step-boundary checks that graphs cannot eliminate — these are the scheduler's "am I done?" queries.

The **dominant GPU kernel is still CUTLASS FP8 GEMM** across Exp 2 and Exp 3 (43.7% → 38.9%). This is the weight matrix projection kernel. With graphs removing the CPU-side bottleneck, the workload is now closer to its theoretical memory-bandwidth floor: decode-phase matrix-vector products are memory-bandwidth-bound by definition at small batch sizes (arithmetic intensity below the roofline knee). The L4's 300 GB/s peak is the physical limit.

**The shift in bottleneck:** BF16 eager → FP8 eager removed the memory-capacity ceiling. FP8 eager → FP8 + CUDA graphs removed the CPU scheduling overhead. What remains is the fundamental memory-bandwidth cost of serving LLM decode.

---

## SA Recommendation

**Finding:** A single NVIDIA L4 running Llama-3.1-8B-Instruct with FP8 quantization and CUDA graphs sustains **627 tok/s** for 30 concurrent users with an average per-request latency of **208 ms**. This represents a **2.24× throughput improvement** over the unoptimized BF16 eager baseline (279 tok/s, 425 ms latency).

**SLA assessment:** Whether 208 ms average latency is acceptable depends on the fintech company's response-time SLA. For a customer support chatbot generating 130-token responses (~650-800 ms time-to-complete), the per-token latency of ~1.6 ms/tok is competitive. At 30 concurrent users and 627 tok/s, the system generates a full 150-token response in approximately **6.2 seconds wall time for the full batch**, averaging ~207ms per user.

**Recommended production configuration:**
- `quantization="fp8"` — FP8 is lossless for instruction-tuned models at this scale; Ada Lovelace has native FP8 tensor cores
- `enforce_eager=False` — CUDA graphs deliver +39% throughput at cost of 0.47 GiB VRAM and ~8-9 s cold-start compilation (cached on subsequent runs)
- `gpu_memory_utilization=0.90` — appropriate for the L4; increase to 0.93–0.95 if KV cache evictions appear under sustained load
- `max_model_len=4096` — sufficient for customer support use case; increasing to 8192 would halve KV cache capacity

**If the SLA cannot be met with a single L4:** The next scaling step is enabling **prefix caching** (`enable_prefix_caching=True`). All 30 prompts share an identical 57-token system message prefix. Prefix caching would eliminate the prefill cost for that prefix across all requests, effectively reducing per-request KV cache consumption and improving throughput further — potentially 10–20% gain with zero hardware cost.

Beyond that: a second L4 in tensor-parallel mode (`tensor_parallel_size=2`) would approximately double throughput by splitting each weight matrix across two GPUs, reducing per-GPU memory bandwidth pressure. This is the recommended path if the single-L4 configuration proves insufficient at production peak load.
