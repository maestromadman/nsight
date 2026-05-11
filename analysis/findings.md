# vLLM Performance Engineering — Findings

## Project Overview

**Business scenario:** A fintech company operates a customer support chatbot handling billing disputes, account inquiries, and transaction questions. Peak concurrent load is 30 simultaneous users. The company is evaluating whether a single NVIDIA L4 GPU can meet production SLAs before committing to a multi-GPU deployment.

**Hardware:** NVIDIA L4 (24 GB GDDR6, 300 W TDP, Ada Lovelace architecture, FP8 tensor core support)  
**Software stack:** CUDA 12.4 · Python 3.10 · vLLM (latest) · Nsight Systems  
**Model:** `meta-llama/Llama-3.1-8B-Instruct` (8B parameters, BF16 ≈ 16 GB weight footprint)  
**Workload:** 30 realistic customer support prompts · `max_tokens=150` · `temperature=0.3`  
**Objective:** Quantify throughput, latency, and GPU utilization for three progressively optimized configurations and determine if a single L4 can sustain 30 concurrent users at acceptable latency.

---

## Experiment 1: BF16 Baseline

### What we expect to find

With `enforce_eager=True` and no quantization, the model runs every transformer layer as individual eager CUDA kernel launches. Each decode step fires hundreds of separate kernels — attention projections, FFN layers, softmax, sampling — and the CPU must enqueue each one individually. On a batched workload of 30 requests, the L4 spends significant time waiting for the host to schedule the next kernel rather than doing compute. We expect:

- High `cudaEventSynchronize` share: the CPU blocks frequently waiting for GPU completion before scheduling the next operation.
- Moderate-to-high `cudaMemcpyAsync` share: BF16 weights occupy ~16 GB, consuming most of the 24 GB VRAM and leaving limited KV cache headroom. Cache evictions or tight paging increase device-to-device copies.
- Decode phase dominated by memory bandwidth rather than compute (roofline: short decode steps are memory-bandwidth-bound on matrix-vector products, not matrix-matrix).
- Throughput and latency that serve as the unoptimized baseline.

### Measured Results

| Metric | Value |
|---|---|
| Total wall time (s) | [PLACEHOLDER: wall time] |
| Total output tokens | [PLACEHOLDER: token count] |
| Throughput (tok/s) | [PLACEHOLDER: throughput] |
| Avg latency per request (s) | [PLACEHOLDER: avg latency] |
| P50 latency per request (s) | [PLACEHOLDER: p50 latency] |
| `cudaEventSynchronize` % of CUDA API time | [PLACEHOLDER: %] |
| `cudaMemcpyAsync` % of CUDA API time | [PLACEHOLDER: %] |
| Top kernel (name + % of GPU time) | [PLACEHOLDER: kernel name + %] |
| 2nd kernel (name + % of GPU time) | [PLACEHOLDER: kernel name + %] |
| 3rd kernel (name + % of GPU time) | [PLACEHOLDER: kernel name + %] |
| Bottleneck classification | [PLACEHOLDER: compute-bound / memory-bound / synchronization-bound] |
| KV cache allocated (GB) | [PLACEHOLDER: GB] |

### Nsight Interpretation

[PLACEHOLDER: narrative — e.g., "cudaEventSynchronize accounts for X% of CUDA API time, confirming the CPU is stalling on GPU completion. The top kernel (flash_attn_varlen_fwd at Y% of GPU time) indicates the attention pass is the compute hotspot, while the memory bandwidth utilization of Z GB/s is well below the L4's peak 300 GB/s, consistent with a synchronization-bound, not compute-bound, profile."]

---

## Experiment 2: FP8 Quantization

### What FP8 changes mechanically

FP8 quantization (E4M3 format) stores each weight in 8 bits instead of 16, cutting the model weight footprint from ~16 GB to ~8 GB. This has two first-order effects on the L4:

1. **KV cache headroom doubles.** With 8 GB freed, vLLM can allocate significantly more KV cache blocks, reducing evictions and recomputation for 30-concurrent-user batches.
2. **Memory bandwidth pressure decreases.** During the decode matrix-vector products (weight × activation), loading a row of weights from HBM takes half the memory transactions. Because decode steps are memory-bandwidth-bound at small batch sizes, this directly improves token throughput.

The compute cost of FP8 dequantization is amortized by the tensor core's native FP8 support on Ada Lovelace — it does not require a software dequant step.

### Measured Results

| Metric | Value |
|---|---|
| Total wall time (s) | [PLACEHOLDER: wall time] |
| Total output tokens | [PLACEHOLDER: token count] |
| Throughput (tok/s) | [PLACEHOLDER: throughput] |
| Avg latency per request (s) | [PLACEHOLDER: avg latency] |
| P50 latency per request (s) | [PLACEHOLDER: p50 latency] |
| `cudaEventSynchronize` % of CUDA API time | [PLACEHOLDER: %] |
| `cudaMemcpyAsync` % of CUDA API time | [PLACEHOLDER: %] |
| Top kernel (name + % of GPU time) | [PLACEHOLDER: kernel name + %] |
| 2nd kernel (name + % of GPU time) | [PLACEHOLDER: kernel name + %] |
| 3rd kernel (name + % of GPU time) | [PLACEHOLDER: kernel name + %] |
| Bottleneck classification | [PLACEHOLDER: compute-bound / memory-bound / synchronization-bound] |
| KV cache allocated (GB) | [PLACEHOLDER: GB] |

### Nsight Interpretation

[PLACEHOLDER: narrative — e.g., "cudaMemcpyAsync % dropped from X% to Y%, reflecting the reduced weight data movement. Throughput improved by Z%, consistent with the L4's memory-bandwidth ceiling being relieved. cudaEventSynchronize remains high (W%), indicating CPU scheduling overhead is still the binding constraint."]

---

## Experiment 3: FP8 + CUDA Graphs

### What CUDA graphs change mechanically

In eager mode (Experiments 1 and 2), each decode step requires the Python runtime and CUDA driver to individually enqueue every kernel — potentially 200–400 separate `cudaLaunchKernel` calls per token, each incurring a CPU→GPU round trip of ~5–20 µs. For a 30-request batch generating up to 150 tokens each, this accumulates to billions of microseconds of pure scheduling overhead.

CUDA graphs solve this by recording the entire decode graph (all kernel launches, dependencies, memory addresses) during a capture phase, then replaying it with a single `cudaGraphLaunch` call. Since the decode graph is static in shape across steps (same operators, same batch size for a given graph slot), the capture is valid for every subsequent token.

Combined effects:
- CPU scheduling overhead drops to near zero during decode.
- The GPU runs continuously without stalling for the host to launch the next kernel.
- SM utilization increases, and `cudaEventSynchronize` share collapses.
- First-token latency (prefill phase) is unaffected — only the decode phase is captured.

### Measured Results

| Metric | Value |
|---|---|
| Total wall time (s) | [PLACEHOLDER: wall time] |
| Total output tokens | [PLACEHOLDER: token count] |
| Throughput (tok/s) | [PLACEHOLDER: throughput] |
| Avg latency per request (s) | [PLACEHOLDER: avg latency] |
| P50 latency per request (s) | [PLACEHOLDER: p50 latency] |
| `cudaEventSynchronize` % of CUDA API time | [PLACEHOLDER: %] |
| `cudaMemcpyAsync` % of CUDA API time | [PLACEHOLDER: %] |
| Top kernel (name + % of GPU time) | [PLACEHOLDER: kernel name + %] |
| 2nd kernel (name + % of GPU time) | [PLACEHOLDER: kernel name + %] |
| 3rd kernel (name + % of GPU time) | [PLACEHOLDER: kernel name + %] |
| Bottleneck classification | [PLACEHOLDER: compute-bound / memory-bound / synchronization-bound] |
| KV cache allocated (GB) | [PLACEHOLDER: GB] |

### Nsight Interpretation

[PLACEHOLDER: narrative — e.g., "cudaEventSynchronize dropped from X% (Exp 2) to Y%, confirming that CPU scheduling was the primary bottleneck eliminated by graph replay. The GPU timeline now shows continuous kernel execution with minimal gaps. The remaining bottleneck is memory bandwidth during decode, which is the theoretical floor for this workload on the L4."]

---

## Bottleneck Analysis

The three experiments expose a layered bottleneck structure that peels away one constraint at a time.

**BF16 Eager (Exp 1) — Synchronization-bound**  
In the baseline, the CPU is the bottleneck. Eager kernel launch serializes work dispatch: the Python scheduler and CUDA driver enqueue kernels one at a time, and `cudaEventSynchronize` shows the CPU stalling on GPU feedback. The GPU is underutilized not because it lacks compute capacity, but because it cannot receive work fast enough. Memory bandwidth is also under pressure: 16 GB of BF16 weights compete directly with the KV cache for VRAM.

**FP8 Quantization (Exp 2) — Synchronization + Memory-bandwidth-bound**  
FP8 halves weight size. The roofline shifts: with more KV cache headroom and fewer bytes transferred per decode step, memory bandwidth utilization approaches the L4's ceiling (300 GB/s) more quickly. Throughput improves, but `cudaEventSynchronize` stays high because the CPU scheduling overhead was not addressed. The GPU is now hitting memory bandwidth before it has a chance to be compute-limited.

**FP8 + CUDA Graphs (Exp 3) — Memory-bandwidth-bound (theoretical floor)**  
Graph replay eliminates the CPU from the decode critical path. Kernels run back-to-back, SM utilization rises, and `cudaEventSynchronize` collapses. The remaining limit is the L4's HBM bandwidth: for small-batch, short-sequence decode steps, matrix-vector products are memory-bound by definition (arithmetic intensity < the roofline knee). This is the physical floor for this workload on this GPU — further gains require a larger batch size, prefix caching, or speculative decoding.

---

## SA Recommendation

[PLACEHOLDER: fill in after running all three experiments]

**Decision framework:**
- If Exp 3 throughput ≥ [target tok/s] and P99 latency ≤ [SLA threshold]: **single L4 is sufficient** for 30 concurrent users at the tested prompt/response length.
- If throughput or latency miss SLAs: document the gap and recommend either (a) enabling prefix caching (the system prompt is identical across all 30 users — this is a strong prefix cache candidate), (b) speculative decoding to reduce decode steps, or (c) a second L4 in tensor-parallel mode.

**Expected recommendation language:**  
"Based on the profiling results, [a single L4 can / cannot] sustain 30 concurrent users at the tested max_tokens=150. The primary gains came from [FP8 quantization / CUDA graphs], which improved throughput by [X]% over baseline. To meet the [SLA] SLA, we recommend deploying with FP8 + CUDA graphs enabled and evaluating prefix caching for the shared system prompt. If load exceeds [N] concurrent users, a second L4 in TP=2 mode is the next scaling step."
