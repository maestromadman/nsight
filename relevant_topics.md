### This is a summary of topics embedded in this project. I wrote this post-project to wrap all the concepts together and strengthen my understanding.

# Topic Summary

## GPU Architecture
### **Ampere** is the GPU arhcitecture my L4 is based on. Everything runs on top of this parallel processor.

## Host vs. Device
### In this context, the CPU is denoted the "host" and the GPU is denoted the "device"; they are separate processors and their memories are separate. Thus, data must be explicitly copied between them. This yields transfer overhead. 
###During inferenc,e the input token IDs are copied from the CPU to the GPU before processing occurs. Model weights also live on the GPU but might paritally be managed from the CPU. In **findings_raw.txt**, you will find "cudaMemcpyAsync", the CUDA function that performs the copy (Async means that it starts the copy and allows other work to proceed in parallel).

## CUDA Kernels
### A CUDA Kernel is a function that executes on the GPU across parallel threads simultaneously. Every GPU operatoin (i.e. matrix multiplicatoin, memory fills, attention computation) is implemented as one or more CUDA kernels. Nsight shows a kernel summary, eveyr GPU function that ran and how long it took. This is helpful for identifying where bottlenecks occur.

## GEMM - General Matrix Multiply 
### Nearly every operation in a transformer NN (e.g. input projection, attention score computatoins, FFNN layers) revolves around matrix multiplication. Thus, GEMM kernels are important for LLM profiling. When they dominate the trace, the GPU is doing useful work. Additionally, **CUTLASS** is NVIDIA's open-source library of hand-optimized GEMM kernels - vLLM calls these under the hood to maximize performance on NVIDIA hardware.

## Transformer Operations (mentioned above)
### Attention projections - input multiplied by Query (Q), Key (K), and Value (V) weight matrices to produce Q,K,V vectors. These are used to compute which tokens should attend to which other tokens. 
### Feed-forward matrix multiplicatoins - after attention, each layer passes through a 2-layer FFNN. These dominate compute time and are compute-bound.

## Prefill vs. Decode
### Prefill - One of the 2 distinct phases of LLM inference, **prefill** processes the entire input prompt at once, in parallel. There are lots of large matrix multiplicatoins occurring simulataneously.
### Decode - The second of the distinct phases of LLM inference, **decode** generates 1 output token at a time, each attending to all previous tokens. This is memory-bandwidth-bound, i.e. small matrix operations taht require reading large amounts of data from GPU memory (this is where most inference time is spent).

## KV Cache
### During the decode portoin of LLM inference, the model uses K and B vectors from every previously generated token to compute attention. Recomputing them from scratch is wasteful, so KV cache stores them in GOU memory as they're computed, growing with each new token generated. This is a primary consumer of VRAM during inference.

## PagedAttention
### Thhis manages KV cache in fixed-size pages, allowing vLLM to pack far more concurrent requests onto the GPU, improving throughput. 

## CPU-GPU Synchronizatoin Overhead
### The CPU and GPU must periodically syncrhonize, and the CPU waits for the GPU to finish before moving to the next step. cudaEvent Synchronize enforces this wait. This can create large overhead, if the CPU schedules work for the GPU in small incrememtns, resulting in frequent wait-and-proceed cycles.

## Batch size
### Batch size is how many prompts are processed togethe rin one GPU pass. Larger batches amortize the fixed overhead (synchronization, scheduling, etc.) across more useful work. For example, a batch of 1 might use 30% of GPU capacity, whereas a batch of 64 might use 90%. In vLLM, batch size is controlled by the max_num_seqs enginer parameter.


#Standard Bottlenecks
### **Compute-Bound** : When kernel compute time is high and GPU utilizzation is near 100% --> Fix: Quantization (reduces arithmetic cost/operation)
### **Memory-bandwidth-bound** : GPU computes faster than reading data from VRAM. --> Fix: KV cache quantization, PagedAttention
### **Synchronization Overhead** : *cudaEventSyncrhonize* consuming a large share of CUDA API time --> Fix: larger batch size
### **Host-to-Device Transfer Overhead** : *cudaMemcpyAsync* consuming lots of time --> Fix: pin weights on GPU, minimize data movement
