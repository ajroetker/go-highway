# MatMul Improvements: Precision and Throughput

## Background

Investigation into Florence-2 vision-language model inference revealed that the Go (simplego+highway) backend produces different autoregressive decoder output than XLA, CoreML, and ONNX Runtime backends. All three native backends produce identical results because they all call `cblas_sgemm` via Apple's Accelerate framework on macOS.

### Root Cause Analysis

**Stage-by-stage diff test** (`e2e/florence2_backend_diff_test.go`) compared Go vs XLA intermediate tensors at each pipeline stage, feeding identical inputs (from XLA) to both backends:

| Stage | Max Abs Diff | Mean Abs Diff | Elements >1% Rel Error |
|-------|-------------|---------------|----------------------|
| vision_encoder (577x768) | 2.16e-03 | 1.94e-05 | 987 / 443,136 (0.22%) |
| embed_tokens (9x768) | 0 (exact) | 0 | 0 |
| encoder_model (586x768) | 2.20e-04 | 2.74e-06 | 30 / 450,048 (0.01%) |
| decoder_first_step (1x51289) | 1.05e-05 | 1.78e-06 | 1 / 51,289 (0.00%) |

Each individual stage has small per-step differences. The decoder first step produces **identical top-5 token predictions** between Go and XLA. However, over the full autoregressive loop (~10+ decoder steps), these tiny per-step errors compound, eventually causing different token selections at some step, which then cascades into completely different output.

**With go-highway disabled** (vanilla simplego), the vision encoder error is **12x worse** (max 2.60e-02 vs 2.16e-03) and **5x slower** (70.7s vs 13.9s). go-highway is already helping significantly with both precision and throughput.

### Why Do Go and BLAS Differ?

Both go-highway and `cblas_sgemm` use **float32 accumulators** for float32 matmul. Neither uses float64 intermediate accumulation. All hardware paths (NEON `FMLA.4S`, SME `FMOPA`, Apple AMX, x86 `VFMADD231PS`) use standard IEEE 754 FMA semantics.

The differences come from:

1. **Accumulation order** (primary cause): Floating-point addition is non-associative. Different blocking strategies group partial sums differently: `(a+b+c) + (d+e+f)` vs `(a+b) + (c+d) + (e+f)` produce different float32 results. go-highway and BLAS use different Kc values, micro-tile shapes, and loop orderings.

2. **K-dimension blocking differences**:
   - `BaseMatMul` / `BlockedMatMul` / SME multi-tile: stream the **full K** in one pass (no intermediate rounding from K-blocking)
   - `PackedMatMul`: blocks K into chunks of Kc=256 (NEON), introducing `ceil(K/Kc)-1` extra roundings per element
   - BLAS: also K-blocks but with different Kc (e.g., Apple Accelerate uses Kc=384)

3. **Per-FMA precision**: Go's `math.FMA` fallback computes at float64 precision (marginally better per-op), while NEON `FMLA.4S` uses float32 FMA. Both avoid rounding the intermediate product before addition.

### Quantitative Error Analysis

For a float32 dot product of length K:
- **Naive sequential**: worst-case error O(K * eps), expected O(sqrt(K) * eps)
- **K-blocked**: adds `ceil(K/Kc)-1` extra ULP-scale roundings
- For K=768: expected per-element error ~sqrt(768) * 2^-24 = ~1.7e-6

This matches observed mean abs diffs (2e-6 for encoder, 2e-5 for vision encoder which has deeper computation).

---

## Precision Improvements

### 1. Pairwise Summation in K Loop (Recommended)

**Error bound**: O(log2(K) * eps) vs O(K * eps) for naive sequential.

For K=768: reduces worst-case error by ~77x (768 / log2(768) = 768/~10).

**Algorithm**: Instead of accumulating all K products sequentially into one accumulator, split into blocks of 128 elements and do a balanced binary-tree reduction of the block results.

```
// Current (naive sequential):
acc = 0
for k = 0 to K-1:
    acc += A[i,k] * B[k,j]
C[i,j] = acc

// Pairwise (block=128):
block_sums = []
for block_start = 0 to K-1 step 128:
    block_acc = 0
    for k = block_start to min(block_start+128, K)-1:
        block_acc += A[i,k] * B[k,j]
    block_sums.append(block_acc)
// Binary tree reduction of block_sums
C[i,j] = pairwise_reduce(block_sums)
```

**Throughput cost**: Essentially zero. Same number of FMA operations execute. Only difference is the order of the final reduction (one extra add per block boundary per output element). This is the default in NumPy and Julia.

**SIMD compatibility**: Excellent. The current kernels already hold multiple accumulators in registers across the full K loop. The change: split K loop into blocks of 128, accumulate each block into fresh accumulators, pairwise-reduce block results.

**Implementation in go-highway**:
- `BaseMatMul`: Split the K loop in the 4-tile column strategy. Each tile's 4 accumulators accumulate for 128 K iterations, then their results feed into a running pairwise tree.
- `BaseBlockedMatMul`: Same approach within the full-K inner loop.
- `PackedMatMul`: Already K-blocked at Kc boundaries. Add pairwise reduction of the Kc partial sums when writing back to C.
- SME multi-tile: Split the full-K FMOPA loop into 128-element blocks, zero ZA between blocks, accumulate block results in Z registers with pairwise reduction.
- NEON assembly: Split K loop, use fresh accumulator registers per block, tree-reduce at end.

### 2. Block-Kahan (Fast2Sum at Block Boundaries)

If pairwise summation is insufficient, apply Fast2Sum at each 128-element block boundary to capture the rounding error from block reduction and inject it into the next block.

**Error bound**: O(eps^2 * K/block) -- near-full float32 precision.

**Throughput cost**: ~10-20% (3 extra operations per block boundary per output element, amortized over 128 FMAs).

**Algorithm** (at each block boundary):
```
// Fast2Sum: compute sum and error of (running_sum + block_result)
// Requires |running_sum| >= |block_result| (typically true for growing sums)
t = running_sum + block_result
error += (running_sum - t) + block_result  // exact rounding error
running_sum = t
// After all blocks:
C[i,j] = running_sum + error
```

**SIMD compatibility**: Fast2Sum is purely arithmetic with no branches, vectorizes trivially across SIMD lanes.

### 3. Full Kahan Summation (Not Recommended)

**Error bound**: O(eps^2 * K) -- best possible.

**Throughput cost**: 3-4x (4 extra operations per FMA step, creates loop-carried dependency chain of 4 ops, making the inner loop latency-bound on Apple Firestorm's 4-cycle FMA latency).

**Recommendation**: Overkill for ML inference. Pairwise or Block-Kahan is sufficient.

### 4. Ozaki Scheme (Not Applicable)

Designed for emulating FP64 GEMM using FP16/FP8 tensor cores. Requires 4-9x more GEMMs. Only worthwhile when dedicated low-precision tensor cores are 10-100x faster than FP32 ALUs. Not applicable to CPU NEON/SME.

### Precision Summary

| Technique | Error Bound | Throughput Cost | SIMD Friendly | Recommendation |
|-----------|------------|-----------------|---------------|----------------|
| Naive sequential | O(K * eps) | Baseline | Yes | Current approach |
| **Pairwise (block=128)** | **O(log(K) * eps)** | **~0%** | **Yes** | **Implement first** |
| Block-Kahan (block=128) | O(eps^2 * K/block) | ~10-20% | Yes | If pairwise insufficient |
| Full Kahan per FMA | O(eps^2 * K) | ~300-400% | Partially | Overkill |
| Ozaki scheme | Configurable | 400-900% | Tensor core only | Not applicable |

---

## NEON Throughput Improvements (M1/M2/M3)

### Current State

The `matmul_klast_neon_f32` kernel uses a **4x4 micro-kernel** (Mr=4, Nr=4) vectorizing along K. Per K iteration:
- 8 loads (4 from A, 4 from B)
- 16 FMAs
- Compute-to-load ratio: 16 FMAs / 8 loads = **2:1**

Each accumulator is a full 128-bit vector holding partial sums across K. The 4x4 kernel uses 16 accumulator registers + 8 load registers = 24/32 NEON registers.

### 1. Switch to GEBP 6x8 Micro-Kernel (~1.5-2x Speedup)

High-performance implementations (XNNPACK, BLIS, ruy, KleidiAI) all use a GEBP (General Block Panel) packed layout instead of K-last dot-product:

**GEBP layout**:
- A packed as Mr x Kc column panels (column-major within panel)
- B packed as Kc x Nr row panels (row-major within panel)
- Micro-kernel computes rank-1 updates: `C[mr,nr] += A_panel[mr,1] * B_panel[1,nr]`

**6x8 GEBP micro-kernel on NEON**:
```
for k in range(Kc):
    a_col = load_6_elements(A_packed + k*6)     // 2 loads (4+2)
    b_row = load_8_elements(B_packed + k*8)     // 2 loads
    // 6*2 = 12 FMAs via broadcast-multiply:
    for i in range(6):
        for j in range(2):  // 2 vectors of 4 floats = 8 columns
            C[i][j] = fmla(C[i][j], broadcast(a_col[i]), b_row[j])
```

**Register budget**:
- C accumulators: 6 * 2 = **12 registers**
- A broadcast: reuse with `fmla v, v, v[lane]` = **2 registers** (loaded as 2 quad-words)
- B row: **2 registers**
- Misc: **4 registers** (pointers, loop counter)
- Total: **20/32 registers**

**Performance**:
- 12 FMAs from 4 loads per K step = **3:1** compute-to-load ratio
- With 2x K-unrolling: 24 FMAs per iteration, saturating Firestorm's 4 FMA pipes (need 16 independent FMAs to fill the 4-cycle pipeline)
- **Expected 1.5-2x speedup** over current 4x4 K-last

**Packing cost**: For M=577, N=768, K=768: packing is O(M*K + N*K) = O(1.2M elements), compute is O(M*N*K) = O(340M FMAs). Packing overhead < 1%.

XNNPACK uses this exact 6x8 shape for their `f32-gemm/6x8-aarch64-neonfma-cortex-a75.S.in` kernel.

### 2. Specialized 1xNr Kernel for Decoder (M=1)

During autoregressive decoding, matmuls have M=1 (vector-matrix multiply). These are memory-bandwidth bound, not compute bound:
- Compute-to-byte ratio: 2*K*N FLOPs / (K*N + K + N) * 4 bytes = ~2 FLOPs/byte
- Apple M-series bandwidth: ~100-200 GB/s → ~200-400 GFLOPS effective

**Optimizations for M=1**:
1. Skip packing entirely (packing overhead dominates)
2. Use 1x8 or 1x16 streaming micro-kernel (stream one row of A against columns of B)
3. Quantized weights (INT8/NF4) halve memory traffic, doubling effective bandwidth

XNNPACK provides `1x8-aarch64-neonfma-cortex-a75.S.in` for exactly this case.

### 3. Cache Blocking Parameters

For Florence-2's dominant shape (577x768x768):

| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| Mc | 48 | 120 | Larger M block for better A panel reuse (120*256*4 = 120KB, fits L2) |
| Kc | 256 | 256 | Good fit for L1 (Mr*Kc*4 = 6*256*4 = 6KB per micro-panel) |
| Nc | - | 768 | Full N in one pass (B panel = 256*768*4 = 768KB, fits L2) |

### 4. K-Unrolling (2x-4x)

Unroll the K loop 2x-4x with software pipelining to hide load latency:

```asm
// K iteration i: compute with previously loaded data
// while loading data for iteration i+1
LDP  q16, q17, [A_ptr, #next]    // prefetch A for k+1
LDP  q18, q19, [B_ptr, #next]    // prefetch B for k+1
FMLA v0.4s, v8.4s, v12.s[0]      // compute k, row 0
FMLA v1.4s, v8.4s, v12.s[1]      // compute k, row 1
...
```

Expected additional ~1.1-1.3x speedup from better instruction-level parallelism.

### 5. Prefetch Strategy

For packed GEBP on Apple Silicon:
- **A panel**: `PRFM PLDL1KEEP` 4-8 K iterations ahead (64 bytes = 16 float32s per cache line)
- **B panel**: Hardware prefetcher handles linear streaming well; explicit `PRFM PLDL2KEEP` for large panels
- **C writeback**: `PRFM PSTL1KEEP` before micro-kernel writes

Apple Firestorm has a strong hardware prefetcher, so explicit prefetch has less impact than on Cortex cores but doesn't hurt.

### NEON Summary

| Change | Current | Proposed | Expected Speedup |
|--------|---------|----------|-----------------|
| Micro-kernel shape | 4x4 K-last | 6x8 GEBP | ~1.5-2x |
| Packing | None | Column-panel A, row-panel B | Enables GEBP |
| Cache blocking | BlockSize=48 | mc=120, kc=256, nc=768 | ~1.1-1.2x |
| K-unrolling | 1x | 2x-4x (with SW pipeline) | ~1.1-1.3x |
| M=1 specialized kernel | Uses general path | 1x8 streaming, no pack | Faster decode |
| **Total** | | | **~2-3x** |

---

## SME FMOPA Throughput Improvements (Apple M4)

### Current State

Apple M4 SME specifications (SVL=512):
- 4 ZA tiles for FP32: ZA0.S through ZA3.S, each 16x16 = 256 elements
- FMOPA FP32 latency: **4 cycles**
- FMOPA FP32 throughput: **1 per cycle** (when rotating across all 4 tiles)
- Peak: **2,009 GFLOPS** per P-core
- Single-tile only: ~502 GFLOPS (1/4 of peak, confirming 4-cycle latency)

Current implementation (`multitile_fmopa_arm64.c`):
- 2x2 tile layout (32x32 output blocks)
- 2 A columns + 2 B rows = 4 FMOPAs per K iteration
- GoAT-generated code is **11-27% slower** than handwritten due to insufficient memory latency hiding

### 1. K-Unroll 4x with Software-Pipelined Loads (~10-25% Speedup)

The `DARWIN_SME.md` documents that GoAT generates a 7-instruction tight K-loop, while handwritten code has 13+ instructions that hide load latency. The fix is interleaving loads for K+1 with FMOPAs for K:

```asm
// K iteration i: FMOPA uses previously loaded data
// while loads for iteration i+1 execute in parallel
LD1W {z_a_next}, p0/z, [A_ptr, #next_offset]
LD1W {z_b_next}, p0/z, [B_ptr, #next_offset]
FMOPA za0.s, p0/m, p0/m, z_a_curr, z_b_curr_0
FMOPA za1.s, p0/m, p0/m, z_a_curr, z_b_curr_1
LD1W {z_b_next2}, p0/z, [B_ptr, #next_offset2]
FMOPA za2.s, p0/m, p0/m, z_a_curr, z_b_curr_2
FMOPA za3.s, p0/m, p0/m, z_a_curr, z_b_curr_3
```

With 4x K-unrolling, this uses ~20 Z registers for buffering (out of 32 available in streaming mode). MpGEMM achieves **94% of peak** with this approach.

### 2. Contiguous Group Loads (~5-10% Speedup)

Apple M4 can load 4 Z registers in a single cycle with `LD1W {z0.s - z3.s}`:
- Loading 1 Z register/cycle: 375 GB/s
- Loading 2 Z registers/cycle (LD1W pair): 625 GB/s
- Loading 4 Z registers/cycle (contiguous group): 900 GB/s

For the 2x2 layout: load both B vectors as a pair (2 loads for A0,A1 + 1 group load for B0,B1) = 3 load ops for 4 FMOPAs.

### 3. Consider 1x4 Tile Layout (16x64) for Large N

| Layout | Tiles | Mr x Nr | FMOPAs/K | Loads/K | Balance |
|--------|-------|---------|----------|---------|---------|
| 1x4 | ZA0-ZA3 | 16x64 | 4 | 5 (1A+4B) | Load-bound |
| **2x2** | ZA0-ZA3 | **32x32** | **4** | **4 (2A+2B)** | **Balanced** |
| 4x1 | ZA0-ZA3 | 64x16 | 4 | 5 (4A+1B) | Load-bound |

The current 2x2 layout (32x32) is naturally balanced at 4 loads per 4 FMOPAs. For Florence-2's N=768:
- 2x2 (Nr=32): 768/32 = 24 tiles along N
- 1x4 (Nr=64): 768/64 = 12 tiles along N (fewer iterations, but load-bound without group loads)

With contiguous group loads, 1x4 becomes 2 load ops for 4 FMOPAs -- better than balanced. Consider benchmarking both for Florence-2 shapes.

### 4. Multi-Core Consideration

The SME unit is **shared per CPU cluster** -- all P-cores share one SME unit. A single core fully saturates it. For multi-threaded inference:
- Don't parallelize a single matmul across cores for SME
- Use different cores for different matmuls (different layers or batch elements)
- NEON matmul can still benefit from multi-core parallelism

### SME Performance Targets

| Implementation | GFLOPS | % of Peak (2009) |
|---------------|--------|------------------|
| FMOPA microbenchmark (all tiles) | 2,009 | 100% |
| LIBXSMM JIT (512x512) | 1,833 | 91% |
| Apple Accelerate (512x512) | 1,825 | 91% |
| MpGEMM mixed-precision (M4 Pro) | ~1,889 | 94% |
| **Target for go-highway** | **1,700-1,830** | **85-91%** |

### SME Summary

| Change | Expected Improvement | Effort |
|--------|---------------------|--------|
| K-unroll 4x + SW pipeline | ~10-25% throughput | Medium |
| Contiguous group loads | ~5-10% throughput | Low |
| 1x4 tile layout (benchmark) | Potentially better for large N | Medium |
| Pairwise summation | ~7x precision, ~0% cost | Low |

---

## Florence-2 Workload-Specific Analysis

### Matrix Shapes

Florence-2 Base (0.23B params): d_model=768, heads=12, head_dim=64, 6 encoder + 6 decoder layers.

| Operation | M | K | N | FLOPs | Notes |
|-----------|---|---|---|-------|-------|
| Q/K/V projection (encoder) | 577 | 768 | 768 | 681M | 3x per layer |
| Attention QK^T (per head) | 577 | 64 | 577 | 43M | 12 heads |
| Attention V multiply | 577 | 577 | 64 | 43M | 12 heads |
| FFN up-projection | 577 | 768 | 3072 | 2.7G | |
| FFN down-projection | 577 | 3072 | 768 | 2.7G | |
| Decoder Q/K/V (autoregressive) | 1-T | 768 | 768 | ~1.5M/tok | M=1 for greedy |
| Cross-attention (decoder) | 1-T | 768 | 768 | ~1.5M/tok | Against encoder out |

### Expected Timing per Shape

| Shape | Strategy | NEON (current) | NEON (improved) | SME (improved) |
|-------|----------|---------------|-----------------|----------------|
| 577x768x768 | GEBP 6x8 | ~15ms | ~7.6ms | ~0.4ms |
| 577x768x3072 | GEBP 6x8 | ~60ms | ~30ms | ~1.6ms |
| 577x3072x768 | GEBP 6x8 | ~60ms | ~30ms | ~1.6ms |
| 12x577x64x577 | K-last 4x4 | ~1ms | ~0.5ms | ~0.03ms |
| 1x768x768 | 1xNr streaming | ~0.02ms | ~0.01ms | BW-bound |

### Attention-Specific Optimizations

For attention QK^T (K=64, small), the current K-last 4x4 kernel is actually well-suited:
- K=64 means only 16 SIMD iterations in the inner loop -- small enough to fully unroll
- No need for K-blocking
- Horizontal reduction overhead is amortized well

Consider pre-transposing K so QK^T can use the same optimized kernel path as other matmuls, avoiding runtime transpose cost.

---

## Implementation Priority

| # | Change | Target | Precision | Throughput | Effort |
|---|--------|--------|-----------|------------|--------|
| 1 | **Pairwise summation (block=128)** | All backends | **~7x better** | ~0% cost | Low |
| 2 | **GEBP 6x8 micro-kernel** | NEON | Same | **~1.5-2x** | Medium |
| 3 | **1xNr decode kernel** | NEON | Same | Faster M=1 | Low |
| 4 | **K-unroll 4x + SW pipeline** | SME | Same | **~10-25%** | Medium |
| 5 | **Contiguous group loads** | SME | Same | **~5-10%** | Low |
| 6 | Block-Kahan at boundaries | All | ~full f32 | ~10-20% cost | Low |

## References

- [Pairwise summation - Wikipedia](https://en.wikipedia.org/wiki/Pairwise_summation)
- [Taming Floating-Point Sums (orlp.net)](https://orlp.net/blog/taming-float-sums/)
- [Fast, accurate summation (Bjornson)](http://blog.zachbjornson.com/2019/08/11/fast-float-summation.html)
- [Kahan summation - Wikipedia](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)
- [2Sum robustness (ACM)](https://dl.acm.org/doi/10.1145/3054947)
- [Accurate Sum and Dot Product on ARMv8](https://www.mdpi.com/2227-7390/13/2/270)
- [Vectorized Kahan and Gill-Moller](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.7763)
- [BLIS Micro-Kernels HowTo](https://github.com/flame/blis/blob/master/docs/KernelsHowTo.md)
- [BLISlab: A Sandbox for Optimizing GEMM](https://arxiv.org/pdf/1609.00076)
- [XNNPACK f32-gemm 6x8 AArch64 kernel](https://github.com/google/XNNPACK/blob/master/src/f32-gemm/6x8-aarch64-neonfma-cortex-a75.S.in)
- [XNNPACK f32-gemm 1x8 AArch64 kernel](https://github.com/google/XNNPACK/blob/master/src/f32-gemm/1x8-aarch64-neonfma-cortex-a75.S.in)
- [KleidiAI (ARM)](https://github.com/ARM-software/kleidiai)
- [Apple M1 Firestorm instruction benchmarks](https://dougallj.github.io/applecpu/firestorm.html)
- [M4 SME Exploration (tzakharko)](https://github.com/tzakharko/m4-sme-exploration)
- [Hello SME! Fast Matrix Multiplication Kernels (LIBXSMM)](https://arxiv.org/pdf/2409.18779)
- [Demystifying ARM SME for GEMM](https://arxiv.org/html/2512.21473)
- [SME Microbenchmarks (Jena)](https://scalable.uni-jena.de/opt/sme/micro.html)
- [SME GEMM Performance (Jena)](https://scalable.uni-jena.de/opt/sme/gemm.html)
- [Apple vs. Oranges: M-Series HPC](https://arxiv.org/html/2502.05317v1)
- [ARM SME Introduction Part 2](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction-p2)
- [Neon, SVE, and SME compared (ARM)](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/matrix-matrix-multiplication-neon-sve-and-sme-compared)
- [Mixed Precision Block FMA error analysis (Higham & Mary)](https://epubs.siam.org/doi/10.1137/19M1289546)
- [Ozaki Scheme (HPCwire)](https://www.hpcwire.com/2025/04/17/have-you-heard-about-the-ozaki-scheme-you-will/)
- [NVIDIA cuBLAS FP Emulation](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
- [Apple AMX Coprocessor Research (meekolab)](https://research.meekolab.com/the-elusive-apple-matrix-coprocessor-amx)
- [Performance Analysis of Apple AMX (MIT)](https://commit.csail.mit.edu/papers/2025/Jonathan_Zhou_SB_Thesis.pdf)
- [Florence-2 CVPR 2024 Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Florence-2_Advancing_a_Unified_Representation_for_a_Variety_of_Vision_CVPR_2024_paper.pdf)
