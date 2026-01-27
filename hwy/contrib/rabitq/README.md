# RaBitQ SIMD Operations

RaBitQ is a state-of-the-art binary quantization method for approximate nearest neighbor (ANN) search. This package provides SIMD-accelerated implementations of the core RaBitQ operations.

## Overview

RaBitQ quantizes high-dimensional vectors into compact binary codes, enabling fast distance estimation using bit operations. The key operations are:

1. **Vector Quantization**: Convert unit vectors to 1-bit codes
2. **Bit Product**: Compute weighted popcount for distance estimation

## Usage

```go
import "github.com/ajroetker/go-highway/hwy/contrib/rabitq"

// Compute weighted bit product for distance estimation
// This is the hot path in RaBitQ search, called for every candidate vector
result := rabitq.BitProduct(code, q1, q2, q3, q4)

// Quantize unit vectors to 1-bit codes
rabitq.QuantizeVectors(
    unitVectors,   // flattened array of unit vectors (count × dims float32s)
    codes,         // output buffer for quantization codes (count × width uint64s)
    dotProducts,   // output buffer for inverted dot products (count float32s)
    codeCounts,    // output buffer for bit counts (count uint32s)
    sqrtDimsInv,   // precomputed 1/√dims
    count,         // number of vectors to process
    dims,          // dimensions per vector
    width,         // number of uint64s per code (typically ⌈dims/64⌉)
)

// Helper to compute code width
width := rabitq.CodeWidth(dims) // returns ⌈dims/64⌉
```

## BitProduct

The `BitProduct` function computes the weighted bit product used in RaBitQ distance estimation:

```
1×popcount(code & q1) + 2×popcount(code & q2) + 4×popcount(code & q3) + 8×popcount(code & q4)
```

This operation is the performance-critical inner loop when searching through candidate vectors.

## QuantizeVectors

The `QuantizeVectors` function converts unit vectors into 1-bit codes:

1. Extracts sign bits (1 for positive/zero, 0 for negative)
2. Packs bits into uint64 codes (MSB-first within each uint64)
3. Computes the dot product between the unit vector and its quantized form
4. Counts the number of 1-bits in the code

The `dotProducts` output contains `1/<o̅,o>` (inverted) for use in distance estimation.

## SIMD Acceleration

This package automatically uses the best available SIMD instructions:

| Architecture | Implementation |
|--------------|----------------|
| AMD64 AVX2 | 256-bit vectors with VPOPCNTDQ emulation |
| AMD64 AVX-512 | 512-bit vectors with native VPOPCNTDQ |
| ARM64 NEON | 128-bit vectors with CNT instruction |
| Fallback | Pure Go with math/bits.OnesCount64 |

## Performance

The SIMD implementations provide significant speedups over scalar code:

- **BitProduct**: 4-8x faster than scalar depending on vector length
- **QuantizeVectors**: 2-4x faster than scalar

## References

- [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2405.12497)
