# Zero-Allocation Transform API Plan

Add a C++ Highway-style Transform API to go-highway that bypasses the `hwy.Vec` wrapper for zero-allocation bulk operations.

## Problem

Current benchmarks show SIMD is **slower** than stdlib due to allocations:

| Function | ns/op | allocs/op | vs Stdlib |
|----------|-------|-----------|-----------|
| Exp32 | 22,113 | 256 | 2.7x slower |
| Exp32_Stdlib | 8,216 | 0 | baseline |

**Root cause**: The `hwy.Vec` wrapper allocates on every operation:
- `make([]float32, n)` in wrapper functions
- `hwy.Load(result)` allocates again
- Polynomial evaluation creates 14+ temporary allocations

**Solution**: The raw `archsimd` functions (`Exp_AVX2_F32x8`) are already zero-allocation. Expose a Transform API that uses them directly.

## Design

### API Style (matching C++ Highway)

```go
// C++ Highway pattern:
// void Transform(const float* in, float* out, size_t n, Op op);

// Go equivalent (no error return, caller ensures correct sizes):
func ExpTransform(input, output []float32)
func LogTransform(input, output []float32)
func SinTransform(input, output []float32)
// ... etc
```

### File Structure

| File | Purpose |
|------|---------|
| `hwy/contrib/transform.go` | Generic Transform helper + scalar fallbacks |
| `hwy/contrib/transform_avx2.go` | AVX2 implementations using archsimd |
| `hwy/contrib/transform_test.go` | Benchmarks comparing Vec vs Transform APIs |

## Implementation

### 1. Core Transform Pattern (`transform_avx2.go`)

```go
//go:build amd64 && goexperiment.simd

package contrib

import (
    "simd/archsimd"
    "github.com/ajroetker/go-highway/hwy"
)

// ExpTransform applies exp(x) to each element with zero allocations.
// Caller must ensure len(output) >= len(input).
func ExpTransform(input, output []float32) {
    if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
        expTransformAVX2(input, output)
    } else {
        expTransformScalar(input, output)
    }
}

func expTransformAVX2(input, output []float32) {
    n := min(len(input), len(output))

    // Process 8 float32s at a time (AVX2)
    for i := 0; i+8 <= n; i += 8 {
        x := archsimd.LoadFloat32x8Slice(input[i:])
        Exp_AVX2_F32x8(x).StoreSlice(output[i:])
    }

    // Scalar tail (no allocation needed)
    for i := (n / 8) * 8; i < n; i++ {
        output[i] = float32(math.Exp(float64(input[i])))
    }
}
```

### 2. Scalar Fallback (`transform.go`)

```go
//go:build !amd64 || !goexperiment.simd

package contrib

import "math"

func ExpTransform(input, output []float32) {
    expTransformScalar(input, output)
}

func expTransformScalar(input, output []float32) {
    for i := range input {
        output[i] = float32(math.Exp(float64(input[i])))
    }
}
```

### 3. Float64 Variants

Same pattern with `Float64x4` and stride of 4.

### 4. In-Place Variant

```go
// ExpTransformInPlace applies exp(x) in-place
func ExpTransformInPlace(data []float32) {
    ExpTransform(data, data)
}
```

## Functions to Implement

| Function | F32 (stride 8) | F64 (stride 4) |
|----------|----------------|----------------|
| ExpTransform | archsimd.Float32x8 | archsimd.Float64x4 |
| LogTransform | archsimd.Float32x8 | archsimd.Float64x4 |
| SinTransform | archsimd.Float32x8 | archsimd.Float64x4 |
| CosTransform | archsimd.Float32x8 | archsimd.Float64x4 |
| TanhTransform | archsimd.Float32x8 | archsimd.Float64x4 |
| SigmoidTransform | archsimd.Float32x8 | archsimd.Float64x4 |
| ErfTransform | archsimd.Float32x8 | archsimd.Float64x4 |

## Benchmarks

Add to `transform_test.go`:

```go
func BenchmarkExpTransform(b *testing.B) {
    input := make([]float32, 1024)
    output := make([]float32, 1024)
    for i := range input {
        input[i] = float32(i) * 0.01
    }

    b.ResetTimer()
    b.ReportAllocs()
    for i := 0; i < b.N; i++ {
        ExpTransform(input, output)
    }
}

func BenchmarkExpTransform_Stdlib(b *testing.B) {
    input := make([]float32, 1024)
    output := make([]float32, 1024)
    for i := range input {
        input[i] = float32(i) * 0.01
    }

    b.ResetTimer()
    b.ReportAllocs()
    for i := 0; i < b.N; i++ {
        for j := range input {
            output[j] = float32(math.Exp(float64(input[j])))
        }
    }
}
```

Expected results:
- Transform API: 0 allocs/op for main loop
- 3-5x speedup over stdlib for large arrays

## Files to Create/Modify

| File | Action |
|------|--------|
| `hwy/contrib/transform.go` | **NEW** - Scalar fallbacks + build tag |
| `hwy/contrib/transform_avx2.go` | **NEW** - AVX2 implementations |
| `hwy/contrib/transform_test.go` | **NEW** - Benchmarks |

## Build & Test

```bash
# Build with SIMD
GOEXPERIMENT=simd go build ./...

# Run benchmarks
GOEXPERIMENT=simd go test -bench=Transform -benchmem ./hwy/contrib/...

# Compare to stdlib
GOEXPERIMENT=simd go test -bench=. -benchmem ./hwy/contrib/...
```

## Future Extensions

1. ~~**Generic Transform**: `Transform[T](input, output []T, fn func(archsimd.Vec) archsimd.Vec)`~~ ✅ **IMPLEMENTED** as `Transform32`/`Transform64`
2. **Buffer Pool**: Reusable buffers for repeated transforms
3. **AVX512**: Same pattern with Float32x16/Float64x8

## Implemented: Generic Transform API

The generic Transform functions are now available:

```go
// Type definitions
type VecFunc32 func(archsimd.Float32x8) archsimd.Float32x8
type VecFunc64 func(archsimd.Float64x4) archsimd.Float64x4
type ScalarFunc32 func(float32) float32
type ScalarFunc64 func(float64) float64

// Generic functions
func Transform32(input, output []float32, simd VecFunc32, scalar ScalarFunc32)
func Transform64(input, output []float64, simd VecFunc64, scalar ScalarFunc64)
```

Usage example:
```go
// Apply x² + x to all elements using SIMD
Transform32(input, output,
    func(x archsimd.Float32x8) archsimd.Float32x8 {
        return x.Mul(x).Add(x)
    },
    func(x float32) float32 { return x*x + x },
)
```

The named transforms (ExpTransform, etc.) are now built on top of Transform32/Transform64.
