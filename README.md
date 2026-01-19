# go-highway

[![Go](https://github.com/ajroetker/go-highway/actions/workflows/go.yml/badge.svg)](https://github.com/ajroetker/go-highway/actions/workflows/go.yml)

A portable SIMD abstraction library for Go, inspired by Google's [Highway](https://github.com/google/highway) C++ library.

Write SIMD code once, run it on AVX2, AVX-512, ARM NEON, or pure Go fallback.

## Requirements

- Go 1.26+ (currently requires `go1.26rc2`)
- `GOEXPERIMENT=simd` for AMD64 hardware acceleration (not needed for ARM64)

## Installation

```bash
go get github.com/ajroetker/go-highway
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/ajroetker/go-highway/hwy"
    "github.com/ajroetker/go-highway/hwy/contrib/algo"
)

func main() {
    // Load data into SIMD vectors
    data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
    v := hwy.Load(data)

    // Vectorized operations
    doubled := hwy.Mul(v, hwy.Set[float32](2.0))
    sum := hwy.ReduceSum(doubled)

    fmt.Printf("Sum of doubled: %v\n", sum)

    // Transcendental functions using transforms
    output := make([]float32, len(data))
    algo.ExpTransform(data, output)
    fmt.Printf("Exp: %v\n", output)
}
```

Build and run:

```bash
GOEXPERIMENT=simd go run main.go
```

## Features

### Core Operations (`hwy` package)

These are fundamental SIMD operations that map directly to hardware instructions:

| Category | Operations |
|----------|------------|
| Load/Store | `Load`, `Store`, `Set`, `Zero`, `MaskLoad`, `MaskStore` |
| Arithmetic | `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Abs`, `Min`, `Max` |
| Math | `Sqrt`, `FMA` |
| Reduction | `ReduceSum`, `ReduceMin`, `ReduceMax` |
| Comparison | `Equal`, `LessThan`, `GreaterThan` |
| Conditional | `IfThenElse` |

Low-level SIMD functions for direct archsimd usage:
- `Sqrt_AVX2_F32x8`, `Sqrt_AVX2_F64x4` - Hardware sqrt (VSQRTPS/VSQRTPD)
- `Sqrt_AVX512_F32x16`, `Sqrt_AVX512_F64x8` - AVX-512 variants

### Extended Math (`hwy/contrib/algo` and `hwy/contrib/math` packages)

The contrib package is organized into two subpackages:

**Algorithm Transforms** (`hwy/contrib/algo`):
| Function | Description |
|----------|-------------|
| `ExpTransform`, `ExpTransform64` | Apply exp(x) to slices |
| `LogTransform`, `LogTransform64` | Apply ln(x) to slices |
| `SinTransform`, `SinTransform64` | Apply sin(x) to slices |
| `CosTransform`, `CosTransform64` | Apply cos(x) to slices |
| `TanhTransform`, `TanhTransform64` | Apply tanh(x) to slices |
| `SigmoidTransform`, `SigmoidTransform64` | Apply 1/(1+e^-x) to slices |
| `ErfTransform`, `ErfTransform64` | Apply erf(x) to slices |
| `Transform32`, `Transform64` | Generic transforms with custom functions |

**Low-Level Math** (`hwy/contrib/math`):
| Function | Description |
|----------|-------------|
| `Exp_AVX2_F32x8`, `Exp_AVX2_F64x4` | Exponential on SIMD vectors |
| `Log_AVX2_F32x8`, `Log_AVX2_F64x4` | Natural logarithm on SIMD vectors |
| `Sin_AVX2_F32x8`, `Cos_AVX2_F32x8` | Trigonometric functions on SIMD vectors |
| `Tanh_AVX2_F32x8` | Hyperbolic tangent on SIMD vectors |
| `Sigmoid_AVX2_F32x8` | Logistic function on SIMD vectors |
| `Erf_AVX2_F32x8` | Error function on SIMD vectors |

All functions support `float32` and `float64` with ~4 ULP accuracy.

## Code Generator (hwygen)

Generate optimized target-specific code from generic implementations:

```bash
go build -o bin/hwygen ./cmd/hwygen
./bin/hwygen -input mycode.go -output . -targets avx2,avx512,neon,fallback
```

### Generic Dispatch

hwygen generates type-safe generic functions that automatically dispatch to the best implementation:

```go
// Write once with generics
func BaseSoftmax[T hwy.Floats](input, output []T) {
    // ... implementation using hwy.Load, hwy.Store, etc.
}

// hwygen generates:
// - BaseSoftmax_avx2, BaseSoftmax_avx2_Float64
// - BaseSoftmax_avx512, BaseSoftmax_avx512_Float64
// - BaseSoftmax_neon, BaseSoftmax_neon_Float64
// - BaseSoftmax_fallback, BaseSoftmax_fallback_Float64

// Plus a generic dispatcher:
func Softmax[T hwy.Floats](input, output []T)  // dispatches by type

// And type-specific function variables:
var SoftmaxFloat32 func(input, output []float32)
var SoftmaxFloat64 func(input, output []float64)

// Tail handling is automatic - remaining elements that don't
// fit a full SIMD width are processed via the fallback path.
```

Usage:

```go
// Generic API - works with any float type
data32 := []float32{1, 2, 3, 4}
out32 := make([]float32, 4)
softmax.Softmax(data32, out32)

data64 := []float64{1, 2, 3, 4}
out64 := make([]float64, 4)
softmax.Softmax(data64, out64)
```

See `examples/gelu` and `examples/softmax` for complete examples.

### Bulk Assembly Mode (ARM64 NEON)

For maximum performance on ARM64, hwygen can generate bulk assembly that processes entire arrays in a single call, eliminating per-vector function call overhead.

**Requirements:**
- [GoAT](https://github.com/gorse-io/goat) - C to Go assembly transpiler (tracked as a tool dependency in go.mod)

```bash
# Install tool dependencies (GoAT is declared in go.mod)
go install tool

# Build hwygen
go build -o bin/hwygen ./cmd/hwygen
```

**Generate bulk assembly:**

```bash
./bin/hwygen -bulk -input examples/gelu/gelu.go -output examples/gelu -targets neon -pkg gelu
```

This generates:
- Assembly files (`.s`) with bulk NEON implementations
- Go wrapper functions (`GELUBulkF32`, `GELUBulkF64`, etc.)

**Performance comparison** (1024 elements on Apple M4 Max):

| Function | Per-Vector | Bulk Assembly | Speedup |
|----------|------------|---------------|---------|
| GELU F32 | 67,581 ns | 577 ns | **117x** |
| GELU F64 | 122,690 ns | 1,793 ns | **68x** |

Bulk mode works best for pure element-wise operations like GELU. Functions with reduction operations (like softmax) are not suitable.

## Building

```bash
# With SIMD acceleration
GOEXPERIMENT=simd go build ./...

# Fallback only (pure Go)
go build ./...

# Run tests
GOEXPERIMENT=simd go test ./...

# Force fallback path (for testing)
HWY_NO_SIMD=1 GOEXPERIMENT=simd go test ./...

# Benchmarks
GOEXPERIMENT=simd go test -bench=. -benchmem ./hwy/contrib/algo/...
GOEXPERIMENT=simd go test -bench=. -benchmem ./hwy/contrib/math/...
```

## Supported Architectures

| Architecture | SIMD Width | Status |
|--------------|------------|--------|
| AMD64 AVX2 | 256-bit | Supported |
| AMD64 AVX-512 | 512-bit | Supported |
| ARM64 NEON | 128-bit | Supported |
| Pure Go | Scalar | Supported (fallback) |

## License

Apache 2.0
