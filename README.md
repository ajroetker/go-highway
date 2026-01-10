# go-highway

[![Go](https://github.com/ajroetker/go-highway/actions/workflows/go.yml/badge.svg)](https://github.com/ajroetker/go-highway/actions/workflows/go.yml)

A portable SIMD abstraction library for Go, inspired by Google's [Highway](https://github.com/google/highway) C++ library.

Write SIMD code once, run it on AVX2, AVX-512, ARM NEON, or pure Go fallback.

## Requirements

- Go 1.26+ (currently requires `go1.26rc1`)
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
| `Horner`, `Horner5`, `Horner7`, `Horner13` | Polynomial evaluation utilities |

All functions support `float32` and `float64` with ~4 ULP accuracy.

## Code Generator (hwygen)

Generate optimized target-specific code from generic implementations:

```bash
go build -o bin/hwygen ./cmd/hwygen
./bin/hwygen -input mycode.go -target avx2 -output mycode_avx2.go
```

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
