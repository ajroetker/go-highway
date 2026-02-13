# go-highway

[![Go](https://github.com/ajroetker/go-highway/actions/workflows/go.yml/badge.svg)](https://github.com/ajroetker/go-highway/actions/workflows/go.yml)

A portable SIMD abstraction library for Go, inspired by Google's [Highway](https://github.com/google/highway) C++ library.

Write SIMD code once, run it on AVX2, AVX-512, ARM NEON, or pure Go fallback.

## Requirements

- Go 1.26+
- `GOEXPERIMENT=simd` for AMD64 hardware acceleration (uses native `simd/archsimd` package)
- ARM64 uses `hwy/asm` with GoAT-generated assembly (no `GOEXPERIMENT` needed)

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
| Load/Store | `Load`, `LoadFull`, `Store`, `StoreFull`, `Set`, `Zero`, `MaskLoad`, `MaskStore`, `Load4` |
| Arithmetic | `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Abs`, `Min`, `Max`, `FMA`, `MulAdd` |
| Math | `Sqrt`, `RSqrt`, `RSqrtNewtonRaphson`, `RSqrtPrecise`, `Pow` |
| Reduction | `ReduceSum`, `ReduceMin`, `ReduceMax` |
| Comparison | `Equal`, `NotEqual`, `LessThan`, `LessEqual`, `GreaterThan`, `GreaterEqual` |
| Conditional | `IfThenElse`, `IfThenElseZero`, `IfThenZeroElse`, `ZeroIfNegative` |
| Bitwise | `And`, `Or`, `Xor`, `Not`, `AndNot`, `ShiftLeft`, `ShiftRight`, `PopCount` |
| Shuffle | `GetLane`, `Reverse`, `Broadcast`, `Iota` |
| Type Cast | `AsInt32`, `AsFloat32`, `AsInt64`, `AsFloat64` |
| Float Check | `IsNaN`, `IsInf`, `IsFinite`, `RoundToEven` |
| Utilities | `NumLanes`, `SignBit`, `Const`, `ConstValue` |

`LoadFull`/`StoreFull` provide bounds-check-free operations when you know the slice has sufficient capacity.

Low-level SIMD functions for direct archsimd usage:
- `Sqrt_AVX2_F32x8`, `Sqrt_AVX2_F64x4` - Hardware sqrt (VSQRTPS/VSQRTPD)
- `Sqrt_AVX512_F32x16`, `Sqrt_AVX512_F64x8` - AVX-512 variants
- `PopCount_AVX2_*`, `PopCount_AVX512_*` - Population count for bit manipulation

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

### Additional Contrib Packages

| Package | Description |
|---------|-------------|
| `hwy/contrib/matmul` | Matrix multiplication with SME/NEON acceleration |
| `hwy/contrib/matvec` | Matrix-vector multiplication |
| `hwy/contrib/rabitq` | RaBitQ SIMD operations for vector quantization (ANN search) |
| `hwy/contrib/activation` | Neural network activation functions |
| `hwy/contrib/nn` | Neural network primitives |
| `hwy/contrib/sort` | SIMD-accelerated sorting algorithms |
| `hwy/contrib/vec` | Vector distance and similarity functions |
| `hwy/contrib/bitpack` | Bit packing/unpacking operations |
| `hwy/contrib/varint` | Variable-length integer encoding |
| `hwy/contrib/image` | Image processing operations |

## Code Generator (hwygen)

Generate optimized target-specific code from generic implementations:

```bash
go build -o bin/hwygen ./cmd/hwygen
./bin/hwygen -input mycode.go -output . -targets avx2,avx512,neon,fallback
```

### Target Modes

Each target supports a generation mode suffix:

- **`neon`** (default GoSimd) — generates Go code calling `hwy/asm` package methods
- **`neon:asm`** — compiles the function to C, transpiles to Go assembly via GoAT, and generates `//go:noescape` wrappers. Use this for compute-heavy kernels where per-vector call overhead matters.
- **`neon:c`** — generates C source only (for inspection)

```bash
# GoSimd mode (default) — portable Go with asm package calls
./bin/hwygen -input dense.go -output . -targets avx2,avx512,neon,fallback

# Assembly mode — C → GoAT → bulk NEON assembly
./bin/hwygen -input matmul.go -output . -targets avx2,avx512,neon:asm,fallback
```

AVX2 and AVX-512 targets use Go 1.26's native `simd/archsimd` package directly. ARM64 targets (NEON, SVE) use the `hwy/asm` package because `simd/archsimd` does not yet support these architectures. The `:asm` mode is available for any target but is primarily useful for ARM64 where bulk assembly avoids per-vector function call overhead.

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

### Assembly Mode (`neon:asm`)

For maximum performance on ARM64, hwygen can generate bulk assembly via the `neon:asm` target that processes entire arrays in a single call, eliminating per-vector function call overhead.

**Requirements:**
- [GoAT](https://github.com/gorse-io/goat) - C to Go assembly transpiler (tracked as a tool dependency in go.mod)

```bash
# Install tool dependencies (GoAT is declared in go.mod)
go install tool

# Build hwygen
go build -o bin/hwygen ./cmd/hwygen
```

**Generate assembly for ARM64 NEON:**

```bash
# Using the :asm target suffix
./bin/hwygen -input matmul.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch matmul

# Or for bulk element-wise operations
./bin/hwygen -bulk -input examples/gelu/gelu.go -output examples/gelu -targets neon -pkg gelu
```

The `neon:asm` target generates:
- C source files (intermediate, kept with `-keepc`)
- Go assembly (`.s`) via GoAT transpilation
- `//go:noescape` wrapper functions for slice-to-pointer conversion
- Dispatch override files (`z_c_slices_*_neon_arm64.gen.go`) that wire assembly into the dispatch table

**Performance comparison** (1024 elements on Apple M4 Max):

| Function | Per-Vector (GoSimd) | Bulk Assembly (`neon:asm`) | Speedup |
|----------|---------------------|---------------------------|---------|
| GELU F32 | 67,581 ns | 577 ns | **117x** |
| GELU F64 | 122,690 ns | 1,793 ns | **68x** |

Assembly mode works best for compute-heavy kernels (matmul, cross-entropy loss, fused quantized ops) and pure element-wise operations. Functions with complex control flow or reduction operations (like softmax) are better suited to the default GoSimd mode.

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

# Disable SME dispatch on ARM64 (falls back to NEON)
HWY_NO_SME=1 go test ./...

# Disable SVE dispatch on ARM64 Linux (falls back to NEON)
HWY_NO_SVE=1 go test ./...

# Benchmarks
GOEXPERIMENT=simd go test -bench=. -benchmem ./hwy/contrib/algo/...
GOEXPERIMENT=simd go test -bench=. -benchmem ./hwy/contrib/math/...
```

## Supported Architectures

| Architecture | SIMD Width | Backend | Status |
|--------------|------------|---------|--------|
| AMD64 AVX2 | 256-bit | Go 1.26 `simd/archsimd` | Supported |
| AMD64 AVX-512 | 512-bit | Go 1.26 `simd/archsimd` | Supported |
| ARM64 NEON | 128-bit | `hwy/asm` (GoAT assembly) | Supported |
| ARM64 SVE (Darwin) | 512-bit fixed | `hwy/asm` (GoAT assembly) | Supported |
| ARM64 SVE (Linux) | Scalable | `hwy/asm` (GoAT assembly) | Supported |
| ARM64 SME | Scalable | `hwy/asm` (GoAT assembly) | Supported (matrix ops) |
| Pure Go | Scalar | — | Supported (fallback) |

AMD64 targets use Go 1.26's native `simd/archsimd` package. ARM64 targets use the `hwy/asm` package with GoAT-generated assembly because `simd/archsimd` does not yet support NEON or SVE. SME (Scalable Matrix Extension) provides dedicated matrix multiplication hardware on Apple Silicon M4 and newer ARM processors.

## License

Apache 2.0
