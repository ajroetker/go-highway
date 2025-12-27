# go-highway/hwy

Portable SIMD operations for Go with runtime CPU dispatch, inspired by Google's [Highway](https://github.com/google/highway) C++ library.

## Installation

```go
import "github.com/ajroetker/go-highway/hwy"
```

Requires Go 1.26+ for generics and SIMD support.

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/ajroetker/go-highway/hwy"
)

func main() {
    a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
    b := []float32{10, 20, 30, 40, 50, 60, 70, 80}
    result := make([]float32, len(a))

    // Process with automatic tail handling
    hwy.ProcessWithTail[float32](len(a),
        func(offset int) {
            // Full vectors
            va := hwy.Load(a[offset:])
            vb := hwy.Load(b[offset:])
            vr := hwy.Add(va, vb)
            hwy.Store(vr, result[offset:])
        },
        func(offset, count int) {
            // Tail elements with mask
            mask := hwy.TailMask[float32](count)
            va := hwy.MaskLoad(mask, a[offset:])
            vb := hwy.MaskLoad(mask, b[offset:])
            vr := hwy.Add(va, vb)
            hwy.MaskStore(mask, vr, result[offset:])
        },
    )

    fmt.Println(result) // [11 22 33 44 55 66 77 88]
}
```

## API Reference

### Types

| Type | Description |
|------|-------------|
| `Vec[T Lanes]` | Portable vector handle |
| `Mask[T Lanes]` | Comparison result mask |
| `Lanes` | Constraint for SIMD-compatible types (floats and integers) |

### Memory Operations

| Function | Description |
|----------|-------------|
| `Load[T](src []T) Vec[T]` | Load vector from slice |
| `Store[T](v Vec[T], dst []T)` | Store vector to slice |
| `Set[T](value T) Vec[T]` | Broadcast value to all lanes |
| `Zero[T]() Vec[T]` | Zero vector |

### Arithmetic

| Function | Description |
|----------|-------------|
| `Add[T](a, b Vec[T]) Vec[T]` | Element-wise addition |
| `Sub[T](a, b Vec[T]) Vec[T]` | Element-wise subtraction |
| `Mul[T](a, b Vec[T]) Vec[T]` | Element-wise multiplication |
| `Div[T Floats](a, b Vec[T]) Vec[T]` | Element-wise division (floats) |
| `Neg[T](v Vec[T]) Vec[T]` | Negate all lanes |
| `Abs[T](v Vec[T]) Vec[T]` | Absolute value |
| `Min[T](a, b Vec[T]) Vec[T]` | Element-wise minimum |
| `Max[T](a, b Vec[T]) Vec[T]` | Element-wise maximum |

### Math (Floats)

| Function | Description |
|----------|-------------|
| `Sqrt[T Floats](v Vec[T]) Vec[T]` | Square root |
| `FMA[T Floats](a, b, c Vec[T]) Vec[T]` | Fused multiply-add: a*b + c |

### Reduction

| Function | Description |
|----------|-------------|
| `ReduceSum[T](v Vec[T]) T` | Sum all lanes |

### Comparison

| Function | Description |
|----------|-------------|
| `Equal[T](a, b Vec[T]) Mask[T]` | Element-wise equality |
| `LessThan[T](a, b Vec[T]) Mask[T]` | Element-wise less-than |
| `GreaterThan[T](a, b Vec[T]) Mask[T]` | Element-wise greater-than |

### Masked Operations

| Function | Description |
|----------|-------------|
| `IfThenElse[T](mask, a, b Vec[T]) Vec[T]` | Conditional selection |
| `MaskLoad[T](mask Mask[T], src []T) Vec[T]` | Load with mask |
| `MaskStore[T](mask Mask[T], v Vec[T], dst []T)` | Store with mask |

### Tail Handling

| Function | Description |
|----------|-------------|
| `TailMask[T](count int) Mask[T]` | Create mask for tail elements |
| `ProcessWithTail[T](size, fullFn, tailFn)` | Process array with tail handling |
| `MaxLanes[T]() int` | Get max lanes for type T |

### Runtime Dispatch

| Function | Description |
|----------|-------------|
| `CurrentLevel() DispatchLevel` | Get detected SIMD level |
| `CurrentWidth() int` | Get SIMD width in bytes |
| `CurrentName() string` | Get SIMD target name |

## Architecture Support

| Architecture | SIMD Levels |
|-------------|-------------|
| amd64 | Scalar, SSE2, AVX2, AVX-512 |
| arm64 | Scalar, NEON, SVE |
| other | Scalar |

## Environment Variables

- `HWY_NO_SIMD=1` - Force scalar fallback (useful for testing/debugging)

## Extended Math (contrib)

The `hwy/contrib` package is organized into subpackages:

### Algorithm Transforms (`hwy/contrib/algo`)

Apply operations to entire slices with zero allocation:

```go
import "github.com/ajroetker/go-highway/hwy/contrib/algo"

input := []float32{1.0, 2.0, 3.0, 4.0}
output := make([]float32, len(input))

// Named transforms
algo.ExpTransform(input, output)
algo.LogTransform(input, output)
algo.SinTransform(input, output)

// Generic transform with custom operation
algo.Transform32(input, output,
    func(x archsimd.Float32x8) archsimd.Float32x8 {
        return x.Mul(x)  // Square each element
    },
    func(x float32) float32 { return x * x },
)
```

### Low-Level Math (`hwy/contrib/math`)

Direct SIMD vector operations:

```go
import "github.com/ajroetker/go-highway/hwy/contrib/math"

// SIMD vector functions
expX := math.Exp_AVX2_F32x8(x)
logX := math.Log_AVX2_F32x8(x)
sinX, cosX := math.SinCos_AVX2_F32x8(x)

// Polynomial evaluation
coeffs := []float32{1.0, 2.0, 3.0}
result := math.Horner(x, coeffs)
```

See package documentation for complete API details.
