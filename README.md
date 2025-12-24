# go-highway

A portable SIMD abstraction library for Go, inspired by Google's [Highway](https://github.com/google/highway) C++ library.

Write SIMD code once, run it on AVX2, AVX-512, or pure Go fallback.

## Requirements

- Go 1.26+ (currently requires `go1.26rc1`)
- `GOEXPERIMENT=simd` for hardware acceleration

## Installation

```bash
go get github.com/go-highway/highway
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/go-highway/highway/hwy"
    "github.com/go-highway/highway/hwy/contrib"
)

func main() {
    // Load data into SIMD vectors
    data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
    v := hwy.Load(data)

    // Vectorized operations
    doubled := hwy.Mul(v, hwy.Set[float32](2.0))
    sum := hwy.ReduceSum(doubled)

    fmt.Printf("Sum of doubled: %v\n", sum)

    // Transcendental functions (exp, log, sin, cos, etc.)
    exp := contrib.Exp(v)
    fmt.Printf("Exp: %v\n", exp.Data())
}
```

Build and run:

```bash
GOEXPERIMENT=simd go run main.go
```

## Features

### Core Operations (`hwy` package)

| Category | Operations |
|----------|------------|
| Load/Store | `Load`, `Store`, `Set`, `Zero`, `MaskLoad`, `MaskStore` |
| Arithmetic | `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Abs`, `Min`, `Max` |
| Math | `Sqrt`, `FMA` |
| Reduction | `ReduceSum`, `ReduceMin`, `ReduceMax` |
| Comparison | `Equal`, `LessThan`, `GreaterThan` |
| Conditional | `IfThenElse` |

### Extended Math (`hwy/contrib` package)

| Function | Description |
|----------|-------------|
| `Exp` | Exponential (e^x) |
| `Log` | Natural logarithm |
| `Sin`, `Cos`, `SinCos` | Trigonometric functions |
| `Tanh` | Hyperbolic tangent |
| `Sigmoid` | Logistic function 1/(1+e^-x) |
| `Erf` | Error function |

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
GOEXPERIMENT=simd go test -bench=. -benchmem ./hwy/contrib/...
```

## Supported Architectures

| Architecture | SIMD Width | Status |
|--------------|------------|--------|
| AMD64 AVX2 | 256-bit | Supported |
| AMD64 AVX-512 | 512-bit | Planned |
| Pure Go | Scalar | Supported (fallback) |

## License

Apache 2.0
