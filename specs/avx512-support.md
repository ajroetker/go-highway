# AVX512 Support

**Status: ✅ Complete** (2025-12-26)

Add AVX512 implementations for math functions in `hwy/contrib`.

## Prerequisites

- [x] Complete AVX2 implementations (see avx2-math-functions.md)
- [x] Verify `RoundToEvenScaled(0)` works as `RoundToEven()` replacement (see API Gap below)

AVX512 uses same algorithms as AVX2, just wider vectors:
- Float32: 8 lanes (AVX2) → 16 lanes (AVX512)
- Float64: 4 lanes (AVX2) → 8 lanes (AVX512)

## API Gap: RoundToEven (Resolved)

The `simd/archsimd` package has an API difference between AVX2 and AVX512 types:

| Method | Float32x8 (AVX2) | Float32x16 (AVX512) |
|--------|------------------|---------------------|
| `RoundToEven()` | ✅ Available | ❌ Not available |
| `RoundToEvenScaled(prec)` | ✅ Available | ✅ Available |

**Workaround**: Use `RoundToEvenScaled(0)` for integer rounding.

**Verification**: Per [Intel VRNDSCALEPS docs](https://www.felixcloutier.com/x86/vrndscaleps),
when `prec=0` (M=0 in imm8[7:4]), the formula becomes:
- `DEST = 2^(-0) × Round(2^0 × SRC) = Round(SRC)`

This is equivalent to VROUNDPS (round-to-nearest-even), confirming the workaround is valid.

## New Files

| File | Purpose |
|------|---------|
| `hwy/contrib/transform_avx512.go` | VecFunc32x16, VecFunc64x8, Transform32x16, Transform64x8 |
| `hwy/contrib/exp_avx512.go` | Exp_AVX512_F32x16, Exp_AVX512_F64x8 |
| `hwy/contrib/log_avx512.go` | Log_AVX512_F32x16, Log_AVX512_F64x8 |
| `hwy/contrib/trig_avx512.go` | Sin/Cos/SinCos for F32x16 and F64x8 |
| `hwy/contrib/special_avx512.go` | Tanh/Sigmoid/Erf for F32x16 and F64x8 |

## Modified Files

| File | Changes |
|------|---------|
| `hwy/contrib/transform_avx2.go` | Update named transforms to dispatch AVX512 → AVX2 |
| `cmd/hwygen/targets.go` | Add contrib ops to AVX512Target OpMap |

## Functions (16 total)

| Function | Notes |
|----------|-------|
| Exp_AVX512_F32x16 | Port from Exp_AVX2_F32x8 |
| Exp_AVX512_F64x8 | Port from Exp_AVX2_F64x4 |
| Log_AVX512_F32x16 | Port from Log_AVX2_F32x8 |
| Log_AVX512_F64x8 | Port from Log_AVX2_F64x4 |
| Sin_AVX512_F32x16 | Port from Sin_AVX2_F32x8 |
| Sin_AVX512_F64x8 | Port from Sin_AVX2_F64x4 |
| Cos_AVX512_F32x16 | Port from Cos_AVX2_F32x8 |
| Cos_AVX512_F64x8 | Port from Cos_AVX2_F64x4 |
| SinCos_AVX512_F32x16 | Port from SinCos_AVX2_F32x8 |
| SinCos_AVX512_F64x8 | Port from SinCos_AVX2_F64x4 |
| Tanh_AVX512_F32x16 | Port from Tanh_AVX2_F32x8 |
| Tanh_AVX512_F64x8 | Port from Tanh_AVX2_F64x4 |
| Sigmoid_AVX512_F32x16 | Port from Sigmoid_AVX2_F32x8 |
| Sigmoid_AVX512_F64x8 | Port from Sigmoid_AVX2_F64x4 |
| Erf_AVX512_F32x16 | Port from Erf_AVX2_F32x8 |
| Erf_AVX512_F64x8 | Port from Erf_AVX2_F64x4 |

## Implementation Phases

### Phase 1: Transform Infrastructure

Create `transform_avx512.go`:

```go
//go:build amd64 && goexperiment.simd

package contrib

import "simd/archsimd"

// Function types for AVX512 Transform operations.
type (
    VecFunc32x16 func(archsimd.Float32x16) archsimd.Float32x16
    VecFunc64x8  func(archsimd.Float64x8) archsimd.Float64x8
)

// Transform32x16 applies a SIMD operation using AVX512 (16 float32s at a time).
func Transform32x16(input, output []float32, simd VecFunc32x16, scalar ScalarFunc32) {
    n := min(len(input), len(output))

    // Process 16 float32s at a time
    for i := 0; i+16 <= n; i += 16 {
        x := archsimd.LoadFloat32x16Slice(input[i:])
        simd(x).StoreSlice(output[i:])
    }

    // Scalar tail (0-15 elements)
    for i := (n / 16) * 16; i < n; i++ {
        output[i] = scalar(input[i])
    }
}

// Transform64x8 applies a SIMD operation using AVX512 (8 float64s at a time).
func Transform64x8(input, output []float64, simd VecFunc64x8, scalar ScalarFunc64) {
    n := min(len(input), len(output))

    // Process 8 float64s at a time
    for i := 0; i+8 <= n; i += 8 {
        x := archsimd.LoadFloat64x8Slice(input[i:])
        simd(x).StoreSlice(output[i:])
    }

    // Scalar tail (0-7 elements)
    for i := (n / 8) * 8; i < n; i++ {
        output[i] = scalar(input[i])
    }
}
```

Update `transform_avx2.go` named transforms:

```go
func ExpTransform(input, output []float32) {
    if hwy.CurrentLevel() >= hwy.DispatchAVX512 {
        Transform32x16(input, output, Exp_AVX512_F32x16, exp32Scalar)
    } else {
        Transform32(input, output, Exp_AVX2_F32x8, exp32Scalar)
    }
}
```

### Phase 2: Math Functions

Port each AVX2 file to AVX512 with these substitutions:

| AVX2 | AVX512 |
|------|--------|
| `Float32x8` | `Float32x16` |
| `Float64x4` | `Float64x8` |
| `Int32x8` | `Int32x16` |
| `Int64x4` | `Int64x8` |
| `BroadcastFloat32x8` | `BroadcastFloat32x16` |
| `LoadFloat32x8Slice` | `LoadFloat32x16Slice` |
| `RoundToEven()` | `RoundToEvenScaled(0)` |

### Phase 3: hwygen Integration

Update `cmd/hwygen/targets.go` AVX512Target OpMap:

```go
OpMap: map[string]OpInfo{
    // ... existing ops ...

    // Contrib math functions
    "Exp":     {Package: "contrib", Name: "Exp", IsMethod: false},
    "Log":     {Package: "contrib", Name: "Log", IsMethod: false},
    "Sin":     {Package: "contrib", Name: "Sin", IsMethod: false},
    "Cos":     {Package: "contrib", Name: "Cos", IsMethod: false},
    "Tanh":    {Package: "contrib", Name: "Tanh", IsMethod: false},
    "Sigmoid": {Package: "contrib", Name: "Sigmoid", IsMethod: false},
    "Erf":     {Package: "contrib", Name: "Erf", IsMethod: false},
},
```

### Phase 4: Testing

Create `avx512_test.go`:

```go
//go:build amd64 && goexperiment.simd

func TestExp_AVX512_F32x16(t *testing.T) {
    if hwy.CurrentLevel() < hwy.DispatchAVX512 {
        t.Skip("AVX-512 not available")
    }
    // Same test vectors as AVX2
}

func BenchmarkExpTransform_AVX512(b *testing.B) {
    if hwy.CurrentLevel() < hwy.DispatchAVX512 {
        b.Skip("AVX-512 not available")
    }
    // Benchmark with 1024+ elements
}
```

## Build Tags

All AVX512 files use the same build tag as AVX2 (runtime dispatch):

```go
//go:build amd64 && goexperiment.simd
```

## Build & Test

```bash
# Build
GOEXPERIMENT=simd go build ./hwy/contrib/...

# Test (skips AVX512 tests if CPU doesn't support)
GOEXPERIMENT=simd go test -v ./hwy/contrib/...

# Benchmarks
GOEXPERIMENT=simd go test -bench=. -benchmem ./hwy/contrib/...
```

## File Summary

| File | Action | Contents |
|------|--------|----------|
| `transform_avx512.go` | **NEW** | VecFunc32x16, VecFunc64x8, Transform32x16, Transform64x8 |
| `exp_avx512.go` | **NEW** | Exp_AVX512_F32x16, Exp_AVX512_F64x8, constants |
| `log_avx512.go` | **NEW** | Log_AVX512_F32x16, Log_AVX512_F64x8, constants |
| `trig_avx512.go` | **NEW** | Sin/Cos/SinCos for F32x16, F64x8, constants |
| `special_avx512.go` | **NEW** | Tanh/Sigmoid/Erf for F32x16, F64x8, constants |
| `avx512_test.go` | **NEW** | Tests and benchmarks |
| `transform_avx2.go` | **MODIFY** | Add AVX512 dispatch to named transforms |
| `targets.go` | **MODIFY** | Add contrib ops to AVX512Target |
