# AVX2 Math Functions

Remaining AVX2 implementations needed in `hwy/contrib`.

## Status

**Completed**: `Exp_AVX2_F32x8`, `Exp_AVX2_F64x4`

**TODO** (14 functions using scalar fallback):

| Function | File | Notes |
|----------|------|-------|
| Log_AVX2_F32x8 | log_avx2.go | Range reduction + atanh polynomial |
| Log_AVX2_F64x4 | log_avx2.go | Higher-degree polynomial |
| Sin_AVX2_F32x8 | trig_avx2.go | Payne-Hanek + minimax poly |
| Sin_AVX2_F64x4 | trig_avx2.go | Higher-degree polynomial |
| Cos_AVX2_F32x8 | trig_avx2.go | Phase-shifted Sin |
| Cos_AVX2_F64x4 | trig_avx2.go | Phase-shifted Sin |
| SinCos_AVX2_F32x8 | trig_avx2.go | Combined (shares computation) |
| SinCos_AVX2_F64x4 | trig_avx2.go | Combined |
| Sigmoid_AVX2_F32x8 | special_avx2.go | Uses Exp |
| Sigmoid_AVX2_F64x4 | special_avx2.go | Uses Exp |
| Tanh_AVX2_F32x8 | special_avx2.go | Polynomial + Exp fallback |
| Tanh_AVX2_F64x4 | special_avx2.go | Uses Exp |
| Erf_AVX2_F32x8 | special_avx2.go | Abramowitz & Stegun |
| Erf_AVX2_F64x4 | special_avx2.go | Uses Exp |

## Algorithms

### Log (Natural Logarithm)
```
1. Range reduction: x = 2^e * m where 1 ≤ m < 2
2. Transform: y = (m-1)/(m+1) for atanh series
3. Polynomial: ln(m) = 2y(1 + y²/3 + y⁴/5 + ...)
4. Reconstruct: ln(x) = e*ln(2) + ln(m)
Edge cases: x<0→NaN, x=0→-Inf, x=+Inf→+Inf
```

### Sin/Cos (Trigonometric)
```
1. Range reduction: k = round(x * 4/π), r = x - k*π/4
2. Octant selection: k mod 4 determines sin/cos/neg
3. Polynomial: sin(r) odd, cos(r) even minimax
Edge cases: ±Inf→NaN, NaN→NaN, 0→0
```

### Sigmoid (Logistic)
```
sigmoid(x) = 1 / (1 + exp(-x))
For stability: clamp x to [-20, 20], use Exp_AVX2
```

### Tanh (Hyperbolic Tangent)
```
|x| < 0.625: polynomial x*(1 + t₁x² + t₂x⁴ + ...)
|x| < 20: 2*sigmoid(2x) - 1 (uses Exp)
|x| ≥ 20: sign(x)
```

### Erf (Error Function)
```
|x| < 0.5: Taylor series
0.5 ≤ |x| < 4: Abramowitz & Stegun 7.1.26
|x| ≥ 4: sign(x)
Uses Exp for medium range
```

## Polynomial Coefficients

### Log (float32)
```go
log32_c1 = 0.6666666666666735130  // 2/3
log32_c2 = 0.3999999999940941908  // 2/5
log32_c3 = 0.2857142874366239149  // 2/7
log32_c4 = 0.2222219843214978396  // 2/9
log32_c5 = 0.1818357216161805012  // 2/11
```

### Sin (float32)
```go
sin32_s1 = -0.16666666641626524   // -1/3!
sin32_s2 =  0.008333329385889463  // 1/5!
sin32_s3 = -0.00019839334836096632 // -1/7!
sin32_s4 =  2.718311493989822e-6  // 1/9!
```

### Cos (float32)
```go
cos32_c1 = -0.4999999963229337    // -1/2!
cos32_c2 =  0.04166662453689337   // 1/4!
cos32_c3 = -0.001388731625493765  // -1/6!
cos32_c4 =  2.443315711809948e-5  // 1/8!
```

### Erf (float32, Abramowitz & Stegun)
```go
erf32_p1 =  0.254829592
erf32_p2 = -0.284496736
erf32_p3 =  1.421413741
erf32_p4 = -1.453152027
erf32_p5 =  1.061405429
erf32_t  =  0.3275911
```

## Implementation Pattern

```go
// 1. Declare vectorized constants
var (
    log32_c1 = archsimd.BroadcastFloat32x8(0.666...)
    // ...
)

// 2. Register in init()
func init() {
    if hwy.CurrentLevel() >= hwy.DispatchAVX2 {
        Log32 = log32AVX2
    }
}

// 3. Wrapper function (hwy.Vec interface)
func log32AVX2(v hwy.Vec[float32]) hwy.Vec[float32] {
    // Process 8 elements, tail with scalar
}

// 4. Core SIMD function (for hwygen)
func Log_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
    // Range reduction
    // Polynomial (Horner's method with MulAdd)
    // Reconstruction
    // Edge case handling with Merge
}
```

## Testing

Each function needs:
1. Accuracy test: Compare to math.* within 4 ULP
2. Edge cases: 0, ±Inf, NaN, denormals
3. Range boundaries: Transition points
4. Vectorized: Mixed values in same vector
5. Cross-check: AVX2 vs fallback results

## Build Commands

```bash
# Build and test
GOEXPERIMENT=simd go test ./hwy/contrib/...

# Test fallback path
HWY_NO_SIMD=1 GOEXPERIMENT=simd go test ./hwy/contrib/...

# Benchmark
GOEXPERIMENT=simd go test -bench=. ./hwy/contrib/...
```
