# Float16 and BFloat16 Support

**Status: Planned**

Add IEEE 754 half-precision (float16) and Google Brain float (bfloat16) support to go-highway for ML inference and training workloads.

## Motivation

Half-precision floating-point formats are essential for modern machine learning:

- **float16 (IEEE 754)**: Used in inference, offers 2x memory bandwidth vs float32
- **bfloat16 (Brain Float)**: Used in training, same exponent range as float32 for stability

Both formats pack 2x more elements per SIMD register, enabling significant throughput improvements.

## Format Specifications

### IEEE 754 Float16

```
Sign (1 bit) | Exponent (5 bits) | Mantissa (10 bits)
    S        |     EEEEE         |    MMMMMMMMMM
```

| Property | Value |
|----------|-------|
| Total bits | 16 |
| Exponent bits | 5 |
| Mantissa bits | 10 |
| Exponent bias | 15 |
| Max value | 65504 |
| Min positive normal | 6.10e-5 |
| Precision | ~3.3 decimal digits |

### BFloat16 (Brain Float)

```
Sign (1 bit) | Exponent (8 bits) | Mantissa (7 bits)
    S        |     EEEEEEEE      |    MMMMMMM
```

| Property | Value |
|----------|-------|
| Total bits | 16 |
| Exponent bits | 8 (same as float32) |
| Mantissa bits | 7 |
| Exponent bias | 127 (same as float32) |
| Max value | ~3.4e38 (same as float32) |
| Min positive normal | ~1.2e-38 (same as float32) |
| Precision | ~2.4 decimal digits |

**Key insight**: BFloat16 is simply float32 with the lower 16 mantissa bits truncated. This makes conversions trivial.

---

## Hardware Support

### x86-64

| Extension | Float16 | BFloat16 | Availability |
|-----------|---------|----------|--------------|
| F16C | Convert only | - | Intel Haswell+ (2013), AMD Excavator+ (2015) |
| AVX-512 BF16 | - | Convert + dot product | Intel Cooper Lake (2020), AMD Zen4+ |
| AVX-512 FP16 | Full arithmetic | - | Intel Sapphire Rapids (2023), AMD Zen5+ |
| AMX | - | Matrix ops | Intel Sapphire Rapids (2023) |

### ARM

| Extension | Float16 | BFloat16 | Availability |
|-----------|---------|----------|--------------|
| ARMv8.2-A FP16 | Full arithmetic | - | Apple A11+, Cortex-A75+ |
| ARMv8.6-A BF16 | - | Convert + dot product | Apple M2+, Cortex-A710+ |
| NEON | Via FP16 extension | Via software | All ARMv8-A |
| SVE/SVE2 | Full arithmetic | Full arithmetic | AWS Graviton3+, Fujitsu A64FX |

### Key Instructions

**Float16 Conversions (F16C)**:
- `VCVTPH2PS`: float16 -> float32 (4 or 8 elements)
- `VCVTPS2PH`: float32 -> float16 (4 or 8 elements, with rounding mode)

**BFloat16 Conversions (AVX-512 BF16)**:
- `VCVTNE2PS2BF16`: Two float32 vectors -> one bfloat16 vector
- `VCVTNEPS2BF16`: float32 -> bfloat16 (round to nearest even)

**Float16 Arithmetic (AVX-512 FP16)**:
- `VADDPH`, `VSUBPH`, `VMULPH`, `VDIVPH`: Basic arithmetic
- `VFMADD132PH`, etc.: Fused multiply-add variants
- `VSQRTPH`, `VRCPPH`, `VRSQRTPH`: Math functions
- `VCMPPH`: Comparisons

---

## Type System Design

### New Types

Since Go lacks native float16/bfloat16 types, we define wrapper types:

```go
// hwy/float16.go

// Float16 represents an IEEE 754 half-precision floating-point number.
// It wraps uint16 for storage but provides float semantics.
type Float16 uint16

// BFloat16 represents a Brain Float 16 number.
// It has the same exponent range as float32 but reduced precision.
type BFloat16 uint16

// Constants
const (
    Float16MaxValue   Float16 = 0x7BFF // 65504
    Float16MinNormal  Float16 = 0x0400 // 2^-14
    Float16Inf        Float16 = 0x7C00
    Float16NegInf     Float16 = 0xFC00
    Float16NaN        Float16 = 0x7E00 // Quiet NaN

    BFloat16MaxValue  BFloat16 = 0x7F7F // ~3.39e38
    BFloat16MinNormal BFloat16 = 0x0080 // ~1.18e-38
    BFloat16Inf       BFloat16 = 0x7F80
    BFloat16NegInf    BFloat16 = 0xFF80
    BFloat16NaN       BFloat16 = 0x7FC0 // Quiet NaN
)
```

### Updated Type Constraints

```go
// hwy/types.go

// Float16Types is a constraint for half-precision float types.
type Float16Types interface {
    Float16 | BFloat16
}

// Floats is a constraint for all floating-point types.
type Floats interface {
    Float16 | BFloat16 | ~float32 | ~float64
}

// FloatsNative is a constraint for Go-native float types (for arithmetic).
type FloatsNative interface {
    ~float32 | ~float64
}

// Lanes is updated to include all types
type Lanes interface {
    Floats | Integers
}
```

**Design decision**: Float16 and BFloat16 are in the `Floats` constraint but NOT in a hypothetical `FloatsNative` constraint. This allows type-safe vectors while preventing direct arithmetic (which must go through promote-compute-demote).

---

## Conversion Functions

### Scalar Conversions (Pure Go)

```go
// hwy/float16.go

// Float16ToFloat32 converts a single Float16 to float32.
func Float16ToFloat32(h Float16) float32 {
    bits := uint32(h)
    sign := bits >> 15
    exp := (bits >> 10) & 0x1F
    mant := bits & 0x3FF

    if exp == 0 {
        if mant == 0 {
            // Zero (positive or negative)
            return math.Float32frombits(sign << 31)
        }
        // Denormalized: convert to normalized float32
        exp = 1
        for mant&0x400 == 0 {
            mant <<= 1
            exp--
        }
        mant &= 0x3FF
        exp += 127 - 15
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            return math.Float32frombits((sign << 31) | 0x7F800000)
        }
        return math.Float32frombits((sign << 31) | 0x7FC00000 | (mant << 13))
    } else {
        // Normalized: rebias exponent
        exp += 127 - 15
    }

    return math.Float32frombits((sign << 31) | (exp << 23) | (mant << 13))
}

// Float32ToFloat16 converts a float32 to Float16 with round-to-nearest-even.
func Float32ToFloat16(f float32) Float16 {
    bits := math.Float32bits(f)
    sign := (bits >> 16) & 0x8000
    exp := int((bits >> 23) & 0xFF) - 127 + 15
    mant := bits & 0x7FFFFF

    if exp <= 0 {
        if exp < -10 {
            // Underflow to zero
            return Float16(sign)
        }
        // Denormalized
        mant = (mant | 0x800000) >> uint(1-exp)
        // Round to nearest even
        if mant&0x1000 != 0 && (mant&0x2FFF) != 0 {
            mant += 0x2000
        }
        return Float16(sign | (mant >> 13))
    } else if exp >= 31 {
        if exp == 128 && mant != 0 {
            // NaN
            return Float16(sign | 0x7E00 | (mant >> 13))
        }
        // Overflow to infinity
        return Float16(sign | 0x7C00)
    }

    // Round to nearest even
    if mant&0x1000 != 0 && (mant&0x2FFF) != 0 {
        mant += 0x2000
        if mant&0x800000 != 0 {
            mant = 0
            exp++
            if exp >= 31 {
                return Float16(sign | 0x7C00)
            }
        }
    }

    return Float16(sign | uint32(exp<<10) | (mant >> 13))
}
```

BFloat16 conversions are much simpler:

```go
// BFloat16ToFloat32 converts a single BFloat16 to float32.
// This is a simple bit shift since bfloat16 is truncated float32.
func BFloat16ToFloat32(b BFloat16) float32 {
    return math.Float32frombits(uint32(b) << 16)
}

// Float32ToBFloat16 converts a float32 to BFloat16.
// Uses round-to-nearest-even on the truncated bits.
func Float32ToBFloat16(f float32) BFloat16 {
    bits := math.Float32bits(f)

    // Handle special cases
    if bits&0x7FFFFFFF > 0x7F800000 {
        // NaN: preserve sign, set quiet NaN
        return BFloat16((bits >> 16) | 0x0040)
    }

    // Round to nearest even
    // Add 0x7FFF + (bit 16 of original) for rounding
    rounding := uint32(0x7FFF) + ((bits >> 16) & 1)
    bits += rounding

    return BFloat16(bits >> 16)
}
```

---

## Vector Operations

### Promotion/Demotion (Vector Level)

```go
// hwy/promote.go (additions)

// PromoteF16ToF32 widens Float16 to float32.
// Input: N Float16 lanes -> Output: N float32 lanes (in two vectors for AVX2).
func PromoteF16ToF32(v Vec[Float16]) Vec[float32] {
    result := make([]float32, len(v.data))
    for i := range v.data {
        result[i] = Float16ToFloat32(v.data[i])
    }
    return Vec[float32]{data: result}
}

// PromoteLowerF16ToF32 promotes lower half of Float16 lanes to float32.
func PromoteLowerF16ToF32(v Vec[Float16]) Vec[float32] {
    n := len(v.data) / 2
    result := make([]float32, n)
    for i := 0; i < n; i++ {
        result[i] = Float16ToFloat32(v.data[i])
    }
    return Vec[float32]{data: result}
}

// PromoteUpperF16ToF32 promotes upper half of Float16 lanes to float32.
func PromoteUpperF16ToF32(v Vec[Float16]) Vec[float32] {
    half := len(v.data) / 2
    n := len(v.data) - half
    result := make([]float32, n)
    for i := 0; i < n; i++ {
        result[i] = Float16ToFloat32(v.data[half+i])
    }
    return Vec[float32]{data: result}
}

// DemoteF32ToF16 narrows float32 to Float16.
func DemoteF32ToF16(v Vec[float32]) Vec[Float16] {
    result := make([]Float16, len(v.data))
    for i := range v.data {
        result[i] = Float32ToFloat16(v.data[i])
    }
    return Vec[Float16]{data: result}
}

// DemoteTwoF32ToF16 demotes two float32 vectors to one Float16 vector.
// Input: 2 vectors of N float32 -> Output: 1 vector of 2N Float16.
func DemoteTwoF32ToF16(lo, hi Vec[float32]) Vec[Float16] {
    n := len(lo.data) + len(hi.data)
    result := make([]Float16, n)
    for i := 0; i < len(lo.data); i++ {
        result[i] = Float32ToFloat16(lo.data[i])
    }
    for i := 0; i < len(hi.data); i++ {
        result[len(lo.data)+i] = Float32ToFloat16(hi.data[i])
    }
    return Vec[Float16]{data: result}
}

// Similar functions for BFloat16...
// PromoteBF16ToF32, PromoteLowerBF16ToF32, PromoteUpperBF16ToF32
// DemoteF32ToBF16, DemoteTwoF32ToBF16
```

### Arithmetic via Promote-Compute-Demote

For platforms without native float16 arithmetic, operations follow the pattern:

```go
// hwy/ops_float16.go

// AddF16 adds two Float16 vectors.
// On platforms without native FP16, promotes to float32, computes, demotes.
func AddF16(a, b Vec[Float16]) Vec[Float16] {
    af32 := PromoteF16ToF32(a)
    bf32 := PromoteF16ToF32(b)
    return DemoteF32ToF16(Add(af32, bf32))
}

// MulF16 multiplies two Float16 vectors.
func MulF16(a, b Vec[Float16]) Vec[Float16] {
    af32 := PromoteF16ToF32(a)
    bf32 := PromoteF16ToF32(b)
    return DemoteF32ToF16(Mul(af32, bf32))
}

// FMAF16 performs fused multiply-add: a*b + c
func FMAF16(a, b, c Vec[Float16]) Vec[Float16] {
    af32 := PromoteF16ToF32(a)
    bf32 := PromoteF16ToF32(b)
    cf32 := PromoteF16ToF32(c)
    return DemoteF32ToF16(FMA(af32, bf32, cf32))
}

// DotF16 computes dot product of two Float16 vectors, returning float32.
// This is the common ML pattern: accumulate in higher precision.
func DotF16(a, b Vec[Float16]) float32 {
    af32 := PromoteF16ToF32(a)
    bf32 := PromoteF16ToF32(b)
    return ReduceSum(Mul(af32, bf32))
}
```

---

## SIMD Implementations

### GoAT C Source for ARM NEON Float16

```c
// hwy/c/ops_f16_neon_arm64.c
// Float16 operations for ARM64 with FP16 extension (ARMv8.2-A+)
// Compile with: -march=armv8.2-a+fp16

#include <arm_neon.h>

// ============================================================================
// Float16 Conversions
// ============================================================================

// Promote float16 to float32: result[i] = (float32)a[i]
void promote_f16_to_f32_neon(unsigned short *a, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 float16 -> 8 float32 at a time
    for (; i + 7 < n; i += 8) {
        float16x8_t h = vld1q_f16((float16_t*)(a + i));
        // Split into two float32x4
        float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
        float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
        vst1q_f32(result + i, lo);
        vst1q_f32(result + i + 4, hi);
    }

    // Process 4 at a time
    for (; i + 3 < n; i += 4) {
        float16x4_t h = vld1_f16((float16_t*)(a + i));
        float32x4_t f = vcvt_f32_f16(h);
        vst1q_f32(result + i, f);
    }

    // Scalar remainder (manual conversion)
    for (; i < n; i++) {
        // Inline float16 to float32 conversion
        unsigned int bits = a[i];
        unsigned int sign = (bits >> 15) & 1;
        unsigned int exp = (bits >> 10) & 0x1F;
        unsigned int mant = bits & 0x3FF;

        unsigned int f32_bits;
        if (exp == 0) {
            if (mant == 0) {
                f32_bits = sign << 31;
            }
            // Denormal handling simplified - treat as zero for now
            f32_bits = sign << 31;
        }
        if (exp == 31) {
            f32_bits = (sign << 31) | 0x7F800000 | (mant << 13);
        }
        if (exp != 0 && exp != 31) {
            f32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
        // Store via pointer cast
        *(unsigned int*)(result + i) = f32_bits;
    }
}

// Demote float32 to float16: result[i] = (float16)a[i]
void demote_f32_to_f16_neon(float *a, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 float32 -> 8 float16 at a time
    for (; i + 7 < n; i += 8) {
        float32x4_t lo = vld1q_f32(a + i);
        float32x4_t hi = vld1q_f32(a + i + 4);
        float16x4_t h_lo = vcvt_f16_f32(lo);
        float16x4_t h_hi = vcvt_f16_f32(hi);
        float16x8_t h = vcombine_f16(h_lo, h_hi);
        vst1q_f16((float16_t*)(result + i), h);
    }

    // Process 4 at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t f = vld1q_f32(a + i);
        float16x4_t h = vcvt_f16_f32(f);
        vst1_f16((float16_t*)(result + i), h);
    }

    // Scalar remainder
    for (; i < n; i++) {
        // Simplified scalar conversion
        unsigned int bits = *(unsigned int*)(a + i);
        unsigned int sign = (bits >> 16) & 0x8000;
        int exp = ((bits >> 23) & 0xFF) - 127 + 15;
        unsigned int mant = bits & 0x7FFFFF;

        unsigned short h;
        if (exp <= 0) {
            h = sign; // Underflow to zero
        }
        if (exp >= 31) {
            h = sign | 0x7C00; // Overflow to infinity
        }
        if (exp > 0 && exp < 31) {
            h = sign | (exp << 10) | (mant >> 13);
        }
        result[i] = h;
    }
}

// ============================================================================
// Float16 Arithmetic (Native - requires ARMv8.2-A+fp16)
// ============================================================================

// Vector addition: result[i] = a[i] + b[i]
void add_f16_neon(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 32 float16 at a time (4 vectors)
    for (; i + 31 < n; i += 32) {
        float16x8_t a0 = vld1q_f16((float16_t*)(a + i));
        float16x8_t a1 = vld1q_f16((float16_t*)(a + i + 8));
        float16x8_t a2 = vld1q_f16((float16_t*)(a + i + 16));
        float16x8_t a3 = vld1q_f16((float16_t*)(a + i + 24));

        float16x8_t b0 = vld1q_f16((float16_t*)(b + i));
        float16x8_t b1 = vld1q_f16((float16_t*)(b + i + 8));
        float16x8_t b2 = vld1q_f16((float16_t*)(b + i + 16));
        float16x8_t b3 = vld1q_f16((float16_t*)(b + i + 24));

        vst1q_f16((float16_t*)(result + i), vaddq_f16(a0, b0));
        vst1q_f16((float16_t*)(result + i + 8), vaddq_f16(a1, b1));
        vst1q_f16((float16_t*)(result + i + 16), vaddq_f16(a2, b2));
        vst1q_f16((float16_t*)(result + i + 24), vaddq_f16(a3, b3));
    }

    // Process 8 at a time
    for (; i + 7 < n; i += 8) {
        float16x8_t av = vld1q_f16((float16_t*)(a + i));
        float16x8_t bv = vld1q_f16((float16_t*)(b + i));
        vst1q_f16((float16_t*)(result + i), vaddq_f16(av, bv));
    }

    // Scalar remainder via promote-compute-demote
    for (; i < n; i++) {
        float16x4_t av = vld1_dup_f16((float16_t*)(a + i));
        float16x4_t bv = vld1_dup_f16((float16_t*)(b + i));
        float16x4_t rv = vadd_f16(av, bv);
        vst1_lane_f16((float16_t*)(result + i), rv, 0);
    }
}

// FMA: result[i] = a[i] * b[i] + c[i]
void fma_f16_neon(unsigned short *a, unsigned short *b, unsigned short *c,
                  unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 7 < n; i += 8) {
        float16x8_t av = vld1q_f16((float16_t*)(a + i));
        float16x8_t bv = vld1q_f16((float16_t*)(b + i));
        float16x8_t cv = vld1q_f16((float16_t*)(c + i));
        // vfmaq_f16: c + a*b
        vst1q_f16((float16_t*)(result + i), vfmaq_f16(cv, av, bv));
    }

    for (; i < n; i++) {
        float16x4_t av = vld1_dup_f16((float16_t*)(a + i));
        float16x4_t bv = vld1_dup_f16((float16_t*)(b + i));
        float16x4_t cv = vld1_dup_f16((float16_t*)(c + i));
        float16x4_t rv = vfma_f16(cv, av, bv);
        vst1_lane_f16((float16_t*)(result + i), rv, 0);
    }
}
```

### GoAT C Source for x86 F16C

```c
// hwy/c/ops_f16_x86.c
// Float16 conversions for x86 with F16C extension
// Compile with: -mf16c -mavx2

#include <immintrin.h>

// Promote float16 to float32 using F16C
void promote_f16_to_f32_f16c(unsigned short *a, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 float16 -> 8 float32 at a time (AVX2)
    for (; i + 7 < n; i += 8) {
        __m128i h = _mm_loadu_si128((__m128i*)(a + i));
        __m256 f = _mm256_cvtph_ps(h);  // VCVTPH2PS
        _mm256_storeu_ps(result + i, f);
    }

    // Process 4 at a time (SSE)
    for (; i + 3 < n; i += 4) {
        __m128i h = _mm_loadl_epi64((__m128i*)(a + i));
        __m128 f = _mm_cvtph_ps(h);  // VCVTPH2PS
        _mm_storeu_ps(result + i, f);
    }

    // Scalar remainder (inline conversion)
    for (; i < n; i++) {
        unsigned int bits = a[i];
        unsigned int sign = (bits >> 15) & 1;
        unsigned int exp = (bits >> 10) & 0x1F;
        unsigned int mant = bits & 0x3FF;

        unsigned int f32_bits = 0;
        if (exp == 0 && mant == 0) {
            f32_bits = sign << 31;
        }
        if (exp == 0 && mant != 0) {
            f32_bits = sign << 31; // Simplified: denormal -> zero
        }
        if (exp == 31) {
            f32_bits = (sign << 31) | 0x7F800000 | (mant << 13);
        }
        if (exp > 0 && exp < 31) {
            f32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
        *(unsigned int*)(result + i) = f32_bits;
    }
}

// Demote float32 to float16 using F16C
void demote_f32_to_f16_f16c(float *a, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 float32 -> 8 float16 at a time (AVX2)
    // imm8 = 0 for round-to-nearest-even
    for (; i + 7 < n; i += 8) {
        __m256 f = _mm256_loadu_ps(a + i);
        __m128i h = _mm256_cvtps_ph(f, 0);  // VCVTPS2PH, round-to-nearest
        _mm_storeu_si128((__m128i*)(result + i), h);
    }

    // Process 4 at a time (SSE)
    for (; i + 3 < n; i += 4) {
        __m128 f = _mm_loadu_ps(a + i);
        __m128i h = _mm_cvtps_ph(f, 0);  // VCVTPS2PH
        _mm_storel_epi64((__m128i*)(result + i), h);
    }

    // Scalar remainder
    for (; i < n; i++) {
        unsigned int bits = *(unsigned int*)(a + i);
        unsigned int sign = (bits >> 16) & 0x8000;
        int exp = ((bits >> 23) & 0xFF) - 127 + 15;
        unsigned int mant = bits & 0x7FFFFF;

        unsigned short h = 0;
        if (exp <= 0) {
            h = sign;
        }
        if (exp >= 31) {
            h = sign | 0x7C00;
        }
        if (exp > 0 && exp < 31) {
            h = sign | (exp << 10) | (mant >> 13);
        }
        result[i] = h;
    }
}
```

### Go Wrappers for GoAT-Generated Assembly

```go
// hwy/asm/f16_neon_wrappers.go
//go:build arm64

package asm

import "unsafe"

//go:generate go tool goat ../c/ops_f16_neon_arm64.c -O3 -e="--target=arm64 -march=armv8.2-a+fp16"

// PromoteF16ToF32NEON converts float16 to float32 using NEON.
func PromoteF16ToF32NEON(a []uint16, result []float32) {
    if len(a) == 0 {
        return
    }
    n := int64(min(len(a), len(result)))
    promote_f16_to_f32_neon(
        unsafe.Pointer(&a[0]),
        unsafe.Pointer(&result[0]),
        unsafe.Pointer(&n),
    )
}

// DemoteF32ToF16NEON converts float32 to float16 using NEON.
func DemoteF32ToF16NEON(a []float32, result []uint16) {
    if len(a) == 0 {
        return
    }
    n := int64(min(len(a), len(result)))
    demote_f32_to_f16_neon(
        unsafe.Pointer(&a[0]),
        unsafe.Pointer(&result[0]),
        unsafe.Pointer(&n),
    )
}

// AddF16NEON adds two float16 vectors using native NEON FP16.
func AddF16NEON(a, b, result []uint16) {
    if len(a) == 0 {
        return
    }
    n := int64(min(len(a), min(len(b), len(result))))
    add_f16_neon(
        unsafe.Pointer(&a[0]),
        unsafe.Pointer(&b[0]),
        unsafe.Pointer(&result[0]),
        unsafe.Pointer(&n),
    )
}

// Assembly function declarations (generated by GoAT)
//go:noescape
func promote_f16_to_f32_neon(a, result, len unsafe.Pointer)

//go:noescape
func demote_f32_to_f16_neon(a, result, len unsafe.Pointer)

//go:noescape
func add_f16_neon(a, b, result, len unsafe.Pointer)

//go:noescape
func fma_f16_neon(a, b, c, result, len unsafe.Pointer)
```

### GoAT x86 F16C Conversions

x86 F16C provides conversion-only instructions (no native F16 arithmetic). Arithmetic uses promote→compute→demote.

```c
// hwy/c/ops_f16_x86.c
// Compile with: -mf16c -mavx2

#include <immintrin.h>

// Promote float16 to float32 using F16C (VCVTPH2PS)
void promote_f16_to_f32_f16c(unsigned short *a, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 float16 -> 8 float32 at a time (AVX2)
    for (; i + 7 < n; i += 8) {
        __m128i h = _mm_loadu_si128((__m128i*)(a + i));
        __m256 f = _mm256_cvtph_ps(h);  // VCVTPH2PS ymm, xmm
        _mm256_storeu_ps(result + i, f);
    }

    // Process 4 at a time (SSE)
    for (; i + 3 < n; i += 4) {
        __m128i h = _mm_loadl_epi64((__m128i*)(a + i));
        __m128 f = _mm_cvtph_ps(h);
        _mm_storeu_ps(result + i, f);
    }

    // Scalar remainder (inline bit conversion)
    for (; i < n; i++) {
        unsigned int bits = a[i];
        unsigned int sign = (bits >> 15) & 1;
        unsigned int exp = (bits >> 10) & 0x1F;
        unsigned int mant = bits & 0x3FF;

        unsigned int f32_bits = 0;
        if (exp == 0 && mant == 0) { f32_bits = sign << 31; }
        if (exp == 31) { f32_bits = (sign << 31) | 0x7F800000 | (mant << 13); }
        if (exp > 0 && exp < 31) { f32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13); }
        *(unsigned int*)(result + i) = f32_bits;
    }
}

// Demote float32 to float16 using F16C (VCVTPS2PH)
void demote_f32_to_f16_f16c(float *a, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 8 float32 -> 8 float16 at a time (AVX2)
    // imm8 = 0 for round-to-nearest-even
    for (; i + 7 < n; i += 8) {
        __m256 f = _mm256_loadu_ps(a + i);
        __m128i h = _mm256_cvtps_ph(f, 0);  // VCVTPS2PH xmm, ymm, imm8
        _mm_storeu_si128((__m128i*)(result + i), h);
    }

    // Process 4 at a time (SSE)
    for (; i + 3 < n; i += 4) {
        __m128 f = _mm_loadu_ps(a + i);
        __m128i h = _mm_cvtps_ph(f, 0);
        _mm_storel_epi64((__m128i*)(result + i), h);
    }

    // Scalar remainder
    for (; i < n; i++) {
        // ... bit manipulation for F32→F16
    }
}
```

### GoAT AVX-512 FP16 (Native Arithmetic)

AVX-512 FP16 (Sapphire Rapids+) provides native float16 arithmetic with 32 lanes per vector.

```c
// hwy/c/ops_f16_avx512.c
// Compile with: -mavx512fp16

#include <immintrin.h>

// Native Float16 addition: result[i] = a[i] + b[i]
void add_f16_avx512(unsigned short *a, unsigned short *b, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 128 float16 at a time (4 vectors of 32 lanes)
    for (; i + 127 < n; i += 128) {
        __m512h a0 = _mm512_loadu_ph(a + i);
        __m512h a1 = _mm512_loadu_ph(a + i + 32);
        __m512h a2 = _mm512_loadu_ph(a + i + 64);
        __m512h a3 = _mm512_loadu_ph(a + i + 96);

        __m512h b0 = _mm512_loadu_ph(b + i);
        __m512h b1 = _mm512_loadu_ph(b + i + 32);
        __m512h b2 = _mm512_loadu_ph(b + i + 64);
        __m512h b3 = _mm512_loadu_ph(b + i + 96);

        _mm512_storeu_ph(result + i, _mm512_add_ph(a0, b0));        // VADDPH
        _mm512_storeu_ph(result + i + 32, _mm512_add_ph(a1, b1));
        _mm512_storeu_ph(result + i + 64, _mm512_add_ph(a2, b2));
        _mm512_storeu_ph(result + i + 96, _mm512_add_ph(a3, b3));
    }

    // Process 32 at a time
    for (; i + 31 < n; i += 32) {
        __m512h av = _mm512_loadu_ph(a + i);
        __m512h bv = _mm512_loadu_ph(b + i);
        _mm512_storeu_ph(result + i, _mm512_add_ph(av, bv));
    }

    // Scalar remainder via promote-compute-demote
    for (; i < n; i++) {
        // Convert F16→F32, add, convert F32→F16
    }
}

// Native Float16 FMA: result[i] = a[i] * b[i] + c[i]
void fma_f16_avx512(unsigned short *a, unsigned short *b, unsigned short *c,
                    unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    for (; i + 31 < n; i += 32) {
        __m512h av = _mm512_loadu_ph(a + i);
        __m512h bv = _mm512_loadu_ph(b + i);
        __m512h cv = _mm512_loadu_ph(c + i);
        // VFMADD132PH: result = a * b + c
        _mm512_storeu_ph(result + i, _mm512_fmadd_ph(av, bv, cv));
    }

    // Scalar remainder
    for (; i < n; i++) { /* ... */ }
}
```

### GoAT AVX-512 BF16 (Conversions + Dot Product)

AVX-512 BF16 (Cooper Lake+) provides BF16 conversions and dot product with F32 accumulator.

```c
// hwy/c/ops_bf16_avx512.c
// Compile with: -mavx512bf16

#include <immintrin.h>

// Demote float32 to bfloat16: VCVTNEPS2BF16
void demote_f32_to_bf16_avx512(float *a, unsigned short *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 float32 -> 16 bfloat16 at a time
    for (; i + 15 < n; i += 16) {
        __m512 f = _mm512_loadu_ps(a + i);
        __m256bh h = _mm512_cvtneps_pbh(f);  // VCVTNEPS2BF16 ymm, zmm
        _mm256_storeu_si256((__m256i*)(result + i), (__m256i)h);
    }

    // Scalar remainder
    for (; i < n; i++) {
        unsigned int bits = *(unsigned int*)(a + i);
        // Round to nearest even and truncate
        unsigned int rounding = 0x7FFF + ((bits >> 16) & 1);
        bits += rounding;
        result[i] = (unsigned short)(bits >> 16);
    }
}

// BF16 dot product with F32 accumulator: VDPBF16PS
// acc = acc + sum(a[i] * b[i]) for i in [0, 31]
void dot_bf16_avx512(unsigned short *a, unsigned short *b, float *acc, long *len) {
    long n = *len;
    long i = 0;

    __m512 sum = _mm512_setzero_ps();

    // Process 32 bf16 pairs at a time (16 F32 accumulator lanes)
    for (; i + 31 < n; i += 32) {
        __m512bh av = (__m512bh)_mm512_loadu_si512(a + i);
        __m512bh bv = (__m512bh)_mm512_loadu_si512(b + i);
        sum = _mm512_dpbf16_ps(sum, av, bv);  // VDPBF16PS
    }

    // Reduce and store
    *acc += _mm512_reduce_add_ps(sum);

    // Scalar remainder (promote to F32, multiply, accumulate)
    for (; i < n; i++) {
        // ...
    }
}
```

---

## CPU Feature Detection

Update `dispatch_amd64_simd.go`:

```go
// hwy/dispatch_amd64_simd.go

// Feature flags for half-precision support
var (
    hasF16C       bool  // F16C: float16 <-> float32 conversions
    hasAVX512FP16 bool  // AVX-512 FP16: native float16 arithmetic
    hasAVX512BF16 bool  // AVX-512 BF16: bfloat16 dot products
)

func init() {
    // Existing detection...

    // Half-precision features
    hasF16C = archsimd.X86.F16C()

    if archsimd.X86.AVX512() {
        hasAVX512FP16 = archsimd.X86.AVX512FP16()
        hasAVX512BF16 = archsimd.X86.AVX512BF16()
    }
}

// HasF16C returns true if F16C (float16 conversions) is available.
func HasF16C() bool {
    return hasF16C
}

// HasAVX512FP16 returns true if native float16 arithmetic is available.
func HasAVX512FP16() bool {
    return hasAVX512FP16
}

// HasAVX512BF16 returns true if bfloat16 dot products are available.
func HasAVX512BF16() bool {
    return hasAVX512BF16
}
```

---

## Lane Counts

With 16-bit types, lane counts double compared to float32:

| SIMD Width | float32 Lanes | Float16/BFloat16 Lanes |
|------------|---------------|------------------------|
| SSE (128-bit) | 4 | 8 |
| AVX2 (256-bit) | 8 | 16 |
| AVX-512 (512-bit) | 16 | 32 |
| NEON (128-bit) | 4 | 8 |
| SVE (256-bit) | 8 | 16 |

The existing `MaxLanes[T]()` function will automatically return correct values since it uses `unsafe.Sizeof(dummy)` which returns 2 for Float16/BFloat16.

---

## Implementation Phases

### Phase 1: Type Definitions and Scalar Conversions

**Files to create:**
- `hwy/float16.go` - Float16 type, constants, scalar conversions
- `hwy/bfloat16.go` - BFloat16 type, constants, scalar conversions

**Files to modify:**
- `hwy/types.go` - Add Float16Types constraint, update Floats constraint

**Deliverables:**
- Float16 and BFloat16 types with constants
- Scalar conversion functions (Go-native, no SIMD)
- Unit tests for edge cases (NaN, Inf, denormals, rounding)

**Test cases:**
```go
func TestFloat16ToFloat32(t *testing.T) {
    tests := []struct {
        input    Float16
        expected float32
    }{
        {0x0000, 0.0},           // Positive zero
        {0x8000, -0.0},          // Negative zero (bit pattern)
        {0x3C00, 1.0},           // One
        {0xBC00, -1.0},          // Negative one
        {0x7BFF, 65504.0},       // Max normal
        {0x0400, 6.103515625e-5},// Min normal
        {0x0001, 5.96046e-8},    // Min denormal
        {0x7C00, inf},           // Positive infinity
        {0xFC00, -inf},          // Negative infinity
        {0x7E00, NaN},           // Quiet NaN
    }
    // ...
}
```

### Phase 2: Vector Conversions (Scalar Fallback)

**Files to create:**
- `hwy/promote_f16.go` - Vector promotion/demotion for Float16
- `hwy/promote_bf16.go` - Vector promotion/demotion for BFloat16

**Deliverables:**
- `PromoteF16ToF32`, `PromoteLowerF16ToF32`, `PromoteUpperF16ToF32`
- `DemoteF32ToF16`, `DemoteTwoF32ToF16`
- Same for BFloat16
- Cross-format: `ConvertF16ToBF16`, `ConvertBF16ToF16`

### Phase 3: Promote-Compute-Demote Arithmetic

**Files to create:**
- `hwy/ops_f16.go` - Float16 arithmetic (Add, Sub, Mul, Div, FMA, etc.)
- `hwy/ops_bf16.go` - BFloat16 arithmetic

**Deliverables:**
- Basic arithmetic: Add, Sub, Mul, Div, Neg, Abs
- FMA: FMAF16, FMABF16
- Comparisons: Equal, LessThan, etc. for Float16/BFloat16
- Reductions: SumF16, MinF16, MaxF16

**Note:** Scalar fallback uses promote-to-float32 -> compute -> demote pattern.

### Phase 4: GoAT NEON Float16 (ARM64)

**Files to create:**
- `hwy/c/ops_f16_neon_arm64.c` - NEON FP16 operations
- `hwy/asm/f16_neon_wrappers.go` - Go wrappers

**Generated assembly:**
- `hwy/asm/ops_f16_neon_arm64.s` - Generated by GoAT

**Compile flags:** `-march=armv8.2-a+fp16`

**Deliverables:**
- `promote_f16_to_f32_neon` - F16→F32 conversion using `vcvt_f32_f16`
- `demote_f32_to_f16_neon` - F32→F16 conversion using `vcvt_f16_f32`
- `add_f16_neon`, `mul_f16_neon`, `fma_f16_neon` - Native FP16 arithmetic

**Loop structure (matching existing patterns):**
```c
// Process 32 float16 at a time (4 vectors of 8 lanes)
for (; i + 31 < n; i += 32) { ... }
// Process 8 float16 at a time (1 vector)
for (; i + 7 < n; i += 8) { ... }
// Scalar remainder (promote-compute-demote)
for (; i < n; i++) { ... }
```

### Phase 5: GoAT x86 F16C Conversions

**Files to create:**
- `hwy/c/ops_f16_x86.c` - x86 F16C conversion operations
- `hwy/asm/f16_x86_wrappers.go` - Go wrappers

**Generated assembly:**
- `hwy/asm/ops_f16_x86.s` - Generated by GoAT

**Compile flags:** `-mf16c -mavx2`

**Deliverables:**
- `promote_f16_to_f32_f16c` using `_mm256_cvtph_ps` (VCVTPH2PS)
- `demote_f32_to_f16_f16c` using `_mm256_cvtps_ph` (VCVTPS2PH)
- 8-16x speedup over scalar for conversions

**Note:** x86 F16C only provides conversions. Arithmetic uses promote→float32 compute→demote pattern.

### Phase 6: GoAT AVX-512 FP16 Native Arithmetic

**Files to create:**
- `hwy/c/ops_f16_avx512.c` - AVX-512 FP16 native operations
- `hwy/asm/f16_avx512_wrappers.go` - Go wrappers

**Generated assembly:**
- `hwy/asm/ops_f16_avx512.s` - Generated by GoAT

**Compile flags:** `-mavx512fp16`

**Deliverables:**
- `add_f16_avx512`, `mul_f16_avx512`, `div_f16_avx512` using VADDPH, VMULPH, VDIVPH
- `fma_f16_avx512` using VFMADD132PH
- Native conversions with AVX-512 widths (32 lanes)
- Eliminate promote-compute-demote overhead on Sapphire Rapids+

### Phase 7: GoAT AVX-512 BF16 Support

**Files to create:**
- `hwy/c/ops_bf16_avx512.c` - AVX-512 BF16 operations
- `hwy/asm/bf16_avx512_wrappers.go` - Go wrappers

**Generated assembly:**
- `hwy/asm/ops_bf16_avx512.s` - Generated by GoAT

**Compile flags:** `-mavx512bf16`

**Deliverables:**
- `demote_f32_to_bf16_avx512` using VCVTNEPS2BF16
- `demote_two_f32_to_bf16_avx512` using VCVTNE2PS2BF16
- `dot_bf16_avx512` using VDPBF16PS for ML dot product accumulation

### Phase 8: GoAT NEON BFloat16 (ARM64)

**Files to create:**
- `hwy/c/ops_bf16_neon_arm64.c` - NEON BF16 operations
- `hwy/asm/bf16_neon_wrappers.go` - Go wrappers

**Generated assembly:**
- `hwy/asm/ops_bf16_neon_arm64.s` - Generated by GoAT

**Compile flags:** `-march=armv8.6-a+bf16`

**Deliverables:**
- BF16↔F32 conversions using `vcvt_bf16_f32`, `vcvt_f32_bf16`
- `vbfdot` for BF16 dot product (Apple M2+, Cortex-A710+)

### Phase 9: hwygen Integration

**Files to modify:**
- `cmd/hwygen/targets.go` - Add Float16/BFloat16 to code generator

**Deliverables:**
- Support for `hwy.Float16` and `hwy.BFloat16` in generated code
- Automatic dispatch selection based on CPU features
- Proper lane count handling (F16 has 2x lanes vs F32)

### Phase 10: Math Functions (contrib)

**Files to create:**
- `hwy/c/math_f16_neon_arm64.c` - F16 math via GoAT
- `hwy/contrib/math/exp_f16.go` - Exp/Log for Float16
- `hwy/contrib/math/trig_f16.go` - Sin/Cos/Tan for Float16
- `hwy/contrib/math/special_f16.go` - Tanh/Sigmoid for Float16 (ML critical)

**GoAT Constraint:** No `__builtin_expf`, `__builtin_sqrtf`, etc.

**Approach:**
- Use polynomial approximations (Chebyshev, minimax, or Remez)
- For scalar/AVX2: Promote to float32, use existing contrib math, demote
- For native F16: Implement reduced-precision polynomial coefficients

**Example (Exp approximation for F16):**
```c
// Uses float32 compute with F16 I/O
void exp_f16_neon(unsigned short *a, unsigned short *result, long *len) {
    // Promote to F32, apply polynomial exp, demote to F16
    // Polynomial coefficients are constants (no __builtin_expf)
}
```

---

## File Summary

### New Files - Pure Go (Scalar Fallback)

| File | Contents | Build Tags |
|------|----------|------------|
| `hwy/float16.go` | Float16 type, constants, scalar F16↔F32 conversions | none |
| `hwy/bfloat16.go` | BFloat16 type, constants, scalar BF16↔F32 conversions | none |
| `hwy/promote_f16.go` | Scalar Float16 vector promotions/demotions | none |
| `hwy/promote_bf16.go` | Scalar BFloat16 vector promotions/demotions | none |
| `hwy/ops_f16.go` | Float16 arithmetic (promote-compute-demote) | none |
| `hwy/ops_bf16.go` | BFloat16 arithmetic (promote-compute-demote) | none |
| `hwy/float16_test.go` | Comprehensive F16 tests | none |
| `hwy/bfloat16_test.go` | Comprehensive BF16 tests | none |

### New Files - GoAT C Sources

| File | Contents | Compile Flags |
|------|----------|---------------|
| `hwy/c/ops_f16_neon_arm64.c` | NEON FP16 conversions + native arithmetic | `-march=armv8.2-a+fp16` |
| `hwy/c/ops_bf16_neon_arm64.c` | NEON BF16 conversions + dot product | `-march=armv8.6-a+bf16` |
| `hwy/c/ops_f16_x86.c` | x86 F16C conversions (AVX2) | `-mf16c -mavx2` |
| `hwy/c/ops_f16_avx512.c` | AVX-512 FP16 native arithmetic | `-mavx512fp16` |
| `hwy/c/ops_bf16_avx512.c` | AVX-512 BF16 conversions + dot product | `-mavx512bf16` |
| `hwy/c/math_f16_neon_arm64.c` | F16 math functions (polynomial approx) | `-march=armv8.2-a+fp16` |

### New Files - GoAT Generated Assembly

Generated via: `go tool goat hwy/c/<source>.c -O3 -e="<compile_flags>"`

| File | Generated From | Target |
|------|----------------|--------|
| `hwy/asm/ops_f16_neon_arm64.s` | `ops_f16_neon_arm64.c` | ARM64 |
| `hwy/asm/ops_bf16_neon_arm64.s` | `ops_bf16_neon_arm64.c` | ARM64 |
| `hwy/asm/ops_f16_x86.s` | `ops_f16_x86.c` | AMD64 |
| `hwy/asm/ops_f16_avx512.s` | `ops_f16_avx512.c` | AMD64 |
| `hwy/asm/ops_bf16_avx512.s` | `ops_bf16_avx512.c` | AMD64 |

### New Files - Go Wrappers for Assembly

| File | Contents | Build Tags |
|------|----------|------------|
| `hwy/asm/f16_neon_wrappers.go` | Go wrappers for NEON FP16 assembly | `arm64` |
| `hwy/asm/bf16_neon_wrappers.go` | Go wrappers for NEON BF16 assembly | `arm64` |
| `hwy/asm/f16_x86_wrappers.go` | Go wrappers for x86 F16C assembly | `amd64` |
| `hwy/asm/f16_avx512_wrappers.go` | Go wrappers for AVX-512 FP16 assembly | `amd64` |
| `hwy/asm/bf16_avx512_wrappers.go` | Go wrappers for AVX-512 BF16 assembly | `amd64` |

### New Files - Contrib Math (F16)

| File | Contents |
|------|----------|
| `hwy/contrib/math/exp_f16.go` | Exp/Log for Float16 (promote-compute-demote) |
| `hwy/contrib/math/trig_f16.go` | Sin/Cos/Tan for Float16 |
| `hwy/contrib/math/special_f16.go` | Tanh/Sigmoid for Float16 (ML critical) |

### Modified Files

| File | Changes |
|------|---------|
| `hwy/types.go` | Add Float16Types constraint, update Floats, update Lanes |
| `hwy/dispatch.go` | Add DispatchLevel comments for FP16/BF16 features |
| `hwy/dispatch_amd64_simd.go` | Add F16C, AVX512_FP16, AVX512_BF16 detection via archsimd |
| `hwy/dispatch_arm64.go` | Add FP16, BF16 extension detection |
| `cmd/hwygen/targets.go` | Add Float16/BFloat16 support, lane count handling |
| `specs/feature-gaps.md` | Update float16/bfloat16 status to in-progress/done |

---

## Testing Strategy

### Unit Tests

```go
// hwy/float16_test.go

func TestFloat16Conversions(t *testing.T) {
    // Round-trip: float32 -> float16 -> float32
    // Should preserve value within precision limits
}

func TestFloat16SpecialValues(t *testing.T) {
    // NaN, Inf, -Inf, zero, negative zero
}

func TestFloat16Denormals(t *testing.T) {
    // Denormalized numbers near zero
}

func TestFloat16Rounding(t *testing.T) {
    // Verify round-to-nearest-even behavior
}

func TestFloat16Saturation(t *testing.T) {
    // Values outside float16 range
}
```

### Accuracy Tests

```go
func TestFloat16ArithmeticAccuracy(t *testing.T) {
    // Compare float16 arithmetic against float32 reference
    // Verify ULP (units in last place) error bounds
}

func TestFloat16MathAccuracy(t *testing.T) {
    // Exp, Log, Sin, Cos, Tanh
    // Verify accuracy for ML-relevant ranges
}
```

### Benchmarks

```go
func BenchmarkFloat16Promote(b *testing.B) {
    // Measure F16->F32 conversion throughput
}

func BenchmarkFloat16Demote(b *testing.B) {
    // Measure F32->F16 conversion throughput
}

func BenchmarkFloat16FMA(b *testing.B) {
    // Measure FMA throughput (key ML operation)
}

func BenchmarkFloat16VsFloat32(b *testing.B) {
    // Compare F16 promote-compute-demote vs native F32
}
```

---

## Dependencies

### Go Version

- Go 1.22+: Generics support
- Go 1.26+ (goexperiment.simd): Required for archsimd CPU feature detection

### GoAT (Go Assembly Transpiler)

**GoAT is the primary mechanism for Float16/BFloat16 SIMD support.** Go 1.26's `simd/archsimd` package does not include Float16, BFloat16, or unsigned integer vector types. We use GoAT to fill this gap by transpiling C code with intrinsics to Go assembly.

This is the established pattern in go-highway (see `hwy/c/ops_neon_arm64.c` → `hwy/asm/ops_neon_arm64.s`).

**Advantages:**
- Leverage clang optimizations and auto-vectorization
- Use native intrinsics (`arm_neon.h`, `immintrin.h`) for F16/BF16 support
- Maintainable C source with generated assembly
- No CGO runtime overhead
- Consistent with existing go-highway patterns

**GoAT Constraints for Float16/BFloat16:**

| Constraint | Impact | Workaround |
|------------|--------|------------|
| No `else` clauses | Low | Use multiple `if` statements |
| No `__builtin_*` functions | High for math | Use polynomial approximations for exp/log/sin/cos |
| No `static inline` helpers | Low | Inline all code directly |
| C functions must return `void` | Low | Pass results via pointer parameters |
| Scalar remainder may become `memset` | Medium | Use SIMD store for scalar loop or add complexity |
| Double constants cause rodata panic | Low (F16 uses F32) | Use `volatile` + hex bit patterns for F64 |

**Float16/BFloat16 operations are straightforward with GoAT** - conversions and arithmetic don't require any forbidden constructs. Only math functions (exp, log, trigonometry) need polynomial implementations.

### archsimd Package (Feature Detection Only)

Go 1.26's `simd/archsimd` provides CPU feature detection but **NOT** Float16/BFloat16 vector types:

| Feature | Available | Notes |
|---------|-----------|-------|
| `archsimd.X86.F16C()` | Yes | Detect F16C conversion support |
| `archsimd.X86.AVX512FP16()` | Yes | Detect native FP16 arithmetic |
| `archsimd.X86.AVX512BF16()` | Yes | Detect BF16 dot product support |
| `archsimd.Float16x*` types | **No** | Use GoAT-generated assembly instead |
| `archsimd.BFloat16x*` types | **No** | Use GoAT-generated assembly instead |

For Float16/BFloat16 vector operations, GoAT-generated assembly is required. We use archsimd only for runtime CPU feature detection to select appropriate implementations.

---

## Performance Expectations

### Conversion Throughput (elements/cycle)

| Method | Float16->Float32 | Float32->Float16 |
|--------|------------------|------------------|
| Scalar | ~0.5 | ~0.3 |
| AVX2/F16C | ~8 | ~8 |
| AVX-512/F16C | ~16 | ~16 |

### Arithmetic Throughput (elements/cycle)

| Method | Add/Mul | FMA |
|--------|---------|-----|
| Scalar (via F32) | ~0.25 | ~0.2 |
| AVX2 (via F32) | ~4 | ~4 |
| AVX-512 FP16 (native) | ~32 | ~32 |

### Memory Bandwidth

Float16 provides 2x elements per memory transfer:
- AVX2 load: 16 Float16 vs 8 float32
- AVX-512 load: 32 Float16 vs 16 float32

This is often the dominant benefit for memory-bound ML workloads.

---

## Open Questions

### Resolved

1. ~~**archsimd support**: What's the current state of F16C/AVX-512 FP16 support in the simd/archsimd package?~~

   **Answer:** Go 1.26's archsimd provides CPU feature detection (`X86.F16C()`, `X86.AVX512FP16()`, etc.) but does NOT include Float16/BFloat16 vector types. **Use GoAT** to generate SIMD assembly from C intrinsics.

2. ~~**Go assembly vs CGO**: If archsimd lacks support, should we write Go assembly or use CGO?~~

   **Answer:** Use **GoAT** (C-to-Go assembly transpiler). This is the established pattern in go-highway and provides:
   - Zero CGO overhead
   - Consistent with existing `hwy/c/*.c` → `hwy/asm/*.s` workflow
   - Leverages clang optimizations

### Still Open

3. **Rounding modes**: Should we expose rounding mode control for conversions, or always use round-to-nearest-even?

   *Recommendation:* Start with round-to-nearest-even only (IEEE 754 default). Add rounding mode parameter later if needed for ML training workloads.

4. **NaN propagation**: What's the correct behavior for NaN in arithmetic operations?

   *Recommendation:* Follow IEEE 754 semantics - NaN propagates through operations. Use quiet NaN (0x7E00 for F16) as the canonical NaN value.

5. **Mixed precision API**: Should operations like `MulF16F32(Vec[Float16], Vec[float32])` be supported?

   *Recommendation:* Defer to Phase 2. Initially require explicit promotion: `Mul(PromoteF16ToF32(a), b)`. Consider mixed-precision API later if use cases emerge.

6. **ARM64 FP16 detection**: Does `golang.org/x/sys/cpu` expose ARM FP16 capability flags?

   *Note:* Need to verify `cpu.ARM64.HasFP16` or equivalent. If not available, may need runtime probe via NEON instruction test.

7. **macOS NEON FP16 compatibility**: Do Apple M1+ processors support NEON FP16 (ARMv8.2-A+fp16)?

   *Note:* Apple A11+ and M1+ support FP16. Verify GoAT-generated code works on macOS (no SME streaming mode issues since FP16 uses regular NEON).

---

## References

- [IEEE 754-2008 Standard](https://ieeexplore.ieee.org/document/4610935)
- [Intel Intrinsics Guide - F16C](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=F16C)
- [Intel Intrinsics Guide - AVX-512 FP16](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX512_FP16)
- [BFloat16 Format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
- [Highway C++ Float16 Implementation](https://github.com/google/highway/blob/master/hwy/ops/generic_ops-inl.h)
