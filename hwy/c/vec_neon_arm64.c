// Per-vector NEON operations for ARM64
// Used with GoAT to generate Go assembly with typed vector wrappers
// These functions pass NEON vectors by value (register-resident operations)

#include <arm_neon.h>

// ============================================================================
// Float32x4 Operations (128-bit, 4 lanes)
// ============================================================================

float32x4_t add_f32x4(float32x4_t a, float32x4_t b) {
    return vaddq_f32(a, b);
}

float32x4_t sub_f32x4(float32x4_t a, float32x4_t b) {
    return vsubq_f32(a, b);
}

float32x4_t mul_f32x4(float32x4_t a, float32x4_t b) {
    return vmulq_f32(a, b);
}

float32x4_t div_f32x4(float32x4_t a, float32x4_t b) {
    return vdivq_f32(a, b);
}

// Fused multiply-add: a * b + c
float32x4_t fma_f32x4(float32x4_t a, float32x4_t b, float32x4_t c) {
    return vfmaq_f32(c, a, b);
}

// Fused multiply-subtract: a * b - c
float32x4_t fms_f32x4(float32x4_t a, float32x4_t b, float32x4_t c) {
    return vfmsq_f32(c, a, b);
}

float32x4_t min_f32x4(float32x4_t a, float32x4_t b) {
    return vminq_f32(a, b);
}

float32x4_t max_f32x4(float32x4_t a, float32x4_t b) {
    return vmaxq_f32(a, b);
}

float32x4_t abs_f32x4(float32x4_t a) {
    return vabsq_f32(a);
}

float32x4_t neg_f32x4(float32x4_t a) {
    return vnegq_f32(a);
}

float32x4_t sqrt_f32x4(float32x4_t a) {
    return vsqrtq_f32(a);
}

// Reciprocal estimate (1/x)
float32x4_t recip_f32x4(float32x4_t a) {
    return vrecpeq_f32(a);
}

// Reciprocal square root estimate (1/sqrt(x))
float32x4_t rsqrt_f32x4(float32x4_t a) {
    return vrsqrteq_f32(a);
}

// Horizontal sum - returns scalar
float hsum_f32x4(float32x4_t v) {
    return vaddvq_f32(v);
}

// Horizontal min - returns scalar
float hmin_f32x4(float32x4_t v) {
    return vminvq_f32(v);
}

// Horizontal max - returns scalar
float hmax_f32x4(float32x4_t v) {
    return vmaxvq_f32(v);
}

// Dot product
float dot_f32x4(float32x4_t a, float32x4_t b) {
    return vaddvq_f32(vmulq_f32(a, b));
}

// ============================================================================
// Float32x4 Comparison Operations (return int32x4_t mask)
// ============================================================================

int32x4_t eq_f32x4(float32x4_t a, float32x4_t b) {
    return vreinterpretq_s32_u32(vceqq_f32(a, b));
}

int32x4_t ne_f32x4(float32x4_t a, float32x4_t b) {
    return vmvnq_s32(vreinterpretq_s32_u32(vceqq_f32(a, b)));
}

int32x4_t lt_f32x4(float32x4_t a, float32x4_t b) {
    return vreinterpretq_s32_u32(vcltq_f32(a, b));
}

int32x4_t le_f32x4(float32x4_t a, float32x4_t b) {
    return vreinterpretq_s32_u32(vcleq_f32(a, b));
}

int32x4_t gt_f32x4(float32x4_t a, float32x4_t b) {
    return vreinterpretq_s32_u32(vcgtq_f32(a, b));
}

int32x4_t ge_f32x4(float32x4_t a, float32x4_t b) {
    return vreinterpretq_s32_u32(vcgeq_f32(a, b));
}

// ============================================================================
// Float32x4 Bitwise/Select Operations
// ============================================================================

// Bitwise AND (reinterpret as int)
float32x4_t and_f32x4(float32x4_t a, float32x4_t b) {
    return vreinterpretq_f32_s32(vandq_s32(
        vreinterpretq_s32_f32(a),
        vreinterpretq_s32_f32(b)));
}

// Bitwise OR
float32x4_t or_f32x4(float32x4_t a, float32x4_t b) {
    return vreinterpretq_f32_s32(vorrq_s32(
        vreinterpretq_s32_f32(a),
        vreinterpretq_s32_f32(b)));
}

// Bitwise XOR
float32x4_t xor_f32x4(float32x4_t a, float32x4_t b) {
    return vreinterpretq_f32_s32(veorq_s32(
        vreinterpretq_s32_f32(a),
        vreinterpretq_s32_f32(b)));
}

// Select: mask ? yes : no
float32x4_t sel_f32x4(int32x4_t mask, float32x4_t yes, float32x4_t no) {
    return vbslq_f32(vreinterpretq_u32_s32(mask), yes, no);
}

// ============================================================================
// Float64x2 Operations (128-bit, 2 lanes)
// ============================================================================

float64x2_t add_f64x2(float64x2_t a, float64x2_t b) {
    return vaddq_f64(a, b);
}

float64x2_t sub_f64x2(float64x2_t a, float64x2_t b) {
    return vsubq_f64(a, b);
}

float64x2_t mul_f64x2(float64x2_t a, float64x2_t b) {
    return vmulq_f64(a, b);
}

float64x2_t div_f64x2(float64x2_t a, float64x2_t b) {
    return vdivq_f64(a, b);
}

float64x2_t fma_f64x2(float64x2_t a, float64x2_t b, float64x2_t c) {
    return vfmaq_f64(c, a, b);
}

float64x2_t min_f64x2(float64x2_t a, float64x2_t b) {
    return vminq_f64(a, b);
}

float64x2_t max_f64x2(float64x2_t a, float64x2_t b) {
    return vmaxq_f64(a, b);
}

float64x2_t abs_f64x2(float64x2_t a) {
    return vabsq_f64(a);
}

float64x2_t neg_f64x2(float64x2_t a) {
    return vnegq_f64(a);
}

float64x2_t sqrt_f64x2(float64x2_t a) {
    return vsqrtq_f64(a);
}

double hsum_f64x2(float64x2_t v) {
    return vaddvq_f64(v);
}

double dot_f64x2(float64x2_t a, float64x2_t b) {
    return vaddvq_f64(vmulq_f64(a, b));
}

// ============================================================================
// Int32x4 Operations (128-bit, 4 lanes)
// ============================================================================

int32x4_t add_i32x4(int32x4_t a, int32x4_t b) {
    return vaddq_s32(a, b);
}

int32x4_t sub_i32x4(int32x4_t a, int32x4_t b) {
    return vsubq_s32(a, b);
}

int32x4_t mul_i32x4(int32x4_t a, int32x4_t b) {
    return vmulq_s32(a, b);
}

int32x4_t min_i32x4(int32x4_t a, int32x4_t b) {
    return vminq_s32(a, b);
}

int32x4_t max_i32x4(int32x4_t a, int32x4_t b) {
    return vmaxq_s32(a, b);
}

int32x4_t abs_i32x4(int32x4_t a) {
    return vabsq_s32(a);
}

int32x4_t neg_i32x4(int32x4_t a) {
    return vnegq_s32(a);
}

// Bitwise operations
int32x4_t and_i32x4(int32x4_t a, int32x4_t b) {
    return vandq_s32(a, b);
}

int32x4_t or_i32x4(int32x4_t a, int32x4_t b) {
    return vorrq_s32(a, b);
}

int32x4_t xor_i32x4(int32x4_t a, int32x4_t b) {
    return veorq_s32(a, b);
}

int32x4_t not_i32x4(int32x4_t a) {
    return vmvnq_s32(a);
}

int32x4_t andnot_i32x4(int32x4_t a, int32x4_t b) {
    return vbicq_s32(a, b);  // a & ~b
}

// Comparisons
int32x4_t eq_i32x4(int32x4_t a, int32x4_t b) {
    return vreinterpretq_s32_u32(vceqq_s32(a, b));
}

int32x4_t lt_i32x4(int32x4_t a, int32x4_t b) {
    return vreinterpretq_s32_u32(vcltq_s32(a, b));
}

int32x4_t gt_i32x4(int32x4_t a, int32x4_t b) {
    return vreinterpretq_s32_u32(vcgtq_s32(a, b));
}

// Select
int32x4_t sel_i32x4(int32x4_t mask, int32x4_t yes, int32x4_t no) {
    return vbslq_s32(vreinterpretq_u32_s32(mask), yes, no);
}

// Horizontal sum
long hsum_i32x4(int32x4_t v) {
    return vaddvq_s32(v);
}

// ============================================================================
// Int64x2 Operations (128-bit, 2 lanes)
// ============================================================================

int64x2_t add_i64x2(int64x2_t a, int64x2_t b) {
    return vaddq_s64(a, b);
}

int64x2_t sub_i64x2(int64x2_t a, int64x2_t b) {
    return vsubq_s64(a, b);
}

int64x2_t and_i64x2(int64x2_t a, int64x2_t b) {
    return vandq_s64(a, b);
}

int64x2_t or_i64x2(int64x2_t a, int64x2_t b) {
    return vorrq_s64(a, b);
}

int64x2_t xor_i64x2(int64x2_t a, int64x2_t b) {
    return veorq_s64(a, b);
}

int64x2_t eq_i64x2(int64x2_t a, int64x2_t b) {
    return vreinterpretq_s64_u64(vceqq_s64(a, b));
}

// ============================================================================
// Float32x2 Operations (64-bit, 2 lanes)
// ============================================================================

float32x2_t add_f32x2(float32x2_t a, float32x2_t b) {
    return vadd_f32(a, b);
}

float32x2_t sub_f32x2(float32x2_t a, float32x2_t b) {
    return vsub_f32(a, b);
}

float32x2_t mul_f32x2(float32x2_t a, float32x2_t b) {
    return vmul_f32(a, b);
}

float32x2_t div_f32x2(float32x2_t a, float32x2_t b) {
    return vdiv_f32(a, b);
}

float32x2_t min_f32x2(float32x2_t a, float32x2_t b) {
    return vmin_f32(a, b);
}

float32x2_t max_f32x2(float32x2_t a, float32x2_t b) {
    return vmax_f32(a, b);
}

float hsum_f32x2(float32x2_t v) {
    return vaddv_f32(v);
}

float dot_f32x2(float32x2_t a, float32x2_t b) {
    return vaddv_f32(vmul_f32(a, b));
}

// ============================================================================
// Mask Operations
// ============================================================================

// Count true lanes in mask
long counttrue_i32x4(int32x4_t mask) {
    // Each true lane is -1 (all bits set), count non-zero lanes
    uint32x4_t bits = vreinterpretq_u32_s32(mask);
    uint32x4_t ones = vshrq_n_u32(bits, 31);  // Get sign bit
    return vaddvq_u32(ones);
}

// All lanes true?
long alltrue_i32x4(int32x4_t mask) {
    return vminvq_u32(vreinterpretq_u32_s32(mask)) != 0;
}

// Any lane true?
long anytrue_i32x4(int32x4_t mask) {
    return vmaxvq_u32(vreinterpretq_u32_s32(mask)) != 0;
}

// ============================================================================
// Type Conversions
// ============================================================================

// Float32 <-> Int32
int32x4_t cvt_f32x4_i32x4(float32x4_t v) {
    return vcvtq_s32_f32(v);
}

float32x4_t cvt_i32x4_f32x4(int32x4_t v) {
    return vcvtq_f32_s32(v);
}

// Float32 <-> Float64 (converts 2 lanes)
float64x2_t cvt_f32x2_f64x2(float32x2_t v) {
    return vcvt_f64_f32(v);
}

float32x2_t cvt_f64x2_f32x2(float64x2_t v) {
    return vcvt_f32_f64(v);
}

// Rounding
float32x4_t round_f32x4(float32x4_t v) {
    return vrndnq_f32(v);  // Round to nearest
}

float32x4_t floor_f32x4(float32x4_t v) {
    return vrndmq_f32(v);  // Round toward -inf
}

float32x4_t ceil_f32x4(float32x4_t v) {
    return vrndpq_f32(v);  // Round toward +inf
}

float32x4_t trunc_f32x4(float32x4_t v) {
    return vrndq_f32(v);   // Round toward zero
}
