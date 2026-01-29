/*
 * Copyright 2025 go-highway Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

// Reciprocal square root estimate (1/sqrt(x)) - ~8-bit precision
float64x2_t rsqrt_f64x2(float64x2_t a) {
    return vrsqrteq_f64(a);
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

// ============================================================================
// Uint8x16 Operations (128-bit, 16 lanes)
// ============================================================================

// Comparisons (return uint8x16_t mask)
uint8x16_t lt_u8x16(uint8x16_t a, uint8x16_t b) {
    return vcltq_u8(a, b);
}

uint8x16_t gt_u8x16(uint8x16_t a, uint8x16_t b) {
    return vcgtq_u8(a, b);
}

uint8x16_t le_u8x16(uint8x16_t a, uint8x16_t b) {
    return vcleq_u8(a, b);
}

uint8x16_t ge_u8x16(uint8x16_t a, uint8x16_t b) {
    return vcgeq_u8(a, b);
}

uint8x16_t eq_u8x16(uint8x16_t a, uint8x16_t b) {
    return vceqq_u8(a, b);
}

// Min/Max
uint8x16_t min_u8x16(uint8x16_t a, uint8x16_t b) {
    return vminq_u8(a, b);
}

uint8x16_t max_u8x16(uint8x16_t a, uint8x16_t b) {
    return vmaxq_u8(a, b);
}

// Saturating arithmetic
uint8x16_t adds_u8x16(uint8x16_t a, uint8x16_t b) {
    return vqaddq_u8(a, b);
}

uint8x16_t subs_u8x16(uint8x16_t a, uint8x16_t b) {
    return vqsubq_u8(a, b);
}

// Bitwise (same instructions as signed, but type-correct declarations)
uint8x16_t and_u8x16(uint8x16_t a, uint8x16_t b) {
    return vandq_u8(a, b);
}

uint8x16_t or_u8x16(uint8x16_t a, uint8x16_t b) {
    return vorrq_u8(a, b);
}

uint8x16_t xor_u8x16(uint8x16_t a, uint8x16_t b) {
    return veorq_u8(a, b);
}

uint8x16_t not_u8x16(uint8x16_t a) {
    return vmvnq_u8(a);
}

// ============================================================================
// Uint16x8 Operations (128-bit, 8 lanes)
// ============================================================================

// Comparisons
uint16x8_t lt_u16x8(uint16x8_t a, uint16x8_t b) {
    return vcltq_u16(a, b);
}

uint16x8_t gt_u16x8(uint16x8_t a, uint16x8_t b) {
    return vcgtq_u16(a, b);
}

uint16x8_t le_u16x8(uint16x8_t a, uint16x8_t b) {
    return vcleq_u16(a, b);
}

uint16x8_t ge_u16x8(uint16x8_t a, uint16x8_t b) {
    return vcgeq_u16(a, b);
}

uint16x8_t eq_u16x8(uint16x8_t a, uint16x8_t b) {
    return vceqq_u16(a, b);
}

// Min/Max
uint16x8_t min_u16x8(uint16x8_t a, uint16x8_t b) {
    return vminq_u16(a, b);
}

uint16x8_t max_u16x8(uint16x8_t a, uint16x8_t b) {
    return vmaxq_u16(a, b);
}

// Saturating arithmetic
uint16x8_t adds_u16x8(uint16x8_t a, uint16x8_t b) {
    return vqaddq_u16(a, b);
}

uint16x8_t subs_u16x8(uint16x8_t a, uint16x8_t b) {
    return vqsubq_u16(a, b);
}

// Bitwise
uint16x8_t and_u16x8(uint16x8_t a, uint16x8_t b) {
    return vandq_u16(a, b);
}

uint16x8_t or_u16x8(uint16x8_t a, uint16x8_t b) {
    return vorrq_u16(a, b);
}

uint16x8_t xor_u16x8(uint16x8_t a, uint16x8_t b) {
    return veorq_u16(a, b);
}

uint16x8_t not_u16x8(uint16x8_t a) {
    return vmvnq_u16(a);
}

// ============================================================================
// Uint32x4 Operations (128-bit, 4 lanes)
// ============================================================================

// Arithmetic
uint32x4_t add_u32x4(uint32x4_t a, uint32x4_t b) {
    return vaddq_u32(a, b);
}

uint32x4_t sub_u32x4(uint32x4_t a, uint32x4_t b) {
    return vsubq_u32(a, b);
}

uint32x4_t mul_u32x4(uint32x4_t a, uint32x4_t b) {
    return vmulq_u32(a, b);
}

// Comparisons
uint32x4_t lt_u32x4(uint32x4_t a, uint32x4_t b) {
    return vcltq_u32(a, b);
}

uint32x4_t gt_u32x4(uint32x4_t a, uint32x4_t b) {
    return vcgtq_u32(a, b);
}

uint32x4_t le_u32x4(uint32x4_t a, uint32x4_t b) {
    return vcleq_u32(a, b);
}

uint32x4_t ge_u32x4(uint32x4_t a, uint32x4_t b) {
    return vcgeq_u32(a, b);
}

uint32x4_t eq_u32x4(uint32x4_t a, uint32x4_t b) {
    return vceqq_u32(a, b);
}

// Min/Max
uint32x4_t min_u32x4(uint32x4_t a, uint32x4_t b) {
    return vminq_u32(a, b);
}

uint32x4_t max_u32x4(uint32x4_t a, uint32x4_t b) {
    return vmaxq_u32(a, b);
}

// Saturating arithmetic
uint32x4_t adds_u32x4(uint32x4_t a, uint32x4_t b) {
    return vqaddq_u32(a, b);
}

uint32x4_t subs_u32x4(uint32x4_t a, uint32x4_t b) {
    return vqsubq_u32(a, b);
}

// Bitwise
uint32x4_t and_u32x4(uint32x4_t a, uint32x4_t b) {
    return vandq_u32(a, b);
}

uint32x4_t or_u32x4(uint32x4_t a, uint32x4_t b) {
    return vorrq_u32(a, b);
}

uint32x4_t xor_u32x4(uint32x4_t a, uint32x4_t b) {
    return veorq_u32(a, b);
}

uint32x4_t not_u32x4(uint32x4_t a) {
    return vmvnq_u32(a);
}

uint32x4_t andnot_u32x4(uint32x4_t a, uint32x4_t b) {
    return vbicq_u32(a, b);  // a & ~b
}

// Horizontal operations
long hsum_u32x4(uint32x4_t v) {
    return vaddvq_u32(v);
}

// ============================================================================
// Uint64x2 Operations (128-bit, 2 lanes)
// ============================================================================

// Arithmetic
uint64x2_t add_u64x2(uint64x2_t a, uint64x2_t b) {
    return vaddq_u64(a, b);
}

uint64x2_t sub_u64x2(uint64x2_t a, uint64x2_t b) {
    return vsubq_u64(a, b);
}

// Comparisons
uint64x2_t lt_u64x2(uint64x2_t a, uint64x2_t b) {
    return vcltq_u64(a, b);
}

uint64x2_t gt_u64x2(uint64x2_t a, uint64x2_t b) {
    return vcgtq_u64(a, b);
}

uint64x2_t le_u64x2(uint64x2_t a, uint64x2_t b) {
    return vcleq_u64(a, b);
}

uint64x2_t ge_u64x2(uint64x2_t a, uint64x2_t b) {
    return vcgeq_u64(a, b);
}

uint64x2_t eq_u64x2(uint64x2_t a, uint64x2_t b) {
    return vceqq_u64(a, b);
}

// Min/Max using comparison + select (no native vminq_u64/vmaxq_u64)
uint64x2_t min_u64x2(uint64x2_t a, uint64x2_t b) {
    uint64x2_t mask = vcltq_u64(a, b);
    return vbslq_u64(mask, a, b);
}

uint64x2_t max_u64x2(uint64x2_t a, uint64x2_t b) {
    uint64x2_t mask = vcgtq_u64(a, b);
    return vbslq_u64(mask, a, b);
}

// Saturating arithmetic
uint64x2_t adds_u64x2(uint64x2_t a, uint64x2_t b) {
    return vqaddq_u64(a, b);
}

uint64x2_t subs_u64x2(uint64x2_t a, uint64x2_t b) {
    return vqsubq_u64(a, b);
}

// Bitwise
uint64x2_t and_u64x2(uint64x2_t a, uint64x2_t b) {
    return vandq_u64(a, b);
}

uint64x2_t or_u64x2(uint64x2_t a, uint64x2_t b) {
    return vorrq_u64(a, b);
}

uint64x2_t xor_u64x2(uint64x2_t a, uint64x2_t b) {
    return veorq_u64(a, b);
}

// Select
uint64x2_t sel_u64x2(uint64x2_t mask, uint64x2_t yes, uint64x2_t no) {
    return vbslq_u64(mask, yes, no);
}

// ============================================================================
// Slide/Extract Operations (for prefix sum, etc.)
// ============================================================================
// vextq extracts from concatenation: result[i] = (i < 16-n) ? b[i+n] : a[i+n-16]
// For slide up by 1: we want [0, v[0], v[1], v[2]] from [v[0], v[1], v[2], v[3]]

// SlideUp: shift elements up (toward higher indices), fill lower with zero
// slide_up_1_f32x4([a,b,c,d]) = [0,a,b,c]
float32x4_t slide_up_1_f32x4(float32x4_t v) {
    float32x4_t zero = vdupq_n_f32(0);
    return vextq_f32(zero, v, 3);  // Take last 3 from zero, first 1 from v? No...
    // vextq_f32(a, b, n) = [a[n], a[n+1], ..., b[0], b[1], ...]
    // We want [0, v[0], v[1], v[2]]
    // vextq_f32(zero, v, 3) = [zero[3], v[0], v[1], v[2]] = [0, v[0], v[1], v[2]] ✓
}

float32x4_t slide_up_2_f32x4(float32x4_t v) {
    float32x4_t zero = vdupq_n_f32(0);
    return vextq_f32(zero, v, 2);  // [0, 0, v[0], v[1]]
}

float64x2_t slide_up_1_f64x2(float64x2_t v) {
    float64x2_t zero = vdupq_n_f64(0);
    return vextq_f64(zero, v, 1);  // [0, v[0]]
}

int32x4_t slide_up_1_i32x4(int32x4_t v) {
    int32x4_t zero = vdupq_n_s32(0);
    return vextq_s32(zero, v, 3);  // [0, v[0], v[1], v[2]]
}

int32x4_t slide_up_2_i32x4(int32x4_t v) {
    int32x4_t zero = vdupq_n_s32(0);
    return vextq_s32(zero, v, 2);  // [0, 0, v[0], v[1]]
}

int64x2_t slide_up_1_i64x2(int64x2_t v) {
    int64x2_t zero = vdupq_n_s64(0);
    return vextq_s64(zero, v, 1);  // [0, v[0]]
}

uint32x4_t slide_up_1_u32x4(uint32x4_t v) {
    uint32x4_t zero = vdupq_n_u32(0);
    return vextq_u32(zero, v, 3);  // [0, v[0], v[1], v[2]]
}

uint32x4_t slide_up_2_u32x4(uint32x4_t v) {
    uint32x4_t zero = vdupq_n_u32(0);
    return vextq_u32(zero, v, 2);  // [0, 0, v[0], v[1]]
}

uint64x2_t slide_up_1_u64x2(uint64x2_t v) {
    uint64x2_t zero = vdupq_n_u64(0);
    return vextq_u64(zero, v, 1);  // [0, v[0]]
}

// ============================================================================
// In-Place Arithmetic Operations (avoid return allocation overhead)
// ============================================================================
// These functions write results directly to an output pointer, avoiding the
// stack allocation overhead of returning [16]byte values in Go.

// Float32x4 in-place operations
void add_f32x4_ip(float32x4_t a, float32x4_t b, float32x4_t *result) {
    *result = vaddq_f32(a, b);
}

void sub_f32x4_ip(float32x4_t a, float32x4_t b, float32x4_t *result) {
    *result = vsubq_f32(a, b);
}

void mul_f32x4_ip(float32x4_t a, float32x4_t b, float32x4_t *result) {
    *result = vmulq_f32(a, b);
}

void div_f32x4_ip(float32x4_t a, float32x4_t b, float32x4_t *result) {
    *result = vdivq_f32(a, b);
}

void min_f32x4_ip(float32x4_t a, float32x4_t b, float32x4_t *result) {
    *result = vminq_f32(a, b);
}

void max_f32x4_ip(float32x4_t a, float32x4_t b, float32x4_t *result) {
    *result = vmaxq_f32(a, b);
}

// Fused multiply-add with accumulator: *acc = a * b + *acc
void muladd_f32x4_acc(float32x4_t a, float32x4_t b, float32x4_t *acc) {
    *acc = vfmaq_f32(*acc, a, b);
}

// Fused multiply-add to output: *result = a * b + c
void muladd_f32x4_ip(float32x4_t a, float32x4_t b, float32x4_t c, float32x4_t *result) {
    *result = vfmaq_f32(c, a, b);
}

// Float64x2 in-place operations
void add_f64x2_ip(float64x2_t a, float64x2_t b, float64x2_t *result) {
    *result = vaddq_f64(a, b);
}

void sub_f64x2_ip(float64x2_t a, float64x2_t b, float64x2_t *result) {
    *result = vsubq_f64(a, b);
}

void mul_f64x2_ip(float64x2_t a, float64x2_t b, float64x2_t *result) {
    *result = vmulq_f64(a, b);
}

void div_f64x2_ip(float64x2_t a, float64x2_t b, float64x2_t *result) {
    *result = vdivq_f64(a, b);
}

void min_f64x2_ip(float64x2_t a, float64x2_t b, float64x2_t *result) {
    *result = vminq_f64(a, b);
}

void max_f64x2_ip(float64x2_t a, float64x2_t b, float64x2_t *result) {
    *result = vmaxq_f64(a, b);
}

// Fused multiply-add with accumulator: *acc = a * b + *acc
void muladd_f64x2_acc(float64x2_t a, float64x2_t b, float64x2_t *acc) {
    *acc = vfmaq_f64(*acc, a, b);
}

// Fused multiply-add to output: *result = a * b + c
void muladd_f64x2_ip(float64x2_t a, float64x2_t b, float64x2_t c, float64x2_t *result) {
    *result = vfmaq_f64(c, a, b);
}

// Int32x4 in-place operations
void add_i32x4_ip(int32x4_t a, int32x4_t b, int32x4_t *result) {
    *result = vaddq_s32(a, b);
}

void sub_i32x4_ip(int32x4_t a, int32x4_t b, int32x4_t *result) {
    *result = vsubq_s32(a, b);
}

void mul_i32x4_ip(int32x4_t a, int32x4_t b, int32x4_t *result) {
    *result = vmulq_s32(a, b);
}

void min_i32x4_ip(int32x4_t a, int32x4_t b, int32x4_t *result) {
    *result = vminq_s32(a, b);
}

void max_i32x4_ip(int32x4_t a, int32x4_t b, int32x4_t *result) {
    *result = vmaxq_s32(a, b);
}

// ============================================================================
// Multi-Register Load Operations (for better memory bandwidth)
// ============================================================================
// These use ld1 with 4 registers which loads 64 bytes in a single instruction,
// providing better memory bandwidth than 4 separate 16-byte loads.

// Float32x4: 4 vectors = 16 floats = 64 bytes
void load4_f32x4(float *ptr, float32x4_t *out0, float32x4_t *out1, float32x4_t *out2, float32x4_t *out3) {
    float32x4x4_t v = vld1q_f32_x4(ptr);
    *out0 = v.val[0];
    *out1 = v.val[1];
    *out2 = v.val[2];
    *out3 = v.val[3];
}

// Float64x2: 4 vectors = 8 doubles = 64 bytes
void load4_f64x2(double *ptr, float64x2_t *out0, float64x2_t *out1, float64x2_t *out2, float64x2_t *out3) {
    float64x2x4_t v = vld1q_f64_x4(ptr);
    *out0 = v.val[0];
    *out1 = v.val[1];
    *out2 = v.val[2];
    *out3 = v.val[3];
}

// Int32x4: 4 vectors = 16 int32s = 64 bytes
void load4_i32x4(int *ptr, int32x4_t *out0, int32x4_t *out1, int32x4_t *out2, int32x4_t *out3) {
    int32x4x4_t v = vld1q_s32_x4(ptr);
    *out0 = v.val[0];
    *out1 = v.val[1];
    *out2 = v.val[2];
    *out3 = v.val[3];
}

// Int64x2: 4 vectors = 8 int64s = 64 bytes
void load4_i64x2(long long *ptr, int64x2_t *out0, int64x2_t *out1, int64x2_t *out2, int64x2_t *out3) {
    int64x2x4_t v = vld1q_s64_x4(ptr);
    *out0 = v.val[0];
    *out1 = v.val[1];
    *out2 = v.val[2];
    *out3 = v.val[3];
}

// Uint32x4: 4 vectors = 16 uint32s = 64 bytes
void load4_u32x4(unsigned int *ptr, uint32x4_t *out0, uint32x4_t *out1, uint32x4_t *out2, uint32x4_t *out3) {
    uint32x4x4_t v = vld1q_u32_x4(ptr);
    *out0 = v.val[0];
    *out1 = v.val[1];
    *out2 = v.val[2];
    *out3 = v.val[3];
}

// Uint64x2: 4 vectors = 8 uint64s = 64 bytes
void load4_u64x2(unsigned long long *ptr, uint64x2_t *out0, uint64x2_t *out1, uint64x2_t *out2, uint64x2_t *out3) {
    uint64x2x4_t v = vld1q_u64_x4(ptr);
    *out0 = v.val[0];
    *out1 = v.val[1];
    *out2 = v.val[2];
    *out3 = v.val[3];
}

// Uint8x16: 4 vectors = 64 bytes
void load4_u8x16(unsigned char *ptr, uint8x16_t *out0, uint8x16_t *out1, uint8x16_t *out2, uint8x16_t *out3) {
    uint8x16x4_t v = vld1q_u8_x4(ptr);
    *out0 = v.val[0];
    *out1 = v.val[1];
    *out2 = v.val[2];
    *out3 = v.val[3];
}

// ============================================================================
// Mask Operations for Uint8x16
// ============================================================================
// NOTE: movmsk_u8x16 (NEON equivalent of x86 pmovmskb) is implemented in pure Go
// as BitsFromMaskFast in hwy/asm/bits_from_mask_fast.go. The C implementation was
// removed because GoAT doesn't properly handle static const arrays - it generates
// empty data labels, causing the assembly to load zeros instead of the weight constants.
// The pure Go implementation uses bit manipulation (AND + multiply trick) which
// is equally fast and doesn't need constant data from memory.
// ============================================================================

// Uint16x8: 4 vectors = 32 uint16s = 64 bytes
void load4_u16x8(unsigned short *ptr, uint16x8_t *out0, uint16x8_t *out1, uint16x8_t *out2, uint16x8_t *out3) {
    uint16x8x4_t v = vld1q_u16_x4(ptr);
    *out0 = v.val[0];
    *out1 = v.val[1];
    *out2 = v.val[2];
    *out3 = v.val[3];
}

