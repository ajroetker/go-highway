package main

import "slices"

// GOAT-safe inline C math helpers.
//
// GOAT (C-to-Go-assembly transpiler) cannot handle external function calls
// like expf(), erff(), etc. These generate `bl _expf` instructions that GOAT
// can't link. Instead, we provide static inline polynomial implementations
// that clang inlines at -O3 before GOAT sees the compiled assembly.
//
// Each precision has vector variants (_v_<name>_<prec>) using NEON intrinsics
// and scalar variants (_s_<name>_<prec>) for tail processing.
//
// Float16 and BFloat16 promote to float32 for math, so they reuse the f32
// helpers directly.

// goatSafeMathHelper maps C math function base names to whether a GOAT-safe
// inline polynomial implementation exists. When true, the C AST translator
// emits _v_<name>_<prec>() / _s_<name>_<prec>() calls instead of
// <name>f() / <name>() calls.
var goatSafeMathHelper = map[string]bool{
	"exp":     true,
	"log":     true,
	"sqrt":    true,
	"sigmoid": true,
	"erf":     true,
	"pow":     true,
	"tanh":    true,
}

// goatMathSuffix returns the precision suffix for GOAT-safe math helpers
// based on the element type. Float16 and BFloat16 use f32 because their
// math is computed in promoted float32. Returns "" if no helpers exist.
func goatMathSuffix(elemType string) string {
	switch elemType {
	case "float32", "hwy.Float16", "hwy.BFloat16":
		return "_f32"
	case "float64":
		return "_f64"
	default:
		return ""
	}
}

// ---------------------------------------------------------------------------
// NEON float32 math helpers (also used by float16 and bfloat16 via promotion)
// ---------------------------------------------------------------------------

var neonF32MathHelpers = []string{
	// NEON vectorized exp(x) using Horner's polynomial approximation.
	`static inline float32x4_t _v_exp_f32(float32x4_t x) {
    float32x4_t invLn2 = vdupq_n_f32(1.44269504088896341f);
    float32x4_t ln2Hi = vdupq_n_f32(0.693359375f);
    float32x4_t ln2Lo = vdupq_n_f32(-2.12194440e-4f);
    float32x4_t overflow = vdupq_n_f32(88.72283905206835f);
    float32x4_t underflow = vdupq_n_f32(-87.33654475055310f);
    float32x4_t c1 = vdupq_n_f32(1.0f);
    float32x4_t c2 = vdupq_n_f32(0.5f);
    float32x4_t c3 = vdupq_n_f32(0.16666666666666666f);
    float32x4_t c4 = vdupq_n_f32(0.041666666666666664f);
    float32x4_t c5 = vdupq_n_f32(0.008333333333333333f);
    float32x4_t c6 = vdupq_n_f32(0.001388888888888889f);
    int32x4_t bias = vdupq_n_s32(127);
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t inf_val = vdupq_n_f32(1.0f / 0.0f);
    uint32x4_t over = vcgtq_f32(x, overflow);
    uint32x4_t under = vcltq_f32(x, underflow);
    float32x4_t kf = vrndnq_f32(vmulq_f32(x, invLn2));
    float32x4_t r = vsubq_f32(x, vmulq_f32(kf, ln2Hi));
    r = vsubq_f32(r, vmulq_f32(kf, ln2Lo));
    float32x4_t ep = vfmaq_f32(c5, c6, r);
    ep = vfmaq_f32(c4, ep, r);
    ep = vfmaq_f32(c3, ep, r);
    ep = vfmaq_f32(c2, ep, r);
    ep = vfmaq_f32(c1, ep, r);
    ep = vfmaq_f32(c1, ep, r);
    int32x4_t ki = vcvtnq_s32_f32(kf);
    int32x4_t scale_bits = vshlq_n_s32(vaddq_s32(ki, bias), 23);
    float32x4_t scale = vreinterpretq_f32_s32(scale_bits);
    float32x4_t result = vmulq_f32(ep, scale);
    result = vbslq_f32(over, inf_val, result);
    result = vbslq_f32(under, zero, result);
    return result;
}`,
	// NEON vectorized sigmoid(x) = 1 / (1 + exp(-x)).
	`static inline float32x4_t _v_sigmoid_f32(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t exp_neg = _v_exp_f32(vnegq_f32(x));
    return vdivq_f32(one, vaddq_f32(one, exp_neg));
}`,
	// NEON vectorized erf(x) using Abramowitz & Stegun approximation.
	`static inline float32x4_t _v_erf_f32(float32x4_t x) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t abs_x = vabsq_f32(x);
    uint32x4_t neg_mask = vcltq_f32(x, zero);
    float32x4_t sign = vbslq_f32(neg_mask, vdupq_n_f32(-1.0f), one);
    float32x4_t t = vdivq_f32(one, vfmaq_f32(one, vdupq_n_f32(0.3275911f), abs_x));
    float32x4_t t2 = vmulq_f32(t, t);
    float32x4_t t3 = vmulq_f32(t2, t);
    float32x4_t t4 = vmulq_f32(t3, t);
    float32x4_t t5 = vmulq_f32(t4, t);
    float32x4_t poly = vmulq_f32(vdupq_n_f32(0.254829592f), t);
    poly = vfmaq_f32(poly, vdupq_n_f32(-0.284496736f), t2);
    poly = vfmaq_f32(poly, vdupq_n_f32(1.421413741f), t3);
    poly = vfmaq_f32(poly, vdupq_n_f32(-1.453152027f), t4);
    poly = vfmaq_f32(poly, vdupq_n_f32(1.061405429f), t5);
    float32x4_t exp_neg_x2 = _v_exp_f32(vnegq_f32(vmulq_f32(abs_x, abs_x)));
    float32x4_t result = vsubq_f32(one, vmulq_f32(poly, exp_neg_x2));
    return vmulq_f32(sign, result);
}`,
	// NEON vectorized log(x) using mantissa extraction + polynomial.
	// log(x) = log(m * 2^e) = log(m) + e * ln(2), m in [1,2)
	`static inline float32x4_t _v_log_f32(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t ln2 = vdupq_n_f32(0.6931471805599453f);
    /* Extract exponent: e = ((bits >> 23) & 0xFF) - 127 */
    int32x4_t bits = vreinterpretq_s32_f32(x);
    int32x4_t exp_i = vsubq_s32(vandq_s32(vshrq_n_s32(bits, 23), vdupq_n_s32(0xFF)), vdupq_n_s32(127));
    float32x4_t e = vcvtq_f32_s32(exp_i);
    /* Normalize mantissa to [1,2): m = (bits & 0x7FFFFF) | 0x3F800000 */
    int32x4_t m_bits = vorrq_s32(vandq_s32(bits, vdupq_n_s32(0x007FFFFF)), vdupq_n_s32(0x3F800000));
    float32x4_t m = vreinterpretq_f32_s32(m_bits);
    /* Polynomial for log(m), m in [1,2): use (m-1) as argument */
    float32x4_t f = vsubq_f32(m, one);
    /* Minimax polynomial for log(1+f), f in [0,1) */
    float32x4_t p = vdupq_n_f32(-0.2401093292f);
    p = vfmaq_f32(vdupq_n_f32(0.3317990258f), p, f);
    p = vfmaq_f32(vdupq_n_f32(-0.4998741238f), p, f);
    p = vfmaq_f32(vdupq_n_f32(0.9999964239f), p, f);
    /* result = p + e * ln(2) */
    return vfmaq_f32(p, e, ln2);
}`,
	// NEON vectorized sqrt(x) using hardware vsqrt instruction.
	`static inline float32x4_t _v_sqrt_f32(float32x4_t x) {
    return vsqrtq_f32(x);
}`,
	// Scalar sqrt(x) using hardware fsqrt instruction.
	`static inline float _s_sqrt_f32(float x) {
    return __builtin_sqrtf(x);
}`,
	// Scalar exp(x) using Horner's polynomial.
	`static inline float _s_exp_f32(float x) {
    if (x > 88.0f) return 1.0f / 0.0f;
    if (x < -88.0f) return 0.0f;
    float kf = __builtin_roundf(x * 1.44269504088896341f);
    float r = x - kf * 0.693359375f;
    r = r - kf * (-2.12194440e-4f);
    float ep = (1.0f/720.0f) * r + (1.0f/120.0f);
    ep = ep * r + (1.0f/24.0f);
    ep = ep * r + (1.0f/6.0f);
    ep = ep * r + 0.5f;
    ep = ep * r + 1.0f;
    ep = ep * r + 1.0f;
    int ki = (int)kf;
    unsigned int bits = (unsigned int)(ki + 127) << 23;
    float scale;
    __builtin_memcpy(&scale, &bits, 4);
    return ep * scale;
}`,
	// Scalar sigmoid(x) = 1 / (1 + exp(-x)).
	`static inline float _s_sigmoid_f32(float x) {
    return 1.0f / (1.0f + _s_exp_f32(-x));
}`,
	// Scalar erf(x) using Abramowitz & Stegun approximation.
	`static inline float _s_erf_f32(float x) {
    float sign = 1.0f;
    float ax = x;
    if (x < 0.0f) { sign = -1.0f; ax = -x; }
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float t2 = t * t;
    float t3 = t2 * t;
    float t4 = t3 * t;
    float t5 = t4 * t;
    float y = 1.0f - (0.254829592f * t - 0.284496736f * t2 +
        1.421413741f * t3 - 1.453152027f * t4 + 1.061405429f * t5) *
        _s_exp_f32(-ax * ax);
    return sign * y;
}`,
	// Scalar log(x) using mantissa extraction + polynomial.
	`static inline float _s_log_f32(float x) {
    unsigned int bits;
    __builtin_memcpy(&bits, &x, 4);
    int exp_i = (int)((bits >> 23) & 0xFF) - 127;
    float e = (float)exp_i;
    unsigned int m_bits = (bits & 0x007FFFFF) | 0x3F800000;
    float m;
    __builtin_memcpy(&m, &m_bits, 4);
    float f = m - 1.0f;
    float p = -0.2401093292f;
    p = p * f + 0.3317990258f;
    p = p * f + -0.4998741238f;
    p = p * f + 0.9999964239f;
    p = p * f;
    return p + e * 0.6931471805599453f;
}`,
	// NEON vectorized pow(base, exp) = exp(exp * log(base)).
	`static inline float32x4_t _v_pow_f32(float32x4_t base, float32x4_t exponent) {
    return _v_exp_f32(vmulq_f32(exponent, _v_log_f32(base)));
}`,
	// Scalar pow(base, exp) = exp(exp * log(base)).
	`static inline float _s_pow_f32(float base, float exponent) {
    return _s_exp_f32(exponent * _s_log_f32(base));
}`,
	// NEON vectorized tanh(x) = 2*sigmoid(2x) - 1.
	`static inline float32x4_t _v_tanh_f32(float32x4_t x) {
    float32x4_t two = vdupq_n_f32(2.0f);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t sig2x = _v_sigmoid_f32(vmulq_f32(two, x));
    return vsubq_f32(vmulq_f32(two, sig2x), one);
}`,
	// Scalar tanh(x) = 2*sigmoid(2x) - 1.
	`static inline float _s_tanh_f32(float x) {
    return 2.0f * _s_sigmoid_f32(2.0f * x) - 1.0f;
}`,
}

// ---------------------------------------------------------------------------
// Scalar float64 math helpers (included in f32/f16/bf16 profiles too)
//
// Go's stdmath (math.Exp, math.Log, etc.) always operates on float64. When
// the C AST translator emits _s_exp_f64/_s_log_f64 in f32 code, these helpers
// must be available. They use only scalar double arithmetic — no NEON vector
// types — so they're safe to include in any profile.
// ---------------------------------------------------------------------------

var scalarF64MathHelpers = []string{
	`static inline double _s_exp_f64(double x) {
    if (x > 709.0) return 1.0 / 0.0;
    if (x < -709.0) return 0.0;
    double kf = __builtin_round(x * 1.4426950408889634);
    double r = x - kf * 6.93147180369123816490e-01;
    r = r - kf * 1.90821492927058500170e-10;
    double ep = (1.0/479001600.0) * r + (1.0/39916800.0);
    ep = ep * r + (1.0/3628800.0);
    ep = ep * r + (1.0/362880.0);
    ep = ep * r + (1.0/40320.0);
    ep = ep * r + (1.0/5040.0);
    ep = ep * r + (1.0/720.0);
    ep = ep * r + (1.0/120.0);
    ep = ep * r + (1.0/24.0);
    ep = ep * r + (1.0/6.0);
    ep = ep * r + 0.5;
    ep = ep * r + 1.0;
    ep = ep * r + 1.0;
    long ki = (long)kf;
    unsigned long bits = (unsigned long)(ki + 1023) << 52;
    double scale;
    __builtin_memcpy(&scale, &bits, 8);
    return ep * scale;
}`,
	`static inline double _s_sigmoid_f64(double x) {
    return 1.0 / (1.0 + _s_exp_f64(-x));
}`,
	`static inline double _s_erf_f64(double x) {
    double sign = 1.0;
    double ax = x;
    if (x < 0.0) { sign = -1.0; ax = -x; }
    double t = 1.0 / (1.0 + 0.3275911 * ax);
    double t2 = t * t;
    double t3 = t2 * t;
    double t4 = t3 * t;
    double t5 = t4 * t;
    double y = 1.0 - (0.254829592 * t - 0.284496736 * t2 +
        1.421413741 * t3 - 1.453152027 * t4 + 1.061405429 * t5) *
        _s_exp_f64(-ax * ax);
    return sign * y;
}`,
	`static inline double _s_log_f64(double x) {
    unsigned long bits;
    __builtin_memcpy(&bits, &x, 8);
    long exp_i = (long)((bits >> 52) & 0x7FF) - 1023;
    double e = (double)exp_i;
    unsigned long m_bits = (bits & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000;
    double m;
    __builtin_memcpy(&m, &m_bits, 8);
    double f = m - 1.0;
    double p = 0.1484794514;
    p = p * f + -0.1792383373;
    p = p * f + 0.2211827839;
    p = p * f + -0.2857142857;
    p = p * f + 0.3999999999;
    p = p * f + -0.4999999999;
    p = p * f + 0.9999999999;
    p = p * f;
    return p + e * 0.6931471805599453;
}`,
	// Scalar sqrt(x) using hardware fsqrt instruction for double.
	`static inline double _s_sqrt_f64(double x) {
    return __builtin_sqrt(x);
}`,
	// Scalar tanh(x) = 2*sigmoid(2x) - 1 for double.
	`static inline double _s_tanh_f64(double x) {
    return 2.0 * _s_sigmoid_f64(2.0 * x) - 1.0;
}`,
}

// ---------------------------------------------------------------------------
// NEON float64 math helpers (vector + scalar)
// ---------------------------------------------------------------------------

var neonF64MathHelpers = []string{
	// NEON vectorized exp(x) for double precision.
	// Uses 11-term Horner polynomial (1/2! through 1/12!) for ~15 digits.
	`static inline float64x2_t _v_exp_f64(float64x2_t x) {
    float64x2_t invLn2 = vdupq_n_f64(1.4426950408889634);
    float64x2_t ln2Hi = vdupq_n_f64(6.93147180369123816490e-01);
    float64x2_t ln2Lo = vdupq_n_f64(1.90821492927058500170e-10);
    float64x2_t overflow = vdupq_n_f64(709.7827128933840);
    float64x2_t underflow = vdupq_n_f64(-708.3964185322641);
    float64x2_t one = vdupq_n_f64(1.0);
    float64x2_t zero = vdupq_n_f64(0.0);
    float64x2_t inf_val = vdupq_n_f64(1.0 / 0.0);
    uint64x2_t over = vcgtq_f64(x, overflow);
    uint64x2_t under = vcltq_f64(x, underflow);
    float64x2_t kf = vrndnq_f64(vmulq_f64(x, invLn2));
    float64x2_t r = vsubq_f64(x, vmulq_f64(kf, ln2Hi));
    r = vsubq_f64(r, vmulq_f64(kf, ln2Lo));
    /* Horner: p = c12*r + c11; p = p*r + c10; ... p = p*r + c2; p = p*r + 1; p = p*r + 1 */
    float64x2_t ep = vfmaq_f64(vdupq_n_f64(1.0/39916800.0), vdupq_n_f64(1.0/479001600.0), r);
    ep = vfmaq_f64(vdupq_n_f64(1.0/3628800.0), ep, r);
    ep = vfmaq_f64(vdupq_n_f64(1.0/362880.0), ep, r);
    ep = vfmaq_f64(vdupq_n_f64(1.0/40320.0), ep, r);
    ep = vfmaq_f64(vdupq_n_f64(1.0/5040.0), ep, r);
    ep = vfmaq_f64(vdupq_n_f64(1.0/720.0), ep, r);
    ep = vfmaq_f64(vdupq_n_f64(1.0/120.0), ep, r);
    ep = vfmaq_f64(vdupq_n_f64(1.0/24.0), ep, r);
    ep = vfmaq_f64(vdupq_n_f64(1.0/6.0), ep, r);
    ep = vfmaq_f64(vdupq_n_f64(0.5), ep, r);
    ep = vfmaq_f64(one, ep, r);
    ep = vfmaq_f64(one, ep, r);
    /* Construct 2^k: ((k + 1023) << 52) reinterpreted as double */
    int64x2_t ki = vcvtnq_s64_f64(kf);
    int64x2_t scale_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
    float64x2_t scale = vreinterpretq_f64_s64(scale_bits);
    float64x2_t result = vmulq_f64(ep, scale);
    result = vbslq_f64(over, inf_val, result);
    result = vbslq_f64(under, zero, result);
    return result;
}`,
	// NEON vectorized sigmoid(x) = 1 / (1 + exp(-x)) for double.
	`static inline float64x2_t _v_sigmoid_f64(float64x2_t x) {
    float64x2_t one = vdupq_n_f64(1.0);
    float64x2_t exp_neg = _v_exp_f64(vnegq_f64(x));
    return vdivq_f64(one, vaddq_f64(one, exp_neg));
}`,
	// NEON vectorized erf(x) using Abramowitz & Stegun for double.
	`static inline float64x2_t _v_erf_f64(float64x2_t x) {
    float64x2_t zero = vdupq_n_f64(0.0);
    float64x2_t one = vdupq_n_f64(1.0);
    float64x2_t abs_x = vabsq_f64(x);
    uint64x2_t neg_mask = vcltq_f64(x, zero);
    float64x2_t sign = vbslq_f64(neg_mask, vdupq_n_f64(-1.0), one);
    float64x2_t t = vdivq_f64(one, vfmaq_f64(one, vdupq_n_f64(0.3275911), abs_x));
    float64x2_t t2 = vmulq_f64(t, t);
    float64x2_t t3 = vmulq_f64(t2, t);
    float64x2_t t4 = vmulq_f64(t3, t);
    float64x2_t t5 = vmulq_f64(t4, t);
    float64x2_t poly = vmulq_f64(vdupq_n_f64(0.254829592), t);
    poly = vfmaq_f64(poly, vdupq_n_f64(-0.284496736), t2);
    poly = vfmaq_f64(poly, vdupq_n_f64(1.421413741), t3);
    poly = vfmaq_f64(poly, vdupq_n_f64(-1.453152027), t4);
    poly = vfmaq_f64(poly, vdupq_n_f64(1.061405429), t5);
    float64x2_t exp_neg_x2 = _v_exp_f64(vnegq_f64(vmulq_f64(abs_x, abs_x)));
    float64x2_t result = vsubq_f64(one, vmulq_f64(poly, exp_neg_x2));
    return vmulq_f64(sign, result);
}`,
	// Scalar exp(x) for double precision.
	`static inline double _s_exp_f64(double x) {
    if (x > 709.0) return 1.0 / 0.0;
    if (x < -709.0) return 0.0;
    double kf = __builtin_round(x * 1.4426950408889634);
    double r = x - kf * 6.93147180369123816490e-01;
    r = r - kf * 1.90821492927058500170e-10;
    double ep = (1.0/479001600.0) * r + (1.0/39916800.0);
    ep = ep * r + (1.0/3628800.0);
    ep = ep * r + (1.0/362880.0);
    ep = ep * r + (1.0/40320.0);
    ep = ep * r + (1.0/5040.0);
    ep = ep * r + (1.0/720.0);
    ep = ep * r + (1.0/120.0);
    ep = ep * r + (1.0/24.0);
    ep = ep * r + (1.0/6.0);
    ep = ep * r + 0.5;
    ep = ep * r + 1.0;
    ep = ep * r + 1.0;
    long ki = (long)kf;
    unsigned long bits = (unsigned long)(ki + 1023) << 52;
    double scale;
    __builtin_memcpy(&scale, &bits, 8);
    return ep * scale;
}`,
	// Scalar sigmoid(x) = 1 / (1 + exp(-x)) for double.
	`static inline double _s_sigmoid_f64(double x) {
    return 1.0 / (1.0 + _s_exp_f64(-x));
}`,
	// Scalar erf(x) using Abramowitz & Stegun for double.
	`static inline double _s_erf_f64(double x) {
    double sign = 1.0;
    double ax = x;
    if (x < 0.0) { sign = -1.0; ax = -x; }
    double t = 1.0 / (1.0 + 0.3275911 * ax);
    double t2 = t * t;
    double t3 = t2 * t;
    double t4 = t3 * t;
    double t5 = t4 * t;
    double y = 1.0 - (0.254829592 * t - 0.284496736 * t2 +
        1.421413741 * t3 - 1.453152027 * t4 + 1.061405429 * t5) *
        _s_exp_f64(-ax * ax);
    return sign * y;
}`,
	// NEON vectorized log(x) for double precision.
	`static inline float64x2_t _v_log_f64(float64x2_t x) {
    float64x2_t one = vdupq_n_f64(1.0);
    float64x2_t ln2 = vdupq_n_f64(0.6931471805599453);
    /* Extract exponent: e = ((bits >> 52) & 0x7FF) - 1023 */
    int64x2_t bits = vreinterpretq_s64_f64(x);
    int64x2_t exp_i = vsubq_s64(vandq_s64(vshrq_n_s64(bits, 52), vdupq_n_s64(0x7FF)), vdupq_n_s64(1023));
    float64x2_t e = vcvtq_f64_s64(exp_i);
    /* Normalize mantissa to [1,2) */
    int64x2_t m_bits = vorrq_s64(vandq_s64(bits, vdupq_n_s64(0x000FFFFFFFFFFFFF)), vdupq_n_s64(0x3FF0000000000000));
    float64x2_t m = vreinterpretq_f64_s64(m_bits);
    float64x2_t f = vsubq_f64(m, one);
    /* Higher-order minimax polynomial for log(1+f) in double */
    float64x2_t p = vdupq_n_f64(0.1484794514);
    p = vfmaq_f64(vdupq_n_f64(-0.1792383373), p, f);
    p = vfmaq_f64(vdupq_n_f64(0.2211827839), p, f);
    p = vfmaq_f64(vdupq_n_f64(-0.2857142857), p, f);
    p = vfmaq_f64(vdupq_n_f64(0.3999999999), p, f);
    p = vfmaq_f64(vdupq_n_f64(-0.4999999999), p, f);
    p = vfmaq_f64(vdupq_n_f64(0.9999999999), p, f);
    p = vmulq_f64(p, f);
    return vfmaq_f64(p, e, ln2);
}`,
	// Scalar log(x) for double precision.
	`static inline double _s_log_f64(double x) {
    unsigned long bits;
    __builtin_memcpy(&bits, &x, 8);
    long exp_i = (long)((bits >> 52) & 0x7FF) - 1023;
    double e = (double)exp_i;
    unsigned long m_bits = (bits & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000;
    double m;
    __builtin_memcpy(&m, &m_bits, 8);
    double f = m - 1.0;
    double p = 0.1484794514;
    p = p * f + -0.1792383373;
    p = p * f + 0.2211827839;
    p = p * f + -0.2857142857;
    p = p * f + 0.3999999999;
    p = p * f + -0.4999999999;
    p = p * f + 0.9999999999;
    p = p * f;
    return p + e * 0.6931471805599453;
}`,
	// NEON vectorized pow(base, exp) = exp(exp * log(base)) for double.
	`static inline float64x2_t _v_pow_f64(float64x2_t base, float64x2_t exponent) {
    return _v_exp_f64(vmulq_f64(exponent, _v_log_f64(base)));
}`,
	// Scalar pow(base, exp) = exp(exp * log(base)) for double.
	`static inline double _s_pow_f64(double base, double exponent) {
    return _s_exp_f64(exponent * _s_log_f64(base));
}`,
	// NEON vectorized sqrt(x) using hardware vsqrt instruction for double.
	`static inline float64x2_t _v_sqrt_f64(float64x2_t x) {
    return vsqrtq_f64(x);
}`,
	// Scalar sqrt(x) using hardware fsqrt instruction for double.
	`static inline double _s_sqrt_f64(double x) {
    return __builtin_sqrt(x);
}`,
	// NEON vectorized tanh(x) = 2*sigmoid(2x) - 1 for double.
	`static inline float64x2_t _v_tanh_f64(float64x2_t x) {
    float64x2_t two = vdupq_n_f64(2.0);
    float64x2_t one = vdupq_n_f64(1.0);
    float64x2_t sig2x = _v_sigmoid_f64(vmulq_f64(two, x));
    return vsubq_f64(vmulq_f64(two, sig2x), one);
}`,
	// Scalar tanh(x) = 2*sigmoid(2x) - 1 for double.
	`static inline double _s_tanh_f64(double x) {
    return 2.0 * _s_sigmoid_f64(2.0 * x) - 1.0;
}`,
}

// ---------------------------------------------------------------------------
// NEON mask/iota helpers for float16 (8 lanes, uint16x8_t mask)
// ---------------------------------------------------------------------------

var neonF16MaskHelpers = []string{
	`static inline float16x8_t hwy_iota_f16(void) {
    float16x8_t v = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    return v;
}`,
	`static inline long hwy_all_true_u16(uint16x8_t mask) {
    return vminvq_u16(mask) != 0;
}`,
	`static inline long hwy_all_false_u16(uint16x8_t mask) {
    return vmaxvq_u16(mask) == 0;
}`,
	`static inline long hwy_find_first_true_u16(uint16x8_t mask) {
    if (vgetq_lane_u16(mask, 0)) return 0;
    if (vgetq_lane_u16(mask, 1)) return 1;
    if (vgetq_lane_u16(mask, 2)) return 2;
    if (vgetq_lane_u16(mask, 3)) return 3;
    if (vgetq_lane_u16(mask, 4)) return 4;
    if (vgetq_lane_u16(mask, 5)) return 5;
    if (vgetq_lane_u16(mask, 6)) return 6;
    if (vgetq_lane_u16(mask, 7)) return 7;
    return -1;
}`,
	`static inline long hwy_count_true_u16(uint16x8_t mask) {
    uint16x8_t bits = vshrq_n_u16(mask, 15);
    return (long)vaddvq_u16(bits);
}`,
	`static inline uint16x8_t hwy_first_n_u16(long n) {
    uint16x8_t iota = {0, 1, 2, 3, 4, 5, 6, 7};
    return vcltq_u16(iota, vdupq_n_u16((unsigned short)n));
}`,
	`static inline long hwy_compress_store_f16(float16x8_t v, uint16x8_t mask, float16_t *dst) {
    long count = 0;
    if (vgetq_lane_u16(mask, 0)) { dst[count++] = vgetq_lane_f16(v, 0); }
    if (vgetq_lane_u16(mask, 1)) { dst[count++] = vgetq_lane_f16(v, 1); }
    if (vgetq_lane_u16(mask, 2)) { dst[count++] = vgetq_lane_f16(v, 2); }
    if (vgetq_lane_u16(mask, 3)) { dst[count++] = vgetq_lane_f16(v, 3); }
    if (vgetq_lane_u16(mask, 4)) { dst[count++] = vgetq_lane_f16(v, 4); }
    if (vgetq_lane_u16(mask, 5)) { dst[count++] = vgetq_lane_f16(v, 5); }
    if (vgetq_lane_u16(mask, 6)) { dst[count++] = vgetq_lane_f16(v, 6); }
    if (vgetq_lane_u16(mask, 7)) { dst[count++] = vgetq_lane_f16(v, 7); }
    return count;
}`,
}

// ---------------------------------------------------------------------------
// NEON mask/iota helpers for bfloat16 (8 lanes, uint16x8_t mask)
// ---------------------------------------------------------------------------

var neonBF16MaskHelpers = []string{
	// BF16 iota creates {0,1,2,...,7} as bfloat16 values via promotion from f32
	`static inline bfloat16x8_t hwy_iota_bf16(void) {
    float32x4_t lo = {0.0f, 1.0f, 2.0f, 3.0f};
    float32x4_t hi = {4.0f, 5.0f, 6.0f, 7.0f};
    return vcombine_bf16(vcvt_bf16_f32(lo), vcvt_bf16_f32(hi));
}`,
	// BF16 comparison returns uint16x8_t so we reuse the u16 mask helpers
}

// ---------------------------------------------------------------------------
// Shared NEON mask query helpers for uint32x4_t masks
// (used by f32, s32, u32 profiles — comparison results are always unsigned)
// ---------------------------------------------------------------------------

var neonU32MaskQueryHelpers = []string{
	`static inline long hwy_all_true_u32(uint32x4_t mask) {
    return vminvq_u32(mask) != 0;
}`,
	`static inline long hwy_all_false_u32(uint32x4_t mask) {
    return vmaxvq_u32(mask) == 0;
}`,
	`static inline long hwy_find_first_true_u32(uint32x4_t mask) {
    if (vgetq_lane_u32(mask, 0)) return 0;
    if (vgetq_lane_u32(mask, 1)) return 1;
    if (vgetq_lane_u32(mask, 2)) return 2;
    if (vgetq_lane_u32(mask, 3)) return 3;
    return -1;
}`,
	`static inline long hwy_count_true_u32(uint32x4_t mask) {
    uint32x4_t bits = vshrq_n_u32(mask, 31);
    return (long)vaddvq_u32(bits);
}`,
	`static inline uint32x4_t hwy_first_n_u32(long n) {
    uint32x4_t iota = {0, 1, 2, 3};
    return vcltq_u32(iota, vdupq_n_u32((unsigned int)n));
}`,
}

// ---------------------------------------------------------------------------
// Shared NEON mask query helpers for uint64x2_t masks
// (used by f64, s64, u64 profiles)
// ---------------------------------------------------------------------------

var neonU64MaskQueryHelpers = []string{
	`static inline long hwy_all_true_u64(uint64x2_t mask) {
    return (vgetq_lane_u64(mask, 0) != 0) && (vgetq_lane_u64(mask, 1) != 0);
}`,
	`static inline long hwy_all_false_u64(uint64x2_t mask) {
    return (vgetq_lane_u64(mask, 0) == 0) && (vgetq_lane_u64(mask, 1) == 0);
}`,
	`static inline long hwy_find_first_true_u64(uint64x2_t mask) {
    if (vgetq_lane_u64(mask, 0)) return 0;
    if (vgetq_lane_u64(mask, 1)) return 1;
    return -1;
}`,
	`static inline long hwy_count_true_u64(uint64x2_t mask) {
    long count = 0;
    if (vgetq_lane_u64(mask, 0)) count++;
    if (vgetq_lane_u64(mask, 1)) count++;
    return count;
}`,
	`static inline uint64x2_t hwy_first_n_u64(long n) {
    uint64x2_t iota = {0, 1};
    return vcltq_u64(iota, vdupq_n_u64((unsigned long)n));
}`,
}

// ---------------------------------------------------------------------------
// NEON mask/iota helpers for float32 (4 lanes, uint32x4_t mask)
// ---------------------------------------------------------------------------

var neonF32MaskHelpers = slices.Concat(neonU32MaskQueryHelpers, []string{
	`static inline float32x4_t hwy_iota_f32(void) {
    float32x4_t v = {0.0f, 1.0f, 2.0f, 3.0f};
    return v;
}`,
	`static inline long hwy_compress_store_f32(float32x4_t v, uint32x4_t mask, float *dst) {
    long count = 0;
    if (vgetq_lane_u32(mask, 0)) { dst[count++] = vgetq_lane_f32(v, 0); }
    if (vgetq_lane_u32(mask, 1)) { dst[count++] = vgetq_lane_f32(v, 1); }
    if (vgetq_lane_u32(mask, 2)) { dst[count++] = vgetq_lane_f32(v, 2); }
    if (vgetq_lane_u32(mask, 3)) { dst[count++] = vgetq_lane_f32(v, 3); }
    return count;
}`,
})

// ---------------------------------------------------------------------------
// NEON mask/iota helpers for float64 (2 lanes, uint64x2_t mask)
// ---------------------------------------------------------------------------

var neonF64MaskHelpers = slices.Concat(neonU64MaskQueryHelpers, []string{
	`static inline float64x2_t hwy_iota_f64(void) {
    float64x2_t v = {0.0, 1.0};
    return v;
}`,
	`static inline long hwy_compress_store_f64(float64x2_t v, uint64x2_t mask, double *dst) {
    long count = 0;
    if (vgetq_lane_u64(mask, 0)) { dst[count++] = vgetq_lane_f64(v, 0); }
    if (vgetq_lane_u64(mask, 1)) { dst[count++] = vgetq_lane_f64(v, 1); }
    return count;
}`,
})

// ---------------------------------------------------------------------------
// NEON mask/iota helpers for int32 (4 lanes, uint32x4_t mask)
// ---------------------------------------------------------------------------

var neonS32MaskHelpers = slices.Concat(neonU32MaskQueryHelpers, []string{
	`static inline int32x4_t hwy_iota_s32(void) {
    int32x4_t v = {0, 1, 2, 3};
    return v;
}`,
	`static inline long hwy_compress_store_s32(int32x4_t v, uint32x4_t mask, int *dst) {
    long count = 0;
    if (vgetq_lane_u32(mask, 0)) { dst[count++] = vgetq_lane_s32(v, 0); }
    if (vgetq_lane_u32(mask, 1)) { dst[count++] = vgetq_lane_s32(v, 1); }
    if (vgetq_lane_u32(mask, 2)) { dst[count++] = vgetq_lane_s32(v, 2); }
    if (vgetq_lane_u32(mask, 3)) { dst[count++] = vgetq_lane_s32(v, 3); }
    return count;
}`,
})

// ---------------------------------------------------------------------------
// NEON mask/iota helpers for uint32 (4 lanes, uint32x4_t mask)
// ---------------------------------------------------------------------------

var neonU32MaskHelpers = slices.Concat(neonU32MaskQueryHelpers, []string{
	`static inline uint32x4_t hwy_iota_u32(void) {
    uint32x4_t v = {0, 1, 2, 3};
    return v;
}`,
	`static inline long hwy_compress_store_u32(uint32x4_t v, uint32x4_t mask, unsigned int *dst) {
    long count = 0;
    if (vgetq_lane_u32(mask, 0)) { dst[count++] = vgetq_lane_u32(v, 0); }
    if (vgetq_lane_u32(mask, 1)) { dst[count++] = vgetq_lane_u32(v, 1); }
    if (vgetq_lane_u32(mask, 2)) { dst[count++] = vgetq_lane_u32(v, 2); }
    if (vgetq_lane_u32(mask, 3)) { dst[count++] = vgetq_lane_u32(v, 3); }
    return count;
}`,
})

// ---------------------------------------------------------------------------
// NEON max/min helpers for 64-bit integers (no native NEON horizontal max/min)
// ---------------------------------------------------------------------------

var neonS64MaxMinHelpers = []string{
	`static inline int64x2_t hwy_max_s64(int64x2_t a, int64x2_t b) {
    uint64x2_t mask = vcgtq_s64(a, b);
    return vbslq_s64(mask, a, b);
}`,
	`static inline int64x2_t hwy_min_s64(int64x2_t a, int64x2_t b) {
    uint64x2_t mask = vcltq_s64(a, b);
    return vbslq_s64(mask, a, b);
}`,
	`static inline long hwy_reducemax_s64(int64x2_t v) {
    long a = vgetq_lane_s64(v, 0);
    long b = vgetq_lane_s64(v, 1);
    return a > b ? a : b;
}`,
	`static inline long hwy_reducemin_s64(int64x2_t v) {
    long a = vgetq_lane_s64(v, 0);
    long b = vgetq_lane_s64(v, 1);
    return a < b ? a : b;
}`,
}

var neonU64MaxMinHelpers = []string{
	`static inline uint64x2_t hwy_max_u64(uint64x2_t a, uint64x2_t b) {
    uint64x2_t mask = vcgtq_u64(a, b);
    return vbslq_u64(mask, a, b);
}`,
	`static inline uint64x2_t hwy_min_u64(uint64x2_t a, uint64x2_t b) {
    uint64x2_t mask = vcltq_u64(a, b);
    return vbslq_u64(mask, a, b);
}`,
	`static inline unsigned long hwy_reducemax_u64(uint64x2_t v) {
    unsigned long a = vgetq_lane_u64(v, 0);
    unsigned long b = vgetq_lane_u64(v, 1);
    return a > b ? a : b;
}`,
	`static inline unsigned long hwy_reducemin_u64(uint64x2_t v) {
    unsigned long a = vgetq_lane_u64(v, 0);
    unsigned long b = vgetq_lane_u64(v, 1);
    return a < b ? a : b;
}`,
}

// ---------------------------------------------------------------------------
// NEON mask/iota helpers for int64 (2 lanes, uint64x2_t mask)
// ---------------------------------------------------------------------------

var neonS64MaskHelpers = slices.Concat(neonU64MaskQueryHelpers, []string{
	`static inline int64x2_t hwy_iota_s64(void) {
    int64x2_t v = {0, 1};
    return v;
}`,
	`static inline long hwy_compress_store_s64(int64x2_t v, uint64x2_t mask, long *dst) {
    long count = 0;
    if (vgetq_lane_u64(mask, 0)) { dst[count++] = vgetq_lane_s64(v, 0); }
    if (vgetq_lane_u64(mask, 1)) { dst[count++] = vgetq_lane_s64(v, 1); }
    return count;
}`,
})

// ---------------------------------------------------------------------------
// NEON mask/iota helpers for uint64 (2 lanes, uint64x2_t mask)
// ---------------------------------------------------------------------------

var neonU64MaskHelpers = slices.Concat(neonU64MaskQueryHelpers, []string{
	`static inline uint64x2_t hwy_iota_u64(void) {
    uint64x2_t v = {0, 1};
    return v;
}`,
	`static inline long hwy_compress_store_u64(uint64x2_t v, uint64x2_t mask, unsigned long *dst) {
    long count = 0;
    if (vgetq_lane_u64(mask, 0)) { dst[count++] = vgetq_lane_u64(v, 0); }
    if (vgetq_lane_u64(mask, 1)) { dst[count++] = vgetq_lane_u64(v, 1); }
    return count;
}`,
})
