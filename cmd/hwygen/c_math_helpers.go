package main

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
	"sigmoid": true,
	"erf":     true,
	"pow":     true,
	"sin":     true,
	"cos":     true,
	"tanh":    true,
	"rsqrt":   true,
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
    float32x4_t two = vdupq_n_f32(2.0f);
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t ln2Hi = vdupq_n_f32(0.693359375f);
    float32x4_t ln2Lo = vdupq_n_f32(-2.12194440e-4f);
    float32x4_t sqrt2 = vdupq_n_f32(1.414f);
    /* Extract exponent and mantissa */
    int32x4_t bits = vreinterpretq_s32_f32(x);
    int32x4_t exp_i = vsubq_s32(vandq_s32(vshrq_n_s32(bits, 23), vdupq_n_s32(0xFF)), vdupq_n_s32(127));
    float32x4_t e = vcvtq_f32_s32(exp_i);
    int32x4_t m_bits = vorrq_s32(vandq_s32(bits, vdupq_n_s32(0x007FFFFF)), vdupq_n_s32(0x3F800000));
    float32x4_t m = vreinterpretq_f32_s32(m_bits);
    /* Adjust for m > sqrt(2): halve m and increment exponent */
    uint32x4_t mLarge = vcgtq_f32(m, sqrt2);
    m = vbslq_f32(mLarge, vmulq_f32(m, half), m);
    e = vbslq_f32(mLarge, vaddq_f32(e, one), e);
    /* y = (m - 1) / (m + 1) */
    float32x4_t y = vdivq_f32(vsubq_f32(m, one), vaddq_f32(m, one));
    float32x4_t y2 = vmulq_f32(y, y);
    /* Polynomial in y^2: log(m) = 2*y*(1 + y^2/3 + y^4/5 + ...) */
    float32x4_t poly = vfmaq_f32(vdupq_n_f32(0.1428571437183119574f), vdupq_n_f32(0.1111109921607489198f), y2);
    poly = vfmaq_f32(vdupq_n_f32(0.1999999999970470954f), poly, y2);
    poly = vfmaq_f32(vdupq_n_f32(0.3333333333333367565f), poly, y2);
    poly = vfmaq_f32(one, poly, y2);
    float32x4_t logM = vmulq_f32(vmulq_f32(two, y), poly);
    /* result = e*ln2Hi + logM + e*ln2Lo */
    return vaddq_f32(vfmaq_f32(logM, e, ln2Hi), vmulq_f32(e, ln2Lo));
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
    if (m > 1.414f) { m = m * 0.5f; e = e + 1.0f; }
    float y = (m - 1.0f) / (m + 1.0f);
    float y2 = y * y;
    float poly = 0.1111109921607489198f;
    poly = poly * y2 + 0.1428571437183119574f;
    poly = poly * y2 + 0.1999999999970470954f;
    poly = poly * y2 + 0.3333333333333367565f;
    poly = poly * y2 + 1.0f;
    float logM = 2.0f * y * poly;
    return e * 0.693359375f + logM + e * (-2.12194440e-4f);
}`,
	// NEON vectorized pow(base, exp) = exp(exp * log(base)).
	`static inline float32x4_t _v_pow_f32(float32x4_t base, float32x4_t exponent) {
    return _v_exp_f32(vmulq_f32(exponent, _v_log_f32(base)));
}`,
	// Scalar pow(base, exp) = exp(exp * log(base)).
	`static inline float _s_pow_f32(float base, float exponent) {
    return _s_exp_f32(exponent * _s_log_f32(base));
}`,
	// NEON vectorized tanh(x) = 2*sigmoid(2*x) - 1.
	`static inline float32x4_t _v_tanh_f32(float32x4_t x) {
    float32x4_t two = vdupq_n_f32(2.0f);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t sig2x = _v_sigmoid_f32(vmulq_f32(two, x));
    return vsubq_f32(vmulq_f32(two, sig2x), one);
}`,
	// Scalar tanh(x) = 2*sigmoid(2*x) - 1.
	`static inline float _s_tanh_f32(float x) {
    return 2.0f * _s_sigmoid_f32(2.0f * x) - 1.0f;
}`,
	// NEON vectorized rsqrt(x) = 1/sqrt(x) using vrsqrte + 2 Newton-Raphson steps.
	`static inline float32x4_t _v_rsqrt_f32(float32x4_t x) {
    float32x4_t est = vrsqrteq_f32(x);
    est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(x, est), est));
    est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(x, est), est));
    return est;
}`,
	// Scalar rsqrt(x) = 1/sqrt(x). Uses __builtin_sqrtf to avoid <math.h> dependency.
	`static inline float _s_rsqrt_f32(float x) {
    return 1.0f / __builtin_sqrtf(x);
}`,
	// NEON vectorized sin(x) using range reduction to [-pi/4, pi/4] + Chebyshev polynomial.
	// Octant selection via vbslq_f32 for branchless quadrant handling.
	`static inline float32x4_t _v_sin_f32(float32x4_t x) {
    float32x4_t two_over_pi = vdupq_n_f32(0.6366197723675814f);
    float32x4_t pi_over_2_hi = vdupq_n_f32(1.5707963267948966f);
    float32x4_t pi_over_2_lo = vdupq_n_f32(6.123233995736766e-17f);
    /* Range reduction: k = round(x * 2/pi) */
    float32x4_t kf = vrndnq_f32(vmulq_f32(x, two_over_pi));
    int32x4_t ki = vcvtnq_s32_f32(kf);
    /* r = x - k * pi/2 (Cody-Waite two-step) */
    float32x4_t r = vsubq_f32(x, vmulq_f32(kf, pi_over_2_hi));
    r = vsubq_f32(r, vmulq_f32(kf, pi_over_2_lo));
    float32x4_t r2 = vmulq_f32(r, r);
    /* Sin polynomial: r * (1 + r^2*(s1 + r^2*(s2 + r^2*(s3 + r^2*s4)))) */
    float32x4_t sp = vdupq_n_f32(2.718311493989822e-6f);
    sp = vfmaq_f32(vdupq_n_f32(-0.00019839334836096632f), sp, r2);
    sp = vfmaq_f32(vdupq_n_f32(0.008333329385889463f), sp, r2);
    sp = vfmaq_f32(vdupq_n_f32(-0.16666666641626524f), sp, r2);
    float32x4_t sinR = vfmaq_f32(r, vmulq_f32(r, r2), sp);
    /* Cos polynomial: 1 + r^2*(c1 + r^2*(c2 + r^2*(c3 + r^2*c4))) */
    float32x4_t cp = vdupq_n_f32(2.443315711809948e-5f);
    cp = vfmaq_f32(vdupq_n_f32(-0.001388731625493765f), cp, r2);
    cp = vfmaq_f32(vdupq_n_f32(0.04166662453689337f), cp, r2);
    cp = vfmaq_f32(vdupq_n_f32(-0.4999999963229337f), cp, r2);
    float32x4_t cosR = vfmaq_f32(vdupq_n_f32(1.0f), r2, cp);
    /* Octant selection: bit 0 of k -> swap sin/cos, bit 1 -> negate */
    int32x4_t one_i = vdupq_n_s32(1);
    int32x4_t two_i = vdupq_n_s32(2);
    uint32x4_t swap = vtstq_s32(ki, one_i);
    uint32x4_t neg = vtstq_s32(ki, two_i);
    float32x4_t result = vbslq_f32(swap, cosR, sinR);
    result = vbslq_f32(neg, vnegq_f32(result), result);
    return result;
}`,
	// Scalar sin(x) using range reduction + polynomial.
	`static inline float _s_sin_f32(float x) {
    float kf = __builtin_roundf(x * 0.6366197723675814f);
    int ki = (int)kf;
    float r = x - kf * 1.5707963267948966f;
    r = r - kf * 6.123233995736766e-17f;
    float r2 = r * r;
    float sp = 2.718311493989822e-6f;
    sp = sp * r2 + (-0.00019839334836096632f);
    sp = sp * r2 + 0.008333329385889463f;
    sp = sp * r2 + (-0.16666666641626524f);
    float sinR = r + r * r2 * sp;
    float cp = 2.443315711809948e-5f;
    cp = cp * r2 + (-0.001388731625493765f);
    cp = cp * r2 + 0.04166662453689337f;
    cp = cp * r2 + (-0.4999999963229337f);
    float cosR = 1.0f + r2 * cp;
    float result = (ki & 1) ? cosR : sinR;
    if (ki & 2) result = -result;
    return result;
}`,
	// NEON vectorized cos(x) = sin(x + pi/2), implemented via octant offset.
	`static inline float32x4_t _v_cos_f32(float32x4_t x) {
    float32x4_t two_over_pi = vdupq_n_f32(0.6366197723675814f);
    float32x4_t pi_over_2_hi = vdupq_n_f32(1.5707963267948966f);
    float32x4_t pi_over_2_lo = vdupq_n_f32(6.123233995736766e-17f);
    float32x4_t kf = vrndnq_f32(vmulq_f32(x, two_over_pi));
    int32x4_t ki = vcvtnq_s32_f32(kf);
    ki = vaddq_s32(ki, vdupq_n_s32(1)); /* offset by 1 for cos */
    float32x4_t r = vsubq_f32(x, vmulq_f32(kf, pi_over_2_hi));
    r = vsubq_f32(r, vmulq_f32(kf, pi_over_2_lo));
    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t sp = vdupq_n_f32(2.718311493989822e-6f);
    sp = vfmaq_f32(vdupq_n_f32(-0.00019839334836096632f), sp, r2);
    sp = vfmaq_f32(vdupq_n_f32(0.008333329385889463f), sp, r2);
    sp = vfmaq_f32(vdupq_n_f32(-0.16666666641626524f), sp, r2);
    float32x4_t sinR = vfmaq_f32(r, vmulq_f32(r, r2), sp);
    float32x4_t cp = vdupq_n_f32(2.443315711809948e-5f);
    cp = vfmaq_f32(vdupq_n_f32(-0.001388731625493765f), cp, r2);
    cp = vfmaq_f32(vdupq_n_f32(0.04166662453689337f), cp, r2);
    cp = vfmaq_f32(vdupq_n_f32(-0.4999999963229337f), cp, r2);
    float32x4_t cosR = vfmaq_f32(vdupq_n_f32(1.0f), r2, cp);
    int32x4_t one_i = vdupq_n_s32(1);
    int32x4_t two_i = vdupq_n_s32(2);
    uint32x4_t swap = vtstq_s32(ki, one_i);
    uint32x4_t neg = vtstq_s32(ki, two_i);
    float32x4_t result = vbslq_f32(swap, cosR, sinR);
    result = vbslq_f32(neg, vnegq_f32(result), result);
    return result;
}`,
	// Scalar cos(x) = sin(x + pi/2) via octant offset.
	`static inline float _s_cos_f32(float x) {
    float kf = __builtin_roundf(x * 0.6366197723675814f);
    int ki = (int)kf + 1; /* offset by 1 for cos */
    float r = x - kf * 1.5707963267948966f;
    r = r - kf * 6.123233995736766e-17f;
    float r2 = r * r;
    float sp = 2.718311493989822e-6f;
    sp = sp * r2 + (-0.00019839334836096632f);
    sp = sp * r2 + 0.008333329385889463f;
    sp = sp * r2 + (-0.16666666641626524f);
    float sinR = r + r * r2 * sp;
    float cp = 2.443315711809948e-5f;
    cp = cp * r2 + (-0.001388731625493765f);
    cp = cp * r2 + 0.04166662453689337f;
    cp = cp * r2 + (-0.4999999963229337f);
    float cosR = 1.0f + r2 * cp;
    float result = (ki & 1) ? cosR : sinR;
    if (ki & 2) result = -result;
    return result;
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
    if (m > 1.4142135623730951) { m = m * 0.5; e = e + 1.0; }
    double y = (m - 1.0) / (m + 1.0);
    double y2 = y * y;
    double poly = 0.0765691884960468666;
    poly = poly * y2 + 0.0909178608080902506;
    poly = poly * y2 + 0.1111109921607489198;
    poly = poly * y2 + 0.1428571437183119574;
    poly = poly * y2 + 0.1999999999970470954;
    poly = poly * y2 + 0.3333333333333367565;
    poly = poly * y2 + 1.0;
    double logM = 2.0 * y * poly;
    return e * 0.6931471803691238 + logM + e * 1.9082149292705877e-10;
}`,
	`static inline double _s_tanh_f64(double x) {
    return 2.0 * _s_sigmoid_f64(2.0 * x) - 1.0;
}`,
	`static inline double _s_sin_f64(double x) {
    double kf = __builtin_round(x * 0.6366197723675814);
    long ki = (long)kf;
    double r = x - kf * 1.5707963267948966192313216916398;
    r = r - kf * 6.123233995736766035868820147292e-17;
    double r2 = r * r;
    double sp = 2.7557316103728803e-6;
    sp = sp * r2 + (-0.00019841269840885721);
    sp = sp * r2 + 0.008333333333332249;
    sp = sp * r2 + (-0.16666666666666632);
    double sinR = r + r * r2 * sp;
    double cp = 2.4801587288851704e-5;
    cp = cp * r2 + (-0.001388888888887411);
    cp = cp * r2 + 0.04166666666666621;
    cp = cp * r2 + (-0.5);
    double cosR = 1.0 + r2 * cp;
    double result = (ki & 1) ? cosR : sinR;
    if (ki & 2) result = -result;
    return result;
}`,
	`static inline double _s_cos_f64(double x) {
    double kf = __builtin_round(x * 0.6366197723675814);
    long ki = (long)kf + 1;
    double r = x - kf * 1.5707963267948966192313216916398;
    r = r - kf * 6.123233995736766035868820147292e-17;
    double r2 = r * r;
    double sp = 2.7557316103728803e-6;
    sp = sp * r2 + (-0.00019841269840885721);
    sp = sp * r2 + 0.008333333333332249;
    sp = sp * r2 + (-0.16666666666666632);
    double sinR = r + r * r2 * sp;
    double cp = 2.4801587288851704e-5;
    cp = cp * r2 + (-0.001388888888887411);
    cp = cp * r2 + 0.04166666666666621;
    cp = cp * r2 + (-0.5);
    double cosR = 1.0 + r2 * cp;
    double result = (ki & 1) ? cosR : sinR;
    if (ki & 2) result = -result;
    return result;
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
    float64x2_t two = vdupq_n_f64(2.0);
    float64x2_t half = vdupq_n_f64(0.5);
    float64x2_t ln2Hi = vdupq_n_f64(0.6931471803691238);
    float64x2_t ln2Lo = vdupq_n_f64(1.9082149292705877e-10);
    float64x2_t sqrt2 = vdupq_n_f64(1.4142135623730951);
    /* Extract exponent and mantissa */
    int64x2_t bits = vreinterpretq_s64_f64(x);
    int64x2_t exp_i = vsubq_s64(vandq_s64(vshrq_n_s64(bits, 52), vdupq_n_s64(0x7FF)), vdupq_n_s64(1023));
    float64x2_t e = vcvtq_f64_s64(exp_i);
    int64x2_t m_bits = vorrq_s64(vandq_s64(bits, vdupq_n_s64(0x000FFFFFFFFFFFFF)), vdupq_n_s64(0x3FF0000000000000));
    float64x2_t m = vreinterpretq_f64_s64(m_bits);
    /* Adjust for m > sqrt(2) */
    uint64x2_t mLarge = vcgtq_f64(m, sqrt2);
    m = vbslq_f64(mLarge, vmulq_f64(m, half), m);
    e = vbslq_f64(mLarge, vaddq_f64(e, one), e);
    /* y = (m - 1) / (m + 1) */
    float64x2_t y = vdivq_f64(vsubq_f64(m, one), vaddq_f64(m, one));
    float64x2_t y2 = vmulq_f64(y, y);
    /* Polynomial in y^2 with 7 terms for double precision */
    float64x2_t poly = vfmaq_f64(vdupq_n_f64(0.0909178608080902506), vdupq_n_f64(0.0765691884960468666), y2);
    poly = vfmaq_f64(vdupq_n_f64(0.1111109921607489198), poly, y2);
    poly = vfmaq_f64(vdupq_n_f64(0.1428571437183119574), poly, y2);
    poly = vfmaq_f64(vdupq_n_f64(0.1999999999970470954), poly, y2);
    poly = vfmaq_f64(vdupq_n_f64(0.3333333333333367565), poly, y2);
    poly = vfmaq_f64(one, poly, y2);
    float64x2_t logM = vmulq_f64(vmulq_f64(two, y), poly);
    return vaddq_f64(vfmaq_f64(logM, e, ln2Hi), vmulq_f64(e, ln2Lo));
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
    if (m > 1.4142135623730951) { m = m * 0.5; e = e + 1.0; }
    double y = (m - 1.0) / (m + 1.0);
    double y2 = y * y;
    double poly = 0.0765691884960468666;
    poly = poly * y2 + 0.0909178608080902506;
    poly = poly * y2 + 0.1111109921607489198;
    poly = poly * y2 + 0.1428571437183119574;
    poly = poly * y2 + 0.1999999999970470954;
    poly = poly * y2 + 0.3333333333333367565;
    poly = poly * y2 + 1.0;
    double logM = 2.0 * y * poly;
    return e * 0.6931471803691238 + logM + e * 1.9082149292705877e-10;
}`,
	// NEON vectorized pow(base, exp) = exp(exp * log(base)) for double.
	`static inline float64x2_t _v_pow_f64(float64x2_t base, float64x2_t exponent) {
    return _v_exp_f64(vmulq_f64(exponent, _v_log_f64(base)));
}`,
	// Scalar pow(base, exp) = exp(exp * log(base)) for double.
	`static inline double _s_pow_f64(double base, double exponent) {
    return _s_exp_f64(exponent * _s_log_f64(base));
}`,
	// NEON vectorized tanh(x) = 2*sigmoid(2*x) - 1 for double.
	`static inline float64x2_t _v_tanh_f64(float64x2_t x) {
    float64x2_t two = vdupq_n_f64(2.0);
    float64x2_t one = vdupq_n_f64(1.0);
    float64x2_t sig2x = _v_sigmoid_f64(vmulq_f64(two, x));
    return vsubq_f64(vmulq_f64(two, sig2x), one);
}`,
	// Scalar tanh(x) = 2*sigmoid(2*x) - 1 for double.
	`static inline double _s_tanh_f64(double x) {
    return 2.0 * _s_sigmoid_f64(2.0 * x) - 1.0;
}`,
	// NEON vectorized rsqrt(x) = 1/sqrt(x) using vrsqrte + 3 Newton-Raphson steps for double precision.
	`static inline float64x2_t _v_rsqrt_f64(float64x2_t x) {
    float64x2_t est = vrsqrteq_f64(x);
    est = vmulq_f64(est, vrsqrtsq_f64(vmulq_f64(x, est), est));
    est = vmulq_f64(est, vrsqrtsq_f64(vmulq_f64(x, est), est));
    est = vmulq_f64(est, vrsqrtsq_f64(vmulq_f64(x, est), est));
    return est;
}`,
	// Scalar rsqrt(x) = 1/sqrt(x) for double. Uses __builtin_sqrt to avoid <math.h> dependency.
	`static inline double _s_rsqrt_f64(double x) {
    return 1.0 / __builtin_sqrt(x);
}`,
	// NEON vectorized sin(x) using range reduction + Chebyshev for double.
	`static inline float64x2_t _v_sin_f64(float64x2_t x) {
    float64x2_t two_over_pi = vdupq_n_f64(0.6366197723675814);
    float64x2_t pi_over_2_hi = vdupq_n_f64(1.5707963267948966192313216916398);
    float64x2_t pi_over_2_lo = vdupq_n_f64(6.123233995736766035868820147292e-17);
    float64x2_t kf = vrndnq_f64(vmulq_f64(x, two_over_pi));
    int64x2_t ki = vcvtnq_s64_f64(kf);
    float64x2_t r = vsubq_f64(x, vmulq_f64(kf, pi_over_2_hi));
    r = vsubq_f64(r, vmulq_f64(kf, pi_over_2_lo));
    float64x2_t r2 = vmulq_f64(r, r);
    float64x2_t sp = vdupq_n_f64(2.7557316103728803e-6);
    sp = vfmaq_f64(vdupq_n_f64(-0.00019841269840885721), sp, r2);
    sp = vfmaq_f64(vdupq_n_f64(0.008333333333332249), sp, r2);
    sp = vfmaq_f64(vdupq_n_f64(-0.16666666666666632), sp, r2);
    float64x2_t sinR = vfmaq_f64(r, vmulq_f64(r, r2), sp);
    float64x2_t cp = vdupq_n_f64(2.4801587288851704e-5);
    cp = vfmaq_f64(vdupq_n_f64(-0.001388888888887411), cp, r2);
    cp = vfmaq_f64(vdupq_n_f64(0.04166666666666621), cp, r2);
    cp = vfmaq_f64(vdupq_n_f64(-0.5), cp, r2);
    float64x2_t cosR = vfmaq_f64(vdupq_n_f64(1.0), r2, cp);
    int64x2_t one_i = vdupq_n_s64(1);
    int64x2_t two_i = vdupq_n_s64(2);
    uint64x2_t swap = vtstq_s64(ki, one_i);
    uint64x2_t neg = vtstq_s64(ki, two_i);
    float64x2_t result = vbslq_f64(swap, cosR, sinR);
    result = vbslq_f64(neg, vnegq_f64(result), result);
    return result;
}`,
	// Scalar sin(x) for double.
	`static inline double _s_sin_f64(double x) {
    double kf = __builtin_round(x * 0.6366197723675814);
    long ki = (long)kf;
    double r = x - kf * 1.5707963267948966192313216916398;
    r = r - kf * 6.123233995736766035868820147292e-17;
    double r2 = r * r;
    double sp = 2.7557316103728803e-6;
    sp = sp * r2 + (-0.00019841269840885721);
    sp = sp * r2 + 0.008333333333332249;
    sp = sp * r2 + (-0.16666666666666632);
    double sinR = r + r * r2 * sp;
    double cp = 2.4801587288851704e-5;
    cp = cp * r2 + (-0.001388888888887411);
    cp = cp * r2 + 0.04166666666666621;
    cp = cp * r2 + (-0.5);
    double cosR = 1.0 + r2 * cp;
    double result = (ki & 1) ? cosR : sinR;
    if (ki & 2) result = -result;
    return result;
}`,
	// NEON vectorized cos(x) = sin(x + pi/2) via octant offset for double.
	`static inline float64x2_t _v_cos_f64(float64x2_t x) {
    float64x2_t two_over_pi = vdupq_n_f64(0.6366197723675814);
    float64x2_t pi_over_2_hi = vdupq_n_f64(1.5707963267948966192313216916398);
    float64x2_t pi_over_2_lo = vdupq_n_f64(6.123233995736766035868820147292e-17);
    float64x2_t kf = vrndnq_f64(vmulq_f64(x, two_over_pi));
    int64x2_t ki = vcvtnq_s64_f64(kf);
    ki = vaddq_s64(ki, vdupq_n_s64(1));
    float64x2_t r = vsubq_f64(x, vmulq_f64(kf, pi_over_2_hi));
    r = vsubq_f64(r, vmulq_f64(kf, pi_over_2_lo));
    float64x2_t r2 = vmulq_f64(r, r);
    float64x2_t sp = vdupq_n_f64(2.7557316103728803e-6);
    sp = vfmaq_f64(vdupq_n_f64(-0.00019841269840885721), sp, r2);
    sp = vfmaq_f64(vdupq_n_f64(0.008333333333332249), sp, r2);
    sp = vfmaq_f64(vdupq_n_f64(-0.16666666666666632), sp, r2);
    float64x2_t sinR = vfmaq_f64(r, vmulq_f64(r, r2), sp);
    float64x2_t cp = vdupq_n_f64(2.4801587288851704e-5);
    cp = vfmaq_f64(vdupq_n_f64(-0.001388888888887411), cp, r2);
    cp = vfmaq_f64(vdupq_n_f64(0.04166666666666621), cp, r2);
    cp = vfmaq_f64(vdupq_n_f64(-0.5), cp, r2);
    float64x2_t cosR = vfmaq_f64(vdupq_n_f64(1.0), r2, cp);
    int64x2_t one_i = vdupq_n_s64(1);
    int64x2_t two_i = vdupq_n_s64(2);
    uint64x2_t swap = vtstq_s64(ki, one_i);
    uint64x2_t neg = vtstq_s64(ki, two_i);
    float64x2_t result = vbslq_f64(swap, cosR, sinR);
    result = vbslq_f64(neg, vnegq_f64(result), result);
    return result;
}`,
	// Scalar cos(x) for double.
	`static inline double _s_cos_f64(double x) {
    double kf = __builtin_round(x * 0.6366197723675814);
    long ki = (long)kf + 1;
    double r = x - kf * 1.5707963267948966192313216916398;
    r = r - kf * 6.123233995736766035868820147292e-17;
    double r2 = r * r;
    double sp = 2.7557316103728803e-6;
    sp = sp * r2 + (-0.00019841269840885721);
    sp = sp * r2 + 0.008333333333332249;
    sp = sp * r2 + (-0.16666666666666632);
    double sinR = r + r * r2 * sp;
    double cp = 2.4801587288851704e-5;
    cp = cp * r2 + (-0.001388888888887411);
    cp = cp * r2 + 0.04166666666666621;
    cp = cp * r2 + (-0.5);
    double cosR = 1.0 + r2 * cp;
    double result = (ki & 1) ? cosR : sinR;
    if (ki & 2) result = -result;
    return result;
}`,
}
