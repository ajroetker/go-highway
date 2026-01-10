// Float64 transcendental math functions for ARM64 NEON
// NOTE: All F64 functions process 2 elements at a time (SIMD only, no scalar remainder)
// Callers must ensure length is a multiple of 2.

#include <arm_neon.h>

// Exp2 float64: result[i] = 2^input[i]
// Uses range reduction: 2^x = 2^k * 2^r where k = round(x) and r = x - k
void exp2_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // ln2 bits: 0x3FE62E42FEFA39EF
    float64x2_t v_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE62E42FEFA39EFLL));

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Clamp input to prevent overflow/underflow
        x = vmaxq_f64(x, vdupq_n_f64(-1022.0));
        x = vminq_f64(x, vdupq_n_f64(1023.0));

        // k = round(x)
        float64x2_t k = vrndnq_f64(x);

        // r = x - k, so r is in [-0.5, 0.5]
        float64x2_t r = vsubq_f64(x, k);

        // Compute 2^r = exp(r * ln(2)) using polynomial
        float64x2_t y = vmulq_f64(r, v_ln2);

        // exp(y) using Horner's method polynomial (more terms for double precision)
        float64x2_t exp_r = vdupq_n_f64(2.7557319223985893e-6);   // 1/9!
        exp_r = vfmaq_f64(vdupq_n_f64(2.48015873015873e-5), exp_r, y);   // 1/8!
        exp_r = vfmaq_f64(vdupq_n_f64(1.984126984126984e-4), exp_r, y);  // 1/7!
        exp_r = vfmaq_f64(vdupq_n_f64(1.388888888888889e-3), exp_r, y);  // 1/6!
        exp_r = vfmaq_f64(vdupq_n_f64(8.333333333333333e-3), exp_r, y);  // 1/5!
        exp_r = vfmaq_f64(vdupq_n_f64(4.166666666666667e-2), exp_r, y);  // 1/4!
        exp_r = vfmaq_f64(vdupq_n_f64(0.16666666666666666), exp_r, y);   // 1/3!
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, y);                    // 1/2!
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, y);                    // 1/1!
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, y);                    // 1/0!

        // Scale by 2^k using IEEE double bit manipulation
        // Convert k to int64, add to exponent bias (1023), shift to exponent position
        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);

        vst1q_f64(result + i, vmulq_f64(exp_r, scale));
    }
}

// Log2 float64: result[i] = log2(input[i])
// Uses range reduction: log2(x) = k + log2(m) where x = m * 2^k, 1 <= m < 2
void log2_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // inv_ln2 bits: 0x3FF71547652B82FE
    float64x2_t v_inv_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF71547652B82FELL));
    float64x2_t v_one = vdupq_n_f64(1.0);

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Extract exponent and mantissa from IEEE double
        int64x2_t xi = vreinterpretq_s64_f64(x);
        int64x2_t exp_bits = vshrq_n_s64(xi, 52);
        int64x2_t k = vsubq_s64(vandq_s64(exp_bits, vdupq_n_s64(0x7FF)), vdupq_n_s64(1023));

        // Set exponent to 0 (bias 1023) to get mantissa in [1, 2)
        int64x2_t mantissa_bits = vorrq_s64(
            vandq_s64(xi, vdupq_n_s64(0x000FFFFFFFFFFFFFLL)),
            vdupq_n_s64(0x3FF0000000000000LL)
        );
        float64x2_t m = vreinterpretq_f64_s64(mantissa_bits);

        // f = m - 1, so we compute log(1 + f)
        float64x2_t f = vsubq_f64(m, v_one);

        // Polynomial approximation for log(1+f) with more terms for double precision
        float64x2_t f2 = vmulq_f64(f, f);
        float64x2_t f3 = vmulq_f64(f2, f);
        float64x2_t f4 = vmulq_f64(f2, f2);
        float64x2_t f5 = vmulq_f64(f4, f);
        float64x2_t f6 = vmulq_f64(f3, f3);
        float64x2_t f7 = vmulq_f64(f6, f);
        float64x2_t f8 = vmulq_f64(f4, f4);

        float64x2_t log_m = vmulq_f64(f, vdupq_n_f64(1.0));
        log_m = vfmaq_f64(log_m, f2, vdupq_n_f64(-0.5));
        log_m = vfmaq_f64(log_m, f3, vdupq_n_f64(0.3333333333333333));
        log_m = vfmaq_f64(log_m, f4, vdupq_n_f64(-0.25));
        log_m = vfmaq_f64(log_m, f5, vdupq_n_f64(0.2));
        log_m = vfmaq_f64(log_m, f6, vdupq_n_f64(-0.16666666666666666));
        log_m = vfmaq_f64(log_m, f7, vdupq_n_f64(0.14285714285714285));
        log_m = vfmaq_f64(log_m, f8, vdupq_n_f64(-0.125));

        // log2(x) = k + log(m) / ln(2) = k + log(m) * inv_ln2
        float64x2_t kf = vcvtq_f64_s64(k);
        float64x2_t res = vfmaq_f64(kf, log_m, v_inv_ln2);

        vst1q_f64(result + i, res);
    }
}

// ============================================================================
// Float64 Transcendental Operations (2 lanes per 128-bit vector)
// ============================================================================

// Exp float64: result[i] = exp(input[i])
// Uses range reduction: exp(x) = 2^k * exp(r), where k = round(x/ln(2)), r = x - k*ln(2)
void exp_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants for exp approximation (using bit patterns)
    // ln2 = 0.6931471805599453, bits: 0x3FE62E42FEFA39EF
    // inv_ln2 = 1.4426950408889634, bits: 0x3FF71547652B82FE
    float64x2_t v_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE62E42FEFA39EFLL));
    float64x2_t v_inv_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF71547652B82FELL));

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Clamp input to prevent overflow/underflow
        x = vmaxq_f64(x, vdupq_n_f64(-709.0));
        x = vminq_f64(x, vdupq_n_f64(709.0));

        // k = round(x / ln(2))
        float64x2_t k = vrndnq_f64(vmulq_f64(x, v_inv_ln2));

        // r = x - k * ln(2)
        float64x2_t r = vfmsq_f64(x, k, v_ln2);

        // exp(r) using polynomial (Horner's method) - more terms for double precision
        // exp(r) ≈ 1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5! + r^6/6! + r^7/7! + r^8/8!
        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);  // 1/8!
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);  // 1/7!
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);  // 1/6!
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);  // 1/5!
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);  // 1/4!
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);  // 1/3!
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);                         // 1/2!
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);                         // 1/1!
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);                         // 1/0!

        // Scale by 2^k
        // Convert k to int, add to exponent bias (1023), shift to exponent position
        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);

        vst1q_f64(result + i, vmulq_f64(exp_r, scale));
    }
}

// Log float64: result[i] = log(input[i]) (natural logarithm)
// Uses range reduction: log(x) = k*ln(2) + log(m) where x = m * 2^k, 1 <= m < 2
void log_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // ln2 bits: 0x3FE62E42FEFA39EF
    float64x2_t v_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE62E42FEFA39EFLL));
    float64x2_t v_one = vdupq_n_f64(1.0);

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Extract exponent and mantissa
        int64x2_t xi = vreinterpretq_s64_f64(x);
        int64x2_t exp_bits = vshrq_n_s64(xi, 52);
        int64x2_t k = vsubq_s64(vandq_s64(exp_bits, vdupq_n_s64(0x7FF)), vdupq_n_s64(1023));

        // Set exponent to 0 (bias 1023) to get mantissa in [1, 2)
        int64x2_t mantissa_bits = vorrq_s64(
            vandq_s64(xi, vdupq_n_s64(0x000FFFFFFFFFFFFFLL)),
            vdupq_n_s64(0x3FF0000000000000LL)
        );
        float64x2_t m = vreinterpretq_f64_s64(mantissa_bits);

        // f = m - 1, so we compute log(1 + f)
        float64x2_t f = vsubq_f64(m, v_one);

        // Polynomial approximation for log(1+f)
        float64x2_t f2 = vmulq_f64(f, f);
        float64x2_t f3 = vmulq_f64(f2, f);
        float64x2_t f4 = vmulq_f64(f2, f2);
        float64x2_t f5 = vmulq_f64(f4, f);
        float64x2_t f6 = vmulq_f64(f3, f3);
        float64x2_t f7 = vmulq_f64(f6, f);
        float64x2_t f8 = vmulq_f64(f4, f4);

        float64x2_t log_m = vmulq_f64(f, vdupq_n_f64(1.0));
        log_m = vfmaq_f64(log_m, f2, vdupq_n_f64(-0.5));
        log_m = vfmaq_f64(log_m, f3, vdupq_n_f64(0.3333333333333333));
        log_m = vfmaq_f64(log_m, f4, vdupq_n_f64(-0.25));
        log_m = vfmaq_f64(log_m, f5, vdupq_n_f64(0.2));
        log_m = vfmaq_f64(log_m, f6, vdupq_n_f64(-0.16666666666666666));
        log_m = vfmaq_f64(log_m, f7, vdupq_n_f64(0.14285714285714285));
        log_m = vfmaq_f64(log_m, f8, vdupq_n_f64(-0.125));

        // log(x) = k * ln(2) + log(m)
        float64x2_t kf = vcvtq_f64_s64(k);
        float64x2_t res = vfmaq_f64(log_m, kf, v_ln2);

        vst1q_f64(result + i, res);
    }
}

// Sin float64: result[i] = sin(input[i])
// Uses range reduction to [-pi, pi], reflection to [-pi/2, pi/2], and polynomial
void sin_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants (using bit patterns for non-immediate values)
    // pi = 3.14159265358979323846, bits: 0x400921FB54442D18
    // inv_pi = 0.3183098861837907, bits: 0x3FD45F306DC9C883
    // half_pi = 1.5707963267948966, bits: 0x3FF921FB54442D18
    float64x2_t v_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x400921FB54442D18LL));
    float64x2_t v_neg_pi = vnegq_f64(v_pi);
    float64x2_t v_half_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF921FB54442D18LL));
    float64x2_t v_neg_half_pi = vnegq_f64(v_half_pi);
    float64x2_t v_inv_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FD45F306DC9C883LL));
    float64x2_t v_two = vdupq_n_f64(2.0);

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Range reduction: x = x - 2*pi*round(x/(2*pi)) -> x in [-pi, pi]
        float64x2_t k = vrndnq_f64(vmulq_f64(x, vmulq_f64(vdupq_n_f64(0.5), v_inv_pi)));
        x = vfmsq_f64(x, k, vmulq_f64(v_two, v_pi));

        // Reflection to [-pi/2, pi/2]:
        // if x > pi/2:  sin(x) = sin(pi - x)
        // if x < -pi/2: sin(x) = sin(-pi - x)
        uint64x2_t need_pos_reflect = vcgtq_f64(x, v_half_pi);
        uint64x2_t need_neg_reflect = vcltq_f64(x, v_neg_half_pi);
        float64x2_t x_pos_reflected = vsubq_f64(v_pi, x);
        float64x2_t x_neg_reflected = vsubq_f64(v_neg_pi, x);
        x = vbslq_f64(need_pos_reflect, x_pos_reflected, x);
        x = vbslq_f64(need_neg_reflect, x_neg_reflected, x);

        // sin(x) using polynomial
        float64x2_t x2 = vmulq_f64(x, x);

        // Coefficients: s11 = -2.5052108385441718e-8, s9 = 2.7557319223985893e-6, etc.
        float64x2_t p = vdupq_n_f64(-2.5052108385441718e-8);   // s11
        p = vfmaq_f64(vdupq_n_f64(2.7557319223985893e-6), p, x2);   // s9
        p = vfmaq_f64(vdupq_n_f64(-0.0001984126984126984), p, x2);  // s7
        p = vfmaq_f64(vdupq_n_f64(0.008333333333333333), p, x2);    // s5
        p = vfmaq_f64(vdupq_n_f64(-0.16666666666666666), p, x2);    // s3
        p = vfmaq_f64(vdupq_n_f64(1.0), p, x2);                     // s1
        p = vmulq_f64(p, x);

        vst1q_f64(result + i, p);
    }
}

// Cos float64: result[i] = cos(input[i])
// Uses range reduction to [-pi, pi], reflection to [0, pi/2], and polynomial
void cos_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // Constants
    float64x2_t v_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x400921FB54442D18LL));
    float64x2_t v_half_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF921FB54442D18LL));
    float64x2_t v_inv_pi = vreinterpretq_f64_s64(vdupq_n_s64(0x3FD45F306DC9C883LL));
    float64x2_t v_two = vdupq_n_f64(2.0);
    float64x2_t v_neg_one = vdupq_n_f64(-1.0);
    float64x2_t v_one = vdupq_n_f64(1.0);

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Range reduction: x = x - 2*pi*round(x/(2*pi)) -> x in [-pi, pi]
        float64x2_t k = vrndnq_f64(vmulq_f64(x, vmulq_f64(vdupq_n_f64(0.5), v_inv_pi)));
        x = vfmsq_f64(x, k, vmulq_f64(v_two, v_pi));

        // cos(x) = cos(|x|) since cosine is even
        x = vabsq_f64(x);

        // Reflection: if |x| > pi/2, use cos(|x|) = -cos(pi - |x|)
        uint64x2_t need_reflect = vcgtq_f64(x, v_half_pi);
        float64x2_t x_reflected = vsubq_f64(v_pi, x);
        x = vbslq_f64(need_reflect, x_reflected, x);
        float64x2_t sign = vbslq_f64(need_reflect, v_neg_one, v_one);

        // cos(x) using polynomial: 1 + x^2*(c2 + x^2*(c4 + x^2*(c6 + x^2*(c8 + x^2*c10))))
        float64x2_t x2 = vmulq_f64(x, x);

        float64x2_t p = vdupq_n_f64(-2.7557319223985888e-7);   // c10
        p = vfmaq_f64(vdupq_n_f64(2.48015873015873016e-5), p, x2);   // c8
        p = vfmaq_f64(vdupq_n_f64(-0.001388888888888889), p, x2);    // c6
        p = vfmaq_f64(vdupq_n_f64(0.041666666666666664), p, x2);     // c4
        p = vfmaq_f64(vdupq_n_f64(-0.5), p, x2);                     // c2
        p = vfmaq_f64(vdupq_n_f64(1.0), p, x2);                      // c0

        // Apply sign from reflection
        p = vmulq_f64(p, sign);

        vst1q_f64(result + i, p);
    }
}

// Tanh float64: result[i] = tanh(input[i])
// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
void tanh_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    // For |x| > 19, tanh(x) ≈ sign(x)
    float64x2_t v_one = vdupq_n_f64(1.0);
    float64x2_t v_limit = vdupq_n_f64(19.0);
    float64x2_t v_neg_limit = vdupq_n_f64(-19.0);

    // Constants (using bit patterns)
    float64x2_t v_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE62E42FEFA39EFLL));
    float64x2_t v_inv_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF71547652B82FELL));

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Clamp to prevent overflow
        float64x2_t x_clamped = vmaxq_f64(vminq_f64(x, v_limit), v_neg_limit);

        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        float64x2_t two_x = vmulq_f64(x_clamped, vdupq_n_f64(2.0));

        // Range reduction for exp
        float64x2_t k = vrndnq_f64(vmulq_f64(two_x, v_inv_ln2));
        float64x2_t r = vfmsq_f64(two_x, k, v_ln2);

        // exp(r) polynomial - higher precision for double
        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);  // 1/8!
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);  // 1/7!
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);  // 1/6!
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);  // 1/5!
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);  // 1/4!
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);  // 1/3!
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

        // Scale
        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t exp2x = vmulq_f64(exp_r, scale);

        // tanh = (exp2x - 1) / (exp2x + 1)
        float64x2_t num = vsubq_f64(exp2x, v_one);
        float64x2_t den = vaddq_f64(exp2x, v_one);
        float64x2_t res = vdivq_f64(num, den);

        vst1q_f64(result + i, res);
    }
}

// Sigmoid float64: result[i] = 1 / (1 + exp(-input[i]))
void sigmoid_f64_neon(double *input, double *result, long *len) {
    long n = *len;
    long i = 0;

    float64x2_t v_one = vdupq_n_f64(1.0);

    // Constants (using bit patterns)
    float64x2_t v_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE62E42FEFA39EFLL));
    float64x2_t v_inv_ln2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FF71547652B82FELL));

    // Process 2 doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(input + i);

        // Clamp to prevent overflow
        x = vmaxq_f64(x, vdupq_n_f64(-709.0));
        x = vminq_f64(x, vdupq_n_f64(709.0));

        // exp(-x)
        float64x2_t neg_x = vnegq_f64(x);

        // Range reduction for exp
        float64x2_t k = vrndnq_f64(vmulq_f64(neg_x, v_inv_ln2));
        float64x2_t r = vfmsq_f64(neg_x, k, v_ln2);

        // exp(r) polynomial - higher precision for double
        float64x2_t exp_r = vdupq_n_f64(2.48015873015873015873e-5);  // 1/8!
        exp_r = vfmaq_f64(vdupq_n_f64(1.98412698412698412698e-4), exp_r, r);  // 1/7!
        exp_r = vfmaq_f64(vdupq_n_f64(1.38888888888888888889e-3), exp_r, r);  // 1/6!
        exp_r = vfmaq_f64(vdupq_n_f64(8.33333333333333333333e-3), exp_r, r);  // 1/5!
        exp_r = vfmaq_f64(vdupq_n_f64(4.16666666666666666667e-2), exp_r, r);  // 1/4!
        exp_r = vfmaq_f64(vdupq_n_f64(1.66666666666666666667e-1), exp_r, r);  // 1/3!
        exp_r = vfmaq_f64(vdupq_n_f64(0.5), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);
        exp_r = vfmaq_f64(vdupq_n_f64(1.0), exp_r, r);

        // Scale
        int64x2_t ki = vcvtq_s64_f64(k);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        float64x2_t exp_neg_x = vmulq_f64(exp_r, scale);

        // sigmoid = 1 / (1 + exp(-x))
        float64x2_t res = vdivq_f64(v_one, vaddq_f64(v_one, exp_neg_x));

        vst1q_f64(result + i, res);
    }
}
