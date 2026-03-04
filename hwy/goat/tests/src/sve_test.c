// SVE/SME test file for GOAT
// Compile with: -march=armv9-a+sme
// This tests automatic streaming mode injection

// Simple vector addition - should auto-vectorize to SVE
void add_f32_sve(float *a, float *b, float *result, long *len) {
    long n = *len;
    for (long i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

// Vector multiply - should auto-vectorize to SVE
void mul_f32_sve(float *a, float *b, float *result, long *len) {
    long n = *len;
    for (long i = 0; i < n; i++) {
        result[i] = a[i] * b[i];
    }
}

// FMA (fused multiply-add) - should auto-vectorize to SVE
void fma_f32_sve(float *a, float *b, float *c, float *result, long *len) {
    long n = *len;
    for (long i = 0; i < n; i++) {
        result[i] = a[i] * b[i] + c[i];
    }
}

// Dot product - common ML operation
void dot_f32_sve(float *a, float *b, float *result, long *len) {
    long n = *len;
    float sum = 0.0f;
    for (long i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    *result = sum;
}
