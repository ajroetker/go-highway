// SME Matrix Multiplication test for GOAT
// Compile with: -march=armv9-a+sme
//
// This implements a simple 16x16 tile matrix multiply using SME FMOPA.
// It mirrors the hand-written assembly in go-highway/hwy/contrib/matmul/
//
// Algorithm: C = A * B where A, B, C are M x N matrices
// Uses outer product: for each k, accumulate: C += A[:,k] * B[k,:]
//
// Note: Using __arm_streaming means the function expects to be called in
// streaming mode. GOAT will inject smstart/smstop around the function.

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
// Parser phase: use stub definitions from prologue
// Compile phase: use real arm_sme.h
#ifndef GOAT_PARSER
#include <arm_sme.h>
#endif

// Simple 16x16 tile FMOPA test
// Computes C[16x16] = A[16x1] * B[1x16] using outer product
// Expected result: C[i][j] = a[i] * b[j] for all i,j
//
// func sme_fmopa_tile(a *float32, b *float32, c *float32)
void sme_fmopa_tile(float *a, float *b, float *c) __arm_streaming __arm_out("za") {
    // Load A column into z0 (16 floats)
    svfloat32_t za = svld1_f32(svptrue_b32(), a);

    // Load B row into z1 (16 floats)
    svfloat32_t zb = svld1_f32(svptrue_b32(), b);

    // Zero the ZA tile
    svzero_za();

    // Outer product accumulate: ZA0 += za * zb^T
    // This computes a 16x16 result in one instruction!
    svmopa_za32_f32_m(0, svptrue_b32(), svptrue_b32(), za, zb);

    // Extract and store all 16 rows of ZA0 to C
    for (int row = 0; row < 16; row++) {
        svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, row);
        svst1_f32(svptrue_b32(), c + row * 16, zrow);
    }
}

// Matrix multiply: C[M,N] = A[M,K] * B[K,N]
// Requires M, N, K to be multiples of 16
// A is transposed (AT[K,M]) for contiguous column access
//
// func sme_matmul_f32(at *float32, b *float32, c *float32, m *int64, n *int64, k *int64)
void sme_matmul_f32(float *at, float *b, float *c, long *pm, long *pn, long *pk) __arm_streaming __arm_out("za") {
    long m = *pm;
    long n = *pn;
    long k = *pk;

    // Process 16x16 tiles
    for (long ti = 0; ti < m; ti += 16) {
        for (long tj = 0; tj < n; tj += 16) {
            // Zero accumulator tile
            svzero_za();

            // Accumulate over K dimension
            for (long kk = 0; kk < k; kk++) {
                // Load A column (from transposed AT): AT[kk, ti:ti+16]
                svfloat32_t za_col = svld1_f32(svptrue_b32(), at + kk * m + ti);

                // Load B row: B[kk, tj:tj+16]
                svfloat32_t zb_row = svld1_f32(svptrue_b32(), b + kk * n + tj);

                // Outer product accumulate
                svmopa_za32_f32_m(0, svptrue_b32(), svptrue_b32(), za_col, zb_row);
            }

            // Store result tile to C[ti:ti+16, tj:tj+16]
            for (int row = 0; row < 16; row++) {
                svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, row);
                svst1_f32(svptrue_b32(), c + (ti + row) * n + tj, zrow);
            }
        }
    }
}

// Simpler version: just test that SME instructions work
// Computes outer product and stores first row
//
// func sme_dot16(a *float32, b *float32, c *float32)
void sme_dot16(float *a, float *b, float *c) __arm_streaming __arm_out("za") {
    // Load vectors
    svfloat32_t za = svld1_f32(svptrue_b32(), a);
    svfloat32_t zb = svld1_f32(svptrue_b32(), b);

    // Zero ZA
    svzero_za();

    // Outer product - result[i][j] = a[i] * b[j]
    svmopa_za32_f32_m(0, svptrue_b32(), svptrue_b32(), za, zb);

    // Extract first row
    svfloat32_t zrow = svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, 0);

    // Store first row
    svst1_f32(svptrue_b32(), c, zrow);
}
