// SME/SVE RaBitQ bit product for ARM64
// Compile with: -march=armv9-a+sve+sme
//
// Computes: 1*popcount(code & q1) + 2*popcount(code & q2) + 4*popcount(code & q3) + 8*popcount(code & q4)
//
// Uses SVE instructions with automatic streaming mode injection by GoAT.
// On Apple M4+, SVE requires streaming mode - GoAT handles smstart/smstop.

// GOAT's C parser uses GOAT_PARSER=1, clang doesn't
#ifndef GOAT_PARSER
#include <arm_sve.h>
#endif

// RaBitQ bit product using SVE
// Processes vectors using 64-bit SVE operations.
//
// Algorithm:
// 1. Load code and query vectors as 64-bit elements
// 2. AND code with each query
// 3. Count bits per 64-bit lane
// 4. Accumulate sums
// 5. Horizontal reduction
// 6. Compute weighted sum: sum1 + 2*sum2 + 4*sum4 + 8*sum8
//
// func rabitq_bit_product_sme(code, q1, q2, q3, q4, res, len unsafe.Pointer)
void rabitq_bit_product_sme(unsigned long *code,
                            unsigned long *q1, unsigned long *q2,
                            unsigned long *q3, unsigned long *q4,
                            unsigned long *res, long *plen) __arm_streaming {
    long len = *plen;

    // Initialize accumulators
    unsigned long sum1 = 0;
    unsigned long sum2 = 0;
    unsigned long sum4 = 0;
    unsigned long sum8 = 0;

    // Get SVE vector length in 64-bit elements
    long vl = svcntd();

    long i = 0;

    if (len >= vl) {
        // SVE vector accumulators (64-bit lanes)
        svuint64_t acc1 = svdup_u64(0);
        svuint64_t acc2 = svdup_u64(0);
        svuint64_t acc4 = svdup_u64(0);
        svuint64_t acc8 = svdup_u64(0);

        svbool_t pg = svptrue_b64();

        // Main vector loop
        for (; i + vl <= len; i += vl) {
            // Load as 64-bit elements
            svuint64_t vc = svld1_u64(pg, code + i);
            svuint64_t vq1 = svld1_u64(pg, q1 + i);
            svuint64_t vq2 = svld1_u64(pg, q2 + i);
            svuint64_t vq3 = svld1_u64(pg, q3 + i);
            svuint64_t vq4 = svld1_u64(pg, q4 + i);

            // AND and count bits
            svuint64_t and1 = svand_u64_z(pg, vc, vq1);
            svuint64_t and2 = svand_u64_z(pg, vc, vq2);
            svuint64_t and4 = svand_u64_z(pg, vc, vq3);
            svuint64_t and8 = svand_u64_z(pg, vc, vq4);

            // Popcount per 64-bit lane
            svuint64_t cnt1 = svcnt_u64_z(pg, and1);
            svuint64_t cnt2 = svcnt_u64_z(pg, and2);
            svuint64_t cnt4 = svcnt_u64_z(pg, and4);
            svuint64_t cnt8 = svcnt_u64_z(pg, and8);

            // Accumulate
            acc1 = svadd_u64_z(pg, acc1, cnt1);
            acc2 = svadd_u64_z(pg, acc2, cnt2);
            acc4 = svadd_u64_z(pg, acc4, cnt4);
            acc8 = svadd_u64_z(pg, acc8, cnt8);
        }

        // Horizontal reduction
        sum1 = svaddv_u64(pg, acc1);
        sum2 = svaddv_u64(pg, acc2);
        sum4 = svaddv_u64(pg, acc4);
        sum8 = svaddv_u64(pg, acc8);
    }

    // Scalar fallback for remaining elements
    for (; i < len; i++) {
        unsigned long c = code[i];
        sum1 += __builtin_popcountll(c & q1[i]);
        sum2 += __builtin_popcountll(c & q2[i]);
        sum4 += __builtin_popcountll(c & q3[i]);
        sum8 += __builtin_popcountll(c & q4[i]);
    }

    // Compute weighted sum: 1*sum1 + 2*sum2 + 4*sum4 + 8*sum8
    *res = sum1 + (sum2 << 1) + (sum4 << 2) + (sum8 << 3);
}
