#include <arm_neon.h>
#include <stdint.h>

// Helper union for safe type punning
typedef union {
    float f;
    uint32_t u;
} float_uint32_t;

// Optimized RaBitQ bit product calculation using NEON
// Computes: 1*popcount(code & q1) + 2*popcount(code & q2) + 4*popcount(code & q3) + 8*popcount(code & q4)
void rabitq_bit_product_neon(unsigned long long *code,
                             unsigned long long *q1, unsigned long long *q2,
                             unsigned long long *q3, unsigned long long *q4,
                             unsigned long long *res, long *len) {
  int size = *len;

  // Initialize accumulators for each weight
  uint32x4_t sum1_vec0 = vdupq_n_u32(0);
  uint32x4_t sum1_vec1 = vdupq_n_u32(0);
  uint32x4_t sum1_vec2 = vdupq_n_u32(0);
  uint32x4_t sum1_vec3 = vdupq_n_u32(0);

  uint32x4_t sum2_vec0 = vdupq_n_u32(0);
  uint32x4_t sum2_vec1 = vdupq_n_u32(0);
  uint32x4_t sum2_vec2 = vdupq_n_u32(0);
  uint32x4_t sum2_vec3 = vdupq_n_u32(0);

  uint32x4_t sum4_vec0 = vdupq_n_u32(0);
  uint32x4_t sum4_vec1 = vdupq_n_u32(0);
  uint32x4_t sum4_vec2 = vdupq_n_u32(0);
  uint32x4_t sum4_vec3 = vdupq_n_u32(0);

  uint32x4_t sum8_vec0 = vdupq_n_u32(0);
  uint32x4_t sum8_vec1 = vdupq_n_u32(0);
  uint32x4_t sum8_vec2 = vdupq_n_u32(0);
  uint32x4_t sum8_vec3 = vdupq_n_u32(0);

  int i = 0;

  // Process 8 uint64s at a time (4x2 vectors)
  while (i + 8 <= size) {
    // Load 8 uint64s from each array
    uint64x2x4_t code_vec = vld1q_u64_x4(code + i);
    uint64x2x4_t q1_vec = vld1q_u64_x4(q1 + i);
    uint64x2x4_t q2_vec = vld1q_u64_x4(q2 + i);
    uint64x2x4_t q3_vec = vld1q_u64_x4(q3 + i);
    uint64x2x4_t q4_vec = vld1q_u64_x4(q4 + i);

    // Compute code & q1 and popcount for weight 1
    sum1_vec0 = vaddq_u32(sum1_vec0, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[0], q1_vec.val[0]))))));
    sum1_vec1 = vaddq_u32(sum1_vec1, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[1], q1_vec.val[1]))))));
    sum1_vec2 = vaddq_u32(sum1_vec2, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[2], q1_vec.val[2]))))));
    sum1_vec3 = vaddq_u32(sum1_vec3, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[3], q1_vec.val[3]))))));

    // Compute code & q2 and popcount for weight 2
    sum2_vec0 = vaddq_u32(sum2_vec0, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[0], q2_vec.val[0]))))));
    sum2_vec1 = vaddq_u32(sum2_vec1, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[1], q2_vec.val[1]))))));
    sum2_vec2 = vaddq_u32(sum2_vec2, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[2], q2_vec.val[2]))))));
    sum2_vec3 = vaddq_u32(sum2_vec3, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[3], q2_vec.val[3]))))));

    // Compute code & q3 and popcount for weight 4
    sum4_vec0 = vaddq_u32(sum4_vec0, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[0], q3_vec.val[0]))))));
    sum4_vec1 = vaddq_u32(sum4_vec1, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[1], q3_vec.val[1]))))));
    sum4_vec2 = vaddq_u32(sum4_vec2, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[2], q3_vec.val[2]))))));
    sum4_vec3 = vaddq_u32(sum4_vec3, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[3], q3_vec.val[3]))))));

    // Compute code & q4 and popcount for weight 8
    sum8_vec0 = vaddq_u32(sum8_vec0, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[0], q4_vec.val[0]))))));
    sum8_vec1 = vaddq_u32(sum8_vec1, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[1], q4_vec.val[1]))))));
    sum8_vec2 = vaddq_u32(sum8_vec2, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[2], q4_vec.val[2]))))));
    sum8_vec3 = vaddq_u32(sum8_vec3, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec.val[3], q4_vec.val[3]))))));

    i += 8;
  }

  // Process 2 uint64s at a time
  while (i + 2 <= size) {
    uint64x2_t code_vec = vld1q_u64(code + i);
    uint64x2_t q1_vec = vld1q_u64(q1 + i);
    uint64x2_t q2_vec = vld1q_u64(q2 + i);
    uint64x2_t q3_vec = vld1q_u64(q3 + i);
    uint64x2_t q4_vec = vld1q_u64(q4 + i);

    sum1_vec0 = vaddq_u32(sum1_vec0, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec, q1_vec))))));
    sum2_vec0 = vaddq_u32(sum2_vec0, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec, q2_vec))))));
    sum4_vec0 = vaddq_u32(sum4_vec0, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec, q3_vec))))));
    sum8_vec0 = vaddq_u32(sum8_vec0, vpaddlq_u16(vpaddlq_u8(
        vcntq_u8(vreinterpretq_u8_u64(vandq_u64(code_vec, q4_vec))))));

    i += 2;
  }

  // Sum up all the vector accumulators
  uint32_t sum1 = vaddvq_u32(sum1_vec0) + vaddvq_u32(sum1_vec1) +
                  vaddvq_u32(sum1_vec2) + vaddvq_u32(sum1_vec3);
  uint32_t sum2 = vaddvq_u32(sum2_vec0) + vaddvq_u32(sum2_vec1) +
                  vaddvq_u32(sum2_vec2) + vaddvq_u32(sum2_vec3);
  uint32_t sum4 = vaddvq_u32(sum4_vec0) + vaddvq_u32(sum4_vec1) +
                  vaddvq_u32(sum4_vec2) + vaddvq_u32(sum4_vec3);
  uint32_t sum8 = vaddvq_u32(sum8_vec0) + vaddvq_u32(sum8_vec1) +
                  vaddvq_u32(sum8_vec2) + vaddvq_u32(sum8_vec3);

  // Process remaining elements
  for (; i < size; i++) {
    sum1 += __builtin_popcountll(code[i] & q1[i]);
    sum2 += __builtin_popcountll(code[i] & q2[i]);
    sum4 += __builtin_popcountll(code[i] & q3[i]);
    sum8 += __builtin_popcountll(code[i] & q4[i]);
  }

  // Compute weighted sum: 1*sum1 + 2*sum2 + 4*sum4 + 8*sum8
  res[0] = sum1 + (sum2 << 1) + (sum4 << 2) + (sum8 << 3);
}
