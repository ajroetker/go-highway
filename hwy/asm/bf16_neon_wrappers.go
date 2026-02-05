// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !noasm && arm64

// BFloat16 NEON operations for ARM64 with BF16 extension (ARMv8.6-A+)
package asm

import "unsafe"

// -march=armv8.6-a+bf16 enables BFDOT and BFMMLA instructions
//go:generate go tool goat ../c/ops_bf16_neon_arm64.c -O3 --target arm64 -e="-march=armv8.6-a+bf16"

// ============================================================================
// BFloat16 Conversions
// ============================================================================

// PromoteBF16ToF32NEON converts bfloat16 to float32 using NEON bit shifts.
// BF16 is simply F32 with the lower 16 bits truncated, so conversion is
// just shifting the 16-bit value left by 16 to form the upper bits of F32.
func PromoteBF16ToF32NEON(a []uint16, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), len(result)))
	promote_bf16_to_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// DemoteF32ToBF16NEON converts float32 to bfloat16 using round-to-nearest-even.
// This properly rounds the F32 value before truncating to 16 bits.
func DemoteF32ToBF16NEON(a []float32, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), len(result)))
	demote_f32_to_bf16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ============================================================================
// BFloat16 Dot Product (ARMv8.6-A BFDOT instruction)
// ============================================================================

// DotBF16NEON computes the dot product of two BF16 vectors, accumulating to F32.
// Uses the BFDOT instruction for optimal performance on ARMv8.6-A+.
// The result is added to the existing value in acc: acc += sum(a[i] * b[i])
func DotBF16NEON(a, b []uint16, acc *float32, n int) {
	if n == 0 || len(a) == 0 || len(b) == 0 {
		return
	}
	count := int64(min(n, min(len(a), len(b))))
	dot_bf16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(acc), unsafe.Pointer(&count))
}

// ============================================================================
// BFloat16 Matrix Multiply (ARMv8.6-A BFMMLA instruction)
// ============================================================================

// MatMulBF16NEON performs tiled matrix multiplication: C += A * B
// A is MxK (row-major), B is KxN (row-major), C is MxN (row-major)
// Uses BFMMLA instruction for 2x2 tiled accumulation when possible.
//
// Parameters:
//   - a: M x K matrix of BF16 values (row-major, leading dimension lda)
//   - b: K x N matrix of BF16 values (row-major, leading dimension ldb)
//   - c: M x N matrix of F32 values (row-major, leading dimension ldc)
//   - m, n, k: matrix dimensions
//   - lda, ldb, ldc: leading dimensions (number of elements between row starts)
func MatMulBF16NEON(a, b []uint16, c []float32, m, n, k, lda, ldb, ldc int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(a) == 0 || len(b) == 0 || len(c) == 0 {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	ldaVal := int64(lda)
	ldbVal := int64(ldb)
	ldcVal := int64(ldc)
	matmul_bf16_neon(
		unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal), unsafe.Pointer(&nVal), unsafe.Pointer(&kVal),
		unsafe.Pointer(&ldaVal), unsafe.Pointer(&ldbVal), unsafe.Pointer(&ldcVal),
	)
}

// ============================================================================
// BFloat16 Vector Load/Store Operations
// ============================================================================

// Load4BF16x8NEON loads 4 consecutive bfloat16x8 vectors (32 bfloat16 values = 64 bytes).
// Uses vld1q_bf16_x4 which loads 64 bytes in a single instruction.
// Returns 4 vectors as [16]byte (each holding 8 bfloat16 values).
func Load4BF16x8NEON(ptr []uint16) (v0, v1, v2, v3 [16]byte) {
	if len(ptr) < 32 {
		return
	}
	load4_bf16x8(unsafe.Pointer(&ptr[0]),
		unsafe.Pointer(&v0[0]), unsafe.Pointer(&v1[0]),
		unsafe.Pointer(&v2[0]), unsafe.Pointer(&v3[0]))
	return
}

// Store4BF16x8NEON stores 4 consecutive bfloat16x8 vectors (32 bfloat16 values = 64 bytes).
// Uses vst1q_bf16_x4 which stores 64 bytes in a single instruction.
func Store4BF16x8NEON(ptr []uint16, v0, v1, v2, v3 [16]byte) {
	if len(ptr) < 32 {
		return
	}
	store4_bf16x8(unsafe.Pointer(&ptr[0]), v0, v1, v2, v3)
}

// Load4BFloat16x8 loads 4 consecutive BFloat16x8 vectors from an unsafe.Pointer.
// ptr must point to at least 32 bfloat16 values (64 bytes).
func Load4BFloat16x8(ptr unsafe.Pointer) (BFloat16x8, BFloat16x8, BFloat16x8, BFloat16x8) {
	u16 := unsafe.Slice((*uint16)(ptr), 32)
	v0, v1, v2, v3 := Load4BF16x8NEON(u16)
	return BFloat16x8(v0), BFloat16x8(v1), BFloat16x8(v2), BFloat16x8(v3)
}

// ============================================================================
// BFloat16x8 Single-Vector Type and Operations
// ============================================================================
// BFloat16x8 represents a 128-bit NEON vector of 8 bfloat16 values.
// Uses [16]byte backing for efficient register passing via GoAT-generated assembly.
//
// Note: BFloat16 does NOT have native SIMD arithmetic. Operations use
// promote-to-F32 -> compute -> demote-to-BF16 pattern. For ML workloads,
// prefer using BFDOT with F32 accumulators for better precision.

// BFloat16x8 represents a 128-bit NEON vector of 8 bfloat16 values.
type BFloat16x8 [16]byte

// ZeroBFloat16x8 returns a zero vector.
func ZeroBFloat16x8() BFloat16x8 {
	return BFloat16x8{}
}

// BroadcastBFloat16x8 broadcasts a scalar bfloat16 to all 8 lanes.
func BroadcastBFloat16x8(val uint16) BFloat16x8 {
	return BFloat16x8(broadcast_bf16x8(unsafe.Pointer(&val)))
}

// LoadBFloat16x8 loads 8 bfloat16 values from a slice (has bounds check).
func LoadBFloat16x8(s []uint16) BFloat16x8 {
	return *(*BFloat16x8)(unsafe.Pointer(&s[0]))
}

// LoadBFloat16x8Slice loads 8 bfloat16 values from a slice (alias for LoadBFloat16x8).
func LoadBFloat16x8Slice(s []uint16) BFloat16x8 {
	return *(*BFloat16x8)(unsafe.Pointer(&s[0]))
}

// LoadBFloat16x8Ptr loads 8 bfloat16 values from an unsafe.Pointer (no bounds check).
// Use this when you have a slice of hwy.BFloat16 and need to avoid type conversion.
func LoadBFloat16x8Ptr(ptr unsafe.Pointer) BFloat16x8 {
	return *(*BFloat16x8)(ptr)
}

// Store stores the vector to a slice (has bounds check).
func (v BFloat16x8) Store(s []uint16) {
	*(*BFloat16x8)(unsafe.Pointer(&s[0])) = v
}

// StoreSlice stores the vector to a slice (alias for Store).
func (v BFloat16x8) StoreSlice(s []uint16) {
	*(*BFloat16x8)(unsafe.Pointer(&s[0])) = v
}

// StorePtr stores the vector to an unsafe.Pointer (no bounds check).
// Use this when you have a slice of hwy.BFloat16 and need to avoid type conversion.
func (v BFloat16x8) StorePtr(ptr unsafe.Pointer) {
	*(*BFloat16x8)(ptr) = v
}

// ===== BFloat16x8 bitwise methods =====

// Neg performs element-wise negation by flipping the sign bit.
func (v BFloat16x8) Neg() BFloat16x8 {
	var result BFloat16x8
	u := (*[8]uint16)(unsafe.Pointer(&v))
	r := (*[8]uint16)(unsafe.Pointer(&result))
	for i := range 8 {
		r[i] = u[i] ^ 0x8000
	}
	return result
}

// Abs performs element-wise absolute value by clearing the sign bit.
func (v BFloat16x8) Abs() BFloat16x8 {
	var result BFloat16x8
	u := (*[8]uint16)(unsafe.Pointer(&v))
	r := (*[8]uint16)(unsafe.Pointer(&result))
	for i := range 8 {
		r[i] = u[i] & 0x7FFF
	}
	return result
}

// Not performs bitwise NOT on the vector bytes.
func (v BFloat16x8) Not() BFloat16x8 {
	var result BFloat16x8
	for i := range 16 {
		result[i] = ^v[i]
	}
	return result
}

// Xor performs bitwise XOR with another vector.
func (v BFloat16x8) Xor(other BFloat16x8) BFloat16x8 {
	var result BFloat16x8
	for i := range 16 {
		result[i] = v[i] ^ other[i]
	}
	return result
}

// And performs bitwise AND with another vector.
func (v BFloat16x8) And(other BFloat16x8) BFloat16x8 {
	var result BFloat16x8
	for i := range 16 {
		result[i] = v[i] & other[i]
	}
	return result
}

// ===== BFloat16x8 return-value methods =====

// Add performs element-wise addition (via F32).
func (v BFloat16x8) Add(other BFloat16x8) BFloat16x8 {
	return BFloat16x8(add_bf16x8([16]byte(v), [16]byte(other)))
}

// Sub performs element-wise subtraction (via F32).
func (v BFloat16x8) Sub(other BFloat16x8) BFloat16x8 {
	return BFloat16x8(sub_bf16x8([16]byte(v), [16]byte(other)))
}

// Mul performs element-wise multiplication (via F32).
func (v BFloat16x8) Mul(other BFloat16x8) BFloat16x8 {
	return BFloat16x8(mul_bf16x8([16]byte(v), [16]byte(other)))
}

// Div performs element-wise division (via F32).
func (v BFloat16x8) Div(other BFloat16x8) BFloat16x8 {
	return BFloat16x8(div_bf16x8([16]byte(v), [16]byte(other)))
}

// MulAdd performs fused multiply-add: v * a + b (via F32 FMA).
func (v BFloat16x8) MulAdd(a, b BFloat16x8) BFloat16x8 {
	return BFloat16x8(fma_bf16x8([16]byte(v), [16]byte(a), [16]byte(b)))
}

// Min performs element-wise minimum (via F32 promote, compare, demote).
func (v BFloat16x8) Min(other BFloat16x8) BFloat16x8 {
	var af, bf, rf [8]float32
	au := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	bu := unsafe.Slice((*uint16)(unsafe.Pointer(&other)), 8)
	PromoteBF16ToF32NEON(au, af[:])
	PromoteBF16ToF32NEON(bu, bf[:])
	for i := range 8 {
		if af[i] < bf[i] {
			rf[i] = af[i]
		} else {
			rf[i] = bf[i]
		}
	}
	var result BFloat16x8
	ru := unsafe.Slice((*uint16)(unsafe.Pointer(&result)), 8)
	DemoteF32ToBF16NEON(rf[:], ru)
	return result
}

// Max performs element-wise maximum (via F32 promote, compare, demote).
func (v BFloat16x8) Max(other BFloat16x8) BFloat16x8 {
	var af, bf, rf [8]float32
	au := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	bu := unsafe.Slice((*uint16)(unsafe.Pointer(&other)), 8)
	PromoteBF16ToF32NEON(au, af[:])
	PromoteBF16ToF32NEON(bu, bf[:])
	for i := range 8 {
		if af[i] > bf[i] {
			rf[i] = af[i]
		} else {
			rf[i] = bf[i]
		}
	}
	var result BFloat16x8
	ru := unsafe.Slice((*uint16)(unsafe.Pointer(&result)), 8)
	DemoteF32ToBF16NEON(rf[:], ru)
	return result
}

// ===== BFloat16x8 in-place methods (allocation-free) =====

// AddInto performs element-wise addition, storing result in *result.
func (v BFloat16x8) AddInto(other BFloat16x8, result *BFloat16x8) {
	add_bf16x8_ip([16]byte(v), [16]byte(other), unsafe.Pointer(result))
}

// SubInto performs element-wise subtraction, storing result in *result.
func (v BFloat16x8) SubInto(other BFloat16x8, result *BFloat16x8) {
	sub_bf16x8_ip([16]byte(v), [16]byte(other), unsafe.Pointer(result))
}

// MulInto performs element-wise multiplication, storing result in *result.
func (v BFloat16x8) MulInto(other BFloat16x8, result *BFloat16x8) {
	mul_bf16x8_ip([16]byte(v), [16]byte(other), unsafe.Pointer(result))
}

// DivInto performs element-wise division, storing result in *result.
func (v BFloat16x8) DivInto(other BFloat16x8, result *BFloat16x8) {
	div_bf16x8_ip([16]byte(v), [16]byte(other), unsafe.Pointer(result))
}

// MulAddAcc performs fused multiply-add accumulation: *acc = v * a + *acc.
// Note: For ML workloads, prefer BFDotF32Acc with F32 accumulator for better precision.
func (v BFloat16x8) MulAddAcc(a BFloat16x8, acc *BFloat16x8) {
	muladd_bf16x8_acc([16]byte(v), [16]byte(a), unsafe.Pointer(acc))
}

// MulAddInto performs fused multiply-add: *result = v * a + b.
func (v BFloat16x8) MulAddInto(a, b BFloat16x8, result *BFloat16x8) {
	muladd_bf16x8_ip([16]byte(v), [16]byte(a), [16]byte(b), unsafe.Pointer(result))
}

// ===== BFloat16x8 interleave =====

// InterleaveLower interleaves the lower halves of two vectors.
// [a0,a1,a2,a3,a4,a5,a6,a7], [b0,b1,b2,b3,b4,b5,b6,b7] -> [a0,b0,a1,b1,a2,b2,a3,b3]
func (v BFloat16x8) InterleaveLower(other BFloat16x8) BFloat16x8 {
	a := (*[8]uint16)(unsafe.Pointer(&v))
	b := (*[8]uint16)(unsafe.Pointer(&other))
	var result BFloat16x8
	r := (*[8]uint16)(unsafe.Pointer(&result))
	r[0] = a[0]
	r[1] = b[0]
	r[2] = a[1]
	r[3] = b[1]
	r[4] = a[2]
	r[5] = b[2]
	r[6] = a[3]
	r[7] = b[3]
	return result
}

// InterleaveUpper interleaves the upper halves of two vectors.
// [a0,a1,a2,a3,a4,a5,a6,a7], [b0,b1,b2,b3,b4,b5,b6,b7] -> [a4,b4,a5,b5,a6,b6,a7,b7]
func (v BFloat16x8) InterleaveUpper(other BFloat16x8) BFloat16x8 {
	a := (*[8]uint16)(unsafe.Pointer(&v))
	b := (*[8]uint16)(unsafe.Pointer(&other))
	var result BFloat16x8
	r := (*[8]uint16)(unsafe.Pointer(&result))
	r[0] = a[4]
	r[1] = b[4]
	r[2] = a[5]
	r[3] = b[5]
	r[4] = a[6]
	r[5] = b[6]
	r[6] = a[7]
	r[7] = b[7]
	return result
}

// ===== BFloat16x8 reduction =====

// ReduceSum returns the sum of all 8 bfloat16 lanes as a float32.
// Promotes to F32 using NEON, then sums scalar values.
func (v BFloat16x8) ReduceSum() float32 {
	var f32 [8]float32
	u16 := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	PromoteBF16ToF32NEON(u16, f32[:])
	return f32[0] + f32[1] + f32[2] + f32[3] + f32[4] + f32[5] + f32[6] + f32[7]
}

// ===== BFloat16x8 comparison =====

// GreaterThan compares element-wise: result[i] = (v[i] > other[i]) ? 0xFFFF : 0x0000.
// Returns a Uint16x8 mask suitable for IfThenElseBFloat16.
func (v BFloat16x8) GreaterThan(other BFloat16x8) Uint16x8 {
	var f32a, f32b [8]float32
	u16a := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	u16b := unsafe.Slice((*uint16)(unsafe.Pointer(&other)), 8)
	PromoteBF16ToF32NEON(u16a, f32a[:])
	PromoteBF16ToF32NEON(u16b, f32b[:])
	var result Uint16x8
	r := (*[8]uint16)(unsafe.Pointer(&result))
	for i := range 8 {
		if f32a[i] > f32b[i] {
			r[i] = 0xFFFF
		}
	}
	return result
}

// LessThan compares element-wise: result[i] = (v[i] < other[i]) ? 0xFFFF : 0x0000.
func (v BFloat16x8) LessThan(other BFloat16x8) Uint16x8 {
	var f32a, f32b [8]float32
	u16a := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	u16b := unsafe.Slice((*uint16)(unsafe.Pointer(&other)), 8)
	PromoteBF16ToF32NEON(u16a, f32a[:])
	PromoteBF16ToF32NEON(u16b, f32b[:])
	var result Uint16x8
	r := (*[8]uint16)(unsafe.Pointer(&result))
	for i := range 8 {
		if f32a[i] < f32b[i] {
			r[i] = 0xFFFF
		}
	}
	return result
}

// GreaterThanOrEqual compares element-wise: result[i] = (v[i] >= other[i]) ? 0xFFFF : 0x0000.
func (v BFloat16x8) GreaterThanOrEqual(other BFloat16x8) Uint16x8 {
	var f32a, f32b [8]float32
	u16a := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	u16b := unsafe.Slice((*uint16)(unsafe.Pointer(&other)), 8)
	PromoteBF16ToF32NEON(u16a, f32a[:])
	PromoteBF16ToF32NEON(u16b, f32b[:])
	var result Uint16x8
	r := (*[8]uint16)(unsafe.Pointer(&result))
	for i := range 8 {
		if f32a[i] >= f32b[i] {
			r[i] = 0xFFFF
		}
	}
	return result
}

// LessThanOrEqual compares element-wise: result[i] = (v[i] <= other[i]) ? 0xFFFF : 0x0000.
func (v BFloat16x8) LessThanOrEqual(other BFloat16x8) Uint16x8 {
	var f32a, f32b [8]float32
	u16a := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	u16b := unsafe.Slice((*uint16)(unsafe.Pointer(&other)), 8)
	PromoteBF16ToF32NEON(u16a, f32a[:])
	PromoteBF16ToF32NEON(u16b, f32b[:])
	var result Uint16x8
	r := (*[8]uint16)(unsafe.Pointer(&result))
	for i := range 8 {
		if f32a[i] <= f32b[i] {
			r[i] = 0xFFFF
		}
	}
	return result
}

// NotEqual compares element-wise: result[i] = (v[i] != other[i]) ? 0xFFFF : 0x0000.
func (v BFloat16x8) NotEqual(other BFloat16x8) Uint16x8 {
	var f32a, f32b [8]float32
	u16a := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	u16b := unsafe.Slice((*uint16)(unsafe.Pointer(&other)), 8)
	PromoteBF16ToF32NEON(u16a, f32a[:])
	PromoteBF16ToF32NEON(u16b, f32b[:])
	var result Uint16x8
	r := (*[8]uint16)(unsafe.Pointer(&result))
	for i := range 8 {
		if f32a[i] != f32b[i] {
			r[i] = 0xFFFF
		}
	}
	return result
}

// ===== BFloat16x8 conditional select =====

// IotaBFloat16x8 returns a vector with lane indices [0, 1, 2, ..., 7] as bfloat16.
func IotaBFloat16x8() BFloat16x8 {
	var f32 [8]float32
	for i := range 8 {
		f32[i] = float32(i)
	}
	var u16 [8]uint16
	DemoteF32ToBF16NEON(f32[:], u16[:])
	return *(*BFloat16x8)(unsafe.Pointer(&u16[0]))
}

// IfThenElseBFloat16 selects elements based on mask: result = mask ? yes : no.
// mask lanes should be 0xFFFF (true) or 0x0000 (false).
func IfThenElseBFloat16(mask Uint16x8, yes, no BFloat16x8) BFloat16x8 {
	m := (*[8]uint16)(unsafe.Pointer(&mask))
	y := (*[8]uint16)(unsafe.Pointer(&yes))
	n := (*[8]uint16)(unsafe.Pointer(&no))
	var result BFloat16x8
	r := (*[8]uint16)(unsafe.Pointer(&result))
	for i := range 8 {
		if m[i] != 0 {
			r[i] = y[i]
		} else {
			r[i] = n[i]
		}
	}
	return result
}

// ===== BFloat16x8 horizontal reductions =====

// ReduceMax returns the maximum of all 8 bfloat16 lanes as a float32.
func (v BFloat16x8) ReduceMax() float32 {
	var f32 [8]float32
	u16 := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	PromoteBF16ToF32NEON(u16, f32[:])
	max := f32[0]
	for i := 1; i < 8; i++ {
		if f32[i] > max {
			max = f32[i]
		}
	}
	return max
}

// ReduceMin returns the minimum of all 8 bfloat16 lanes as a float32.
func (v BFloat16x8) ReduceMin() float32 {
	var f32 [8]float32
	u16 := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	PromoteBF16ToF32NEON(u16, f32[:])
	min := f32[0]
	for i := 1; i < 8; i++ {
		if f32[i] < min {
			min = f32[i]
		}
	}
	return min
}

// ===== BFloat16x8 Data access =====

// Data returns the 8 bfloat16 values as an array of uint16.
func (v BFloat16x8) Data() [8]uint16 {
	return *(*[8]uint16)(unsafe.Pointer(&v))
}

// ===== BFloat16x8 dot product with F32 accumulator (preferred for ML) =====

// BFDotF32Acc accumulates dot product into F32x4: *acc += dot(v, other).
// Uses BFDOT instruction which is optimal for ML workloads.
// Each F32 lane receives dot product of 2 BF16 pairs.
func (v BFloat16x8) BFDotF32Acc(other BFloat16x8, acc *Float32x4) {
	bfdot_bf16x8_f32x4_acc([16]byte(v), [16]byte(other), unsafe.Pointer(acc))
}

// SignBitBFloat16x8 returns a vector with the sign bit (0x8000) set in each lane.
func SignBitBFloat16x8() BFloat16x8 {
	return BroadcastBFloat16x8(0x8000)
}

// Assembly function declarations are in ops_bf16_neon_arm64.go (generated by GoAT)
