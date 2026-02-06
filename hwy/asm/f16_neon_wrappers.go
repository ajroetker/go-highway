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

// Float16 NEON operations for ARM64 with FP16 extension (ARMv8.2-A+)
package asm

import "unsafe"

// -march=armv8.2-a+fp16 enables native float16 arithmetic instructions
//go:generate go tool goat ../c/ops_f16_neon_arm64.c -O3 --target arm64 -e="-march=armv8.2-a+fp16"

// ============================================================================
// Float16 Conversions
// ============================================================================

// PromoteF16ToF32NEON converts float16 to float32 using NEON vcvt_f32_f16.
func PromoteF16ToF32NEON(a []uint16, result []float32) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), len(result)))
	promote_f16_to_f32_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// DemoteF32ToF16NEON converts float32 to float16 using NEON vcvt_f16_f32.
func DemoteF32ToF16NEON(a []float32, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), len(result)))
	demote_f32_to_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ============================================================================
// Native Float16 Arithmetic (requires ARMv8.2-A+fp16)
// ============================================================================

// AddF16NEON performs element-wise addition: result[i] = a[i] + b[i]
func AddF16NEON(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	add_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SubF16NEON performs element-wise subtraction: result[i] = a[i] - b[i]
func SubF16NEON(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	sub_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MulF16NEON performs element-wise multiplication: result[i] = a[i] * b[i]
func MulF16NEON(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	mul_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// DivF16NEON performs element-wise division: result[i] = a[i] / b[i]
func DivF16NEON(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	div_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// FmaF16NEON performs fused multiply-add: result[i] = a[i] * b[i] + c[i]
func FmaF16NEON(a, b, c, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), min(len(c), len(result)))))
	fma_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// NegF16NEON performs element-wise negation: result[i] = -a[i]
func NegF16NEON(a, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), len(result)))
	neg_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// AbsF16NEON performs element-wise absolute value: result[i] = abs(a[i])
func AbsF16NEON(a, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), len(result)))
	abs_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MinF16NEON performs element-wise minimum: result[i] = min(a[i], b[i])
func MinF16NEON(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	min_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// MaxF16NEON performs element-wise maximum: result[i] = max(a[i], b[i])
func MaxF16NEON(a, b, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), min(len(b), len(result))))
	max_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// SqrtF16NEON performs element-wise square root: result[i] = sqrt(a[i])
func SqrtF16NEON(a, result []uint16) {
	if len(a) == 0 {
		return
	}
	n := int64(min(len(a), len(result)))
	sqrt_f16_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&result[0]), unsafe.Pointer(&n))
}

// ============================================================================
// Float16 Vector Load/Store Operations
// ============================================================================

// Load4F16x8NEON loads 4 consecutive float16x8 vectors (32 float16 values = 64 bytes).
// Uses vld1q_f16_x4 which loads 64 bytes in a single instruction.
// Returns 4 vectors as [16]byte (each holding 8 float16 values).
func Load4F16x8NEON(ptr []uint16) (v0, v1, v2, v3 [16]byte) {
	if len(ptr) < 32 {
		return
	}
	load4_f16x8(unsafe.Pointer(&ptr[0]),
		unsafe.Pointer(&v0[0]), unsafe.Pointer(&v1[0]),
		unsafe.Pointer(&v2[0]), unsafe.Pointer(&v3[0]))
	return
}

// Store4F16x8NEON stores 4 consecutive float16x8 vectors (32 float16 values = 64 bytes).
// Uses vst1q_f16_x4 which stores 64 bytes in a single instruction.
func Store4F16x8NEON(ptr []uint16, v0, v1, v2, v3 [16]byte) {
	if len(ptr) < 32 {
		return
	}
	store4_f16x8(unsafe.Pointer(&ptr[0]), v0, v1, v2, v3)
}

// Load4Float16x8 loads 4 consecutive Float16x8 vectors from an unsafe.Pointer.
// ptr must point to at least 32 float16 values (64 bytes).
func Load4Float16x8(ptr unsafe.Pointer) (Float16x8, Float16x8, Float16x8, Float16x8) {
	u16 := unsafe.Slice((*uint16)(ptr), 32)
	v0, v1, v2, v3 := Load4F16x8NEON(u16)
	return Float16x8(v0), Float16x8(v1), Float16x8(v2), Float16x8(v3)
}

// ============================================================================
// Float16x8 Single-Vector Type and Operations
// ============================================================================
// Float16x8 represents a 128-bit NEON vector of 8 float16 values.
// Uses [16]byte backing for efficient register passing via GoAT-generated assembly.

// Float16x8 represents a 128-bit NEON vector of 8 float16 values.
type Float16x8 [16]byte

// ZeroFloat16x8 returns a zero vector.
func ZeroFloat16x8() Float16x8 {
	return Float16x8{}
}

// BroadcastFloat16x8 broadcasts a scalar float16 to all 8 lanes.
func BroadcastFloat16x8(val uint16) Float16x8 {
	return Float16x8(broadcast_f16x8(unsafe.Pointer(&val)))
}

// LoadFloat16x8 loads 8 float16 values from a slice (has bounds check).
func LoadFloat16x8(s []uint16) Float16x8 {
	return *(*Float16x8)(unsafe.Pointer(&s[0]))
}

// LoadFloat16x8Slice loads 8 float16 values from a slice (alias for LoadFloat16x8).
func LoadFloat16x8Slice(s []uint16) Float16x8 {
	return *(*Float16x8)(unsafe.Pointer(&s[0]))
}

// LoadFloat16x8Ptr loads 8 float16 values from an unsafe.Pointer (no bounds check).
// Use this when you have a slice of hwy.Float16 and need to avoid type conversion.
func LoadFloat16x8Ptr(ptr unsafe.Pointer) Float16x8 {
	return *(*Float16x8)(ptr)
}

// Store stores the vector to a slice (has bounds check).
func (v Float16x8) Store(s []uint16) {
	*(*Float16x8)(unsafe.Pointer(&s[0])) = v
}

// StoreSlice stores the vector to a slice (alias for Store).
func (v Float16x8) StoreSlice(s []uint16) {
	*(*Float16x8)(unsafe.Pointer(&s[0])) = v
}

// StorePtr stores the vector to an unsafe.Pointer (no bounds check).
// Use this when you have a slice of hwy.Float16 and need to avoid type conversion.
func (v Float16x8) StorePtr(ptr unsafe.Pointer) {
	*(*Float16x8)(ptr) = v
}

// ===== Float16x8 return-value methods =====

// Add performs element-wise addition.
func (v Float16x8) Add(other Float16x8) Float16x8 {
	return Float16x8(add_f16x8([16]byte(v), [16]byte(other)))
}

// Sub performs element-wise subtraction.
func (v Float16x8) Sub(other Float16x8) Float16x8 {
	return Float16x8(sub_f16x8([16]byte(v), [16]byte(other)))
}

// Mul performs element-wise multiplication.
func (v Float16x8) Mul(other Float16x8) Float16x8 {
	return Float16x8(mul_f16x8([16]byte(v), [16]byte(other)))
}

// Div performs element-wise division.
func (v Float16x8) Div(other Float16x8) Float16x8 {
	return Float16x8(div_f16x8([16]byte(v), [16]byte(other)))
}

// Min performs element-wise minimum.
func (v Float16x8) Min(other Float16x8) Float16x8 {
	return Float16x8(min_f16x8([16]byte(v), [16]byte(other)))
}

// Max performs element-wise maximum.
func (v Float16x8) Max(other Float16x8) Float16x8 {
	return Float16x8(max_f16x8([16]byte(v), [16]byte(other)))
}

// Abs performs element-wise absolute value.
func (v Float16x8) Abs() Float16x8 {
	return Float16x8(abs_f16x8([16]byte(v)))
}

// Neg performs element-wise negation.
func (v Float16x8) Neg() Float16x8 {
	return Float16x8(neg_f16x8([16]byte(v)))
}

// Sqrt performs element-wise square root.
func (v Float16x8) Sqrt() Float16x8 {
	return Float16x8(sqrt_f16x8([16]byte(v)))
}

// MulAdd performs fused multiply-add: v * a + b
func (v Float16x8) MulAdd(a, b Float16x8) Float16x8 {
	return Float16x8(fma_f16x8([16]byte(v), [16]byte(a), [16]byte(b)))
}

// MulSub performs fused multiply-subtract: v * a - b
func (v Float16x8) MulSub(a, b Float16x8) Float16x8 {
	return Float16x8(fms_f16x8([16]byte(v), [16]byte(a), [16]byte(b)))
}

// ===== Float16x8 bitwise methods =====

// Not performs bitwise NOT on the vector bytes.
func (v Float16x8) Not() Float16x8 {
	var result Float16x8
	for i := range 16 {
		result[i] = ^v[i]
	}
	return result
}

// Xor performs bitwise XOR with another vector.
func (v Float16x8) Xor(other Float16x8) Float16x8 {
	var result Float16x8
	for i := range 16 {
		result[i] = v[i] ^ other[i]
	}
	return result
}

// And performs bitwise AND with another vector.
func (v Float16x8) And(other Float16x8) Float16x8 {
	var result Float16x8
	for i := range 16 {
		result[i] = v[i] & other[i]
	}
	return result
}

// ===== Float16x8 in-place methods (allocation-free) =====

// AddInto performs element-wise addition, storing result in *result.
func (v Float16x8) AddInto(other Float16x8, result *Float16x8) {
	add_f16x8_ip([16]byte(v), [16]byte(other), unsafe.Pointer(result))
}

// SubInto performs element-wise subtraction, storing result in *result.
func (v Float16x8) SubInto(other Float16x8, result *Float16x8) {
	sub_f16x8_ip([16]byte(v), [16]byte(other), unsafe.Pointer(result))
}

// MulInto performs element-wise multiplication, storing result in *result.
func (v Float16x8) MulInto(other Float16x8, result *Float16x8) {
	mul_f16x8_ip([16]byte(v), [16]byte(other), unsafe.Pointer(result))
}

// DivInto performs element-wise division, storing result in *result.
func (v Float16x8) DivInto(other Float16x8, result *Float16x8) {
	div_f16x8_ip([16]byte(v), [16]byte(other), unsafe.Pointer(result))
}

// MinInto performs element-wise minimum, storing result in *result.
func (v Float16x8) MinInto(other Float16x8, result *Float16x8) {
	min_f16x8_ip([16]byte(v), [16]byte(other), unsafe.Pointer(result))
}

// MaxInto performs element-wise maximum, storing result in *result.
func (v Float16x8) MaxInto(other Float16x8, result *Float16x8) {
	max_f16x8_ip([16]byte(v), [16]byte(other), unsafe.Pointer(result))
}

// MulAddAcc performs fused multiply-add accumulation: *acc = v * a + *acc.
// This is the most efficient pattern for inner loops (no return allocation).
func (v Float16x8) MulAddAcc(a Float16x8, acc *Float16x8) {
	muladd_f16x8_acc([16]byte(v), [16]byte(a), unsafe.Pointer(acc))
}

// MulAddInto performs fused multiply-add: *result = v * a + b.
func (v Float16x8) MulAddInto(a, b Float16x8, result *Float16x8) {
	muladd_f16x8_ip([16]byte(v), [16]byte(a), [16]byte(b), unsafe.Pointer(result))
}

// ===== Float16x8 interleave =====

// InterleaveLower interleaves the lower halves of two vectors.
// [a0,a1,a2,a3,a4,a5,a6,a7], [b0,b1,b2,b3,b4,b5,b6,b7] -> [a0,b0,a1,b1,a2,b2,a3,b3]
func (v Float16x8) InterleaveLower(other Float16x8) Float16x8 {
	a := (*[8]uint16)(unsafe.Pointer(&v))
	b := (*[8]uint16)(unsafe.Pointer(&other))
	var result Float16x8
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
func (v Float16x8) InterleaveUpper(other Float16x8) Float16x8 {
	a := (*[8]uint16)(unsafe.Pointer(&v))
	b := (*[8]uint16)(unsafe.Pointer(&other))
	var result Float16x8
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

// ===== Float16x8 reduction =====

// ReduceSum returns the sum of all 8 float16 lanes as a float32.
// Promotes to F32 using NEON, then sums scalar values.
func (v Float16x8) ReduceSum() float32 {
	var f32 [8]float32
	u16 := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	PromoteF16ToF32NEON(u16, f32[:])
	return f32[0] + f32[1] + f32[2] + f32[3] + f32[4] + f32[5] + f32[6] + f32[7]
}

// ===== Float16x8 comparison =====

// GreaterThan compares element-wise: result[i] = (v[i] > other[i]) ? 0xFFFF : 0x0000.
// Returns a Uint16x8 mask suitable for IfThenElseFloat16.
func (v Float16x8) GreaterThan(other Float16x8) Uint16x8 {
	var f32a, f32b [8]float32
	u16a := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	u16b := unsafe.Slice((*uint16)(unsafe.Pointer(&other)), 8)
	PromoteF16ToF32NEON(u16a, f32a[:])
	PromoteF16ToF32NEON(u16b, f32b[:])
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
func (v Float16x8) LessThan(other Float16x8) Uint16x8 {
	var f32a, f32b [8]float32
	u16a := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	u16b := unsafe.Slice((*uint16)(unsafe.Pointer(&other)), 8)
	PromoteF16ToF32NEON(u16a, f32a[:])
	PromoteF16ToF32NEON(u16b, f32b[:])
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
func (v Float16x8) GreaterThanOrEqual(other Float16x8) Uint16x8 {
	var f32a, f32b [8]float32
	u16a := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	u16b := unsafe.Slice((*uint16)(unsafe.Pointer(&other)), 8)
	PromoteF16ToF32NEON(u16a, f32a[:])
	PromoteF16ToF32NEON(u16b, f32b[:])
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
func (v Float16x8) LessThanOrEqual(other Float16x8) Uint16x8 {
	var f32a, f32b [8]float32
	u16a := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	u16b := unsafe.Slice((*uint16)(unsafe.Pointer(&other)), 8)
	PromoteF16ToF32NEON(u16a, f32a[:])
	PromoteF16ToF32NEON(u16b, f32b[:])
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
func (v Float16x8) NotEqual(other Float16x8) Uint16x8 {
	var f32a, f32b [8]float32
	u16a := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	u16b := unsafe.Slice((*uint16)(unsafe.Pointer(&other)), 8)
	PromoteF16ToF32NEON(u16a, f32a[:])
	PromoteF16ToF32NEON(u16b, f32b[:])
	var result Uint16x8
	r := (*[8]uint16)(unsafe.Pointer(&result))
	for i := range 8 {
		if f32a[i] != f32b[i] {
			r[i] = 0xFFFF
		}
	}
	return result
}

// ===== Float16x8 conditional select =====

// IotaFloat16x8 returns a vector with lane indices [0, 1, 2, ..., 7] as float16.
func IotaFloat16x8() Float16x8 {
	// Convert integer indices to float16 via float32
	var f32 [8]float32
	for i := range 8 {
		f32[i] = float32(i)
	}
	var u16 [8]uint16
	DemoteF32ToF16NEON(f32[:], u16[:])
	return *(*Float16x8)(unsafe.Pointer(&u16[0]))
}

// IfThenElseFloat16 selects elements based on mask: result = mask ? yes : no.
// mask lanes should be 0xFFFF (true) or 0x0000 (false).
func IfThenElseFloat16(mask Uint16x8, yes, no Float16x8) Float16x8 {
	m := (*[8]uint16)(unsafe.Pointer(&mask))
	y := (*[8]uint16)(unsafe.Pointer(&yes))
	n := (*[8]uint16)(unsafe.Pointer(&no))
	var result Float16x8
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

// ===== Float16x8 horizontal reductions =====

// ReduceMax returns the maximum of all 8 float16 lanes as a float32.
func (v Float16x8) ReduceMax() float32 {
	var f32 [8]float32
	u16 := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	PromoteF16ToF32NEON(u16, f32[:])
	max := f32[0]
	for i := 1; i < 8; i++ {
		if f32[i] > max {
			max = f32[i]
		}
	}
	return max
}

// ReduceMin returns the minimum of all 8 float16 lanes as a float32.
func (v Float16x8) ReduceMin() float32 {
	var f32 [8]float32
	u16 := unsafe.Slice((*uint16)(unsafe.Pointer(&v)), 8)
	PromoteF16ToF32NEON(u16, f32[:])
	min := f32[0]
	for i := 1; i < 8; i++ {
		if f32[i] < min {
			min = f32[i]
		}
	}
	return min
}

// ===== Float16x8 Data access =====

// Data returns the 8 float16 values as a slice of uint16.
func (v Float16x8) Data() [8]uint16 {
	return *(*[8]uint16)(unsafe.Pointer(&v))
}

// SignBitFloat16x8 returns a vector with the sign bit (0x8000) set in each lane.
func SignBitFloat16x8() Float16x8 {
	return BroadcastFloat16x8(0x8000)
}

// Assembly function declarations are in ops_f16_neon_arm64.go (generated by GoAT)
