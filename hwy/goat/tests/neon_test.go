//go:build arm64

package tests

import (
	"math"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

// Helper to convert float32 slice to [16]byte for NEON
func f32x4(a, b, c, d float32) [16]byte {
	var result [16]byte
	*(*float32)(unsafe.Pointer(&result[0])) = a
	*(*float32)(unsafe.Pointer(&result[4])) = b
	*(*float32)(unsafe.Pointer(&result[8])) = c
	*(*float32)(unsafe.Pointer(&result[12])) = d
	return result
}

// Helper to convert [16]byte back to float32 slice
func toF32x4(v [16]byte) [4]float32 {
	return [4]float32{
		*(*float32)(unsafe.Pointer(&v[0])),
		*(*float32)(unsafe.Pointer(&v[4])),
		*(*float32)(unsafe.Pointer(&v[8])),
		*(*float32)(unsafe.Pointer(&v[12])),
	}
}

// Helper to convert int32 slice to [16]byte for NEON
func i32x4(a, b, c, d int32) [16]byte {
	var result [16]byte
	*(*int32)(unsafe.Pointer(&result[0])) = a
	*(*int32)(unsafe.Pointer(&result[4])) = b
	*(*int32)(unsafe.Pointer(&result[8])) = c
	*(*int32)(unsafe.Pointer(&result[12])) = d
	return result
}

// Helper to convert [16]byte back to int32 slice
func toI32x4(v [16]byte) [4]int32 {
	return [4]int32{
		*(*int32)(unsafe.Pointer(&v[0])),
		*(*int32)(unsafe.Pointer(&v[4])),
		*(*int32)(unsafe.Pointer(&v[8])),
		*(*int32)(unsafe.Pointer(&v[12])),
	}
}

func TestAddF32x4(t *testing.T) {
	a := f32x4(1, 2, 3, 4)
	b := f32x4(5, 6, 7, 8)
	result := add_f32x4(a, b)
	expected := [4]float32{6, 8, 10, 12}
	assert.Equal(t, expected, toF32x4(result))
}

func TestMulF32x4(t *testing.T) {
	a := f32x4(1, 2, 3, 4)
	b := f32x4(5, 6, 7, 8)
	result := mul_f32x4(a, b)
	expected := [4]float32{5, 12, 21, 32}
	assert.Equal(t, expected, toF32x4(result))
}

func TestFmaF32x4(t *testing.T) {
	a := f32x4(1, 2, 3, 4)   // multiplicand
	b := f32x4(2, 2, 2, 2)   // multiplier
	c := f32x4(10, 20, 30, 40) // addend
	result := fma_f32x4(a, b, c)
	// fma = a * b + c = (1*2+10, 2*2+20, 3*2+30, 4*2+40) = (12, 24, 36, 48)
	expected := [4]float32{12, 24, 36, 48}
	assert.Equal(t, expected, toF32x4(result))
}

func TestAddI32x4(t *testing.T) {
	a := i32x4(1, -2, 3, -4)
	b := i32x4(5, 6, -7, 8)
	result := add_i32x4(a, b)
	expected := [4]int32{6, 4, -4, 4}
	assert.Equal(t, expected, toI32x4(result))
}

func TestHsumF32x4(t *testing.T) {
	v := f32x4(1, 2, 3, 4)
	result := hsum_f32x4(v)
	assert.Equal(t, float32(10), result)
}

func TestDotF32x4(t *testing.T) {
	a := f32x4(1, 2, 3, 4)
	b := f32x4(5, 6, 7, 8)
	result := dot_f32x4(a, b)
	// dot = 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
	assert.Equal(t, float32(70), result)
}

func TestAddF32x4_NegativeZero(t *testing.T) {
	// Test handling of special float values
	negZero := float32(math.Copysign(0, -1))
	a := f32x4(negZero, 0, 1, -1)
	b := f32x4(0, negZero, -1, 1)
	result := add_f32x4(a, b)
	got := toF32x4(result)
	assert.Equal(t, float32(0), got[0])
	assert.Equal(t, float32(0), got[1])
	assert.Equal(t, float32(0), got[2])
	assert.Equal(t, float32(0), got[3])
}

// ============ 64-bit NEON type tests ============

// Helper to convert float32 pair to [8]byte for 64-bit NEON
func f32x2(a, b float32) [8]byte {
	var result [8]byte
	*(*float32)(unsafe.Pointer(&result[0])) = a
	*(*float32)(unsafe.Pointer(&result[4])) = b
	return result
}

// Helper to convert [8]byte back to float32 pair
func toF32x2(v [8]byte) [2]float32 {
	return [2]float32{
		*(*float32)(unsafe.Pointer(&v[0])),
		*(*float32)(unsafe.Pointer(&v[4])),
	}
}

// Helper to convert int32 pair to [8]byte for 64-bit NEON
func i32x2(a, b int32) [8]byte {
	var result [8]byte
	*(*int32)(unsafe.Pointer(&result[0])) = a
	*(*int32)(unsafe.Pointer(&result[4])) = b
	return result
}

// Helper to convert [8]byte back to int32 pair
func toI32x2(v [8]byte) [2]int32 {
	return [2]int32{
		*(*int32)(unsafe.Pointer(&v[0])),
		*(*int32)(unsafe.Pointer(&v[4])),
	}
}

func TestAddF32x2(t *testing.T) {
	a := f32x2(1, 2)
	b := f32x2(3, 4)
	result := add_f32x2(a, b)
	expected := [2]float32{4, 6}
	assert.Equal(t, expected, toF32x2(result))
}

func TestMulF32x2(t *testing.T) {
	a := f32x2(2, 3)
	b := f32x2(4, 5)
	result := mul_f32x2(a, b)
	expected := [2]float32{8, 15}
	assert.Equal(t, expected, toF32x2(result))
}

func TestAddI32x2(t *testing.T) {
	a := i32x2(10, -20)
	b := i32x2(5, 30)
	result := add_i32x2(a, b)
	expected := [2]int32{15, 10}
	assert.Equal(t, expected, toI32x2(result))
}

func TestHsumF32x2(t *testing.T) {
	v := f32x2(3, 7)
	result := hsum_f32x2(v)
	assert.Equal(t, float32(10), result)
}

// ============ NEON array type tests (x2) ============

// Helper to create [32]byte for float32x4x2 from two [16]byte vectors
func f32x4x2(v0, v1 [16]byte) [32]byte {
	var result [32]byte
	copy(result[0:16], v0[:])
	copy(result[16:32], v1[:])
	return result
}

// Helper to extract two [16]byte vectors from [32]byte
func toF32x4Pair(v [32]byte) ([4]float32, [4]float32) {
	var v0, v1 [16]byte
	copy(v0[:], v[0:16])
	copy(v1[:], v[16:32])
	return toF32x4(v0), toF32x4(v1)
}

func TestAddF32x4x2(t *testing.T) {
	a0 := f32x4(1, 2, 3, 4)
	a1 := f32x4(5, 6, 7, 8)
	b0 := f32x4(10, 20, 30, 40)
	b1 := f32x4(50, 60, 70, 80)

	a := f32x4x2(a0, a1)
	b := f32x4x2(b0, b1)

	result := add_f32x4x2(a, b)
	r0, r1 := toF32x4Pair(result)

	assert.Equal(t, [4]float32{11, 22, 33, 44}, r0)
	assert.Equal(t, [4]float32{55, 66, 77, 88}, r1)
}

func TestMakeF32x4x2(t *testing.T) {
	a := f32x4(1, 2, 3, 4)
	b := f32x4(5, 6, 7, 8)

	result := make_f32x4x2(a, b)
	r0, r1 := toF32x4Pair(result)

	assert.Equal(t, [4]float32{1, 2, 3, 4}, r0)
	assert.Equal(t, [4]float32{5, 6, 7, 8}, r1)
}

func TestSumF32x4x2(t *testing.T) {
	v0 := f32x4(1, 2, 3, 4)   // sum = 10
	v1 := f32x4(5, 6, 7, 8)   // sum = 26
	v := f32x4x2(v0, v1)

	result := sum_f32x4x2(v)
	assert.Equal(t, float32(36), result) // 10 + 26 = 36
}
