//go:build amd64

package tests

import (
	"math"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

// ============ SSE 128-bit helpers ============

// Helper to create [16]byte from 4 float32s (SSE __m128)
func m128ps(a, b, c, d float32) [16]byte {
	var result [16]byte
	*(*float32)(unsafe.Pointer(&result[0])) = a
	*(*float32)(unsafe.Pointer(&result[4])) = b
	*(*float32)(unsafe.Pointer(&result[8])) = c
	*(*float32)(unsafe.Pointer(&result[12])) = d
	return result
}

// Helper to convert [16]byte back to [4]float32
func toM128ps(v [16]byte) [4]float32 {
	return [4]float32{
		*(*float32)(unsafe.Pointer(&v[0])),
		*(*float32)(unsafe.Pointer(&v[4])),
		*(*float32)(unsafe.Pointer(&v[8])),
		*(*float32)(unsafe.Pointer(&v[12])),
	}
}

// Helper to create [16]byte from 2 float64s (SSE __m128d)
func m128pd(a, b float64) [16]byte {
	var result [16]byte
	*(*float64)(unsafe.Pointer(&result[0])) = a
	*(*float64)(unsafe.Pointer(&result[8])) = b
	return result
}

// Helper to convert [16]byte back to [2]float64
func toM128pd(v [16]byte) [2]float64 {
	return [2]float64{
		*(*float64)(unsafe.Pointer(&v[0])),
		*(*float64)(unsafe.Pointer(&v[8])),
	}
}

// Helper to create [16]byte from 4 int32s (SSE __m128i)
func m128i32(a, b, c, d int32) [16]byte {
	var result [16]byte
	*(*int32)(unsafe.Pointer(&result[0])) = a
	*(*int32)(unsafe.Pointer(&result[4])) = b
	*(*int32)(unsafe.Pointer(&result[8])) = c
	*(*int32)(unsafe.Pointer(&result[12])) = d
	return result
}

// Helper to convert [16]byte back to [4]int32
func toM128i32(v [16]byte) [4]int32 {
	return [4]int32{
		*(*int32)(unsafe.Pointer(&v[0])),
		*(*int32)(unsafe.Pointer(&v[4])),
		*(*int32)(unsafe.Pointer(&v[8])),
		*(*int32)(unsafe.Pointer(&v[12])),
	}
}

// ============ AVX 256-bit helpers ============

// Helper to create [32]byte from 8 float32s (AVX __m256)
func m256ps(a, b, c, d, e, f, g, h float32) [32]byte {
	var result [32]byte
	*(*float32)(unsafe.Pointer(&result[0])) = a
	*(*float32)(unsafe.Pointer(&result[4])) = b
	*(*float32)(unsafe.Pointer(&result[8])) = c
	*(*float32)(unsafe.Pointer(&result[12])) = d
	*(*float32)(unsafe.Pointer(&result[16])) = e
	*(*float32)(unsafe.Pointer(&result[20])) = f
	*(*float32)(unsafe.Pointer(&result[24])) = g
	*(*float32)(unsafe.Pointer(&result[28])) = h
	return result
}

// Helper to convert [32]byte back to [8]float32
func toM256ps(v [32]byte) [8]float32 {
	return [8]float32{
		*(*float32)(unsafe.Pointer(&v[0])),
		*(*float32)(unsafe.Pointer(&v[4])),
		*(*float32)(unsafe.Pointer(&v[8])),
		*(*float32)(unsafe.Pointer(&v[12])),
		*(*float32)(unsafe.Pointer(&v[16])),
		*(*float32)(unsafe.Pointer(&v[20])),
		*(*float32)(unsafe.Pointer(&v[24])),
		*(*float32)(unsafe.Pointer(&v[28])),
	}
}

// Helper to create [32]byte from 4 float64s (AVX __m256d)
func m256pd(a, b, c, d float64) [32]byte {
	var result [32]byte
	*(*float64)(unsafe.Pointer(&result[0])) = a
	*(*float64)(unsafe.Pointer(&result[8])) = b
	*(*float64)(unsafe.Pointer(&result[16])) = c
	*(*float64)(unsafe.Pointer(&result[24])) = d
	return result
}

// Helper to convert [32]byte back to [4]float64
func toM256pd(v [32]byte) [4]float64 {
	return [4]float64{
		*(*float64)(unsafe.Pointer(&v[0])),
		*(*float64)(unsafe.Pointer(&v[8])),
		*(*float64)(unsafe.Pointer(&v[16])),
		*(*float64)(unsafe.Pointer(&v[24])),
	}
}

// Helper to create [32]byte from 8 int32s (AVX __m256i)
func m256i32(a, b, c, d, e, f, g, h int32) [32]byte {
	var result [32]byte
	*(*int32)(unsafe.Pointer(&result[0])) = a
	*(*int32)(unsafe.Pointer(&result[4])) = b
	*(*int32)(unsafe.Pointer(&result[8])) = c
	*(*int32)(unsafe.Pointer(&result[12])) = d
	*(*int32)(unsafe.Pointer(&result[16])) = e
	*(*int32)(unsafe.Pointer(&result[20])) = f
	*(*int32)(unsafe.Pointer(&result[24])) = g
	*(*int32)(unsafe.Pointer(&result[28])) = h
	return result
}

// Helper to convert [32]byte back to [8]int32
func toM256i32(v [32]byte) [8]int32 {
	return [8]int32{
		*(*int32)(unsafe.Pointer(&v[0])),
		*(*int32)(unsafe.Pointer(&v[4])),
		*(*int32)(unsafe.Pointer(&v[8])),
		*(*int32)(unsafe.Pointer(&v[12])),
		*(*int32)(unsafe.Pointer(&v[16])),
		*(*int32)(unsafe.Pointer(&v[20])),
		*(*int32)(unsafe.Pointer(&v[24])),
		*(*int32)(unsafe.Pointer(&v[28])),
	}
}

// ============ SSE 128-bit tests ============

func TestAddPs(t *testing.T) {
	a := m128ps(1, 2, 3, 4)
	b := m128ps(5, 6, 7, 8)
	result := add_ps(a, b)
	expected := [4]float32{6, 8, 10, 12}
	assert.Equal(t, expected, toM128ps(result))
}

func TestMulPs(t *testing.T) {
	a := m128ps(1, 2, 3, 4)
	b := m128ps(5, 6, 7, 8)
	result := mul_ps(a, b)
	expected := [4]float32{5, 12, 21, 32}
	assert.Equal(t, expected, toM128ps(result))
}

func TestFmaPs(t *testing.T) {
	a := m128ps(1, 2, 3, 4)   // multiplicand
	b := m128ps(2, 2, 2, 2)   // multiplier
	c := m128ps(10, 20, 30, 40) // addend
	result := fma_ps(a, b, c)
	// fma = a * b + c = (1*2+10, 2*2+20, 3*2+30, 4*2+40) = (12, 24, 36, 48)
	expected := [4]float32{12, 24, 36, 48}
	assert.Equal(t, expected, toM128ps(result))
}

func TestAddPd(t *testing.T) {
	a := m128pd(1.5, 2.5)
	b := m128pd(3.5, 4.5)
	result := add_pd(a, b)
	expected := [2]float64{5.0, 7.0}
	assert.Equal(t, expected, toM128pd(result))
}

func TestMulPd(t *testing.T) {
	a := m128pd(2.0, 3.0)
	b := m128pd(4.0, 5.0)
	result := mul_pd(a, b)
	expected := [2]float64{8.0, 15.0}
	assert.Equal(t, expected, toM128pd(result))
}

func TestAddEpi32(t *testing.T) {
	a := m128i32(1, -2, 3, -4)
	b := m128i32(5, 6, -7, 8)
	result := add_epi32(a, b)
	expected := [4]int32{6, 4, -4, 4}
	assert.Equal(t, expected, toM128i32(result))
}

func TestHsumPs(t *testing.T) {
	v := m128ps(1, 2, 3, 4)
	result := hsum_ps(v)
	assert.Equal(t, float32(10), result)
}

func TestDotPs(t *testing.T) {
	a := m128ps(1, 2, 3, 4)
	b := m128ps(5, 6, 7, 8)
	result := dot_ps(a, b)
	// dot = 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
	assert.Equal(t, float32(70), result)
}

func TestAddPs_NegativeZero(t *testing.T) {
	// Test handling of special float values
	negZero := float32(math.Copysign(0, -1))
	a := m128ps(negZero, 0, 1, -1)
	b := m128ps(0, negZero, -1, 1)
	result := add_ps(a, b)
	got := toM128ps(result)
	assert.Equal(t, float32(0), got[0])
	assert.Equal(t, float32(0), got[1])
	assert.Equal(t, float32(0), got[2])
	assert.Equal(t, float32(0), got[3])
}

// ============ AVX 256-bit tests ============

func TestAdd256Ps(t *testing.T) {
	a := m256ps(1, 2, 3, 4, 5, 6, 7, 8)
	b := m256ps(10, 20, 30, 40, 50, 60, 70, 80)
	result := add256_ps(a, b)
	expected := [8]float32{11, 22, 33, 44, 55, 66, 77, 88}
	assert.Equal(t, expected, toM256ps(result))
}

func TestMul256Ps(t *testing.T) {
	a := m256ps(1, 2, 3, 4, 5, 6, 7, 8)
	b := m256ps(2, 2, 2, 2, 2, 2, 2, 2)
	result := mul256_ps(a, b)
	expected := [8]float32{2, 4, 6, 8, 10, 12, 14, 16}
	assert.Equal(t, expected, toM256ps(result))
}

func TestFma256Ps(t *testing.T) {
	a := m256ps(1, 2, 3, 4, 5, 6, 7, 8)
	b := m256ps(2, 2, 2, 2, 2, 2, 2, 2)
	c := m256ps(10, 20, 30, 40, 50, 60, 70, 80)
	result := fma256_ps(a, b, c)
	// fma = a * b + c
	expected := [8]float32{12, 24, 36, 48, 60, 72, 84, 96}
	assert.Equal(t, expected, toM256ps(result))
}

func TestAdd256Pd(t *testing.T) {
	a := m256pd(1.5, 2.5, 3.5, 4.5)
	b := m256pd(10.5, 20.5, 30.5, 40.5)
	result := add256_pd(a, b)
	expected := [4]float64{12.0, 23.0, 34.0, 45.0}
	assert.Equal(t, expected, toM256pd(result))
}

func TestMul256Pd(t *testing.T) {
	a := m256pd(2.0, 3.0, 4.0, 5.0)
	b := m256pd(10.0, 10.0, 10.0, 10.0)
	result := mul256_pd(a, b)
	expected := [4]float64{20.0, 30.0, 40.0, 50.0}
	assert.Equal(t, expected, toM256pd(result))
}

func TestAdd256Epi32(t *testing.T) {
	a := m256i32(1, -2, 3, -4, 5, -6, 7, -8)
	b := m256i32(10, 20, 30, 40, 50, 60, 70, 80)
	result := add256_epi32(a, b)
	expected := [8]int32{11, 18, 33, 36, 55, 54, 77, 72}
	assert.Equal(t, expected, toM256i32(result))
}

func TestHsum256Ps(t *testing.T) {
	v := m256ps(1, 2, 3, 4, 5, 6, 7, 8)
	result := hsum256_ps(v)
	// 1+2+3+4+5+6+7+8 = 36
	assert.Equal(t, float32(36), result)
}

func TestDot256Ps(t *testing.T) {
	a := m256ps(1, 2, 3, 4, 5, 6, 7, 8)
	b := m256ps(1, 1, 1, 1, 1, 1, 1, 1)
	result := dot256_ps(a, b)
	// dot = 1+2+3+4+5+6+7+8 = 36
	assert.Equal(t, float32(36), result)
}
