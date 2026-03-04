//go:build !noasm && amd64

package tests

import (
	"math"
	"testing"
)

// Helper: convert 8 float32s to [32]byte (AVX-256 format)
func f32x8(a, b, c, d, e, f, g, h float32) [32]byte {
	var result [32]byte
	putFloat32(result[:], 0, a)
	putFloat32(result[:], 4, b)
	putFloat32(result[:], 8, c)
	putFloat32(result[:], 12, d)
	putFloat32(result[:], 16, e)
	putFloat32(result[:], 20, f)
	putFloat32(result[:], 24, g)
	putFloat32(result[:], 28, h)
	return result
}

// Helper: convert [32]byte to [8]float32
func toF32x8(v [32]byte) [8]float32 {
	return [8]float32{
		getFloat32(v[:], 0),
		getFloat32(v[:], 4),
		getFloat32(v[:], 8),
		getFloat32(v[:], 12),
		getFloat32(v[:], 16),
		getFloat32(v[:], 20),
		getFloat32(v[:], 24),
		getFloat32(v[:], 28),
	}
}

// putFloat32 writes a float32 at the specified offset in little-endian order
func putFloat32(b []byte, offset int, v float32) {
	bits := math.Float32bits(v)
	b[offset+0] = byte(bits)
	b[offset+1] = byte(bits >> 8)
	b[offset+2] = byte(bits >> 16)
	b[offset+3] = byte(bits >> 24)
}

// getFloat32 reads a float32 from the specified offset in little-endian order
func getFloat32(b []byte, offset int) float32 {
	bits := uint32(b[offset+0]) | uint32(b[offset+1])<<8 | uint32(b[offset+2])<<16 | uint32(b[offset+3])<<24
	return math.Float32frombits(bits)
}

// TestAddConstPS256 tests the AVX constant pool add function
// add_const_ps256 adds {10, 20, 30, 40, 50, 60, 70, 80} to input
func TestAddConstPS256(t *testing.T) {
	v := f32x8(1, 2, 3, 4, 5, 6, 7, 8)
	result := add_const_ps256(v)
	got := toF32x8(result)
	expected := [8]float32{11, 22, 33, 44, 55, 66, 77, 88}

	for i := 0; i < 8; i++ {
		if got[i] != expected[i] {
			t.Errorf("add_const_ps256()[%d] = %v, want %v", i, got[i], expected[i])
		}
	}
}

// TestMulIndexPS256 tests the AVX constant pool multiply function
// mul_index_ps256 multiplies by {1, 2, 3, 4, 5, 6, 7, 8}
func TestMulIndexPS256(t *testing.T) {
	v := f32x8(10, 10, 10, 10, 10, 10, 10, 10)
	result := mul_index_ps256(v)
	got := toF32x8(result)
	expected := [8]float32{10, 20, 30, 40, 50, 60, 70, 80}

	for i := 0; i < 8; i++ {
		if got[i] != expected[i] {
			t.Errorf("mul_index_ps256()[%d] = %v, want %v", i, got[i], expected[i])
		}
	}
}

// TestHsumPS256 tests the AVX horizontal sum function
func TestHsumPS256(t *testing.T) {
	// Sum of 1+2+3+4+5+6+7+8 = 36
	v := f32x8(1, 2, 3, 4, 5, 6, 7, 8)
	result := hsum_ps256(v)

	if result != 36.0 {
		t.Errorf("hsum_ps256({1,2,3,4,5,6,7,8}) = %v, want 36.0", result)
	}
}

// TestConstPoolWithZeros tests edge case with zeros
func TestConstPoolWithZeros(t *testing.T) {
	v := f32x8(0, 0, 0, 0, 0, 0, 0, 0)
	result := add_const_ps256(v)
	got := toF32x8(result)
	// Adding to zeros should give us the constant pool values
	expected := [8]float32{10, 20, 30, 40, 50, 60, 70, 80}

	for i := 0; i < 8; i++ {
		if got[i] != expected[i] {
			t.Errorf("add_const_ps256(zeros)[%d] = %v, want %v", i, got[i], expected[i])
		}
	}
}

// TestConstPoolWithNegatives tests with negative values
func TestConstPoolWithNegatives(t *testing.T) {
	v := f32x8(-10, -20, -30, -40, -50, -60, -70, -80)
	result := add_const_ps256(v)
	got := toF32x8(result)
	// -10+10=0, -20+20=0, etc.
	expected := [8]float32{0, 0, 0, 0, 0, 0, 0, 0}

	for i := 0; i < 8; i++ {
		if got[i] != expected[i] {
			t.Errorf("add_const_ps256(negatives)[%d] = %v, want %v", i, got[i], expected[i])
		}
	}
}
