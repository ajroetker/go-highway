//go:build arm64

package tests

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAddConstF32x4(t *testing.T) {
	// add_const_f32x4 adds {10, 20, 30, 40} from a constant pool
	v := f32x4(1, 2, 3, 4)
	result := add_const_f32x4(v)
	expected := [4]float32{11, 22, 33, 44}
	assert.Equal(t, expected, toF32x4(result))
}

func TestAddConstF32x4_Zeros(t *testing.T) {
	// Verify with zero input - result should be the constant itself
	v := f32x4(0, 0, 0, 0)
	result := add_const_f32x4(v)
	expected := [4]float32{10, 20, 30, 40}
	assert.Equal(t, expected, toF32x4(result))
}

func TestMulIndexF32x4(t *testing.T) {
	// mul_index_f32x4 multiplies by {1, 2, 3, 4} from a constant pool
	v := f32x4(10, 10, 10, 10)
	result := mul_index_f32x4(v)
	expected := [4]float32{10, 20, 30, 40}
	assert.Equal(t, expected, toF32x4(result))
}

func TestMulIndexF32x4_Identity(t *testing.T) {
	// Multiplying {1, 2, 3, 4} by the constant {1, 2, 3, 4}
	v := f32x4(1, 2, 3, 4)
	result := mul_index_f32x4(v)
	expected := [4]float32{1, 4, 9, 16}
	assert.Equal(t, expected, toF32x4(result))
}

func TestWeightedSumF32x4(t *testing.T) {
	// weighted_sum multiplies by 0.25 and sums all lanes
	v := f32x4(4, 8, 12, 16)
	result := weighted_sum_f32x4(v)
	// (4*0.25) + (8*0.25) + (12*0.25) + (16*0.25) = 1 + 2 + 3 + 4 = 10
	assert.Equal(t, float32(10), result)
}
