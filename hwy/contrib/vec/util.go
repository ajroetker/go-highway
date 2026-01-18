package vec

import (
	"math"
)

// Zero sets every element of the given slice to zero.
func Zero[T ~float32 | ~float64](dst []T) {
	clear(dst)
}

// MinIdx returns the index of the minimum value in the input slice.
// If several entries have the minimum value, the first such index is returned.
// It panics if s is zero length.
func MinIdx(s []float32) int {
	if len(s) == 0 {
		panic("MinIdx input cannot be zero")
	}

	// Set min index to first position that's not NaN.
	var ind int
	var min float32
	for i, v := range s {
		min = v
		ind = i
		if !math.IsNaN(float64(v)) {
			break
		}
	}

	// Now look for smaller value. Further NaN values will be ignored.
	for i := ind + 1; i < len(s); i++ {
		if s[i] < min {
			min = s[i]
			ind = i
		}
	}
	return ind
}

// MaxIdx returns the index of the maximum value in the input slice.
// If several entries have the maximum value, the first such index is returned.
// It panics if s is zero length.
func MaxIdx(s []float32) int {
	if len(s) == 0 {
		panic("MaxIdx input cannot be zero")
	}

	// Set max index to first position that's not NaN.
	var ind int
	var max float32
	for i, v := range s {
		max = v
		ind = i
		if !math.IsNaN(float64(v)) {
			break
		}
	}

	// Now look for larger value. Further NaN values will be ignored.
	for i := ind + 1; i < len(s); i++ {
		if s[i] > max {
			max = s[i]
			ind = i
		}
	}
	return ind
}

// MinIdx64 returns the index of the minimum value in the input slice.
func MinIdx64(s []float64) int {
	if len(s) == 0 {
		panic("MinIdx64 input cannot be zero")
	}

	var ind int
	var min float64
	for i, v := range s {
		min = v
		ind = i
		if !math.IsNaN(v) {
			break
		}
	}

	for i := ind + 1; i < len(s); i++ {
		if s[i] < min {
			min = s[i]
			ind = i
		}
	}
	return ind
}

// MaxIdx64 returns the index of the maximum value in the input slice.
func MaxIdx64(s []float64) int {
	if len(s) == 0 {
		panic("MaxIdx64 input cannot be zero")
	}

	var ind int
	var max float64
	for i, v := range s {
		max = v
		ind = i
		if !math.IsNaN(v) {
			break
		}
	}

	for i := ind + 1; i < len(s); i++ {
		if s[i] > max {
			max = s[i]
			ind = i
		}
	}
	return ind
}

// EncodeFloat32s encodes a slice of float32 values into a byte slice (little-endian).
func EncodeFloat32s(dst []byte, src []float32) {
	if len(dst) < len(src)*4 {
		panic("dst is too short")
	}
	for i, v := range src {
		bits := math.Float32bits(v)
		dst[i*4] = byte(bits)
		dst[i*4+1] = byte(bits >> 8)
		dst[i*4+2] = byte(bits >> 16)
		dst[i*4+3] = byte(bits >> 24)
	}
}

// DecodeFloat32s decodes a byte slice into a slice of float32 values (little-endian).
func DecodeFloat32s(src []byte, dst []float32) {
	if len(src) < len(dst)*4 {
		panic("src is too short")
	}
	for i := range dst {
		bits := uint32(src[i*4]) | uint32(src[i*4+1])<<8 | uint32(src[i*4+2])<<16 | uint32(src[i*4+3])<<24
		dst[i] = math.Float32frombits(bits)
	}
}
