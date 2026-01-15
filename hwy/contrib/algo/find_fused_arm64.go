//go:build arm64

package algo

import (
	"os"

	"github.com/ajroetker/go-highway/hwy/asm"
)

// This file provides fused Find/Count implementations for ARM64 that process
// entire slices in a single assembly call, avoiding per-iteration function overhead.
// These are significantly faster than the generated NEON implementations which
// use array-backed Vec types with per-iteration assembly calls.

// FusedFindFloat32 finds the first element equal to value in a float32 slice.
func FusedFindFloat32(slice []float32, value float32) int {
	return asm.FindEqualF32(slice, value)
}

// FusedFindFloat64 finds the first element equal to value in a float64 slice.
func FusedFindFloat64(slice []float64, value float64) int {
	return asm.FindEqualF64(slice, value)
}

// FusedFindInt32 finds the first element equal to value in an int32 slice.
func FusedFindInt32(slice []int32, value int32) int {
	return asm.FindEqualI32(slice, value)
}

// FusedFindInt64 finds the first element equal to value in an int64 slice.
func FusedFindInt64(slice []int64, value int64) int {
	return asm.FindEqualI64(slice, value)
}

// FusedCountFloat32 counts elements equal to value in a float32 slice.
func FusedCountFloat32(slice []float32, value float32) int {
	return asm.CountEqualF32(slice, value)
}

// FusedCountFloat64 counts elements equal to value in a float64 slice.
func FusedCountFloat64(slice []float64, value float64) int {
	return asm.CountEqualF64(slice, value)
}

// FusedCountInt32 counts elements equal to value in an int32 slice.
func FusedCountInt32(slice []int32, value int32) int {
	return asm.CountEqualI32(slice, value)
}

// FusedCountInt64 counts elements equal to value in an int64 slice.
func FusedCountInt64(slice []int64, value int64) int {
	return asm.CountEqualI64(slice, value)
}

// FusedContainsFloat32 returns true if value exists in the float32 slice.
func FusedContainsFloat32(slice []float32, value float32) bool {
	return asm.FindEqualF32(slice, value) >= 0
}

// FusedContainsFloat64 returns true if value exists in the float64 slice.
func FusedContainsFloat64(slice []float64, value float64) bool {
	return asm.FindEqualF64(slice, value) >= 0
}

// FusedContainsInt32 returns true if value exists in the int32 slice.
func FusedContainsInt32(slice []int32, value int32) bool {
	return asm.FindEqualI32(slice, value) >= 0
}

// FusedContainsInt64 returns true if value exists in the int64 slice.
func FusedContainsInt64(slice []int64, value int64) bool {
	return asm.FindEqualI64(slice, value) >= 0
}

func init() {
	// Skip fused implementations if SIMD is disabled
	if os.Getenv("HWY_NO_SIMD") != "" {
		return
	}

	// Override the dispatch function variables with fused implementations
	// for all types that have fused assembly available
	FindFloat32 = FusedFindFloat32
	FindFloat64 = FusedFindFloat64
	FindInt32 = FusedFindInt32
	FindInt64 = FusedFindInt64
	CountFloat32 = FusedCountFloat32
	CountFloat64 = FusedCountFloat64
	CountInt32 = FusedCountInt32
	CountInt64 = FusedCountInt64
	ContainsFloat32 = FusedContainsFloat32
	ContainsFloat64 = FusedContainsFloat64
	ContainsInt32 = FusedContainsInt32
	ContainsInt64 = FusedContainsInt64
}
