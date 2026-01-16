//go:build !amd64 || !goexperiment.simd

package asm

// Stub implementations for non-AMD64 or non-SIMD builds.
// These should never be called - the hwy package will use scalar fallbacks.

// PromoteF16ToF32F16C stub for non-F16C platforms.
func PromoteF16ToF32F16C(a []uint16, result []float32) {
	panic("F16C not available")
}

// DemoteF32ToF16F16C stub for non-F16C platforms.
func DemoteF32ToF16F16C(a []float32, result []uint16) {
	panic("F16C not available")
}
