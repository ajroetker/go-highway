//go:build amd64 && goexperiment.simd

package hwy

import "simd/archsimd"

func init() {
	// Check if SIMD is disabled via environment variable
	if NoSimdEnv() {
		setScalarMode()
		return
	}

	detectCPUFeatures()
}

func detectCPUFeatures() {
	// Use actual CPU detection from archsimd package
	if archsimd.X86.AVX512() {
		currentLevel = DispatchAVX512
		currentWidth = 64
		currentName = "avx512"
	} else if archsimd.X86.AVX2() {
		currentLevel = DispatchAVX2
		currentWidth = 32
		currentName = "avx2"
	} else if archsimd.X86.AVX() {
		// AVX without AVX2 - use 256-bit but limited ops
		currentLevel = DispatchSSE2 // Treat as SSE2 for safety
		currentWidth = 16
		currentName = "sse2"
	} else {
		// SSE2 is baseline for amd64
		currentLevel = DispatchSSE2
		currentWidth = 16
		currentName = "sse2"
	}
}

func setScalarMode() {
	currentLevel = DispatchScalar
	currentWidth = 16 // Use 16-byte vectors even in scalar mode for consistency
	currentName = "scalar"
}
