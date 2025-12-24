//go:build amd64 && !goexperiment.simd

package hwy

// Fallback for when GOEXPERIMENT=simd is not enabled.
// This version assumes AVX2 is available (common on modern x86-64).
// For actual CPU detection, build with GOEXPERIMENT=simd.

func init() {
	// Check if SIMD is disabled via environment variable
	if NoSimdEnv() {
		setScalarMode()
		return
	}

	detectCPUFeatures()
}

func detectCPUFeatures() {
	// Without GOEXPERIMENT=simd, we can't use archsimd for CPU detection.
	// Default to SSE2 which is baseline for all amd64 CPUs.
	// Build with GOEXPERIMENT=simd for proper AVX2/AVX512 detection.
	currentLevel = DispatchSSE2
	currentWidth = 16
	currentName = "sse2"
}

func setScalarMode() {
	currentLevel = DispatchScalar
	currentWidth = 16 // Use 16-byte vectors even in scalar mode for consistency
	currentName = "scalar"
}
