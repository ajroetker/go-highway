//go:build arm64

package hwy

import "golang.org/x/sys/cpu"

func init() {
	// Check for HWY_NO_SIMD environment variable first
	if NoSimdEnv() {
		currentLevel = DispatchScalar
		currentWidth = 16
		currentName = "scalar"
		return
	}

	// ARM64 (AArch64) always has NEON (ASIMD) available.
	// It's part of the ARMv8-A base architecture.
	// We still check the cpu package for future SVE support.

	// Note: cpu.ARM64.HasASIMD is always true for ARMv8+
	// We check it for consistency and to enable SVE detection later.
	if cpu.ARM64.HasASIMD {
		currentLevel = DispatchNEON
		currentWidth = 16 // NEON is 128-bit (16 bytes)
		currentName = "neon"
	} else {
		// Fallback to scalar (should never happen on ARMv8+)
		currentLevel = DispatchScalar
		currentWidth = 16
		currentName = "scalar"
	}

	// Future: SVE support
	// if cpu.ARM64.HasSVE {
	//     currentLevel = DispatchSVE
	//     currentWidth = ... // SVE width is variable
	//     currentName = "sve"
	// }
}
