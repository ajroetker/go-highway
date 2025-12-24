package hwy

import (
	"os"
	"strconv"
	"unsafe"
)

// DispatchLevel represents the current SIMD instruction set being used.
type DispatchLevel int

const (
	// DispatchScalar indicates no SIMD, pure Go implementation.
	DispatchScalar DispatchLevel = iota

	// DispatchSSE2 indicates SSE2 instructions (x86-64 baseline).
	DispatchSSE2

	// DispatchAVX2 indicates AVX2 instructions (256-bit SIMD).
	DispatchAVX2

	// DispatchAVX512 indicates AVX-512 instructions (512-bit SIMD).
	DispatchAVX512

	// DispatchNEON indicates ARM NEON instructions (128-bit SIMD).
	DispatchNEON

	// DispatchSVE indicates ARM SVE instructions (scalable vector).
	DispatchSVE
)

// String returns a human-readable name for the dispatch level.
func (d DispatchLevel) String() string {
	switch d {
	case DispatchScalar:
		return "scalar"
	case DispatchSSE2:
		return "sse2"
	case DispatchAVX2:
		return "avx2"
	case DispatchAVX512:
		return "avx512"
	case DispatchNEON:
		return "neon"
	case DispatchSVE:
		return "sve"
	default:
		return "unknown"
	}
}

// currentLevel is the detected SIMD level for this runtime.
// Set by init() in dispatch_*.go files.
var currentLevel DispatchLevel

// currentWidth is the SIMD register width in bytes for the current level.
// Set by init() in dispatch_*.go files.
var currentWidth int

// currentName is the human-readable name of the current SIMD level.
// Set by init() in dispatch_*.go files.
var currentName string

// CurrentLevel returns the SIMD instruction set being used.
func CurrentLevel() DispatchLevel {
	return currentLevel
}

// CurrentWidth returns the SIMD register width in bytes.
// For example: 16 for SSE2/NEON, 32 for AVX2, 64 for AVX-512.
func CurrentWidth() int {
	return currentWidth
}

// CurrentName returns a human-readable name for the current SIMD target.
// For example: "avx2", "neon", "scalar".
func CurrentName() string {
	return currentName
}

// NoSimdEnv checks if the HWY_NO_SIMD environment variable is set.
// When set, Highway will use scalar fallback regardless of CPU capabilities.
// This is useful for testing and debugging.
func NoSimdEnv() bool {
	val := os.Getenv("HWY_NO_SIMD")
	if val == "" {
		return false
	}
	// Any non-empty value is considered true, but also parse as bool
	if b, err := strconv.ParseBool(val); err == nil {
		return b
	}
	return true
}

// MaxLanes returns the maximum number of lanes for type T with the current SIMD width.
//
// For example, with AVX2 (256 bits / 32 bytes):
//   - float32: 32/4 = 8 lanes
//   - float64: 32/8 = 4 lanes
//   - int32: 32/4 = 8 lanes
func MaxLanes[T Lanes]() int {
	var dummy T
	elementSize := int(unsafe.Sizeof(dummy))
	if elementSize == 0 {
		return 0
	}
	return currentWidth / elementSize
}
