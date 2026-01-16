//go:build amd64 && !goexperiment.simd

package hwy

import "golang.org/x/sys/cpu"

// Fallback for when GOEXPERIMENT=simd is not enabled.
// This version assumes AVX2 is available (common on modern x86-64).
// For actual CPU detection, build with GOEXPERIMENT=simd.

// CPU feature flags for float16/bfloat16 support
var (
	// hasF16C indicates F16C support: float16 <-> float32 conversions (Haswell+)
	hasF16C bool

	// hasAVX512FP16 indicates AVX-512 FP16 support: native float16 arithmetic (Sapphire Rapids+)
	hasAVX512FP16 bool

	// hasAVX512BF16 indicates AVX-512 BF16 support: bfloat16 dot products (Cooper Lake+)
	hasAVX512BF16 bool
)

func init() {
	// Check if SIMD is disabled via environment variable
	if NoSimdEnv() {
		setScalarMode()
		return
	}

	detectCPUFeatures()
	detectFP16BF16Features()
}

func detectCPUFeatures() {
	// Without GOEXPERIMENT=simd, we can't use archsimd for CPU detection.
	// Default to SSE2 which is baseline for all amd64 CPUs.
	// Build with GOEXPERIMENT=simd for proper AVX2/AVX512 detection.
	currentLevel = DispatchSSE2
	currentWidth = 16
	currentName = "sse2"
}

func detectFP16BF16Features() {
	// F16C detection: use FMA as a proxy (F16C is present on all FMA-capable CPUs)
	if cpu.X86.HasAVX {
		hasF16C = cpu.X86.HasFMA
	}

	// AVX-512 BF16 detection via x/sys/cpu
	if cpu.X86.HasAVX512 {
		hasAVX512BF16 = cpu.X86.HasAVX512BF16
	}

	// AVX-512 FP16: not yet supported in x/sys/cpu
	hasAVX512FP16 = false
}

func setScalarMode() {
	currentLevel = DispatchScalar
	currentWidth = 16 // Use 16-byte vectors even in scalar mode for consistency
	currentName = "scalar"
}

// HasF16C returns true if the CPU supports F16C instructions.
// F16C provides hardware-accelerated float16 <-> float32 conversions.
// Present on Intel Haswell+ and AMD Piledriver+ CPUs.
func HasF16C() bool {
	return hasF16C
}

// HasAVX512FP16 returns true if the CPU supports AVX-512 FP16 instructions.
// AVX-512 FP16 provides native float16 arithmetic operations.
// Present on Intel Sapphire Rapids+ CPUs (2023+).
func HasAVX512FP16() bool {
	return hasAVX512FP16
}

// HasAVX512BF16 returns true if the CPU supports AVX-512 BF16 instructions.
// AVX-512 BF16 provides bfloat16 dot product operations.
// Present on Intel Cooper Lake+ and AMD Zen 4+ CPUs.
func HasAVX512BF16() bool {
	return hasAVX512BF16
}

// HasARMFP16 returns false on x86 (ARM FP16 is ARM-specific).
// Use HasF16C() or HasAVX512FP16() for x86 float16 support.
func HasARMFP16() bool {
	return false
}

// HasARMBF16 returns false on x86 (ARM BF16 is ARM-specific).
// Use HasAVX512BF16() for x86 bfloat16 support.
func HasARMBF16() bool {
	return false
}
