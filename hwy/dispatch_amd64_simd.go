//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"

	"golang.org/x/sys/cpu"
)

// CPU feature flags for float16/bfloat16 support
var (
	// hasF16C indicates F16C support: float16 <-> float32 conversions (Haswell+)
	// F16C is detected via CPUID leaf 1, ECX bit 29
	hasF16C bool

	// hasAVX512FP16 indicates AVX-512 FP16 support: native float16 arithmetic (Sapphire Rapids+)
	// AVX-512 FP16 is detected via CPUID leaf 7, subleaf 0, EDX bit 23
	hasAVX512FP16 bool

	// hasAVX512BF16 indicates AVX-512 BF16 support: bfloat16 dot products (Cooper Lake+)
	// Available from golang.org/x/sys/cpu
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

func detectFP16BF16Features() {
	// F16C detection: CPUID leaf 1, ECX bit 29
	// F16C provides VCVTPH2PS and VCVTPS2PH instructions for float16 <-> float32 conversion
	// F16C was introduced with Intel Ivy Bridge (2012) and is present on all CPUs with AVX2
	// Note: F16C requires AVX support to be usable
	if cpu.X86.HasAVX {
		hasF16C = cpu.X86.HasFMA // F16C is typically present with FMA (Haswell+)
		// More accurate: check CPUID directly, but FMA is a good proxy
		// since all FMA-capable CPUs also have F16C
	}

	// AVX-512 BF16 detection via x/sys/cpu
	if cpu.X86.HasAVX512 {
		hasAVX512BF16 = cpu.X86.HasAVX512BF16
	}

	// AVX-512 FP16 detection
	// Note: golang.org/x/sys/cpu doesn't have AVX512FP16 yet
	// AVX-512 FP16 is very new (Sapphire Rapids, 2023) and requires:
	// - CPUID leaf 7, subleaf 0, EDX bit 23
	// For now, we leave this as false until x/sys/cpu adds support
	// or we add direct CPUID detection
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
