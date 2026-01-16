//go:build !amd64 && !arm64

package hwy

func init() {
	// Non-amd64 architectures fall back to scalar mode for now.
	// Future implementations will add:
	// - arm64: NEON and SVE support
	// - wasm: SIMD128 support
	// - riscv64: Vector extension support

	currentLevel = DispatchScalar
	currentWidth = 16 // Use 16-byte vectors even in scalar mode for consistency
	currentName = "scalar"
}

// HasF16C returns false on non-x86 platforms (F16C is an x86-specific feature).
func HasF16C() bool {
	return false
}

// HasAVX512FP16 returns false on non-x86 platforms (AVX-512 is x86-specific).
func HasAVX512FP16() bool {
	return false
}

// HasAVX512BF16 returns false on non-x86 platforms (AVX-512 is x86-specific).
func HasAVX512BF16() bool {
	return false
}

// HasARMFP16 returns false on non-ARM64 platforms (ARM FP16 is ARM-specific).
func HasARMFP16() bool {
	return false
}

// HasARMBF16 returns false on non-ARM64 platforms (ARM BF16 is ARM-specific).
func HasARMBF16() bool {
	return false
}
