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
