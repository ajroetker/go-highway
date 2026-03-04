package main

// x86 SSE types (128-bit / 16 bytes)
var sse128Types = map[string]int{
	"__m128":  16, // 4x float32
	"__m128d": 16, // 2x float64
	"__m128i": 16, // various integer types
}

// x86 AVX types (256-bit / 32 bytes)
var avx256Types = map[string]int{
	"__m256":  32, // 8x float32
	"__m256d": 32, // 4x float64
	"__m256i": 32, // various integer types
}

// x86 AVX-512 types (512-bit / 64 bytes)
var avx512Types = map[string]int{
	"__m512":  64, // 16x float32
	"__m512d": 64, // 8x float64
	"__m512i": 64, // various integer types
}

// IsX86SIMDType returns true if the type is any x86 SIMD vector type
func IsX86SIMDType(t string) bool {
	if _, ok := sse128Types[t]; ok {
		return true
	}
	if _, ok := avx256Types[t]; ok {
		return true
	}
	if _, ok := avx512Types[t]; ok {
		return true
	}
	return false
}

// X86SIMDTypeSize returns the size in bytes for an x86 SIMD type, or 0 if not an x86 SIMD type
func X86SIMDTypeSize(t string) int {
	if sz, ok := sse128Types[t]; ok {
		return sz
	}
	if sz, ok := avx256Types[t]; ok {
		return sz
	}
	if sz, ok := avx512Types[t]; ok {
		return sz
	}
	return 0
}

// X86SIMDAlignment returns the required alignment for an x86 SIMD type
func X86SIMDAlignment(t string) int {
	if _, ok := sse128Types[t]; ok {
		return 16
	}
	if _, ok := avx256Types[t]; ok {
		return 32
	}
	if _, ok := avx512Types[t]; ok {
		return 64
	}
	return 0
}
