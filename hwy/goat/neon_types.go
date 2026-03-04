package main

// NEON vector types (128-bit / 16 bytes)
var neon128Types = map[string]int{
	// Integer vectors
	"int8x16_t":  16,
	"int16x8_t":  16,
	"int32x4_t":  16,
	"int64x2_t":  16,
	"uint8x16_t": 16,
	"uint16x8_t": 16,
	"uint32x4_t": 16,
	"uint64x2_t": 16,
	// Float vectors
	"float32x4_t": 16,
	"float64x2_t": 16,
	// Half-precision (float16)
	"float16x8_t": 16,
	// BFloat16
	"bfloat16x8_t": 16,
	// Polynomial types
	"poly8x16_t": 16,
	"poly16x8_t": 16,
	"poly64x2_t": 16,
	"poly128_t":  16,
}

// NEON vector types (64-bit / 8 bytes)
var neon64Types = map[string]int{
	// Integer vectors
	"int8x8_t":   8,
	"int16x4_t":  8,
	"int32x2_t":  8,
	"int64x1_t":  8,
	"uint8x8_t":  8,
	"uint16x4_t": 8,
	"uint32x2_t": 8,
	"uint64x1_t": 8,
	// Float vectors
	"float32x2_t": 8,
	"float64x1_t": 8,
	// Half-precision (float16)
	"float16x4_t": 8,
	// BFloat16
	"bfloat16x4_t": 8,
	// Polynomial types
	"poly8x8_t":  8,
	"poly16x4_t": 8,
	"poly64x1_t": 8,
}

// NEON array types (multiple 128-bit vectors)
var neonArrayTypes = map[string]int{
	// 2 vectors (32 bytes)
	"int8x16x2_t":    32,
	"int16x8x2_t":    32,
	"int32x4x2_t":    32,
	"int64x2x2_t":    32,
	"uint8x16x2_t":   32,
	"uint16x8x2_t":   32,
	"uint32x4x2_t":   32,
	"uint64x2x2_t":   32,
	"float32x4x2_t":  32,
	"float64x2x2_t":  32,
	"float16x8x2_t":  32,
	"bfloat16x8x2_t": 32,
	"poly8x16x2_t":   32,
	"poly16x8x2_t":   32,
	"poly64x2x2_t":   32,
	// 3 vectors (48 bytes)
	"int8x16x3_t":    48,
	"int16x8x3_t":    48,
	"int32x4x3_t":    48,
	"int64x2x3_t":    48,
	"uint8x16x3_t":   48,
	"uint16x8x3_t":   48,
	"uint32x4x3_t":   48,
	"uint64x2x3_t":   48,
	"float32x4x3_t":  48,
	"float64x2x3_t":  48,
	"float16x8x3_t":  48,
	"bfloat16x8x3_t": 48,
	"poly8x16x3_t":   48,
	"poly16x8x3_t":   48,
	"poly64x2x3_t":   48,
	// 4 vectors (64 bytes)
	"int8x16x4_t":    64,
	"int16x8x4_t":    64,
	"int32x4x4_t":    64,
	"int64x2x4_t":    64,
	"uint8x16x4_t":   64,
	"uint16x8x4_t":   64,
	"uint32x4x4_t":   64,
	"uint64x2x4_t":   64,
	"float32x4x4_t":  64,
	"float64x2x4_t":  64,
	"float16x8x4_t":  64,
	"bfloat16x8x4_t": 64,
	"poly8x16x4_t":   64,
	"poly16x8x4_t":   64,
	"poly64x2x4_t":   64,
}

// NEON 64-bit array types (multiple 64-bit vectors)
var neon64ArrayTypes = map[string]int{
	// 2 vectors (16 bytes)
	"int8x8x2_t":     16,
	"int16x4x2_t":    16,
	"int32x2x2_t":    16,
	"int64x1x2_t":    16,
	"uint8x8x2_t":    16,
	"uint16x4x2_t":   16,
	"uint32x2x2_t":   16,
	"uint64x1x2_t":   16,
	"float32x2x2_t":  16,
	"float64x1x2_t":  16,
	"float16x4x2_t":  16,
	"bfloat16x4x2_t": 16,
	"poly8x8x2_t":    16,
	"poly16x4x2_t":   16,
	"poly64x1x2_t":   16,
	// 3 vectors (24 bytes)
	"int8x8x3_t":     24,
	"int16x4x3_t":    24,
	"int32x2x3_t":    24,
	"int64x1x3_t":    24,
	"uint8x8x3_t":    24,
	"uint16x4x3_t":   24,
	"uint32x2x3_t":   24,
	"uint64x1x3_t":   24,
	"float32x2x3_t":  24,
	"float64x1x3_t":  24,
	"float16x4x3_t":  24,
	"bfloat16x4x3_t": 24,
	"poly8x8x3_t":    24,
	"poly16x4x3_t":   24,
	"poly64x1x3_t":   24,
	// 4 vectors (32 bytes)
	"int8x8x4_t":     32,
	"int16x4x4_t":    32,
	"int32x2x4_t":    32,
	"int64x1x4_t":    32,
	"uint8x8x4_t":    32,
	"uint16x4x4_t":   32,
	"uint32x2x4_t":   32,
	"uint64x1x4_t":   32,
	"float32x2x4_t":  32,
	"float64x1x4_t":  32,
	"float16x4x4_t":  32,
	"bfloat16x4x4_t": 32,
	"poly8x8x4_t":    32,
	"poly16x4x4_t":   32,
	"poly64x1x4_t":   32,
}

// IsNeonType returns true if the type is any NEON vector type
func IsNeonType(t string) bool {
	if _, ok := neon128Types[t]; ok {
		return true
	}
	if _, ok := neon64Types[t]; ok {
		return true
	}
	if _, ok := neonArrayTypes[t]; ok {
		return true
	}
	if _, ok := neon64ArrayTypes[t]; ok {
		return true
	}
	return false
}

// NeonTypeSize returns the size in bytes for a NEON type, or 0 if not a NEON type
func NeonTypeSize(t string) int {
	if sz, ok := neon128Types[t]; ok {
		return sz
	}
	if sz, ok := neon64Types[t]; ok {
		return sz
	}
	if sz, ok := neonArrayTypes[t]; ok {
		return sz
	}
	if sz, ok := neon64ArrayTypes[t]; ok {
		return sz
	}
	return 0
}

// NeonVectorCount returns the number of vectors in a NEON type (1 for single, 2-4 for arrays)
func NeonVectorCount(t string) int {
	if _, ok := neon128Types[t]; ok {
		return 1
	}
	if _, ok := neon64Types[t]; ok {
		return 1
	}
	if sz, ok := neonArrayTypes[t]; ok {
		return sz / 16 // 128-bit arrays: size / 16 bytes per vector
	}
	if sz, ok := neon64ArrayTypes[t]; ok {
		return sz / 8 // 64-bit arrays: size / 8 bytes per vector
	}
	return 0
}

// IsNeon64Type returns true if this is a 64-bit (not 128-bit) NEON base type
func IsNeon64Type(t string) bool {
	if _, ok := neon64Types[t]; ok {
		return true
	}
	if _, ok := neon64ArrayTypes[t]; ok {
		return true
	}
	return false
}
