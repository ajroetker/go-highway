package sort

// =============================================================================
// Constants for radix sort
// =============================================================================

// 8-bit mask constants for radix sort
var (
	radixMask8_i32 int32 = 0xFF
	radixMask8_i64 int64 = 0xFF
)

// 16-bit mask constants for radix sort
var (
	radixMask16_i32 int32 = 0xFFFF
	radixMask16_i64 int64 = 0xFFFF
)
