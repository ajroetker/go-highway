package bitpack

// This file provides the public API for the bitpack package.
// It wraps the generated dispatch functions with cleaner names.

// MaxBits64 returns the minimum bits needed to represent all values in src.
func MaxBits64(src []uint64) int {
	return MaxBits(src)
}

// Pack32 packs uint32 values using the specified bit width.
func Pack32(src []uint32, bitWidth int, dst []byte) int {
	return Pack32Float32(src, bitWidth, dst)
}

// Unpack32 unpacks uint32 values using the specified bit width.
func Unpack32(src []byte, bitWidth int, dst []uint32) int {
	return Unpack32Float32(src, bitWidth, dst)
}

// Pack64 packs uint64 values using the specified bit width.
func Pack64(src []uint64, bitWidth int, dst []byte) int {
	return Pack64Float32(src, bitWidth, dst)
}

// Unpack64 unpacks uint64 values using the specified bit width.
func Unpack64(src []byte, bitWidth int, dst []uint64) int {
	return Unpack64Float32(src, bitWidth, dst)
}

// DeltaEncode32 computes deltas between consecutive uint32 values.
func DeltaEncode32(src []uint32, base uint32, dst []uint32) {
	DeltaEncode32Float32(src, base, dst)
}

// DeltaEncode64 computes deltas between consecutive uint64 values.
func DeltaEncode64(src []uint64, base uint64, dst []uint64) {
	DeltaEncode64Float32(src, base, dst)
}

