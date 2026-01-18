// Package bitpack provides high-performance SIMD bit-packing operations for integer compression.
// This package corresponds to Google Highway's hwy/contrib/bit_pack directory.
//
// # Bit-Packing Overview
//
// Bit-packing is a compression technique that stores integers using only the minimum
// number of bits required. For example, if all values in a block fit in 5 bits,
// the pack operation stores each value using only 5 bits instead of 32, achieving
// a compression ratio of 32/5 = 6.4x.
//
// # Core Functions
//
// The package provides vectorized bit-packing operations for uint8, uint16, uint32, and uint64:
//   - Pack[T](src []T, bitWidth int, dst []byte) int - Pack integers to bit stream
//   - Unpack[T](src []byte, bitWidth int, dst []T) int - Unpack bit stream to integers
//   - MaxBits[T](src []T) int - Find minimum bits needed for a slice
//
// # Delta Encoding
//
// For sorted sequences, delta encoding dramatically improves compression:
//   - DeltaEncode[T](src []T, base T, dst []T) - Compute deltas from base value
//   - DeltaDecode[T](src []T, base T, dst []T) - Reconstruct values from deltas
//
// # Algorithm
//
// The implementation uses SIMD shift and mask operations:
//   1. Process integers in blocks matching the SIMD width (8 for AVX2, 16 for AVX-512)
//   2. Use SIMD shifts to align bits within output words
//   3. Use SIMD OR operations to combine multiple packed values
//   4. Handle cross-word boundaries with appropriate masking
//
// # Example Usage
//
//	import "github.com/ajroetker/go-highway/hwy/contrib/bitpack"
//
//	// Find the bit width needed
//	values := []uint32{5, 12, 3, 15, 7, 2, 9, 11}
//	bitWidth := bitpack.MaxBits(values)  // Returns 4 (max value 15 fits in 4 bits)
//
//	// Pack the values
//	packed := make([]byte, bitpack.PackedSize(len(values), bitWidth))
//	bitpack.Pack(values, bitWidth, packed)
//
//	// Unpack the values
//	unpacked := make([]uint32, len(values))
//	bitpack.Unpack(packed, bitWidth, unpacked)
//
// # Delta Encoding Example
//
//	// For sorted data, use delta encoding first
//	sorted := []uint32{100, 102, 105, 106, 110, 115, 118, 120}
//
//	// Encode deltas (differences from base value)
//	deltas := make([]uint32, len(sorted))
//	bitpack.DeltaEncode(sorted, sorted[0], deltas)  // [0, 2, 5, 6, 10, 15, 18, 20]
//
//	// Now the deltas have smaller values, needing fewer bits
//	bitWidth := bitpack.MaxBits(deltas)  // Only 5 bits instead of 7
//
// # Performance
//
// The SIMD implementation provides significant speedups:
//   - AVX2: ~4-6 billion integers/second
//   - AVX-512: ~6-8 billion integers/second
//   - Performance scales with bit width (lower bit widths = more compression = higher throughput)
//
// # Build Requirements
//
// The SIMD implementations require:
//   - GOEXPERIMENT=simd build flag
//   - AMD64 architecture with AVX2 or AVX-512 support, or ARM64 with NEON
package bitpack
