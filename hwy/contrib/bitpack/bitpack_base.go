package bitpack

//go:generate go run ../../../cmd/hwygen -input bitpack_base.go -output . -targets avx2,avx512,neon,fallback -dispatch bitpack

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/vec"
)

// MaxBits finds the minimum number of bits required to represent
// all values in the slice. Returns 0 for empty slices or slices containing only zeros.
//
// This is computed by finding the maximum value and then determining
// the bit position of its highest set bit plus one.
//
// Uses SIMD acceleration via vec.Max when available.
//
// Example:
//
//	values := []uint32{5, 12, 3, 15, 7}
//	bits := MaxBits(values)  // Returns 4 (max value 15 needs 4 bits)
func MaxBits[T hwy.UnsignedInts](src []T) int {
	if len(src) == 0 {
		return 0
	}

	// Find the maximum value using SIMD-accelerated vec.Max
	maxVal := vec.Max(src)

	if maxVal == 0 {
		return 0
	}

	// Count bits needed: position of highest set bit + 1
	return bitsNeeded(maxVal)
}

// bitsNeeded returns the number of bits required to represent a value.
func bitsNeeded[T hwy.UnsignedInts](val T) int {
	if val == 0 {
		return 0
	}
	bits := 0
	for val > 0 {
		bits++
		val >>= 1
	}
	return bits
}

// PackedSize returns the number of bytes needed to store n integers
// using the given bit width.
func PackedSize(n, bitWidth int) int {
	if bitWidth == 0 || n == 0 {
		return 0
	}
	totalBits := n * bitWidth
	return (totalBits + 7) / 8
}

// BasePack32 packs uint32 values into a byte slice using the specified bit width.
// Each value is stored using exactly bitWidth bits, tightly packed.
// Returns the number of bytes written to dst.
//
// The bit width must be between 1 and 32 inclusive.
// dst must have at least PackedSize(len(src), bitWidth) bytes available.
//
// Example:
//
//	src := []uint32{5, 12, 3, 15}  // values fit in 4 bits
//	dst := make([]byte, PackedSize(len(src), 4))
//	Pack32(src, 4, dst)  // Packs to 2 bytes: [0x5C, 0x3F] (little-endian)
func BasePack32(src []uint32, bitWidth int, dst []byte) int {
	if len(src) == 0 || bitWidth == 0 {
		return 0
	}
	if bitWidth > 32 {
		bitWidth = 32
	}

	lanes := hwy.Zero[uint32]().NumLanes()
	mask := uint32((1 << bitWidth) - 1)
	maskVec := hwy.Set[uint32](mask)

	bitPos := 0
	bytePos := 0

	// Process SIMD-width blocks
	var i int
	for i = 0; i+lanes <= len(src); i += lanes {
		v := hwy.Load(src[i:])
		v = hwy.And(v, maskVec) // Ensure values fit in bitWidth

		// Pack each lane - this part remains scalar for cross-lane bit packing
		for lane := range lanes {
			val := hwy.GetLane(v, lane)
			packValue32(val, bitWidth, &bitPos, &bytePos, dst)
		}
	}

	// Handle tail elements
	for ; i < len(src); i++ {
		val := src[i] & mask
		packValue32(val, bitWidth, &bitPos, &bytePos, dst)
	}

	// Return bytes written
	if bitPos > 0 {
		return bytePos + 1
	}
	return bytePos
}

// packValue32 packs a single value into the byte stream.
func packValue32(val uint32, bitWidth int, bitPos, bytePos *int, dst []byte) {
	remaining := bitWidth
	for remaining > 0 {
		bitsAvailable := 8 - *bitPos
		bitsToWrite := min(remaining, bitsAvailable)

		// Extract the bits we want to write
		writeMask := uint32((1 << bitsToWrite) - 1)
		bits := val & writeMask
		val >>= bitsToWrite
		remaining -= bitsToWrite

		// Write to current byte position
		dst[*bytePos] |= byte(bits << *bitPos)

		*bitPos += bitsToWrite
		if *bitPos >= 8 {
			*bitPos = 0
			*bytePos++
		}
	}
}

// BaseUnpack32 unpacks uint32 values from a bit-packed byte slice.
// Each value is read using exactly bitWidth bits.
// Returns the number of values unpacked to dst.
//
// The bit width must be between 1 and 32 inclusive.
//
// Example:
//
//	packed := []byte{0x5C, 0x3F}  // 4 values at 4 bits each
//	dst := make([]uint32, 4)
//	Unpack32(packed, 4, dst)  // Unpacks to [5, 12, 3, 15]
func BaseUnpack32(src []byte, bitWidth int, dst []uint32) int {
	if len(src) == 0 || bitWidth == 0 || len(dst) == 0 {
		return 0
	}
	if bitWidth > 32 {
		bitWidth = 32
	}

	lanes := hwy.Zero[uint32]().NumLanes()
	mask := uint32((1 << bitWidth) - 1)

	bitPos := 0
	bytePos := 0
	totalBits := len(src) * 8

	// Unpack values
	var i int
	for i = 0; i < len(dst); i++ {
		if bytePos*8+bitPos+bitWidth > totalBits {
			break
		}
		dst[i] = unpackValue32(mask, bitWidth, &bitPos, &bytePos, src)
	}

	// If we unpacked a full SIMD width, we could potentially use SIMD for masking
	// but the cross-byte nature of bit unpacking limits SIMD benefit here
	_ = lanes

	return i
}

// unpackValue32 unpacks a single value from the byte stream.
func unpackValue32(mask uint32, bitWidth int, bitPos, bytePos *int, src []byte) uint32 {
	var val uint32
	remaining := bitWidth
	shift := 0

	for remaining > 0 && *bytePos < len(src) {
		bitsAvailable := 8 - *bitPos
		bitsToRead := min(remaining, bitsAvailable)

		// Read bits from current byte
		readMask := byte((1 << bitsToRead) - 1)
		bits := (src[*bytePos] >> *bitPos) & readMask
		val |= uint32(bits) << shift

		shift += bitsToRead
		remaining -= bitsToRead
		*bitPos += bitsToRead

		if *bitPos >= 8 {
			*bitPos = 0
			*bytePos++
		}
	}

	return val & mask
}

// BasePack64 packs uint64 values into a byte slice using the specified bit width.
// Each value is stored using exactly bitWidth bits, tightly packed.
// Returns the number of bytes written to dst.
//
// The bit width must be between 1 and 64 inclusive.
func BasePack64(src []uint64, bitWidth int, dst []byte) int {
	if len(src) == 0 || bitWidth == 0 {
		return 0
	}
	if bitWidth > 64 {
		bitWidth = 64
	}

	lanes := hwy.Zero[uint64]().NumLanes()
	var mask uint64
	if bitWidth == 64 {
		mask = ^uint64(0)
	} else {
		mask = (1 << bitWidth) - 1
	}
	maskVec := hwy.Set[uint64](mask)

	bitPos := 0
	bytePos := 0

	// Process SIMD-width blocks
	var i int
	for i = 0; i+lanes <= len(src); i += lanes {
		v := hwy.Load(src[i:])
		v = hwy.And(v, maskVec)

		for lane := range lanes {
			val := hwy.GetLane(v, lane)
			packValue64(val, bitWidth, &bitPos, &bytePos, dst)
		}
	}

	// Handle tail
	for ; i < len(src); i++ {
		val := src[i] & mask
		packValue64(val, bitWidth, &bitPos, &bytePos, dst)
	}

	if bitPos > 0 {
		return bytePos + 1
	}
	return bytePos
}

// packValue64 packs a single uint64 value into the byte stream.
func packValue64(val uint64, bitWidth int, bitPos, bytePos *int, dst []byte) {
	remaining := bitWidth
	for remaining > 0 {
		bitsAvailable := 8 - *bitPos
		bitsToWrite := min(remaining, bitsAvailable)

		writeMask := uint64((1 << bitsToWrite) - 1)
		bits := val & writeMask
		val >>= bitsToWrite
		remaining -= bitsToWrite

		dst[*bytePos] |= byte(bits << *bitPos)

		*bitPos += bitsToWrite
		if *bitPos >= 8 {
			*bitPos = 0
			*bytePos++
		}
	}
}

// BaseUnpack64 unpacks uint64 values from a bit-packed byte slice.
// Each value is read using exactly bitWidth bits.
// Returns the number of values unpacked to dst.
func BaseUnpack64(src []byte, bitWidth int, dst []uint64) int {
	if len(src) == 0 || bitWidth == 0 || len(dst) == 0 {
		return 0
	}
	if bitWidth > 64 {
		bitWidth = 64
	}

	lanes := hwy.Zero[uint64]().NumLanes()
	var mask uint64
	if bitWidth == 64 {
		mask = ^uint64(0)
	} else {
		mask = (1 << bitWidth) - 1
	}

	bitPos := 0
	bytePos := 0
	totalBits := len(src) * 8

	var i int
	for i = 0; i < len(dst); i++ {
		if bytePos*8+bitPos+bitWidth > totalBits {
			break
		}
		dst[i] = unpackValue64(mask, bitWidth, &bitPos, &bytePos, src)
	}

	_ = lanes
	return i
}

// unpackValue64 unpacks a single uint64 value from the byte stream.
func unpackValue64(mask uint64, bitWidth int, bitPos, bytePos *int, src []byte) uint64 {
	var val uint64
	remaining := bitWidth
	shift := 0

	for remaining > 0 && *bytePos < len(src) {
		bitsAvailable := 8 - *bitPos
		bitsToRead := min(remaining, bitsAvailable)

		readMask := byte((1 << bitsToRead) - 1)
		bits := (src[*bytePos] >> *bitPos) & readMask
		val |= uint64(bits) << shift

		shift += bitsToRead
		remaining -= bitsToRead
		*bitPos += bitsToRead

		if *bitPos >= 8 {
			*bitPos = 0
			*bytePos++
		}
	}

	return val & mask
}

// BaseDeltaEncode32 computes the difference between consecutive values.
// For sorted sequences, this produces smaller values that compress better.
//
// dst[0] = src[0] - base
// dst[i] = src[i] - src[i-1] for i > 0
//
// The base value is typically src[0] for sorted sequences starting at an arbitrary value.
//
// Example:
//
//	src := []uint32{100, 102, 105, 106, 110}
//	dst := make([]uint32, len(src))
//	DeltaEncode32(src, 100, dst)  // dst = [0, 2, 3, 1, 4]
func BaseDeltaEncode32(src []uint32, base uint32, dst []uint32) {
	if len(src) == 0 {
		return
	}
	if len(dst) < len(src) {
		return
	}

	lanes := hwy.Zero[uint32]().NumLanes()

	// First element is relative to base
	dst[0] = src[0] - base
	prev := src[0]

	// Process remaining elements
	// Note: Delta encoding has sequential dependencies, limiting SIMD benefit
	// However, we can still vectorize the subtraction if we rearrange
	var i int
	for i = 1; i+lanes <= len(src); i += lanes {
		// Load current block
		curr := hwy.Load(src[i:])

		// Load previous block (shifted by 1)
		// We need src[i-1], src[i], src[i+1], ..., src[i+lanes-2]
		prevVec := hwy.Load(src[i-1:])

		// Compute deltas: curr - prev
		delta := hwy.Sub(curr, prevVec)
		hwy.Store(delta, dst[i:])

		prev = src[i+lanes-1]
	}

	// Handle tail elements
	for ; i < len(src); i++ {
		dst[i] = src[i] - prev
		prev = src[i]
	}
}

// DeltaDecode reconstructs original values from delta-encoded data.
// This is the inverse of DeltaEncode.
//
// dst[0] = base + src[0]
// dst[i] = dst[i-1] + src[i] for i > 0
//
// Note: Delta decoding has inherent sequential dependencies (each output
// depends on the previous output), so this uses scalar code rather than SIMD.
//
// Example:
//
//	src := []uint32{0, 2, 3, 1, 4}  // delta-encoded
//	dst := make([]uint32, len(src))
//	DeltaDecode(src, 100, dst)  // dst = [100, 102, 105, 106, 110]
func DeltaDecode[T hwy.UnsignedInts](src []T, base T, dst []T) {
	if len(src) == 0 {
		return
	}
	if len(dst) < len(src) {
		return
	}

	// Delta decoding has inherent sequential dependencies
	// Each output depends on the previous output
	// This limits SIMD parallelism, so we use a simple scalar loop
	// Future: Consider prefix-sum SIMD algorithms for larger blocks

	current := base
	for i := range src {
		current += src[i]
		dst[i] = current
	}
}

// BaseDeltaEncode64 computes deltas for uint64 values.
func BaseDeltaEncode64(src []uint64, base uint64, dst []uint64) {
	if len(src) == 0 {
		return
	}
	if len(dst) < len(src) {
		return
	}

	lanes := hwy.Zero[uint64]().NumLanes()

	dst[0] = src[0] - base
	prev := src[0]

	var i int
	for i = 1; i+lanes <= len(src); i += lanes {
		curr := hwy.Load(src[i:])
		prevVec := hwy.Load(src[i-1:])
		delta := hwy.Sub(curr, prevVec)
		hwy.Store(delta, dst[i:])
		prev = src[i+lanes-1]
	}

	for ; i < len(src); i++ {
		dst[i] = src[i] - prev
		prev = src[i]
	}
}

