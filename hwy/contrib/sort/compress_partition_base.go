//go:build ignore

package sort

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input compress_partition_base.go -output . -targets avx2,avx512,neon,fallback -dispatch compress_partition

// BaseCompressPartition3Way partitions data into three regions around pivot.
// For 3-way partitioning, we use the scalar Dutch National Flag algorithm
// which is faster than SIMD compress on narrow vectors.
// Returns (lt, gt) where:
//   - data[0:lt] < pivot
//   - data[lt:gt] == pivot
//   - data[gt:n] > pivot
func BaseCompressPartition3Way[T hwy.Lanes](data []T, pivot T) (int, int) {
	// 3-way partition doesn't benefit from SIMD compress on narrow vectors
	// Use scalar Dutch National Flag which is cache-friendly and branchless
	lt := 0
	gt := len(data)
	i := 0
	for i < gt {
		if data[i] < pivot {
			data[lt], data[i] = data[i], data[lt]
			lt++
			i++
		} else if data[i] > pivot {
			gt--
			data[i], data[gt] = data[gt], data[i]
		} else {
			i++
		}
	}
	return lt, gt
}

// BaseCompressPartition partitions data using Highway's double-store technique.
// Returns idx where data[0:idx] < pivot and data[idx:n] >= pivot.
// This is an in-place O(1) space algorithm.
func BaseCompressPartition[T hwy.Lanes](data []T, pivot T) int {
	n := len(data)
	if n == 0 {
		return 0
	}

	lanes := hwy.MaxLanes[T]()
	kUnroll := 4
	preloadSize := kUnroll * lanes

	// Scalar fallback for small arrays
	if n < 2*preloadSize {
		return scalarPartition2Way(data, pivot)
	}

	// Handle non-aligned sizes with scalar
	middleSize := n - 2*preloadSize
	if middleSize%lanes != 0 {
		return scalarPartition2Way(data, pivot)
	}

	pivotVec := hwy.Set(pivot)

	// Preload kUnroll vectors from each end to avoid overwriting unread data
	// We store them in a buffer and process after the main loop
	preloadL := make([]T, preloadSize)
	preloadR := make([]T, preloadSize)
	copy(preloadL, data[:preloadSize])
	copy(preloadR, data[n-preloadSize:])

	// Read pointers for middle section
	readL := preloadSize
	readR := n - preloadSize

	// Write state: writeL is left write position, remaining is unpartitioned space
	writeL := 0
	remaining := n

	// Main loop: process middle region with double-store
	for readL < readR {
		var v hwy.Vec[T]

		// Read from the side with more capacity to avoid overwrites
		capacityL := readL - writeL
		if capacityL > preloadSize {
			readR -= lanes
			v = hwy.Load(data[readR:])
		} else {
			v = hwy.Load(data[readL:])
			readL += lanes
		}

		// Compress elements < pivot
		maskLess := hwy.LessThan(v, pivotVec)
		numLess := hwy.CountTrue(maskLess)
		compressed, _ := hwy.Compress(v, maskLess)

		// Double-store: write to both left and right positions
		// Elements < pivot stay at writeL, elements >= pivot end up at right
		remaining -= lanes
		hwy.Store(compressed, data[writeL:])
		hwy.Store(compressed, data[remaining+writeL:])
		writeL += numLess
	}

	// Process preloaded left vectors
	for i := 0; i < preloadSize; i += lanes {
		v := hwy.Load(preloadL[i:])
		maskLess := hwy.LessThan(v, pivotVec)
		numLess := hwy.CountTrue(maskLess)
		compressed, _ := hwy.Compress(v, maskLess)
		remaining -= lanes
		hwy.Store(compressed, data[writeL:])
		hwy.Store(compressed, data[remaining+writeL:])
		writeL += numLess
	}

	// Process preloaded right vectors (except last 2 which need buffer)
	for i := 0; i < preloadSize-2*lanes; i += lanes {
		v := hwy.Load(preloadR[i:])
		maskLess := hwy.LessThan(v, pivotVec)
		numLess := hwy.CountTrue(maskLess)
		compressed, _ := hwy.Compress(v, maskLess)
		remaining -= lanes
		hwy.Store(compressed, data[writeL:])
		hwy.Store(compressed, data[remaining+writeL:])
		writeL += numLess
	}

	// Last 2 vectors: use buffer to avoid overwrites
	var buf [32]T // enough for 2 vectors of max width
	bufL := 0
	writeR := writeL + remaining

	for i := preloadSize - 2*lanes; i < preloadSize; i += lanes {
		v := hwy.Load(preloadR[i:])
		maskLess := hwy.LessThan(v, pivotVec)
		numLess := hwy.CountTrue(maskLess)
		compressed, _ := hwy.Compress(v, maskLess)

		// Store compressed to temp buffer
		hwy.Store(compressed, buf[bufL:])

		// Left elements stay in buffer
		bufL += numLess

		// Right elements go directly to output
		numRight := lanes - numLess
		writeR -= numRight
		copy(data[writeR:], buf[bufL:bufL+numRight])
	}

	// Copy buffered left elements to final position
	copy(data[writeL:], buf[:bufL])

	return writeL + bufL
}

// scalarPartition2Way is the scalar fallback.
func scalarPartition2Way[T hwy.Lanes](data []T, pivot T) int {
	lt := 0
	gt := len(data)
	for lt < gt {
		if data[lt] < pivot {
			lt++
		} else {
			gt--
			data[lt], data[gt] = data[gt], data[lt]
		}
	}
	return lt
}
