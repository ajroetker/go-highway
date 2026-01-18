package hwy

// TailMask creates a mask with the first 'count' lanes active.
// This is useful for handling the tail (remainder) of an array
// when the size is not a multiple of the vector width.
//
// Example:
//
//	maxLanes := hwy.MaxLanes[float32]()
//	remaining := len(data) % maxLanes
//	if remaining > 0 {
//	    mask := hwy.TailMask[float32](remaining)
//	    v := hwy.MaskLoad(mask, data[len(data)-remaining:])
//	    // ... process tail
//	    hwy.MaskStore(mask, result, output[len(output)-remaining:])
//	}
func TailMask[T Lanes](count int) Mask[T] {
	maxLanes := MaxLanes[T]()
	if count < 0 {
		count = 0
	}
	if count > maxLanes {
		count = maxLanes
	}

	bits := make([]bool, maxLanes)
	for i := 0; i < count; i++ {
		bits[i] = true
	}
	return Mask[T]{bits: bits}
}

// ProcessWithTail is a helper for processing arrays with SIMD that handles
// both full vectors and the tail (remainder) automatically.
//
// It calls:
//   - fullFn(offset) for each full vector (offset is the starting index)
//   - tailFn(offset, count) once for the tail if size is not a multiple of vector width
//
// Example:
//
//	hwy.ProcessWithTail[float32](len(data),
//	    func(offset int) {
//	        // Process full vector at data[offset:]
//	        v := hwy.Load(data[offset:])
//	        result := hwy.Add(v, v)
//	        hwy.Store(result, output[offset:])
//	    },
//	    func(offset, count int) {
//	        // Process tail with mask
//	        mask := hwy.TailMask[float32](count)
//	        v := hwy.MaskLoad(mask, data[offset:])
//	        result := hwy.Add(v, v)
//	        hwy.MaskStore(mask, result, output[offset:])
//	    },
//	)
func ProcessWithTail[T Lanes](size int, fullFn func(offset int), tailFn func(offset, count int)) {
	maxLanes := MaxLanes[T]()

	// Process full vectors
	fullVectors := size / maxLanes
	for i := range fullVectors {
		fullFn(i * maxLanes)
	}

	// Process tail if any
	remaining := size % maxLanes
	if remaining > 0 {
		tailFn(fullVectors*maxLanes, remaining)
	}
}

// ProcessWithTailNoMask is similar to ProcessWithTail but doesn't require
// a tail function. Instead, it processes overlapping vectors for the tail.
// This is simpler but may do redundant work for the last few elements.
//
// Example:
//
//	hwy.ProcessWithTailNoMask[float32](len(data),
//	    func(offset int) {
//	        v := hwy.Load(data[offset:])
//	        result := hwy.Add(v, v)
//	        hwy.Store(result, output[offset:])
//	    },
//	)
func ProcessWithTailNoMask[T Lanes](size int, fullFn func(offset int)) {
	maxLanes := MaxLanes[T]()

	if size < maxLanes {
		// Single partial vector
		fullFn(0)
		return
	}

	// Process full vectors
	fullVectors := size / maxLanes
	for i := range fullVectors {
		fullFn(i * maxLanes)
	}

	// Process tail with overlapping vector if needed
	remaining := size % maxLanes
	if remaining > 0 {
		// Process last full vector, which overlaps with the previous one
		fullFn(size - maxLanes)
	}
}

// AlignedSize rounds up size to the next multiple of vector width.
// This is useful for allocating buffers that will be processed with SIMD.
func AlignedSize[T Lanes](size int) int {
	maxLanes := MaxLanes[T]()
	if maxLanes == 0 {
		return size
	}
	return ((size + maxLanes - 1) / maxLanes) * maxLanes
}

// IsAligned returns true if size is a multiple of vector width.
func IsAligned[T Lanes](size int) bool {
	maxLanes := MaxLanes[T]()
	if maxLanes == 0 {
		return true
	}
	return size%maxLanes == 0
}
