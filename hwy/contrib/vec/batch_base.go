package vec

//go:generate go run ../../../cmd/hwygen -input batch_base.go -output . -targets avx2,avx512,neon,fallback -dispatch batch

import "github.com/ajroetker/go-highway/hwy"

// BaseBatchL2SquaredDistance computes the L2 squared distance from a single query
// vector to multiple data vectors using SIMD primitives.
//
// Parameters:
//   - query: a single vector of length dims
//   - data: a flattened array of count vectors, each of length dims (total: count*dims)
//   - distances: output buffer of length count, must be pre-allocated
//   - count: number of data vectors to compare against
//   - dims: dimensionality of each vector
//
// For each i in [0, count):
//
//	distances[i] = sum((query[j] - data[i*dims + j])^2 for j in [0, dims))
//
// The computation is SIMD-accelerated along the dims dimension. The outer loop
// over count is sequential, but each individual distance computation uses
// vectorized operations.
//
// Edge cases:
//   - Returns immediately if count <= 0 or dims <= 0
//   - Validates that data has at least count*dims elements
//   - Validates that distances has at least count elements
//
// Works with float32 and float64 slices.
//
// Example:
//
//	query := []float32{1, 2, 3}
//	data := []float32{4, 5, 6, 1, 2, 3, 0, 0, 0}  // 3 vectors of dims=3
//	distances := make([]float32, 3)
//	BaseBatchL2SquaredDistance(query, data, distances, 3, 3)
//	// distances[0] = (1-4)^2 + (2-5)^2 + (3-6)^2 = 27
//	// distances[1] = 0 (same as query)
//	// distances[2] = 1 + 4 + 9 = 14
func BaseBatchL2SquaredDistance[T hwy.Floats](query, data []T, distances []T, count, dims int) {
	// Handle edge cases
	if count <= 0 || dims <= 0 {
		return
	}

	// Validate input sizes
	if len(data) < count*dims {
		return
	}
	if len(distances) < count {
		return
	}
	if len(query) < dims {
		return
	}

	// Get SIMD lane count for this type
	sum := hwy.Zero[T]()
	lanes := sum.NumLanes()

	// Process each data vector
	for i := range count {
		dataStart := i * dims
		dataVec := data[dataStart : dataStart+dims]

		// Reset accumulator for this distance
		sum = hwy.Zero[T]()

		// Process full vectors using SIMD
		var j int
		for j = 0; j+lanes <= dims; j += lanes {
			vq := hwy.Load(query[j:])
			vd := hwy.Load(dataVec[j:])
			diff := hwy.Sub(vq, vd)
			diffSq := hwy.Mul(diff, diff)
			sum = hwy.Add(sum, diffSq)
		}

		// Reduce vector sum to scalar
		result := hwy.ReduceSum(sum)

		// Handle tail elements with scalar code
		for ; j < dims; j++ {
			diff := query[j] - dataVec[j]
			result += diff * diff
		}

		distances[i] = result
	}
}

// BaseBatchDot computes the dot product of a single query vector with multiple
// data vectors using SIMD primitives.
//
// Parameters:
//   - query: a single vector of length dims
//   - data: a flattened array of count vectors, each of length dims (total: count*dims)
//   - dots: output buffer of length count, must be pre-allocated
//   - count: number of data vectors to compute dot products with
//   - dims: dimensionality of each vector
//
// For each i in [0, count):
//
//	dots[i] = sum(query[j] * data[i*dims + j] for j in [0, dims))
//
// The computation is SIMD-accelerated along the dims dimension. The outer loop
// over count is sequential, but each individual dot product computation uses
// vectorized operations.
//
// Edge cases:
//   - Returns immediately if count <= 0 or dims <= 0
//   - Validates that data has at least count*dims elements
//   - Validates that dots has at least count elements
//
// Works with float32 and float64 slices.
//
// Example:
//
//	query := []float32{1, 2, 3}
//	data := []float32{4, 5, 6, 1, 0, 0, 2, 2, 2}  // 3 vectors of dims=3
//	dots := make([]float32, 3)
//	BaseBatchDot(query, data, dots, 3, 3)
//	// dots[0] = 1*4 + 2*5 + 3*6 = 32
//	// dots[1] = 1*1 + 2*0 + 3*0 = 1
//	// dots[2] = 1*2 + 2*2 + 3*2 = 12
func BaseBatchDot[T hwy.Floats](query, data []T, dots []T, count, dims int) {
	// Handle edge cases
	if count <= 0 || dims <= 0 {
		return
	}

	// Validate input sizes
	if len(data) < count*dims {
		return
	}
	if len(dots) < count {
		return
	}
	if len(query) < dims {
		return
	}

	// Get SIMD lane count for this type
	sum := hwy.Zero[T]()
	lanes := sum.NumLanes()

	// Process each data vector
	for i := range count {
		dataStart := i * dims
		dataVec := data[dataStart : dataStart+dims]

		// Reset accumulator for this dot product
		sum = hwy.Zero[T]()

		// Process full vectors using SIMD
		var j int
		for j = 0; j+lanes <= dims; j += lanes {
			vq := hwy.Load(query[j:])
			vd := hwy.Load(dataVec[j:])
			prod := hwy.Mul(vq, vd)
			sum = hwy.Add(sum, prod)
		}

		// Reduce vector sum to scalar
		result := hwy.ReduceSum(sum)

		// Handle tail elements with scalar code
		for ; j < dims; j++ {
			result += query[j] * dataVec[j]
		}

		dots[i] = result
	}
}
