package dot

// Dot computes the dot product of two float32 slices.
// The result is the sum of element-wise products: Σ(a[i] * b[i]).
//
// If the slices have different lengths, the computation uses the minimum length.
// Returns 0 if either slice is empty.
//
// Example:
//
//	a := []float32{1, 2, 3}
//	b := []float32{4, 5, 6}
//	result := Dot(a, b)  // 1*4 + 2*5 + 3*6 = 32
func Dot(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	return dotImpl32(a, b)
}

// Dot64 computes the dot product of two float64 slices.
// The result is the sum of element-wise products: Σ(a[i] * b[i]).
//
// If the slices have different lengths, the computation uses the minimum length.
// Returns 0 if either slice is empty.
func Dot64(a, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	return dotImpl64(a, b)
}

// DotBatch computes multiple dot products efficiently.
// For each i, computes the dot product of queries[i] and keys[i].
//
// This is useful for batch operations in ML applications (e.g., attention mechanisms).
// Returns a slice of results with length min(len(queries), len(keys)).
//
// Example:
//
//	queries := [][]float32{{1, 2}, {3, 4}}
//	keys := [][]float32{{5, 6}, {7, 8}}
//	results := DotBatch(queries, keys)  // [17, 53]
func DotBatch(queries, keys [][]float32) []float32 {
	n := min(len(queries), len(keys))
	results := make([]float32, n)

	for i := 0; i < n; i++ {
		results[i] = Dot(queries[i], keys[i])
	}

	return results
}
