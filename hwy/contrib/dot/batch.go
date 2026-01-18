package dot

// DotBatch computes multiple dot products efficiently.
// For each i, computes the dot product of queries[i] and keys[i].
//
// This is useful for batch operations in ML applications (e.g., attention mechanisms).
// Returns a slice of results with length min(len(queries), len(keys)).
func DotBatch(queries, keys [][]float32) []float32 {
	n := min(len(queries), len(keys))
	results := make([]float32, n)

	for i := range n {
		results[i] = Dot(queries[i], keys[i])
	}

	return results
}

// DotBatchFloat64 computes multiple dot products for float64 slices.
func DotBatchFloat64(queries, keys [][]float64) []float64 {
	n := min(len(queries), len(keys))
	results := make([]float64, n)

	for i := range n {
		results[i] = DotFloat64(queries[i], keys[i])
	}

	return results
}
