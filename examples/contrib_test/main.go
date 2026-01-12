// Example demonstrating the refactored contrib package with AVX2 support.
// This shows how hwygen-generated code can call contrib functions directly.

//go:build amd64 && goexperiment.simd

package main

import (
	"fmt"
	"simd/archsimd"

	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

func main() {
	// Test BaseExpVec_avx2
	input := archsimd.BroadcastFloat32x8(1.0)
	result := math.BaseExpVec_avx2(input)

	var values [8]float32
	result.StoreSlice(values[:])

	fmt.Printf("Exp(1.0) = %v\n", values[0])
	fmt.Printf("Expected: ~2.71828\n")

	// Test BaseExpVec_avx2_Float64
	input64 := archsimd.BroadcastFloat64x4(2.0)
	result64 := math.BaseExpVec_avx2_Float64(input64)

	var values64 [4]float64
	result64.StoreSlice(values64[:])

	fmt.Printf("Exp(2.0) = %v\n", values64[0])
	fmt.Printf("Expected: ~7.38906\n")

	fmt.Println("\nContrib package refactoring successful!")
	fmt.Println("AVX2-native functions are exported and callable directly.")
}
