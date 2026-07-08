// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
	result.Store(values[:])

	fmt.Printf("Exp(1.0) = %v\n", values[0])
	fmt.Printf("Expected: ~2.71828\n")

	// Test BaseExpVec_avx2_Float64
	input64 := archsimd.BroadcastFloat64x4(2.0)
	result64 := math.BaseExpVec_avx2_Float64(input64)

	var values64 [4]float64
	result64.Store(values64[:])

	fmt.Printf("Exp(2.0) = %v\n", values64[0])
	fmt.Printf("Expected: ~7.38906\n")

	fmt.Println("\nContrib package refactoring successful!")
	fmt.Println("AVX2-native functions are exported and callable directly.")
}
