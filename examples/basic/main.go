// Package main demonstrates basic usage of go-highway SIMD operations.
package main

import (
	"fmt"

	"github.com/go-highway/highway/hwy"
)

func main() {
	fmt.Println("=== go-highway Basic Example ===")
	fmt.Printf("SIMD Level: %s, Width: %d bytes\n\n", hwy.CurrentName(), hwy.CurrentWidth())

	// Example 1: Vector addition
	fmt.Println("1. Vector Addition:")
	data1 := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	data2 := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	result := make([]float32, len(data1))

	hwy.ProcessWithTail[float32](len(data1),
		func(offset int) {
			a := hwy.Load(data1[offset:])
			b := hwy.Load(data2[offset:])
			sum := hwy.Add(a, b)
			hwy.Store(sum, result[offset:])
		},
		func(offset, count int) {
			mask := hwy.TailMask[float32](count)
			a := hwy.MaskLoad(mask, data1[offset:])
			b := hwy.MaskLoad(mask, data2[offset:])
			sum := hwy.Add(a, b)
			hwy.MaskStore(mask, sum, result[offset:])
		},
	)

	fmt.Printf("  %v + %v = %v\n\n", data1, data2, result)

	// Example 2: Fused multiply-add
	fmt.Println("2. Fused Multiply-Add (a * b + c):")
	a := []float32{2, 3, 4, 5}
	b := []float32{10, 10, 10, 10}
	c := []float32{1, 2, 3, 4}
	fmaResult := make([]float32, len(a))

	v_a := hwy.Load(a)
	v_b := hwy.Load(b)
	v_c := hwy.Load(c)
	v_result := hwy.FMA(v_a, v_b, v_c)
	hwy.Store(v_result, fmaResult)

	fmt.Printf("  %v * %v + %v = %v\n\n", a, b, c, fmaResult)

	// Example 3: Finding maximum values
	fmt.Println("3. Element-wise Maximum:")
	data3 := []float32{15, 5, 25, 10}
	data4 := []float32{10, 20, 15, 30}
	maxResult := make([]float32, len(data3))

	v1 := hwy.Load(data3)
	v2 := hwy.Load(data4)
	vMax := hwy.Max(v1, v2)
	hwy.Store(vMax, maxResult)

	fmt.Printf("  max(%v, %v) = %v\n\n", data3, data4, maxResult)

	// Example 4: Conditional operations with masks
	fmt.Println("4. Conditional Selection (IfThenElse):")
	values := []float32{5, 15, 10, 20}
	threshold := float32(12)
	smallValues := []float32{0, 0, 0, 0}
	largeValues := []float32{100, 100, 100, 100}
	condResult := make([]float32, len(values))

	v := hwy.Load(values)
	thresh := hwy.Set[float32](threshold)
	mask := hwy.GreaterThan(v, thresh)

	v_large := hwy.Load(largeValues)
	v_small := hwy.Load(smallValues)
	v_cond := hwy.IfThenElse(mask, v_large, v_small)
	hwy.Store(v_cond, condResult)

	fmt.Printf("  values: %v\n", values)
	fmt.Printf("  threshold: %.0f\n", threshold)
	fmt.Printf("  result (>threshold ? 100 : 0): %v\n\n", condResult)

	// Example 5: Reduction (sum all elements)
	fmt.Println("5. Reduction (Sum):")
	numbers := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	totalSum := float32(0)

	hwy.ProcessWithTail[float32](len(numbers),
		func(offset int) {
			v := hwy.Load(numbers[offset:])
			totalSum += hwy.ReduceSum(v)
		},
		func(offset, count int) {
			mask := hwy.TailMask[float32](count)
			v := hwy.MaskLoad(mask, numbers[offset:])
			totalSum += hwy.ReduceSum(v)
		},
	)

	fmt.Printf("  sum(%v) = %.0f\n\n", numbers, totalSum)

	// Example 6: Vector operations with different types
	fmt.Println("6. Integer Operations:")
	intData1 := []int32{10, 20, 30, 40}
	intData2 := []int32{5, 15, 25, 35}
	intResult := make([]int32, len(intData1))

	vi1 := hwy.Load(intData1)
	vi2 := hwy.Load(intData2)
	viSum := hwy.Add(vi1, vi2)
	hwy.Store(viSum, intResult)

	fmt.Printf("  %v + %v = %v\n\n", intData1, intData2, intResult)

	fmt.Println("=== Example Complete ===")
}
