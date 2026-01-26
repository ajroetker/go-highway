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

//go:build !noasm && arm64

// Override hwygen-generated NEON implementations with GoAT full-function assembly.
// This eliminates per-operation function call overhead by keeping entire loops in assembly.
package vec

import (
	"math"

	"github.com/ajroetker/go-highway/hwy/contrib/vec/asm"
)

func init() {
	// Override norm operations for float32/float64
	SquaredNormFloat32 = asm.SquaredNormF32
	SquaredNormFloat64 = asm.SquaredNormF64
	NormFloat32 = goatNormF32
	NormFloat64 = goatNormF64

	// Override reduce operations for float32/float64
	SumFloat32 = asm.SumF32
	SumFloat64 = asm.SumF64
	MinFloat32 = asm.MinF32
	MaxFloat32 = asm.MaxF32

	// Override unsigned integer reduce operations with GoAT whole-loop assembly
	MaxUint32 = asmReduceMaxU32
	MaxUint64 = asmReduceMaxU64

	// Override distance operations for float32/float64
	L2SquaredDistanceFloat32 = asm.L2SquaredDistanceF32
	L2SquaredDistanceFloat64 = asm.L2SquaredDistanceF64
	L2DistanceFloat32 = goatL2DistanceF32
	L2DistanceFloat64 = goatL2DistanceF64

	// Override dot product operations for float32/float64
	DotFloat32 = asm.DotF32
	DotFloat64 = asm.DotF64

	// Override argmax/argmin with GoAT whole-loop assembly
	ArgmaxFloat32 = asm.ArgmaxF32
	ArgmaxFloat64 = asm.ArgmaxF64
	ArgminFloat32 = asm.ArgminF32
	ArgminFloat64 = asm.ArgminF64

	// Override 2-arg in-place arithmetic operations for float32
	// These use the 3-arg GoAT functions with dst as both destination and first source
	AddFloat32 = goatAddF32
	SubFloat32 = goatSubF32
	MulFloat32 = goatMulF32

	// Override 3-arg arithmetic operations for float32
	AddToFloat32 = goatAddToF32
	SubToFloat32 = goatSubToF32
	MulToFloat32 = goatMulToF32
	ScaleFloat32 = goatScaleF32
	ScaleToFloat32 = goatScaleToF32
	AddConstFloat32 = goatAddConstF32
	MulConstAddToFloat32 = goatMulConstAddToF32
}

// Norm using GoAT squared norm + sqrt
func goatNormF32(v []float32) float32 {
	sqNorm := asm.SquaredNormF32(v)
	if sqNorm == 0 {
		return 0
	}
	return float32(math.Sqrt(float64(sqNorm)))
}

func goatNormF64(v []float64) float64 {
	sqNorm := asm.SquaredNormF64(v)
	if sqNorm == 0 {
		return 0
	}
	return math.Sqrt(sqNorm)
}

// L2 distance using GoAT squared distance + sqrt
func goatL2DistanceF32(a, b []float32) float32 {
	sqDist := asm.L2SquaredDistanceF32(a, b)
	return float32(math.Sqrt(float64(sqDist)))
}

func goatL2DistanceF64(a, b []float64) float64 {
	sqDist := asm.L2SquaredDistanceF64(a, b)
	return math.Sqrt(sqDist)
}

// 2-arg in-place arithmetic wrappers using GoAT assembly
// These pass dst as both destination and first source for in-place operation
func goatAddF32(dst, s []float32) {
	asm.AddSlicesF32(dst, dst, s)
}

func goatSubF32(dst, s []float32) {
	asm.SubSlicesF32(dst, dst, s)
}

func goatMulF32(dst, s []float32) {
	asm.MulSlicesF32(dst, dst, s)
}

// 3-arg arithmetic wrappers that call into GoAT assembly
func goatAddToF32(dst, a, b []float32) {
	asm.AddSlicesF32(dst, a, b)
}

func goatSubToF32(dst, a, b []float32) {
	asm.SubSlicesF32(dst, a, b)
}

func goatMulToF32(dst, a, b []float32) {
	asm.MulSlicesF32(dst, a, b)
}

func goatScaleF32(c float32, dst []float32) {
	// Scale in-place: dst = c * dst
	asm.ScaleF32(dst, dst, c)
}

func goatScaleToF32(dst []float32, c float32, s []float32) {
	asm.ScaleF32(dst, s, c)
}

func goatAddConstF32(c float32, dst []float32) {
	asm.AddConstF32(dst, c)
}

func goatMulConstAddToF32(dst []float32, a float32, x []float32) {
	asm.AxpyF32(dst, x, a)
}

// Unsigned integer reduce using GoAT whole-loop assembly
func asmReduceMaxU32(v []uint32) uint32 {
	return asm.ReduceMaxU32(v)
}

func asmReduceMaxU64(v []uint64) uint64 {
	return asm.ReduceMaxU64(v)
}
