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

//go:build arm64 && !goexperiment.simd

package math

import (
	stdmath "math"
	"testing"

	"github.com/ajroetker/go-highway/hwy/asm"
)

// Mirror of the archsimd-variant tests/benchmarks for the hwy/asm-backed
// NEON variant (!goexperiment.simd builds). Same function names, same
// benchmark shapes — run with and without GOEXPERIMENT=simd and compare
// with benchstat.

func checkF32(t *testing.T, name string, fn func(asm.Float32x4) asm.Float32x4, ref func(float64) float64, inputs []float32, relTol float64) {
	t.Helper()
	for i := 0; i+4 <= len(inputs); i += 4 {
		v := asm.LoadFloat32x4Slice(inputs[i : i+4])
		var got [4]float32
		fn(v).StoreSlice(got[:])
		for lane := range 4 {
			x := float64(inputs[i+lane])
			want := ref(x)
			g := float64(got[lane])
			if stdmath.IsNaN(want) {
				if !stdmath.IsNaN(g) {
					t.Errorf("%s(%v) = %v, want NaN", name, x, g)
				}
				continue
			}
			diff := stdmath.Abs(g - want)
			if diff > relTol*stdmath.Max(stdmath.Abs(want), 1) {
				t.Errorf("%s(%v) = %v, want %v (diff %v)", name, x, g, want, diff)
			}
		}
	}
}

var testInputsF32 = []float32{
	0.001, 0.1, 0.5, 1.0, 1.5, 2.0, 3.14159, 5.0,
	0.25, 0.75, 1.25, 4.0, 6.5, 8.0, 0.01, 2.5,
}

var testInputsSignedF32 = []float32{
	-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0,
	-3.14159, -0.1, 0.1, 3.14159, -1.5, 1.5, -0.25, 0.25,
}

func TestNEONGoatMathF32(t *testing.T) {
	checkF32(t, "Exp", BaseExpVec_neon, stdmath.Exp, testInputsSignedF32, 1e-5)
	checkF32(t, "Log", BaseLogVec_neon, stdmath.Log, testInputsF32, 1e-5)
	checkF32(t, "Sin", BaseSinVec_neon, stdmath.Sin, testInputsSignedF32, 1e-5)
	checkF32(t, "Cos", BaseCosVec_neon, stdmath.Cos, testInputsSignedF32, 1e-5)
	checkF32(t, "Tanh", BaseTanhVec_neon, stdmath.Tanh, testInputsSignedF32, 1e-5)
	checkF32(t, "Erf", BaseErfVec_neon, stdmath.Erf, testInputsSignedF32, 1e-5)
	checkF32(t, "Sigmoid", BaseSigmoidVec_neon, func(x float64) float64 {
		return 1 / (1 + stdmath.Exp(-x))
	}, testInputsSignedF32, 1e-5)
}

func BenchmarkNEONSimdExpF32(b *testing.B) {
	v := asm.LoadFloat32x4Slice([]float32{0.5, 1.0, 1.5, 2.0})
	scale := asm.BroadcastFloat32x4(0.001)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v = BaseExpVec_neon(v.Mul(scale))
	}
	var sink [4]float32
	v.StoreSlice(sink[:])
	_ = sink
}

func BenchmarkNEONSimdSinF32(b *testing.B) {
	v := asm.LoadFloat32x4Slice([]float32{0.5, 1.0, 1.5, 2.0})
	scale := asm.BroadcastFloat32x4(0.999)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v = BaseSinVec_neon(v.Mul(scale))
	}
	var sink [4]float32
	v.StoreSlice(sink[:])
	_ = sink
}
