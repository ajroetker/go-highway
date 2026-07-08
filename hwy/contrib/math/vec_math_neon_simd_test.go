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

//go:build arm64 && goexperiment.simd

package math

import (
	stdmath "math"
	"simd/archsimd"
	"testing"
)

// These tests exercise the archsimd-backed NEON variants (Go 1.27+,
// GOEXPERIMENT=simd). The hwy/asm-backed variants of the same functions are
// selected under !goexperiment.simd builds and share symbol names, so the
// same assertions run against whichever variant the build tags picked.

func checkF32(t *testing.T, name string, fn func(archsimd.Float32x4) archsimd.Float32x4, ref func(float64) float64, inputs []float32, relTol float64) {
	t.Helper()
	for i := 0; i+4 <= len(inputs); i += 4 {
		v := archsimd.LoadFloat32x4(inputs[i : i+4])
		var got [4]float32
		fn(v).Store(got[:])
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

func checkF64(t *testing.T, name string, fn func(archsimd.Float64x2) archsimd.Float64x2, ref func(float64) float64, inputs []float64, relTol float64) {
	t.Helper()
	for i := 0; i+2 <= len(inputs); i += 2 {
		v := archsimd.LoadFloat64x2(inputs[i : i+2])
		var got [2]float64
		fn(v).Store(got[:])
		for lane := range 2 {
			x := inputs[i+lane]
			want := ref(x)
			g := got[lane]
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

var testInputsF64 = []float64{
	0.001, 0.1, 0.5, 1.0, 1.5, 2.0, 3.14159, 5.0,
	0.25, 0.75, 1.25, 4.0, 6.5, 8.0, 0.01, 2.5,
}

var testInputsSignedF64 = []float64{
	-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0,
	-3.14159, -0.1, 0.1, 3.14159, -1.5, 1.5, -0.25, 0.25,
}

func TestNEONSimdMathF32(t *testing.T) {
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

func TestNEONSimdMathF64(t *testing.T) {
	// The shared base algorithms use single-precision polynomial constants
	// for all lane types, so float64 lanes carry ~1e-7 relative accuracy by
	// design (both the archsimd and hwy/asm NEON variants).
	checkF64(t, "Exp", BaseExpVec_neon_Float64, stdmath.Exp, testInputsSignedF64, 1e-5)
	checkF64(t, "Log", BaseLogVec_neon_Float64, stdmath.Log, testInputsF64, 1e-5)
	checkF64(t, "Sin", BaseSinVec_neon_Float64, stdmath.Sin, testInputsSignedF64, 1e-5)
	checkF64(t, "Cos", BaseCosVec_neon_Float64, stdmath.Cos, testInputsSignedF64, 1e-5)
	checkF64(t, "Tanh", BaseTanhVec_neon_Float64, stdmath.Tanh, testInputsSignedF64, 1e-5)
	checkF64(t, "Erf", BaseErfVec_neon_Float64, stdmath.Erf, testInputsSignedF64, 1e-5)
	checkF64(t, "Sigmoid", BaseSigmoidVec_neon_Float64, func(x float64) float64 {
		return 1 / (1 + stdmath.Exp(-x))
	}, testInputsSignedF64, 1e-5)
}

func BenchmarkNEONSimdExpF32(b *testing.B) {
	v := archsimd.LoadFloat32x4([]float32{0.5, 1.0, 1.5, 2.0})
	scale := archsimd.BroadcastFloat32x4(0.001)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v = BaseExpVec_neon(v.Mul(scale))
	}
	var sink [4]float32
	v.Store(sink[:])
	_ = sink
}

func BenchmarkNEONSimdSinF32(b *testing.B) {
	v := archsimd.LoadFloat32x4([]float32{0.5, 1.0, 1.5, 2.0})
	scale := archsimd.BroadcastFloat32x4(0.999)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v = BaseSinVec_neon(v.Mul(scale))
	}
	var sink [4]float32
	v.Store(sink[:])
	_ = sink
}
