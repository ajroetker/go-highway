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

package wavelet

import "github.com/ajroetker/go-highway/hwy"

// CDF 9/7 wavelet lifting coefficients.
// These are the standard Cohen-Daubechies-Feauveau 9/7 biorthogonal wavelet
// coefficients used in JPEG 2000 lossy mode.

// Lifting step coefficients (in order of application for synthesis)
const (
	// Alpha97 is the first predict step coefficient
	Alpha97 = -1.586134342059924

	// Beta97 is the first update step coefficient
	Beta97 = -0.052980118572961

	// Gamma97 is the second predict step coefficient
	Gamma97 = 0.882911075530934

	// Delta97 is the second update step coefficient
	Delta97 = 0.443506852043971

	// K97 is the normalization constant applied to low-pass coefficients
	K97 = 1.230174104914001

	// InvK97 is 1/K, applied to high-pass coefficients
	InvK97 = 1.0 / K97
)

// Typed variables for hwygen code generation.
// These are derived from the public constants above.

// Float32 variants
var (
	liftAlpha97_f32 float32 = Alpha97
	liftBeta97_f32  float32 = Beta97
	liftGamma97_f32 float32 = Gamma97
	liftDelta97_f32 float32 = Delta97
	liftK97_f32     float32 = K97
	liftInvK97_f32  float32 = InvK97
)

// Float64 variants
var (
	liftAlpha97_f64 float64 = Alpha97
	liftBeta97_f64  float64 = Beta97
	liftGamma97_f64 float64 = Gamma97
	liftDelta97_f64 float64 = Delta97
	liftK97_f64     float64 = K97
	liftInvK97_f64  float64 = InvK97
)

// Float16 variants
var (
	liftAlpha97_f16 hwy.Float16 = hwy.Float32ToFloat16(Alpha97)
	liftBeta97_f16  hwy.Float16 = hwy.Float32ToFloat16(Beta97)
	liftGamma97_f16 hwy.Float16 = hwy.Float32ToFloat16(Gamma97)
	liftDelta97_f16 hwy.Float16 = hwy.Float32ToFloat16(Delta97)
	liftK97_f16     hwy.Float16 = hwy.Float32ToFloat16(K97)
	liftInvK97_f16  hwy.Float16 = hwy.Float32ToFloat16(InvK97)
)

// BFloat16 variants
var (
	liftAlpha97_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(Alpha97)
	liftBeta97_bf16  hwy.BFloat16 = hwy.Float32ToBFloat16(Beta97)
	liftGamma97_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(Gamma97)
	liftDelta97_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(Delta97)
	liftK97_bf16     hwy.BFloat16 = hwy.Float32ToBFloat16(K97)
	liftInvK97_bf16  hwy.BFloat16 = hwy.Float32ToBFloat16(InvK97)
)

// lift97Coeffs returns the typed 9/7 lifting coefficients for a given float type.
func lift97Coeffs[T hwy.Floats]() (alpha, beta, gamma, delta, k, invK T) {
	var zero T
	switch any(zero).(type) {
	case float32:
		return any(liftAlpha97_f32).(T), any(liftBeta97_f32).(T),
			any(liftGamma97_f32).(T), any(liftDelta97_f32).(T),
			any(liftK97_f32).(T), any(liftInvK97_f32).(T)
	case float64:
		return any(liftAlpha97_f64).(T), any(liftBeta97_f64).(T),
			any(liftGamma97_f64).(T), any(liftDelta97_f64).(T),
			any(liftK97_f64).(T), any(liftInvK97_f64).(T)
	case hwy.Float16:
		return any(liftAlpha97_f16).(T), any(liftBeta97_f16).(T),
			any(liftGamma97_f16).(T), any(liftDelta97_f16).(T),
			any(liftK97_f16).(T), any(liftInvK97_f16).(T)
	case hwy.BFloat16:
		return any(liftAlpha97_bf16).(T), any(liftBeta97_bf16).(T),
			any(liftGamma97_bf16).(T), any(liftDelta97_bf16).(T),
			any(liftK97_bf16).(T), any(liftInvK97_bf16).(T)
	default:
		var z T
		return z, z, z, z, z, z
	}
}
