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

package activation

import "github.com/ajroetker/go-highway/hwy"

// =============================================================================
// Per-type constants for activation functions
//
// Using typed constants avoids precision loss from hwy.Const's float32
// parameter when generating float64 specializations.
// =============================================================================

// Float16 constants for activations
var (
	actZero_f16           hwy.Float16 = hwy.Float32ToFloat16(0.0)
	actOne_f16            hwy.Float16 = hwy.Float32ToFloat16(1.0)
	actHalf_f16           hwy.Float16 = hwy.Float32ToFloat16(0.5)
	actInvSqrt2_f16       hwy.Float16 = hwy.Float32ToFloat16(0.7071067811865476)
	actGeluApproxCoeff_f16 hwy.Float16 = hwy.Float32ToFloat16(1.702)
	actHardSwishScale_f16 hwy.Float16 = hwy.Float32ToFloat16(0.16666666666666666)
)

// BFloat16 constants for activations
var (
	actZero_bf16           hwy.BFloat16 = hwy.Float32ToBFloat16(0.0)
	actOne_bf16            hwy.BFloat16 = hwy.Float32ToBFloat16(1.0)
	actHalf_bf16           hwy.BFloat16 = hwy.Float32ToBFloat16(0.5)
	actInvSqrt2_bf16       hwy.BFloat16 = hwy.Float32ToBFloat16(0.7071067811865476)
	actGeluApproxCoeff_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(1.702)
	actHardSwishScale_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(0.16666666666666666)
)

// Float32 constants for activations
var (
	actZero_f32           float32 = 0.0
	actOne_f32            float32 = 1.0
	actHalf_f32           float32 = 0.5
	actInvSqrt2_f32       float32 = 0.7071067811865476
	actGeluApproxCoeff_f32 float32 = 1.702
	actHardSwishScale_f32 float32 = 0.16666666666666666
)

// Float64 constants for activations
var (
	actZero_f64           float64 = 0.0
	actOne_f64            float64 = 1.0
	actHalf_f64           float64 = 0.5
	actInvSqrt2_f64       float64 = 0.7071067811865475244008443621048490392848359376884740365883398689
	actGeluApproxCoeff_f64 float64 = 1.702
	actHardSwishScale_f64 float64 = 0.16666666666666666666666666666666666666666666666666666666666666666
)
