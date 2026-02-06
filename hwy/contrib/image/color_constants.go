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

package image

import "github.com/ajroetker/go-highway/hwy"

// ICT (Irreversible Color Transform) coefficients from ITU-T T.800 Table G.2.
// These are the standard JPEG 2000 RGB to YCbCr conversion coefficients.

// Forward ICT coefficients: RGB → YCbCr
const (
	// Y = RtoY*R + GtoY*G + BtoY*B
	ICT_RtoY = 0.299
	ICT_GtoY = 0.587
	ICT_BtoY = 0.114

	// Cb = RtoCb*R + GtoCb*G + BtoCb*B
	ICT_RtoCb = -0.16875
	ICT_GtoCb = -0.33126
	ICT_BtoCb = 0.5

	// Cr = RtoCr*R + GtoCr*G + BtoCr*B
	ICT_RtoCr = 0.5
	ICT_GtoCr = -0.41869
	ICT_BtoCr = -0.08131
)

// Inverse ICT coefficients: YCbCr → RGB
const (
	// R = Y + CrtoR*Cr
	ICT_CrtoR = 1.402

	// G = Y + CbtoG*Cb + CrtoG*Cr
	ICT_CbtoG = -0.344136
	ICT_CrtoG = -0.714136

	// B = Y + CbtoB*Cb
	ICT_CbtoB = 1.772
)

// Typed variables for hwygen code generation.
// These are derived from the public constants above and used by the SIMD base functions.

// Float32 variants
var (
	ictRtoY_f32 float32 = ICT_RtoY
	ictGtoY_f32 float32 = ICT_GtoY
	ictBtoY_f32 float32 = ICT_BtoY

	ictRtoCb_f32 float32 = ICT_RtoCb
	ictGtoCb_f32 float32 = ICT_GtoCb
	ictBtoCb_f32 float32 = ICT_BtoCb

	ictRtoCr_f32 float32 = ICT_RtoCr
	ictGtoCr_f32 float32 = ICT_GtoCr
	ictBtoCr_f32 float32 = ICT_BtoCr

	ictCrtoR_f32 float32 = ICT_CrtoR
	ictCbtoG_f32 float32 = ICT_CbtoG
	ictCrtoG_f32 float32 = ICT_CrtoG
	ictCbtoB_f32 float32 = ICT_CbtoB
)

// Float64 variants
var (
	ictRtoY_f64 float64 = ICT_RtoY
	ictGtoY_f64 float64 = ICT_GtoY
	ictBtoY_f64 float64 = ICT_BtoY

	ictRtoCb_f64 float64 = ICT_RtoCb
	ictGtoCb_f64 float64 = ICT_GtoCb
	ictBtoCb_f64 float64 = ICT_BtoCb

	ictRtoCr_f64 float64 = ICT_RtoCr
	ictGtoCr_f64 float64 = ICT_GtoCr
	ictBtoCr_f64 float64 = ICT_BtoCr

	ictCrtoR_f64 float64 = ICT_CrtoR
	ictCbtoG_f64 float64 = ICT_CbtoG
	ictCrtoG_f64 float64 = ICT_CrtoG
	ictCbtoB_f64 float64 = ICT_CbtoB
)

// Float16 variants
var (
	ictRtoY_f16 hwy.Float16 = hwy.Float32ToFloat16(ICT_RtoY)
	ictGtoY_f16 hwy.Float16 = hwy.Float32ToFloat16(ICT_GtoY)
	ictBtoY_f16 hwy.Float16 = hwy.Float32ToFloat16(ICT_BtoY)

	ictRtoCb_f16 hwy.Float16 = hwy.Float32ToFloat16(ICT_RtoCb)
	ictGtoCb_f16 hwy.Float16 = hwy.Float32ToFloat16(ICT_GtoCb)
	ictBtoCb_f16 hwy.Float16 = hwy.Float32ToFloat16(ICT_BtoCb)

	ictRtoCr_f16 hwy.Float16 = hwy.Float32ToFloat16(ICT_RtoCr)
	ictGtoCr_f16 hwy.Float16 = hwy.Float32ToFloat16(ICT_GtoCr)
	ictBtoCr_f16 hwy.Float16 = hwy.Float32ToFloat16(ICT_BtoCr)

	ictCrtoR_f16 hwy.Float16 = hwy.Float32ToFloat16(ICT_CrtoR)
	ictCbtoG_f16 hwy.Float16 = hwy.Float32ToFloat16(ICT_CbtoG)
	ictCrtoG_f16 hwy.Float16 = hwy.Float32ToFloat16(ICT_CrtoG)
	ictCbtoB_f16 hwy.Float16 = hwy.Float32ToFloat16(ICT_CbtoB)
)

// BFloat16 variants
var (
	ictRtoY_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(ICT_RtoY)
	ictGtoY_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(ICT_GtoY)
	ictBtoY_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(ICT_BtoY)

	ictRtoCb_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(ICT_RtoCb)
	ictGtoCb_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(ICT_GtoCb)
	ictBtoCb_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(ICT_BtoCb)

	ictRtoCr_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(ICT_RtoCr)
	ictGtoCr_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(ICT_GtoCr)
	ictBtoCr_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(ICT_BtoCr)

	ictCrtoR_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(ICT_CrtoR)
	ictCbtoG_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(ICT_CbtoG)
	ictCrtoG_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(ICT_CrtoG)
	ictCbtoB_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(ICT_CbtoB)
)

// ictCoeffs returns the typed ICT coefficients for a given float type.
// This is a helper for generic code that needs type-appropriate coefficients.
func ictCoeffs[T hwy.Floats]() (
	rToY, gToY, bToY,
	rToCb, gToCb, bToCb,
	rToCr, gToCr, bToCr,
	crToR, cbToG, crToG, cbToB T,
) {
	var zero T
	switch any(zero).(type) {
	case float32:
		return any(ictRtoY_f32).(T), any(ictGtoY_f32).(T), any(ictBtoY_f32).(T),
			any(ictRtoCb_f32).(T), any(ictGtoCb_f32).(T), any(ictBtoCb_f32).(T),
			any(ictRtoCr_f32).(T), any(ictGtoCr_f32).(T), any(ictBtoCr_f32).(T),
			any(ictCrtoR_f32).(T), any(ictCbtoG_f32).(T), any(ictCrtoG_f32).(T), any(ictCbtoB_f32).(T)
	case float64:
		return any(ictRtoY_f64).(T), any(ictGtoY_f64).(T), any(ictBtoY_f64).(T),
			any(ictRtoCb_f64).(T), any(ictGtoCb_f64).(T), any(ictBtoCb_f64).(T),
			any(ictRtoCr_f64).(T), any(ictGtoCr_f64).(T), any(ictBtoCr_f64).(T),
			any(ictCrtoR_f64).(T), any(ictCbtoG_f64).(T), any(ictCrtoG_f64).(T), any(ictCbtoB_f64).(T)
	case hwy.Float16:
		return any(ictRtoY_f16).(T), any(ictGtoY_f16).(T), any(ictBtoY_f16).(T),
			any(ictRtoCb_f16).(T), any(ictGtoCb_f16).(T), any(ictBtoCb_f16).(T),
			any(ictRtoCr_f16).(T), any(ictGtoCr_f16).(T), any(ictBtoCr_f16).(T),
			any(ictCrtoR_f16).(T), any(ictCbtoG_f16).(T), any(ictCrtoG_f16).(T), any(ictCbtoB_f16).(T)
	case hwy.BFloat16:
		return any(ictRtoY_bf16).(T), any(ictGtoY_bf16).(T), any(ictBtoY_bf16).(T),
			any(ictRtoCb_bf16).(T), any(ictGtoCb_bf16).(T), any(ictBtoCb_bf16).(T),
			any(ictRtoCr_bf16).(T), any(ictGtoCr_bf16).(T), any(ictBtoCr_bf16).(T),
			any(ictCrtoR_bf16).(T), any(ictCbtoG_bf16).(T), any(ictCrtoG_bf16).(T), any(ictCbtoB_bf16).(T)
	default:
		// Should not reach here with valid Floats types
		var z T
		return z, z, z, z, z, z, z, z, z, z, z, z, z
	}
}
