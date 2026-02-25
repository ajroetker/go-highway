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

package quantize

//go:generate go run ../../../cmd/hwygen -input quantize_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch quantize

import (
	"github.com/ajroetker/go-highway/hwy"
)

// BaseDequantizeUint8 converts quantized uint8 values to float32.
//
//	output[i] = min + float32(input[i]) * scale
func BaseDequantizeUint8(input []uint8, output []float32, min, scale float32) {
	if len(input) == 0 {
		return
	}
	n := len(input)
	if len(output) < n {
		n = len(output)
	}

	lanes := hwy.NumLanes[float32]()
	minVec := hwy.Set[float32](min)
	scaleVec := hwy.Set[float32](scale)

	buf := make([]float32, lanes)

	i := 0
	for ; i+lanes <= n; i += lanes {
		// Promote uint8 → float32 into buffer
		for j := range lanes {
			buf[j] = float32(input[i+j])
		}

		// Load float32 vector and apply: output = min + val * scale
		v := hwy.Load(buf)
		result := hwy.MulAdd(v, scaleVec, minVec)
		hwy.Store(result, output[i:])
	}

	// Scalar tail
	for ; i < n; i++ {
		output[i] = min + float32(input[i])*scale
	}
}

// BaseQuantizeFloat32 converts float32 values to quantized uint8.
//
//	output[i] = uint8(round(clamp((input[i] - min) / scale, 0, 255)))
func BaseQuantizeFloat32(input []float32, output []uint8, min, scale float32) {
	if len(input) == 0 {
		return
	}
	n := len(input)
	if len(output) < n {
		n = len(output)
	}

	lanes := hwy.NumLanes[float32]()
	minVec := hwy.Set[float32](min)
	invScaleVec := hwy.Set[float32](1.0 / scale)
	zeroVec := hwy.Zero[float32]()
	max255Vec := hwy.Set[float32](255.0)

	buf := make([]float32, lanes)

	i := 0
	for ; i+lanes <= n; i += lanes {
		v := hwy.Load(input[i:])

		// (input - min) / scale
		diff := hwy.Mul(hwy.Sub(v, minVec), invScaleVec)

		// Round and clamp to [0, 255]
		rounded := hwy.Clamp(hwy.Round(diff), zeroVec, max255Vec)

		// Store to buffer and narrow float32 → uint8
		hwy.Store(rounded, buf)
		for j := range lanes {
			output[i+j] = uint8(buf[j])
		}
	}

	// Scalar tail
	for ; i < n; i++ {
		val := (input[i] - min) / scale
		rounded := float32(int32(val + 0.5))
		if rounded < 0 {
			rounded = 0
		}
		if rounded > 255 {
			rounded = 255
		}
		output[i] = uint8(rounded)
	}
}
