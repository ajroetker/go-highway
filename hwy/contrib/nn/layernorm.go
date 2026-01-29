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

package nn

import (
	stdmath "math"

	"github.com/ajroetker/go-highway/hwy"
)

// LayerNormScalar is a scalar reference implementation for comparison and testing.
func LayerNormScalar[T hwy.Floats](input, output []T, normSize int, gamma, beta []T, epsilon T) {
	size := min(len(input), len(output))
	if size == 0 || normSize <= 0 {
		return
	}

	numGroups := size / normSize

	for g := range numGroups {
		off := g * normSize

		// Compute mean
		var sum float64
		for i := range normSize {
			sum += float64(input[off+i])
		}
		mean := sum / float64(normSize)

		// Compute variance
		var variance float64
		for i := range normSize {
			diff := float64(input[off+i]) - mean
			variance += diff * diff
		}
		variance /= float64(normSize)

		// Normalize
		invStd := 1.0 / stdmath.Sqrt(variance+float64(epsilon))

		if gamma != nil && beta != nil {
			for i := range normSize {
				normed := (float64(input[off+i]) - mean) * invStd
				output[off+i] = T(normed*float64(gamma[i]) + float64(beta[i]))
			}
		} else if gamma != nil {
			for i := range normSize {
				normed := (float64(input[off+i]) - mean) * invStd
				output[off+i] = T(normed * float64(gamma[i]))
			}
		} else {
			for i := range normSize {
				output[off+i] = T((float64(input[off+i]) - mean) * invStd)
			}
		}
	}
}
