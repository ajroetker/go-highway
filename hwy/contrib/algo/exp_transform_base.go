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

package algo

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

//go:generate go run ../../../cmd/hwygen -input exp_transform_base.go -output . -targets avx2,avx512,neon:asm,fallback

// BaseExpTransform applies exp(x) to each element using SIMD.
// Uses Apply for loop and buffer-based tail handling - no scalar fallback needed.
func BaseExpTransform[T hwy.Floats](in, out []T) {
	BaseApply(in, out, math.BaseExpVec)
}

// BaseLogTransform applies ln(x) to each element using SIMD.
func BaseLogTransform[T hwy.Floats](in, out []T) {
	BaseApply(in, out, math.BaseLogVec)
}

// BaseSinTransform applies sin(x) to each element using SIMD.
func BaseSinTransform[T hwy.Floats](in, out []T) {
	BaseApply(in, out, math.BaseSinVec)
}

// BaseCosTransform applies cos(x) to each element using SIMD.
func BaseCosTransform[T hwy.Floats](in, out []T) {
	BaseApply(in, out, math.BaseCosVec)
}

// BaseTanhTransform applies tanh(x) to each element using SIMD.
func BaseTanhTransform[T hwy.Floats](in, out []T) {
	BaseApply(in, out, math.BaseTanhVec)
}

// BaseSigmoidTransform applies sigmoid(x) to each element using SIMD.
func BaseSigmoidTransform[T hwy.Floats](in, out []T) {
	BaseApply(in, out, math.BaseSigmoidVec)
}

// BaseErfTransform applies erf(x) to each element using SIMD.
func BaseErfTransform[T hwy.Floats](in, out []T) {
	BaseApply(in, out, math.BaseErfVec)
}
