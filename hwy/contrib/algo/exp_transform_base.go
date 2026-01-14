package algo

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/math"
)

//go:generate go run ../../../cmd/hwygen -input exp_transform_base.go -output . -targets avx2,avx512,neon,fallback

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
