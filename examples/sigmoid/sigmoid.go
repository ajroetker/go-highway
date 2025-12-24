//go:generate ../../bin/hwygen -input $GOFILE -output . -targets avx2,fallback

package sigmoid

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib"
)

// BaseSigmoid computes the sigmoid activation function: 1 / (1 + exp(-x))
// This is a portable implementation that will be transformed into
// architecture-specific SIMD code by hwygen.
func BaseSigmoid[T hwy.Floats](in, out []T) {
	size := min(len(in), len(out))
	vOne := hwy.Set(T(1))
	for ii := 0; ii < size; ii += vOne.NumLanes() {
		x := hwy.Load(in[ii:])
		negX := hwy.Neg(x)
		// Compute exp(-x) using the contrib package
		expNegX := contrib.Exp(negX)
		// y = 1 / (1 + exp(-x))
		y := hwy.Div(vOne, hwy.Add(vOne, expNegX))
		hwy.Store(y, out[ii:])
	}
}
