package image

import (
	"github.com/ajroetker/go-highway/hwy"
)

//go:generate go run ../../../cmd/hwygen -input point_ops_base.go -output . -targets avx2,avx512,neon,fallback -dispatch pointops

// BaseBrightnessContrast applies linear transformation: out = in * scale + offset.
// This is the fundamental point operation for adjusting image brightness and contrast.
func BaseBrightnessContrast[T hwy.Floats](img, out *Image[T], scale, offset T) {
	if img == nil || out == nil || img.data == nil || out.data == nil {
		return
	}

	scaleVec := hwy.Set(scale)
	offsetVec := hwy.Set(offset)
	lanes := hwy.MaxLanes[T]()

	for y := 0; y < img.height; y++ {
		inRow := img.Row(y)
		outRow := out.Row(y)
		width := img.width
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			v := hwy.Load(inRow[i:])
			// result = v * scale + offset (FMA)
			result := hwy.FMA(v, scaleVec, offsetVec)
			hwy.Store(result, outRow[i:])
		}

		// Handle tail elements via buffer
		if remaining := width - i; remaining > 0 {
			buf := make([]T, lanes)
			copy(buf, inRow[i:i+remaining])
			v := hwy.Load(buf)
			result := hwy.FMA(v, scaleVec, offsetVec)
			hwy.Store(result, buf)
			copy(outRow[i:i+remaining], buf[:remaining])
		}
	}
}

// BaseClampImage clamps pixel values to [minVal, maxVal].
func BaseClampImage[T hwy.Floats](img, out *Image[T], minVal, maxVal T) {
	if img == nil || out == nil || img.data == nil || out.data == nil {
		return
	}

	minVec := hwy.Set(minVal)
	maxVec := hwy.Set(maxVal)
	lanes := hwy.MaxLanes[T]()

	for y := 0; y < img.height; y++ {
		inRow := img.Row(y)
		outRow := out.Row(y)
		width := img.width
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			v := hwy.Load(inRow[i:])
			// Clamp: max(min(v, maxVec), minVec)
			result := hwy.Max(hwy.Min(v, maxVec), minVec)
			hwy.Store(result, outRow[i:])
		}

		// Handle tail elements
		if remaining := width - i; remaining > 0 {
			buf := make([]T, lanes)
			copy(buf, inRow[i:i+remaining])
			v := hwy.Load(buf)
			result := hwy.Max(hwy.Min(v, maxVec), minVec)
			hwy.Store(result, buf)
			copy(outRow[i:i+remaining], buf[:remaining])
		}
	}
}

// BaseThreshold applies binary threshold: out = (in >= threshold) ? above : below.
func BaseThreshold[T hwy.Floats](img, out *Image[T], threshold, below, above T) {
	if img == nil || out == nil || img.data == nil || out.data == nil {
		return
	}

	threshVec := hwy.Set(threshold)
	belowVec := hwy.Set(below)
	aboveVec := hwy.Set(above)
	lanes := hwy.MaxLanes[T]()

	for y := 0; y < img.height; y++ {
		inRow := img.Row(y)
		outRow := out.Row(y)
		width := img.width
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			v := hwy.Load(inRow[i:])
			// mask = v >= threshold
			mask := hwy.GreaterEqual(v, threshVec)
			// result = mask ? above : below
			result := hwy.IfThenElse(mask, aboveVec, belowVec)
			hwy.Store(result, outRow[i:])
		}

		// Handle tail elements
		if remaining := width - i; remaining > 0 {
			buf := make([]T, lanes)
			copy(buf, inRow[i:i+remaining])
			v := hwy.Load(buf)
			mask := hwy.GreaterEqual(v, threshVec)
			result := hwy.IfThenElse(mask, aboveVec, belowVec)
			hwy.Store(result, buf)
			copy(outRow[i:i+remaining], buf[:remaining])
		}
	}
}

// BaseInvert inverts pixel values: out = maxVal - in.
// For normalized images [0,1], use maxVal=1.
func BaseInvert[T hwy.Floats](img, out *Image[T], maxVal T) {
	if img == nil || out == nil || img.data == nil || out.data == nil {
		return
	}

	maxVec := hwy.Set(maxVal)
	lanes := hwy.MaxLanes[T]()

	for y := 0; y < img.height; y++ {
		inRow := img.Row(y)
		outRow := out.Row(y)
		width := img.width
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			v := hwy.Load(inRow[i:])
			result := hwy.Sub(maxVec, v)
			hwy.Store(result, outRow[i:])
		}

		// Handle tail elements
		if remaining := width - i; remaining > 0 {
			buf := make([]T, lanes)
			copy(buf, inRow[i:i+remaining])
			v := hwy.Load(buf)
			result := hwy.Sub(maxVec, v)
			hwy.Store(result, buf)
			copy(outRow[i:i+remaining], buf[:remaining])
		}
	}
}

// BaseAbs computes absolute value of each pixel: out = |in|.
func BaseAbs[T hwy.Floats](img, out *Image[T]) {
	if img == nil || out == nil || img.data == nil || out.data == nil {
		return
	}

	lanes := hwy.MaxLanes[T]()

	for y := 0; y < img.height; y++ {
		inRow := img.Row(y)
		outRow := out.Row(y)
		width := img.width
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			v := hwy.Load(inRow[i:])
			result := hwy.Abs(v)
			hwy.Store(result, outRow[i:])
		}

		// Handle tail elements
		if remaining := width - i; remaining > 0 {
			buf := make([]T, lanes)
			copy(buf, inRow[i:i+remaining])
			v := hwy.Load(buf)
			result := hwy.Abs(v)
			hwy.Store(result, buf)
			copy(outRow[i:i+remaining], buf[:remaining])
		}
	}
}

// BaseScale multiplies all pixels by a constant: out = in * scale.
func BaseScale[T hwy.Floats](img, out *Image[T], scale T) {
	if img == nil || out == nil || img.data == nil || out.data == nil {
		return
	}

	scaleVec := hwy.Set(scale)
	lanes := hwy.MaxLanes[T]()

	for y := 0; y < img.height; y++ {
		inRow := img.Row(y)
		outRow := out.Row(y)
		width := img.width
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			v := hwy.Load(inRow[i:])
			result := hwy.Mul(v, scaleVec)
			hwy.Store(result, outRow[i:])
		}

		// Handle tail elements
		if remaining := width - i; remaining > 0 {
			buf := make([]T, lanes)
			copy(buf, inRow[i:i+remaining])
			v := hwy.Load(buf)
			result := hwy.Mul(v, scaleVec)
			hwy.Store(result, buf)
			copy(outRow[i:i+remaining], buf[:remaining])
		}
	}
}

// BaseOffset adds a constant to all pixels: out = in + offset.
func BaseOffset[T hwy.Floats](img, out *Image[T], offset T) {
	if img == nil || out == nil || img.data == nil || out.data == nil {
		return
	}

	offsetVec := hwy.Set(offset)
	lanes := hwy.MaxLanes[T]()

	for y := 0; y < img.height; y++ {
		inRow := img.Row(y)
		outRow := out.Row(y)
		width := img.width
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			v := hwy.Load(inRow[i:])
			result := hwy.Add(v, offsetVec)
			hwy.Store(result, outRow[i:])
		}

		// Handle tail elements
		if remaining := width - i; remaining > 0 {
			buf := make([]T, lanes)
			copy(buf, inRow[i:i+remaining])
			v := hwy.Load(buf)
			result := hwy.Add(v, offsetVec)
			hwy.Store(result, buf)
			copy(outRow[i:i+remaining], buf[:remaining])
		}
	}
}

// BaseGamma applies gamma correction: out = pow(in, gamma).
// Input should be in [0, 1] range for proper gamma correction.
// Uses SIMD pow for vectorized processing.
func BaseGamma[T hwy.Floats](img, out *Image[T], gamma T) {
	if img == nil || out == nil || img.data == nil || out.data == nil {
		return
	}

	gammaVec := hwy.Set(gamma)
	lanes := hwy.MaxLanes[T]()

	for y := 0; y < img.height; y++ {
		inRow := img.Row(y)
		outRow := out.Row(y)
		width := img.width
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			v := hwy.Load(inRow[i:])
			result := hwy.Pow(v, gammaVec)
			hwy.Store(result, outRow[i:])
		}

		// Handle tail elements via buffer
		if remaining := width - i; remaining > 0 {
			buf := make([]T, lanes)
			copy(buf, inRow[i:i+remaining])
			v := hwy.Load(buf)
			result := hwy.Pow(v, gammaVec)
			hwy.Store(result, buf)
			copy(outRow[i:i+remaining], buf[:remaining])
		}
	}
}

// BaseMinImage computes element-wise minimum: out = min(a, b).
func BaseMinImage[T hwy.Floats](a, b, out *Image[T]) {
	if a == nil || b == nil || out == nil || a.data == nil || b.data == nil || out.data == nil {
		return
	}

	lanes := hwy.MaxLanes[T]()
	height := min(a.height, b.height)

	for y := range height {
		aRow := a.Row(y)
		bRow := b.Row(y)
		outRow := out.Row(y)
		width := min(a.width, b.width)
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			va := hwy.Load(aRow[i:])
			vb := hwy.Load(bRow[i:])
			result := hwy.Min(va, vb)
			hwy.Store(result, outRow[i:])
		}

		// Handle tail elements
		if remaining := width - i; remaining > 0 {
			bufA := make([]T, lanes)
			bufB := make([]T, lanes)
			bufOut := make([]T, lanes)
			copy(bufA, aRow[i:i+remaining])
			copy(bufB, bRow[i:i+remaining])
			va := hwy.Load(bufA)
			vb := hwy.Load(bufB)
			result := hwy.Min(va, vb)
			hwy.Store(result, bufOut)
			copy(outRow[i:i+remaining], bufOut[:remaining])
		}
	}
}

// BaseMaxImage computes element-wise maximum: out = max(a, b).
func BaseMaxImage[T hwy.Floats](a, b, out *Image[T]) {
	if a == nil || b == nil || out == nil || a.data == nil || b.data == nil || out.data == nil {
		return
	}

	lanes := hwy.MaxLanes[T]()
	height := min(a.height, b.height)

	for y := range height {
		aRow := a.Row(y)
		bRow := b.Row(y)
		outRow := out.Row(y)
		width := min(a.width, b.width)
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			va := hwy.Load(aRow[i:])
			vb := hwy.Load(bRow[i:])
			result := hwy.Max(va, vb)
			hwy.Store(result, outRow[i:])
		}

		// Handle tail elements
		if remaining := width - i; remaining > 0 {
			bufA := make([]T, lanes)
			bufB := make([]T, lanes)
			bufOut := make([]T, lanes)
			copy(bufA, aRow[i:i+remaining])
			copy(bufB, bRow[i:i+remaining])
			va := hwy.Load(bufA)
			vb := hwy.Load(bufB)
			result := hwy.Max(va, vb)
			hwy.Store(result, bufOut)
			copy(outRow[i:i+remaining], bufOut[:remaining])
		}
	}
}
