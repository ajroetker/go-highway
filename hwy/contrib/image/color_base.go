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

import (
	"github.com/ajroetker/go-highway/hwy"
)

//go:generate go run ../../../cmd/hwygen -input color_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch color

// BaseForwardRCT applies the Reversible Color Transform (RCT) from JPEG 2000.
// This transforms RGB to YCbCr using integer arithmetic:
//
//	Y  = (R + 2*G + B) >> 2
//	Cb = B - G
//	Cr = R - G
//
// The transform is lossless for integer types.
func BaseForwardRCT[T hwy.SignedInts](r, g, b, outY, outCb, outCr *Image[T]) {
	if r == nil || g == nil || b == nil || outY == nil || outCb == nil || outCr == nil {
		return
	}
	if r.data == nil || g.data == nil || b.data == nil ||
		outY.data == nil || outCb.data == nil || outCr.data == nil {
		return
	}

	twoVec := hwy.Set(T(2))
	lanes := hwy.MaxLanes[T]()
	height := r.height
	width := r.width

	for y := range height {
		rRow := r.Row(y)
		gRow := g.Row(y)
		bRow := b.Row(y)
		yRow := outY.Row(y)
		cbRow := outCb.Row(y)
		crRow := outCr.Row(y)
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			vr := hwy.Load(rRow[i:])
			vg := hwy.Load(gRow[i:])
			vb := hwy.Load(bRow[i:])

			// Y = (R + 2*G + B) >> 2
			twoG := hwy.Mul(vg, twoVec)
			sum := hwy.Add(hwy.Add(vr, twoG), vb)
			vy := hwy.ShiftRight(sum, 2)

			// Cb = B - G
			vcb := hwy.Sub(vb, vg)

			// Cr = R - G
			vcr := hwy.Sub(vr, vg)

			hwy.Store(vy, yRow[i:])
			hwy.Store(vcb, cbRow[i:])
			hwy.Store(vcr, crRow[i:])
		}

		// Handle tail elements via buffer
		if remaining := width - i; remaining > 0 {
			bufR := make([]T, lanes)
			bufG := make([]T, lanes)
			bufB := make([]T, lanes)
			bufY := make([]T, lanes)
			bufCb := make([]T, lanes)
			bufCr := make([]T, lanes)

			copy(bufR, rRow[i:i+remaining])
			copy(bufG, gRow[i:i+remaining])
			copy(bufB, bRow[i:i+remaining])

			vr := hwy.Load(bufR)
			vg := hwy.Load(bufG)
			vb := hwy.Load(bufB)

			twoG := hwy.Mul(vg, twoVec)
			sum := hwy.Add(hwy.Add(vr, twoG), vb)
			vy := hwy.ShiftRight(sum, 2)
			vcb := hwy.Sub(vb, vg)
			vcr := hwy.Sub(vr, vg)

			hwy.Store(vy, bufY)
			hwy.Store(vcb, bufCb)
			hwy.Store(vcr, bufCr)

			copy(yRow[i:i+remaining], bufY[:remaining])
			copy(cbRow[i:i+remaining], bufCb[:remaining])
			copy(crRow[i:i+remaining], bufCr[:remaining])
		}
	}
}

// BaseInverseRCT applies the inverse Reversible Color Transform (RCT).
// This transforms YCbCr back to RGB using integer arithmetic:
//
//	G = Y - ((Cb + Cr) >> 2)
//	R = Cr + G
//	B = Cb + G
//
// The transform is lossless for integer types.
func BaseInverseRCT[T hwy.SignedInts](y, cb, cr, outR, outG, outB *Image[T]) {
	if y == nil || cb == nil || cr == nil || outR == nil || outG == nil || outB == nil {
		return
	}
	if y.data == nil || cb.data == nil || cr.data == nil ||
		outR.data == nil || outG.data == nil || outB.data == nil {
		return
	}

	lanes := hwy.MaxLanes[T]()
	height := y.height
	width := y.width

	for row := range height {
		yRow := y.Row(row)
		cbRow := cb.Row(row)
		crRow := cr.Row(row)
		rRow := outR.Row(row)
		gRow := outG.Row(row)
		bRow := outB.Row(row)
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			vy := hwy.Load(yRow[i:])
			vcb := hwy.Load(cbRow[i:])
			vcr := hwy.Load(crRow[i:])

			// G = Y - ((Cb + Cr) >> 2)
			cbPlusCr := hwy.Add(vcb, vcr)
			shift := hwy.ShiftRight(cbPlusCr, 2)
			vg := hwy.Sub(vy, shift)

			// R = Cr + G
			vr := hwy.Add(vcr, vg)

			// B = Cb + G
			vb := hwy.Add(vcb, vg)

			hwy.Store(vr, rRow[i:])
			hwy.Store(vg, gRow[i:])
			hwy.Store(vb, bRow[i:])
		}

		// Handle tail elements via buffer
		if remaining := width - i; remaining > 0 {
			bufY := make([]T, lanes)
			bufCb := make([]T, lanes)
			bufCr := make([]T, lanes)
			bufR := make([]T, lanes)
			bufG := make([]T, lanes)
			bufB := make([]T, lanes)

			copy(bufY, yRow[i:i+remaining])
			copy(bufCb, cbRow[i:i+remaining])
			copy(bufCr, crRow[i:i+remaining])

			vy := hwy.Load(bufY)
			vcb := hwy.Load(bufCb)
			vcr := hwy.Load(bufCr)

			cbPlusCr := hwy.Add(vcb, vcr)
			shift := hwy.ShiftRight(cbPlusCr, 2)
			vg := hwy.Sub(vy, shift)
			vr := hwy.Add(vcr, vg)
			vb := hwy.Add(vcb, vg)

			hwy.Store(vr, bufR)
			hwy.Store(vg, bufG)
			hwy.Store(vb, bufB)

			copy(rRow[i:i+remaining], bufR[:remaining])
			copy(gRow[i:i+remaining], bufG[:remaining])
			copy(bRow[i:i+remaining], bufB[:remaining])
		}
	}
}

// BaseForwardICT applies the Irreversible Color Transform (ICT) from JPEG 2000.
// This transforms RGB to YCbCr using floating-point arithmetic:
//
//	Y  = 0.299*R + 0.587*G + 0.114*B
//	Cb = -0.16875*R - 0.33126*G + 0.5*B
//	Cr = 0.5*R - 0.41869*G - 0.08131*B
//
// Coefficients are from ITU-T T.800 Table G.2.
func BaseForwardICT[T hwy.Floats](r, g, b, outY, outCb, outCr *Image[T]) {
	if r == nil || g == nil || b == nil || outY == nil || outCb == nil || outCr == nil {
		return
	}
	if r.data == nil || g.data == nil || b.data == nil ||
		outY.data == nil || outCb.data == nil || outCr.data == nil {
		return
	}

	// Get coefficients for this type
	rToY, gToY, bToY,
		rToCb, gToCb, bToCb,
		rToCr, gToCr, bToCr,
		_, _, _, _ := ictCoeffs[T]()

	rToYVec := hwy.Set(rToY)
	gToYVec := hwy.Set(gToY)
	bToYVec := hwy.Set(bToY)
	rToCbVec := hwy.Set(rToCb)
	gToCbVec := hwy.Set(gToCb)
	bToCbVec := hwy.Set(bToCb)
	rToCrVec := hwy.Set(rToCr)
	gToCrVec := hwy.Set(gToCr)
	bToCrVec := hwy.Set(bToCr)

	lanes := hwy.MaxLanes[T]()
	height := r.height
	width := r.width

	for row := range height {
		rRow := r.Row(row)
		gRow := g.Row(row)
		bRow := b.Row(row)
		yRow := outY.Row(row)
		cbRow := outCb.Row(row)
		crRow := outCr.Row(row)
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			vr := hwy.Load(rRow[i:])
			vg := hwy.Load(gRow[i:])
			vb := hwy.Load(bRow[i:])

			// Y = rToY*R + gToY*G + bToY*B
			// Using FMA chains for efficiency
			vy := hwy.FMA(vr, rToYVec, hwy.FMA(vg, gToYVec, hwy.Mul(vb, bToYVec)))

			// Cb = rToCb*R + gToCb*G + bToCb*B
			vcb := hwy.FMA(vr, rToCbVec, hwy.FMA(vg, gToCbVec, hwy.Mul(vb, bToCbVec)))

			// Cr = rToCr*R + gToCr*G + bToCr*B
			vcr := hwy.FMA(vr, rToCrVec, hwy.FMA(vg, gToCrVec, hwy.Mul(vb, bToCrVec)))

			hwy.Store(vy, yRow[i:])
			hwy.Store(vcb, cbRow[i:])
			hwy.Store(vcr, crRow[i:])
		}

		// Handle tail elements via buffer
		if remaining := width - i; remaining > 0 {
			bufR := make([]T, lanes)
			bufG := make([]T, lanes)
			bufB := make([]T, lanes)
			bufY := make([]T, lanes)
			bufCb := make([]T, lanes)
			bufCr := make([]T, lanes)

			copy(bufR, rRow[i:i+remaining])
			copy(bufG, gRow[i:i+remaining])
			copy(bufB, bRow[i:i+remaining])

			vr := hwy.Load(bufR)
			vg := hwy.Load(bufG)
			vb := hwy.Load(bufB)

			vy := hwy.FMA(vr, rToYVec, hwy.FMA(vg, gToYVec, hwy.Mul(vb, bToYVec)))
			vcb := hwy.FMA(vr, rToCbVec, hwy.FMA(vg, gToCbVec, hwy.Mul(vb, bToCbVec)))
			vcr := hwy.FMA(vr, rToCrVec, hwy.FMA(vg, gToCrVec, hwy.Mul(vb, bToCrVec)))

			hwy.Store(vy, bufY)
			hwy.Store(vcb, bufCb)
			hwy.Store(vcr, bufCr)

			copy(yRow[i:i+remaining], bufY[:remaining])
			copy(cbRow[i:i+remaining], bufCb[:remaining])
			copy(crRow[i:i+remaining], bufCr[:remaining])
		}
	}
}

// BaseInverseICT applies the inverse Irreversible Color Transform (ICT).
// This transforms YCbCr back to RGB using floating-point arithmetic:
//
//	R = Y + 1.402*Cr
//	G = Y - 0.344136*Cb - 0.714136*Cr
//	B = Y + 1.772*Cb
//
// Coefficients are from ITU-T T.800 Table G.2.
func BaseInverseICT[T hwy.Floats](y, cb, cr, outR, outG, outB *Image[T]) {
	if y == nil || cb == nil || cr == nil || outR == nil || outG == nil || outB == nil {
		return
	}
	if y.data == nil || cb.data == nil || cr.data == nil ||
		outR.data == nil || outG.data == nil || outB.data == nil {
		return
	}

	// Get coefficients for this type
	_, _, _,
		_, _, _,
		_, _, _,
		crToR, cbToG, crToG, cbToB := ictCoeffs[T]()

	crToRVec := hwy.Set(crToR)
	cbToGVec := hwy.Set(cbToG)
	crToGVec := hwy.Set(crToG)
	cbToBVec := hwy.Set(cbToB)

	lanes := hwy.MaxLanes[T]()
	height := y.height
	width := y.width

	for row := range height {
		yRow := y.Row(row)
		cbRow := cb.Row(row)
		crRow := cr.Row(row)
		rRow := outR.Row(row)
		gRow := outG.Row(row)
		bRow := outB.Row(row)
		i := 0

		// Process full vectors
		for ; i+lanes <= width; i += lanes {
			vy := hwy.Load(yRow[i:])
			vcb := hwy.Load(cbRow[i:])
			vcr := hwy.Load(crRow[i:])

			// R = Y + crToR*Cr
			vr := hwy.FMA(vcr, crToRVec, vy)

			// G = Y + cbToG*Cb + crToG*Cr
			// = Y + (cbToG*Cb + crToG*Cr)
			vg := hwy.FMA(vcb, cbToGVec, hwy.FMA(vcr, crToGVec, vy))

			// B = Y + cbToB*Cb
			vb := hwy.FMA(vcb, cbToBVec, vy)

			hwy.Store(vr, rRow[i:])
			hwy.Store(vg, gRow[i:])
			hwy.Store(vb, bRow[i:])
		}

		// Handle tail elements via buffer
		if remaining := width - i; remaining > 0 {
			bufY := make([]T, lanes)
			bufCb := make([]T, lanes)
			bufCr := make([]T, lanes)
			bufR := make([]T, lanes)
			bufG := make([]T, lanes)
			bufB := make([]T, lanes)

			copy(bufY, yRow[i:i+remaining])
			copy(bufCb, cbRow[i:i+remaining])
			copy(bufCr, crRow[i:i+remaining])

			vy := hwy.Load(bufY)
			vcb := hwy.Load(bufCb)
			vcr := hwy.Load(bufCr)

			vr := hwy.FMA(vcr, crToRVec, vy)
			vg := hwy.FMA(vcb, cbToGVec, hwy.FMA(vcr, crToGVec, vy))
			vb := hwy.FMA(vcb, cbToBVec, vy)

			hwy.Store(vr, bufR)
			hwy.Store(vg, bufG)
			hwy.Store(vb, bufB)

			copy(rRow[i:i+remaining], bufR[:remaining])
			copy(gRow[i:i+remaining], bufG[:remaining])
			copy(bRow[i:i+remaining], bufB[:remaining])
		}
	}
}
