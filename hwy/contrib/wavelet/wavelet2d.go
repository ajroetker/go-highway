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

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/image"
)

// PhaseFunc returns the horizontal and vertical phase for a given decomposition level.
// Level 0 is the finest level (original resolution), higher levels are coarser.
type PhaseFunc func(level int) (phaseH, phaseV int)

// Synthesize2D_53 applies the inverse 5/3 2D wavelet transform.
// Reconstructs from levels decomposition levels to the original image.
// phaseFn provides the phase for each level based on tile/subband position.
func Synthesize2D_53(img *image.Image[int32], levels int, phaseFn PhaseFunc) {
	if img == nil || img.Width() == 0 || img.Height() == 0 || levels == 0 {
		return
	}

	// Process from coarsest to finest level
	// level goes from (levels-1) down to 0
	// At level (levels-1) we operate on the smallest LL subband
	// At level 0 we operate on the full image
	for level := levels - 1; level >= 0; level-- {
		phaseH, phaseV := phaseFn(level)
		// levelDim(dim, level) calculates size after 'level' halvings
		// For synthesis at current level, we need size after (level) halvings
		levelWidth := levelDim(img.Width(), level)
		levelHeight := levelDim(img.Height(), level)

		// Vertical pass first (on columns)
		col := make([]int32, levelHeight)
		for x := 0; x < levelWidth; x++ {
			// Extract column
			for y := 0; y < levelHeight; y++ {
				col[y] = img.At(x, y)
			}
			// Transform
			Synthesize53(col, phaseV)
			// Write back
			for y := 0; y < levelHeight; y++ {
				img.Set(x, y, col[y])
			}
		}

		// Horizontal pass (on rows)
		for y := 0; y < levelHeight; y++ {
			row := img.Row(y)[:levelWidth]
			Synthesize53(row, phaseH)
		}
	}
}

// Analyze2D_53 applies the forward 5/3 2D wavelet transform.
// Decomposes the image into the specified number of levels.
// phaseFn provides the phase for each level based on tile/subband position.
func Analyze2D_53(img *image.Image[int32], levels int, phaseFn PhaseFunc) {
	if img == nil || img.Width() == 0 || img.Height() == 0 || levels == 0 {
		return
	}

	// Process from finest to coarsest level
	for level := 0; level < levels; level++ {
		phaseH, phaseV := phaseFn(level)
		levelWidth := levelDim(img.Width(), level)
		levelHeight := levelDim(img.Height(), level)

		if levelWidth < 2 || levelHeight < 2 {
			break
		}

		// Horizontal pass first (on rows)
		for y := 0; y < levelHeight; y++ {
			row := img.Row(y)[:levelWidth]
			Analyze53(row, phaseH)
		}

		// Vertical pass (on columns)
		col := make([]int32, levelHeight)
		for x := 0; x < levelWidth; x++ {
			// Extract column
			for y := 0; y < levelHeight; y++ {
				col[y] = img.At(x, y)
			}
			// Transform
			Analyze53(col, phaseV)
			// Write back
			for y := 0; y < levelHeight; y++ {
				img.Set(x, y, col[y])
			}
		}
	}
}

// Synthesize2D_97 applies the inverse 9/7 2D wavelet transform.
// Reconstructs from levels decomposition levels to the original image.
// phaseFn provides the phase for each level based on tile/subband position.
func Synthesize2D_97[T hwy.Floats](img *image.Image[T], levels int, phaseFn PhaseFunc) {
	if img == nil || img.Width() == 0 || img.Height() == 0 || levels == 0 {
		return
	}

	// Process from coarsest to finest level
	for level := levels - 1; level >= 0; level-- {
		phaseH, phaseV := phaseFn(level)
		levelWidth := levelDim(img.Width(), level)
		levelHeight := levelDim(img.Height(), level)

		// Vertical pass first
		col := make([]T, levelHeight)
		for x := 0; x < levelWidth; x++ {
			// Extract column
			for y := 0; y < levelHeight; y++ {
				col[y] = img.At(x, y)
			}
			// Transform
			Synthesize97(col, phaseV)
			// Write back
			for y := 0; y < levelHeight; y++ {
				img.Set(x, y, col[y])
			}
		}

		// Horizontal pass
		for y := 0; y < levelHeight; y++ {
			row := img.Row(y)[:levelWidth]
			Synthesize97(row, phaseH)
		}
	}
}

// Analyze2D_97 applies the forward 9/7 2D wavelet transform.
// Decomposes the image into the specified number of levels.
// phaseFn provides the phase for each level based on tile/subband position.
func Analyze2D_97[T hwy.Floats](img *image.Image[T], levels int, phaseFn PhaseFunc) {
	if img == nil || img.Width() == 0 || img.Height() == 0 || levels == 0 {
		return
	}

	// Process from finest to coarsest level
	for level := 0; level < levels; level++ {
		phaseH, phaseV := phaseFn(level)
		levelWidth := levelDim(img.Width(), level)
		levelHeight := levelDim(img.Height(), level)

		if levelWidth < 2 || levelHeight < 2 {
			break
		}

		// Horizontal pass first
		for y := 0; y < levelHeight; y++ {
			row := img.Row(y)[:levelWidth]
			Analyze97(row, phaseH)
		}

		// Vertical pass
		col := make([]T, levelHeight)
		for x := 0; x < levelWidth; x++ {
			// Extract column
			for y := 0; y < levelHeight; y++ {
				col[y] = img.At(x, y)
			}
			// Transform
			Analyze97(col, phaseV)
			// Write back
			for y := 0; y < levelHeight; y++ {
				img.Set(x, y, col[y])
			}
		}
	}
}

// levelDim calculates the dimension at a given decomposition level.
// Level 0 is the original dimension, level 1 is (dim+1)/2, etc.
func levelDim(dim, level int) int {
	for i := 0; i < level; i++ {
		dim = (dim + 1) / 2
	}
	return dim
}
