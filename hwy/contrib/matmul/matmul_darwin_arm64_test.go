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

//go:build darwin && arm64

package matmul

import (
	"math"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// TestMultiTileFMOPADirect calls the multi-tile assembly kernel directly.
func TestMultiTileFMOPADirect(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	m, n, k := 32, 32, 32
	// AT is K x M (already transposed)
	at := make([]float32, k*m)
	b := make([]float32, k*n)
	c := make([]float32, m*n)

	// Fill with simple values: identity-like
	for i := range k {
		for j := range m {
			if i == j {
				at[i*m+j] = 1.0
			}
		}
	}
	for i := range k {
		for j := range n {
			b[i*n+j] = float32(i*n + j)
		}
	}

	defer hwy.SMEGuard()()
	asm.MultiTileMatMulFMOPAF32(at, b, c, m, n, k)

	// With AT = identity transposed, C should equal B (first 32 rows)
	var maxErr float32
	for i := range m {
		for j := range n {
			expected := b[i*n+j]
			err := float32(math.Abs(float64(c[i*n+j] - expected)))
			if err > maxErr {
				maxErr = err
			}
		}
	}
	t.Logf("32x32 multi-tile direct: max error %e", maxErr)
	if maxErr > 1e-4 {
		t.Errorf("max error %e exceeds threshold", maxErr)
	}
}
