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

package matmul

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// TestKernelDirect tests the micro-kernel directly with known inputs.
// This helps diagnose platform-specific issues.
func TestKernelDirect(t *testing.T) {
	lanes := hwy.Zero[float32]().NumLanes()
	t.Logf("lanes=%d, CurrentName=%s", lanes, hwy.CurrentName())

	// Simple 2x2 matmul: C = A * B
	// A = [[1, 2], [3, 4]]  (2x2)
	// B = [[5, 6], [7, 8]]  (2x2)
	// C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]

	// Pack A with mr=4 (padding with zeros)
	// Layout: [k, mr] -> for k=0: [1,3,0,0], for k=1: [2,4,0,0]
	packedA := []float32{1, 3, 0, 0, 2, 4, 0, 0}

	// Pack B with nr=8 (padding with zeros)
	// Layout: [k, nr] -> for k=0: [5,6,0,0,0,0,0,0], for k=1: [7,8,0,0,0,0,0,0]
	packedB := []float32{5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0}

	// Output C (n=2, so row stride is 2)
	c := make([]float32, 4*2) // 4 rows (mr) x 2 cols (n)
	n := 2

	// Call kernel: ir=0, jr=0, kc=2, mr=4, nr=8
	// But we only have 2 active rows and 2 active columns
	t.Logf("Calling PackedMicroKernelPartial with activeRows=2, activeCols=2")
	PackedMicroKernelPartial(packedA, packedB, c, n, 0, 0, 2, 4, 8, 2, 2)

	// Check results
	expected := []float32{19, 22, 43, 50, 0, 0, 0, 0}
	t.Logf("c = %v", c)
	t.Logf("expected = %v", expected)

	for i := 0; i < 4; i++ {
		if c[i] != expected[i] {
			t.Errorf("c[%d] = %f, want %f", i, c[i], expected[i])
		}
	}
}

// TestBaseKernelGeneral directly tests basePackedMicroKernelGeneral
// which is used when lanes != nr/2.
func TestBaseKernelGeneral(t *testing.T) {
	lanes := hwy.Zero[float32]().NumLanes()
	t.Logf("lanes=%d", lanes)

	// Simple 2x2 matmul with mr=4, nr=8 (like fallback params)
	// A = [[1, 2], [3, 4]]  packed as [k, mr]
	// B = [[5, 6], [7, 8]]  packed as [k, nr]
	packedA := []float32{1, 3, 0, 0, 2, 4, 0, 0} // k=0: [1,3,0,0], k=1: [2,4,0,0]
	packedB := []float32{5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0}

	c := make([]float32, 4*8) // 4 rows x 8 cols (full micro-tile output space)
	n := 8                    // Leading dimension of C

	// Call the general kernel directly
	basePackedMicroKernelGeneral(packedA, packedB, c, n, 0, 0, 2, 4, 8)

	// Expected: C[0,0]=19, C[0,1]=22, C[1,0]=43, C[1,1]=50, rest=0
	// In row-major with n=8: c[0]=19, c[1]=22, c[8]=43, c[9]=50
	t.Logf("c[0:16] = %v", c[0:16])

	if c[0] != 19 {
		t.Errorf("c[0,0] = %f, want 19", c[0])
	}
	if c[1] != 22 {
		t.Errorf("c[0,1] = %f, want 22", c[1])
	}
	if c[8] != 43 {
		t.Errorf("c[1,0] = %f, want 43", c[8])
	}
	if c[9] != 50 {
		t.Errorf("c[1,1] = %f, want 50", c[9])
	}
}

// TestScalarMatmulReference computes the expected result using pure scalar operations.
// This serves as a known-good reference for comparison.
func TestScalarMatmulReference(t *testing.T) {
	lanes := hwy.Zero[float32]().NumLanes()
	t.Logf("lanes=%d, CurrentName=%s", lanes, hwy.CurrentName())

	// Use the packed data layout as in other tests
	// A = [[1, 2], [3, 4]] (2 rows, k=2)
	// B = [[5, 6], [7, 8]] (k=2, 2 cols)
	// C = A*B = [[19, 22], [43, 50]]

	// Packed A with mr=4: [k, mr] layout
	packedA := []float32{1, 3, 0, 0, 2, 4, 0, 0}
	// Packed B with nr=8: [k, nr] layout
	packedB := []float32{5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0}

	mr, nr, kc := 4, 8, 2

	// Compute C manually using scalar loop (same algorithm as basePackedMicroKernelGeneral but without hwy)
	c := make([]float32, 4*8)
	n := 8

	for r := 0; r < mr; r++ {
		cRowStart := r * n
		for col := 0; col < nr; col++ {
			var sum float32
			for p := 0; p < kc; p++ {
				aVal := packedA[p*mr+r]
				bVal := packedB[p*nr+col]
				sum += aVal * bVal
			}
			c[cRowStart+col] += sum
		}
	}

	t.Logf("Scalar result c[0:16] = %v", c[0:16])

	// Verify scalar computation matches expected
	if c[0] != 19 {
		t.Errorf("Scalar: c[0,0] = %f, want 19", c[0])
	}
	if c[1] != 22 {
		t.Errorf("Scalar: c[0,1] = %f, want 22", c[1])
	}
	if c[8] != 43 {
		t.Errorf("Scalar: c[1,0] = %f, want 43", c[8])
	}
	if c[9] != 50 {
		t.Errorf("Scalar: c[1,1] = %f, want 50", c[9])
	}

	// Now test the general kernel and compare
	c2 := make([]float32, 4*8)
	basePackedMicroKernelGeneral(packedA, packedB, c2, n, 0, 0, kc, mr, nr)

	t.Logf("General kernel c2[0:16] = %v", c2[0:16])

	// Compare scalar vs general kernel
	for i := 0; i < 16; i++ {
		if c[i] != c2[i] {
			t.Errorf("Mismatch at c[%d]: scalar=%f, general=%f", i, c[i], c2[i])
		}
	}
}

// TestPackedMatMulSmall tests packed matmul with a small matrix that spans
// multiple micro-panels to exercise the GEBP loop structure.
func TestPackedMatMulSmall(t *testing.T) {
	lanes := hwy.Zero[float32]().NumLanes()
	params := getCacheParams[float32]()
	t.Logf("lanes=%d, CurrentName=%s", lanes, hwy.CurrentName())
	t.Logf("CacheParams: Mr=%d, Nr=%d, Kc=%d, Mc=%d, Nc=%d",
		params.Mr, params.Nr, params.Kc, params.Mc, params.Nc)

	// 8x8 matmul with simple values
	// This spans multiple micro-panels: with Mr=4, we have 2 row panels
	// With Nr=8, we have 1 column panel
	m, n, k := 8, 8, 4

	// A is 8x4, B is 4x8, C is 8x8
	// Use simple values: A[i,j] = i*k + j + 1, B[i,j] = i*n + j + 1
	a := make([]float32, m*k)
	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			a[i*k+j] = float32(i*k + j + 1)
		}
	}

	b := make([]float32, k*n)
	for i := 0; i < k; i++ {
		for j := 0; j < n; j++ {
			b[i*n+j] = float32(i*n + j + 1)
		}
	}

	// Compute expected result using simple triple loop
	expected := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for kk := 0; kk < k; kk++ {
				sum += a[i*k+kk] * b[kk*n+j]
			}
			expected[i*n+j] = sum
		}
	}

	t.Logf("A (8x4):")
	for i := 0; i < m; i++ {
		t.Logf("  row %d: %v", i, a[i*k:(i+1)*k])
	}

	t.Logf("B (4x8):")
	for i := 0; i < k; i++ {
		t.Logf("  row %d: %v", i, b[i*n:(i+1)*n])
	}

	t.Logf("Expected C (8x8):")
	for i := 0; i < m; i++ {
		t.Logf("  row %d: %v", i, expected[i*n:(i+1)*n])
	}

	// Run packed matmul
	c := make([]float32, m*n)
	PackedMatMul(a, b, c, m, n, k)

	t.Logf("Actual C (8x8):")
	for i := 0; i < m; i++ {
		t.Logf("  row %d: %v", i, c[i*n:(i+1)*n])
	}

	// Compare
	var maxErr float32
	var maxErrI, maxErrJ int
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			idx := i*n + j
			diff := c[idx] - expected[idx]
			if diff < 0 {
				diff = -diff
			}
			if diff > maxErr {
				maxErr = diff
				maxErrI, maxErrJ = i, j
			}
		}
	}

	t.Logf("Max error: %f at C[%d,%d] (expected=%f, got=%f)",
		maxErr, maxErrI, maxErrJ, expected[maxErrI*n+maxErrJ], c[maxErrI*n+maxErrJ])

	if maxErr > 1e-4 {
		t.Errorf("max error %f exceeds threshold", maxErr)
	}
}

// TestPackedMatMul16x16Deterministic tests with 16x16 deterministic values
// to isolate whether the bug is size-dependent or random-value-dependent.
func TestPackedMatMul16x16Deterministic(t *testing.T) {
	lanes := hwy.Zero[float32]().NumLanes()
	params := getCacheParams[float32]()
	t.Logf("HWY_NO_SIMD=%q", os.Getenv("HWY_NO_SIMD"))
	t.Logf("lanes=%d, CurrentName=%s", lanes, hwy.CurrentName())
	t.Logf("CacheParams: Mr=%d, Nr=%d, Kc=%d, Mc=%d, Nc=%d",
		params.Mr, params.Nr, params.Kc, params.Mc, params.Nc)

	// 16x16 matmul - this spans:
	// - 4 row panels (16/4 = 4) with Mr=4
	// - 2 column panels (16/8 = 2) with Nr=8
	m, n, k := 16, 16, 16

	// Use simple deterministic values
	a := make([]float32, m*k)
	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			a[i*k+j] = float32(i*k + j + 1)
		}
	}

	b := make([]float32, k*n)
	for i := 0; i < k; i++ {
		for j := 0; j < n; j++ {
			b[i*n+j] = float32(i*n + j + 1)
		}
	}

	// Compute expected result using simple triple loop
	expected := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for kk := 0; kk < k; kk++ {
				sum += a[i*k+kk] * b[kk*n+j]
			}
			expected[i*n+j] = sum
		}
	}

	// Run packed matmul
	c := make([]float32, m*n)
	PackedMatMul(a, b, c, m, n, k)

	// Compare and find max error
	var maxErr float32
	var maxErrI, maxErrJ int
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			idx := i*n + j
			diff := c[idx] - expected[idx]
			if diff < 0 {
				diff = -diff
			}
			if diff > maxErr {
				maxErr = diff
				maxErrI, maxErrJ = i, j
			}
		}
	}

	t.Logf("Max error: %f at C[%d,%d] (expected=%f, got=%f)",
		maxErr, maxErrI, maxErrJ, expected[maxErrI*n+maxErrJ], c[maxErrI*n+maxErrJ])

	// If error is non-zero, print the first few mismatches
	if maxErr > 1e-4 {
		t.Logf("First mismatches:")
		count := 0
		for i := 0; i < m && count < 10; i++ {
			for j := 0; j < n && count < 10; j++ {
				idx := i*n + j
				diff := c[idx] - expected[idx]
				if diff < 0 {
					diff = -diff
				}
				if diff > 1e-4 {
					t.Logf("  C[%d,%d]: expected=%f, got=%f, diff=%f", i, j, expected[idx], c[idx], diff)
					count++
				}
			}
		}
		t.Errorf("max error %f exceeds threshold", maxErr)
	}
}

// TestPackedMatMulDiagnostic isolates packing and micro-kernel to find bugs.
func TestPackedMatMulDiagnostic(t *testing.T) {
	t.Logf("=== Environment ===")
	t.Logf("HWY_NO_SIMD=%q", os.Getenv("HWY_NO_SIMD"))
	t.Logf("lanes=%d, CurrentName=%s", hwy.Zero[float32]().NumLanes(), hwy.CurrentName())

	params := getCacheParams[float32]()
	mr, nr := params.Mr, params.Nr
	t.Logf("CacheParams: Mr=%d, Nr=%d", mr, nr)

	// Simple 16x16 case: 2 B panels (columns 0-7 and 8-15)
	m, n, k := 16, 16, 16

	// Create deterministic inputs
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32(i + 1)
	}
	for i := range b {
		b[i] = float32(i + 1)
	}

	// Test RHS packing for both B panels
	t.Logf("=== Testing RHS Packing ===")
	packedBSize := params.PackedBSize()
	packedB := make([]float32, packedBSize)

	// Pack B: all 16 columns
	activeColsLast := PackRHS(b, packedB, k, n, 0, 0, k, n, nr)
	t.Logf("PackRHS returned activeColsLast=%d", activeColsLast)

	// Check first B panel (columns 0-7) at offset 0
	t.Logf("First B panel (cols 0-7), first k-row:")
	t.Logf("  packedB[0:8] = %v", packedB[0:8])

	// Check second B panel (columns 8-15) at offset k*nr = 16*8 = 128
	bPanel1Offset := k * nr
	t.Logf("Second B panel (cols 8-15), first k-row (offset=%d):", bPanel1Offset)
	t.Logf("  packedB[%d:%d] = %v", bPanel1Offset, bPanel1Offset+8, packedB[bPanel1Offset:bPanel1Offset+8])

	// Verify packing is correct
	// For k-row 0, columns 8-15: b[0*16+8..15] = b[8:16] = [9,10,11,12,13,14,15,16]
	expectedB1 := []float32{9, 10, 11, 12, 13, 14, 15, 16}
	packingOK := true
	for i, exp := range expectedB1 {
		if packedB[bPanel1Offset+i] != exp {
			t.Errorf("packedB[%d] = %f, want %f", bPanel1Offset+i, packedB[bPanel1Offset+i], exp)
			packingOK = false
		}
	}
	if packingOK {
		t.Logf("Second B panel packing: CORRECT")
	}

	// Test micro-kernel for second B panel
	t.Logf("=== Testing Micro-Kernel for Second B Panel ===")

	// Pack ALL of A (all 16 rows)
	packedASize := params.PackedASize()
	packedA := make([]float32, packedASize)
	panelRows := m // all 16 rows
	activeRowsLast := PackLHS(a, packedA, m, k, 0, 0, panelRows, k, mr)
	t.Logf("PackLHS returned activeRowsLast=%d", activeRowsLast)

	// Initialize C to zero
	c := make([]float32, m*n)

	// Test iPanel=0 (rows 0-3) with jPanel=1 (columns 8-15)
	t.Logf("--- Testing iPanel=0 (rows 0-3) with jPanel=1 (cols 8-15) ---")
	aPanelOffset0 := 0 * k * mr // 0
	PackedMicroKernel(packedA[aPanelOffset0:], packedB[bPanel1Offset:], c, n, 0, 8, k, mr, nr)
	t.Logf("After micro-kernel, C[0, 8:16] = %v", c[8:16])

	// Compute expected for C[0,8]
	var expectedC08 float32
	for kk := 0; kk < k; kk++ {
		expectedC08 += a[0*k+kk] * b[kk*n+8]
	}
	t.Logf("Expected C[0,8] = %f, got %f", expectedC08, c[8])
	if c[8] == 0 && expectedC08 != 0 {
		t.Errorf("MICRO-KERNEL BUG (iPanel=0): C[0,8] is 0 but should be %f", expectedC08)
	}

	// Test iPanel=3 (rows 12-15) with jPanel=1 (columns 8-15) - THE FAILING CASE
	t.Logf("--- Testing iPanel=3 (rows 12-15) with jPanel=1 (cols 8-15) ---")
	aPanelOffset3 := 3 * k * mr // 3 * 16 * 4 = 192
	t.Logf("aPanelOffset3 = %d", aPanelOffset3)
	t.Logf("packedA[%d:%d] = %v", aPanelOffset3, aPanelOffset3+8, packedA[aPanelOffset3:aPanelOffset3+8])

	PackedMicroKernel(packedA[aPanelOffset3:], packedB[bPanel1Offset:], c, n, 12, 8, k, mr, nr)
	t.Logf("After micro-kernel, C[12, 8:16] = %v", c[12*n+8:12*n+16])

	// Compute expected for C[12,8]
	var expectedC128 float32
	for kk := 0; kk < k; kk++ {
		expectedC128 += a[12*k+kk] * b[kk*n+8]
	}
	t.Logf("Expected C[12,8] = %f, got %f", expectedC128, c[12*n+8])
	if c[12*n+8] == 0 && expectedC128 != 0 {
		t.Errorf("MICRO-KERNEL BUG (iPanel=3): C[12,8] is 0 but should be %f", expectedC128)
	}
}

// TestPackLHS verifies that LHS packing produces the expected layout.
func TestPackLHS(t *testing.T) {
	// 6x4 matrix A, Mr=2
	// A = [[ 1  2  3  4]
	//      [ 5  6  7  8]
	//      [ 9 10 11 12]
	//      [13 14 15 16]
	//      [17 18 19 20]
	//      [21 22 23 24]]
	m, k := 6, 4
	mr := 2
	a := make([]float32, m*k)
	for i := range a {
		a[i] = float32(i + 1)
	}

	// Pack entire A (rowStart=0, colStart=0, panelRows=m, panelK=k)
	numPanels := (m + mr - 1) / mr
	packed := make([]float32, numPanels*k*mr)

	activeRows := BasePackLHS(a, packed, m, k, 0, 0, m, k, mr)

	// Expected packed layout: [panel, k, mr]
	// Panel 0: rows 0-1 -> for k=0: [1,5], k=1: [2,6], k=2: [3,7], k=3: [4,8]
	// Panel 1: rows 2-3 -> for k=0: [9,13], k=1: [10,14], k=2: [11,15], k=3: [12,16]
	// Panel 2: rows 4-5 -> for k=0: [17,21], k=1: [18,22], k=2: [19,23], k=3: [20,24]
	expected := []float32{
		1, 5, 2, 6, 3, 7, 4, 8, // Panel 0
		9, 13, 10, 14, 11, 15, 12, 16, // Panel 1
		17, 21, 18, 22, 19, 23, 20, 24, // Panel 2
	}

	if activeRows != mr {
		t.Errorf("activeRows = %d, want %d", activeRows, mr)
	}

	for i := range expected {
		if packed[i] != expected[i] {
			t.Errorf("packed[%d] = %f, want %f", i, packed[i], expected[i])
		}
	}
}

// TestPackRHS verifies that RHS packing produces the expected layout.
func TestPackRHS(t *testing.T) {
	// 4x6 matrix B, Nr=2
	// B = [[ 1  2  3  4  5  6]
	//      [ 7  8  9 10 11 12]
	//      [13 14 15 16 17 18]
	//      [19 20 21 22 23 24]]
	k, n := 4, 6
	nr := 2
	b := make([]float32, k*n)
	for i := range b {
		b[i] = float32(i + 1)
	}

	// Pack entire B (rowStart=0, colStart=0, panelK=k, panelCols=n)
	numPanels := (n + nr - 1) / nr
	packed := make([]float32, numPanels*k*nr)

	activeCols := BasePackRHS(b, packed, k, n, 0, 0, k, n, nr)

	// Expected packed layout: [panel, k, nr]
	// Panel 0: cols 0-1 -> for k=0: [1,2], k=1: [7,8], k=2: [13,14], k=3: [19,20]
	// Panel 1: cols 2-3 -> for k=0: [3,4], k=1: [9,10], k=2: [15,16], k=3: [21,22]
	// Panel 2: cols 4-5 -> for k=0: [5,6], k=1: [11,12], k=2: [17,18], k=3: [23,24]
	expected := []float32{
		1, 2, 7, 8, 13, 14, 19, 20, // Panel 0
		3, 4, 9, 10, 15, 16, 21, 22, // Panel 1
		5, 6, 11, 12, 17, 18, 23, 24, // Panel 2
	}

	if activeCols != nr {
		t.Errorf("activeCols = %d, want %d", activeCols, nr)
	}

	for i := range expected {
		if packed[i] != expected[i] {
			t.Errorf("packed[%d] = %f, want %f", i, packed[i], expected[i])
		}
	}
}

// TestPackedMatMul verifies packed matmul produces correct results.
func TestPackedMatMul(t *testing.T) {
	t.Logf("Dispatch level: %s", hwy.CurrentName())

	// Use a seeded random source for reproducibility across environments
	rng := rand.New(rand.NewSource(42))

	sizes := []int{16, 32, 48, 64, 96, 128, 256}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			b := make([]float32, k*n)
			c := make([]float32, m*n)
			expected := make([]float32, m*n)

			for i := range a {
				a[i] = rng.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rng.Float32()*2 - 1
			}

			matmulReference(a, b, expected, m, n, k)
			PackedMatMul(a, b, c, m, n, k)

			var maxErr float32
			var maxErrIdx int
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
					maxErrIdx = i
				}
			}

			t.Logf("size %dx%d: max error %e at index %d (expected=%f, got=%f)",
				size, size, maxErr, maxErrIdx, expected[maxErrIdx], c[maxErrIdx])

			// Allow accumulated floating point error proportional to K
			tolerance := float32(1e-4) * float32(k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

// TestPackedMatMulNonSquare verifies packed matmul with non-square matrices.
func TestPackedMatMulNonSquare(t *testing.T) {
	testCases := []struct {
		m, n, k int
	}{
		{64, 128, 32},
		{128, 64, 96},
		{100, 200, 150},
		{37, 53, 41}, // Odd sizes to test edge handling
		{256, 512, 128},
	}

	for _, tc := range testCases {
		name := sizeStr(tc.m) + "x" + sizeStr(tc.n) + "x" + sizeStr(tc.k)
		t.Run(name, func(t *testing.T) {
			a := make([]float32, tc.m*tc.k)
			b := make([]float32, tc.k*tc.n)
			c := make([]float32, tc.m*tc.n)
			expected := make([]float32, tc.m*tc.n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulReference(a, b, expected, tc.m, tc.n, tc.k)
			PackedMatMul(a, b, c, tc.m, tc.n, tc.k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			t.Logf("size %dx%dx%d: max error %e", tc.m, tc.n, tc.k, maxErr)

			tolerance := float32(1e-4) * float32(tc.k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

// TestParallelPackedMatMul verifies parallel packed matmul produces correct results.
func TestParallelPackedMatMul(t *testing.T) {
	sizes := []int{256, 512}

	for _, size := range sizes {
		t.Run(sizeStr(size), func(t *testing.T) {
			m, n, k := size, size, size

			a := make([]float32, m*k)
			b := make([]float32, k*n)
			c := make([]float32, m*n)
			expected := make([]float32, m*n)

			for i := range a {
				a[i] = rand.Float32()*2 - 1
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}

			matmulReference(a, b, expected, m, n, k)
			ParallelPackedMatMul(a, b, c, m, n, k)

			var maxErr float32
			for i := range c {
				err := float32(math.Abs(float64(c[i] - expected[i])))
				if err > maxErr {
					maxErr = err
				}
			}

			t.Logf("size %dx%d: max error %e", size, size, maxErr)

			tolerance := float32(1e-4) * float32(k)
			if maxErr > tolerance {
				t.Errorf("max error %e exceeds threshold %e", maxErr, tolerance)
			}
		})
	}
}

// BenchmarkPackedMatMul benchmarks the packed matmul.
func BenchmarkPackedMatMul(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{128, 256, 512, 1024}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size), func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				PackedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkParallelPackedMatMul benchmarks the parallel packed matmul.
func BenchmarkParallelPackedMatMul(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{256, 512, 1024}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size), func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				ParallelPackedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkPackedVsBlocked compares packed and blocked matmul side-by-side.
func BenchmarkPackedVsBlocked(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{256, 512, 1024}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size)+"/Blocked", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				BlockedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/Packed", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				PackedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/ParallelBlocked", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				ParallelMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/ParallelPacked", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				ParallelPackedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkAllAlgorithms compares all matmul algorithms.
func BenchmarkAllAlgorithms(b *testing.B) {
	b.Logf("Dispatch level: %s", hwy.CurrentName())

	sizes := []int{64, 128, 256, 512, 1024}

	for _, size := range sizes {
		m, n, k := size, size, size

		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(sizeStr(size)+"/Streaming", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/Blocked", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				BlockedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/Packed", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				PackedMatMul(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(sizeStr(size)+"/Auto", func(b *testing.B) {
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatMulAuto(a, bMat, c, m, n, k)
			}

			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkPacking benchmarks the packing operations themselves.
func BenchmarkPacking(b *testing.B) {
	size := 512
	m, n, k := size, size, size

	a := make([]float32, m*k)
	bMat := make([]float32, k*n)

	for i := range a {
		a[i] = rand.Float32()
	}
	for i := range bMat {
		bMat[i] = rand.Float32()
	}

	params := getCacheParams[float32]()
	packedA := make([]float32, params.PackedASize())
	packedB := make([]float32, params.PackedBSize())

	b.Run("PackLHS", func(b *testing.B) {
		b.SetBytes(int64(params.Mc * params.Kc * 4))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			PackLHS(a, packedA, m, k, 0, 0, params.Mc, params.Kc, params.Mr)
		}
	})

	b.Run("PackRHS", func(b *testing.B) {
		b.SetBytes(int64(params.Nc * params.Kc * 4))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			PackRHS(bMat, packedB, k, n, 0, 0, params.Kc, params.Nc, params.Nr)
		}
	})
}
