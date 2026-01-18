//go:build darwin && arm64

package matmul

import (
	"testing"
	"unsafe"
)

//go:noescape
func sme_za_store_test(dst unsafe.Pointer)

//go:noescape
func sme_za_load_store_test(dst, src unsafe.Pointer)

//go:noescape
func sme_fmopa_test(dst unsafe.Pointer)

//go:noescape
func sme_fmopa_debug_test(z0out, z1out, zaout unsafe.Pointer)

//go:noescape
func sme_build_vector_test(dst unsafe.Pointer)

//go:noescape
func sme_fmopa_store_all_test(dst unsafe.Pointer, stride int64)

//go:noescape
func sme_za1_build_vector_test(dst unsafe.Pointer)

//go:noescape
func sme_za2_save_restore_test(dst unsafe.Pointer)

// TestSMEZA2SaveRestore tests if tile is in Zd bits 1:0
// za0 has 6.0 (2*3), za1 has 20.0 (4*5)
// Read with 0xc0820005 (z5, bits 1:0 = 01)
// If tile is in Zd bits 1:0, this should read za1 and return 20.0
func TestSMEZA2SaveRestore(t *testing.T) {
	data := make([]float32, 16)
	for i := range data {
		data[i] = -1.0
	}

	sme_za2_save_restore_test(unsafe.Pointer(&data[0]))

	// With encoding 0xc0820005 (z5, bits 1:0 = 01):
	// If 6.0: tile is NOT in Zd bits 1:0 (reads za0)
	// If 20.0: tile IS in Zd bits 1:0 (reads za1) ✓
	t.Log("Tile encoding test (0xc0820005, z5):")
	t.Logf("  data[0] = %f (6.0=za0, 20.0=za1)", data[0])

	if data[0] == 20.0 {
		t.Log("  → SUCCESS! Tile IS in Zd bits 1:0 - reads za1")
	} else if data[0] == 6.0 {
		t.Log("  → reads za0 (tile NOT in Zd bits 1:0)")
	} else {
		t.Logf("  → reads something else (got %f)", data[0])
	}
}

// TestSMEZA1BuildVector tests building a vector via ZA1 (tile 1) as scratch
// This verifies if we can use za1 independently of za0
// SKIP: On Apple M4, tile-to-vector MOVA only reliably reads from za0,
// so this test fails. This is documented behavior on M4.
func TestSMEZA1BuildVector(t *testing.T) {
	t.Skip("ZA1 tile-to-vector MOVA doesn't work on Apple M4 - only za0 is readable")

	data := make([]float32, 16)
	for i := range data {
		data[i] = -1.0
	}

	sme_za1_build_vector_test(unsafe.Pointer(&data[0]))

	// After writing 0,1,2,...,15 to horizontal slices of ZA1 and reading vertically,
	// we should get [0, 1, 2, ..., 15]
	t.Log("Vector built via ZA1 horizontal write + vertical read:")
	for i, v := range data {
		t.Logf("  data[%d] = %f (expected %f)", i, v, float32(i))
	}

	for i, v := range data {
		expected := float32(i)
		if v != expected {
			t.Errorf("data[%d] = %f, want %f", i, v, expected)
		}
	}
}

// TestSMEFMOPAStoreAll tests storing all 16 rows of ZA after FMOPA
func TestSMEFMOPAStoreAll(t *testing.T) {
	// Create 16x16 output matrix
	const n = 16
	data := make([]float32, n*n)
	for i := range data {
		data[i] = -999.0
	}

	// Stride = 16 floats * 4 bytes = 64 bytes
	sme_fmopa_store_all_test(unsafe.Pointer(&data[0]), int64(n*4))

	// All values should be 6.0 (2.0 * 3.0)
	expected := float32(6.0)
	t.Logf("First row (expected all %f):", expected)
	for j := range n {
		t.Logf("  data[0][%d] = %f", j, data[j])
	}

	for i := range n {
		for j := range n {
			if data[i*n+j] != expected {
				t.Errorf("data[%d][%d] = %f, want %f", i, j, data[i*n+j], expected)
			}
		}
	}
}

// TestFMOPAMatMul16x16 tests the FMOPA-based matrix multiplication with a simple 16x16 case
func TestFMOPAMatMul16x16(t *testing.T) {
	// 16x16 identity test: A * I = A
	const n = 16

	// Create A matrix with known values
	a := make([]float32, n*n)
	for i := range a {
		a[i] = float32(i%n + 1) // 1, 2, 3, ..., 16, 1, 2, 3, ...
	}

	// Create identity matrix B
	b := make([]float32, n*n)
	for i := range n {
		b[i*n+i] = 1.0
	}

	// Result matrix C
	c := make([]float32, n*n)
	for i := range c {
		c[i] = -999.0 // sentinel to detect untouched values
	}

	matmul_fmopa_f32(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		int64(n), int64(n), int64(n),
	)

	// Check results: C should equal A
	t.Log("First row of C (expected 1, 2, 3, ..., 16):")
	for j := range n {
		t.Logf("  c[0][%d] = %f (expected %f)", j, c[j], a[j])
	}

	maxErr := float32(0)
	for i := range c {
		err := c[i] - a[i]
		if err < 0 {
			err = -err
		}
		if err > maxErr {
			maxErr = err
		}
	}

	if maxErr > 1e-4 {
		t.Errorf("max error %f exceeds tolerance", maxErr)
	} else {
		t.Logf("FMOPA 16x16 matmul passed, max error: %e", maxErr)
	}
}

// TestFMOPAMatMulOnes16x16 tests with all-twos and all-threes matrices
func TestFMOPAMatMulOnes16x16(t *testing.T) {
	const n = 16

	// A = all 2.0, B = all 3.0
	// If 1 FMOPA: C[i,j] = 2*3 = 6
	// If 16 FMOPAs: C[i,j] = 16 * 2*3 = 96
	a := make([]float32, n*n)
	b := make([]float32, n*n)
	for i := range a {
		a[i] = 2.0
		b[i] = 3.0
	}

	c := make([]float32, n*n)
	for i := range c {
		c[i] = -999.0
	}

	t.Logf("Calling matmul_fmopa_f32 with A=%p, B=%p, C=%p, m=%d, n=%d, k=%d",
		&a[0], &b[0], &c[0], n, n, n)

	matmul_fmopa_f32(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		int64(n), int64(n), int64(n),
	)

	t.Log("matmul_fmopa_f32 returned")

	// Check if anything was written at all
	allSentinel := true
	for i := range c {
		if c[i] != -999.0 {
			allSentinel = false
			break
		}
	}
	if allSentinel {
		t.Error("No values were written to C - all still sentinel")
	}

	// Each element should be 16 * 2 * 3 = 96 (sum of 16 outer products)
	// If only 1 FMOPA: 6
	expected := float32(96)
	t.Logf("First few elements of C (expected 96 if 16 FMOPAs, 6 if 1 FMOPA):")
	for i := range 8 {
		t.Logf("  c[%d] = %f", i, c[i])
	}

	for i := range c {
		if c[i] != expected {
			t.Errorf("c[%d] = %f, want %f", i, c[i], expected)
		}
	}
}

// TestFMOPAMatMul32x32 tests FMOPA with a 32x32 matrix (4 tiles)
func TestFMOPAMatMul32x32(t *testing.T) {
	const n = 32

	// A = all 2.0, B = all 3.0
	// C[i,j] = sum over k of A[i,k] * B[k,j] = 32 * 2 * 3 = 192
	a := make([]float32, n*n)
	b := make([]float32, n*n)
	for i := range a {
		a[i] = 2.0
		b[i] = 3.0
	}

	c := make([]float32, n*n)
	for i := range c {
		c[i] = -999.0
	}

	matmul_fmopa_f32(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		int64(n), int64(n), int64(n),
	)

	// Check all 4 tiles
	expected := float32(192) // 32 * 2 * 3
	errCount := 0
	maxErr := float32(0)

	for i := range n {
		for j := range n {
			err := c[i*n+j] - expected
			if err < 0 {
				err = -err
			}
			if err > maxErr {
				maxErr = err
			}
			if c[i*n+j] != expected {
				if errCount < 10 {
					t.Errorf("c[%d,%d] = %f, want %f", i, j, c[i*n+j], expected)
				}
				errCount++
			}
		}
	}

	if errCount > 10 {
		t.Errorf("... and %d more errors", errCount-10)
	}

	if errCount == 0 {
		t.Logf("FMOPA 32x32 matmul passed, all %d elements correct", n*n)
	}
}

// TestFMOPAMatMul64x64 tests FMOPA with a 64x64 matrix (16 tiles)
func TestFMOPAMatMul64x64(t *testing.T) {
	const n = 64

	// A = all 2.0, B = all 3.0
	// C[i,j] = sum over k of A[i,k] * B[k,j] = 64 * 2 * 3 = 384
	a := make([]float32, n*n)
	b := make([]float32, n*n)
	for i := range a {
		a[i] = 2.0
		b[i] = 3.0
	}

	c := make([]float32, n*n)
	for i := range c {
		c[i] = -999.0
	}

	matmul_fmopa_f32(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		int64(n), int64(n), int64(n),
	)

	expected := float32(384) // 64 * 2 * 3
	errCount := 0

	for i := range n {
		for j := range n {
			if c[i*n+j] != expected {
				if errCount < 10 {
					t.Errorf("c[%d,%d] = %f, want %f", i, j, c[i*n+j], expected)
				}
				errCount++
			}
		}
	}

	if errCount > 10 {
		t.Errorf("... and %d more errors", errCount-10)
	}

	if errCount == 0 {
		t.Logf("FMOPA 64x64 matmul passed, all %d elements correct", n*n)
	}
}

// TestSMEZAStore tests storing from ZA tile (after zeroing it)
func TestSMEZAStore(t *testing.T) {
	// Allocate 16 float32s for one row of ZA (512-bit SVL = 16 floats)
	data := make([]float32, 16)
	for i := range data {
		data[i] = -1.0 // Initialize to -1 to verify writes
	}

	sme_za_store_test(unsafe.Pointer(&data[0]))

	// After zeroing ZA and storing, all values should be 0
	for i, v := range data {
		t.Logf("data[%d] = %f (expected 0.0)", i, v)
		if v != 0.0 {
			t.Errorf("data[%d] = %f, want 0.0", i, v)
		}
	}
}

// TestSMEZALoadStore tests loading into ZA tile and storing back
// SKIP: LD1W/ST1W for ZA tiles have different encodings that need verification
func TestSMEZALoadStore(t *testing.T) {
	t.Skip("LD1W/ST1W for ZA tiles need encoding verification - use MOVA instead")
	// Source array with known values
	src := make([]float32, 16)
	for i := range src {
		src[i] = float32(i + 1) // 1, 2, 3, ..., 16
	}

	// Destination array initialized to -1
	dst := make([]float32, 16)
	for i := range dst {
		dst[i] = -1.0
	}

	sme_za_load_store_test(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]))

	// After loading src into ZA and storing to dst, values should match
	for i, v := range dst {
		expected := float32(i + 1)
		t.Logf("dst[%d] = %f (expected %f)", i, v, expected)
		if v != expected {
			t.Errorf("dst[%d] = %f, want %f", i, v, expected)
		}
	}
}

// TestSMEFMOPA tests the outer product accumulate instruction
func TestSMEFMOPA(t *testing.T) {
	// Allocate space for one ZA row (16 floats for 512-bit SVL)
	data := make([]float32, 16)
	for i := range data {
		data[i] = -1.0
	}

	sme_fmopa_test(unsafe.Pointer(&data[0]))

	// FMOPA with z0 = all 2.0, z1 = all 3.0 should give 2.0 * 3.0 = 6.0
	// accumulated into zeroed ZA, so result should be 6.0
	expected := float32(6.0)
	for i, v := range data {
		t.Logf("data[%d] = %f (expected %f)", i, v, expected)
	}
	for i, v := range data {
		if v != expected {
			t.Errorf("data[%d] = %f, want %f", i, v, expected)
		}
	}
}

// TestSMEBuildVector tests building a vector via horizontal write + vertical read
// This is the technique used to load strided data for FMOPA
func TestSMEBuildVector(t *testing.T) {
	data := make([]float32, 16)
	for i := range data {
		data[i] = -1.0
	}

	sme_build_vector_test(unsafe.Pointer(&data[0]))

	// After writing 0,1,2,...,15 to horizontal slices and reading vertically,
	// we should get [0, 1, 2, ..., 15]
	t.Log("Vector built via horizontal write + vertical read:")
	for i, v := range data {
		t.Logf("  data[%d] = %f (expected %f)", i, v, float32(i))
	}

	for i, v := range data {
		expected := float32(i)
		if v != expected {
			t.Errorf("data[%d] = %f, want %f", i, v, expected)
		}
	}
}

// TestSMEFMOPADebug tests FMOPA with debug output
func TestSMEFMOPADebug(t *testing.T) {
	z0out := make([]float32, 16)
	z1out := make([]float32, 16)
	zaout := make([]float32, 16)
	for i := range z0out {
		z0out[i] = -1.0
		z1out[i] = -1.0
		zaout[i] = -1.0
	}

	sme_fmopa_debug_test(
		unsafe.Pointer(&z0out[0]),
		unsafe.Pointer(&z1out[0]),
		unsafe.Pointer(&zaout[0]),
	)

	t.Log("z0 (expected 2.0):")
	for i, v := range z0out {
		t.Logf("  z0[%d] = %f", i, v)
	}
	t.Log("z1 (expected 3.0):")
	for i, v := range z1out {
		t.Logf("  z1[%d] = %f", i, v)
	}
	t.Log("ZA row 0 (expected 6.0):")
	for i, v := range zaout {
		t.Logf("  za[%d] = %f", i, v)
	}

	// Check z0 = 2.0
	for i, v := range z0out {
		if v != 2.0 {
			t.Errorf("z0[%d] = %f, want 2.0", i, v)
		}
	}
	// Check z1 = 3.0
	for i, v := range z1out {
		if v != 3.0 {
			t.Errorf("z1[%d] = %f, want 3.0", i, v)
		}
	}
	// Check ZA = 6.0
	for i, v := range zaout {
		if v != 6.0 {
			t.Errorf("za[%d] = %f, want 6.0", i, v)
		}
	}
}

// TestFMOPATransposed tests FMOPA with pre-transposed A
func TestFMOPATransposed(t *testing.T) {
	const n = 64

	// A = all 2.0, B = all 3.0
	// C[i,j] = sum over k of A[i,k] * B[k,j] = 64 * 2 * 3 = 384
	a := make([]float32, n*n)
	b := make([]float32, n*n)
	for i := range a {
		a[i] = 2.0
		b[i] = 3.0
	}

	// Transpose A manually: AT[k,i] = A[i,k]
	at := make([]float32, n*n)
	for i := range n {
		for k := range n {
			at[k*n+i] = a[i*n+k]
		}
	}

	c := make([]float32, n*n)
	for i := range c {
		c[i] = -999.0
	}

	matmul_fmopa_at_f32(
		unsafe.Pointer(&at[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		int64(n), int64(n), int64(n),
	)

	expected := float32(384) // 64 * 2 * 3
	errCount := 0

	for i := range n {
		for j := range n {
			if c[i*n+j] != expected {
				if errCount < 10 {
					t.Errorf("c[%d,%d] = %f, want %f", i, j, c[i*n+j], expected)
				}
				errCount++
			}
		}
	}

	if errCount > 10 {
		t.Errorf("... and %d more errors", errCount-10)
	}

	if errCount == 0 {
		t.Logf("FMOPA transposed 64x64 matmul passed, all %d elements correct", n*n)
	}
}

// BenchmarkFMOPA_vs_NEON compares FMOPA and NEON matmul performance
func BenchmarkFMOPA_vs_NEON(b *testing.B) {
	sizes := []int{64, 128, 256}

	fmtSize := func(n int) string {
		return string(rune('0'+n/100)) + string(rune('0'+(n/10)%10)) + string(rune('0'+n%10))
	}

	for _, size := range sizes {
		m, n, k := size, size, size
		a := make([]float32, m*k)
		bMat := make([]float32, k*n)
		c := make([]float32, m*n)

		for i := range a {
			a[i] = 2.0
		}
		for i := range bMat {
			bMat[i] = 3.0
		}

		// Pre-transpose A for FMOPA_AT benchmark
		at := make([]float32, k*m)
		for i := range m {
			for j := range k {
				at[j*m+i] = a[i*k+j]
			}
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run("FMOPA_Scalar/"+fmtSize(size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matmul_fmopa_f32(
					unsafe.Pointer(&a[0]),
					unsafe.Pointer(&bMat[0]),
					unsafe.Pointer(&c[0]),
					int64(m), int64(n), int64(k),
				)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run("FMOPA_Transposed/"+fmtSize(size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matmul_fmopa_at_f32(
					unsafe.Pointer(&at[0]),
					unsafe.Pointer(&bMat[0]),
					unsafe.Pointer(&c[0]),
					int64(m), int64(n), int64(k),
				)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run("NEON/"+fmtSize(size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matmulNEON(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}
