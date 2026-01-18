//go:build !noasm && darwin && arm64

package matmul

import (
	"sync"
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/matmul/asm"
)

// NOTE: SME implementation confirmed working on Apple M4!
//
// Initial testing showed zeros from ZA due to incorrect MOVA encodings.
// After fixing instruction encodings from LLVM reference:
//
// What WORKS on Apple M4:
//   - SMSTART/SMSTOP (streaming mode entry/exit)
//   - PTRUE, ZERO {ZA}, DUP
//   - ST1W from Z registers (SVE stores in streaming mode)
//   - FMOPA (outer product accumulate) - 512 FP32 ops per instruction!
//   - MOVA Z→ZA (write to ZA): encoding 0xc080XXXX
//   - MOVA ZA→Z (read from ZA): encoding 0xc082XXXX (bit 17 set!)
//
// Key encoding corrections:
//   - MOVA tile-to-vector requires bit 17 set (0xc082 prefix, not 0xc080)
//   - FMOPA encoding: 0x8081XXXX for za.s with z registers
//   - PTRUE p0.s for FP32 operations, not p0.b
//
// Apple M4 SVL = 512 bits, meaning:
//   - Z registers hold 16 × float32
//   - ZA tiles are 16×16 = 256 float32 elements
//   - Single FMOPA does 16×16×2 = 512 FP32 ops
//
// Two implementations available:
//   1. matmul_fmopa_f32: original with scalar A column loading (slower)
//   2. matmul_fmopa_at_f32: with pre-transposed A for contiguous loads (faster!)

// Minimum dimensions to use SME FMOPA MatMul
// With pre-transposed A, FMOPA uses contiguous loads and is competitive with NEON.
// Below this threshold, NEON is used (streaming mode has fixed overhead).
const minDimForSME = 64

//go:noescape
func matmul_sme_f32(a, b, c unsafe.Pointer, m, n, k int64)

//go:noescape
func matmul_fmopa_f32(a, b, c unsafe.Pointer, m, n, k int64)

//go:noescape
func matmul_fmopa_at_f32(at, b, c unsafe.Pointer, m, n, k int64)

//go:noescape
func matmul_fmopa_at_f64(at, b, c unsafe.Pointer, m, n, k int64)


// Transpose buffer pools to avoid allocations
var transposePool32 = sync.Pool{
	New: func() any {
		return make([]float32, 0, 256*256)
	},
}

var transposePool64 = sync.Pool{
	New: func() any {
		return make([]float64, 0, 256*256)
	},
}

var transposePoolF16 = sync.Pool{
	New: func() any {
		return make([]hwy.Float16, 0, 256*256)
	},
}

var transposePoolBF16 = sync.Pool{
	New: func() any {
		return make([]hwy.BFloat16, 0, 256*256)
	},
}

// transposeMatrix transposes M×K matrix A into K×M matrix AT (row-major to column-major)
// AT[k,i] = A[i,k]
// Uses generics to support all float types.
func transposeMatrix[T hwy.Floats](a []T, m, k int, at []T) {
	for i := range m {
		for j := range k {
			at[j*m+i] = a[i*k+j]
		}
	}
}

// matmulStreamingSVE uses ARM streaming SVE mode with SVE instructions.
// NOTE: This does NOT work on Apple M4 - ld1w doesn't function in streaming mode.
// Kept for reference, but use matmulFMOPA instead.
func matmulStreamingSVE(a, b, c []float32, m, n, k int) {
	// For small matrices, fall back to NEON (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulNEON(a, b, c, m, n, k)
		return
	}

	matmul_sme_f32(
		unsafe.Pointer(unsafe.SliceData(a)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		int64(m),
		int64(n),
		int64(k),
	)
}

// matmulFMOPA uses ARM SME FMOPA instruction for matrix multiplication.
// Uses outer product accumulate with ZA tiles - confirmed working on Apple M4!
// Processes matrices in 16x16 tiles using the ZA accumulator.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPA(a, b, c []float32, m, n, k int) {
	// For non-aligned sizes, fall back to generated NEON
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		matmulNEON(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulNEON(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePool32.Get().([]float32)
	if cap(atBuf) < atSize {
		atBuf = make([]float32, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Call FMOPA with transposed A
	matmul_fmopa_at_f32(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		int64(m),
		int64(n),
		int64(k),
	)

	// Return buffer to pool
	transposePool32.Put(atBuf)
}

// matmulFMOPA64 uses ARM SME FMOPA instruction for float64 matrix multiplication.
// Uses outer product accumulate with ZA tiles - 8×8 tiles for float64.
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPA64(a, b, c []float64, m, n, k int) {
	// For non-aligned sizes (8×8 tiles for float64), fall back to scalar
	if m%8 != 0 || n%8 != 0 || k%8 != 0 {
		matmulScalar64(a, b, c, m, n, k)
		return
	}

	// For small matrices, scalar is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulScalar64(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePool64.Get().([]float64)
	if cap(atBuf) < atSize {
		atBuf = make([]float64, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Call FMOPA with transposed A
	matmul_fmopa_at_f64(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		int64(m),
		int64(n),
		int64(k),
	)

	// Return buffer to pool
	transposePool64.Put(atBuf)
}

// matmulFMOPAF16 uses ARM SME FMOPA instruction for float16 matrix multiplication.
// Uses widening: f16 -> f32 FMOPA -> f16, with 16×16 tiles (f32 accumulator).
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPAF16(a, b, c []hwy.Float16, m, n, k int) {
	// For non-aligned sizes (16×16 tiles), fall back to NEON
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		matmulNEONF16(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulNEONF16(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePoolF16.Get().([]hwy.Float16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.Float16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Scratch buffer for f32->f16 conversion
	var scratch [16]float32
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)

	// Call FMOPA with transposed A (from asm package)
	asm.MatmulFmopaAtF16(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)

	// Return buffer to pool
	transposePoolF16.Put(atBuf)
}

// matmulFMOPABF16 uses ARM SME BFMOPA instruction for bfloat16 matrix multiplication.
// Uses widening: bf16 -> f32 FMOPA -> bf16, with 16×16 tiles (f32 accumulator).
// Pre-transposes A for contiguous column access, enabling fast vector loads.
func matmulFMOPABF16(a, b, c []hwy.BFloat16, m, n, k int) {
	// For non-aligned sizes (16×16 tiles), fall back to NEON
	if m%16 != 0 || n%16 != 0 || k%16 != 0 {
		matmulNEONBF16(a, b, c, m, n, k)
		return
	}

	// For small matrices, NEON is faster (streaming mode has overhead)
	if m < minDimForSME || n < minDimForSME || k < minDimForSME {
		matmulNEONBF16(a, b, c, m, n, k)
		return
	}

	// Get transpose buffer from pool
	atSize := m * k
	atBuf := transposePoolBF16.Get().([]hwy.BFloat16)
	if cap(atBuf) < atSize {
		atBuf = make([]hwy.BFloat16, atSize)
	} else {
		atBuf = atBuf[:atSize]
	}

	// Transpose A (M×K) to AT (K×M) for contiguous column access
	transposeMatrix(a, m, k, atBuf)

	// Scratch buffer for f32->bf16 conversion
	var scratch [16]float32
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)

	// Call BFMOPA with transposed A (from asm package)
	asm.MatmulBfmopaAtBF16(
		unsafe.Pointer(unsafe.SliceData(atBuf)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(unsafe.SliceData(c)),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
		unsafe.Pointer(&scratch[0]),
	)

	// Return buffer to pool
	transposePoolBF16.Put(atBuf)
}

func init() {
	if hwy.HasSME() {
		// Use FMOPA implementation which works on Apple M4
		// This overrides the generated dispatch for large aligned matrices
		MatMulFloat32 = matmulFMOPA
		MatMulFloat64 = matmulFMOPA64
		MatMulFloat16 = matmulFMOPAF16
		MatMulBFloat16 = matmulFMOPABF16
	}
}
