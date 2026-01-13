//go:build !noasm && arm64

#include "textflag.h"

// func block_muladd_neon_f32(aT, b, c unsafe.Pointer, blockDim int64)
//
// Computes C += A * B for square blocks where:
//   - aT is PRE-TRANSPOSED A (rows are original A columns)
//   - b is normal B (rows are B rows)
//
// This computes C += (aT)^T * b = A * B
//
// Memory layout advantages:
//   - aT[k, i:i+4] gives us A[i:i+4, k] contiguously
//   - b[k, j:j+4] gives us B[k, j:j+4] contiguously
//   - Both A and B accesses are now contiguous!
//
// Register usage:
//   R19: aT base pointer
//   R20: B base pointer
//   R21: C base pointer
//   R22: blockDim
//   R23: blockDim * 4 (row stride in bytes)
//   R0: i (row counter, steps by 4)
//   R1: j (column counter, steps by 4)
//   R2: k (reduction loop counter)
//   R3-R6: temp pointers
//   R8-R11: C row pointers for rows i, i+1, i+2, i+3
//   V0-V3: A[i+0..3, k] broadcasts
//   V4: B[k, j:j+4]
//   V16-V19: C accumulators for rows i, i+1, i+2, i+3
//
TEXT ·block_muladd_neon_f32(SB), NOSPLIT, $64-32
	// Save callee-saved registers
	MOVD	R19, 0(RSP)
	MOVD	R20, 8(RSP)
	MOVD	R21, 16(RSP)
	MOVD	R22, 24(RSP)
	MOVD	R23, 32(RSP)
	MOVD	R24, 40(RSP)
	MOVD	R25, 48(RSP)
	MOVD	R26, 56(RSP)

	// Load arguments
	MOVD	aT+0(FP), R19     // aT matrix base (transposed A)
	MOVD	b+8(FP), R20      // B matrix base (normal)
	MOVD	c+16(FP), R21     // C matrix base
	MOVD	blockDim+24(FP), R22  // blockDim

	// Calculate row stride in bytes
	LSL	$2, R22, R23       // R23 = blockDim * 4

	// Outer loop: i over rows of C (step by 4)
	MOVD	$0, R0             // R0 = i

row_loop:
	// Check if we have at least 4 rows left
	ADD	$4, R0, R24
	CMP	R22, R24
	BGT	row_tail           // Less than 4 rows left

	// Calculate C row pointers for rows i, i+1, i+2, i+3
	MUL	R23, R0, R8
	ADD	R21, R8, R8        // R8 = &C[i, 0]
	ADD	R23, R8, R9        // R9 = &C[i+1, 0]
	ADD	R23, R9, R10       // R10 = &C[i+2, 0]
	ADD	R23, R10, R11      // R11 = &C[i+3, 0]

	// Inner loop: j over columns of C (step by 4)
	MOVD	$0, R1             // R1 = j

col_loop:
	ADD	$4, R1, R24
	CMP	R22, R24
	BGT	col_tail

	// Load C[i:i+4, j:j+4] into accumulators
	LSL	$2, R1, R24        // R24 = j * 4 (byte offset)
	ADD	R8, R24, R25       // &C[i, j]
	VLD1	(R25), [V16.S4]
	ADD	R9, R24, R25       // &C[i+1, j]
	VLD1	(R25), [V17.S4]
	ADD	R10, R24, R25      // &C[i+2, j]
	VLD1	(R25), [V18.S4]
	ADD	R11, R24, R25      // &C[i+3, j]
	VLD1	(R25), [V19.S4]

	// K-loop: accumulate over k
	MOVD	$0, R2             // R2 = k

k_loop:
	CMP	R22, R2
	BGE	k_done

	// Load A[i:i+4, k] = aT[k, i:i+4] (contiguous in aT!)
	// aT[k, i] is at aT + k*stride + i*4
	MUL	R23, R2, R3        // R3 = k * stride
	ADD	R19, R3, R3        // R3 = &aT[k, 0]
	LSL	$2, R0, R24        // R24 = i * 4
	ADD	R3, R24, R3        // R3 = &aT[k, i]

	// Load 4 consecutive A values and broadcast each
	VLD1	(R3), [V5.S4]      // V5 = [A[i,k], A[i+1,k], A[i+2,k], A[i+3,k]]
	VDUP	V5.S[0], V0.S4     // V0 = A[i, k] broadcast
	VDUP	V5.S[1], V1.S4     // V1 = A[i+1, k] broadcast
	VDUP	V5.S[2], V2.S4     // V2 = A[i+2, k] broadcast
	VDUP	V5.S[3], V3.S4     // V3 = A[i+3, k] broadcast

	// Load B[k, j:j+4] (contiguous in B)
	MUL	R23, R2, R25       // R25 = k * stride
	ADD	R20, R25, R25      // R25 = &B[k, 0]
	LSL	$2, R1, R26        // R26 = j * 4
	ADD	R25, R26, R25      // R25 = &B[k, j]
	VLD1	(R25), [V4.S4]

	// Accumulate: C[i+r, j:j+4] += A[i+r, k] * B[k, j:j+4]
	VFMLA	V0.S4, V4.S4, V16.S4
	VFMLA	V1.S4, V4.S4, V17.S4
	VFMLA	V2.S4, V4.S4, V18.S4
	VFMLA	V3.S4, V4.S4, V19.S4

	ADD	$1, R2, R2
	B	k_loop

k_done:
	// Store C[i:i+4, j:j+4] back
	LSL	$2, R1, R24
	ADD	R8, R24, R25
	VST1	[V16.S4], (R25)
	ADD	R9, R24, R25
	VST1	[V17.S4], (R25)
	ADD	R10, R24, R25
	VST1	[V18.S4], (R25)
	ADD	R11, R24, R25
	VST1	[V19.S4], (R25)

	// Next column block
	ADD	$4, R1, R1
	B	col_loop

col_tail:
	// Handle remaining columns (j to blockDim) - scalar fallback
	CMP	R22, R1
	BGE	next_row_block

col_tail_loop:
	CMP	R22, R1
	BGE	next_row_block

	// Load C[i:i+4, j] scalars
	LSL	$2, R1, R24
	ADD	R8, R24, R25
	FMOVS	(R25), F16         // C[i, j]
	ADD	R9, R24, R25
	FMOVS	(R25), F17         // C[i+1, j]
	ADD	R10, R24, R25
	FMOVS	(R25), F18         // C[i+2, j]
	ADD	R11, R24, R25
	FMOVS	(R25), F19         // C[i+3, j]

	// K-loop scalar
	MOVD	$0, R2

k_tail_loop:
	CMP	R22, R2
	BGE	k_tail_done

	// Load A[i:i+4, k] = aT[k, i:i+4]
	MUL	R23, R2, R3
	ADD	R19, R3, R3
	LSL	$2, R0, R24
	ADD	R3, R24, R3

	FMOVS	(R3), F0           // A[i, k]
	FMOVS	4(R3), F1          // A[i+1, k]
	FMOVS	8(R3), F2          // A[i+2, k]
	FMOVS	12(R3), F3         // A[i+3, k]

	// Load B[k, j]
	MUL	R23, R2, R25
	ADD	R20, R25, R25
	LSL	$2, R1, R26
	ADD	R25, R26, R25
	FMOVS	(R25), F4

	// Accumulate
	FMADDS	F0, F4, F16, F16
	FMADDS	F1, F4, F17, F17
	FMADDS	F2, F4, F18, F18
	FMADDS	F3, F4, F19, F19

	ADD	$1, R2, R2
	B	k_tail_loop

k_tail_done:
	// Store back
	LSL	$2, R1, R24
	ADD	R8, R24, R25
	FMOVS	F16, (R25)
	ADD	R9, R24, R25
	FMOVS	F17, (R25)
	ADD	R10, R24, R25
	FMOVS	F18, (R25)
	ADD	R11, R24, R25
	FMOVS	F19, (R25)

	ADD	$1, R1, R1
	B	col_tail_loop

next_row_block:
	ADD	$4, R0, R0
	B	row_loop

row_tail:
	// Handle remaining rows (less than 4)
	CMP	R22, R0
	BGE	done

row_tail_loop:
	CMP	R22, R0
	BGE	done

	// Single row processing
	MUL	R23, R0, R8
	ADD	R21, R8, R8        // R8 = &C[i, 0]

	MOVD	$0, R1             // j = 0

single_col_loop:
	ADD	$4, R1, R24
	CMP	R22, R24
	BGT	single_col_tail

	// Load C[i, j:j+4]
	LSL	$2, R1, R24
	ADD	R8, R24, R25
	VLD1	(R25), [V16.S4]

	// K-loop
	MOVD	$0, R2

single_k_loop:
	CMP	R22, R2
	BGE	single_k_done

	// Load and broadcast A[i, k] = aT[k, i]
	MUL	R23, R2, R3
	ADD	R19, R3, R3
	LSL	$2, R0, R24
	ADD	R3, R24, R3
	FMOVS	(R3), F0
	VDUP	V0.S[0], V0.S4

	// Load B[k, j:j+4]
	MUL	R23, R2, R25
	ADD	R20, R25, R25
	LSL	$2, R1, R26
	ADD	R25, R26, R25
	VLD1	(R25), [V4.S4]

	// Accumulate
	VFMLA	V0.S4, V4.S4, V16.S4

	ADD	$1, R2, R2
	B	single_k_loop

single_k_done:
	// Store C[i, j:j+4]
	LSL	$2, R1, R24
	ADD	R8, R24, R25
	VST1	[V16.S4], (R25)

	ADD	$4, R1, R1
	B	single_col_loop

single_col_tail:
	// Scalar tail for single row
	CMP	R22, R1
	BGE	next_single_row

single_scalar_loop:
	CMP	R22, R1
	BGE	next_single_row

	LSL	$2, R1, R24
	ADD	R8, R24, R25
	FMOVS	(R25), F16

	MOVD	$0, R2

single_scalar_k:
	CMP	R22, R2
	BGE	single_scalar_k_done

	// A[i, k] = aT[k, i]
	MUL	R23, R2, R3
	ADD	R19, R3, R3
	LSL	$2, R0, R24
	ADD	R3, R24, R3
	FMOVS	(R3), F0

	// B[k, j]
	MUL	R23, R2, R25
	ADD	R20, R25, R25
	LSL	$2, R1, R26
	ADD	R25, R26, R25
	FMOVS	(R25), F4
	FMADDS	F0, F4, F16, F16

	ADD	$1, R2, R2
	B	single_scalar_k

single_scalar_k_done:
	LSL	$2, R1, R24
	ADD	R8, R24, R25
	FMOVS	F16, (R25)

	ADD	$1, R1, R1
	B	single_scalar_loop

next_single_row:
	ADD	$1, R0, R0
	B	row_tail_loop

done:
	// Restore callee-saved registers
	MOVD	0(RSP), R19
	MOVD	8(RSP), R20
	MOVD	16(RSP), R21
	MOVD	24(RSP), R22
	MOVD	32(RSP), R23
	MOVD	40(RSP), R24
	MOVD	48(RSP), R25
	MOVD	56(RSP), R26

	RET

// =============================================================================
// Float64 NEON version
// =============================================================================

// func block_muladd_neon_f64(aT, b, c unsafe.Pointer, blockDim int64)
//
// Computes C += A * B for square blocks where:
//   - aT is PRE-TRANSPOSED A (rows are original A columns)
//   - b is normal B (rows are B rows)
//
// Float64 version: processes 2 doubles per NEON vector (128-bit).
//
TEXT ·block_muladd_neon_f64(SB), NOSPLIT, $64-32
	// Save callee-saved registers
	MOVD	R19, 0(RSP)
	MOVD	R20, 8(RSP)
	MOVD	R21, 16(RSP)
	MOVD	R22, 24(RSP)
	MOVD	R23, 32(RSP)
	MOVD	R24, 40(RSP)
	MOVD	R25, 48(RSP)
	MOVD	R26, 56(RSP)

	// Load arguments
	MOVD	aT+0(FP), R19     // aT matrix base (transposed A)
	MOVD	b+8(FP), R20      // B matrix base (normal)
	MOVD	c+16(FP), R21     // C matrix base
	MOVD	blockDim+24(FP), R22  // blockDim

	// Calculate row stride in bytes (blockDim * 8 for f64)
	LSL	$3, R22, R23       // R23 = blockDim * 8

	// Outer loop: i over rows of C (step by 2 for f64)
	MOVD	$0, R0             // R0 = i

f64_row_loop:
	// Check if we have at least 2 rows left
	ADD	$2, R0, R24
	CMP	R22, R24
	BGT	f64_row_tail

	// Calculate C row pointers for rows i, i+1
	MUL	R23, R0, R8
	ADD	R21, R8, R8        // R8 = &C[i, 0]
	ADD	R23, R8, R9        // R9 = &C[i+1, 0]

	// Inner loop: j over columns of C (step by 2 for f64 vectors)
	MOVD	$0, R1             // R1 = j

f64_col_loop:
	ADD	$2, R1, R24
	CMP	R22, R24
	BGT	f64_col_tail

	// Load C[i:i+2, j:j+2] into accumulators
	LSL	$3, R1, R24        // R24 = j * 8 (byte offset)
	ADD	R8, R24, R25       // &C[i, j]
	VLD1	(R25), [V16.D2]
	ADD	R9, R24, R25       // &C[i+1, j]
	VLD1	(R25), [V17.D2]

	// K-loop: accumulate over k
	MOVD	$0, R2             // R2 = k

f64_k_loop:
	CMP	R22, R2
	BGE	f64_k_done

	// Load A[i:i+2, k] = aT[k, i:i+2] (contiguous in aT!)
	MUL	R23, R2, R3        // R3 = k * stride
	ADD	R19, R3, R3        // R3 = &aT[k, 0]
	LSL	$3, R0, R24        // R24 = i * 8
	ADD	R3, R24, R3        // R3 = &aT[k, i]

	// Load 2 consecutive A values and broadcast each
	VLD1	(R3), [V5.D2]      // V5 = [A[i,k], A[i+1,k]]
	VDUP	V5.D[0], V0.D2     // V0 = A[i, k] broadcast
	VDUP	V5.D[1], V1.D2     // V1 = A[i+1, k] broadcast

	// Load B[k, j:j+2] (contiguous in B)
	MUL	R23, R2, R25       // R25 = k * stride
	ADD	R20, R25, R25      // R25 = &B[k, 0]
	LSL	$3, R1, R26        // R26 = j * 8
	ADD	R25, R26, R25      // R25 = &B[k, j]
	VLD1	(R25), [V4.D2]

	// Accumulate: C[i+r, j:j+2] += A[i+r, k] * B[k, j:j+2]
	VFMLA	V0.D2, V4.D2, V16.D2
	VFMLA	V1.D2, V4.D2, V17.D2

	ADD	$1, R2, R2
	B	f64_k_loop

f64_k_done:
	// Store C[i:i+2, j:j+2] back
	LSL	$3, R1, R24
	ADD	R8, R24, R25
	VST1	[V16.D2], (R25)
	ADD	R9, R24, R25
	VST1	[V17.D2], (R25)

	// Next column block
	ADD	$2, R1, R1
	B	f64_col_loop

f64_col_tail:
	// Handle remaining columns - scalar fallback
	CMP	R22, R1
	BGE	f64_next_row_block

f64_col_tail_loop:
	CMP	R22, R1
	BGE	f64_next_row_block

	// Load C[i:i+2, j] scalars
	LSL	$3, R1, R24
	ADD	R8, R24, R25
	FMOVD	(R25), F16         // C[i, j]
	ADD	R9, R24, R25
	FMOVD	(R25), F17         // C[i+1, j]

	// K-loop scalar
	MOVD	$0, R2

f64_k_tail_loop:
	CMP	R22, R2
	BGE	f64_k_tail_done

	// Load A[i:i+2, k] = aT[k, i:i+2]
	MUL	R23, R2, R3
	ADD	R19, R3, R3
	LSL	$3, R0, R24
	ADD	R3, R24, R3

	FMOVD	(R3), F0           // A[i, k]
	FMOVD	8(R3), F1          // A[i+1, k]

	// Load B[k, j]
	MUL	R23, R2, R25
	ADD	R20, R25, R25
	LSL	$3, R1, R26
	ADD	R25, R26, R25
	FMOVD	(R25), F4

	// Accumulate
	FMADDD	F0, F4, F16, F16
	FMADDD	F1, F4, F17, F17

	ADD	$1, R2, R2
	B	f64_k_tail_loop

f64_k_tail_done:
	// Store back
	LSL	$3, R1, R24
	ADD	R8, R24, R25
	FMOVD	F16, (R25)
	ADD	R9, R24, R25
	FMOVD	F17, (R25)

	ADD	$1, R1, R1
	B	f64_col_tail_loop

f64_next_row_block:
	ADD	$2, R0, R0
	B	f64_row_loop

f64_row_tail:
	// Handle remaining single row
	CMP	R22, R0
	BGE	f64_done

	// Single row processing
	MUL	R23, R0, R8
	ADD	R21, R8, R8        // R8 = &C[i, 0]

	MOVD	$0, R1             // j = 0

f64_single_col_loop:
	ADD	$2, R1, R24
	CMP	R22, R24
	BGT	f64_single_col_tail

	// Load C[i, j:j+2]
	LSL	$3, R1, R24
	ADD	R8, R24, R25
	VLD1	(R25), [V16.D2]

	// K-loop
	MOVD	$0, R2

f64_single_k_loop:
	CMP	R22, R2
	BGE	f64_single_k_done

	// Load and broadcast A[i, k] = aT[k, i]
	MUL	R23, R2, R3
	ADD	R19, R3, R3
	LSL	$3, R0, R24
	ADD	R3, R24, R3
	FMOVD	(R3), F0
	VDUP	V0.D[0], V0.D2

	// Load B[k, j:j+2]
	MUL	R23, R2, R25
	ADD	R20, R25, R25
	LSL	$3, R1, R26
	ADD	R25, R26, R25
	VLD1	(R25), [V4.D2]

	// Accumulate
	VFMLA	V0.D2, V4.D2, V16.D2

	ADD	$1, R2, R2
	B	f64_single_k_loop

f64_single_k_done:
	// Store C[i, j:j+2]
	LSL	$3, R1, R24
	ADD	R8, R24, R25
	VST1	[V16.D2], (R25)

	ADD	$2, R1, R1
	B	f64_single_col_loop

f64_single_col_tail:
	// Scalar tail for single row
	CMP	R22, R1
	BGE	f64_done

f64_single_scalar_loop:
	CMP	R22, R1
	BGE	f64_done

	LSL	$3, R1, R24
	ADD	R8, R24, R25
	FMOVD	(R25), F16

	MOVD	$0, R2

f64_single_scalar_k:
	CMP	R22, R2
	BGE	f64_single_scalar_k_done

	// A[i, k] = aT[k, i]
	MUL	R23, R2, R3
	ADD	R19, R3, R3
	LSL	$3, R0, R24
	ADD	R3, R24, R3
	FMOVD	(R3), F0

	// B[k, j]
	MUL	R23, R2, R25
	ADD	R20, R25, R25
	LSL	$3, R1, R26
	ADD	R25, R26, R25
	FMOVD	(R25), F4
	FMADDD	F0, F4, F16, F16

	ADD	$1, R2, R2
	B	f64_single_scalar_k

f64_single_scalar_k_done:
	LSL	$3, R1, R24
	ADD	R8, R24, R25
	FMOVD	F16, (R25)

	ADD	$1, R1, R1
	B	f64_single_scalar_loop

f64_done:
	// Restore callee-saved registers
	MOVD	0(RSP), R19
	MOVD	8(RSP), R20
	MOVD	16(RSP), R21
	MOVD	24(RSP), R22
	MOVD	32(RSP), R23
	MOVD	40(RSP), R24
	MOVD	48(RSP), R25
	MOVD	56(RSP), R26

	RET
