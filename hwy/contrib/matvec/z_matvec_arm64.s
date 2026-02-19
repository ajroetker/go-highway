//go:build !noasm && darwin && arm64

#include "textflag.h"

// func matvec_sme_f32(mt, v, result unsafe.Pointer, rows, cols int64)
//
// FMOPA-based matrix-vector multiplication with M TRANSPOSED: result = MT^T * v = M * v
//   MT is cols x rows (row-major) = M transposed
//   v is cols (input vector)
//   result is rows (output vector)
//
// With M transposed, loading M columns becomes contiguous!
//   M[row:row+16, k] = MT[k, row:row+16] = contiguous in memory
//
// Algorithm (matches matmul pattern):
//   For each 16-row tile:
//     zero ZA
//     for k in 0..cols:
//       z2 = MT[k, row:row+16]  (contiguous load - like A column in matmul)
//       z0 = broadcast(v[k])   (like B row in matmul, but broadcast)
//       FMOPA za0 += z2 ⊗ z0
//     Extract za0 rows and store first element of each to result
//
// Apple M4 SVL = 512 bits = 16 × float32
// ZA tiles = 16×16 = 256 float32 elements
//
// Register usage (matching matmul):
//   R19: MT base pointer
//   R20: v base pointer
//   R21: result base pointer
//   R22: rows dimension
//   R23: cols dimension
//   R24: rows * 4 (MT row stride in bytes)
//   R0: row (tile row index, increments by 16)
//   R3: k (column loop index)
//   R10: zero register for LD1W offset
//
TEXT ·matvec_sme_f32(SB), NOSPLIT, $64-40
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
	MOVD	mt+0(FP), R19      // MT matrix base (cols x rows, row-major)
	MOVD	v+8(FP), R20       // v vector base
	MOVD	result+16(FP), R21 // result vector base
	MOVD	rows+24(FP), R22   // rows dimension
	MOVD	cols+32(FP), R23   // cols dimension

	// Calculate stride (in bytes)
	LSL	$2, R22, R26       // R26 = rows * 4 (MT row stride)

	// Enter streaming mode with ZA enabled
	WORD	$0xd503477f        // smstart

	// Set up predicates for 32-bit elements
	WORD	$0x2598e3e0        // ptrue p0.s
	WORD	$0x2598e3e1        // ptrue p1.s

	// Zero R10 for use as offset register in LD1W
	MOVD	$0, R10

	// Outer loop: for row = 0; row < rows; row += 16
	MOVD	$0, R0             // R0 = row (tile row index)

tile_loop:
	CMP	R22, R0            // row - rows
	BGE	done

	// Refresh predicates
	WORD	$0x2598e3e0        // ptrue p0.s
	WORD	$0x2598e3e1        // ptrue p1.s

	// Zero all ZA tiles for accumulation
	WORD	$0xc00800ff        // zero {za}

	// Inner loop: for k = 0; k < cols; k++
	MOVD	$0, R3             // R3 = k

k_loop:
	CMP	R23, R3            // k - cols
	BGE	store_result

	// === Load M column: M[row:row+16, k] into z2 (like matmul loads A column) ===
	// With MT (cols x rows, row-major), M[row:row+16, k] = MT[k, row:row+16]
	// This is CONTIGUOUS at: MT_base + k * rows * 4 + row * 4
	MUL	R26, R3, R4        // R4 = k * rows * 4
	ADD	R19, R4, R4        // R4 = &MT[k, 0]
	LSL	$2, R0, R5         // R5 = row * 4
	ADD	R4, R5, R4         // R4 = &MT[k, row] (contiguous 16 floats!)

	// LD1W z2.s, p0/z, [x4, x10, lsl #2] - same as matmul
	WORD	$0xa54a4082        // ld1w {z2.s}, p0/z, [x4, x10, lsl #2]

	// === Load and broadcast v[k] into z0 ===
	LSL	$2, R3, R6         // R6 = k * 4
	ADD	R20, R6, R6        // R6 = &v[k]

	// Load v[k] into z0 via LD1RW (broadcast load)
	// LD1RW z0.s, p0/z, [x6] - broadcast scalar to all lanes
	WORD	$0x8540c0c0        // ld1rw {z0.s}, p0/z, [x6]

	// === FMOPA: za0 += z2 ⊗ z0 (outer product) - EXACT SAME AS MATMUL ===
	// fmopa za0.s, p0/m, p1/m, z2.s, z0.s
	WORD	$0x80802040

	ADD	$1, R3, R3
	B	k_loop

store_result:
	// Extract za0 rows and store first element of each to result
	// After accumulation, za[i][j] = dot(row i, v) for all j (all columns same)
	// We use scalar store via FMOV to extract just element 0 of each row

	// Calculate result pointer: &result[row]
	LSL	$2, R0, R4         // R4 = row * 4
	ADD	R21, R4, R4        // R4 = &result[row]

	MOVD	$0, R12            // W12 = slice index
	MOVD	R4, R6             // R6 = current result pointer

store_rows:
	CMP	$16, R12
	BGE	next_tile

	// MOVA z2.s, p0/m, za0h.s[w12, 0] - extract horizontal slice (same as matmul)
	WORD	$0xc0820002

	// Use FMOV to extract s2 (first 32-bit element) to w7
	// In streaming mode, the lower 128 bits of Z registers are V registers
	// FMOV w7, s2 - encoding: 0x1e260047
	WORD	$0x1e260047        // fmov w7, s2

	// Store w7 to memory
	MOVW	R7, (R6)

	// Move to next result element
	ADD	$4, R6, R6         // Next result (stride = 4 bytes)
	ADD	$1, R12, R12       // Next ZA slice
	B	store_rows

next_tile:
	ADD	$16, R0, R0
	B	tile_loop

done:
	// Exit streaming mode
	WORD	$0xd503467f        // smstop

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


// func matvec_sme_f64(mt, v, result unsafe.Pointer, rows, cols int64)
//
// FMOPA-based matrix-vector multiplication for float64.
// Uses 8x8 tiles instead of 16x16.
//
// Apple M4 SVL = 512 bits = 8 × float64
// ZA tiles = 8×8 = 64 float64 elements
//
TEXT ·matvec_sme_f64(SB), NOSPLIT, $64-40
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
	MOVD	mt+0(FP), R19      // MT matrix base (cols x rows, row-major)
	MOVD	v+8(FP), R20       // v vector base
	MOVD	result+16(FP), R21 // result vector base
	MOVD	rows+24(FP), R22   // rows dimension
	MOVD	cols+32(FP), R23   // cols dimension

	// Calculate stride (in bytes for float64)
	LSL	$3, R22, R26       // R26 = rows * 8 (MT row stride)

	// Enter streaming mode with ZA enabled
	WORD	$0xd503477f        // smstart

	// Set up predicates for 64-bit elements
	WORD	$0x25d8e3e0        // ptrue p0.d
	WORD	$0x25d8e3e1        // ptrue p1.d

	// Zero R10 for use as offset register in LD1D
	MOVD	$0, R10

	// Outer loop: for row = 0; row < rows; row += 8
	MOVD	$0, R0             // R0 = row (tile row index)

tile_loop_f64:
	CMP	R22, R0            // row - rows
	BGE	done_f64

	// Refresh predicates
	WORD	$0x25d8e3e0        // ptrue p0.d
	WORD	$0x25d8e3e1        // ptrue p1.d

	// Zero all ZA tiles for accumulation
	WORD	$0xc00800ff        // zero {za}

	// Inner loop: for k = 0; k < cols; k++
	MOVD	$0, R3             // R3 = k

k_loop_f64:
	CMP	R23, R3            // k - cols
	BGE	store_result_f64

	// === Load M column into z2 ===
	MUL	R26, R3, R4        // R4 = k * rows * 8
	ADD	R19, R4, R4        // R4 = &MT[k, 0]
	LSL	$3, R0, R5         // R5 = row * 8
	ADD	R4, R5, R4         // R4 = &MT[k, row] (contiguous 8 doubles!)

	// LD1D z2.d, p0/z, [x4, x10, lsl #3]
	WORD	$0xa5ea4082        // ld1d {z2.d}, p0/z, [x4, x10, lsl #3]

	// === Load and broadcast v[k] into z0 ===
	LSL	$3, R3, R6         // R6 = k * 8
	ADD	R20, R6, R6        // R6 = &v[k]

	// Load scalar v[k] into x7
	MOVD	(R6), R7

	// DUP z0.d, x7 - broadcast scalar to all lanes
	// Encoding for 64-bit: 0x05e03800 + (Rn << 5) + Rd
	// DUP z0.d, x7: 0x05e038e0
	WORD	$0x05e038e0        // dup z0.d, x7

	// === FMOPA: za0 += z2 ⊗ z0 (outer product for float64) ===
	// fmopa za0.d, p0/m, p1/m, z2.d, z0.d
	WORD	$0x80c02040

	ADD	$1, R3, R3
	B	k_loop_f64

store_result_f64:
	// Extract horizontal slices from za0 and store first element of each

	// Calculate result pointer: &result[row]
	LSL	$3, R0, R4         // R4 = row * 8
	ADD	R21, R4, R4        // R4 = &result[row]

	MOVD	$0, R12            // W12 = slice index
	MOVD	R4, R6             // R6 = current result pointer

store_rows_f64:
	CMP	$8, R12
	BGE	next_tile_f64

	// MOVA z2.d, p0/m, za0h.d[w12, 0] - extract horizontal slice (row)
	WORD	$0xc0c20002        // mova z2.d, p0/m, za0h.d[w12, 0]

	// Use FMOV to extract d2 (first 64-bit element) to x7
	// FMOV x7, d2 - encoding: 0x9e660047
	WORD	$0x9e660047        // fmov x7, d2

	// Store x7 to memory
	MOVD	R7, (R6)

	// Move to next result element
	ADD	$8, R6, R6         // Next result (stride = 8 bytes)
	ADD	$1, R12, R12       // Next ZA slice
	B	store_rows_f64

next_tile_f64:
	ADD	$8, R0, R0
	B	tile_loop_f64

done_f64:
	// Exit streaming mode
	WORD	$0xd503467f        // smstop

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
