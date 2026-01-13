//go:build !noasm && darwin && arm64

#include "textflag.h"

// =============================================================================
// Float32 SME FMOPA - TRUE outer product accumulate into ZA tile
// =============================================================================

// func block_muladd_fmopa_f32(aT, b, c unsafe.Pointer, blockDim int64)
//
// Computes C += A * B for square blocks where:
//   - aT is PRE-TRANSPOSED A (rows are original A columns)
//   - b is normal B (rows are B rows)
//
// Uses TRUE FMOPA outer product instruction:
//   - Each FMOPA computes 16×16 = 256 FMAs in one instruction!
//   - ZA tile accumulates results, then added to C
//
// Memory layout (optimal for outer product):
//   - aT[k, i:i+16] gives A[i:i+16, k] contiguously → z2
//   - b[k, j:j+16] gives B[k, j:j+16] contiguously → z0
//   - FMOPA ZA0.S += z2 ⊗ z0 (outer product)
//
// Requires blockDim to be multiple of 16 (SVL on M4).
//
TEXT ·block_muladd_fmopa_f32(SB), NOSPLIT, $64-32
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
	LSL	$2, R22, R23       // R23 = blockDim * 4 (stride in bytes)

	// Enter streaming mode with ZA
	WORD	$0xd503477f        // smstart

	// Set up predicates for 32-bit elements
	WORD	$0x2598e3e0        // ptrue p0.s
	WORD	$0x2598e3e1        // ptrue p1.s

	// Zero R10 for scalar+scalar addressing
	MOVD	$0, R10

	// =================================================================
	// Tile loop: process 16×16 tiles
	// =================================================================

	// Outer loop: i over tile rows (step by 16)
	MOVD	$0, R0             // R0 = i (tile row start)

fmopa_tile_row_loop:
	CMP	R22, R0            // Compare: if i >= blockDim, done
	BGE	fmopa_done

	// Inner loop: j over tile columns (step by 16)
	MOVD	$0, R2             // R2 = j (tile col start)

fmopa_tile_col_loop:
	CMP	R22, R2            // Compare: if j >= blockDim, next row
	BGE	fmopa_next_tile_row

	// Refresh predicates
	WORD	$0x2598e3e0        // ptrue p0.s
	WORD	$0x2598e3e1        // ptrue p1.s

	// Zero ZA tile for this 16×16 block
	WORD	$0xc00800ff        // zero {za}

	// K-loop: accumulate outer products into ZA
	MOVD	$0, R3             // R3 = k

fmopa_k_loop:
	CMP	R22, R3            // Compare: if k >= blockDim, k-loop done
	BGE	fmopa_k_done

	// Load z2 = aT[k, i:i+16] (16 values from column k of A)
	// Address: aT + k*stride + i*4
	MUL	R23, R3, R4        // R4 = k * stride
	ADD	R19, R4, R4        // R4 = &aT[k, 0]
	LSL	$2, R0, R5         // R5 = i * 4
	ADD	R4, R5, R4         // R4 = &aT[k, i]
	WORD	$0xa54a4082        // ld1w {z2.s}, p0/z, [x4, x10, lsl #2]

	// Load z0 = b[k, j:j+16] (16 values from row k of B)
	// Address: b + k*stride + j*4
	MUL	R23, R3, R6        // R6 = k * stride
	ADD	R20, R6, R6        // R6 = &B[k, 0]
	LSL	$2, R2, R7         // R7 = j * 4
	ADD	R6, R7, R6         // R6 = &B[k, j]
	WORD	$0xa54a40c0        // ld1w {z0.s}, p0/z, [x6, x10, lsl #2]

	// FMOPA za0.s, p0/m, p1/m, z2.s, z0.s
	// Outer product: ZA0[r,c] += z2[r] * z0[c] for all r,c
	// This is 16×16 = 256 FMAs in ONE instruction!
	WORD	$0x80802040        // fmopa za0.s, p0/m, p1/m, z2.s, z0.s

	ADD	$1, R3, R3
	B	fmopa_k_loop

fmopa_k_done:
	// ---------------------------------------------------------
	// Store loop: add ZA tile to C and store back
	// For each row r: C[i+r, j:j+16] += ZA0[r, :]
	// ---------------------------------------------------------

	// Calculate base C pointer: &C[i, j]
	MUL	R23, R0, R4        // R4 = i * stride
	ADD	R21, R4, R4        // R4 = &C[i, 0]
	LSL	$2, R2, R5         // R5 = j * 4
	ADD	R4, R5, R4         // R4 = &C[i, j]

	MOVD	$0, R12            // R12 = row counter (w12 for MOVA slice index)
	MOVD	R4, R6             // R6 = current C row pointer

fmopa_store_loop:
	CMP	$16, R12           // Compare: if r >= 16, store done
	BGE	fmopa_store_done

	// Load current C row into z3 (using scalar+scalar with x10=0)
	WORD	$0xa54a40c3        // ld1w {z3.s}, p0/z, [x6, x10, lsl #2]

	// MOVA z2.s, p0/m, za0h.s[w12, 0]
	// Extract horizontal row w12 from ZA0 into z2
	WORD	$0xc0820002        // mova z2.s, p0/m, za0h.s[w12, 0]

	// FADD z3.s, z3.s, z2.s (add ZA row to C row)
	WORD	$0x65820063        // fadd z3.s, z3.s, z2.s

	// ST1W {z3.s}, p0, [x6, x10, lsl #2] (using scalar+scalar with x10=0)
	WORD	$0xe54a40c3        // st1w {z3.s}, p0, [x6, x10, lsl #2]

	// Move to next row
	ADD	R23, R6, R6        // Next C row (stride = blockDim*4)
	ADD	$1, R12, R12       // Next ZA slice

	B	fmopa_store_loop

fmopa_store_done:
	// Next tile column
	ADD	$16, R2, R2        // j += 16
	B	fmopa_tile_col_loop

fmopa_next_tile_row:
	ADD	$16, R0, R0        // i += 16
	B	fmopa_tile_row_loop

fmopa_done:
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


// =============================================================================
// Float64 SME FMOPA - TRUE outer product for double precision
// =============================================================================

// func block_muladd_fmopa_f64(aT, b, c unsafe.Pointer, blockDim int64)
//
// Same algorithm as f32 but with 8×8 tiles (8 doubles per SVE vector on M4).
//
TEXT ·block_muladd_fmopa_f64(SB), NOSPLIT, $64-32
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

	// Enter streaming mode with ZA
	WORD	$0xd503477f        // smstart

	// Set up predicates for 64-bit elements
	WORD	$0x25d8e3e0        // ptrue p0.d
	WORD	$0x25d8e3e1        // ptrue p1.d

	// Zero R10 for scalar+scalar addressing
	MOVD	$0, R10

	// Tile loop: process 8×8 tiles (8 doubles per vector on M4)
	MOVD	$0, R0             // R0 = i (tile row start)

fmopa64_tile_row_loop:
	CMP	R22, R0
	BGE	fmopa64_done

	MOVD	$0, R2             // R2 = j (tile col start)

fmopa64_tile_col_loop:
	CMP	R22, R2
	BGE	fmopa64_next_tile_row

	// Refresh predicates
	WORD	$0x25d8e3e0        // ptrue p0.d
	WORD	$0x25d8e3e1        // ptrue p1.d

	// Zero ZA tile
	WORD	$0xc00800ff        // zero {za}

	// K-loop
	MOVD	$0, R3             // R3 = k

fmopa64_k_loop:
	CMP	R22, R3
	BGE	fmopa64_k_done

	// Load z2 = aT[k, i:i+8]
	MUL	R23, R3, R4
	ADD	R19, R4, R4
	LSL	$3, R0, R5
	ADD	R4, R5, R4
	WORD	$0xa5ea4082        // ld1d {z2.d}, p0/z, [x4, x10, lsl #3]

	// Load z0 = b[k, j:j+8]
	MUL	R23, R3, R6
	ADD	R20, R6, R6
	LSL	$3, R2, R7
	ADD	R6, R7, R6
	WORD	$0xa5ea40c0        // ld1d {z0.d}, p0/z, [x6, x10, lsl #3]

	// FMOPA za0.d, p0/m, p1/m, z2.d, z0.d
	// 64-bit outer product (8×8 = 64 FMAs per instruction)
	WORD	$0x80c02040        // fmopa za0.d, p0/m, p1/m, z2.d, z0.d

	ADD	$1, R3, R3
	B	fmopa64_k_loop

fmopa64_k_done:
	// Calculate base C pointer
	MUL	R23, R0, R4
	ADD	R21, R4, R4
	LSL	$3, R2, R5
	ADD	R4, R5, R4

	MOVD	$0, R12
	MOVD	R4, R6

fmopa64_store_loop:
	CMP	$8, R12            // 8 rows for f64
	BGE	fmopa64_store_done

	// Load current C row (using scalar+scalar with x10=0)
	// Derived from working 0xa5ea4082: change Rn x4→x6, Zt z2→z3
	WORD	$0xa5ea40c3        // ld1d {z3.d}, p0/z, [x6, x10, lsl #3]

	// MOVA z2.d, p0/m, za0h.d[w12, 0]
	WORD	$0xc0c20002        // mova z2.d, p0/m, za0h.d[w12, 0]

	// FADD z3.d, z3.d, z2.d
	WORD	$0x65c20063        // fadd z3.d, z3.d, z2.d

	// Store C row (using scalar+scalar with x10=0)
	// ST1D: change LD opcode 0xa5 → 0xe5
	WORD	$0xe5ea40c3        // st1d {z3.d}, p0, [x6, x10, lsl #3]

	ADD	R23, R6, R6
	ADD	$1, R12, R12

	B	fmopa64_store_loop

fmopa64_store_done:
	ADD	$8, R2, R2         // j += 8 for f64
	B	fmopa64_tile_col_loop

fmopa64_next_tile_row:
	ADD	$8, R0, R0         // i += 8 for f64
	B	fmopa64_tile_row_loop

fmopa64_done:
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
