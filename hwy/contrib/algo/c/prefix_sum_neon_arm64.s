	.build_version macos, 15, 0	sdk_version 15, 5
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_prefix_sum_inplace_f32         ; -- Begin function prefix_sum_inplace_f32
	.p2align	2
_prefix_sum_inplace_f32:                ; @prefix_sum_inplace_f32
; %bb.0:
	cmp	x1, #4
	b.ge	LBB0_2
; %bb.1:
	mov	x10, #0                         ; =0x0
	movi.2d	v0, #0000000000000000
	b	LBB0_4
LBB0_2:
	mov	x8, #0                          ; =0x0
	movi.2d	v1, #0000000000000000
	mov	x9, x0
	movi.2d	v0, #0000000000000000
LBB0_3:                                 ; =>This Inner Loop Header: Depth=1
	ldr	q2, [x9]
	ext.16b	v3, v1, v2, #12
	fadd.4s	v2, v2, v3
	ext.16b	v3, v1, v2, #8
	fadd.4s	v2, v2, v3
	fadd.4s	v0, v0, v2
	str	q0, [x9], #16
	dup.4s	v0, v0[3]
	add	x10, x8, #4
	add	x11, x8, #8
	mov	x8, x10
	cmp	x11, x1
	b.le	LBB0_3
LBB0_4:
	subs	x8, x1, x10
	b.le	LBB0_7
; %bb.5:
	add	x9, x0, x10, lsl #2
LBB0_6:                                 ; =>This Inner Loop Header: Depth=1
	ldr	s1, [x9]
	fadd	s0, s0, s1
	str	s0, [x9], #4
	subs	x8, x8, #1
	b.ne	LBB0_6
LBB0_7:
	ret
                                        ; -- End function
	.globl	_prefix_sum_inplace_f64         ; -- Begin function prefix_sum_inplace_f64
	.p2align	2
_prefix_sum_inplace_f64:                ; @prefix_sum_inplace_f64
; %bb.0:
	cmp	x1, #2
	b.ge	LBB1_2
; %bb.1:
	mov	x8, #0                          ; =0x0
	movi.2d	v0, #0000000000000000
	b	LBB1_4
LBB1_2:
	mov	x9, #0                          ; =0x0
	movi.2d	v1, #0000000000000000
	mov	x10, x0
	movi.2d	v0, #0000000000000000
LBB1_3:                                 ; =>This Inner Loop Header: Depth=1
	ldr	q2, [x10]
	ext.16b	v3, v1, v2, #8
	fadd.2d	v2, v2, v3
	fadd.2d	v0, v0, v2
	str	q0, [x10], #16
	dup.2d	v0, v0[1]
	add	x8, x9, #2
	add	x11, x9, #4
	mov	x9, x8
	cmp	x11, x1
	b.le	LBB1_3
LBB1_4:
	subs	x9, x1, x8
	b.le	LBB1_7
; %bb.5:
	add	x8, x0, x8, lsl #3
LBB1_6:                                 ; =>This Inner Loop Header: Depth=1
	ldr	d1, [x8]
	fadd	d0, d0, d1
	str	d0, [x8], #8
	subs	x9, x9, #1
	b.ne	LBB1_6
LBB1_7:
	ret
                                        ; -- End function
	.globl	_prefix_sum_inplace_i32         ; -- Begin function prefix_sum_inplace_i32
	.p2align	2
_prefix_sum_inplace_i32:                ; @prefix_sum_inplace_i32
; %bb.0:
	cmp	x1, #4
	b.ge	LBB2_2
; %bb.1:
	mov	x10, #0                         ; =0x0
	movi.2d	v0, #0000000000000000
	b	LBB2_4
LBB2_2:
	mov	x8, #0                          ; =0x0
	movi.2d	v1, #0000000000000000
	mov	x9, x0
	movi.2d	v0, #0000000000000000
LBB2_3:                                 ; =>This Inner Loop Header: Depth=1
	ldr	q2, [x9]
	ext.16b	v3, v1, v2, #12
	add.4s	v2, v3, v2
	ext.16b	v3, v1, v2, #8
	add.4s	v0, v2, v0
	add.4s	v0, v0, v3
	str	q0, [x9], #16
	dup.4s	v0, v0[3]
	add	x10, x8, #4
	add	x11, x8, #8
	mov	x8, x10
	cmp	x11, x1
	b.le	LBB2_3
LBB2_4:
	subs	x8, x1, x10
	b.le	LBB2_7
; %bb.5:
	fmov	w9, s0
	add	x10, x0, x10, lsl #2
LBB2_6:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w11, [x10]
	add	w9, w11, w9
	str	w9, [x10], #4
	subs	x8, x8, #1
	b.ne	LBB2_6
LBB2_7:
	ret
                                        ; -- End function
	.globl	_prefix_sum_inplace_i64         ; -- Begin function prefix_sum_inplace_i64
	.p2align	2
_prefix_sum_inplace_i64:                ; @prefix_sum_inplace_i64
; %bb.0:
	cmp	x1, #1
	b.lt	LBB3_3
; %bb.1:
	mov	x8, #0                          ; =0x0
LBB3_2:                                 ; =>This Inner Loop Header: Depth=1
	ldr	x9, [x0]
	add	x8, x9, x8
	str	x8, [x0], #8
	subs	x1, x1, #1
	b.ne	LBB3_2
LBB3_3:
	ret
                                        ; -- End function
	.globl	_prefix_sum_inplace_u32         ; -- Begin function prefix_sum_inplace_u32
	.p2align	2
_prefix_sum_inplace_u32:                ; @prefix_sum_inplace_u32
; %bb.0:
	cmp	x1, #4
	b.ge	LBB4_2
; %bb.1:
	mov	x10, #0                         ; =0x0
	movi.2d	v0, #0000000000000000
	b	LBB4_4
LBB4_2:
	mov	x8, #0                          ; =0x0
	movi.2d	v1, #0000000000000000
	mov	x9, x0
	movi.2d	v0, #0000000000000000
LBB4_3:                                 ; =>This Inner Loop Header: Depth=1
	ldr	q2, [x9]
	ext.16b	v3, v1, v2, #12
	add.4s	v2, v3, v2
	ext.16b	v3, v1, v2, #8
	add.4s	v0, v2, v0
	add.4s	v0, v0, v3
	str	q0, [x9], #16
	dup.4s	v0, v0[3]
	add	x10, x8, #4
	add	x11, x8, #8
	mov	x8, x10
	cmp	x11, x1
	b.le	LBB4_3
LBB4_4:
	subs	x8, x1, x10
	b.le	LBB4_7
; %bb.5:
	fmov	w9, s0
	add	x10, x0, x10, lsl #2
LBB4_6:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w11, [x10]
	add	w9, w11, w9
	str	w9, [x10], #4
	subs	x8, x8, #1
	b.ne	LBB4_6
LBB4_7:
	ret
                                        ; -- End function
	.globl	_prefix_sum_inplace_u64         ; -- Begin function prefix_sum_inplace_u64
	.p2align	2
_prefix_sum_inplace_u64:                ; @prefix_sum_inplace_u64
; %bb.0:
	cmp	x1, #1
	b.lt	LBB5_3
; %bb.1:
	mov	x8, #0                          ; =0x0
LBB5_2:                                 ; =>This Inner Loop Header: Depth=1
	ldr	x9, [x0]
	add	x8, x9, x8
	str	x8, [x0], #8
	subs	x1, x1, #1
	b.ne	LBB5_2
LBB5_3:
	ret
                                        ; -- End function
	.globl	_delta_decode_inplace_i32       ; -- Begin function delta_decode_inplace_i32
	.p2align	2
_delta_decode_inplace_i32:              ; @delta_decode_inplace_i32
; %bb.0:
	dup.4s	v0, w2
	cmp	x1, #4
	b.ge	LBB6_2
; %bb.1:
	mov	x10, #0                         ; =0x0
	b	LBB6_4
LBB6_2:
	mov	x8, #0                          ; =0x0
	movi.2d	v1, #0000000000000000
	mov	x9, x0
LBB6_3:                                 ; =>This Inner Loop Header: Depth=1
	ldr	q2, [x9]
	ext.16b	v3, v1, v2, #12
	add.4s	v2, v3, v2
	ext.16b	v3, v1, v2, #8
	add.4s	v0, v2, v0
	add.4s	v0, v0, v3
	str	q0, [x9], #16
	dup.4s	v0, v0[3]
	add	x10, x8, #4
	add	x11, x8, #8
	mov	x8, x10
	cmp	x11, x1
	b.le	LBB6_3
LBB6_4:
	subs	x8, x1, x10
	b.le	LBB6_7
; %bb.5:
	fmov	w9, s0
	add	x10, x0, x10, lsl #2
LBB6_6:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w11, [x10]
	add	w9, w11, w9
	str	w9, [x10], #4
	subs	x8, x8, #1
	b.ne	LBB6_6
LBB6_7:
	ret
                                        ; -- End function
	.globl	_delta_decode_inplace_i64       ; -- Begin function delta_decode_inplace_i64
	.p2align	2
_delta_decode_inplace_i64:              ; @delta_decode_inplace_i64
; %bb.0:
	cmp	x1, #1
	b.lt	LBB7_2
LBB7_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	x8, [x0]
	add	x2, x8, x2
	str	x2, [x0], #8
	subs	x1, x1, #1
	b.ne	LBB7_1
LBB7_2:
	ret
                                        ; -- End function
	.globl	_delta_decode_inplace_u32       ; -- Begin function delta_decode_inplace_u32
	.p2align	2
_delta_decode_inplace_u32:              ; @delta_decode_inplace_u32
; %bb.0:
	dup.4s	v0, w2
	cmp	x1, #4
	b.ge	LBB8_2
; %bb.1:
	mov	x10, #0                         ; =0x0
	b	LBB8_4
LBB8_2:
	mov	x8, #0                          ; =0x0
	movi.2d	v1, #0000000000000000
	mov	x9, x0
LBB8_3:                                 ; =>This Inner Loop Header: Depth=1
	ldr	q2, [x9]
	ext.16b	v3, v1, v2, #12
	add.4s	v2, v3, v2
	ext.16b	v3, v1, v2, #8
	add.4s	v0, v2, v0
	add.4s	v0, v0, v3
	str	q0, [x9], #16
	dup.4s	v0, v0[3]
	add	x10, x8, #4
	add	x11, x8, #8
	mov	x8, x10
	cmp	x11, x1
	b.le	LBB8_3
LBB8_4:
	subs	x8, x1, x10
	b.le	LBB8_7
; %bb.5:
	fmov	w9, s0
	add	x10, x0, x10, lsl #2
LBB8_6:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w11, [x10]
	add	w9, w11, w9
	str	w9, [x10], #4
	subs	x8, x8, #1
	b.ne	LBB8_6
LBB8_7:
	ret
                                        ; -- End function
	.globl	_delta_decode_inplace_u64       ; -- Begin function delta_decode_inplace_u64
	.p2align	2
_delta_decode_inplace_u64:              ; @delta_decode_inplace_u64
; %bb.0:
	cmp	x1, #1
	b.lt	LBB9_2
LBB9_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	x8, [x0]
	add	x2, x8, x2
	str	x2, [x0], #8
	subs	x1, x1, #1
	b.ne	LBB9_1
LBB9_2:
	ret
                                        ; -- End function
.subsections_via_symbols
