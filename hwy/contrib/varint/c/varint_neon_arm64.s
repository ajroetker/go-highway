	.build_version macos, 15, 0	sdk_version 15, 5
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3, 0x0                          ; -- Begin function find_varint_ends_u8
lCPI0_0:
	.short	2                               ; 0x2
	.short	3                               ; 0x3
	.short	6                               ; 0x6
	.short	7                               ; 0x7
	.section	__TEXT,__literal16,16byte_literals
	.p2align	4, 0x0
lCPI0_1:
	.quad	12                              ; 0xc
	.quad	13                              ; 0xd
lCPI0_2:
	.quad	8                               ; 0x8
	.quad	9                               ; 0x9
lCPI0_3:
	.quad	14                              ; 0xe
	.quad	15                              ; 0xf
lCPI0_4:
	.quad	10                              ; 0xa
	.quad	11                              ; 0xb
lCPI0_5:
	.quad	6                               ; 0x6
	.quad	7                               ; 0x7
lCPI0_6:
	.quad	4                               ; 0x4
	.quad	5                               ; 0x5
lCPI0_7:
	.quad	2                               ; 0x2
	.quad	3                               ; 0x3
lCPI0_8:
	.quad	0                               ; 0x0
	.quad	1                               ; 0x1
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_find_varint_ends_u8
	.p2align	2
_find_varint_ends_u8:                   ; @find_varint_ends_u8
; %bb.0:
	cmp	x1, #1
	b.lt	LBB0_3
; %bb.1:
	stp	d11, d10, [sp, #-32]!           ; 16-byte Folded Spill
	stp	d9, d8, [sp, #16]               ; 16-byte Folded Spill
	mov	w8, #64                         ; =0x40
	cmp	x1, #64
	csel	x8, x1, x8, lo
	cmp	x1, #15
	b.hi	LBB0_4
; %bb.2:
	mov	x10, #0                         ; =0x0
	mov	x9, #0                          ; =0x0
	b	LBB0_11
LBB0_3:
	str	xzr, [x2]
	ret
LBB0_4:
	ldr	q0, [x0]
	cmge.16b	v0, v0, #0
	movi.16b	v5, #1
	and.16b	v1, v0, v5
	umov.b	w9, v1[0]
	umov.b	w10, v1[1]
	umov.b	w11, v1[4]
	umov.b	w12, v1[5]
	mov	b2, v1[2]
	mov.b	v2[2], v1[3]
	mov.b	v2[4], v1[6]
	lsl	w12, w12, #5
	mov.b	v2[6], v1[7]
Lloh0:
	adrp	x13, lCPI0_0@PAGE
Lloh1:
	ldr	d0, [x13, lCPI0_0@PAGEOFF]
	ushl.4h	v2, v2, v0
	fmov	x13, d2
	orr	x13, x13, x13, lsr #32
	lsr	x14, x13, #16
	orr	w9, w13, w9
	orr	w9, w9, w14
	orr	w10, w12, w10, lsl #1
	orr	w9, w9, w10
	orr	w9, w9, w11, lsl #4
	and	x9, x9, #0xff
	ushll2.8h	v1, v1, #0
	ushll.4s	v2, v1, #0
	ushll2.2d	v6, v2, #0
	ushll2.4s	v1, v1, #0
	ushll2.2d	v4, v1, #0
	ushll.2d	v3, v2, #0
	ushll.2d	v2, v1, #0
Lloh2:
	adrp	x10, lCPI0_1@PAGE
Lloh3:
	ldr	q1, [x10, lCPI0_1@PAGEOFF]
	ushl.2d	v7, v2, v1
Lloh4:
	adrp	x10, lCPI0_2@PAGE
Lloh5:
	ldr	q2, [x10, lCPI0_2@PAGEOFF]
	ushl.2d	v16, v3, v2
Lloh6:
	adrp	x10, lCPI0_3@PAGE
Lloh7:
	ldr	q3, [x10, lCPI0_3@PAGEOFF]
	ushl.2d	v17, v4, v3
Lloh8:
	adrp	x10, lCPI0_4@PAGE
Lloh9:
	ldr	q4, [x10, lCPI0_4@PAGEOFF]
	ushl.2d	v6, v6, v4
	orr.16b	v6, v6, v17
	orr.16b	v7, v16, v7
	orr.16b	v6, v7, v6
	ext.16b	v7, v6, v6, #8
	orr.8b	v6, v6, v7
	fmov	x10, d6
	orr	x9, x10, x9
	cmp	x1, #32
	b.hs	LBB0_6
; %bb.5:
	mov	w10, #16                        ; =0x10
	b	LBB0_11
LBB0_6:
	ldr	q6, [x0, #16]
	cmge.16b	v6, v6, #0
	and.16b	v5, v6, v5
	umov.b	w10, v5[0]
	umov.b	w11, v5[1]
	umov.b	w12, v5[4]
	umov.b	w13, v5[5]
	lsl	w13, w13, #5
	mov	b6, v5[2]
	mov.b	v6[2], v5[3]
	mov.b	v6[4], v5[6]
	mov.b	v6[6], v5[7]
	ushl.4h	v6, v6, v0
	fmov	x14, d6
	orr	x14, x14, x14, lsr #32
	lsr	x15, x14, #16
	orr	w10, w14, w10
	orr	w10, w10, w15
	orr	w11, w13, w11, lsl #1
	orr	w10, w10, w11
	orr	w10, w10, w12, lsl #4
	and	x10, x10, #0xff
	ushll2.8h	v5, v5, #0
	ushll.4s	v6, v5, #0
	ushll2.2d	v7, v6, #0
	ushll2.4s	v5, v5, #0
	ushll2.2d	v16, v5, #0
	ushll.2d	v6, v6, #0
	ushll.2d	v5, v5, #0
	ushl.2d	v5, v5, v1
	ushl.2d	v6, v6, v2
	ushl.2d	v16, v16, v3
	ushl.2d	v7, v7, v4
	orr.16b	v7, v7, v16
	orr.16b	v5, v6, v5
	orr.16b	v5, v5, v7
	ext.16b	v6, v5, v5, #8
	orr.8b	v5, v5, v6
	fmov	x11, d5
	orr	x10, x11, x10
	orr	x9, x9, x10, lsl #16
	cmp	x1, #48
	b.hs	LBB0_8
; %bb.7:
	mov	w10, #32                        ; =0x20
	b	LBB0_11
LBB0_8:
	ldr	q5, [x0, #32]
	cmge.16b	v6, v5, #0
	movi.16b	v5, #1
	and.16b	v6, v6, v5
	umov.b	w10, v6[0]
	umov.b	w11, v6[1]
	umov.b	w12, v6[4]
	umov.b	w13, v6[5]
	lsl	w13, w13, #5
	mov	b7, v6[2]
	mov.b	v7[2], v6[3]
	mov.b	v7[4], v6[6]
	mov.b	v7[6], v6[7]
	ushl.4h	v7, v7, v0
	fmov	x14, d7
	lsr	x15, x14, #32
	orr	w14, w14, w15
	orr	w14, w14, w14, lsr #16
	orr	w11, w13, w11, lsl #1
	orr	w10, w10, w11
	orr	w10, w14, w10
	orr	w10, w10, w12, lsl #4
	ushll2.8h	v6, v6, #0
	ushll.4s	v7, v6, #0
	ushll2.2d	v16, v7, #0
	ushll2.4s	v6, v6, #0
	ushll2.2d	v17, v6, #0
	ushll.2d	v7, v7, #0
	ushll.2d	v6, v6, #0
	ushl.2d	v6, v6, v1
	ushl.2d	v7, v7, v2
	ushl.2d	v17, v17, v3
	ushl.2d	v16, v16, v4
	orr.16b	v16, v16, v17
	orr.16b	v6, v7, v6
	orr.16b	v6, v6, v16
	ext.16b	v7, v6, v6, #8
	orr.8b	v6, v6, v7
	fmov	x11, d6
	and	w10, w10, #0xff
	orr	w10, w11, w10
	orr	x9, x9, x10, lsl #32
	cmp	x1, #64
	b.hs	LBB0_10
; %bb.9:
	mov	w10, #48                        ; =0x30
	b	LBB0_11
LBB0_10:
	ldr	q6, [x0, #48]
	cmge.16b	v6, v6, #0
	and.16b	v5, v6, v5
	umov.b	w10, v5[0]
	umov.b	w11, v5[1]
	umov.b	w12, v5[4]
	umov.b	w13, v5[5]
	lsl	w13, w13, #5
	mov	b6, v5[2]
	mov.b	v6[2], v5[3]
	mov.b	v6[4], v5[6]
	mov.b	v6[6], v5[7]
	ushl.4h	v0, v6, v0
	fmov	x14, d0
	lsr	x15, x14, #32
	orr	w14, w14, w15
	orr	w14, w14, w14, lsr #16
	orr	w11, w13, w11, lsl #1
	orr	w10, w10, w11
	orr	w10, w14, w10
	orr	w10, w10, w12, lsl #4
	ushll2.8h	v0, v5, #0
	ushll.4s	v5, v0, #0
	ushll2.2d	v6, v5, #0
	ushll2.4s	v0, v0, #0
	ushll2.2d	v7, v0, #0
	ushll.2d	v5, v5, #0
	ushll.2d	v0, v0, #0
	ushl.2d	v0, v0, v1
	ushl.2d	v1, v5, v2
	ushl.2d	v2, v7, v3
	ushl.2d	v3, v6, v4
	orr.16b	v2, v3, v2
	orr.16b	v0, v1, v0
	orr.16b	v0, v0, v2
	ext.16b	v1, v0, v0, #8
	orr.8b	v0, v0, v1
	fmov	x11, d0
	and	w10, w10, #0xff
	orr	w10, w11, w10
	orr	x9, x9, x10, lsl #48
	mov	w10, #64                        ; =0x40
LBB0_11:
	subs	x12, x8, x10
	b.ls	LBB0_26
; %bb.12:
	cmp	x12, #3
	b.hi	LBB0_14
; %bb.13:
	mov	x12, x10
	b	LBB0_24
LBB0_14:
	adrp	x13, lCPI0_7@PAGE
	adrp	x11, lCPI0_8@PAGE
	cmp	x12, #16
	b.hs	LBB0_16
; %bb.15:
	mov	x14, #0                         ; =0x0
	mov	x15, x10
	b	LBB0_21
LBB0_16:
	movi.2d	v0, #0000000000000000
	movi.2d	v1, #0000000000000000
	mov.d	v1[0], x9
	dup.2d	v17, x10
Lloh10:
	adrp	x9, lCPI0_3@PAGE
Lloh11:
	ldr	q2, [x9, lCPI0_3@PAGEOFF]
	orr.16b	v2, v17, v2
Lloh12:
	adrp	x9, lCPI0_1@PAGE
Lloh13:
	ldr	q3, [x9, lCPI0_1@PAGEOFF]
	orr.16b	v3, v17, v3
Lloh14:
	adrp	x9, lCPI0_4@PAGE
Lloh15:
	ldr	q4, [x9, lCPI0_4@PAGEOFF]
	orr.16b	v4, v17, v4
Lloh16:
	adrp	x9, lCPI0_2@PAGE
Lloh17:
	ldr	q5, [x9, lCPI0_2@PAGEOFF]
	orr.16b	v5, v17, v5
Lloh18:
	adrp	x9, lCPI0_5@PAGE
Lloh19:
	ldr	q6, [x9, lCPI0_5@PAGEOFF]
	orr.16b	v6, v17, v6
Lloh20:
	adrp	x9, lCPI0_6@PAGE
Lloh21:
	ldr	q7, [x9, lCPI0_6@PAGEOFF]
	orr.16b	v7, v17, v7
	ldr	q16, [x13, lCPI0_7@PAGEOFF]
	orr.16b	v16, v17, v16
	ldr	q18, [x11, lCPI0_8@PAGEOFF]
	orr.16b	v17, v17, v18
	mov	w9, #1                          ; =0x1
	dup.2d	v18, x9
	mov	w9, #16                         ; =0x10
	dup.2d	v19, x9
	and	x16, x8, #0xf
	sub	x14, x12, x16
	add	x15, x10, x14
	add	x9, x0, x10
	add	x17, x10, x16
	sub	x17, x17, x8
	movi.2d	v21, #0000000000000000
	movi.2d	v20, #0000000000000000
	movi.2d	v24, #0000000000000000
	movi.2d	v22, #0000000000000000
	movi.2d	v25, #0000000000000000
	movi.2d	v23, #0000000000000000
LBB0_17:                                ; =>This Inner Loop Header: Depth=1
	ushl.2d	v26, v18, v5
	ushl.2d	v27, v18, v7
	ldr	q28, [x9], #16
	ushl.2d	v29, v18, v2
	ushl.2d	v30, v18, v17
	cmlt.16b	v28, v28, #0
	sshll2.8h	v31, v28, #0
	sshll.8h	v28, v28, #0
	sshll2.4s	v8, v31, #0
	sshll.4s	v9, v28, #0
	sshll.2d	v10, v9, #0
	bic.16b	v30, v30, v10
	sshll2.2d	v10, v8, #0
	bic.16b	v29, v29, v10
	ushl.2d	v10, v18, v16
	sshll.4s	v31, v31, #0
	sshll2.4s	v28, v28, #0
	sshll2.2d	v9, v9, #0
	bic.16b	v9, v10, v9
	sshll.2d	v10, v28, #0
	bic.16b	v27, v27, v10
	sshll.2d	v10, v31, #0
	bic.16b	v26, v26, v10
	ushl.2d	v10, v18, v3
	sshll.2d	v8, v8, #0
	bic.16b	v8, v10, v8
	ushl.2d	v10, v18, v6
	sshll2.2d	v28, v28, #0
	bic.16b	v28, v10, v28
	ushl.2d	v10, v18, v4
	sshll2.2d	v31, v31, #0
	bic.16b	v31, v10, v31
	orr.16b	v22, v31, v22
	orr.16b	v20, v28, v20
	orr.16b	v25, v8, v25
	orr.16b	v24, v26, v24
	orr.16b	v21, v27, v21
	orr.16b	v0, v9, v0
	orr.16b	v23, v29, v23
	orr.16b	v1, v30, v1
	add.2d	v7, v7, v19
	add.2d	v16, v16, v19
	add.2d	v17, v17, v19
	add.2d	v6, v6, v19
	add.2d	v5, v5, v19
	add.2d	v4, v4, v19
	add.2d	v3, v3, v19
	add.2d	v2, v2, v19
	adds	x17, x17, #16
	b.ne	LBB0_17
; %bb.18:
	orr.16b	v1, v1, v24
	orr.16b	v2, v21, v25
	orr.16b	v1, v1, v2
	orr.16b	v0, v0, v22
	orr.16b	v2, v20, v23
	orr.16b	v0, v0, v2
	orr.16b	v0, v1, v0
	ext.16b	v1, v0, v0, #8
	orr.8b	v0, v0, v1
	fmov	x9, d0
	cbz	x16, LBB0_26
; %bb.19:
	cmp	x16, #4
	b.hs	LBB0_21
; %bb.20:
	add	x12, x10, x14
	b	LBB0_24
LBB0_21:
	and	x16, x8, #0x3
	sub	x12, x12, x16
	add	x12, x10, x12
	movi.2d	v0, #0000000000000000
	movi.2d	v1, #0000000000000000
	mov.d	v1[0], x9
	dup.2d	v3, x15
	ldr	q2, [x13, lCPI0_7@PAGEOFF]
	add.2d	v2, v3, v2
	ldr	q4, [x11, lCPI0_8@PAGEOFF]
	add.2d	v3, v3, v4
	add	x10, x10, x14
	add	x9, x0, x10
	add	x10, x10, x16
	sub	x10, x10, x8
	mov	w11, #1                         ; =0x1
	dup.2d	v4, x11
	mov	w11, #4                         ; =0x4
	dup.2d	v5, x11
LBB0_22:                                ; =>This Inner Loop Header: Depth=1
	ldr	s6, [x9], #4
	sshll.8h	v6, v6, #0
	ushl.2d	v7, v4, v2
	ushl.2d	v16, v4, v3
	cmlt.4h	v6, v6, #0
	ushll.4s	v6, v6, #0
	ushll2.2d	v17, v6, #0
	shl.2d	v17, v17, #56
	sshr.2d	v17, v17, #56
	ushll.2d	v6, v6, #0
	shl.2d	v6, v6, #56
	sshr.2d	v6, v6, #56
	bic.16b	v6, v16, v6
	bic.16b	v7, v7, v17
	orr.16b	v0, v7, v0
	orr.16b	v1, v6, v1
	add.2d	v2, v2, v5
	add.2d	v3, v3, v5
	adds	x10, x10, #4
	b.ne	LBB0_22
; %bb.23:
	orr.16b	v0, v1, v0
	ext.16b	v1, v0, v0, #8
	orr.8b	v0, v0, v1
	fmov	x9, d0
	cbz	x16, LBB0_26
LBB0_24:
	mov	w10, #1                         ; =0x1
LBB0_25:                                ; =>This Inner Loop Header: Depth=1
	ldrsb	w11, [x0, x12]
	lsl	x13, x10, x12
	cmp	w11, #0
	csel	x11, xzr, x13, lt
	orr	x9, x11, x9
	add	x12, x12, #1
	cmp	x8, x12
	b.ne	LBB0_25
LBB0_26:
	ldp	d9, d8, [sp, #16]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp], #32             ; 16-byte Folded Reload
	str	x9, [x2]
	ret
	.loh AdrpLdr	Lloh8, Lloh9
	.loh AdrpAdrp	Lloh6, Lloh8
	.loh AdrpLdr	Lloh6, Lloh7
	.loh AdrpAdrp	Lloh4, Lloh6
	.loh AdrpLdr	Lloh4, Lloh5
	.loh AdrpAdrp	Lloh2, Lloh4
	.loh AdrpLdr	Lloh2, Lloh3
	.loh AdrpLdr	Lloh0, Lloh1
	.loh AdrpLdr	Lloh20, Lloh21
	.loh AdrpAdrp	Lloh18, Lloh20
	.loh AdrpLdr	Lloh18, Lloh19
	.loh AdrpAdrp	Lloh16, Lloh18
	.loh AdrpLdr	Lloh16, Lloh17
	.loh AdrpAdrp	Lloh14, Lloh16
	.loh AdrpLdr	Lloh14, Lloh15
	.loh AdrpAdrp	Lloh12, Lloh14
	.loh AdrpLdr	Lloh12, Lloh13
	.loh AdrpAdrp	Lloh10, Lloh12
	.loh AdrpLdr	Lloh10, Lloh11
                                        ; -- End function
	.globl	_decode_uvarint64_batch         ; -- Begin function decode_uvarint64_batch
	.p2align	2
_decode_uvarint64_batch:                ; @decode_uvarint64_batch
; %bb.0:
	str	xzr, [x5]
	str	xzr, [x6]
	cmp	x1, #1
	b.lt	LBB1_37
; %bb.1:
	cmp	x3, #1
	b.lt	LBB1_37
; %bb.2:
	cmp	x4, #1
	b.lt	LBB1_37
; %bb.3:
	stp	x22, x21, [sp, #-32]!           ; 16-byte Folded Spill
	stp	x20, x19, [sp, #16]             ; 16-byte Folded Spill
	mov	x8, #0                          ; =0x0
	mov	x9, #0                          ; =0x0
	cmp	x4, x3
	csel	x10, x4, x3, lo
	add	x11, x0, #1
	add	x12, x0, #2
	add	x13, x0, #3
	add	x14, x0, #4
	add	x15, x0, #5
	add	x16, x0, #6
	add	x17, x0, #7
	add	x3, x0, #8
	add	x4, x0, #9
LBB1_4:                                 ; =>This Inner Loop Header: Depth=1
	cmp	x9, x1
	csel	x19, x9, x1, gt
	cmp	x9, x1
	b.ge	LBB1_36
; %bb.5:                                ;   in Loop: Header=BB1_4 Depth=1
	ldrb	w20, [x0, x9]
	and	x7, x20, #0x7f
	tbnz	w20, #7, LBB1_7
; %bb.6:                                ;   in Loop: Header=BB1_4 Depth=1
	mov	w19, #1                         ; =0x1
	b	LBB1_34
LBB1_7:                                 ;   in Loop: Header=BB1_4 Depth=1
	sub	x19, x19, x9
	cmp	x19, #1
	b.eq	LBB1_36
; %bb.8:                                ;   in Loop: Header=BB1_4 Depth=1
	ldrsb	w20, [x11, x9]
	and	w21, w20, #0x7f
	orr	x7, x7, x21, lsl #7
	tbnz	w20, #31, LBB1_10
; %bb.9:                                ;   in Loop: Header=BB1_4 Depth=1
	mov	w19, #2                         ; =0x2
	b	LBB1_34
LBB1_10:                                ;   in Loop: Header=BB1_4 Depth=1
	cmp	x19, #2
	b.eq	LBB1_36
; %bb.11:                               ;   in Loop: Header=BB1_4 Depth=1
	ldrsb	w20, [x12, x9]
	and	w21, w20, #0x7f
	orr	x7, x7, x21, lsl #14
	tbnz	w20, #31, LBB1_13
; %bb.12:                               ;   in Loop: Header=BB1_4 Depth=1
	mov	w19, #3                         ; =0x3
	b	LBB1_34
LBB1_13:                                ;   in Loop: Header=BB1_4 Depth=1
	cmp	x19, #3
	b.eq	LBB1_36
; %bb.14:                               ;   in Loop: Header=BB1_4 Depth=1
	ldrsb	w20, [x13, x9]
	and	w21, w20, #0x7f
	orr	x7, x7, x21, lsl #21
	tbnz	w20, #31, LBB1_16
; %bb.15:                               ;   in Loop: Header=BB1_4 Depth=1
	mov	w19, #4                         ; =0x4
	b	LBB1_34
LBB1_16:                                ;   in Loop: Header=BB1_4 Depth=1
	cmp	x19, #4
	b.eq	LBB1_36
; %bb.17:                               ;   in Loop: Header=BB1_4 Depth=1
	ldrsb	w20, [x14, x9]
	and	w21, w20, #0x7f
	orr	x7, x7, x21, lsl #28
	tbnz	w20, #31, LBB1_19
; %bb.18:                               ;   in Loop: Header=BB1_4 Depth=1
	mov	w19, #5                         ; =0x5
	b	LBB1_34
LBB1_19:                                ;   in Loop: Header=BB1_4 Depth=1
	cmp	x19, #5
	b.eq	LBB1_36
; %bb.20:                               ;   in Loop: Header=BB1_4 Depth=1
	ldrsb	w20, [x15, x9]
	and	w21, w20, #0x7f
	orr	x7, x7, x21, lsl #35
	tbnz	w20, #31, LBB1_22
; %bb.21:                               ;   in Loop: Header=BB1_4 Depth=1
	mov	w19, #6                         ; =0x6
	b	LBB1_34
LBB1_22:                                ;   in Loop: Header=BB1_4 Depth=1
	cmp	x19, #6
	b.eq	LBB1_36
; %bb.23:                               ;   in Loop: Header=BB1_4 Depth=1
	ldrsb	w20, [x16, x9]
	and	w21, w20, #0x7f
	orr	x7, x7, x21, lsl #42
	tbnz	w20, #31, LBB1_25
; %bb.24:                               ;   in Loop: Header=BB1_4 Depth=1
	mov	w19, #7                         ; =0x7
	b	LBB1_34
LBB1_25:                                ;   in Loop: Header=BB1_4 Depth=1
	cmp	x19, #7
	b.eq	LBB1_36
; %bb.26:                               ;   in Loop: Header=BB1_4 Depth=1
	ldrsb	w20, [x17, x9]
	and	w21, w20, #0x7f
	orr	x7, x7, x21, lsl #49
	tbnz	w20, #31, LBB1_28
; %bb.27:                               ;   in Loop: Header=BB1_4 Depth=1
	mov	w19, #8                         ; =0x8
	b	LBB1_34
LBB1_28:                                ;   in Loop: Header=BB1_4 Depth=1
	cmp	x19, #8
	b.eq	LBB1_36
; %bb.29:                               ;   in Loop: Header=BB1_4 Depth=1
	ldrsb	w20, [x3, x9]
	and	w21, w20, #0x7f
	orr	x7, x7, x21, lsl #56
	tbnz	w20, #31, LBB1_31
; %bb.30:                               ;   in Loop: Header=BB1_4 Depth=1
	mov	w19, #9                         ; =0x9
	b	LBB1_34
LBB1_31:                                ;   in Loop: Header=BB1_4 Depth=1
	cmp	x19, #9
	b.eq	LBB1_36
; %bb.32:                               ;   in Loop: Header=BB1_4 Depth=1
	ldrb	w19, [x4, x9]
	cmp	x19, #1
	b.hi	LBB1_36
; %bb.33:                               ;   in Loop: Header=BB1_4 Depth=1
	orr	x7, x7, x19, lsl #63
	mov	w19, #10                        ; =0xa
LBB1_34:                                ;   in Loop: Header=BB1_4 Depth=1
	str	x7, [x2, x8, lsl #3]
	add	x9, x19, x9
	add	x8, x8, #1
	cmp	x8, x10
	b.ge	LBB1_36
; %bb.35:                               ;   in Loop: Header=BB1_4 Depth=1
	cmp	x9, x1
	b.lt	LBB1_4
LBB1_36:
	str	x8, [x5]
	str	x9, [x6]
	ldp	x20, x19, [sp, #16]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp], #32             ; 16-byte Folded Reload
LBB1_37:
	ret
                                        ; -- End function
	.globl	_decode_group_varint32          ; -- Begin function decode_group_varint32
	.p2align	2
_decode_group_varint32:                 ; @decode_group_varint32
; %bb.0:
	str	xzr, [x3]
	cmp	x1, #1
	b.lt	LBB2_19
; %bb.1:
	ldrb	w10, [x0]
	and	x4, x10, #0x3
	ubfx	x16, x10, #2, #2
	add	x15, x16, #1
	ubfx	x13, x10, #4, #2
	add	x12, x13, #1
	lsr	w8, w10, #6
	add	w9, w8, #1
	add	x17, x4, #2
	add	x14, x17, x15
	add	x11, x14, x12
	add	w8, w11, w9
	cmp	x1, x8
	b.lo	LBB2_19
; %bb.2:
	ldrb	w1, [x0, #1]
	cbz	w4, LBB2_6
; %bb.3:
	add	w4, w4, #1
	ldrb	w5, [x0, #2]
	orr	w1, w1, w5, lsl #8
	cmp	w4, #2
	b.eq	LBB2_6
; %bb.4:
	ldrb	w5, [x0, #3]
	orr	w1, w1, w5, lsl #16
	cmp	w4, #3
	b.eq	LBB2_6
; %bb.5:
	ldrb	w4, [x0, #4]
	orr	w1, w1, w4, lsl #24
LBB2_6:
	str	w1, [x2]
	add	x1, x0, x17
	ldrb	w17, [x1]
	cbz	w16, LBB2_10
; %bb.7:
	ldrb	w16, [x1, #1]
	orr	w17, w17, w16, lsl #8
	cmp	w15, #2
	b.eq	LBB2_10
; %bb.8:
	ldrb	w16, [x1, #2]
	orr	w17, w17, w16, lsl #16
	cmp	w15, #3
	b.eq	LBB2_10
; %bb.9:
	ldrb	w15, [x1, #3]
	orr	w17, w17, w15, lsl #24
LBB2_10:
	str	w17, [x2, #4]
	add	x15, x0, x14
	ldrb	w14, [x15]
	cbz	w13, LBB2_14
; %bb.11:
	ldrb	w13, [x15, #1]
	orr	w14, w14, w13, lsl #8
	cmp	w12, #2
	b.eq	LBB2_14
; %bb.12:
	ldrb	w13, [x15, #2]
	orr	w14, w14, w13, lsl #16
	cmp	w12, #3
	b.eq	LBB2_14
; %bb.13:
	ldrb	w12, [x15, #3]
	orr	w14, w14, w12, lsl #24
LBB2_14:
	str	w14, [x2, #8]
	add	x12, x0, x11
	ldrb	w11, [x12]
	cmp	w10, #64
	b.lo	LBB2_18
; %bb.15:
	ldrb	w10, [x12, #1]
	orr	w11, w11, w10, lsl #8
	cmp	w9, #2
	b.eq	LBB2_18
; %bb.16:
	ldrb	w10, [x12, #2]
	orr	w11, w11, w10, lsl #16
	cmp	w9, #3
	b.eq	LBB2_18
; %bb.17:
	ldrb	w9, [x12, #3]
	orr	w11, w11, w9, lsl #24
LBB2_18:
	str	w11, [x2, #12]
	str	x8, [x3]
LBB2_19:
	ret
                                        ; -- End function
	.globl	_decode_group_varint64          ; -- Begin function decode_group_varint64
	.p2align	2
_decode_group_varint64:                 ; @decode_group_varint64
; %bb.0:
	str	xzr, [x3]
	cmp	x1, #2
	b.lt	LBB3_35
; %bb.1:
	ldrb	w8, [x0]
	ldrb	w9, [x0, #1]
	orr	w10, w8, w9, lsl #8
	and	x4, x8, #0x7
	ubfx	x16, x8, #3, #3
	add	x15, x16, #1
	ubfx	w13, w10, #6, #3
	add	w12, w13, #1
	ubfx	w10, w9, #1, #3
	add	w9, w10, #1
	add	x17, x4, #3
	add	x14, x17, x15
	add	x11, x14, x12
	add	w8, w11, w9
	cmp	x1, x8
	b.lo	LBB3_35
; %bb.2:
	ldrb	w1, [x0, #2]
	cbz	w4, LBB3_10
; %bb.3:
	add	w4, w4, #1
	ldrb	w5, [x0, #3]
	orr	x1, x1, x5, lsl #8
	cmp	w4, #2
	b.eq	LBB3_10
; %bb.4:
	ldrb	w5, [x0, #4]
	orr	x1, x1, x5, lsl #16
	cmp	w4, #3
	b.eq	LBB3_10
; %bb.5:
	ldrb	w5, [x0, #5]
	orr	x1, x1, x5, lsl #24
	cmp	w4, #4
	b.eq	LBB3_10
; %bb.6:
	ldrb	w5, [x0, #6]
	orr	x1, x1, x5, lsl #32
	cmp	w4, #5
	b.eq	LBB3_10
; %bb.7:
	ldrb	w5, [x0, #7]
	orr	x1, x1, x5, lsl #40
	cmp	w4, #6
	b.eq	LBB3_10
; %bb.8:
	ldrb	w5, [x0, #8]
	orr	x1, x1, x5, lsl #48
	cmp	w4, #7
	b.eq	LBB3_10
; %bb.9:
	ldrb	w4, [x0, #9]
	orr	x1, x1, x4, lsl #56
LBB3_10:
	str	x1, [x2]
	add	x1, x0, x17
	ldrb	w17, [x1]
	cbz	w16, LBB3_18
; %bb.11:
	ldrb	w16, [x1, #1]
	orr	x17, x17, x16, lsl #8
	cmp	w15, #2
	b.eq	LBB3_18
; %bb.12:
	ldrb	w16, [x1, #2]
	orr	x17, x17, x16, lsl #16
	cmp	w15, #3
	b.eq	LBB3_18
; %bb.13:
	ldrb	w16, [x1, #3]
	orr	x17, x17, x16, lsl #24
	cmp	w15, #4
	b.eq	LBB3_18
; %bb.14:
	ldrb	w16, [x1, #4]
	orr	x17, x17, x16, lsl #32
	cmp	w15, #5
	b.eq	LBB3_18
; %bb.15:
	ldrb	w16, [x1, #5]
	orr	x17, x17, x16, lsl #40
	cmp	w15, #6
	b.eq	LBB3_18
; %bb.16:
	ldrb	w16, [x1, #6]
	orr	x17, x17, x16, lsl #48
	cmp	w15, #7
	b.eq	LBB3_18
; %bb.17:
	ldrb	w15, [x1, #7]
	orr	x17, x17, x15, lsl #56
LBB3_18:
	str	x17, [x2, #8]
	add	x15, x0, x14
	ldrb	w14, [x15]
	cbz	w13, LBB3_26
; %bb.19:
	ldrb	w13, [x15, #1]
	orr	x14, x14, x13, lsl #8
	cmp	w12, #2
	b.eq	LBB3_26
; %bb.20:
	ldrb	w13, [x15, #2]
	orr	x14, x14, x13, lsl #16
	cmp	w12, #3
	b.eq	LBB3_26
; %bb.21:
	ldrb	w13, [x15, #3]
	orr	x14, x14, x13, lsl #24
	cmp	w12, #4
	b.eq	LBB3_26
; %bb.22:
	ldrb	w13, [x15, #4]
	orr	x14, x14, x13, lsl #32
	cmp	w12, #5
	b.eq	LBB3_26
; %bb.23:
	ldrb	w13, [x15, #5]
	orr	x14, x14, x13, lsl #40
	cmp	w12, #6
	b.eq	LBB3_26
; %bb.24:
	ldrb	w13, [x15, #6]
	orr	x14, x14, x13, lsl #48
	cmp	w12, #7
	b.eq	LBB3_26
; %bb.25:
	ldrb	w12, [x15, #7]
	orr	x14, x14, x12, lsl #56
LBB3_26:
	str	x14, [x2, #16]
	add	x12, x0, x11
	ldrb	w11, [x12]
	cbz	w10, LBB3_34
; %bb.27:
	ldrb	w10, [x12, #1]
	orr	x11, x11, x10, lsl #8
	cmp	w9, #2
	b.eq	LBB3_34
; %bb.28:
	ldrb	w10, [x12, #2]
	orr	x11, x11, x10, lsl #16
	cmp	w9, #3
	b.eq	LBB3_34
; %bb.29:
	ldrb	w10, [x12, #3]
	orr	x11, x11, x10, lsl #24
	cmp	w9, #4
	b.eq	LBB3_34
; %bb.30:
	ldrb	w10, [x12, #4]
	orr	x11, x11, x10, lsl #32
	cmp	w9, #5
	b.eq	LBB3_34
; %bb.31:
	ldrb	w10, [x12, #5]
	orr	x11, x11, x10, lsl #40
	cmp	w9, #6
	b.eq	LBB3_34
; %bb.32:
	ldrb	w10, [x12, #6]
	orr	x11, x11, x10, lsl #48
	cmp	w9, #7
	b.eq	LBB3_34
; %bb.33:
	ldrb	w9, [x12, #7]
	orr	x11, x11, x9, lsl #56
LBB3_34:
	str	x11, [x2, #24]
	str	x8, [x3]
LBB3_35:
	ret
                                        ; -- End function
	.globl	_decode_uvarint64               ; -- Begin function decode_uvarint64
	.p2align	2
_decode_uvarint64:                      ; @decode_uvarint64
; %bb.0:
	str	xzr, [x2]
	str	xzr, [x3]
	subs	x11, x1, #1
	b.lt	LBB4_6
; %bb.1:
	mov	x8, #0                          ; =0x0
	mov	x9, #0                          ; =0x0
	mov	x10, #0                         ; =0x0
	mov	w12, #9                         ; =0x9
	cmp	x11, #9
	csel	x11, x11, x12, lo
	add	x11, x11, #1
LBB4_2:                                 ; =>This Inner Loop Header: Depth=1
	ldrb	w12, [x0, x9]
	cmp	x9, #9
	b.ne	LBB4_4
; %bb.3:                                ;   in Loop: Header=BB4_2 Depth=1
	cmp	w12, #1
	b.hi	LBB4_6
LBB4_4:                                 ;   in Loop: Header=BB4_2 Depth=1
	and	x13, x12, #0x7f
	lsl	x13, x13, x10
	orr	x8, x13, x8
	add	x9, x9, #1
	tbz	w12, #7, LBB4_7
; %bb.5:                                ;   in Loop: Header=BB4_2 Depth=1
	add	x10, x10, #7
	cmp	x11, x9
	b.ne	LBB4_2
LBB4_6:
	ret
LBB4_7:
	str	x8, [x2]
	str	x9, [x3]
	ret
                                        ; -- End function
	.globl	_decode_2uvarint64              ; -- Begin function decode_2uvarint64
	.p2align	2
_decode_2uvarint64:                     ; @decode_2uvarint64
; %bb.0:
	str	xzr, [x2]
	str	xzr, [x3]
	str	xzr, [x4]
	cmp	x1, #1
	b.lt	LBB5_6
; %bb.1:
	mov	x8, #0                          ; =0x0
	mov	x10, #0                         ; =0x0
	mov	x9, #0                          ; =0x0
	mov	w11, #10                        ; =0xa
	cmp	x1, #10
	csel	x11, x1, x11, lo
	neg	x11, x11
	b	LBB5_3
LBB5_2:                                 ;   in Loop: Header=BB5_3 Depth=1
	and	x13, x12, #0x7f
	lsl	x13, x13, x10
	orr	x9, x13, x9
	add	x10, x10, #7
	sub	x8, x8, #1
	add	x0, x0, #1
	tbz	w12, #7, LBB5_7
LBB5_3:                                 ; =>This Inner Loop Header: Depth=1
	cmp	x11, x8
	b.eq	LBB5_6
; %bb.4:                                ;   in Loop: Header=BB5_3 Depth=1
	ldrb	w12, [x0]
	cmn	x8, #9
	b.ne	LBB5_2
; %bb.5:                                ;   in Loop: Header=BB5_3 Depth=1
	cmp	w12, #1
	b.ls	LBB5_2
LBB5_6:
	ret
LBB5_7:
	mov	x10, #0                         ; =0x0
	mov	x11, #0                         ; =0x0
	mov	x12, #0                         ; =0x0
	neg	x13, x8
	b	LBB5_9
LBB5_8:                                 ;   in Loop: Header=BB5_9 Depth=1
	and	x15, x14, #0x7f
	lsl	x15, x15, x12
	orr	x11, x15, x11
	add	x12, x12, #7
	add	x10, x10, #1
	tbz	w14, #7, LBB5_12
LBB5_9:                                 ; =>This Inner Loop Header: Depth=1
	add	x14, x13, x10
	cmp	x14, x1
	ccmp	x10, #9, #2, lt
	b.hi	LBB5_6
; %bb.10:                               ;   in Loop: Header=BB5_9 Depth=1
	ldrb	w14, [x0, x10]
	cmp	x10, #9
	b.ne	LBB5_8
; %bb.11:                               ;   in Loop: Header=BB5_9 Depth=1
	cmp	w14, #1
	b.ls	LBB5_8
	b	LBB5_6
LBB5_12:
	str	x9, [x2]
	str	x11, [x3]
	sub	x8, x10, x8
	str	x8, [x4]
	ret
                                        ; -- End function
	.globl	_decode_5uvarint64              ; -- Begin function decode_5uvarint64
	.p2align	2
_decode_5uvarint64:                     ; @decode_5uvarint64
; %bb.0:
	movi.2d	v0, #0000000000000000
	stp	q0, q0, [x2]
	str	xzr, [x2, #32]
	str	xzr, [x3]
	cmp	x1, #1
	b.lt	LBB6_106
; %bb.1:
	ldrb	w9, [x0]
	and	x8, x9, #0x7f
	tbnz	w9, #7, LBB6_14
; %bb.2:
	mov	w9, #1                          ; =0x1
LBB6_3:
	str	x8, [x2]
	cmp	x9, x1
	b.hs	LBB6_104
; %bb.4:
	add	x8, x9, #1
	ldrb	w11, [x0, x9]
	and	x10, x11, #0x7f
	tbnz	w11, #7, LBB6_17
LBB6_5:
	str	x10, [x2, #8]
	cmp	x8, x1
	b.hs	LBB6_104
; %bb.6:
	add	x9, x8, #1
	ldrb	w11, [x0, x8]
	and	x10, x11, #0x7f
	tbnz	w11, #7, LBB6_23
LBB6_7:
	str	x10, [x2, #16]
	cmp	x9, x1
	b.hs	LBB6_104
; %bb.8:
	add	x8, x9, #1
	ldrb	w11, [x0, x9]
	and	x10, x11, #0x7f
	tbnz	w11, #7, LBB6_43
LBB6_9:
	str	x10, [x2, #24]
	cmp	x8, x1
	b.hs	LBB6_104
; %bb.10:
	add	x10, x8, #1
	ldrb	w11, [x0, x8]
	and	x9, x11, #0x7f
	tbz	w11, #7, LBB6_105
; %bb.11:
	cmp	x1, x10
	b.eq	LBB6_104
; %bb.12:
	add	x11, x8, #2
	ldrsb	w10, [x0, x10]
	and	w12, w10, #0x7f
	orr	x9, x9, x12, lsl #7
	tbnz	w10, #31, LBB6_86
LBB6_13:
	mov	x10, x11
	b	LBB6_105
LBB6_14:
	cmp	x1, #1
	b.eq	LBB6_104
; %bb.15:
	ldrsb	w9, [x0, #1]
	and	w10, w9, #0x7f
	orr	x8, x8, x10, lsl #7
	tbnz	w9, #31, LBB6_20
; %bb.16:
	mov	w9, #2                          ; =0x2
	b	LBB6_3
LBB6_17:
	cmp	x1, x8
	b.eq	LBB6_104
; %bb.18:
	add	x11, x9, #2
	ldrsb	w8, [x0, x8]
	and	w12, w8, #0x7f
	orr	x10, x10, x12, lsl #7
	tbnz	w8, #31, LBB6_26
LBB6_19:
	mov	x8, x11
	b	LBB6_5
LBB6_20:
	cmp	x1, #2
	b.eq	LBB6_104
; %bb.21:
	ldrsb	w9, [x0, #2]
	and	w10, w9, #0x7f
	orr	x8, x8, x10, lsl #14
	tbnz	w9, #31, LBB6_46
; %bb.22:
	mov	w9, #3                          ; =0x3
	b	LBB6_3
LBB6_23:
	cmp	x1, x9
	b.eq	LBB6_104
; %bb.24:
	add	x11, x8, #2
	ldrsb	w9, [x0, x9]
	and	w12, w9, #0x7f
	orr	x10, x10, x12, lsl #7
	tbnz	w9, #31, LBB6_49
LBB6_25:
	mov	x9, x11
	b	LBB6_7
LBB6_26:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.27:
	add	x8, x9, #3
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x10, x10, x12, lsl #14
	tbz	w11, #31, LBB6_5
; %bb.28:
	cmp	x1, x8
	b.eq	LBB6_104
; %bb.29:
	add	x11, x9, #4
	ldrsb	w8, [x0, x8]
	and	w12, w8, #0x7f
	orr	x10, x10, x12, lsl #21
	tbz	w8, #31, LBB6_19
; %bb.30:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.31:
	add	x8, x9, #5
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x10, x10, x12, lsl #28
	tbz	w11, #31, LBB6_5
; %bb.32:
	cmp	x1, x8
	b.eq	LBB6_104
; %bb.33:
	add	x11, x9, #6
	ldrsb	w8, [x0, x8]
	and	w12, w8, #0x7f
	orr	x10, x10, x12, lsl #35
	tbz	w8, #31, LBB6_19
; %bb.34:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.35:
	add	x8, x9, #7
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x10, x10, x12, lsl #42
	tbz	w11, #31, LBB6_5
; %bb.36:
	cmp	x1, x8
	b.eq	LBB6_104
; %bb.37:
	add	x11, x9, #8
	ldrsb	w8, [x0, x8]
	and	w12, w8, #0x7f
	orr	x10, x10, x12, lsl #49
	tbz	w8, #31, LBB6_19
; %bb.38:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.39:
	add	x8, x9, #9
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x10, x10, x12, lsl #56
	tbz	w11, #31, LBB6_5
; %bb.40:
	cmp	x1, x8
	b.eq	LBB6_104
; %bb.41:
	ldrb	w11, [x0, x8]
	cmp	x11, #1
	b.hi	LBB6_104
; %bb.42:
	add	x8, x9, #10
	orr	x10, x10, x11, lsl #63
	b	LBB6_5
LBB6_43:
	cmp	x1, x8
	b.eq	LBB6_104
; %bb.44:
	add	x11, x9, #2
	ldrsb	w8, [x0, x8]
	and	w12, w8, #0x7f
	orr	x10, x10, x12, lsl #7
	tbnz	w8, #31, LBB6_66
LBB6_45:
	mov	x8, x11
	b	LBB6_9
LBB6_46:
	cmp	x1, #3
	b.eq	LBB6_104
; %bb.47:
	ldrsb	w9, [x0, #3]
	and	w10, w9, #0x7f
	orr	x8, x8, x10, lsl #21
	tbnz	w9, #31, LBB6_83
; %bb.48:
	mov	w9, #4                          ; =0x4
	b	LBB6_3
LBB6_49:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.50:
	add	x9, x8, #3
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x10, x10, x12, lsl #14
	tbz	w11, #31, LBB6_7
; %bb.51:
	cmp	x1, x9
	b.eq	LBB6_104
; %bb.52:
	add	x11, x8, #4
	ldrsb	w9, [x0, x9]
	and	w12, w9, #0x7f
	orr	x10, x10, x12, lsl #21
	tbz	w9, #31, LBB6_25
; %bb.53:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.54:
	add	x9, x8, #5
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x10, x10, x12, lsl #28
	tbz	w11, #31, LBB6_7
; %bb.55:
	cmp	x1, x9
	b.eq	LBB6_104
; %bb.56:
	add	x11, x8, #6
	ldrsb	w9, [x0, x9]
	and	w12, w9, #0x7f
	orr	x10, x10, x12, lsl #35
	tbz	w9, #31, LBB6_25
; %bb.57:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.58:
	add	x9, x8, #7
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x10, x10, x12, lsl #42
	tbz	w11, #31, LBB6_7
; %bb.59:
	cmp	x1, x9
	b.eq	LBB6_104
; %bb.60:
	add	x11, x8, #8
	ldrsb	w9, [x0, x9]
	and	w12, w9, #0x7f
	orr	x10, x10, x12, lsl #49
	tbz	w9, #31, LBB6_25
; %bb.61:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.62:
	add	x9, x8, #9
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x10, x10, x12, lsl #56
	tbz	w11, #31, LBB6_7
; %bb.63:
	cmp	x1, x9
	b.eq	LBB6_104
; %bb.64:
	ldrb	w11, [x0, x9]
	cmp	x11, #1
	b.hi	LBB6_104
; %bb.65:
	add	x9, x8, #10
	orr	x10, x10, x11, lsl #63
	b	LBB6_7
LBB6_66:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.67:
	add	x8, x9, #3
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x10, x10, x12, lsl #14
	tbz	w11, #31, LBB6_9
; %bb.68:
	cmp	x1, x8
	b.eq	LBB6_104
; %bb.69:
	add	x11, x9, #4
	ldrsb	w8, [x0, x8]
	and	w12, w8, #0x7f
	orr	x10, x10, x12, lsl #21
	tbz	w8, #31, LBB6_45
; %bb.70:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.71:
	add	x8, x9, #5
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x10, x10, x12, lsl #28
	tbz	w11, #31, LBB6_9
; %bb.72:
	cmp	x1, x8
	b.eq	LBB6_104
; %bb.73:
	add	x11, x9, #6
	ldrsb	w8, [x0, x8]
	and	w12, w8, #0x7f
	orr	x10, x10, x12, lsl #35
	tbz	w8, #31, LBB6_45
; %bb.74:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.75:
	add	x8, x9, #7
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x10, x10, x12, lsl #42
	tbz	w11, #31, LBB6_9
; %bb.76:
	cmp	x1, x8
	b.eq	LBB6_104
; %bb.77:
	add	x11, x9, #8
	ldrsb	w8, [x0, x8]
	and	w12, w8, #0x7f
	orr	x10, x10, x12, lsl #49
	tbz	w8, #31, LBB6_45
; %bb.78:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.79:
	add	x8, x9, #9
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x10, x10, x12, lsl #56
	tbz	w11, #31, LBB6_9
; %bb.80:
	cmp	x1, x8
	b.eq	LBB6_104
; %bb.81:
	ldrb	w11, [x0, x8]
	cmp	x11, #1
	b.hi	LBB6_104
; %bb.82:
	add	x8, x9, #10
	orr	x10, x10, x11, lsl #63
	b	LBB6_9
LBB6_83:
	cmp	x1, #4
	b.eq	LBB6_104
; %bb.84:
	ldrsb	w9, [x0, #4]
	and	w10, w9, #0x7f
	orr	x8, x8, x10, lsl #28
	tbnz	w9, #31, LBB6_103
; %bb.85:
	mov	w9, #5                          ; =0x5
	b	LBB6_3
LBB6_86:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.87:
	add	x10, x8, #3
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x9, x9, x12, lsl #14
	tbz	w11, #31, LBB6_105
; %bb.88:
	cmp	x1, x10
	b.eq	LBB6_104
; %bb.89:
	add	x11, x8, #4
	ldrsb	w10, [x0, x10]
	and	w12, w10, #0x7f
	orr	x9, x9, x12, lsl #21
	tbz	w10, #31, LBB6_13
; %bb.90:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.91:
	add	x10, x8, #5
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x9, x9, x12, lsl #28
	tbz	w11, #31, LBB6_105
; %bb.92:
	cmp	x1, x10
	b.eq	LBB6_104
; %bb.93:
	add	x11, x8, #6
	ldrsb	w10, [x0, x10]
	and	w12, w10, #0x7f
	orr	x9, x9, x12, lsl #35
	tbz	w10, #31, LBB6_13
; %bb.94:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.95:
	add	x10, x8, #7
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x9, x9, x12, lsl #42
	tbz	w11, #31, LBB6_105
; %bb.96:
	cmp	x1, x10
	b.eq	LBB6_104
; %bb.97:
	add	x11, x8, #8
	ldrsb	w10, [x0, x10]
	and	w12, w10, #0x7f
	orr	x9, x9, x12, lsl #49
	tbz	w10, #31, LBB6_13
; %bb.98:
	cmp	x1, x11
	b.eq	LBB6_104
; %bb.99:
	add	x10, x8, #9
	ldrsb	w11, [x0, x11]
	and	w12, w11, #0x7f
	orr	x9, x9, x12, lsl #56
	tbz	w11, #31, LBB6_105
; %bb.100:
	cmp	x1, x10
	b.eq	LBB6_104
; %bb.101:
	ldrb	w11, [x0, x10]
	cmp	x11, #1
	b.hi	LBB6_104
; %bb.102:
	add	x10, x8, #10
	orr	x9, x9, x11, lsl #63
	b	LBB6_105
LBB6_103:
	cmp	x1, #5
	b.ne	LBB6_107
LBB6_104:
	mov	x9, #0                          ; =0x0
	mov	x10, #0                         ; =0x0
	stp	q0, q0, [x2]
LBB6_105:
	str	x9, [x2, #32]
	str	x10, [x3]
LBB6_106:
	ret
LBB6_107:
	ldrsb	w9, [x0, #5]
	and	w10, w9, #0x7f
	orr	x8, x8, x10, lsl #35
	tbnz	w9, #31, LBB6_109
; %bb.108:
	mov	w9, #6                          ; =0x6
	b	LBB6_3
LBB6_109:
	cmp	x1, #6
	b.eq	LBB6_104
; %bb.110:
	ldrsb	w9, [x0, #6]
	and	w10, w9, #0x7f
	orr	x8, x8, x10, lsl #42
	tbnz	w9, #31, LBB6_112
; %bb.111:
	mov	w9, #7                          ; =0x7
	b	LBB6_3
LBB6_112:
	cmp	x1, #7
	b.eq	LBB6_104
; %bb.113:
	ldrsb	w9, [x0, #7]
	and	w10, w9, #0x7f
	orr	x8, x8, x10, lsl #49
	tbnz	w9, #31, LBB6_115
; %bb.114:
	mov	w9, #8                          ; =0x8
	b	LBB6_3
LBB6_115:
	cmp	x1, #8
	b.eq	LBB6_104
; %bb.116:
	ldrsb	w9, [x0, #8]
	and	w10, w9, #0x7f
	orr	x8, x8, x10, lsl #56
	tbnz	w9, #31, LBB6_118
; %bb.117:
	mov	w9, #9                          ; =0x9
	b	LBB6_3
LBB6_118:
	cmp	x1, #9
	b.eq	LBB6_104
; %bb.119:
	ldrb	w9, [x0, #9]
	cmp	x9, #1
	b.hi	LBB6_104
; %bb.120:
	orr	x8, x8, x9, lsl #63
	mov	w9, #10                         ; =0xa
	b	LBB6_3
                                        ; -- End function
.subsections_via_symbols
