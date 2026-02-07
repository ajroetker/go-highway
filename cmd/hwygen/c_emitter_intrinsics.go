package main

import "fmt"

// ---------------------------------------------------------------------------
// Generic intrinsic helpers for CEmitter.
// These methods abstract NEON and SVE intrinsic formatting, allowing math
// computation code to be written once for all targets.
// ---------------------------------------------------------------------------

// intSuffix returns the integer type suffix corresponding to the float element type.
// f32 → "s32", f64 → "s64".
func (e *CEmitter) intSuffix() string {
	if e.elemType == "float32" {
		return "s32"
	}
	return "s64"
}

// maskType returns the C type for comparison result masks.
func (e *CEmitter) maskType() string {
	if e.isSVE() {
		return "svbool_t"
	}
	if e.elemType == "float32" {
		return "uint32x4_t"
	}
	return "uint64x2_t"
}

// intVecType returns the C type for integer vectors.
func (e *CEmitter) intVecType() string {
	if e.isSVE() {
		if e.elemType == "float32" {
			return "svint32_t"
		}
		return "svint64_t"
	}
	if e.elemType == "float32" {
		return "int32x4_t"
	}
	return "int64x2_t"
}

// expBias returns the exponent bias for the element type: "127" (f32) or "1023" (f64).
func (e *CEmitter) expBias() string {
	if e.elemType == "float32" {
		return "127"
	}
	return "1023"
}

// expShift returns the exponent shift amount: "23" (f32) or "52" (f64).
func (e *CEmitter) expShift() string {
	if e.elemType == "float32" {
		return "23"
	}
	return "52"
}

// volatilePrefix returns "volatile " for NEON f64 constants (GOAT workaround),
// or "" for all other targets.
func (e *CEmitter) volatilePrefix() string {
	if !e.isSVE() && e.elemType == "float64" {
		return "volatile "
	}
	return ""
}

// ---------------------------------------------------------------------------
// Binary float operations (predicated for SVE).
// ---------------------------------------------------------------------------

func (e *CEmitter) fmtMul(a, b string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svmul_%s_x(pg, %s, %s)", s, a, b)
	}
	return fmt.Sprintf("vmulq_%s(%s, %s)", s, a, b)
}

func (e *CEmitter) fmtAdd(a, b string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svadd_%s_x(pg, %s, %s)", s, a, b)
	}
	return fmt.Sprintf("vaddq_%s(%s, %s)", s, a, b)
}

func (e *CEmitter) fmtDiv(a, b string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svdiv_%s_x(pg, %s, %s)", s, a, b)
	}
	return fmt.Sprintf("vdivq_%s(%s, %s)", s, a, b)
}

func (e *CEmitter) fmtMin(a, b string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svmin_%s_x(pg, %s, %s)", s, a, b)
	}
	return fmt.Sprintf("vminq_%s(%s, %s)", s, a, b)
}

func (e *CEmitter) fmtMax(a, b string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svmax_%s_x(pg, %s, %s)", s, a, b)
	}
	return fmt.Sprintf("vmaxq_%s(%s, %s)", s, a, b)
}

// fmtFma returns acc + a*b.
//
//	NEON: vfmaq_f32(acc, a, b)
//	SVE:  svmla_f32_x(pg, acc, a, b)
func (e *CEmitter) fmtFma(acc, a, b string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svmla_%s_x(pg, %s, %s, %s)", s, acc, a, b)
	}
	return fmt.Sprintf("vfmaq_%s(%s, %s, %s)", s, acc, a, b)
}

// fmtFms returns a - b*c.
//
//	NEON: vfmsq_f32(a, b, c)
//	SVE:  svmls_f32_x(pg, a, b, c)
func (e *CEmitter) fmtFms(a, b, c string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svmls_%s_x(pg, %s, %s, %s)", s, a, b, c)
	}
	return fmt.Sprintf("vfmsq_%s(%s, %s, %s)", s, a, b, c)
}

// ---------------------------------------------------------------------------
// Unary float operations (predicated for SVE).
// ---------------------------------------------------------------------------

func (e *CEmitter) fmtNeg(x string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svneg_%s_x(pg, %s)", s, x)
	}
	return fmt.Sprintf("vnegq_%s(%s)", s, x)
}

func (e *CEmitter) fmtAbs(x string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svabs_%s_x(pg, %s)", s, x)
	}
	return fmt.Sprintf("vabsq_%s(%s)", s, x)
}

func (e *CEmitter) fmtRound(x string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svrintn_%s_x(pg, %s)", s, x)
	}
	return fmt.Sprintf("vrndnq_%s(%s)", s, x)
}

// ---------------------------------------------------------------------------
// Comparison operations (predicated for SVE, return mask type).
// ---------------------------------------------------------------------------

func (e *CEmitter) fmtCmpGt(a, b string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svcmpgt_%s(pg, %s, %s)", s, a, b)
	}
	return fmt.Sprintf("vcgtq_%s(%s, %s)", s, a, b)
}

func (e *CEmitter) fmtCmpLt(a, b string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svcmplt_%s(pg, %s, %s)", s, a, b)
	}
	return fmt.Sprintf("vcltq_%s(%s, %s)", s, a, b)
}

// ---------------------------------------------------------------------------
// Conditional selection.
// ---------------------------------------------------------------------------

// fmtSel returns a conditional select: where mask is true, pick a, else b.
//
//	NEON: vbslq_f32(mask, a, b)
//	SVE:  svsel_f32(mask, a, b)
func (e *CEmitter) fmtSel(mask, a, b string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svsel_%s(%s, %s, %s)", s, mask, a, b)
	}
	return fmt.Sprintf("vbslq_%s(%s, %s, %s)", s, mask, a, b)
}

// ---------------------------------------------------------------------------
// Integer vector operations (predicated for SVE).
// ---------------------------------------------------------------------------

// fmtCvtFloatToInt converts a float vector to integer.
//
//	NEON f32: vcvtnq_s32_f32(x) (round to nearest)
//	NEON f64: vcvtq_s64_f64(x)  (truncation; input pre-rounded)
//	SVE:      svcvt_sXX_fXX_x(pg, x)
func (e *CEmitter) fmtCvtFloatToInt(x string) string {
	fs := e.typeSuffix()
	is := e.intSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svcvt_%s_%s_x(pg, %s)", is, fs, x)
	}
	if e.elemType == "float32" {
		return fmt.Sprintf("vcvtnq_%s_%s(%s)", is, fs, x)
	}
	return fmt.Sprintf("vcvtq_%s_%s(%s)", is, fs, x)
}

// fmtIntShl returns integer shift left by immediate.
//
//	NEON: vshlq_n_s32(x, n)
//	SVE:  svlsl_n_s32_x(pg, x, n)
func (e *CEmitter) fmtIntShl(x, n string) string {
	is := e.intSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svlsl_n_%s_x(pg, %s, %s)", is, x, n)
	}
	return fmt.Sprintf("vshlq_n_%s(%s, %s)", is, x, n)
}

// fmtIntAdd returns integer vector addition.
//
//	NEON: vaddq_s32(a, b)
//	SVE:  svadd_s32_x(pg, a, b)
func (e *CEmitter) fmtIntAdd(a, b string) string {
	is := e.intSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svadd_%s_x(pg, %s, %s)", is, a, b)
	}
	return fmt.Sprintf("vaddq_%s(%s, %s)", is, a, b)
}

// ---------------------------------------------------------------------------
// Reinterpret (never predicated on any target).
// ---------------------------------------------------------------------------

// fmtReinterpretFloatFromInt reinterprets an integer vector as float.
//
//	NEON: vreinterpretq_f32_s32(x)
//	SVE:  svreinterpret_f32_s32(x)
func (e *CEmitter) fmtReinterpretFloatFromInt(x string) string {
	fs := e.typeSuffix()
	is := e.intSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svreinterpret_%s_%s(%s)", fs, is, x)
	}
	return fmt.Sprintf("vreinterpretq_%s_%s(%s)", fs, is, x)
}

// ---------------------------------------------------------------------------
// Constant emission helpers.
// ---------------------------------------------------------------------------

// fmtConstHex creates a float vector constant from a hex integer bit pattern.
//
//	NEON: vreinterpretq_f32_s32(vdupq_n_s32(0xHEX))
//	SVE:  svreinterpret_f32_s32(svdup_s32(0xHEX))
func (e *CEmitter) fmtConstHex(hexVal string) string {
	fs := e.typeSuffix()
	is := e.intSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svreinterpret_%s_%s(svdup_%s(%s))", fs, is, is, hexVal)
	}
	return fmt.Sprintf("vreinterpretq_%s_%s(vdupq_n_%s(%s))", fs, is, is, hexVal)
}

// fmtConstFloat creates a float vector constant from a literal value.
//
//	NEON: vdupq_n_f32(val)
//	SVE:  svdup_f32(val)
func (e *CEmitter) fmtConstFloat(val string) string {
	s := e.typeSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svdup_%s(%s)", s, val)
	}
	return fmt.Sprintf("vdupq_n_%s(%s)", s, val)
}

// fmtConstInt creates an integer vector constant.
//
//	NEON: vdupq_n_s32(val)
//	SVE:  svdup_s32(val)
func (e *CEmitter) fmtConstInt(val string) string {
	is := e.intSuffix()
	if e.isSVE() {
		return fmt.Sprintf("svdup_%s(%s)", is, val)
	}
	return fmt.Sprintf("vdupq_n_%s(%s)", is, val)
}
