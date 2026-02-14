// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

//go:build cgo && darwin

// NOTE: This file is named "z_accelerate_darwin.go" (starting with 'z')
// to ensure its init() runs AFTER the generated dispatch files and any
// z_*_arm64.go NEON overrides. Go executes init() functions in lexicographic
// filename order within a package.
//
// On darwin+cgo, we override float32/float64 transform dispatch pointers
// with Apple Accelerate vForce implementations. Float16/BFloat16 stay on
// their existing (NEON or fallback) implementations.

package algo

/*
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
#include <math.h>

// erff/erf loop helpers — Apple's math.h erff is vectorized by clang.
static void vec_erff(const float *in, float *out, int n) {
	for (int i = 0; i < n; i++) {
		out[i] = erff(in[i]);
	}
}
static void vec_erf(const double *in, double *out, int n) {
	for (int i = 0; i < n; i++) {
		out[i] = erf(in[i]);
	}
}

// sigmoid helpers: sigmoid(x) = 1/(1+exp(-x))
static void vec_sigmoidf(const float *in, float *out, int n) {
	// negate
	int ni = n;
	for (int i = 0; i < n; i++) out[i] = -in[i];
	// exp(-x)
	vvexpf(out, out, &ni);
	// 1 + exp(-x), then 1/result
	float one = 1.0f;
	vDSP_vsadd(out, 1, &one, out, 1, (vDSP_Length)n);
	vvrecf(out, out, &ni);
}
static void vec_sigmoid(const double *in, double *out, int n) {
	int ni = n;
	for (int i = 0; i < n; i++) out[i] = -in[i];
	vvexp(out, out, &ni);
	double one = 1.0;
	vDSP_vsaddD(out, 1, &one, out, 1, (vDSP_Length)n);
	vvrec(out, out, &ni);
}
*/
import "C"
import "unsafe"

func init() {
	// Override float32 transforms with vForce
	ExpTransformFloat32 = accelerateExpTransformFloat32
	ExpTransformFloat64 = accelerateExpTransformFloat64
	LogTransformFloat32 = accelerateLogTransformFloat32
	LogTransformFloat64 = accelerateLogTransformFloat64
	SinTransformFloat32 = accelerateSinTransformFloat32
	SinTransformFloat64 = accelerateSinTransformFloat64
	CosTransformFloat32 = accelerateCosTransformFloat32
	CosTransformFloat64 = accelerateCosTransformFloat64
	TanhTransformFloat32 = accelerateTanhTransformFloat32
	TanhTransformFloat64 = accelerateTanhTransformFloat64
	SigmoidTransformFloat32 = accelerateSigmoidTransformFloat32
	SigmoidTransformFloat64 = accelerateSigmoidTransformFloat64
	ErfTransformFloat32 = accelerateErfTransformFloat32
	ErfTransformFloat64 = accelerateErfTransformFloat64
}

func accelerateExpTransformFloat32(in, out []float32) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	ni := C.int(n)
	C.vvexpf((*C.float)(unsafe.Pointer(&out[0])),
		(*C.float)(unsafe.Pointer(&in[0])), &ni)
}

func accelerateExpTransformFloat64(in, out []float64) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	ni := C.int(n)
	C.vvexp((*C.double)(unsafe.Pointer(&out[0])),
		(*C.double)(unsafe.Pointer(&in[0])), &ni)
}

func accelerateLogTransformFloat32(in, out []float32) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	ni := C.int(n)
	C.vvlogf((*C.float)(unsafe.Pointer(&out[0])),
		(*C.float)(unsafe.Pointer(&in[0])), &ni)
}

func accelerateLogTransformFloat64(in, out []float64) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	ni := C.int(n)
	C.vvlog((*C.double)(unsafe.Pointer(&out[0])),
		(*C.double)(unsafe.Pointer(&in[0])), &ni)
}

func accelerateSinTransformFloat32(in, out []float32) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	ni := C.int(n)
	C.vvsinf((*C.float)(unsafe.Pointer(&out[0])),
		(*C.float)(unsafe.Pointer(&in[0])), &ni)
}

func accelerateSinTransformFloat64(in, out []float64) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	ni := C.int(n)
	C.vvsin((*C.double)(unsafe.Pointer(&out[0])),
		(*C.double)(unsafe.Pointer(&in[0])), &ni)
}

func accelerateCosTransformFloat32(in, out []float32) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	ni := C.int(n)
	C.vvcosf((*C.float)(unsafe.Pointer(&out[0])),
		(*C.float)(unsafe.Pointer(&in[0])), &ni)
}

func accelerateCosTransformFloat64(in, out []float64) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	ni := C.int(n)
	C.vvcos((*C.double)(unsafe.Pointer(&out[0])),
		(*C.double)(unsafe.Pointer(&in[0])), &ni)
}

func accelerateTanhTransformFloat32(in, out []float32) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	ni := C.int(n)
	C.vvtanhf((*C.float)(unsafe.Pointer(&out[0])),
		(*C.float)(unsafe.Pointer(&in[0])), &ni)
}

func accelerateTanhTransformFloat64(in, out []float64) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	ni := C.int(n)
	C.vvtanh((*C.double)(unsafe.Pointer(&out[0])),
		(*C.double)(unsafe.Pointer(&in[0])), &ni)
}

func accelerateSigmoidTransformFloat32(in, out []float32) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	C.vec_sigmoidf((*C.float)(unsafe.Pointer(&in[0])),
		(*C.float)(unsafe.Pointer(&out[0])), C.int(n))
}

func accelerateSigmoidTransformFloat64(in, out []float64) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	C.vec_sigmoid((*C.double)(unsafe.Pointer(&in[0])),
		(*C.double)(unsafe.Pointer(&out[0])), C.int(n))
}

func accelerateErfTransformFloat32(in, out []float32) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	C.vec_erff((*C.float)(unsafe.Pointer(&in[0])),
		(*C.float)(unsafe.Pointer(&out[0])), C.int(n))
}

func accelerateErfTransformFloat64(in, out []float64) {
	n := min(len(in), len(out))
	if n == 0 {
		return
	}
	C.vec_erf((*C.double)(unsafe.Pointer(&in[0])),
		(*C.double)(unsafe.Pointer(&out[0])), C.int(n))
}
