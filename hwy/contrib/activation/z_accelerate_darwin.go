// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

//go:build cgo && darwin

// NOTE: This file is named "z_accelerate_darwin.go" (starting with 'z')
// to ensure its init() runs AFTER the generated dispatch files and any
// z_activation_arm64.go NEON overrides. Go executes init() functions in
// lexicographic filename order within a package.
//
// On darwin+cgo, we override float32/float64 activation dispatch pointers
// with Apple Accelerate implementations. Float16/BFloat16 stay on their
// existing (NEON or fallback) implementations.

package activation

/*
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
#include <math.h>

// GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
// Apple's clang auto-vectorizes erff with NEON intrinsics.
static void gelu_f32(const float *in, float *out, int n) {
	const float inv_sqrt2 = 0.7071067811865476f;
	for (int i = 0; i < n; i++) {
		float x = in[i];
		out[i] = x * 0.5f * (1.0f + erff(x * inv_sqrt2));
	}
}

static void gelu_f64(const double *in, double *out, int n) {
	const double inv_sqrt2 = 0.7071067811865476;
	for (int i = 0; i < n; i++) {
		double x = in[i];
		out[i] = x * 0.5 * (1.0 + erf(x * inv_sqrt2));
	}
}

// GELUApprox: x * sigmoid(1.702 * x)
static void gelu_approx_f32(const float *in, float *out, int n) {
	const float alpha = 1.702f;
	// out = -1.702 * x
	float negAlpha = -alpha;
	vDSP_vsmul(in, 1, &negAlpha, out, 1, (vDSP_Length)n);
	// out = exp(-1.702 * x)
	vvexpf(out, out, &n);
	// out = 1 + exp(-1.702*x)
	float one = 1.0f;
	vDSP_vsadd(out, 1, &one, out, 1, (vDSP_Length)n);
	// out = 1 / (1 + exp(-1.702*x)) = sigmoid(1.702*x)
	vvrecf(out, out, &n);
	// out = x * sigmoid(1.702*x)
	vDSP_vmul(in, 1, out, 1, out, 1, (vDSP_Length)n);
}

static void gelu_approx_f64(const double *in, double *out, int n) {
	const double alpha = 1.702;
	double negAlpha = -alpha;
	vDSP_vsmulD(in, 1, &negAlpha, out, 1, (vDSP_Length)n);
	vvexp(out, out, &n);
	double one = 1.0;
	vDSP_vsaddD(out, 1, &one, out, 1, (vDSP_Length)n);
	vvrec(out, out, &n);
	vDSP_vmulD(in, 1, out, 1, out, 1, (vDSP_Length)n);
}

// SiLU: x * sigmoid(x)
static void silu_f32(const float *in, float *out, int n) {
	// out = -x
	float negOne = -1.0f;
	vDSP_vsmul(in, 1, &negOne, out, 1, (vDSP_Length)n);
	// out = exp(-x)
	vvexpf(out, out, &n);
	// out = 1 + exp(-x)
	float one = 1.0f;
	vDSP_vsadd(out, 1, &one, out, 1, (vDSP_Length)n);
	// out = 1/(1+exp(-x)) = sigmoid(x)
	vvrecf(out, out, &n);
	// out = x * sigmoid(x)
	vDSP_vmul(in, 1, out, 1, out, 1, (vDSP_Length)n);
}

static void silu_f64(const double *in, double *out, int n) {
	double negOne = -1.0;
	vDSP_vsmulD(in, 1, &negOne, out, 1, (vDSP_Length)n);
	vvexp(out, out, &n);
	double one = 1.0;
	vDSP_vsaddD(out, 1, &one, out, 1, (vDSP_Length)n);
	vvrec(out, out, &n);
	vDSP_vmulD(in, 1, out, 1, out, 1, (vDSP_Length)n);
}
*/
import "C"
import "unsafe"

func init() {
	GELUFloat32 = accelerateGELUFloat32
	GELUFloat64 = accelerateGELUFloat64
	GELUApproxFloat32 = accelerateGELUApproxFloat32
	GELUApproxFloat64 = accelerateGELUApproxFloat64
	TanhFloat32 = accelerateTanhFloat32
	TanhFloat64 = accelerateTanhFloat64
	SiLUFloat32 = accelerateSiLUFloat32
	SiLUFloat64 = accelerateSiLUFloat64
}

func accelerateGELUFloat32(input, output []float32) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	C.gelu_f32((*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&output[0])), C.int(n))
}

func accelerateGELUFloat64(input, output []float64) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	C.gelu_f64((*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&output[0])), C.int(n))
}

func accelerateGELUApproxFloat32(input, output []float32) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	C.gelu_approx_f32((*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&output[0])), C.int(n))
}

func accelerateGELUApproxFloat64(input, output []float64) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	C.gelu_approx_f64((*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&output[0])), C.int(n))
}

func accelerateTanhFloat32(input, output []float32) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	ni := C.int(n)
	C.vvtanhf((*C.float)(unsafe.Pointer(&output[0])),
		(*C.float)(unsafe.Pointer(&input[0])), &ni)
}

func accelerateTanhFloat64(input, output []float64) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	ni := C.int(n)
	C.vvtanh((*C.double)(unsafe.Pointer(&output[0])),
		(*C.double)(unsafe.Pointer(&input[0])), &ni)
}

func accelerateSiLUFloat32(input, output []float32) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	C.silu_f32((*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&output[0])), C.int(n))
}

func accelerateSiLUFloat64(input, output []float64) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	C.silu_f64((*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&output[0])), C.int(n))
}
