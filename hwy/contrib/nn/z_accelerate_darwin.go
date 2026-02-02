// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

//go:build cgo && darwin

// NOTE: This file is named "z_accelerate_darwin.go" (starting with 'z')
// to ensure its init() runs AFTER the generated dispatch files.
// Go executes init() functions in lexicographic filename order within a package.
//
// On darwin+cgo, we override float32/float64 softmax dispatch pointers with
// Apple Accelerate vDSP + vForce implementations. These fuse max/subtract/exp/sum/divide
// into a single CGo call, eliminating per-pass CGo overhead.

package nn

/*
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>

// Fused softmax: max → subtract → exp → sum → divide
static void accelerate_softmax_f32(const float *input, float *output, int n) {
	vDSP_Length vn = (vDSP_Length)n;
	// 1. Find max
	float maxVal;
	vDSP_maxv(input, 1, &maxVal, vn);
	// 2. Subtract max: output = input + (-max)
	float negMax = -maxVal;
	vDSP_vsadd(input, 1, &negMax, output, 1, vn);
	// 3. Exp
	vvexpf(output, output, &n);
	// 4. Sum
	float sum;
	vDSP_sve(output, 1, &sum, vn);
	// 5. Normalize
	vDSP_vsdiv(output, 1, &sum, output, 1, vn);
}

static void accelerate_softmax_f64(const double *input, double *output, int n) {
	vDSP_Length vn = (vDSP_Length)n;
	double maxVal;
	vDSP_maxvD(input, 1, &maxVal, vn);
	double negMax = -maxVal;
	vDSP_vsaddD(input, 1, &negMax, output, 1, vn);
	vvexp(output, output, &n);
	double sum;
	vDSP_sveD(output, 1, &sum, vn);
	vDSP_vsdivD(output, 1, &sum, output, 1, vn);
}

// Fused log-softmax: max → subtract → exp → sum → log(sum) → subtract log(sum)
static void accelerate_log_softmax_f32(const float *input, float *output, int n) {
	vDSP_Length vn = (vDSP_Length)n;
	// 1. Find max
	float maxVal;
	vDSP_maxv(input, 1, &maxVal, vn);
	// 2. shifted = input - max
	float negMax = -maxVal;
	vDSP_vsadd(input, 1, &negMax, output, 1, vn);
	// 3. exp(shifted) into temp — we need shifted values preserved
	// Actually: log_softmax(x) = (x - max) - log(sum(exp(x - max)))
	// We can compute sum(exp(shifted)) without overwriting shifted
	float *temp = (float *)__builtin_alloca(n * sizeof(float));
	vvexpf(temp, output, &n);
	// 4. Sum of exp
	float sum;
	vDSP_sve(temp, 1, &sum, vn);
	// 5. log(sum)
	float logSum;
	int one = 1;
	vvlogf(&logSum, &sum, &one);
	// 6. output = shifted - log(sum)
	float negLogSum = -logSum;
	vDSP_vsadd(output, 1, &negLogSum, output, 1, vn);
}

static void accelerate_log_softmax_f64(const double *input, double *output, int n) {
	vDSP_Length vn = (vDSP_Length)n;
	double maxVal;
	vDSP_maxvD(input, 1, &maxVal, vn);
	double negMax = -maxVal;
	vDSP_vsaddD(input, 1, &negMax, output, 1, vn);
	double *temp = (double *)__builtin_alloca(n * sizeof(double));
	vvexp(temp, output, &n);
	double sum;
	vDSP_sveD(temp, 1, &sum, vn);
	double logSum;
	int one = 1;
	vvlog(&logSum, &sum, &one);
	double negLogSum = -logSum;
	vDSP_vsaddD(output, 1, &negLogSum, output, 1, vn);
}

// Softmax with temperature: softmax(x/T)
static void accelerate_softmax_temp_f32(const float *input, float *output, int n, float temperature) {
	vDSP_Length vn = (vDSP_Length)n;
	// 1. output = input / temperature
	vDSP_vsdiv(input, 1, &temperature, output, 1, vn);
	// 2. Softmax on scaled values
	float maxVal;
	vDSP_maxv(output, 1, &maxVal, vn);
	float negMax = -maxVal;
	vDSP_vsadd(output, 1, &negMax, output, 1, vn);
	vvexpf(output, output, &n);
	float sum;
	vDSP_sve(output, 1, &sum, vn);
	vDSP_vsdiv(output, 1, &sum, output, 1, vn);
}

static void accelerate_softmax_temp_f64(const double *input, double *output, int n, double temperature) {
	vDSP_Length vn = (vDSP_Length)n;
	vDSP_vsdivD(input, 1, &temperature, output, 1, vn);
	double maxVal;
	vDSP_maxvD(output, 1, &maxVal, vn);
	double negMax = -maxVal;
	vDSP_vsaddD(output, 1, &negMax, output, 1, vn);
	vvexp(output, output, &n);
	double sum;
	vDSP_sveD(output, 1, &sum, vn);
	vDSP_vsdivD(output, 1, &sum, output, 1, vn);
}
*/
import "C"
import "unsafe"

func init() {
	SoftmaxFloat32 = accelerateSoftmaxFloat32
	SoftmaxFloat64 = accelerateSoftmaxFloat64
	SoftmaxInPlaceFloat32 = accelerateSoftmaxInPlaceFloat32
	SoftmaxInPlaceFloat64 = accelerateSoftmaxInPlaceFloat64
	LogSoftmaxFloat32 = accelerateLogSoftmaxFloat32
	LogSoftmaxFloat64 = accelerateLogSoftmaxFloat64
	LogSoftmaxInPlaceFloat32 = accelerateLogSoftmaxInPlaceFloat32
	LogSoftmaxInPlaceFloat64 = accelerateLogSoftmaxInPlaceFloat64
	SoftmaxWithTemperatureFloat32 = accelerateSoftmaxWithTemperatureFloat32
	SoftmaxWithTemperatureFloat64 = accelerateSoftmaxWithTemperatureFloat64
}

func accelerateSoftmaxFloat32(input, output []float32) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	C.accelerate_softmax_f32((*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&output[0])), C.int(n))
}

func accelerateSoftmaxFloat64(input, output []float64) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	C.accelerate_softmax_f64((*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&output[0])), C.int(n))
}

func accelerateSoftmaxInPlaceFloat32(x []float32) {
	if len(x) == 0 {
		return
	}
	C.accelerate_softmax_f32((*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&x[0])), C.int(len(x)))
}

func accelerateSoftmaxInPlaceFloat64(x []float64) {
	if len(x) == 0 {
		return
	}
	C.accelerate_softmax_f64((*C.double)(unsafe.Pointer(&x[0])),
		(*C.double)(unsafe.Pointer(&x[0])), C.int(len(x)))
}

func accelerateLogSoftmaxFloat32(input, output []float32) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	C.accelerate_log_softmax_f32((*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&output[0])), C.int(n))
}

func accelerateLogSoftmaxFloat64(input, output []float64) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	C.accelerate_log_softmax_f64((*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&output[0])), C.int(n))
}

func accelerateLogSoftmaxInPlaceFloat32(x []float32) {
	if len(x) == 0 {
		return
	}
	C.accelerate_log_softmax_f32((*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&x[0])), C.int(len(x)))
}

func accelerateLogSoftmaxInPlaceFloat64(x []float64) {
	if len(x) == 0 {
		return
	}
	C.accelerate_log_softmax_f64((*C.double)(unsafe.Pointer(&x[0])),
		(*C.double)(unsafe.Pointer(&x[0])), C.int(len(x)))
}

func accelerateSoftmaxWithTemperatureFloat32(input, output []float32, temperature float32) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	C.accelerate_softmax_temp_f32((*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&output[0])), C.int(n), C.float(temperature))
}

func accelerateSoftmaxWithTemperatureFloat64(input, output []float64, temperature float64) {
	n := min(len(input), len(output))
	if n == 0 {
		return
	}
	C.accelerate_softmax_temp_f64((*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&output[0])), C.int(n), C.double(temperature))
}
