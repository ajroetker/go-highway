// Copyright 2025 The go-highway Authors. SPDX-License-Identifier: Apache-2.0

//go:build cgo && darwin

package matmul

/*
#cgo LDFLAGS: -framework Accelerate
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
*/
import "C"
import "unsafe"

func init() {
	accelerateSgemm = func(a, b, c []float32, m, n, k int) {
		C.cblas_sgemm(C.CblasRowMajor, C.CblasNoTrans, C.CblasNoTrans,
			C.int(m), C.int(n), C.int(k),
			1.0, (*C.float)(unsafe.Pointer(&a[0])), C.int(k),
			(*C.float)(unsafe.Pointer(&b[0])), C.int(n),
			0.0, (*C.float)(unsafe.Pointer(&c[0])), C.int(n))
	}
	accelerateDgemm = func(a, b, c []float64, m, n, k int) {
		C.cblas_dgemm(C.CblasRowMajor, C.CblasNoTrans, C.CblasNoTrans,
			C.int(m), C.int(n), C.int(k),
			1.0, (*C.double)(unsafe.Pointer(&a[0])), C.int(k),
			(*C.double)(unsafe.Pointer(&b[0])), C.int(n),
			0.0, (*C.double)(unsafe.Pointer(&c[0])), C.int(n))
	}
	accelerateSgemmT = func(a, b, c []float32, m, n, k int) {
		C.cblas_sgemm(C.CblasRowMajor, C.CblasNoTrans, C.CblasTrans,
			C.int(m), C.int(n), C.int(k),
			1.0, (*C.float)(unsafe.Pointer(&a[0])), C.int(k),
			(*C.float)(unsafe.Pointer(&b[0])), C.int(k),
			0.0, (*C.float)(unsafe.Pointer(&c[0])), C.int(n))
	}
	accelerateDgemmT = func(a, b, c []float64, m, n, k int) {
		C.cblas_dgemm(C.CblasRowMajor, C.CblasNoTrans, C.CblasTrans,
			C.int(m), C.int(n), C.int(k),
			1.0, (*C.double)(unsafe.Pointer(&a[0])), C.int(k),
			(*C.double)(unsafe.Pointer(&b[0])), C.int(k),
			0.0, (*C.double)(unsafe.Pointer(&c[0])), C.int(n))
	}
}
