// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !amd64 || !goexperiment.simd

package asm

import "unsafe"

// Float16x16AVX512 stub for non-AMD64 or non-SIMD builds.
type Float16x16AVX512 struct {
	data [64]byte // placeholder for archsimd.Float32x16
}

func LoadFloat16x16AVX512Slice(_ []uint16) Float16x16AVX512     { panic("AVX512 not available") }
func LoadFloat16x16AVX512Ptr(_ unsafe.Pointer) Float16x16AVX512 { panic("AVX512 not available") }
func (v Float16x16AVX512) StoreSlice(_ []uint16)                 { panic("AVX512 not available") }
func (v Float16x16AVX512) StorePtr(_ unsafe.Pointer)             { panic("AVX512 not available") }
func BroadcastFloat16x16AVX512(_ uint16) Float16x16AVX512       { panic("AVX512 not available") }
func ZeroFloat16x16AVX512() Float16x16AVX512                    { panic("AVX512 not available") }

func (v Float16x16AVX512) Add(_ Float16x16AVX512) Float16x16AVX512     { panic("AVX512 not available") }
func (v Float16x16AVX512) Sub(_ Float16x16AVX512) Float16x16AVX512     { panic("AVX512 not available") }
func (v Float16x16AVX512) Mul(_ Float16x16AVX512) Float16x16AVX512     { panic("AVX512 not available") }
func (v Float16x16AVX512) Div(_ Float16x16AVX512) Float16x16AVX512     { panic("AVX512 not available") }
func (v Float16x16AVX512) Min(_ Float16x16AVX512) Float16x16AVX512     { panic("AVX512 not available") }
func (v Float16x16AVX512) Max(_ Float16x16AVX512) Float16x16AVX512     { panic("AVX512 not available") }
func (v Float16x16AVX512) Sqrt() Float16x16AVX512                      { panic("AVX512 not available") }
func (v Float16x16AVX512) Neg() Float16x16AVX512                       { panic("AVX512 not available") }
func (v Float16x16AVX512) Abs() Float16x16AVX512                       { panic("AVX512 not available") }
func (v Float16x16AVX512) MulAdd(_, _ Float16x16AVX512) Float16x16AVX512 {
	panic("AVX512 not available")
}
func (v Float16x16AVX512) MulSub(_, _ Float16x16AVX512) Float16x16AVX512 {
	panic("AVX512 not available")
}
func (v Float16x16AVX512) ReciprocalSqrt() Float16x16AVX512 { panic("AVX512 not available") }

func (v Float16x16AVX512) And(_ Float16x16AVX512) Float16x16AVX512    { panic("AVX512 not available") }
func (v Float16x16AVX512) Or(_ Float16x16AVX512) Float16x16AVX512     { panic("AVX512 not available") }
func (v Float16x16AVX512) Xor(_ Float16x16AVX512) Float16x16AVX512    { panic("AVX512 not available") }
func (v Float16x16AVX512) AndNot(_ Float16x16AVX512) Float16x16AVX512 { panic("AVX512 not available") }
func (v Float16x16AVX512) Not() Float16x16AVX512                      { panic("AVX512 not available") }

func (v Float16x16AVX512) ReduceSum() float32 { panic("AVX512 not available") }
func (v Float16x16AVX512) ReduceMax() float32 { panic("AVX512 not available") }
func (v Float16x16AVX512) ReduceMin() float32 { panic("AVX512 not available") }

func IotaFloat16x16AVX512() Float16x16AVX512    { panic("AVX512 not available") }
func SignBitFloat16x16AVX512() Float16x16AVX512 { panic("AVX512 not available") }

func (v Float16x16AVX512) InterleaveLower(_ Float16x16AVX512) Float16x16AVX512 {
	panic("AVX512 not available")
}
func (v Float16x16AVX512) InterleaveUpper(_ Float16x16AVX512) Float16x16AVX512 {
	panic("AVX512 not available")
}
