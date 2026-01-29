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

// Float16x8AVX2 stub for non-AMD64 or non-SIMD builds.
type Float16x8AVX2 struct {
	data [32]byte // placeholder for archsimd.Float32x8
}

func LoadFloat16x8AVX2Slice(_ []uint16) Float16x8AVX2     { panic("AVX2 not available") }
func LoadFloat16x8AVX2Ptr(_ unsafe.Pointer) Float16x8AVX2 { panic("AVX2 not available") }
func (v Float16x8AVX2) StoreSlice(_ []uint16)              { panic("AVX2 not available") }
func (v Float16x8AVX2) StorePtr(_ unsafe.Pointer)          { panic("AVX2 not available") }
func BroadcastFloat16x8AVX2(_ uint16) Float16x8AVX2       { panic("AVX2 not available") }
func ZeroFloat16x8AVX2() Float16x8AVX2                    { panic("AVX2 not available") }

func (v Float16x8AVX2) Add(_ Float16x8AVX2) Float16x8AVX2     { panic("AVX2 not available") }
func (v Float16x8AVX2) Sub(_ Float16x8AVX2) Float16x8AVX2     { panic("AVX2 not available") }
func (v Float16x8AVX2) Mul(_ Float16x8AVX2) Float16x8AVX2     { panic("AVX2 not available") }
func (v Float16x8AVX2) Div(_ Float16x8AVX2) Float16x8AVX2     { panic("AVX2 not available") }
func (v Float16x8AVX2) Min(_ Float16x8AVX2) Float16x8AVX2     { panic("AVX2 not available") }
func (v Float16x8AVX2) Max(_ Float16x8AVX2) Float16x8AVX2     { panic("AVX2 not available") }
func (v Float16x8AVX2) Sqrt() Float16x8AVX2                   { panic("AVX2 not available") }
func (v Float16x8AVX2) Neg() Float16x8AVX2                    { panic("AVX2 not available") }
func (v Float16x8AVX2) Abs() Float16x8AVX2                    { panic("AVX2 not available") }
func (v Float16x8AVX2) MulAdd(_, _ Float16x8AVX2) Float16x8AVX2 {
	panic("AVX2 not available")
}
func (v Float16x8AVX2) MulSub(_, _ Float16x8AVX2) Float16x8AVX2 {
	panic("AVX2 not available")
}
func (v Float16x8AVX2) ReciprocalSqrt() Float16x8AVX2 { panic("AVX2 not available") }

func (v Float16x8AVX2) And(_ Float16x8AVX2) Float16x8AVX2    { panic("AVX2 not available") }
func (v Float16x8AVX2) Or(_ Float16x8AVX2) Float16x8AVX2     { panic("AVX2 not available") }
func (v Float16x8AVX2) Xor(_ Float16x8AVX2) Float16x8AVX2    { panic("AVX2 not available") }
func (v Float16x8AVX2) AndNot(_ Float16x8AVX2) Float16x8AVX2 { panic("AVX2 not available") }
func (v Float16x8AVX2) Not() Float16x8AVX2                   { panic("AVX2 not available") }

func (v Float16x8AVX2) ReduceSum() float32 { panic("AVX2 not available") }
func (v Float16x8AVX2) ReduceMax() float32 { panic("AVX2 not available") }
func (v Float16x8AVX2) ReduceMin() float32 { panic("AVX2 not available") }

func IotaFloat16x8AVX2() Float16x8AVX2    { panic("AVX2 not available") }
func SignBitFloat16x8AVX2() Float16x8AVX2 { panic("AVX2 not available") }

func (v Float16x8AVX2) InterleaveLower(_ Float16x8AVX2) Float16x8AVX2 {
	panic("AVX2 not available")
}
func (v Float16x8AVX2) InterleaveUpper(_ Float16x8AVX2) Float16x8AVX2 {
	panic("AVX2 not available")
}
