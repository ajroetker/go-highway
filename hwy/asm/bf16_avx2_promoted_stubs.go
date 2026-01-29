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

// BFloat16x8AVX2 stub for non-AMD64 or non-SIMD builds.
type BFloat16x8AVX2 struct {
	data [32]byte // placeholder for archsimd.Float32x8
}

func LoadBFloat16x8AVX2Slice(_ []uint16) BFloat16x8AVX2     { panic("AVX2 not available") }
func LoadBFloat16x8AVX2Ptr(_ unsafe.Pointer) BFloat16x8AVX2 { panic("AVX2 not available") }
func (v BFloat16x8AVX2) StoreSlice(_ []uint16)               { panic("AVX2 not available") }
func (v BFloat16x8AVX2) StorePtr(_ unsafe.Pointer)           { panic("AVX2 not available") }
func BroadcastBFloat16x8AVX2(_ uint16) BFloat16x8AVX2       { panic("AVX2 not available") }
func ZeroBFloat16x8AVX2() BFloat16x8AVX2                    { panic("AVX2 not available") }

func (v BFloat16x8AVX2) Add(_ BFloat16x8AVX2) BFloat16x8AVX2     { panic("AVX2 not available") }
func (v BFloat16x8AVX2) Sub(_ BFloat16x8AVX2) BFloat16x8AVX2     { panic("AVX2 not available") }
func (v BFloat16x8AVX2) Mul(_ BFloat16x8AVX2) BFloat16x8AVX2     { panic("AVX2 not available") }
func (v BFloat16x8AVX2) Div(_ BFloat16x8AVX2) BFloat16x8AVX2     { panic("AVX2 not available") }
func (v BFloat16x8AVX2) Min(_ BFloat16x8AVX2) BFloat16x8AVX2     { panic("AVX2 not available") }
func (v BFloat16x8AVX2) Max(_ BFloat16x8AVX2) BFloat16x8AVX2     { panic("AVX2 not available") }
func (v BFloat16x8AVX2) Sqrt() BFloat16x8AVX2                    { panic("AVX2 not available") }
func (v BFloat16x8AVX2) Neg() BFloat16x8AVX2                     { panic("AVX2 not available") }
func (v BFloat16x8AVX2) Abs() BFloat16x8AVX2                     { panic("AVX2 not available") }
func (v BFloat16x8AVX2) MulAdd(_, _ BFloat16x8AVX2) BFloat16x8AVX2 {
	panic("AVX2 not available")
}
func (v BFloat16x8AVX2) MulSub(_, _ BFloat16x8AVX2) BFloat16x8AVX2 {
	panic("AVX2 not available")
}
func (v BFloat16x8AVX2) ReciprocalSqrt() BFloat16x8AVX2 { panic("AVX2 not available") }

func (v BFloat16x8AVX2) And(_ BFloat16x8AVX2) BFloat16x8AVX2    { panic("AVX2 not available") }
func (v BFloat16x8AVX2) Or(_ BFloat16x8AVX2) BFloat16x8AVX2     { panic("AVX2 not available") }
func (v BFloat16x8AVX2) Xor(_ BFloat16x8AVX2) BFloat16x8AVX2    { panic("AVX2 not available") }
func (v BFloat16x8AVX2) AndNot(_ BFloat16x8AVX2) BFloat16x8AVX2 { panic("AVX2 not available") }
func (v BFloat16x8AVX2) Not() BFloat16x8AVX2                    { panic("AVX2 not available") }

func (v BFloat16x8AVX2) ReduceSum() float32 { panic("AVX2 not available") }
func (v BFloat16x8AVX2) ReduceMax() float32 { panic("AVX2 not available") }
func (v BFloat16x8AVX2) ReduceMin() float32 { panic("AVX2 not available") }

func IotaBFloat16x8AVX2() BFloat16x8AVX2    { panic("AVX2 not available") }
func SignBitBFloat16x8AVX2() BFloat16x8AVX2 { panic("AVX2 not available") }

func (v BFloat16x8AVX2) InterleaveLower(_ BFloat16x8AVX2) BFloat16x8AVX2 {
	panic("AVX2 not available")
}
func (v BFloat16x8AVX2) InterleaveUpper(_ BFloat16x8AVX2) BFloat16x8AVX2 {
	panic("AVX2 not available")
}
