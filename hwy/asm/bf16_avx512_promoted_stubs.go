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

// BFloat16x16AVX512 stub for non-AMD64 or non-SIMD builds.
type BFloat16x16AVX512 struct {
	data [64]byte // placeholder for archsimd.Float32x16
}

func LoadBFloat16x16AVX512Slice(_ []uint16) BFloat16x16AVX512     { panic("AVX512 not available") }
func LoadBFloat16x16AVX512Ptr(_ unsafe.Pointer) BFloat16x16AVX512 { panic("AVX512 not available") }
func (v BFloat16x16AVX512) StoreSlice(_ []uint16)                  { panic("AVX512 not available") }
func (v BFloat16x16AVX512) StorePtr(_ unsafe.Pointer)              { panic("AVX512 not available") }
func BroadcastBFloat16x16AVX512(_ uint16) BFloat16x16AVX512       { panic("AVX512 not available") }
func ZeroBFloat16x16AVX512() BFloat16x16AVX512                    { panic("AVX512 not available") }

func (v BFloat16x16AVX512) Add(_ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
func (v BFloat16x16AVX512) Sub(_ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
func (v BFloat16x16AVX512) Mul(_ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
func (v BFloat16x16AVX512) Div(_ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
func (v BFloat16x16AVX512) Min(_ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
func (v BFloat16x16AVX512) Max(_ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
func (v BFloat16x16AVX512) Sqrt() BFloat16x16AVX512 { panic("AVX512 not available") }
func (v BFloat16x16AVX512) Neg() BFloat16x16AVX512  { panic("AVX512 not available") }
func (v BFloat16x16AVX512) Abs() BFloat16x16AVX512  { panic("AVX512 not available") }
func (v BFloat16x16AVX512) MulAdd(_, _ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
func (v BFloat16x16AVX512) MulSub(_, _ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
func (v BFloat16x16AVX512) ReciprocalSqrt() BFloat16x16AVX512 { panic("AVX512 not available") }

func (v BFloat16x16AVX512) And(_ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
func (v BFloat16x16AVX512) Or(_ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
func (v BFloat16x16AVX512) Xor(_ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
func (v BFloat16x16AVX512) AndNot(_ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
func (v BFloat16x16AVX512) Not() BFloat16x16AVX512 { panic("AVX512 not available") }

func (v BFloat16x16AVX512) ReduceSum() float32 { panic("AVX512 not available") }
func (v BFloat16x16AVX512) ReduceMax() float32 { panic("AVX512 not available") }
func (v BFloat16x16AVX512) ReduceMin() float32 { panic("AVX512 not available") }

func IotaBFloat16x16AVX512() BFloat16x16AVX512    { panic("AVX512 not available") }
func SignBitBFloat16x16AVX512() BFloat16x16AVX512 { panic("AVX512 not available") }

func (v BFloat16x16AVX512) InterleaveLower(_ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
func (v BFloat16x16AVX512) InterleaveUpper(_ BFloat16x16AVX512) BFloat16x16AVX512 {
	panic("AVX512 not available")
}
