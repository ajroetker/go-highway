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

//go:build amd64 && goexperiment.simd

package hwy

import (
	"simd/archsimd"
	"testing"
)

// These tests cover the AVX2 emulations of int64 operations that only exist on
// AVX-512 (VPMULLQ, VPSRAQ). They run on AVX2-only CPUs without a SIGILL and
// must match plain Go scalar semantics.

func extractInt64x4(v archsimd.Int64x4) [4]int64 {
	lo, hi := v.GetLo(), v.GetHi()
	return [4]int64{lo.GetElem(0), lo.GetElem(1), hi.GetElem(0), hi.GetElem(1)}
}

func extractUint64x4(v archsimd.Uint64x4) [4]uint64 {
	lo, hi := v.GetLo(), v.GetHi()
	return [4]uint64{lo.GetElem(0), lo.GetElem(1), hi.GetElem(0), hi.GetElem(1)}
}

func TestMulAVX2Int64x4(t *testing.T) {
	as := [4]int64{0, -1, 123456789, -987654321}
	bs := [4]int64{1, 7, -98765, 0x7FFFFFFF}
	got := extractInt64x4(Mul_AVX2_Int64x4(
		archsimd.LoadInt64x4(as[:]), archsimd.LoadInt64x4(bs[:])))
	for i := range as {
		if want := as[i] * bs[i]; got[i] != want {
			t.Errorf("Mul lane %d: %d*%d = %d, want %d", i, as[i], bs[i], got[i], want)
		}
	}
}

func TestMulAVX2Uint64x4(t *testing.T) {
	as := [4]uint64{0, 0xFFFFFFFFFFFFFFFF, 123456789, 0x1_0000_0001}
	bs := [4]uint64{1, 3, 98765, 0x1_0000_0001}
	got := extractUint64x4(Mul_AVX2_Uint64x4(
		archsimd.LoadUint64x4(as[:]), archsimd.LoadUint64x4(bs[:])))
	for i := range as {
		if want := as[i] * bs[i]; got[i] != want {
			t.Errorf("Mul lane %d: %d*%d = %d, want %d", i, as[i], bs[i], got[i], want)
		}
	}
}

func TestShiftAllRightAVX2Int64x4(t *testing.T) {
	vals := [4]int64{-8, 8, -1, 0x7FFFFFFFFFFFFFFF}
	for _, n := range []uint64{0, 1, 2, 13, 31, 62, 63} {
		got := extractInt64x4(ShiftAllRight_AVX2_Int64x4(archsimd.LoadInt64x4(vals[:]), n))
		for i := range vals {
			if want := vals[i] >> n; got[i] != want {
				t.Errorf("ShiftAllRight n=%d lane %d: %d>>%d = %d, want %d (arithmetic)", n, i, vals[i], n, got[i], want)
			}
		}
	}
}
