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

//go:build !noasm && darwin && arm64

package asm

import (
	"fmt"
	"testing"

	"github.com/ajroetker/go-highway/hwy"
)

// scalarSMOPAS8 computes the reference 16×16 signed int8×int8→int32 outer product.
func scalarSMOPAS8(aPanel, bPanel []int8, kGroups int) [256]int32 {
	var c [256]int32
	for k4 := range kGroups {
		for row := range 16 {
			for col := range 16 {
				for g := range 4 {
					a := int32(aPanel[k4*64+row*4+g])
					b := int32(bPanel[k4*64+col*4+g])
					c[row*16+col] += a * b
				}
			}
		}
	}
	return c
}

// scalarSUMOPAS8U8 computes the reference 16×16 signed×unsigned outer product.
func scalarSUMOPAS8U8(aPanel []int8, bPanel []uint8, kGroups int) [256]int32 {
	var c [256]int32
	for k4 := range kGroups {
		for row := range 16 {
			for col := range 16 {
				for g := range 4 {
					a := int32(aPanel[k4*64+row*4+g])
					b := int32(bPanel[k4*64+col*4+g])
					c[row*16+col] += a * b
				}
			}
		}
	}
	return c
}

func TestTileSMOPAS8_Identity(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	for _, kGroups := range []int{1, 2, 4, 8} {
		panelSize := kGroups * 64
		aPanel := make([]int8, panelSize)
		bPanel := make([]int8, panelSize)

		// Fill with deterministic signed pattern.
		for i := range panelSize {
			aPanel[i] = int8((i*7 + 3) % 256)
			bPanel[i] = int8((i*11 + 5) % 256)
		}

		want := scalarSMOPAS8(aPanel, bPanel, kGroups)
		got := make([]int32, 256)

		defer hwy.SMEGuard()()
		TileSMOPAS8(aPanel, bPanel, got, kGroups)

		for i := range 256 {
			if got[i] != want[i] {
				t.Errorf("kGroups=%d: c[%d,%d] = %d, want %d",
					kGroups, i/16, i%16, got[i], want[i])
			}
		}
	}
}

func TestTileSUMOPAS8U8_Identity(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	for _, kGroups := range []int{1, 2, 4, 8} {
		panelSize := kGroups * 64
		aPanel := make([]int8, panelSize)
		bPanel := make([]uint8, panelSize)

		// Fill with deterministic patterns.
		for i := range panelSize {
			aPanel[i] = int8((i*7 + 3) % 256) // signed [-128, 127]
			bPanel[i] = uint8((i*11 + 5) % 256) // unsigned [0, 255]
		}

		want := scalarSUMOPAS8U8(aPanel, bPanel, kGroups)
		got := make([]int32, 256)

		defer hwy.SMEGuard()()
		TileSUMOPAS8U8(aPanel, bPanel, got, kGroups)

		for i := range 256 {
			if got[i] != want[i] {
				t.Errorf("kGroups=%d: c[%d,%d] = %d, want %d",
					kGroups, i/16, i%16, got[i], want[i])
			}
		}
	}
}

func TestTileSMOPAS8_Zero(t *testing.T) {
	if !hwy.HasSME() {
		t.Skip("SME not available")
	}

	kGroups := 4
	panelSize := kGroups * 64
	aPanel := make([]int8, panelSize)
	bPanel := make([]int8, panelSize)

	got := make([]int32, 256)
	defer hwy.SMEGuard()()
	TileSMOPAS8(aPanel, bPanel, got, kGroups)

	for i := range 256 {
		if got[i] != 0 {
			t.Errorf("zero input: c[%d,%d] = %d, want 0", i/16, i%16, got[i])
		}
	}
}

func BenchmarkTileSMOPAS8(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	for _, kGroups := range []int{4, 8, 16, 64} {
		panelSize := kGroups * 64
		aPanel := make([]int8, panelSize)
		bPanel := make([]int8, panelSize)
		c := make([]int32, 256)

		for i := range panelSize {
			aPanel[i] = int8(i % 127)
			bPanel[i] = int8(i % 127)
		}

		b.Run(fmt.Sprintf("kGroups=%d", kGroups), func(b *testing.B) {
			defer hwy.SMEGuard()()
			b.SetBytes(int64(panelSize * 2))
			for range b.N {
				TileSMOPAS8(aPanel, bPanel, c, kGroups)
			}
		})
	}
}

func BenchmarkTileSUMOPAS8U8(b *testing.B) {
	if !hwy.HasSME() {
		b.Skip("SME not available")
	}

	for _, kGroups := range []int{4, 8, 16, 64} {
		panelSize := kGroups * 64
		aPanel := make([]int8, panelSize)
		bPanel := make([]uint8, panelSize)
		c := make([]int32, 256)

		for i := range panelSize {
			aPanel[i] = int8(i % 127)
			bPanel[i] = uint8(i % 255)
		}

		b.Run(fmt.Sprintf("kGroups=%d", kGroups), func(b *testing.B) {
			defer hwy.SMEGuard()()
			b.SetBytes(int64(panelSize * 2))
			for range b.N {
				TileSUMOPAS8U8(aPanel, bPanel, c, kGroups)
			}
		})
	}
}
