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

package roaring

import (
	"math/bits"
	"math/rand/v2"
	"testing"
)

// bitmapContainerSize is the size of a roaring bitmap container in uint64 words.
const bitmapContainerSize = 1024

// --- Reference scalar implementations for testing ---

func refPopcntSlice(s []uint64) uint64 {
	var count uint64
	for _, v := range s {
		count += uint64(bits.OnesCount64(v))
	}
	return count
}

func refPopcntAndSlice(s, m []uint64) uint64 {
	n := min(len(s), len(m))
	var count uint64
	for i := 0; i < n; i++ {
		count += uint64(bits.OnesCount64(s[i] & m[i]))
	}
	return count
}

func refPopcntOrSlice(s, m []uint64) uint64 {
	n := min(len(s), len(m))
	var count uint64
	for i := 0; i < n; i++ {
		count += uint64(bits.OnesCount64(s[i] | m[i]))
	}
	return count
}

func refPopcntXorSlice(s, m []uint64) uint64 {
	n := min(len(s), len(m))
	var count uint64
	for i := 0; i < n; i++ {
		count += uint64(bits.OnesCount64(s[i] ^ m[i]))
	}
	return count
}

func refPopcntAndNotSlice(s, m []uint64) uint64 {
	n := min(len(s), len(m))
	var count uint64
	for i := 0; i < n; i++ {
		count += uint64(bits.OnesCount64(s[i] &^ m[i]))
	}
	return count
}

func randomBitmap(n int) []uint64 {
	s := make([]uint64, n)
	for i := range s {
		s[i] = rand.Uint64()
	}
	return s
}

// --- Popcount tests ---

func TestPopcntSlice(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 100, 1024} {
		s := randomBitmap(n)
		want := refPopcntSlice(s)
		got := PopcntSlice(s)
		if got != want {
			t.Errorf("PopcntSlice(len=%d): got %d, want %d", n, got, want)
		}
	}
}

func TestPopcntAndSlice(t *testing.T) {
	for _, n := range []int{0, 1, 3, 8, 15, 32, 100, 1024} {
		s := randomBitmap(n)
		m := randomBitmap(n)
		want := refPopcntAndSlice(s, m)
		got := PopcntAndSlice(s, m)
		if got != want {
			t.Errorf("PopcntAndSlice(len=%d): got %d, want %d", n, got, want)
		}
	}
}

func TestPopcntOrSlice(t *testing.T) {
	for _, n := range []int{0, 1, 3, 8, 15, 32, 100, 1024} {
		s := randomBitmap(n)
		m := randomBitmap(n)
		want := refPopcntOrSlice(s, m)
		got := PopcntOrSlice(s, m)
		if got != want {
			t.Errorf("PopcntOrSlice(len=%d): got %d, want %d", n, got, want)
		}
	}
}

func TestPopcntXorSlice(t *testing.T) {
	for _, n := range []int{0, 1, 3, 8, 15, 32, 100, 1024} {
		s := randomBitmap(n)
		m := randomBitmap(n)
		want := refPopcntXorSlice(s, m)
		got := PopcntXorSlice(s, m)
		if got != want {
			t.Errorf("PopcntXorSlice(len=%d): got %d, want %d", n, got, want)
		}
	}
}

func TestPopcntAndNotSlice(t *testing.T) {
	for _, n := range []int{0, 1, 3, 8, 15, 32, 100, 1024} {
		s := randomBitmap(n)
		m := randomBitmap(n)
		want := refPopcntAndNotSlice(s, m)
		got := PopcntAndNotSlice(s, m)
		if got != want {
			t.Errorf("PopcntAndNotSlice(len=%d): got %d, want %d", n, got, want)
		}
	}
}

// --- Bitwise tests ---

func TestAndSlice(t *testing.T) {
	for _, n := range []int{0, 1, 3, 8, 15, 32, 100, 1024} {
		a := randomBitmap(n)
		b := randomBitmap(n)
		got := make([]uint64, n)
		want := make([]uint64, n)
		AndSlice(got, a, b)
		for i := range n {
			want[i] = a[i] & b[i]
		}
		for i := range n {
			if got[i] != want[i] {
				t.Errorf("AndSlice(len=%d)[%d]: got %x, want %x", n, i, got[i], want[i])
				break
			}
		}
	}
}

func TestOrSlice(t *testing.T) {
	for _, n := range []int{0, 1, 3, 8, 15, 32, 100, 1024} {
		a := randomBitmap(n)
		b := randomBitmap(n)
		got := make([]uint64, n)
		want := make([]uint64, n)
		OrSlice(got, a, b)
		for i := range n {
			want[i] = a[i] | b[i]
		}
		for i := range n {
			if got[i] != want[i] {
				t.Errorf("OrSlice(len=%d)[%d]: got %x, want %x", n, i, got[i], want[i])
				break
			}
		}
	}
}

func TestXorSlice(t *testing.T) {
	for _, n := range []int{0, 1, 3, 8, 15, 32, 100, 1024} {
		a := randomBitmap(n)
		b := randomBitmap(n)
		got := make([]uint64, n)
		want := make([]uint64, n)
		XorSlice(got, a, b)
		for i := range n {
			want[i] = a[i] ^ b[i]
		}
		for i := range n {
			if got[i] != want[i] {
				t.Errorf("XorSlice(len=%d)[%d]: got %x, want %x", n, i, got[i], want[i])
				break
			}
		}
	}
}

func TestAndNotSlice(t *testing.T) {
	for _, n := range []int{0, 1, 3, 8, 15, 32, 100, 1024} {
		a := randomBitmap(n)
		b := randomBitmap(n)
		got := make([]uint64, n)
		want := make([]uint64, n)
		AndNotSlice(got, a, b)
		for i := range n {
			want[i] = a[i] &^ b[i]
		}
		for i := range n {
			if got[i] != want[i] {
				t.Errorf("AndNotSlice(len=%d)[%d]: got %x, want %x", n, i, got[i], want[i])
				break
			}
		}
	}
}

func TestMismatchedLengths(t *testing.T) {
	a := []uint64{0xFFFF, 0x0F0F, 0xAAAA, 0x5555}
	b := []uint64{0x00FF, 0x00F0, 0x0FF0}

	dst := []uint64{^uint64(0), ^uint64(0)}
	AndSlice(dst, a, b)
	wantAnd := []uint64{a[0] & b[0], a[1] & b[1]}
	for i := range len(dst) {
		if dst[i] != wantAnd[i] {
			t.Fatalf("AndSlice mismatch[%d]: got %x, want %x", i, dst[i], wantAnd[i])
		}
	}

	dst = []uint64{^uint64(0), ^uint64(0)}
	gotCard := AndPopcntSlice(dst, a, b)
	var wantCard uint64
	for i := range len(dst) {
		v := a[i] & b[i]
		wantCard += uint64(bits.OnesCount64(v))
		if dst[i] != v {
			t.Fatalf("AndPopcntSlice mismatch[%d]: got %x, want %x", i, dst[i], v)
		}
	}
	if gotCard != wantCard {
		t.Fatalf("AndPopcntSlice mismatch cardinality: got %d, want %d", gotCard, wantCard)
	}
}

// --- Edge cases ---

func TestPopcntSliceAllZeros(t *testing.T) {
	s := make([]uint64, 1024)
	if got := PopcntSlice(s); got != 0 {
		t.Errorf("PopcntSlice(zeros): got %d, want 0", got)
	}
}

func TestPopcntSliceAllOnes(t *testing.T) {
	s := make([]uint64, 1024)
	for i := range s {
		s[i] = ^uint64(0)
	}
	want := uint64(1024 * 64)
	if got := PopcntSlice(s); got != want {
		t.Errorf("PopcntSlice(ones): got %d, want %d", got, want)
	}
}

func TestAndNotSliceIdentity(t *testing.T) {
	a := randomBitmap(1024)
	dst := make([]uint64, 1024)
	// a &^ 0 == a
	zeros := make([]uint64, 1024)
	AndNotSlice(dst, a, zeros)
	for i := range 1024 {
		if dst[i] != a[i] {
			t.Errorf("AndNotSlice(a, zeros)[%d]: got %x, want %x", i, dst[i], a[i])
			break
		}
	}
	// a &^ a == 0
	AndNotSlice(dst, a, a)
	for i := range 1024 {
		if dst[i] != 0 {
			t.Errorf("AndNotSlice(a, a)[%d]: got %x, want 0", i, dst[i])
			break
		}
	}
}

// --- Benchmarks at roaring bitmap container size (1024 uint64 = 8KB) ---

func BenchmarkPopcntSlice(b *testing.B) {
	s := randomBitmap(bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8))
	b.ResetTimer()
	for b.Loop() {
		PopcntSlice(s)
	}
}

func BenchmarkPopcntAndSlice(b *testing.B) {
	s := randomBitmap(bitmapContainerSize)
	m := randomBitmap(bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 2))
	b.ResetTimer()
	for b.Loop() {
		PopcntAndSlice(s, m)
	}
}

func BenchmarkPopcntOrSlice(b *testing.B) {
	s := randomBitmap(bitmapContainerSize)
	m := randomBitmap(bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 2))
	b.ResetTimer()
	for b.Loop() {
		PopcntOrSlice(s, m)
	}
}

func BenchmarkPopcntXorSlice(b *testing.B) {
	s := randomBitmap(bitmapContainerSize)
	m := randomBitmap(bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 2))
	b.ResetTimer()
	for b.Loop() {
		PopcntXorSlice(s, m)
	}
}

func BenchmarkPopcntAndNotSlice(b *testing.B) {
	s := randomBitmap(bitmapContainerSize)
	m := randomBitmap(bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 2))
	b.ResetTimer()
	for b.Loop() {
		PopcntAndNotSlice(s, m)
	}
}

func BenchmarkAndSlice(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	dst := make([]uint64, bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 3))
	b.ResetTimer()
	for b.Loop() {
		AndSlice(dst, a, bb)
	}
}

func BenchmarkOrSlice(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	dst := make([]uint64, bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 3))
	b.ResetTimer()
	for b.Loop() {
		OrSlice(dst, a, bb)
	}
}

func BenchmarkXorSlice(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	dst := make([]uint64, bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 3))
	b.ResetTimer()
	for b.Loop() {
		XorSlice(dst, a, bb)
	}
}

func BenchmarkAndNotSlice(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	dst := make([]uint64, bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 3))
	b.ResetTimer()
	for b.Loop() {
		AndNotSlice(dst, a, bb)
	}
}

// --- Fused bitwise+store+popcount tests ---

func testFused(t *testing.T, name string, fn func(dst, a, b []uint64) uint64, refOp func(uint64, uint64) uint64) {
	t.Helper()
	for _, n := range []int{0, 1, 3, 8, 15, 32, 100, 1024} {
		a := randomBitmap(n)
		b := randomBitmap(n)
		got := make([]uint64, n)
		gotCard := fn(got, a, b)

		var wantCard uint64
		for i := range n {
			v := refOp(a[i], b[i])
			if got[i] != v {
				t.Errorf("%s(len=%d)[%d]: got %x, want %x", name, n, i, got[i], v)
				break
			}
			wantCard += uint64(bits.OnesCount64(v))
		}
		if gotCard != wantCard {
			t.Errorf("%s(len=%d) cardinality: got %d, want %d", name, n, gotCard, wantCard)
		}
	}
}

func TestAndPopcntSlice(t *testing.T) {
	testFused(t, "AndPopcntSlice", AndPopcntSlice, func(a, b uint64) uint64 { return a & b })
}

func TestOrPopcntSlice(t *testing.T) {
	testFused(t, "OrPopcntSlice", OrPopcntSlice, func(a, b uint64) uint64 { return a | b })
}

func TestXorPopcntSlice(t *testing.T) {
	testFused(t, "XorPopcntSlice", XorPopcntSlice, func(a, b uint64) uint64 { return a ^ b })
}

func TestAndNotPopcntSlice(t *testing.T) {
	testFused(t, "AndNotPopcntSlice", AndNotPopcntSlice, func(a, b uint64) uint64 { return a &^ b })
}

// --- PopcntSliceRange tests ---

func TestPopcntSliceRange(t *testing.T) {
	s := randomBitmap(1024)
	tests := []struct {
		name       string
		start, end int
	}{
		{"full", 0, 1024},
		{"first half", 0, 512},
		{"second half", 512, 1024},
		{"single", 100, 101},
		{"empty", 50, 50},
		{"out of range", 0, 2000},
		{"negative start", 1024, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PopcntSliceRange(s, tt.start, tt.end)
			// Compute reference
			start := tt.start
			end := tt.end
			if start >= end || start >= len(s) {
				if got != 0 {
					t.Errorf("got %d, want 0", got)
				}
				return
			}
			if end > len(s) {
				end = len(s)
			}
			want := refPopcntSlice(s[start:end])
			if got != want {
				t.Errorf("got %d, want %d", got, want)
			}
		})
	}
}

// --- Fused benchmarks ---

func BenchmarkAndPopcntSlice(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	dst := make([]uint64, bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 3))
	b.ResetTimer()
	for b.Loop() {
		AndPopcntSlice(dst, a, bb)
	}
}

func BenchmarkOrPopcntSlice(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	dst := make([]uint64, bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 3))
	b.ResetTimer()
	for b.Loop() {
		OrPopcntSlice(dst, a, bb)
	}
}

func BenchmarkXorPopcntSlice(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	dst := make([]uint64, bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 3))
	b.ResetTimer()
	for b.Loop() {
		XorPopcntSlice(dst, a, bb)
	}
}

func BenchmarkAndNotPopcntSlice(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	dst := make([]uint64, bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 3))
	b.ResetTimer()
	for b.Loop() {
		AndNotPopcntSlice(dst, a, bb)
	}
}

// Two-pass baseline: what roaring does today (popcount pass + bitwise pass).
func BenchmarkTwoPassAndPopcnt(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	dst := make([]uint64, bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 3))
	b.ResetTimer()
	for b.Loop() {
		PopcntAndSlice(a, bb)
		AndSlice(dst, a, bb)
	}
}

func BenchmarkTwoPassOrPopcnt(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	dst := make([]uint64, bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 3))
	b.ResetTimer()
	for b.Loop() {
		PopcntOrSlice(a, bb)
		OrSlice(dst, a, bb)
	}
}

// Scalar baseline benchmarks for comparison.

func BenchmarkScalarPopcntSlice(b *testing.B) {
	s := randomBitmap(bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8))
	b.ResetTimer()
	for b.Loop() {
		refPopcntSlice(s)
	}
}

func BenchmarkScalarPopcntAndSlice(b *testing.B) {
	s := randomBitmap(bitmapContainerSize)
	m := randomBitmap(bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 2))
	b.ResetTimer()
	for b.Loop() {
		refPopcntAndSlice(s, m)
	}
}

func BenchmarkScalarAndSlice(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	dst := make([]uint64, bitmapContainerSize)
	b.SetBytes(int64(bitmapContainerSize * 8 * 3))
	b.ResetTimer()
	for b.Loop() {
		for i := range bitmapContainerSize {
			dst[i] = a[i] & bb[i]
		}
	}
}

// --- Bit extraction tests ---

// refExtractBitPositions is the scalar reference matching roaring's fillLeastSignificant16bits.
func refExtractBitPositions(bitmap []uint64, out []uint16) int {
	pos := 0
	for k, bitset := range bitmap {
		for bitset != 0 {
			t := bitset & -bitset
			out[pos] = uint16(k*64 + bits.OnesCount64(t-1))
			pos++
			bitset ^= t
		}
	}
	return pos
}

func TestExtractBitPositions(t *testing.T) {
	for _, n := range []int{0, 1, 2, 8, 32, 100, 1024} {
		bitmap := randomBitmap(n)
		maxBits := n * 64
		got := make([]uint16, maxBits)
		want := make([]uint16, maxBits)

		gotCount := ExtractBitPositions(bitmap, got)
		wantCount := refExtractBitPositions(bitmap, want)

		if gotCount != wantCount {
			t.Errorf("ExtractBitPositions(len=%d) count: got %d, want %d", n, gotCount, wantCount)
			continue
		}
		for i := 0; i < gotCount; i++ {
			if got[i] != want[i] {
				t.Errorf("ExtractBitPositions(len=%d)[%d]: got %d, want %d", n, i, got[i], want[i])
				break
			}
		}
	}
}

func TestExtractBitPositionsAllZeros(t *testing.T) {
	bitmap := make([]uint64, 1024)
	out := make([]uint16, 65536)
	count := ExtractBitPositions(bitmap, out)
	if count != 0 {
		t.Errorf("ExtractBitPositions(zeros): got count %d, want 0", count)
	}
}

func TestExtractBitPositionsAllOnes(t *testing.T) {
	bitmap := make([]uint64, 16) // 16 words = 1024 bits
	for i := range bitmap {
		bitmap[i] = ^uint64(0)
	}
	out := make([]uint16, 1024)
	count := ExtractBitPositions(bitmap, out)
	if count != 1024 {
		t.Errorf("ExtractBitPositions(ones): got count %d, want 1024", count)
		return
	}
	for i := 0; i < 1024; i++ {
		if out[i] != uint16(i) {
			t.Errorf("ExtractBitPositions(ones)[%d]: got %d, want %d", i, out[i], i)
			break
		}
	}
}

func TestExtractBitPositionsSingleBits(t *testing.T) {
	// One bit set per word at various positions.
	bitmap := make([]uint64, 4)
	bitmap[0] = 1 << 0  // bit 0
	bitmap[1] = 1 << 63 // bit 127
	bitmap[2] = 1 << 31 // bit 159
	bitmap[3] = 1 << 42 // bit 234

	out := make([]uint16, 256)
	count := ExtractBitPositions(bitmap, out)
	if count != 4 {
		t.Fatalf("count = %d, want 4", count)
	}
	expected := []uint16{0, 127, 159, 234}
	for i, want := range expected {
		if out[i] != want {
			t.Errorf("out[%d] = %d, want %d", i, out[i], want)
		}
	}
}

func testExtractFused(t *testing.T, name string, fn func([]uint16, []uint64, []uint64) int, refOp func(uint64, uint64) uint64) {
	t.Helper()
	for _, n := range []int{0, 1, 8, 32, 100, 1024} {
		a := randomBitmap(n)
		b := randomBitmap(n)

		// Build reference by applying op then extracting.
		combined := make([]uint64, n)
		for i := range n {
			combined[i] = refOp(a[i], b[i])
		}
		maxBits := n * 64
		want := make([]uint16, maxBits)
		wantCount := refExtractBitPositions(combined, want)

		got := make([]uint16, maxBits)
		gotCount := fn(got, a, b)

		if gotCount != wantCount {
			t.Errorf("%s(len=%d) count: got %d, want %d", name, n, gotCount, wantCount)
			continue
		}
		for i := 0; i < gotCount; i++ {
			if got[i] != want[i] {
				t.Errorf("%s(len=%d)[%d]: got %d, want %d", name, n, i, got[i], want[i])
				break
			}
		}
	}
}

func TestExtractBitPositionsAND(t *testing.T) {
	testExtractFused(t, "ExtractAND", ExtractBitPositionsAND, func(a, b uint64) uint64 { return a & b })
}

func TestExtractBitPositionsANDNOT(t *testing.T) {
	testExtractFused(t, "ExtractANDNOT", ExtractBitPositionsANDNOT, func(a, b uint64) uint64 { return a &^ b })
}

func TestExtractBitPositionsXOR(t *testing.T) {
	testExtractFused(t, "ExtractXOR", ExtractBitPositionsXOR, func(a, b uint64) uint64 { return a ^ b })
}

// --- Bit extraction benchmarks ---

func BenchmarkExtractBitPositions(b *testing.B) {
	bitmap := randomBitmap(bitmapContainerSize)
	out := make([]uint16, bitmapContainerSize*64)
	b.SetBytes(int64(bitmapContainerSize * 8))
	b.ResetTimer()
	for b.Loop() {
		ExtractBitPositions(bitmap, out)
	}
}

func BenchmarkExtractBitPositionsAND(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	out := make([]uint16, bitmapContainerSize*64)
	b.SetBytes(int64(bitmapContainerSize * 8 * 2))
	b.ResetTimer()
	for b.Loop() {
		ExtractBitPositionsAND(out, a, bb)
	}
}

func BenchmarkExtractBitPositionsANDNOT(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	out := make([]uint16, bitmapContainerSize*64)
	b.SetBytes(int64(bitmapContainerSize * 8 * 2))
	b.ResetTimer()
	for b.Loop() {
		ExtractBitPositionsANDNOT(out, a, bb)
	}
}

func BenchmarkExtractBitPositionsXOR(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	out := make([]uint16, bitmapContainerSize*64)
	b.SetBytes(int64(bitmapContainerSize * 8 * 2))
	b.ResetTimer()
	for b.Loop() {
		ExtractBitPositionsXOR(out, a, bb)
	}
}

// Scalar baseline using roaring's original popcount(t-1) pattern.
func BenchmarkScalarExtractBitPositions(b *testing.B) {
	bitmap := randomBitmap(bitmapContainerSize)
	out := make([]uint16, bitmapContainerSize*64)
	b.SetBytes(int64(bitmapContainerSize * 8))
	b.ResetTimer()
	for b.Loop() {
		refExtractBitPositions(bitmap, out)
	}
}

// Sparse bitmap: ~1/16 density (typical of intersection results).
func BenchmarkExtractBitPositionsSparse(b *testing.B) {
	bitmap := make([]uint64, bitmapContainerSize)
	for i := range bitmap {
		bitmap[i] = rand.Uint64() & rand.Uint64() & rand.Uint64() & rand.Uint64() // ~4 bits/word
	}
	out := make([]uint16, bitmapContainerSize*64)
	b.SetBytes(int64(bitmapContainerSize * 8))
	b.ResetTimer()
	for b.Loop() {
		ExtractBitPositions(bitmap, out)
	}
}

func BenchmarkScalarExtractBitPositionsSparse(b *testing.B) {
	bitmap := make([]uint64, bitmapContainerSize)
	for i := range bitmap {
		bitmap[i] = rand.Uint64() & rand.Uint64() & rand.Uint64() & rand.Uint64()
	}
	out := make([]uint16, bitmapContainerSize*64)
	b.SetBytes(int64(bitmapContainerSize * 8))
	b.ResetTimer()
	for b.Loop() {
		refExtractBitPositions(bitmap, out)
	}
}

// Two-pass baseline: what roaring does for AND → array conversion
// (popcntAndSlice to check cardinality, then fillArrayAND).
func BenchmarkTwoPassExtractAND(b *testing.B) {
	a := randomBitmap(bitmapContainerSize)
	bb := randomBitmap(bitmapContainerSize)
	out := make([]uint16, bitmapContainerSize*64)
	b.SetBytes(int64(bitmapContainerSize * 8 * 2))
	b.ResetTimer()
	for b.Loop() {
		PopcntAndSlice(a, bb)
		refExtractBitPositions(a, out) // simplified: in reality roaring does fillArrayAND
	}
}
