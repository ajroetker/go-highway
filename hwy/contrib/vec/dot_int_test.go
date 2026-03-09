package vec

import (
	"math/rand"
	"strconv"
	"testing"
)

func TestDotIntInt8_KnownValues(t *testing.T) {
	// 1+2+3+...+16 = 136
	a := make([]int8, 16)
	b := make([]int8, 16)
	for i := range a {
		a[i] = int8(i + 1)
		b[i] = 1
	}
	got := DotInt(a, b)
	if got != 136 {
		t.Errorf("DotInt([1..16], [1..1]) = %d, want 136", got)
	}
}

func TestDotIntUint8_KnownValues(t *testing.T) {
	a := make([]uint8, 16)
	b := make([]uint8, 16)
	for i := range a {
		a[i] = uint8(i + 1)
		b[i] = 1
	}
	got := DotInt(a, b)
	if got != 136 {
		t.Errorf("DotInt([1..16], [1..1]) = %d, want 136", got)
	}
}

func TestDotIntInt8_Negative(t *testing.T) {
	a := []int8{1, -2, 3, -4, 5, -6, 7, -8}
	b := []int8{1, 1, 1, 1, 1, 1, 1, 1}
	// 1-2+3-4+5-6+7-8 = -4
	got := DotInt(a, b)
	if got != -4 {
		t.Errorf("DotInt(alternating signs) = %d, want -4", got)
	}
}

func TestDotIntInt8_AllNegative(t *testing.T) {
	a := []int8{-1, -2, -3, -4}
	b := []int8{-1, -2, -3, -4}
	// 1+4+9+16 = 30
	got := DotInt(a, b)
	if got != 30 {
		t.Errorf("DotInt(all negative) = %d, want 30", got)
	}
}

func TestDotInt_Empty(t *testing.T) {
	got := DotInt[int8](nil, nil)
	if got != 0 {
		t.Errorf("DotInt(nil, nil) = %d, want 0", got)
	}

	got = DotInt[uint8](nil, nil)
	if got != 0 {
		t.Errorf("DotInt(nil, nil) = %d, want 0", got)
	}
}

func TestDotInt_Short(t *testing.T) {
	// Fewer elements than one SIMD vector (16 lanes for int8)
	a := []int8{3, 5, 7}
	b := []int8{2, 4, 6}
	// 6+20+42 = 68
	got := DotInt(a, b)
	if got != 68 {
		t.Errorf("DotInt(short) = %d, want 68", got)
	}
}

func TestDotInt_Large(t *testing.T) {
	n := 10000
	a := make([]int8, n)
	b := make([]int8, n)
	for i := range a {
		a[i] = 1
		b[i] = 1
	}
	got := DotInt(a, b)
	if got != int32(n) {
		t.Errorf("DotInt(1s, 1s, n=%d) = %d, want %d", n, got, n)
	}
}

func TestDotIntUint8_Large(t *testing.T) {
	n := 10000
	a := make([]uint8, n)
	b := make([]uint8, n)
	for i := range a {
		a[i] = 1
		b[i] = 1
	}
	got := DotInt(a, b)
	if got != int32(n) {
		t.Errorf("DotInt(1s, 1s, n=%d) = %d, want %d", n, got, n)
	}
}

func TestDotIntUint8_MaxValues(t *testing.T) {
	// 4 elements at max: 4*255*255 = 260100, well within int32
	a := []uint8{255, 255, 255, 255}
	b := []uint8{255, 255, 255, 255}
	got := DotInt(a, b)
	if got != 260100 {
		t.Errorf("DotInt(255s) = %d, want 260100", got)
	}
}

func TestDotIntInt8_MinMax(t *testing.T) {
	// -128 * -128 = 16384, 4 of these = 65536
	a := []int8{-128, -128, -128, -128}
	b := []int8{-128, -128, -128, -128}
	got := DotInt(a, b)
	if got != 65536 {
		t.Errorf("DotInt(-128s) = %d, want 65536", got)
	}
}

// scalarDotInt computes dot product in pure scalar Go for reference.
func scalarDotInt[T int8 | uint8](a, b []T) int32 {
	n := min(len(a), len(b))
	var sum int32
	for i := 0; i < n; i++ {
		sum += int32(a[i]) * int32(b[i])
	}
	return sum
}

func TestDotIntInt8_MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for _, n := range []int{0, 1, 3, 4, 7, 15, 16, 17, 31, 32, 33, 63, 64, 100, 255, 256, 1000} {
		a := make([]int8, n)
		b := make([]int8, n)
		for i := range a {
			a[i] = int8(rng.Intn(256) - 128)
			b[i] = int8(rng.Intn(256) - 128)
		}
		got := DotInt(a, b)
		want := scalarDotInt(a, b)
		if got != want {
			t.Errorf("DotInt(int8, n=%d) = %d, scalar = %d", n, got, want)
		}
	}
}

func TestDotIntUint8_MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for _, n := range []int{0, 1, 3, 4, 7, 15, 16, 17, 31, 32, 33, 63, 64, 100, 255, 256, 1000} {
		a := make([]uint8, n)
		b := make([]uint8, n)
		for i := range a {
			a[i] = uint8(rng.Intn(256))
			b[i] = uint8(rng.Intn(256))
		}
		got := DotInt(a, b)
		want := scalarDotInt(a, b)
		if got != want {
			t.Errorf("DotInt(uint8, n=%d) = %d, scalar = %d", n, got, want)
		}
	}
}

func TestDotInt_FallbackMatchesDispatch(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	n := 512
	a := make([]int8, n)
	b := make([]int8, n)
	for i := range a {
		a[i] = int8(rng.Intn(256) - 128)
		b[i] = int8(rng.Intn(256) - 128)
	}

	dispatch := DotInt(a, b)
	fallback := BaseDotInt_fallback_Int8(a, b)
	if dispatch != fallback {
		t.Errorf("dispatch=%d, fallback=%d", dispatch, fallback)
	}
}

func BenchmarkDotIntInt8(b *testing.B) {
	for _, size := range []int{256, 1024, 4096, 16384} {
		a := make([]int8, size)
		bv := make([]int8, size)
		for i := range a {
			a[i] = int8(i % 127)
			bv[i] = int8(i % 127)
		}
		b.Run(intSizeName("int8", size), func(b *testing.B) {
			b.SetBytes(int64(size * 2))
			for i := 0; i < b.N; i++ {
				_ = DotInt(a, bv)
			}
		})
	}
}

func BenchmarkDotIntUint8(b *testing.B) {
	for _, size := range []int{256, 1024, 4096, 16384} {
		a := make([]uint8, size)
		bv := make([]uint8, size)
		for i := range a {
			a[i] = uint8(i % 255)
			bv[i] = uint8(i % 255)
		}
		b.Run(intSizeName("uint8", size), func(b *testing.B) {
			b.SetBytes(int64(size * 2))
			for i := 0; i < b.N; i++ {
				_ = DotInt(a, bv)
			}
		})
	}
}

func intSizeName(typ string, size int) string {
	switch {
	case size >= 1024:
		return typ + "_" + strconv.Itoa(size/1024) + "K"
	default:
		return typ + "_" + strconv.Itoa(size)
	}
}
