package hwy

import (
	"math"
	"testing"
)

func TestLoad(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	v := Load(data)

	if v.NumLanes() == 0 {
		t.Error("Load created empty vector")
	}

	for i := 0; i < v.NumLanes() && i < len(data); i++ {
		if v.data[i] != data[i] {
			t.Errorf("Load: lane %d: got %v, want %v", i, v.data[i], data[i])
		}
	}
}

func TestSet(t *testing.T) {
	v := Set[float32](42.0)

	if v.NumLanes() == 0 {
		t.Error("Set created empty vector")
	}

	for i := 0; i < v.NumLanes(); i++ {
		if v.data[i] != 42.0 {
			t.Errorf("Set: lane %d: got %v, want %v", i, v.data[i], 42.0)
		}
	}
}

func TestZero(t *testing.T) {
	v := Zero[int32]()

	if v.NumLanes() == 0 {
		t.Error("Zero created empty vector")
	}

	for i := 0; i < v.NumLanes(); i++ {
		if v.data[i] != 0 {
			t.Errorf("Zero: lane %d: got %v, want 0", i, v.data[i])
		}
	}
}

func TestAdd(t *testing.T) {
	a := Set[float32](10.0)
	b := Set[float32](5.0)
	result := Add(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		if result.data[i] != 15.0 {
			t.Errorf("Add: lane %d: got %v, want 15.0", i, result.data[i])
		}
	}
}

func TestSub(t *testing.T) {
	a := Set[float32](10.0)
	b := Set[float32](3.0)
	result := Sub(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		if result.data[i] != 7.0 {
			t.Errorf("Sub: lane %d: got %v, want 7.0", i, result.data[i])
		}
	}
}

func TestMul(t *testing.T) {
	a := Set[float32](4.0)
	b := Set[float32](5.0)
	result := Mul(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		if result.data[i] != 20.0 {
			t.Errorf("Mul: lane %d: got %v, want 20.0", i, result.data[i])
		}
	}
}

func TestDiv(t *testing.T) {
	a := Set[float32](20.0)
	b := Set[float32](4.0)
	result := Div(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		if result.data[i] != 5.0 {
			t.Errorf("Div: lane %d: got %v, want 5.0", i, result.data[i])
		}
	}
}

func TestNeg(t *testing.T) {
	v := Set[float32](42.0)
	result := Neg(v)

	for i := 0; i < result.NumLanes(); i++ {
		if result.data[i] != -42.0 {
			t.Errorf("Neg: lane %d: got %v, want -42.0", i, result.data[i])
		}
	}
}

func TestAbs(t *testing.T) {
	v := Set[float32](-42.0)
	result := Abs(v)

	for i := 0; i < result.NumLanes(); i++ {
		if result.data[i] != 42.0 {
			t.Errorf("Abs: lane %d: got %v, want 42.0", i, result.data[i])
		}
	}
}

func TestMin(t *testing.T) {
	a := Set[float32](10.0)
	b := Set[float32](5.0)
	result := Min(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		if result.data[i] != 5.0 {
			t.Errorf("Min: lane %d: got %v, want 5.0", i, result.data[i])
		}
	}
}

func TestMax(t *testing.T) {
	a := Set[float32](10.0)
	b := Set[float32](5.0)
	result := Max(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		if result.data[i] != 10.0 {
			t.Errorf("Max: lane %d: got %v, want 10.0", i, result.data[i])
		}
	}
}

func TestSqrt(t *testing.T) {
	v := Set[float32](16.0)
	result := Sqrt(v)

	for i := 0; i < result.NumLanes(); i++ {
		if result.data[i] != 4.0 {
			t.Errorf("Sqrt: lane %d: got %v, want 4.0", i, result.data[i])
		}
	}
}

func TestFMA(t *testing.T) {
	a := Set[float32](2.0)
	b := Set[float32](3.0)
	c := Set[float32](4.0)
	result := FMA(a, b, c) // 2*3 + 4 = 10

	for i := 0; i < result.NumLanes(); i++ {
		if result.data[i] != 10.0 {
			t.Errorf("FMA: lane %d: got %v, want 10.0", i, result.data[i])
		}
	}
}

func TestReduceSum(t *testing.T) {
	data := []float32{1, 2, 3, 4}
	v := Load(data)
	sum := ReduceSum(v)

	// Sum should be at least the first MaxLanes elements
	expectedMin := float32(0)
	for i := 0; i < v.NumLanes() && i < len(data); i++ {
		expectedMin += data[i]
	}

	if sum < expectedMin-0.001 || sum > expectedMin+0.001 {
		t.Errorf("ReduceSum: got %v, want ~%v", sum, expectedMin)
	}
}

func TestEqual(t *testing.T) {
	a := Load([]float32{1, 2, 3, 4})
	b := Load([]float32{1, 5, 3, 7})
	mask := Equal(a, b)

	if !mask.GetBit(0) {
		t.Error("Equal: expected lane 0 to be true")
	}
	if mask.GetBit(1) {
		t.Error("Equal: expected lane 1 to be false")
	}
	if !mask.GetBit(2) {
		t.Error("Equal: expected lane 2 to be true")
	}
	if mask.GetBit(3) {
		t.Error("Equal: expected lane 3 to be false")
	}
}

func TestLessThan(t *testing.T) {
	a := Load([]float32{1, 5, 3, 7})
	b := Load([]float32{2, 4, 4, 6})
	mask := LessThan(a, b)

	if !mask.GetBit(0) {
		t.Error("LessThan: expected lane 0 to be true (1 < 2)")
	}
	if mask.GetBit(1) {
		t.Error("LessThan: expected lane 1 to be false (5 < 4)")
	}
}

func TestGreaterThan(t *testing.T) {
	a := Load([]float32{3, 5, 2, 8})
	b := Load([]float32{2, 6, 3, 7})
	mask := GreaterThan(a, b)

	if !mask.GetBit(0) {
		t.Error("GreaterThan: expected lane 0 to be true (3 > 2)")
	}
	if mask.GetBit(1) {
		t.Error("GreaterThan: expected lane 1 to be false (5 > 6)")
	}
}

func TestIfThenElse(t *testing.T) {
	mask := Equal(
		Load([]float32{1, 2, 3, 4}),
		Load([]float32{1, 0, 3, 0}),
	)
	a := Set[float32](100.0)
	b := Set[float32](200.0)
	result := IfThenElse(mask, a, b)

	if result.data[0] != 100.0 {
		t.Errorf("IfThenElse: lane 0: got %v, want 100.0", result.data[0])
	}
	if result.data[1] != 200.0 {
		t.Errorf("IfThenElse: lane 1: got %v, want 200.0", result.data[1])
	}
	if result.data[2] != 100.0 {
		t.Errorf("IfThenElse: lane 2: got %v, want 100.0", result.data[2])
	}
}

func TestMaskLoad(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	mask := TailMask[float32](3) // First 3 lanes active
	v := MaskLoad(mask, data)

	if v.data[0] != 1.0 {
		t.Errorf("MaskLoad: lane 0: got %v, want 1.0", v.data[0])
	}
	if v.data[1] != 2.0 {
		t.Errorf("MaskLoad: lane 1: got %v, want 2.0", v.data[1])
	}
	if v.data[2] != 3.0 {
		t.Errorf("MaskLoad: lane 2: got %v, want 3.0", v.data[2])
	}
	// Lanes 3+ should be zero
	for i := 3; i < v.NumLanes(); i++ {
		if v.data[i] != 0.0 {
			t.Errorf("MaskLoad: lane %d: got %v, want 0.0", i, v.data[i])
		}
	}
}

func TestMaskStore(t *testing.T) {
	v := Load([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	dst := make([]float32, 8)
	mask := TailMask[float32](3) // First 3 lanes active
	MaskStore(mask, v, dst)

	if dst[0] != 1.0 {
		t.Errorf("MaskStore: dst[0]: got %v, want 1.0", dst[0])
	}
	if dst[1] != 2.0 {
		t.Errorf("MaskStore: dst[1]: got %v, want 2.0", dst[1])
	}
	if dst[2] != 3.0 {
		t.Errorf("MaskStore: dst[2]: got %v, want 3.0", dst[2])
	}
	// dst[3:] should remain zero
	for i := 3; i < len(dst); i++ {
		if dst[i] != 0.0 {
			t.Errorf("MaskStore: dst[%d]: got %v, want 0.0", i, dst[i])
		}
	}
}

func TestMaskAllTrue(t *testing.T) {
	allTrue := TailMask[float32](MaxLanes[float32]())
	if !allTrue.AllTrue() {
		t.Error("AllTrue: expected true for full mask")
	}

	partial := TailMask[float32](2)
	if partial.AllTrue() {
		t.Error("AllTrue: expected false for partial mask")
	}
}

func TestMaskAnyTrue(t *testing.T) {
	partial := TailMask[float32](2)
	if !partial.AnyTrue() {
		t.Error("AnyTrue: expected true for partial mask")
	}

	empty := TailMask[float32](0)
	if empty.AnyTrue() {
		t.Error("AnyTrue: expected false for empty mask")
	}
}

func TestDispatch(t *testing.T) {
	level := CurrentLevel()
	width := CurrentWidth()
	name := CurrentName()

	t.Logf("Dispatch level: %v (%s), width: %d bytes", level, name, width)

	if width <= 0 {
		t.Error("CurrentWidth should be positive")
	}

	if name == "" {
		t.Error("CurrentName should not be empty")
	}
}

func TestMaxLanes(t *testing.T) {
	maxF32 := MaxLanes[float32]()
	maxF64 := MaxLanes[float64]()
	maxI32 := MaxLanes[int32]()

	t.Logf("MaxLanes: float32=%d, float64=%d, int32=%d", maxF32, maxF64, maxI32)

	if maxF32 <= 0 {
		t.Error("MaxLanes[float32] should be positive")
	}

	// float64 uses twice as much space, so should have half the lanes
	if maxF64*2 != maxF32 {
		t.Errorf("MaxLanes: expected float64 lanes (%d) to be half of float32 lanes (%d)", maxF64, maxF32)
	}
}

func TestTailMask(t *testing.T) {
	mask := TailMask[float32](3)

	if !mask.GetBit(0) || !mask.GetBit(1) || !mask.GetBit(2) {
		t.Error("TailMask: first 3 bits should be true")
	}

	for i := 3; i < mask.NumLanes(); i++ {
		if mask.GetBit(i) {
			t.Errorf("TailMask: bit %d should be false", i)
		}
	}
}

func TestProcessWithTail(t *testing.T) {
	data := make([]float32, 100)
	for i := range data {
		data[i] = float32(i)
	}

	output := make([]float32, len(data))

	fullVectors := 0

	ProcessWithTail[float32](len(data),
		func(offset int) {
			fullVectors++
			v := Load(data[offset:])
			result := Add(v, v) // Double each value
			Store(result, output[offset:])
		},
		func(offset, count int) {
			mask := TailMask[float32](count)
			v := MaskLoad(mask, data[offset:])
			result := Add(v, v)
			MaskStore(mask, result, output[offset:])
		},
	)

	if fullVectors == 0 {
		t.Error("ProcessWithTail: no full vectors processed")
	}

	// Verify results
	for i, val := range output {
		expected := float32(i) * 2
		if math.Abs(float64(val-expected)) > 0.001 {
			t.Errorf("ProcessWithTail: output[%d]: got %v, want %v", i, val, expected)
		}
	}
}

func TestAlignedSize(t *testing.T) {
	maxLanes := MaxLanes[float32]()

	tests := []struct {
		input    int
		expected int
	}{
		{0, 0},
		{1, maxLanes},
		{maxLanes, maxLanes},
		{maxLanes + 1, maxLanes * 2},
		{maxLanes * 2, maxLanes * 2},
	}

	for _, tt := range tests {
		result := AlignedSize[float32](tt.input)
		if result != tt.expected {
			t.Errorf("AlignedSize(%d): got %d, want %d", tt.input, result, tt.expected)
		}
	}
}

func TestIsAligned(t *testing.T) {
	maxLanes := MaxLanes[float32]()

	if !IsAligned[float32](0) {
		t.Error("IsAligned: 0 should be aligned")
	}

	if !IsAligned[float32](maxLanes) {
		t.Error("IsAligned: maxLanes should be aligned")
	}

	if !IsAligned[float32](maxLanes * 2) {
		t.Error("IsAligned: maxLanes*2 should be aligned")
	}

	if IsAligned[float32](maxLanes + 1) {
		t.Error("IsAligned: maxLanes+1 should not be aligned")
	}
}

func TestAnd(t *testing.T) {
	// Test with integers
	a := Load([]uint32{0xFF00FF00, 0xAAAAAAAA, 0x12345678, 0xFFFFFFFF})
	b := Load([]uint32{0x00FF00FF, 0x55555555, 0x87654321, 0x0F0F0F0F})
	result := And(a, b)

	expected := []uint32{0x00000000, 0x00000000, 0x02244220, 0x0F0F0F0F}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("And: lane %d: got 0x%08X, want 0x%08X", i, result.data[i], expected[i])
		}
	}

	// Test with floats (bitwise on representation)
	f1 := Set[float32](1.0)
	f2 := Set[float32](2.0)
	_ = And(f1, f2) // Just verify it doesn't panic
}

func TestOr(t *testing.T) {
	a := Load([]uint32{0xFF00FF00, 0xAAAAAAAA, 0x12345678, 0x00000000})
	b := Load([]uint32{0x00FF00FF, 0x55555555, 0x87654321, 0x0F0F0F0F})
	result := Or(a, b)

	expected := []uint32{0xFFFFFFFF, 0xFFFFFFFF, 0x97755779, 0x0F0F0F0F}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("Or: lane %d: got 0x%08X, want 0x%08X", i, result.data[i], expected[i])
		}
	}
}

func TestXor(t *testing.T) {
	a := Load([]uint32{0xFF00FF00, 0xAAAAAAAA, 0x12345678, 0xFFFFFFFF})
	b := Load([]uint32{0x00FF00FF, 0x55555555, 0x12345678, 0xFFFFFFFF})
	result := Xor(a, b)

	expected := []uint32{0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("Xor: lane %d: got 0x%08X, want 0x%08X", i, result.data[i], expected[i])
		}
	}
}

func TestNot(t *testing.T) {
	a := Load([]uint32{0x00000000, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555})
	result := Not(a)

	expected := []uint32{0xFFFFFFFF, 0x00000000, 0x55555555, 0xAAAAAAAA}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("Not: lane %d: got 0x%08X, want 0x%08X", i, result.data[i], expected[i])
		}
	}
}

func TestAndNot(t *testing.T) {
	a := Load([]uint32{0xFF00FF00, 0xAAAAAAAA, 0xFFFFFFFF, 0x00000000})
	b := Load([]uint32{0xFFFFFFFF, 0xFFFFFFFF, 0x0F0F0F0F, 0x0F0F0F0F})
	result := AndNot(a, b)

	// (~a) & b
	expected := []uint32{0x00FF00FF, 0x55555555, 0x00000000, 0x0F0F0F0F}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("AndNot: lane %d: got 0x%08X, want 0x%08X", i, result.data[i], expected[i])
		}
	}
}

func TestShiftLeft(t *testing.T) {
	a := Load([]uint32{1, 2, 3, 4})
	result := ShiftLeft(a, 2)

	expected := []uint32{4, 8, 12, 16}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("ShiftLeft: lane %d: got %d, want %d", i, result.data[i], expected[i])
		}
	}

	// Test with signed integers
	b := Load([]int32{-1, -2, -3, -4})
	result2 := ShiftLeft(b, 1)

	expected2 := []int32{-2, -4, -6, -8}
	for i := 0; i < len(expected2) && i < result2.NumLanes(); i++ {
		if result2.data[i] != expected2[i] {
			t.Errorf("ShiftLeft (signed): lane %d: got %d, want %d", i, result2.data[i], expected2[i])
		}
	}
}

func TestShiftRight(t *testing.T) {
	// Unsigned shift (logical)
	a := Load([]uint32{16, 8, 4, 2})
	result := ShiftRight(a, 2)

	expected := []uint32{4, 2, 1, 0}
	for i := 0; i < len(expected) && i < result.NumLanes(); i++ {
		if result.data[i] != expected[i] {
			t.Errorf("ShiftRight (unsigned): lane %d: got %d, want %d", i, result.data[i], expected[i])
		}
	}

	// Signed shift (arithmetic)
	b := Load([]int32{-16, -8, 8, 4})
	result2 := ShiftRight(b, 2)

	expected2 := []int32{-4, -2, 2, 1}
	for i := 0; i < len(expected2) && i < result2.NumLanes(); i++ {
		if result2.data[i] != expected2[i] {
			t.Errorf("ShiftRight (signed): lane %d: got %d, want %d", i, result2.data[i], expected2[i])
		}
	}
}

func TestIota(t *testing.T) {
	v := Iota[int32]()

	for i := 0; i < v.NumLanes(); i++ {
		if v.data[i] != int32(i) {
			t.Errorf("Iota: lane %d: got %d, want %d", i, v.data[i], i)
		}
	}

	// Test with float32
	vf := Iota[float32]()

	for i := 0; i < vf.NumLanes(); i++ {
		if vf.data[i] != float32(i) {
			t.Errorf("Iota (float32): lane %d: got %f, want %f", i, vf.data[i], float32(i))
		}
	}
}

func TestSignBit(t *testing.T) {
	// Test with float32
	vf32 := SignBit[float32]()
	for i := 0; i < vf32.NumLanes(); i++ {
		// Should be -0.0, which has sign bit set
		bits := math.Float32bits(vf32.data[i])
		if bits != 0x80000000 {
			t.Errorf("SignBit (float32): lane %d: got bits 0x%08X, want 0x80000000", i, bits)
		}
	}

	// Test with float64
	vf64 := SignBit[float64]()
	for i := 0; i < vf64.NumLanes(); i++ {
		bits := math.Float64bits(vf64.data[i])
		if bits != 0x8000000000000000 {
			t.Errorf("SignBit (float64): lane %d: got bits 0x%016X, want 0x8000000000000000", i, bits)
		}
	}

	// Test with signed integer
	vi32 := SignBit[int32]()
	for i := 0; i < vi32.NumLanes(); i++ {
		if vi32.data[i] != int32(-2147483648) {
			t.Errorf("SignBit (int32): lane %d: got %d, want %d", i, vi32.data[i], int32(-2147483648))
		}
	}

	// Test with unsigned integer
	vu32 := SignBit[uint32]()
	for i := 0; i < vu32.NumLanes(); i++ {
		if vu32.data[i] != uint32(0x80000000) {
			t.Errorf("SignBit (uint32): lane %d: got 0x%08X, want 0x80000000", i, vu32.data[i])
		}
	}
}

func TestReverse(t *testing.T) {
	a := Load([]int32{1, 2, 3, 4, 5, 6, 7, 8})
	result := Reverse(a)

	n := result.NumLanes()
	for i := range n {
		expected := int32(n - i)
		if i < len(a.data) {
			expected = a.data[n-1-i]
		}
		if result.data[i] != expected {
			t.Errorf("Reverse: lane %d: got %d, want %d", i, result.data[i], expected)
		}
	}
}

func TestBroadcast(t *testing.T) {
	a := Load([]float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0})

	// Broadcast lane 2 (value 3.0)
	result := Broadcast(a, 2)

	for i := 0; i < result.NumLanes(); i++ {
		if i < len(a.data) && result.data[i] != 3.0 {
			t.Errorf("Broadcast: lane %d: got %f, want 3.0", i, result.data[i])
		}
	}

	// Test out of bounds
	result2 := Broadcast(a, 100)
	for i := 0; i < result2.NumLanes(); i++ {
		if result2.data[i] != 0.0 {
			t.Errorf("Broadcast (out of bounds): lane %d: got %f, want 0.0", i, result2.data[i])
		}
	}
}
