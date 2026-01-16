package hwy

import (
	"math"
	"testing"
)

// TestBFloat16Constants verifies the predefined BFloat16 constants.
func TestBFloat16Constants(t *testing.T) {
	tests := []struct {
		name     string
		value    BFloat16
		expected float32
	}{
		{"Zero", BFloat16Zero, 0.0},
		{"One", BFloat16One, 1.0},
		{"NegOne", BFloat16NegOne, -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BFloat16ToFloat32(tt.value)
			if got != tt.expected {
				t.Errorf("BFloat16%s: got %v, want %v", tt.name, got, tt.expected)
			}
		})
	}

	// Test special values with dedicated checks
	t.Run("Infinity", func(t *testing.T) {
		if !BFloat16Inf.IsInf() || BFloat16Inf.IsNegative() {
			t.Error("BFloat16Inf should be positive infinity")
		}
	})

	t.Run("NegInfinity", func(t *testing.T) {
		if !BFloat16NegInf.IsInf() || !BFloat16NegInf.IsNegative() {
			t.Error("BFloat16NegInf should be negative infinity")
		}
	})

	t.Run("NaN", func(t *testing.T) {
		if !BFloat16NaN.IsNaN() {
			t.Error("BFloat16NaN should be NaN")
		}
	})

	t.Run("MaxValue", func(t *testing.T) {
		max := BFloat16ToFloat32(BFloat16MaxValue)
		// BFloat16 has same range as float32, max is approximately 3.39e38
		if max < 3e38 || max > float32(math.MaxFloat32) {
			t.Errorf("BFloat16MaxValue: got %v, expected ~3.39e38", max)
		}
	})
}

// TestBFloat16ToFloat32 tests conversion from BFloat16 to float32.
func TestBFloat16ToFloat32(t *testing.T) {
	tests := []struct {
		name     string
		input    BFloat16
		expected float32
	}{
		{"Zero", 0x0000, 0.0},
		{"NegZero", 0x8000, float32(math.Copysign(0, -1))},
		{"One", 0x3F80, 1.0},
		{"Two", 0x4000, 2.0},
		{"Half", 0x3F00, 0.5},
		{"NegOne", 0xBF80, -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BFloat16ToFloat32(tt.input)
			if got != tt.expected {
				t.Errorf("BFloat16ToFloat32(0x%04X): got %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}

// TestFloat32ToBFloat16 tests conversion from float32 to BFloat16.
func TestFloat32ToBFloat16(t *testing.T) {
	tests := []struct {
		name     string
		input    float32
		expected BFloat16
	}{
		{"Zero", 0.0, 0x0000},
		{"One", 1.0, 0x3F80},
		{"Two", 2.0, 0x4000},
		{"Half", 0.5, 0x3F00},
		{"NegOne", -1.0, 0xBF80},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Float32ToBFloat16(tt.input)
			if got != tt.expected {
				t.Errorf("Float32ToBFloat16(%v): got 0x%04X, want 0x%04X", tt.input, got, tt.expected)
			}
		})
	}
}

// TestBFloat16RoundTrip tests that round-trip conversion preserves values within precision.
func TestBFloat16RoundTrip(t *testing.T) {
	testValues := []float32{
		0.0, 1.0, -1.0, 0.5, -0.5,
		2.0, 4.0, 8.0, 16.0, 32.0,
		0.25, 0.125,
		100.0, 1000.0, 10000.0, 1e10, 1e20, 1e30,
	}

	for _, f := range testValues {
		b := Float32ToBFloat16(f)
		back := BFloat16ToFloat32(b)

		// BFloat16 has only 7 mantissa bits, so expect ~1% precision loss
		if f != 0 {
			relError := math.Abs(float64(back-f)) / math.Abs(float64(f))
			if relError > 0.01 {
				t.Errorf("Round-trip for %v: got %v, relative error %v", f, back, relError)
			}
		} else if back != 0 {
			t.Errorf("Round-trip for 0: got %v", back)
		}
	}
}

// TestBFloat16Infinity tests infinity handling.
func TestBFloat16Infinity(t *testing.T) {
	// Positive infinity
	posInf := Float32ToBFloat16(float32(math.Inf(1)))
	if !posInf.IsInf() || posInf.IsNegative() {
		t.Error("Float32ToBFloat16(+Inf) should be positive infinity")
	}
	if BFloat16ToFloat32(posInf) != float32(math.Inf(1)) {
		t.Error("BFloat16ToFloat32(BFloat16Inf) should return +Inf")
	}

	// Negative infinity
	negInf := Float32ToBFloat16(float32(math.Inf(-1)))
	if !negInf.IsInf() || !negInf.IsNegative() {
		t.Error("Float32ToBFloat16(-Inf) should be negative infinity")
	}
	if BFloat16ToFloat32(negInf) != float32(math.Inf(-1)) {
		t.Error("BFloat16ToFloat32(BFloat16NegInf) should return -Inf")
	}
}

// TestBFloat16NaN tests NaN handling.
func TestBFloat16NaN(t *testing.T) {
	// Convert NaN to BFloat16
	nan := Float32ToBFloat16(float32(math.NaN()))
	if !nan.IsNaN() {
		t.Error("Float32ToBFloat16(NaN) should be NaN")
	}

	// Convert back
	back := BFloat16ToFloat32(nan)
	if !math.IsNaN(float64(back)) {
		t.Error("BFloat16ToFloat32(NaN) should return NaN")
	}

	// Test that NaN != NaN
	nan1 := BFloat16NaN
	nan2 := BFloat16(0x7FC1) // Different NaN
	if !nan1.IsNaN() || !nan2.IsNaN() {
		t.Error("Both values should be NaN")
	}
}

// TestBFloat16Denormals tests denormalized number handling.
func TestBFloat16Denormals(t *testing.T) {
	// Smallest denormal
	minDenormal := BFloat16MinValue
	if !minDenormal.IsDenormal() {
		t.Error("BFloat16MinValue should be denormal")
	}

	// Convert to float32 and back
	f := BFloat16ToFloat32(minDenormal)
	if f <= 0 {
		t.Errorf("Smallest denormal should be positive, got %v", f)
	}

	// Smallest normal
	smallNormal := BFloat16ToFloat32(BFloat16MinNormal)
	if smallNormal <= 0 {
		t.Errorf("Smallest normal should be positive, got %v", smallNormal)
	}
}

// TestBFloat16Rounding tests round-to-nearest-even behavior.
func TestBFloat16Rounding(t *testing.T) {
	// Test that rounding works correctly
	// BFloat16 has 7 mantissa bits, so values that differ in lower bits should round

	// 1.0 should convert exactly
	one := Float32ToBFloat16(1.0)
	if BFloat16ToFloat32(one) != 1.0 {
		t.Error("1.0 should convert exactly")
	}

	// 1.0 + very small epsilon should still be close to 1.0
	eps := float32(1e-4)
	oneEps := Float32ToBFloat16(1.0 + eps)
	back := BFloat16ToFloat32(oneEps)
	if math.Abs(float64(back-1.0)) > 0.01 {
		t.Errorf("1.0+eps round-trip: got %v, expected ~1.0", back)
	}
}

// TestBFloat16Methods tests the helper methods on BFloat16.
func TestBFloat16Methods(t *testing.T) {
	t.Run("IsZero", func(t *testing.T) {
		if !BFloat16Zero.IsZero() {
			t.Error("BFloat16Zero.IsZero() should be true")
		}
		if !BFloat16NegZero.IsZero() {
			t.Error("BFloat16NegZero.IsZero() should be true")
		}
		if BFloat16One.IsZero() {
			t.Error("BFloat16One.IsZero() should be false")
		}
	})

	t.Run("IsNegative", func(t *testing.T) {
		if BFloat16Zero.IsNegative() {
			t.Error("BFloat16Zero should not be negative")
		}
		if !BFloat16NegZero.IsNegative() {
			t.Error("BFloat16NegZero should be negative")
		}
		if BFloat16One.IsNegative() {
			t.Error("BFloat16One should not be negative")
		}
		if !BFloat16NegOne.IsNegative() {
			t.Error("BFloat16NegOne should be negative")
		}
	})

	t.Run("Float32Method", func(t *testing.T) {
		if BFloat16One.Float32() != 1.0 {
			t.Error("BFloat16One.Float32() should be 1.0")
		}
	})

	t.Run("Float64Method", func(t *testing.T) {
		if BFloat16One.Float64() != 1.0 {
			t.Error("BFloat16One.Float64() should be 1.0")
		}
	})

	t.Run("Bits", func(t *testing.T) {
		if BFloat16One.Bits() != 0x3F80 {
			t.Errorf("BFloat16One.Bits() should be 0x3F80, got 0x%04X", BFloat16One.Bits())
		}
	})
}

// TestBFloat16Constructors tests the constructor functions.
func TestBFloat16Constructors(t *testing.T) {
	t.Run("NewBFloat16", func(t *testing.T) {
		b := NewBFloat16(1.0)
		if b != BFloat16One {
			t.Errorf("NewBFloat16(1.0): got 0x%04X, want 0x%04X", b, BFloat16One)
		}
	})

	t.Run("NewBFloat16FromFloat64", func(t *testing.T) {
		b := NewBFloat16FromFloat64(1.0)
		if b != BFloat16One {
			t.Errorf("NewBFloat16FromFloat64(1.0): got 0x%04X, want 0x%04X", b, BFloat16One)
		}
	})

	t.Run("BFloat16FromBits", func(t *testing.T) {
		b := BFloat16FromBits(0x3F80)
		if b != BFloat16One {
			t.Errorf("BFloat16FromBits(0x3F80): got 0x%04X, want 0x%04X", b, BFloat16One)
		}
	})
}

// TestFloat16ToBFloat16 tests cross-format conversion.
func TestFloat16ToBFloat16(t *testing.T) {
	testCases := []float32{0.0, 1.0, -1.0, 0.5, 2.0, 100.0, -50.0}

	for _, f := range testCases {
		f16 := Float32ToFloat16(f)
		bf16 := Float16ToBFloat16(f16)
		back := BFloat16ToFloat32(bf16)

		// Allow for some precision loss in conversion
		if math.Abs(float64(back-f)) > float64(math.Abs(float64(f)))*0.05+0.01 {
			t.Errorf("Float16ToBFloat16: %v -> %v (expected ~%v)", f, back, f)
		}
	}
}

// TestBFloat16ToFloat16 tests cross-format conversion.
func TestBFloat16ToFloat16(t *testing.T) {
	testCases := []float32{0.0, 1.0, -1.0, 0.5, 2.0, 100.0, -50.0}

	for _, f := range testCases {
		bf16 := Float32ToBFloat16(f)
		f16 := BFloat16ToFloat16(bf16)
		back := Float16ToFloat32(f16)

		// Allow for some precision loss in conversion
		if math.Abs(float64(back-f)) > float64(math.Abs(float64(f)))*0.05+0.01 {
			t.Errorf("BFloat16ToFloat16: %v -> %v (expected ~%v)", f, back, f)
		}
	}
}

// TestAddBF16 tests BFloat16 vector addition.
func TestAddBF16(t *testing.T) {
	a := SetBF16FromF32(2.0)
	b := SetBF16FromF32(3.0)
	result := AddBF16(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		got := BFloat16ToFloat32(result.data[i])
		if math.Abs(float64(got-5.0)) > 0.01 {
			t.Errorf("AddBF16: lane %d: got %v, want 5.0", i, got)
		}
	}
}

// TestSubBF16 tests BFloat16 vector subtraction.
func TestSubBF16(t *testing.T) {
	a := SetBF16FromF32(5.0)
	b := SetBF16FromF32(2.0)
	result := SubBF16(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		got := BFloat16ToFloat32(result.data[i])
		if math.Abs(float64(got-3.0)) > 0.01 {
			t.Errorf("SubBF16: lane %d: got %v, want 3.0", i, got)
		}
	}
}

// TestMulBF16 tests BFloat16 vector multiplication.
func TestMulBF16(t *testing.T) {
	a := SetBF16FromF32(3.0)
	b := SetBF16FromF32(4.0)
	result := MulBF16(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		got := BFloat16ToFloat32(result.data[i])
		if math.Abs(float64(got-12.0)) > 0.1 {
			t.Errorf("MulBF16: lane %d: got %v, want 12.0", i, got)
		}
	}
}

// TestDivBF16 tests BFloat16 vector division.
func TestDivBF16(t *testing.T) {
	a := SetBF16FromF32(12.0)
	b := SetBF16FromF32(3.0)
	result := DivBF16(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		got := BFloat16ToFloat32(result.data[i])
		if math.Abs(float64(got-4.0)) > 0.1 {
			t.Errorf("DivBF16: lane %d: got %v, want 4.0", i, got)
		}
	}
}

// TestFMABF16 tests BFloat16 fused multiply-add.
func TestFMABF16(t *testing.T) {
	a := SetBF16FromF32(2.0)
	b := SetBF16FromF32(3.0)
	c := SetBF16FromF32(4.0)
	result := FMABF16(a, b, c) // 2*3 + 4 = 10

	for i := 0; i < result.NumLanes(); i++ {
		got := BFloat16ToFloat32(result.data[i])
		if math.Abs(float64(got-10.0)) > 0.1 {
			t.Errorf("FMABF16: lane %d: got %v, want 10.0", i, got)
		}
	}
}

// TestNegBF16 tests BFloat16 negation.
func TestNegBF16(t *testing.T) {
	v := SetBF16FromF32(5.0)
	result := NegBF16(v)

	for i := 0; i < result.NumLanes(); i++ {
		got := BFloat16ToFloat32(result.data[i])
		if math.Abs(float64(got-(-5.0))) > 0.01 {
			t.Errorf("NegBF16: lane %d: got %v, want -5.0", i, got)
		}
	}
}

// TestAbsBF16 tests BFloat16 absolute value.
func TestAbsBF16(t *testing.T) {
	v := SetBF16FromF32(-7.0)
	result := AbsBF16(v)

	for i := 0; i < result.NumLanes(); i++ {
		got := BFloat16ToFloat32(result.data[i])
		if math.Abs(float64(got-7.0)) > 0.01 {
			t.Errorf("AbsBF16: lane %d: got %v, want 7.0", i, got)
		}
	}
}

// TestMinMaxBF16 tests BFloat16 min/max operations.
func TestMinMaxBF16(t *testing.T) {
	a := SetBF16FromF32(3.0)
	b := SetBF16FromF32(7.0)

	minResult := MinBF16(a, b)
	maxResult := MaxBF16(a, b)

	for i := 0; i < minResult.NumLanes(); i++ {
		gotMin := BFloat16ToFloat32(minResult.data[i])
		gotMax := BFloat16ToFloat32(maxResult.data[i])
		if math.Abs(float64(gotMin-3.0)) > 0.01 {
			t.Errorf("MinBF16: lane %d: got %v, want 3.0", i, gotMin)
		}
		if math.Abs(float64(gotMax-7.0)) > 0.01 {
			t.Errorf("MaxBF16: lane %d: got %v, want 7.0", i, gotMax)
		}
	}
}

// TestSqrtBF16 tests BFloat16 square root.
func TestSqrtBF16(t *testing.T) {
	v := SetBF16FromF32(16.0)
	result := SqrtBF16(v)

	for i := 0; i < result.NumLanes(); i++ {
		got := BFloat16ToFloat32(result.data[i])
		if math.Abs(float64(got-4.0)) > 0.1 {
			t.Errorf("SqrtBF16: lane %d: got %v, want 4.0", i, got)
		}
	}
}

// TestReciprocalBF16 tests BFloat16 reciprocal.
func TestReciprocalBF16(t *testing.T) {
	v := SetBF16FromF32(4.0)
	result := ReciprocalBF16(v)

	for i := 0; i < result.NumLanes(); i++ {
		got := BFloat16ToFloat32(result.data[i])
		if math.Abs(float64(got-0.25)) > 0.01 {
			t.Errorf("ReciprocalBF16: lane %d: got %v, want 0.25", i, got)
		}
	}
}

// TestReduceSumBF16 tests BFloat16 reduction sum.
func TestReduceSumBF16(t *testing.T) {
	// Create a vector with known values
	data := make([]BFloat16, MaxLanes[BFloat16]())
	for i := range data {
		data[i] = Float32ToBFloat16(float32(i + 1))
	}
	v := Vec[BFloat16]{data: data}

	sum := ReduceSumBF16(v)

	// Calculate expected sum
	var expected float32
	for i := range data {
		expected += float32(i + 1)
	}

	// BFloat16 has lower precision, allow for larger error
	if math.Abs(float64(sum-expected)) > float64(expected)*0.05 {
		t.Errorf("ReduceSumBF16: got %v, want %v", sum, expected)
	}
}

// TestReduceMinMaxBF16 tests BFloat16 min/max reductions.
func TestReduceMinMaxBF16(t *testing.T) {
	data := []BFloat16{
		Float32ToBFloat16(5.0),
		Float32ToBFloat16(2.0),
		Float32ToBFloat16(8.0),
		Float32ToBFloat16(1.0),
	}
	v := Vec[BFloat16]{data: data}

	min := ReduceMinBF16(v)
	max := ReduceMaxBF16(v)

	if BFloat16ToFloat32(min) != 1.0 {
		t.Errorf("ReduceMinBF16: got %v, want 1.0", BFloat16ToFloat32(min))
	}
	if BFloat16ToFloat32(max) != 8.0 {
		t.Errorf("ReduceMaxBF16: got %v, want 8.0", BFloat16ToFloat32(max))
	}
}

// TestDotBF16 tests BFloat16 dot product.
func TestDotBF16(t *testing.T) {
	a := Vec[BFloat16]{data: []BFloat16{
		Float32ToBFloat16(1.0),
		Float32ToBFloat16(2.0),
		Float32ToBFloat16(3.0),
		Float32ToBFloat16(4.0),
	}}
	b := Vec[BFloat16]{data: []BFloat16{
		Float32ToBFloat16(1.0),
		Float32ToBFloat16(1.0),
		Float32ToBFloat16(1.0),
		Float32ToBFloat16(1.0),
	}}

	dot := DotBF16(a, b)
	// 1*1 + 2*1 + 3*1 + 4*1 = 10
	if math.Abs(float64(dot-10.0)) > 0.5 {
		t.Errorf("DotBF16: got %v, want 10.0", dot)
	}
}

// TestComparisonBF16 tests BFloat16 comparison operations.
func TestComparisonBF16(t *testing.T) {
	a := Vec[BFloat16]{data: []BFloat16{
		Float32ToBFloat16(1.0),
		Float32ToBFloat16(5.0),
		Float32ToBFloat16(3.0),
		Float32ToBFloat16(7.0),
	}}
	b := Vec[BFloat16]{data: []BFloat16{
		Float32ToBFloat16(1.0),
		Float32ToBFloat16(3.0),
		Float32ToBFloat16(5.0),
		Float32ToBFloat16(7.0),
	}}

	t.Run("Equal", func(t *testing.T) {
		mask := EqualBF16(a, b)
		if !mask.GetBit(0) {
			t.Error("EqualBF16: lane 0 should be true (1.0 == 1.0)")
		}
		if mask.GetBit(1) {
			t.Error("EqualBF16: lane 1 should be false (5.0 != 3.0)")
		}
		if !mask.GetBit(3) {
			t.Error("EqualBF16: lane 3 should be true (7.0 == 7.0)")
		}
	})

	t.Run("LessThan", func(t *testing.T) {
		mask := LessThanBF16(a, b)
		if mask.GetBit(0) {
			t.Error("LessThanBF16: lane 0 should be false (1.0 < 1.0)")
		}
		if mask.GetBit(1) {
			t.Error("LessThanBF16: lane 1 should be false (5.0 < 3.0)")
		}
		if !mask.GetBit(2) {
			t.Error("LessThanBF16: lane 2 should be true (3.0 < 5.0)")
		}
	})

	t.Run("GreaterThan", func(t *testing.T) {
		mask := GreaterThanBF16(a, b)
		if mask.GetBit(0) {
			t.Error("GreaterThanBF16: lane 0 should be false (1.0 > 1.0)")
		}
		if !mask.GetBit(1) {
			t.Error("GreaterThanBF16: lane 1 should be true (5.0 > 3.0)")
		}
	})
}

// TestIsNaNBF16 tests BFloat16 NaN detection.
func TestIsNaNBF16(t *testing.T) {
	v := Vec[BFloat16]{data: []BFloat16{
		BFloat16One,
		BFloat16NaN,
		BFloat16Inf,
		BFloat16(0x7FC1), // Another NaN
	}}

	mask := IsNaNBF16(v)

	if mask.GetBit(0) {
		t.Error("IsNaNBF16: lane 0 should be false (1.0)")
	}
	if !mask.GetBit(1) {
		t.Error("IsNaNBF16: lane 1 should be true (NaN)")
	}
	if mask.GetBit(2) {
		t.Error("IsNaNBF16: lane 2 should be false (Inf)")
	}
	if !mask.GetBit(3) {
		t.Error("IsNaNBF16: lane 3 should be true (NaN)")
	}
}

// TestIsInfBF16 tests BFloat16 infinity detection.
func TestIsInfBF16(t *testing.T) {
	v := Vec[BFloat16]{data: []BFloat16{
		BFloat16One,
		BFloat16Inf,
		BFloat16NegInf,
		BFloat16NaN,
	}}

	t.Run("AnyInf", func(t *testing.T) {
		mask := IsInfBF16(v, 0)
		if mask.GetBit(0) {
			t.Error("lane 0 should be false (1.0)")
		}
		if !mask.GetBit(1) {
			t.Error("lane 1 should be true (+Inf)")
		}
		if !mask.GetBit(2) {
			t.Error("lane 2 should be true (-Inf)")
		}
		if mask.GetBit(3) {
			t.Error("lane 3 should be false (NaN)")
		}
	})

	t.Run("PosInf", func(t *testing.T) {
		mask := IsInfBF16(v, 1)
		if !mask.GetBit(1) {
			t.Error("lane 1 should be true (+Inf)")
		}
		if mask.GetBit(2) {
			t.Error("lane 2 should be false (for +Inf check)")
		}
	})

	t.Run("NegInf", func(t *testing.T) {
		mask := IsInfBF16(v, -1)
		if mask.GetBit(1) {
			t.Error("lane 1 should be false (for -Inf check)")
		}
		if !mask.GetBit(2) {
			t.Error("lane 2 should be true (-Inf)")
		}
	})
}

// TestIfThenElseBF16 tests BFloat16 conditional selection.
func TestIfThenElseBF16(t *testing.T) {
	a := Vec[BFloat16]{data: []BFloat16{
		Float32ToBFloat16(1.0),
		Float32ToBFloat16(2.0),
		Float32ToBFloat16(3.0),
		Float32ToBFloat16(4.0),
	}}
	b := Vec[BFloat16]{data: []BFloat16{
		Float32ToBFloat16(1.0),
		Float32ToBFloat16(1.0),
		Float32ToBFloat16(3.0),
		Float32ToBFloat16(1.0),
	}}

	mask := EqualBF16(a, b) // true, false, true, false
	yes := SetBF16FromF32(100.0)
	no := SetBF16FromF32(0.0)

	result := IfThenElseBF16(mask, yes, no)

	if BFloat16ToFloat32(result.data[0]) != 100.0 {
		t.Errorf("IfThenElseBF16: lane 0 should be 100.0 (mask true)")
	}
	if BFloat16ToFloat32(result.data[1]) != 0.0 {
		t.Errorf("IfThenElseBF16: lane 1 should be 0.0 (mask false)")
	}
	if BFloat16ToFloat32(result.data[2]) != 100.0 {
		t.Errorf("IfThenElseBF16: lane 2 should be 100.0 (mask true)")
	}
}

// TestPromoteDemoteBF16 tests BFloat16 promotion and demotion operations.
func TestPromoteDemoteBF16(t *testing.T) {
	t.Run("PromoteBF16ToF32", func(t *testing.T) {
		v := Vec[BFloat16]{data: []BFloat16{
			Float32ToBFloat16(1.0),
			Float32ToBFloat16(2.0),
			Float32ToBFloat16(3.0),
			Float32ToBFloat16(4.0),
		}}

		result := PromoteBF16ToF32(v)

		for i := 0; i < len(v.data); i++ {
			expected := float32(i + 1)
			if math.Abs(float64(result.data[i]-expected)) > 0.01 {
				t.Errorf("PromoteBF16ToF32: lane %d: got %v, want %v", i, result.data[i], expected)
			}
		}
	})

	t.Run("DemoteF32ToBF16", func(t *testing.T) {
		v := Vec[float32]{data: []float32{1.0, 2.0, 3.0, 4.0}}

		result := DemoteF32ToBF16(v)

		for i := 0; i < len(v.data); i++ {
			got := BFloat16ToFloat32(result.data[i])
			expected := float32(i + 1)
			if math.Abs(float64(got-expected)) > 0.01 {
				t.Errorf("DemoteF32ToBF16: lane %d: got %v, want %v", i, got, expected)
			}
		}
	})

	t.Run("PromoteLowerBF16ToF32", func(t *testing.T) {
		v := Vec[BFloat16]{data: []BFloat16{
			Float32ToBFloat16(1.0),
			Float32ToBFloat16(2.0),
			Float32ToBFloat16(3.0),
			Float32ToBFloat16(4.0),
		}}

		result := PromoteLowerBF16ToF32(v)

		// Should only contain lower half (1.0, 2.0)
		if len(result.data) != 2 {
			t.Errorf("PromoteLowerBF16ToF32: expected 2 elements, got %d", len(result.data))
		}
		if result.data[0] != 1.0 || result.data[1] != 2.0 {
			t.Errorf("PromoteLowerBF16ToF32: got %v, want [1.0, 2.0]", result.data)
		}
	})

	t.Run("PromoteUpperBF16ToF32", func(t *testing.T) {
		v := Vec[BFloat16]{data: []BFloat16{
			Float32ToBFloat16(1.0),
			Float32ToBFloat16(2.0),
			Float32ToBFloat16(3.0),
			Float32ToBFloat16(4.0),
		}}

		result := PromoteUpperBF16ToF32(v)

		// Should only contain upper half (3.0, 4.0)
		if len(result.data) != 2 {
			t.Errorf("PromoteUpperBF16ToF32: expected 2 elements, got %d", len(result.data))
		}
		if result.data[0] != 3.0 || result.data[1] != 4.0 {
			t.Errorf("PromoteUpperBF16ToF32: got %v, want [3.0, 4.0]", result.data)
		}
	})

	t.Run("DemoteTwoF32ToBF16", func(t *testing.T) {
		lo := Vec[float32]{data: []float32{1.0, 2.0}}
		hi := Vec[float32]{data: []float32{3.0, 4.0}}

		result := DemoteTwoF32ToBF16(lo, hi)

		if len(result.data) != 4 {
			t.Errorf("DemoteTwoF32ToBF16: expected 4 elements, got %d", len(result.data))
		}

		expected := []float32{1.0, 2.0, 3.0, 4.0}
		for i, exp := range expected {
			got := BFloat16ToFloat32(result.data[i])
			if math.Abs(float64(got-exp)) > 0.01 {
				t.Errorf("DemoteTwoF32ToBF16: lane %d: got %v, want %v", i, got, exp)
			}
		}
	})

	t.Run("PromoteBF16ToF64", func(t *testing.T) {
		v := Vec[BFloat16]{data: []BFloat16{
			Float32ToBFloat16(1.0),
			Float32ToBFloat16(2.0),
		}}

		result := PromoteBF16ToF64(v)

		if len(result.data) != 2 {
			t.Errorf("PromoteBF16ToF64: expected 2 elements, got %d", len(result.data))
		}
		if result.data[0] != 1.0 || result.data[1] != 2.0 {
			t.Errorf("PromoteBF16ToF64: got %v, want [1.0, 2.0]", result.data)
		}
	})

	t.Run("DemoteF64ToBF16", func(t *testing.T) {
		v := Vec[float64]{data: []float64{1.0, 2.0}}

		result := DemoteF64ToBF16(v)

		if len(result.data) != 2 {
			t.Errorf("DemoteF64ToBF16: expected 2 elements, got %d", len(result.data))
		}
		got0 := BFloat16ToFloat32(result.data[0])
		got1 := BFloat16ToFloat32(result.data[1])
		if got0 != 1.0 || got1 != 2.0 {
			t.Errorf("DemoteF64ToBF16: got [%v, %v], want [1.0, 2.0]", got0, got1)
		}
	})
}

// TestLoadStoreBF16 tests BFloat16 load and store operations.
func TestLoadStoreBF16(t *testing.T) {
	t.Run("LoadBF16", func(t *testing.T) {
		src := []uint16{0x3F80, 0x4000, 0x4040, 0x4080} // 1, 2, 3, 4
		v := LoadBF16(src)

		for i := 0; i < len(src) && i < v.NumLanes(); i++ {
			if v.data[i] != BFloat16(src[i]) {
				t.Errorf("LoadBF16: lane %d: got 0x%04X, want 0x%04X", i, v.data[i], src[i])
			}
		}
	})

	t.Run("StoreBF16", func(t *testing.T) {
		v := Vec[BFloat16]{data: []BFloat16{0x3F80, 0x4000, 0x4040, 0x4080}}
		dst := make([]uint16, 4)
		StoreBF16(v, dst)

		expected := []uint16{0x3F80, 0x4000, 0x4040, 0x4080}
		for i, exp := range expected {
			if dst[i] != exp {
				t.Errorf("StoreBF16: dst[%d]: got 0x%04X, want 0x%04X", i, dst[i], exp)
			}
		}
	})

	t.Run("LoadBF16FromF32", func(t *testing.T) {
		src := []float32{1.0, 2.0, 3.0, 4.0}
		v := LoadBF16FromF32(src)

		for i := 0; i < len(src) && i < v.NumLanes(); i++ {
			got := BFloat16ToFloat32(v.data[i])
			if math.Abs(float64(got-src[i])) > 0.01 {
				t.Errorf("LoadBF16FromF32: lane %d: got %v, want %v", i, got, src[i])
			}
		}
	})

	t.Run("StoreBF16ToF32", func(t *testing.T) {
		v := Vec[BFloat16]{data: []BFloat16{
			Float32ToBFloat16(1.0),
			Float32ToBFloat16(2.0),
			Float32ToBFloat16(3.0),
			Float32ToBFloat16(4.0),
		}}
		dst := make([]float32, 4)
		StoreBF16ToF32(v, dst)

		expected := []float32{1.0, 2.0, 3.0, 4.0}
		for i, exp := range expected {
			if math.Abs(float64(dst[i]-exp)) > 0.01 {
				t.Errorf("StoreBF16ToF32: dst[%d]: got %v, want %v", i, dst[i], exp)
			}
		}
	})
}

// TestSetZeroBF16 tests BFloat16 Set and Zero operations.
func TestSetZeroBF16(t *testing.T) {
	t.Run("SetBF16", func(t *testing.T) {
		v := SetBF16(BFloat16One)
		for i := 0; i < v.NumLanes(); i++ {
			if v.data[i] != BFloat16One {
				t.Errorf("SetBF16: lane %d: got 0x%04X, want 0x%04X", i, v.data[i], BFloat16One)
			}
		}
	})

	t.Run("SetBF16FromF32", func(t *testing.T) {
		v := SetBF16FromF32(2.0)
		for i := 0; i < v.NumLanes(); i++ {
			got := BFloat16ToFloat32(v.data[i])
			if math.Abs(float64(got-2.0)) > 0.01 {
				t.Errorf("SetBF16FromF32: lane %d: got %v, want 2.0", i, got)
			}
		}
	})

	t.Run("ZeroBF16", func(t *testing.T) {
		v := ZeroBF16()
		for i := 0; i < v.NumLanes(); i++ {
			if v.data[i] != BFloat16Zero {
				t.Errorf("ZeroBF16: lane %d: got 0x%04X, want 0x0000", i, v.data[i])
			}
		}
	})
}

// TestClampBF16 tests BFloat16 clamping.
func TestClampBF16(t *testing.T) {
	v := Vec[BFloat16]{data: []BFloat16{
		Float32ToBFloat16(0.0),
		Float32ToBFloat16(5.0),
		Float32ToBFloat16(15.0),
		Float32ToBFloat16(10.0),
	}}
	lo := SetBF16FromF32(2.0)
	hi := SetBF16FromF32(12.0)

	result := ClampBF16(v, lo, hi)

	expected := []float32{2.0, 5.0, 12.0, 10.0}
	for i, exp := range expected {
		got := BFloat16ToFloat32(result.data[i])
		if math.Abs(float64(got-exp)) > 0.01 {
			t.Errorf("ClampBF16: lane %d: got %v, want %v", i, got, exp)
		}
	}
}

// TestMulAddSubBF16 tests BFloat16 multiply-add and multiply-sub variants.
func TestMulAddSubBF16(t *testing.T) {
	a := SetBF16FromF32(2.0)
	b := SetBF16FromF32(3.0)
	c := SetBF16FromF32(4.0)

	t.Run("MulAddBF16", func(t *testing.T) {
		// 2*3 + 4 = 10
		result := MulAddBF16(a, b, c)
		got := BFloat16ToFloat32(result.data[0])
		if math.Abs(float64(got-10.0)) > 0.1 {
			t.Errorf("MulAddBF16: got %v, want 10.0", got)
		}
	})

	t.Run("MulSubBF16", func(t *testing.T) {
		// 2*3 - 4 = 2
		result := MulSubBF16(a, b, c)
		got := BFloat16ToFloat32(result.data[0])
		if math.Abs(float64(got-2.0)) > 0.1 {
			t.Errorf("MulSubBF16: got %v, want 2.0", got)
		}
	})

	t.Run("NegMulAddBF16", func(t *testing.T) {
		// -2*3 + 4 = -2
		result := NegMulAddBF16(a, b, c)
		got := BFloat16ToFloat32(result.data[0])
		if math.Abs(float64(got-(-2.0))) > 0.1 {
			t.Errorf("NegMulAddBF16: got %v, want -2.0", got)
		}
	})

	t.Run("NegMulSubBF16", func(t *testing.T) {
		// -(2*3 + 4) = -10
		result := NegMulSubBF16(a, b, c)
		got := BFloat16ToFloat32(result.data[0])
		if math.Abs(float64(got-(-10.0))) > 0.1 {
			t.Errorf("NegMulSubBF16: got %v, want -10.0", got)
		}
	})
}

// TestConvertF16BF16 tests cross-format conversion at vector level.
func TestConvertF16BF16(t *testing.T) {
	t.Run("ConvertF16ToBF16", func(t *testing.T) {
		v := Vec[Float16]{data: []Float16{
			Float32ToFloat16(1.0),
			Float32ToFloat16(2.0),
			Float32ToFloat16(3.0),
			Float32ToFloat16(4.0),
		}}

		result := ConvertF16ToBF16(v)

		for i := 0; i < len(v.data); i++ {
			got := BFloat16ToFloat32(result.data[i])
			expected := float32(i + 1)
			if math.Abs(float64(got-expected)) > 0.1 {
				t.Errorf("ConvertF16ToBF16: lane %d: got %v, want %v", i, got, expected)
			}
		}
	})

	t.Run("ConvertBF16ToF16", func(t *testing.T) {
		v := Vec[BFloat16]{data: []BFloat16{
			Float32ToBFloat16(1.0),
			Float32ToBFloat16(2.0),
			Float32ToBFloat16(3.0),
			Float32ToBFloat16(4.0),
		}}

		result := ConvertBF16ToF16(v)

		for i := 0; i < len(v.data); i++ {
			got := Float16ToFloat32(result.data[i])
			expected := float32(i + 1)
			if math.Abs(float64(got-expected)) > 0.1 {
				t.Errorf("ConvertBF16ToF16: lane %d: got %v, want %v", i, got, expected)
			}
		}
	})
}

// TestBFloat16LargeValues tests BFloat16 with values in float32 range but outside float16 range.
func TestBFloat16LargeValues(t *testing.T) {
	largeValues := []float32{1e10, 1e20, 1e30, -1e10, -1e20, -1e30}

	for _, f := range largeValues {
		b := Float32ToBFloat16(f)
		back := BFloat16ToFloat32(b)

		// Check relative error
		if f != 0 {
			relError := math.Abs(float64(back-f)) / math.Abs(float64(f))
			if relError > 0.01 {
				t.Errorf("Large value %v: got %v, relative error %v", f, back, relError)
			}
		}
	}
}
