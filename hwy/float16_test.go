package hwy

import (
	"math"
	"testing"
)

// TestFloat16Constants verifies the predefined Float16 constants.
func TestFloat16Constants(t *testing.T) {
	tests := []struct {
		name     string
		value    Float16
		expected float32
	}{
		{"Zero", Float16Zero, 0.0},
		{"One", Float16One, 1.0},
		{"NegOne", Float16NegOne, -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Float16ToFloat32(tt.value)
			if got != tt.expected {
				t.Errorf("Float16%s: got %v, want %v", tt.name, got, tt.expected)
			}
		})
	}

	// Test special values with dedicated checks
	t.Run("Infinity", func(t *testing.T) {
		if !Float16Inf.IsInf() || Float16Inf.IsNegative() {
			t.Error("Float16Inf should be positive infinity")
		}
	})

	t.Run("NegInfinity", func(t *testing.T) {
		if !Float16NegInf.IsInf() || !Float16NegInf.IsNegative() {
			t.Error("Float16NegInf should be negative infinity")
		}
	})

	t.Run("NaN", func(t *testing.T) {
		if !Float16NaN.IsNaN() {
			t.Error("Float16NaN should be NaN")
		}
	})

	t.Run("MaxValue", func(t *testing.T) {
		max := Float16ToFloat32(Float16MaxValue)
		if max != 65504.0 {
			t.Errorf("Float16MaxValue: got %v, want 65504", max)
		}
	})
}

// TestFloat16ToFloat32 tests conversion from Float16 to float32.
func TestFloat16ToFloat32(t *testing.T) {
	tests := []struct {
		name     string
		input    Float16
		expected float32
	}{
		{"Zero", 0x0000, 0.0},
		{"NegZero", 0x8000, float32(math.Copysign(0, -1))},
		{"One", 0x3C00, 1.0},
		{"Two", 0x4000, 2.0},
		{"Half", 0x3800, 0.5},
		{"NegOne", 0xBC00, -1.0},
		{"Pi", 0x4248, 3.140625}, // Closest representable to pi
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Float16ToFloat32(tt.input)
			if math.Abs(float64(got-tt.expected)) > 1e-6 {
				t.Errorf("Float16ToFloat32(0x%04X): got %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}

// TestFloat32ToFloat16 tests conversion from float32 to Float16.
func TestFloat32ToFloat16(t *testing.T) {
	tests := []struct {
		name     string
		input    float32
		expected Float16
	}{
		{"Zero", 0.0, 0x0000},
		{"One", 1.0, 0x3C00},
		{"Two", 2.0, 0x4000},
		{"Half", 0.5, 0x3800},
		{"NegOne", -1.0, 0xBC00},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Float32ToFloat16(tt.input)
			if got != tt.expected {
				t.Errorf("Float32ToFloat16(%v): got 0x%04X, want 0x%04X", tt.input, got, tt.expected)
			}
		})
	}
}

// TestFloat16RoundTrip tests that round-trip conversion preserves values.
func TestFloat16RoundTrip(t *testing.T) {
	testValues := []float32{
		0.0, 1.0, -1.0, 0.5, -0.5,
		2.0, 4.0, 8.0, 16.0, 32.0,
		0.25, 0.125, 0.0625,
		100.0, 1000.0, 10000.0, 65504.0, // Max float16 value
	}

	for _, f := range testValues {
		h := Float32ToFloat16(f)
		back := Float16ToFloat32(h)
		// Due to precision loss, we check if the round-trip is close
		if !math.IsInf(float64(f), 0) && math.Abs(float64(back-f)) > float64(math.Abs(float64(f)))*0.01+1e-5 {
			t.Errorf("Round-trip for %v: got %v (via 0x%04X)", f, back, h)
		}
	}
}

// TestFloat16Infinity tests infinity handling.
func TestFloat16Infinity(t *testing.T) {
	// Positive infinity
	posInf := Float32ToFloat16(float32(math.Inf(1)))
	if !posInf.IsInf() || posInf.IsNegative() {
		t.Error("Float32ToFloat16(+Inf) should be positive infinity")
	}
	if Float16ToFloat32(posInf) != float32(math.Inf(1)) {
		t.Error("Float16ToFloat32(Float16Inf) should return +Inf")
	}

	// Negative infinity
	negInf := Float32ToFloat16(float32(math.Inf(-1)))
	if !negInf.IsInf() || !negInf.IsNegative() {
		t.Error("Float32ToFloat16(-Inf) should be negative infinity")
	}
	if Float16ToFloat32(negInf) != float32(math.Inf(-1)) {
		t.Error("Float16ToFloat32(Float16NegInf) should return -Inf")
	}

	// Overflow to infinity
	overflow := Float32ToFloat16(100000.0) // Exceeds Float16 max
	if !overflow.IsInf() {
		t.Error("Large values should overflow to infinity")
	}
}

// TestFloat16NaN tests NaN handling.
func TestFloat16NaN(t *testing.T) {
	// Convert NaN to Float16
	nan := Float32ToFloat16(float32(math.NaN()))
	if !nan.IsNaN() {
		t.Error("Float32ToFloat16(NaN) should be NaN")
	}

	// Convert back
	back := Float16ToFloat32(nan)
	if !math.IsNaN(float64(back)) {
		t.Error("Float16ToFloat32(NaN) should return NaN")
	}

	// Test that NaN != NaN
	nan1 := Float16NaN
	nan2 := Float16(0x7E01) // Different NaN
	if !nan1.IsNaN() || !nan2.IsNaN() {
		t.Error("Both values should be NaN")
	}
}

// TestFloat16Denormals tests denormalized number handling.
func TestFloat16Denormals(t *testing.T) {
	// Smallest denormal
	minDenormal := Float16MinValue
	if !minDenormal.IsDenormal() {
		t.Error("Float16MinValue should be denormal")
	}

	// Convert to float32 and back
	f := Float16ToFloat32(minDenormal)
	if f <= 0 {
		t.Errorf("Smallest denormal should be positive, got %v", f)
	}

	// Just below smallest normal should still be representable
	smallNormal := Float16ToFloat32(Float16MinNormal)
	if smallNormal <= 0 {
		t.Errorf("Smallest normal should be positive, got %v", smallNormal)
	}
}

// TestFloat16Underflow tests underflow to zero.
func TestFloat16Underflow(t *testing.T) {
	// Very small float32 should underflow to zero
	verySmall := float32(1e-20)
	h := Float32ToFloat16(verySmall)
	if !h.IsZero() {
		t.Errorf("Very small value should underflow to zero, got 0x%04X", h)
	}
}

// TestFloat16Rounding tests round-to-nearest-even behavior.
func TestFloat16Rounding(t *testing.T) {
	// Test values that require rounding
	// 1.0 + small epsilon should round to 1.0 or the next representable value
	one := Float32ToFloat16(1.0)
	oneEps := Float32ToFloat16(1.0 + 1e-4)

	// Both should be close to 1.0
	oneBack := Float16ToFloat32(one)
	oneEpsBack := Float16ToFloat32(oneEps)

	if math.Abs(float64(oneBack-1.0)) > 0.001 {
		t.Errorf("1.0 round-trip failed: got %v", oneBack)
	}
	if math.Abs(float64(oneEpsBack-1.0)) > 0.01 {
		t.Errorf("1.0+eps round-trip failed: got %v", oneEpsBack)
	}
}

// TestFloat16Methods tests the helper methods on Float16.
func TestFloat16Methods(t *testing.T) {
	t.Run("IsZero", func(t *testing.T) {
		if !Float16Zero.IsZero() {
			t.Error("Float16Zero.IsZero() should be true")
		}
		if !Float16NegZero.IsZero() {
			t.Error("Float16NegZero.IsZero() should be true")
		}
		if Float16One.IsZero() {
			t.Error("Float16One.IsZero() should be false")
		}
	})

	t.Run("IsNegative", func(t *testing.T) {
		if Float16Zero.IsNegative() {
			t.Error("Float16Zero should not be negative")
		}
		if !Float16NegZero.IsNegative() {
			t.Error("Float16NegZero should be negative")
		}
		if Float16One.IsNegative() {
			t.Error("Float16One should not be negative")
		}
		if !Float16NegOne.IsNegative() {
			t.Error("Float16NegOne should be negative")
		}
	})

	t.Run("Float32Method", func(t *testing.T) {
		if Float16One.Float32() != 1.0 {
			t.Error("Float16One.Float32() should be 1.0")
		}
	})

	t.Run("Float64Method", func(t *testing.T) {
		if Float16One.Float64() != 1.0 {
			t.Error("Float16One.Float64() should be 1.0")
		}
	})

	t.Run("Bits", func(t *testing.T) {
		if Float16One.Bits() != 0x3C00 {
			t.Errorf("Float16One.Bits() should be 0x3C00, got 0x%04X", Float16One.Bits())
		}
	})
}

// TestFloat16Constructors tests the constructor functions.
func TestFloat16Constructors(t *testing.T) {
	t.Run("NewFloat16", func(t *testing.T) {
		h := NewFloat16(1.0)
		if h != Float16One {
			t.Errorf("NewFloat16(1.0): got 0x%04X, want 0x%04X", h, Float16One)
		}
	})

	t.Run("NewFloat16FromFloat64", func(t *testing.T) {
		h := NewFloat16FromFloat64(1.0)
		if h != Float16One {
			t.Errorf("NewFloat16FromFloat64(1.0): got 0x%04X, want 0x%04X", h, Float16One)
		}
	})

	t.Run("Float16FromBits", func(t *testing.T) {
		h := Float16FromBits(0x3C00)
		if h != Float16One {
			t.Errorf("Float16FromBits(0x3C00): got 0x%04X, want 0x%04X", h, Float16One)
		}
	})
}

// TestAddF16 tests Float16 vector addition.
func TestAddF16(t *testing.T) {
	a := SetF16FromF32(2.0)
	b := SetF16FromF32(3.0)
	result := AddF16(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		got := Float16ToFloat32(result.data[i])
		if math.Abs(float64(got-5.0)) > 0.01 {
			t.Errorf("AddF16: lane %d: got %v, want 5.0", i, got)
		}
	}
}

// TestSubF16 tests Float16 vector subtraction.
func TestSubF16(t *testing.T) {
	a := SetF16FromF32(5.0)
	b := SetF16FromF32(2.0)
	result := SubF16(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		got := Float16ToFloat32(result.data[i])
		if math.Abs(float64(got-3.0)) > 0.01 {
			t.Errorf("SubF16: lane %d: got %v, want 3.0", i, got)
		}
	}
}

// TestMulF16 tests Float16 vector multiplication.
func TestMulF16(t *testing.T) {
	a := SetF16FromF32(3.0)
	b := SetF16FromF32(4.0)
	result := MulF16(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		got := Float16ToFloat32(result.data[i])
		if math.Abs(float64(got-12.0)) > 0.01 {
			t.Errorf("MulF16: lane %d: got %v, want 12.0", i, got)
		}
	}
}

// TestDivF16 tests Float16 vector division.
func TestDivF16(t *testing.T) {
	a := SetF16FromF32(12.0)
	b := SetF16FromF32(3.0)
	result := DivF16(a, b)

	for i := 0; i < result.NumLanes(); i++ {
		got := Float16ToFloat32(result.data[i])
		if math.Abs(float64(got-4.0)) > 0.01 {
			t.Errorf("DivF16: lane %d: got %v, want 4.0", i, got)
		}
	}
}

// TestFMAF16 tests Float16 fused multiply-add.
func TestFMAF16(t *testing.T) {
	a := SetF16FromF32(2.0)
	b := SetF16FromF32(3.0)
	c := SetF16FromF32(4.0)
	result := FMAF16(a, b, c) // 2*3 + 4 = 10

	for i := 0; i < result.NumLanes(); i++ {
		got := Float16ToFloat32(result.data[i])
		if math.Abs(float64(got-10.0)) > 0.01 {
			t.Errorf("FMAF16: lane %d: got %v, want 10.0", i, got)
		}
	}
}

// TestNegF16 tests Float16 negation.
func TestNegF16(t *testing.T) {
	v := SetF16FromF32(5.0)
	result := NegF16(v)

	for i := 0; i < result.NumLanes(); i++ {
		got := Float16ToFloat32(result.data[i])
		if math.Abs(float64(got-(-5.0))) > 0.01 {
			t.Errorf("NegF16: lane %d: got %v, want -5.0", i, got)
		}
	}
}

// TestAbsF16 tests Float16 absolute value.
func TestAbsF16(t *testing.T) {
	v := SetF16FromF32(-7.0)
	result := AbsF16(v)

	for i := 0; i < result.NumLanes(); i++ {
		got := Float16ToFloat32(result.data[i])
		if math.Abs(float64(got-7.0)) > 0.01 {
			t.Errorf("AbsF16: lane %d: got %v, want 7.0", i, got)
		}
	}
}

// TestMinMaxF16 tests Float16 min/max operations.
func TestMinMaxF16(t *testing.T) {
	a := SetF16FromF32(3.0)
	b := SetF16FromF32(7.0)

	minResult := MinF16(a, b)
	maxResult := MaxF16(a, b)

	for i := 0; i < minResult.NumLanes(); i++ {
		gotMin := Float16ToFloat32(minResult.data[i])
		gotMax := Float16ToFloat32(maxResult.data[i])
		if math.Abs(float64(gotMin-3.0)) > 0.01 {
			t.Errorf("MinF16: lane %d: got %v, want 3.0", i, gotMin)
		}
		if math.Abs(float64(gotMax-7.0)) > 0.01 {
			t.Errorf("MaxF16: lane %d: got %v, want 7.0", i, gotMax)
		}
	}
}

// TestSqrtF16 tests Float16 square root.
func TestSqrtF16(t *testing.T) {
	v := SetF16FromF32(16.0)
	result := SqrtF16(v)

	for i := 0; i < result.NumLanes(); i++ {
		got := Float16ToFloat32(result.data[i])
		if math.Abs(float64(got-4.0)) > 0.01 {
			t.Errorf("SqrtF16: lane %d: got %v, want 4.0", i, got)
		}
	}
}

// TestReciprocalF16 tests Float16 reciprocal.
func TestReciprocalF16(t *testing.T) {
	v := SetF16FromF32(4.0)
	result := ReciprocalF16(v)

	for i := 0; i < result.NumLanes(); i++ {
		got := Float16ToFloat32(result.data[i])
		if math.Abs(float64(got-0.25)) > 0.01 {
			t.Errorf("ReciprocalF16: lane %d: got %v, want 0.25", i, got)
		}
	}
}

// TestReduceSumF16 tests Float16 reduction sum.
func TestReduceSumF16(t *testing.T) {
	// Create a vector with known values
	data := make([]Float16, MaxLanes[Float16]())
	for i := range data {
		data[i] = Float32ToFloat16(float32(i + 1))
	}
	v := Vec[Float16]{data: data}

	sum := ReduceSumF16(v)

	// Calculate expected sum
	var expected float32
	for i := range data {
		expected += float32(i + 1)
	}

	if math.Abs(float64(sum-expected)) > 0.1 {
		t.Errorf("ReduceSumF16: got %v, want %v", sum, expected)
	}
}

// TestReduceMinMaxF16 tests Float16 min/max reductions.
func TestReduceMinMaxF16(t *testing.T) {
	data := []Float16{
		Float32ToFloat16(5.0),
		Float32ToFloat16(2.0),
		Float32ToFloat16(8.0),
		Float32ToFloat16(1.0),
	}
	v := Vec[Float16]{data: data}

	min := ReduceMinF16(v)
	max := ReduceMaxF16(v)

	if Float16ToFloat32(min) != 1.0 {
		t.Errorf("ReduceMinF16: got %v, want 1.0", Float16ToFloat32(min))
	}
	if Float16ToFloat32(max) != 8.0 {
		t.Errorf("ReduceMaxF16: got %v, want 8.0", Float16ToFloat32(max))
	}
}

// TestDotF16 tests Float16 dot product.
func TestDotF16(t *testing.T) {
	a := Vec[Float16]{data: []Float16{
		Float32ToFloat16(1.0),
		Float32ToFloat16(2.0),
		Float32ToFloat16(3.0),
		Float32ToFloat16(4.0),
	}}
	b := Vec[Float16]{data: []Float16{
		Float32ToFloat16(1.0),
		Float32ToFloat16(1.0),
		Float32ToFloat16(1.0),
		Float32ToFloat16(1.0),
	}}

	dot := DotF16(a, b)
	// 1*1 + 2*1 + 3*1 + 4*1 = 10
	if math.Abs(float64(dot-10.0)) > 0.1 {
		t.Errorf("DotF16: got %v, want 10.0", dot)
	}
}

// TestComparisonF16 tests Float16 comparison operations.
func TestComparisonF16(t *testing.T) {
	a := Vec[Float16]{data: []Float16{
		Float32ToFloat16(1.0),
		Float32ToFloat16(5.0),
		Float32ToFloat16(3.0),
		Float32ToFloat16(7.0),
	}}
	b := Vec[Float16]{data: []Float16{
		Float32ToFloat16(1.0),
		Float32ToFloat16(3.0),
		Float32ToFloat16(5.0),
		Float32ToFloat16(7.0),
	}}

	t.Run("Equal", func(t *testing.T) {
		mask := EqualF16(a, b)
		if !mask.GetBit(0) {
			t.Error("EqualF16: lane 0 should be true (1.0 == 1.0)")
		}
		if mask.GetBit(1) {
			t.Error("EqualF16: lane 1 should be false (5.0 != 3.0)")
		}
		if !mask.GetBit(3) {
			t.Error("EqualF16: lane 3 should be true (7.0 == 7.0)")
		}
	})

	t.Run("LessThan", func(t *testing.T) {
		mask := LessThanF16(a, b)
		if mask.GetBit(0) {
			t.Error("LessThanF16: lane 0 should be false (1.0 < 1.0)")
		}
		if mask.GetBit(1) {
			t.Error("LessThanF16: lane 1 should be false (5.0 < 3.0)")
		}
		if !mask.GetBit(2) {
			t.Error("LessThanF16: lane 2 should be true (3.0 < 5.0)")
		}
	})

	t.Run("GreaterThan", func(t *testing.T) {
		mask := GreaterThanF16(a, b)
		if mask.GetBit(0) {
			t.Error("GreaterThanF16: lane 0 should be false (1.0 > 1.0)")
		}
		if !mask.GetBit(1) {
			t.Error("GreaterThanF16: lane 1 should be true (5.0 > 3.0)")
		}
	})
}

// TestIsNaNF16 tests Float16 NaN detection.
func TestIsNaNF16(t *testing.T) {
	v := Vec[Float16]{data: []Float16{
		Float16One,
		Float16NaN,
		Float16Inf,
		Float16(0x7E01), // Another NaN
	}}

	mask := IsNaNF16(v)

	if mask.GetBit(0) {
		t.Error("IsNaNF16: lane 0 should be false (1.0)")
	}
	if !mask.GetBit(1) {
		t.Error("IsNaNF16: lane 1 should be true (NaN)")
	}
	if mask.GetBit(2) {
		t.Error("IsNaNF16: lane 2 should be false (Inf)")
	}
	if !mask.GetBit(3) {
		t.Error("IsNaNF16: lane 3 should be true (NaN)")
	}
}

// TestIsInfF16 tests Float16 infinity detection.
func TestIsInfF16(t *testing.T) {
	v := Vec[Float16]{data: []Float16{
		Float16One,
		Float16Inf,
		Float16NegInf,
		Float16NaN,
	}}

	t.Run("AnyInf", func(t *testing.T) {
		mask := IsInfF16(v, 0)
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
		mask := IsInfF16(v, 1)
		if !mask.GetBit(1) {
			t.Error("lane 1 should be true (+Inf)")
		}
		if mask.GetBit(2) {
			t.Error("lane 2 should be false (for +Inf check)")
		}
	})

	t.Run("NegInf", func(t *testing.T) {
		mask := IsInfF16(v, -1)
		if mask.GetBit(1) {
			t.Error("lane 1 should be false (for -Inf check)")
		}
		if !mask.GetBit(2) {
			t.Error("lane 2 should be true (-Inf)")
		}
	})
}

// TestIfThenElseF16 tests Float16 conditional selection.
func TestIfThenElseF16(t *testing.T) {
	a := Vec[Float16]{data: []Float16{
		Float32ToFloat16(1.0),
		Float32ToFloat16(2.0),
		Float32ToFloat16(3.0),
		Float32ToFloat16(4.0),
	}}
	b := Vec[Float16]{data: []Float16{
		Float32ToFloat16(1.0),
		Float32ToFloat16(1.0),
		Float32ToFloat16(3.0),
		Float32ToFloat16(1.0),
	}}

	mask := EqualF16(a, b) // true, false, true, false
	yes := SetF16FromF32(100.0)
	no := SetF16FromF32(0.0)

	result := IfThenElseF16(mask, yes, no)

	if Float16ToFloat32(result.data[0]) != 100.0 {
		t.Errorf("IfThenElseF16: lane 0 should be 100.0 (mask true)")
	}
	if Float16ToFloat32(result.data[1]) != 0.0 {
		t.Errorf("IfThenElseF16: lane 1 should be 0.0 (mask false)")
	}
	if Float16ToFloat32(result.data[2]) != 100.0 {
		t.Errorf("IfThenElseF16: lane 2 should be 100.0 (mask true)")
	}
}

// TestPromoteDemoteF16 tests Float16 promotion and demotion operations.
func TestPromoteDemoteF16(t *testing.T) {
	t.Run("PromoteF16ToF32", func(t *testing.T) {
		v := Vec[Float16]{data: []Float16{
			Float32ToFloat16(1.0),
			Float32ToFloat16(2.0),
			Float32ToFloat16(3.0),
			Float32ToFloat16(4.0),
		}}

		result := PromoteF16ToF32(v)

		for i := 0; i < len(v.data); i++ {
			expected := float32(i + 1)
			if math.Abs(float64(result.data[i]-expected)) > 0.01 {
				t.Errorf("PromoteF16ToF32: lane %d: got %v, want %v", i, result.data[i], expected)
			}
		}
	})

	t.Run("DemoteF32ToF16", func(t *testing.T) {
		v := Vec[float32]{data: []float32{1.0, 2.0, 3.0, 4.0}}

		result := DemoteF32ToF16(v)

		for i := 0; i < len(v.data); i++ {
			got := Float16ToFloat32(result.data[i])
			expected := float32(i + 1)
			if math.Abs(float64(got-expected)) > 0.01 {
				t.Errorf("DemoteF32ToF16: lane %d: got %v, want %v", i, got, expected)
			}
		}
	})

	t.Run("PromoteLowerF16ToF32", func(t *testing.T) {
		v := Vec[Float16]{data: []Float16{
			Float32ToFloat16(1.0),
			Float32ToFloat16(2.0),
			Float32ToFloat16(3.0),
			Float32ToFloat16(4.0),
		}}

		result := PromoteLowerF16ToF32(v)

		// Should only contain lower half (1.0, 2.0)
		if len(result.data) != 2 {
			t.Errorf("PromoteLowerF16ToF32: expected 2 elements, got %d", len(result.data))
		}
		if result.data[0] != 1.0 || result.data[1] != 2.0 {
			t.Errorf("PromoteLowerF16ToF32: got %v, want [1.0, 2.0]", result.data)
		}
	})

	t.Run("PromoteUpperF16ToF32", func(t *testing.T) {
		v := Vec[Float16]{data: []Float16{
			Float32ToFloat16(1.0),
			Float32ToFloat16(2.0),
			Float32ToFloat16(3.0),
			Float32ToFloat16(4.0),
		}}

		result := PromoteUpperF16ToF32(v)

		// Should only contain upper half (3.0, 4.0)
		if len(result.data) != 2 {
			t.Errorf("PromoteUpperF16ToF32: expected 2 elements, got %d", len(result.data))
		}
		if result.data[0] != 3.0 || result.data[1] != 4.0 {
			t.Errorf("PromoteUpperF16ToF32: got %v, want [3.0, 4.0]", result.data)
		}
	})

	t.Run("DemoteTwoF32ToF16", func(t *testing.T) {
		lo := Vec[float32]{data: []float32{1.0, 2.0}}
		hi := Vec[float32]{data: []float32{3.0, 4.0}}

		result := DemoteTwoF32ToF16(lo, hi)

		if len(result.data) != 4 {
			t.Errorf("DemoteTwoF32ToF16: expected 4 elements, got %d", len(result.data))
		}

		expected := []float32{1.0, 2.0, 3.0, 4.0}
		for i, exp := range expected {
			got := Float16ToFloat32(result.data[i])
			if math.Abs(float64(got-exp)) > 0.01 {
				t.Errorf("DemoteTwoF32ToF16: lane %d: got %v, want %v", i, got, exp)
			}
		}
	})
}

// TestLoadStoreF16 tests Float16 load and store operations.
func TestLoadStoreF16(t *testing.T) {
	t.Run("LoadF16", func(t *testing.T) {
		src := []uint16{0x3C00, 0x4000, 0x4200, 0x4400} // 1, 2, 3, 4
		v := LoadF16(src)

		for i := 0; i < len(src) && i < v.NumLanes(); i++ {
			if v.data[i] != Float16(src[i]) {
				t.Errorf("LoadF16: lane %d: got 0x%04X, want 0x%04X", i, v.data[i], src[i])
			}
		}
	})

	t.Run("StoreF16", func(t *testing.T) {
		v := Vec[Float16]{data: []Float16{0x3C00, 0x4000, 0x4200, 0x4400}}
		dst := make([]uint16, 4)
		StoreF16(v, dst)

		expected := []uint16{0x3C00, 0x4000, 0x4200, 0x4400}
		for i, exp := range expected {
			if dst[i] != exp {
				t.Errorf("StoreF16: dst[%d]: got 0x%04X, want 0x%04X", i, dst[i], exp)
			}
		}
	})

	t.Run("LoadF16FromF32", func(t *testing.T) {
		src := []float32{1.0, 2.0, 3.0, 4.0}
		v := LoadF16FromF32(src)

		for i := 0; i < len(src) && i < v.NumLanes(); i++ {
			got := Float16ToFloat32(v.data[i])
			if math.Abs(float64(got-src[i])) > 0.01 {
				t.Errorf("LoadF16FromF32: lane %d: got %v, want %v", i, got, src[i])
			}
		}
	})

	t.Run("StoreF16ToF32", func(t *testing.T) {
		v := Vec[Float16]{data: []Float16{
			Float32ToFloat16(1.0),
			Float32ToFloat16(2.0),
			Float32ToFloat16(3.0),
			Float32ToFloat16(4.0),
		}}
		dst := make([]float32, 4)
		StoreF16ToF32(v, dst)

		expected := []float32{1.0, 2.0, 3.0, 4.0}
		for i, exp := range expected {
			if math.Abs(float64(dst[i]-exp)) > 0.01 {
				t.Errorf("StoreF16ToF32: dst[%d]: got %v, want %v", i, dst[i], exp)
			}
		}
	})
}

// TestSetZeroF16 tests Float16 Set and Zero operations.
func TestSetZeroF16(t *testing.T) {
	t.Run("SetF16", func(t *testing.T) {
		v := SetF16(Float16One)
		for i := 0; i < v.NumLanes(); i++ {
			if v.data[i] != Float16One {
				t.Errorf("SetF16: lane %d: got 0x%04X, want 0x%04X", i, v.data[i], Float16One)
			}
		}
	})

	t.Run("SetF16FromF32", func(t *testing.T) {
		v := SetF16FromF32(2.0)
		for i := 0; i < v.NumLanes(); i++ {
			got := Float16ToFloat32(v.data[i])
			if math.Abs(float64(got-2.0)) > 0.01 {
				t.Errorf("SetF16FromF32: lane %d: got %v, want 2.0", i, got)
			}
		}
	})

	t.Run("ZeroF16", func(t *testing.T) {
		v := ZeroF16()
		for i := 0; i < v.NumLanes(); i++ {
			if v.data[i] != Float16Zero {
				t.Errorf("ZeroF16: lane %d: got 0x%04X, want 0x0000", i, v.data[i])
			}
		}
	})
}

// TestClampF16 tests Float16 clamping.
func TestClampF16(t *testing.T) {
	v := Vec[Float16]{data: []Float16{
		Float32ToFloat16(0.0),
		Float32ToFloat16(5.0),
		Float32ToFloat16(15.0),
		Float32ToFloat16(10.0),
	}}
	lo := SetF16FromF32(2.0)
	hi := SetF16FromF32(12.0)

	result := ClampF16(v, lo, hi)

	expected := []float32{2.0, 5.0, 12.0, 10.0}
	for i, exp := range expected {
		got := Float16ToFloat32(result.data[i])
		if math.Abs(float64(got-exp)) > 0.01 {
			t.Errorf("ClampF16: lane %d: got %v, want %v", i, got, exp)
		}
	}
}

// TestMulAddSubF16 tests Float16 multiply-add and multiply-sub variants.
func TestMulAddSubF16(t *testing.T) {
	a := SetF16FromF32(2.0)
	b := SetF16FromF32(3.0)
	c := SetF16FromF32(4.0)

	t.Run("MulAddF16", func(t *testing.T) {
		// 2*3 + 4 = 10
		result := MulAddF16(a, b, c)
		got := Float16ToFloat32(result.data[0])
		if math.Abs(float64(got-10.0)) > 0.01 {
			t.Errorf("MulAddF16: got %v, want 10.0", got)
		}
	})

	t.Run("MulSubF16", func(t *testing.T) {
		// 2*3 - 4 = 2
		result := MulSubF16(a, b, c)
		got := Float16ToFloat32(result.data[0])
		if math.Abs(float64(got-2.0)) > 0.01 {
			t.Errorf("MulSubF16: got %v, want 2.0", got)
		}
	})

	t.Run("NegMulAddF16", func(t *testing.T) {
		// -2*3 + 4 = -2
		result := NegMulAddF16(a, b, c)
		got := Float16ToFloat32(result.data[0])
		if math.Abs(float64(got-(-2.0))) > 0.01 {
			t.Errorf("NegMulAddF16: got %v, want -2.0", got)
		}
	})

	t.Run("NegMulSubF16", func(t *testing.T) {
		// -(2*3 + 4) = -10
		result := NegMulSubF16(a, b, c)
		got := Float16ToFloat32(result.data[0])
		if math.Abs(float64(got-(-10.0))) > 0.01 {
			t.Errorf("NegMulSubF16: got %v, want -10.0", got)
		}
	})
}
