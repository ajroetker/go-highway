package algo

import "github.com/ajroetker/go-highway/hwy"

// Predicate defines an operation that can test individual values or vectors.
// Implementations can be specialized by hwygen for different SIMD targets.
type Predicate[T hwy.Lanes] interface {
	// Test returns true if the scalar value satisfies the predicate.
	// Used by scalar fallback code to avoid allocations.
	Test(value T) bool

	// Apply returns a mask indicating which lanes satisfy the predicate.
	// Used by SIMD code paths.
	Apply(v hwy.Vec[T]) hwy.Mask[T]
}

// Preparable is an optional interface for predicates that can pre-compute
// comparison vectors for better performance in loops.
type Preparable[T hwy.Lanes] interface {
	Predicate[T]
	// Prepare returns a version of this predicate with pre-computed comparison vectors.
	// This should be called once before a loop to avoid repeated allocations.
	Prepare() Predicate[T]
}

// GreaterThan returns true for values where v > threshold.
type GreaterThan[T hwy.Lanes] struct {
	Threshold T
}

func (p GreaterThan[T]) Test(value T) bool {
	return value > p.Threshold
}

func (p GreaterThan[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.Greater(v, hwy.Set(p.Threshold))
}

func (p GreaterThan[T]) Prepare() Predicate[T] {
	return preparedGreaterThan[T]{threshold: p.Threshold, thresholdVec: hwy.Set(p.Threshold)}
}

type preparedGreaterThan[T hwy.Lanes] struct {
	threshold    T
	thresholdVec hwy.Vec[T]
}

func (p preparedGreaterThan[T]) Test(value T) bool {
	return value > p.threshold
}

func (p preparedGreaterThan[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.Greater(v, p.thresholdVec)
}

// LessThan returns true for values where v < threshold.
type LessThan[T hwy.Lanes] struct {
	Threshold T
}

func (p LessThan[T]) Test(value T) bool {
	return value < p.Threshold
}

func (p LessThan[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.Less(v, hwy.Set(p.Threshold))
}

func (p LessThan[T]) Prepare() Predicate[T] {
	return preparedLessThan[T]{threshold: p.Threshold, thresholdVec: hwy.Set(p.Threshold)}
}

type preparedLessThan[T hwy.Lanes] struct {
	threshold    T
	thresholdVec hwy.Vec[T]
}

func (p preparedLessThan[T]) Test(value T) bool {
	return value < p.threshold
}

func (p preparedLessThan[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.Less(v, p.thresholdVec)
}

// GreaterEqual returns true for values where v >= threshold.
type GreaterEqual[T hwy.Lanes] struct {
	Threshold T
}

func (p GreaterEqual[T]) Test(value T) bool {
	return value >= p.Threshold
}

func (p GreaterEqual[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.GreaterEqual(v, hwy.Set(p.Threshold))
}

func (p GreaterEqual[T]) Prepare() Predicate[T] {
	return preparedGreaterEqual[T]{threshold: p.Threshold, thresholdVec: hwy.Set(p.Threshold)}
}

type preparedGreaterEqual[T hwy.Lanes] struct {
	threshold    T
	thresholdVec hwy.Vec[T]
}

func (p preparedGreaterEqual[T]) Test(value T) bool {
	return value >= p.threshold
}

func (p preparedGreaterEqual[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.GreaterEqual(v, p.thresholdVec)
}

// LessEqual returns true for values where v <= threshold.
type LessEqual[T hwy.Lanes] struct {
	Threshold T
}

func (p LessEqual[T]) Test(value T) bool {
	return value <= p.Threshold
}

func (p LessEqual[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.LessEqual(v, hwy.Set(p.Threshold))
}

func (p LessEqual[T]) Prepare() Predicate[T] {
	return preparedLessEqual[T]{threshold: p.Threshold, thresholdVec: hwy.Set(p.Threshold)}
}

type preparedLessEqual[T hwy.Lanes] struct {
	threshold    T
	thresholdVec hwy.Vec[T]
}

func (p preparedLessEqual[T]) Test(value T) bool {
	return value <= p.threshold
}

func (p preparedLessEqual[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.LessEqual(v, p.thresholdVec)
}

// Equal returns true for values where v == value.
type Equal[T hwy.Lanes] struct {
	Value T
}

func (p Equal[T]) Test(value T) bool {
	return value == p.Value
}

func (p Equal[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.Equal(v, hwy.Set(p.Value))
}

func (p Equal[T]) Prepare() Predicate[T] {
	return preparedEqual[T]{value: p.Value, valueVec: hwy.Set(p.Value)}
}

type preparedEqual[T hwy.Lanes] struct {
	value    T
	valueVec hwy.Vec[T]
}

func (p preparedEqual[T]) Test(value T) bool {
	return value == p.value
}

func (p preparedEqual[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.Equal(v, p.valueVec)
}

// NotEqual returns true for values where v != value.
type NotEqual[T hwy.Lanes] struct {
	Value T
}

func (p NotEqual[T]) Test(value T) bool {
	return value != p.Value
}

func (p NotEqual[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.NotEqual(v, hwy.Set(p.Value))
}

func (p NotEqual[T]) Prepare() Predicate[T] {
	return preparedNotEqual[T]{value: p.Value, valueVec: hwy.Set(p.Value)}
}

type preparedNotEqual[T hwy.Lanes] struct {
	value    T
	valueVec hwy.Vec[T]
}

func (p preparedNotEqual[T]) Test(value T) bool {
	return value != p.value
}

func (p preparedNotEqual[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.NotEqual(v, p.valueVec)
}

// InRange returns true for values where min <= v <= max.
type InRange[T hwy.Lanes] struct {
	Min T
	Max T
}

func (p InRange[T]) Test(value T) bool {
	return value >= p.Min && value <= p.Max
}

func (p InRange[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	minMask := hwy.GreaterEqual(v, hwy.Set(p.Min))
	maxMask := hwy.LessEqual(v, hwy.Set(p.Max))
	return hwy.MaskAnd(minMask, maxMask)
}

func (p InRange[T]) Prepare() Predicate[T] {
	return preparedInRange[T]{min: p.Min, max: p.Max, minVec: hwy.Set(p.Min), maxVec: hwy.Set(p.Max)}
}

type preparedInRange[T hwy.Lanes] struct {
	min    T
	max    T
	minVec hwy.Vec[T]
	maxVec hwy.Vec[T]
}

func (p preparedInRange[T]) Test(value T) bool {
	return value >= p.min && value <= p.max
}

func (p preparedInRange[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	minMask := hwy.GreaterEqual(v, p.minVec)
	maxMask := hwy.LessEqual(v, p.maxVec)
	return hwy.MaskAnd(minMask, maxMask)
}

// OutOfRange returns true for values where v < min or v > max.
type OutOfRange[T hwy.Lanes] struct {
	Min T
	Max T
}

func (p OutOfRange[T]) Test(value T) bool {
	return value < p.Min || value > p.Max
}

func (p OutOfRange[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	minMask := hwy.Less(v, hwy.Set(p.Min))
	maxMask := hwy.Greater(v, hwy.Set(p.Max))
	return hwy.MaskOr(minMask, maxMask)
}

func (p OutOfRange[T]) Prepare() Predicate[T] {
	return preparedOutOfRange[T]{min: p.Min, max: p.Max, minVec: hwy.Set(p.Min), maxVec: hwy.Set(p.Max)}
}

type preparedOutOfRange[T hwy.Lanes] struct {
	min    T
	max    T
	minVec hwy.Vec[T]
	maxVec hwy.Vec[T]
}

func (p preparedOutOfRange[T]) Test(value T) bool {
	return value < p.min || value > p.max
}

func (p preparedOutOfRange[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	minMask := hwy.Less(v, p.minVec)
	maxMask := hwy.Greater(v, p.maxVec)
	return hwy.MaskOr(minMask, maxMask)
}

// IsZero returns true for values where v == 0.
type IsZero[T hwy.Lanes] struct{}

func (p IsZero[T]) Test(value T) bool {
	var zero T
	return value == zero
}

func (p IsZero[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.Equal(v, hwy.Zero[T]())
}

func (p IsZero[T]) Prepare() Predicate[T] {
	return preparedIsZero[T]{zeroVec: hwy.Zero[T]()}
}

type preparedIsZero[T hwy.Lanes] struct {
	zeroVec hwy.Vec[T]
}

func (p preparedIsZero[T]) Test(value T) bool {
	var zero T
	return value == zero
}

func (p preparedIsZero[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.Equal(v, p.zeroVec)
}

// IsNonZero returns true for values where v != 0.
type IsNonZero[T hwy.Lanes] struct{}

func (p IsNonZero[T]) Test(value T) bool {
	var zero T
	return value != zero
}

func (p IsNonZero[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.NotEqual(v, hwy.Zero[T]())
}

func (p IsNonZero[T]) Prepare() Predicate[T] {
	return preparedIsNonZero[T]{zeroVec: hwy.Zero[T]()}
}

type preparedIsNonZero[T hwy.Lanes] struct {
	zeroVec hwy.Vec[T]
}

func (p preparedIsNonZero[T]) Test(value T) bool {
	var zero T
	return value != zero
}

func (p preparedIsNonZero[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.NotEqual(v, p.zeroVec)
}

// IsPositive returns true for values where v > 0.
type IsPositive[T hwy.Lanes] struct{}

func (p IsPositive[T]) Test(value T) bool {
	var zero T
	return value > zero
}

func (p IsPositive[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.Greater(v, hwy.Zero[T]())
}

func (p IsPositive[T]) Prepare() Predicate[T] {
	return preparedIsPositive[T]{zeroVec: hwy.Zero[T]()}
}

type preparedIsPositive[T hwy.Lanes] struct {
	zeroVec hwy.Vec[T]
}

func (p preparedIsPositive[T]) Test(value T) bool {
	var zero T
	return value > zero
}

func (p preparedIsPositive[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.Greater(v, p.zeroVec)
}

// IsNegative returns true for values where v < 0.
type IsNegative[T hwy.Lanes] struct{}

func (p IsNegative[T]) Test(value T) bool {
	var zero T
	return value < zero
}

func (p IsNegative[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.Less(v, hwy.Zero[T]())
}

func (p IsNegative[T]) Prepare() Predicate[T] {
	return preparedIsNegative[T]{zeroVec: hwy.Zero[T]()}
}

type preparedIsNegative[T hwy.Lanes] struct {
	zeroVec hwy.Vec[T]
}

func (p preparedIsNegative[T]) Test(value T) bool {
	var zero T
	return value < zero
}

func (p preparedIsNegative[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return hwy.Less(v, p.zeroVec)
}

// preparePredicate returns a prepared version of the predicate if it supports
// the Preparable interface, otherwise returns the original predicate.
// This should be called once before a loop to avoid repeated allocations.
func preparePredicate[T hwy.Lanes, P Predicate[T]](pred P) Predicate[T] {
	if p, ok := any(pred).(Preparable[T]); ok {
		return p.Prepare()
	}
	return pred
}
