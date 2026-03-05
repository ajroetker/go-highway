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

package algo

import "github.com/ajroetker/go-highway/hwy"

// =============================================================================
// Callback-based API (for custom predicates)
// =============================================================================

// FuncPredicate wraps a callback function as a Predicate.
// This allows callback-based APIs to use the predicate infrastructure internally.
// Note: FuncPredicate's Test method allocates because it must create a Vec.
// For zero-allocation scalar testing, use the built-in predicate types instead.
type FuncPredicate[T hwy.Lanes] struct {
	Fn func(hwy.Vec[T]) hwy.Mask[T]
}

func (p FuncPredicate[T]) Test(value T) bool {
	// Create single-element vector and test (allocates)
	v := hwy.Set(value)
	mask := p.Fn(v)
	return hwy.FindFirstTrue(mask) >= 0
}

func (p FuncPredicate[T]) Apply(v hwy.Vec[T]) hwy.Mask[T] {
	return p.Fn(v)
}

// FindIf returns the index of the first element where pred returns a mask with any true lane.
// Returns -1 if no element matches.
//
// Example: Find first element greater than 10
//
//	idx := FindIf(data, func(v hwy.Vec[float32]) hwy.Mask[float32] {
//	    return hwy.GreaterThan(v, hwy.Set(float32(10)))
//	})
//
// For better performance with built-in predicates, use FindIfP with predicate types.
func FindIf[T hwy.Lanes](slice []T, pred func(hwy.Vec[T]) hwy.Mask[T]) int {
	return FindIfP(slice, FuncPredicate[T]{Fn: pred})
}

// CountIf returns the number of elements where pred returns a true mask lane.
//
// Example: Count elements greater than 0
//
//	n := CountIf(data, func(v hwy.Vec[float32]) hwy.Mask[float32] {
//	    return hwy.GreaterThan(v, hwy.Zero[float32]())
//	})
//
// For better performance with built-in predicates, use CountIfP with predicate types.
func CountIf[T hwy.Lanes](slice []T, pred func(hwy.Vec[T]) hwy.Mask[T]) int {
	return CountIfP(slice, FuncPredicate[T]{Fn: pred})
}

// All returns true if pred returns true for all elements.
// Short-circuits on first false.
//
// For better performance with built-in predicates, use AllP with predicate types.
func All[T hwy.Lanes](slice []T, pred func(hwy.Vec[T]) hwy.Mask[T]) bool {
	return AllP(slice, FuncPredicate[T]{Fn: pred})
}

// Any returns true if pred returns true for any element.
// Short-circuits on first true.
//
// For better performance with built-in predicates, use AnyP with predicate types.
func Any[T hwy.Lanes](slice []T, pred func(hwy.Vec[T]) hwy.Mask[T]) bool {
	return AnyP(slice, FuncPredicate[T]{Fn: pred})
}

