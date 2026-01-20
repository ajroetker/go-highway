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

//go:build !noasm && arm64

package vec

import (
	"github.com/ajroetker/go-highway/hwy/contrib/vec/asm"
)

// Override hwygen-generated dispatch with optimized GoAT assembly implementations.
// These keep the entire loop in assembly, eliminating per-operation function call overhead.
func init() {
	// Dot product
	DotFloat32 = asm.DotF32
	DotFloat64 = asm.DotF64

	// Squared norm (sum of squares)
	SquaredNormFloat32 = asm.SquaredNormF32
	SquaredNormFloat64 = asm.SquaredNormF64

	// L2 squared distance
	L2SquaredDistanceFloat32 = asm.L2SquaredDistanceF32
	L2SquaredDistanceFloat64 = asm.L2SquaredDistanceF64
}
