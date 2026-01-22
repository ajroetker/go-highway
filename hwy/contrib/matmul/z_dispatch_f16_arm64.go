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

//go:build arm64

package matmul

import "github.com/ajroetker/go-highway/hwy"

// Override Float16/BFloat16 dispatch based on CPU feature detection.
// The generated dispatch files (dispatch_*_arm64.gen.go) unconditionally use
// the NEON versions for Float16/BFloat16, but these require ARMv8.2+ FP16
// and ARMv8.6+ BF16 extensions respectively.
//
// This file runs after the generated dispatch files (alphabetically) and
// downgrades to fallback implementations if the CPU doesn't support the
// required extensions.
func init() {
	// The generated dispatch already set MatMulFloat16/BFloat16 to NEON versions.
	// If the CPU doesn't support FP16/BF16, downgrade to fallback.
	if !hwy.HasARMFP16() {
		MatMulFloat16 = BaseMatMul_fallback_Float16
		BlockedMatMulFloat16 = BaseBlockedMatMul_fallback_Float16
	}
	if !hwy.HasARMBF16() {
		MatMulBFloat16 = BaseMatMul_fallback_BFloat16
		BlockedMatMulBFloat16 = BaseBlockedMatMul_fallback_BFloat16
	}
}
