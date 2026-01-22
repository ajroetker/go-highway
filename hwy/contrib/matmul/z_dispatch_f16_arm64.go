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
//
// The NEON assembly code is compiled with:
// - F16 code: -march=armv8.2-a+fp16 (requires ARMv8.2-A FP16 extension)
// - BF16 code: -march=armv8.6-a+bf16 (requires ARMv8.6-A BF16 extension)
//
// The generated dispatch files (dispatch_*_arm64.gen.go) may unconditionally use
// the NEON versions, but not all ARM64 CPUs support FP16/BF16. This file runs
// after the generated dispatch files (alphabetically via "z_" prefix) and
// sets up the correct dispatch based on CPU capabilities.
func init() {
	// Use optimized NEON path if CPU supports FP16
	if hwy.HasARMFP16() {
		MatMulFloat16 = matmulNEONF16
		BlockedMatMulFloat16 = blockedMatMulNEONF16
	} else {
		MatMulFloat16 = BaseMatMul_fallback_Float16
		BlockedMatMulFloat16 = BaseBlockedMatMul_fallback_Float16
	}

	// BF16 always uses fallback for now - golang.org/x/sys/cpu doesn't detect it yet
	// TODO: Enable optimized path when BF16 detection is available
	MatMulBFloat16 = BaseMatMul_fallback_BFloat16
	BlockedMatMulBFloat16 = BaseBlockedMatMul_fallback_BFloat16
}
