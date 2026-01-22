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

// Override Float16/BFloat16 dispatch to use fallback implementations.
//
// The NEON assembly code for F16/BF16 requires ARMv8.2-A+ FP16 and ARMv8.6-A+ BF16
// extensions respectively. However, CPU feature detection via golang.org/x/sys/cpu
// can be unreliable on some Linux ARM64 systems, leading to crashes when the
// detected features don't match actual hardware capabilities.
//
// This file runs after the generated dispatch files (alphabetically via "z_" prefix)
// and forces fallback implementations for F16/BF16 on all ARM64 systems for safety.
//
// To enable optimized F16/BF16 paths on known-good hardware (like Apple Silicon),
// set HWY_ENABLE_F16=1 environment variable.
func init() {
	// Always use fallback for F16/BF16 by default - CPU feature detection
	// has proven unreliable on some ARM64 systems
	MatMulFloat16 = BaseMatMul_fallback_Float16
	BlockedMatMulFloat16 = BaseBlockedMatMul_fallback_Float16
	MatMulBFloat16 = BaseMatMul_fallback_BFloat16
	BlockedMatMulBFloat16 = BaseBlockedMatMul_fallback_BFloat16

	// Allow opting into optimized paths on known-good hardware
	// This is useful for Apple Silicon and other verified platforms
	if hwy.EnableF16Env() && hwy.HasARMFP16() {
		// Re-enable optimized F16 paths
		// matmul_neon_arm64.go would have set these, but we overwrote them above
		// Let matmul_neon_arm64.go's dispatch stand (it already ran before us)
		// We need to re-set them here since we cleared them
		MatMulFloat16 = matmulNEONF16
		BlockedMatMulFloat16 = blockedMatMulNEONF16
	}
}
