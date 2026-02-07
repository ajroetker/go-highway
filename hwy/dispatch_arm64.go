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

package hwy

import (
	"os"

	"golang.org/x/sys/cpu"
)

// CPU feature flags for ARM float16/bfloat16 support
var (
	// hasARMFP16 indicates ARMv8.2-A FP16 extension support
	// Provides native float16 arithmetic in NEON/ASIMD
	// Detected via cpu.ARM64.HasFPHP (scalar) and cpu.ARM64.HasASIMDHP (vector)
	hasARMFP16 bool

	// hasARMBF16 indicates ARMv8.6-A BF16 extension support
	// Provides bfloat16 dot product operations
	// Note: golang.org/x/sys/cpu doesn't have BF16 detection yet
	// BF16 is available on Apple M2+ and recent ARM Cortex CPUs
	hasARMBF16 bool
)

func init() {
	// Check for HWY_NO_SIMD environment variable first
	if NoSimdEnv() {
		currentLevel = DispatchScalar
		currentWidth = 16
		return
	}

	// ARM64 (AArch64) always has NEON (ASIMD) available.
	// It's part of the ARMv8-A base architecture.
	// We still check the cpu package for future SVE support.

	// Note: cpu.ARM64.HasASIMD is always true for ARMv8+
	// We check it for consistency and to enable SVE detection later.
	if cpu.ARM64.HasASIMD {
		currentLevel = DispatchNEON
		currentWidth = 16 // NEON is 128-bit (16 bytes)
	} else {
		// Fallback to scalar (should never happen on ARMv8+)
		currentLevel = DispatchScalar
		currentWidth = 16
	}

	// SME support (Apple M4+)
	// Check for HWY_NO_SME environment variable to disable SME
	if hasSME && os.Getenv("HWY_NO_SME") == "" {
		currentLevel = DispatchSME
		// Keep currentWidth at NEON width (16 bytes) until hwygen generates
		// SVE/SME-width kernels. MaxLanes must match the dispatched kernel
		// width, and all current arm64 kernels target NEON (128-bit).
		// Packages that need SME use hwy.HasSME() for explicit dispatch.
	}

	// Future: SVE support (without SME streaming mode)
	// if cpu.ARM64.HasSVE {
	//     currentLevel = DispatchSVE
	//     currentWidth = ... // SVE width is variable
	// }

	// Detect FP16/BF16 features
	detectARMFP16BF16Features()
}

func detectARMFP16BF16Features() {
	// ARM FP16 detection via x/sys/cpu
	// HasFPHP: scalar FP16 support (ARMv8.2-A FEAT_FP16)
	// HasASIMDHP: NEON FP16 support (ARMv8.2-A FEAT_FP16)
	// HasASIMDFHM: FP16 to FP32 fused multiply-add (ARMv8.4-A FEAT_FHM)
	hasARMFP16 = cpu.ARM64.HasFPHP && cpu.ARM64.HasASIMDHP

	// ARM BF16 detection
	// golang.org/x/sys/cpu doesn't have explicit BF16 detection yet
	// BF16 (FEAT_BF16) was introduced in ARMv8.6-A
	// On macOS/Apple Silicon, we detect via sysctl (see bf16_detect_darwin.go)
	hasARMBF16 = hasBF16Darwin
}

// HasARMFP16 returns true if the CPU supports ARM FP16 extension.
// ARM FP16 provides native float16 arithmetic in NEON/ASIMD.
// Present on ARMv8.2-A and later CPUs (Apple A11+, Cortex-A75+).
func HasARMFP16() bool {
	return hasARMFP16
}

// HasARMBF16 returns true if the CPU supports ARM BF16 extension.
// ARM BF16 provides bfloat16 dot product operations.
// Present on ARMv8.6-A and later CPUs (Apple M2+, Cortex-X2+).
func HasARMBF16() bool {
	return hasARMBF16
}

// HasF16C returns false on ARM64 (F16C is an x86-specific feature).
// Use HasARMFP16() for ARM float16 support.
func HasF16C() bool {
	return false
}

// HasAVX512FP16 returns false on ARM64 (AVX-512 is x86-specific).
// Use HasARMFP16() for ARM float16 support.
func HasAVX512FP16() bool {
	return false
}

// HasAVX512BF16 returns false on ARM64 (AVX-512 is x86-specific).
// Use HasARMBF16() for ARM bfloat16 support.
func HasAVX512BF16() bool {
	return false
}
