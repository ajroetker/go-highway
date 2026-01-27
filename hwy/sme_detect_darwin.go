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

//go:build darwin && arm64

package hwy

import (
	"os"
	"syscall"
)

// hasSME indicates if ARM SME (Scalable Matrix Extension) is available.
// SME is available on Apple M4 and later processors.
var hasSME = detectSME()

// detectSME checks if ARM SME is available via sysctl on macOS.
func detectSME() bool {
	val, err := syscall.Sysctl("hw.optional.arm.FEAT_SME")
	if err != nil {
		return false
	}
	return len(val) > 0 && val[0] == 1
}

// HasSME returns true if the CPU supports ARM SME instructions and
// SME has not been disabled via environment variables.
// Returns false when HWY_NO_SIMD or HWY_NO_SME is set.
func HasSME() bool {
	if NoSimdEnv() || os.Getenv("HWY_NO_SME") != "" {
		return false
	}
	return hasSME
}
