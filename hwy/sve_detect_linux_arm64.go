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

//go:build linux && arm64

package hwy

import (
	"os"

	"golang.org/x/sys/cpu"
)

// hasSVE indicates if ARM SVE (Scalable Vector Extension) is available.
// SVE is available on processors such as Fujitsu A64FX, ARM Neoverse V1/V2,
// and other ARMv8.2-A+ implementations with SVE support.
var hasSVE = cpu.ARM64.HasSVE

// HasSVE returns true if the CPU supports ARM SVE instructions and
// SVE has not been disabled via environment variables.
// Returns false when HWY_NO_SIMD or HWY_NO_SVE is set.
func HasSVE() bool {
	if NoSimdEnv() || os.Getenv("HWY_NO_SVE") != "" {
		return false
	}
	return hasSVE
}
