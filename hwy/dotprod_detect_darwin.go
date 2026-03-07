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

import "syscall"

// hasDotProdDarwin indicates if ARM DOTPROD is available on macOS.
// DOTPROD (FEAT_DotProd) is available on Apple A11+ processors.
var hasDotProdDarwin = detectDotProd()

// detectDotProd checks if ARM DOTPROD is available via sysctl on macOS.
func detectDotProd() bool {
	val, err := syscall.Sysctl("hw.optional.arm.FEAT_DotProd")
	if err != nil {
		return false
	}
	return len(val) > 0 && val[0] == 1
}
