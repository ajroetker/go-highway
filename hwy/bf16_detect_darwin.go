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

// hasBF16Darwin indicates if ARM BF16 (bfloat16) is available on macOS.
// BF16 is available on Apple M2 and later processors.
var hasBF16Darwin = detectBF16()

// detectBF16 checks if ARM BF16 is available via sysctl on macOS.
func detectBF16() bool {
	val, err := syscall.Sysctl("hw.optional.arm.FEAT_BF16")
	if err != nil {
		return false
	}
	return len(val) > 0 && val[0] == 1
}
