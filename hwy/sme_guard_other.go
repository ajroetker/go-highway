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

//go:build !(darwin && arm64) && !(linux && arm64)

package hwy

// SMEGuard is a no-op on platforms that do not support ARM SME.
// Returns a cleanup function that must be deferred:
//
//	defer hwy.SMEGuard()()
func SMEGuard() func() {
	return func() {}
}
