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

package roaring

// PopcntSliceRange returns the total popcount of s[start:end].
// This is used by roaring's rank() and getCardinalityInRange().
func PopcntSliceRange(s []uint64, start, end int) uint64 {
	if start >= end || start >= len(s) {
		return 0
	}
	if end > len(s) {
		end = len(s)
	}
	return PopcntSlice(s[start:end])
}
