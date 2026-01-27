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

package algo

import "github.com/ajroetker/go-highway/hwy"

//go:generate go run ../../../cmd/hwygen -input copy_base.go -output . -targets avx2,avx512,neon,fallback -dispatch copy

// BaseCopyIf conditionally copies elements from src to dst based on a predicate.
// The predicate receives a vector and returns a mask indicating which elements to copy.
// Elements where the mask is true are packed together in dst (stream compaction).
// Returns the number of elements copied (limited by dst capacity).
func BaseCopyIf[T hwy.Lanes](src, dst []T, pred func(hwy.Vec[T]) hwy.Mask[T]) int {
	n := len(src)
	dstLen := len(dst)
	if n == 0 || dstLen == 0 {
		return 0
	}

	lanes := hwy.MaxLanes[T]()
	dstIdx := 0
	i := 0

	// Process full vectors
	for ; i+lanes <= n && dstIdx < dstLen; i += lanes {
		v := hwy.LoadFull(src[i:])
		mask := pred(v)

		// Limit how many we can store
		remaining := dstLen - dstIdx
		count := min(hwy.CompressStore(v, mask, dst[dstIdx:]), remaining)
		dstIdx += count

		if dstIdx >= dstLen {
			break
		}
	}

	// Handle tail elements
	if remaining := n - i; remaining > 0 && dstIdx < dstLen {
		buf := make([]T, lanes)
		copy(buf, src[i:i+remaining])
		v := hwy.Load(buf)
		mask := pred(v)

		// Create a tail mask to only process valid elements
		tailMask := hwy.FirstN[T](remaining)
		mask = hwy.MaskAnd(mask, tailMask)

		dstRemaining := dstLen - dstIdx
		count := min(hwy.CompressStore(v, mask, dst[dstIdx:]), dstRemaining)
		dstIdx += count
	}

	return dstIdx
}
