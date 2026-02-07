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

package wavelet

import (
	"testing"
)

// 1D benchmark sizes
var bench1DSizes = []int{64, 256, 1024, 4096}

func BenchmarkSynthesize53(b *testing.B) {
	for _, size := range bench1DSizes {
		b.Run(benchSizeName(size), func(b *testing.B) {
			data := make([]int32, size)
			for i := range data {
				data[i] = int32(i % 256)
			}

			maxHalf := (size + 1) / 2
			low := make([]int32, maxHalf)
			high := make([]int32, maxHalf)

			b.ResetTimer()
			b.ReportAllocs()
			for b.Loop() {
				Synthesize53(data, 0, low, high)
			}
			b.SetBytes(int64(size * 4)) // int32 = 4 bytes
		})
	}
}

func BenchmarkAnalyze53(b *testing.B) {
	for _, size := range bench1DSizes {
		b.Run(benchSizeName(size), func(b *testing.B) {
			data := make([]int32, size)
			for i := range data {
				data[i] = int32(i % 256)
			}

			maxHalf := (size + 1) / 2
			low := make([]int32, maxHalf)
			high := make([]int32, maxHalf)

			b.ResetTimer()
			b.ReportAllocs()
			for b.Loop() {
				Analyze53(data, 0, low, high)
			}
			b.SetBytes(int64(size * 4))
		})
	}
}

func benchSizeName(size int) string {
	switch size {
	case 64:
		return "64"
	case 256:
		return "256"
	case 1024:
		return "1024"
	case 4096:
		return "4096"
	default:
		return "unknown"
	}
}
