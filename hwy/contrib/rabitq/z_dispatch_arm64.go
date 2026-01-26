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

//go:build !noasm && darwin && arm64

package rabitq

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/rabitq/asm"
)

// smeThreshold is the minimum number of uint64 elements where SME outperforms NEON.
// Based on benchmarks on Apple M4 Max:
//   - At 256 elements: NEON beats SME due to smstart/smstop overhead
//   - At 512+ elements: SME outperforms NEON
//
// SME has ~40ns overhead from smstart/smstop, so it only pays off for larger inputs.
const smeThreshold = 512

// bitProductAdaptive dispatches to SME or NEON based on input size.
// SME has ~40ns overhead from smstart/smstop, so it only pays off for larger inputs.
func bitProductAdaptive(code, query1, query2, query3, query4 []uint64) uint32 {
	// Handle empty slices (assembly implementations don't)
	if len(code) == 0 {
		return 0
	}
	if len(code) >= smeThreshold {
		return asm.BitProductSME(code, query1, query2, query3, query4)
	}
	return asm.BitProductNEON(code, query1, query2, query3, query4)
}

func init() {
	// Use adaptive dispatch on M4+ (SME available)
	// For smaller vectors, NEON is faster due to SME's smstart/smstop overhead
	// For larger vectors (>= 512 elements), SME outperforms NEON
	if hwy.HasSME() {
		BitProduct = bitProductAdaptive
	}
}
