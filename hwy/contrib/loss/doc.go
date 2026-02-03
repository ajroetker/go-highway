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

// Package loss provides SIMD-accelerated loss function computations.
//
// The primary contribution is Cut Cross-Entropy (CCE), which computes
// cross-entropy loss without materializing the full logits matrix.
// For a model with vocabulary size V and sequence length S, standard
// cross-entropy requires O(V * S) memory for logits. CCE reduces this
// to O(V) by streaming the log-sum-exp computation one position at a time.
//
// This is based on the Apple "Cut Your Losses" paper (ICLR 2025), which
// showed that logits computation can consume up to 90% of training memory.
package loss
