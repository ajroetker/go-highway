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

//go:build arm64

package matmul

// getKernelLanesFloat32 returns the lanes used by the float32 kernel implementation.
// On ARM64, the generated kernels use NEON Float32x4 intrinsics (4 lanes),
// regardless of whether SME is detected (which would report 16 lanes).
func getKernelLanesFloat32() int {
	return 4 // NEON Float32x4 = 4 lanes
}

// getKernelLanesFloat64 returns the lanes used by the float64 kernel implementation.
// On ARM64, the generated kernels use NEON Float64x2 intrinsics (2 lanes),
// regardless of whether SME is detected (which would report 8 lanes).
func getKernelLanesFloat64() int {
	return 2 // NEON Float64x2 = 2 lanes
}
