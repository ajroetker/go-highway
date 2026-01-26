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

package matmul

// CacheParams defines architecture-specific blocking parameters for the
// GotoBLAS-style 5-loop matmul algorithm.
//
// The parameters are tuned for cache hierarchy:
//   - Mr × Nr: Micro-tile dimensions (register blocking)
//   - Kc: K-blocking for L1 cache (packed A panel height)
//   - Mc: M-blocking for L2 cache (packed A panel width)
//   - Nc: N-blocking for L3 cache (packed B panel width)
//
// Memory layout after packing:
//   - Packed A: [ceil(M/Mr), Kc, Mr] - K-first within micro-panels
//   - Packed B: [ceil(N/Nr), Kc, Nr] - K-first within micro-panels
type CacheParams struct {
	Mr int // Micro-tile rows (register blocking)
	Nr int // Micro-tile columns (register blocking, in elements not vectors)
	Kc int // K-blocking (L1 cache)
	Mc int // M-blocking (L2 cache)
	Nc int // N-blocking (L3 cache)
}

// Blocking parameters tuned for different architectures.
// These are conservative estimates that should work well across most CPUs
// in each architecture family.

// CacheParamsAVX512 returns blocking parameters for AVX-512.
// Optimized for 512-bit vectors (16 float32s per vector).
// Assumes: 32KB L1d, 1MB L2, 30+MB L3 (typical for Skylake-X and later)
func CacheParamsAVX512() CacheParams {
	return CacheParams{
		Mr: 4,    // 4 rows per micro-tile
		Nr: 32,   // 2 vectors × 16 lanes = 32 columns
		Kc: 512,  // L1 blocking: 4 * 512 * 4 bytes = 8KB packed A strip
		Mc: 512,  // L2 blocking: 512 * 512 * 4 bytes = 1MB packed A panel
		Nc: 4096, // L3 blocking: 512 * 4096 * 4 bytes = 8MB packed B panel
	}
}

// CacheParamsAVX2 returns blocking parameters for AVX2.
// Optimized for 256-bit vectors (8 float32s per vector).
// Assumes: 32KB L1d, 256KB L2, 8+MB L3 (typical for Haswell and later)
func CacheParamsAVX2() CacheParams {
	return CacheParams{
		Mr: 4,    // 4 rows per micro-tile
		Nr: 16,   // 2 vectors × 8 lanes = 16 columns
		Kc: 256,  // L1 blocking: 4 * 256 * 4 bytes = 4KB packed A strip
		Mc: 256,  // L2 blocking: 256 * 256 * 4 bytes = 256KB packed A panel
		Nc: 2048, // L3 blocking: 256 * 2048 * 4 bytes = 2MB packed B panel
	}
}

// CacheParamsNEON returns blocking parameters for ARM NEON.
// Optimized for 128-bit vectors (4 float32s per vector).
// Assumes: 32-64KB L1d, 256KB-1MB L2, 4+MB L3 (typical for Cortex-A76 and later)
func CacheParamsNEON() CacheParams {
	return CacheParams{
		Mr: 4,    // 4 rows per micro-tile
		Nr: 8,    // 2 vectors × 4 lanes = 8 columns
		Kc: 256,  // L1 blocking: 4 * 256 * 4 bytes = 4KB packed A strip
		Mc: 256,  // L2 blocking: 256 * 256 * 4 bytes = 256KB packed A panel
		Nc: 1024, // L3 blocking: 256 * 1024 * 4 bytes = 1MB packed B panel
	}
}

// CacheParamsFallback returns conservative blocking parameters for fallback.
// Uses smaller blocks that should work on any hardware.
func CacheParamsFallback() CacheParams {
	return CacheParams{
		Mr: 4,   // 4 rows per micro-tile
		Nr: 8,   // 8 columns (no vectorization assumed)
		Kc: 128, // Small K-blocking
		Mc: 128, // Small M-blocking
		Nc: 512, // Small N-blocking
	}
}

// CacheParamsFloat64AVX512 returns blocking parameters for AVX-512 with float64.
// Optimized for 512-bit vectors (8 float64s per vector).
func CacheParamsFloat64AVX512() CacheParams {
	return CacheParams{
		Mr: 4,    // 4 rows per micro-tile
		Nr: 16,   // 2 vectors × 8 lanes = 16 columns
		Kc: 256,  // L1 blocking: 4 * 256 * 8 bytes = 8KB packed A strip
		Mc: 256,  // L2 blocking: 256 * 256 * 8 bytes = 512KB packed A panel
		Nc: 2048, // L3 blocking: 256 * 2048 * 8 bytes = 4MB packed B panel
	}
}

// CacheParamsFloat64AVX2 returns blocking parameters for AVX2 with float64.
// Optimized for 256-bit vectors (4 float64s per vector).
func CacheParamsFloat64AVX2() CacheParams {
	return CacheParams{
		Mr: 4,    // 4 rows per micro-tile
		Nr: 8,    // 2 vectors × 4 lanes = 8 columns
		Kc: 128,  // L1 blocking: 4 * 128 * 8 bytes = 4KB packed A strip
		Mc: 128,  // L2 blocking: 128 * 128 * 8 bytes = 128KB packed A panel
		Nc: 1024, // L3 blocking: 128 * 1024 * 8 bytes = 1MB packed B panel
	}
}

// CacheParamsFloat64NEON returns blocking parameters for ARM NEON with float64.
// Optimized for 128-bit vectors (2 float64s per vector).
func CacheParamsFloat64NEON() CacheParams {
	return CacheParams{
		Mr: 4,   // 4 rows per micro-tile
		Nr: 4,   // 2 vectors × 2 lanes = 4 columns
		Kc: 128, // L1 blocking: 4 * 128 * 8 bytes = 4KB packed A strip
		Mc: 128, // L2 blocking: 128 * 128 * 8 bytes = 128KB packed A panel
		Nc: 512, // L3 blocking: 128 * 512 * 8 bytes = 512KB packed B panel
	}
}

// CacheParamsFloat16NEON returns blocking parameters for ARM NEON with float16.
// Optimized for 128-bit vectors (8 float16s per vector).
func CacheParamsFloat16NEON() CacheParams {
	return CacheParams{
		Mr: 4,    // 4 rows per micro-tile
		Nr: 16,   // 2 vectors × 8 lanes = 16 columns
		Kc: 512,  // L1 blocking: 4 * 512 * 2 bytes = 4KB packed A strip
		Mc: 512,  // L2 blocking: 512 * 512 * 2 bytes = 512KB packed A panel
		Nc: 2048, // L3 blocking: 512 * 2048 * 2 bytes = 2MB packed B panel
	}
}

// CacheParamsBFloat16NEON returns blocking parameters for ARM NEON with bfloat16.
// Uses f32 accumulation, so Nr matches f32 (8 columns).
func CacheParamsBFloat16NEON() CacheParams {
	return CacheParams{
		Mr: 4,    // 4 rows per micro-tile
		Nr: 8,    // 2 f32 vectors × 4 lanes = 8 columns (f32 accumulation)
		Kc: 256,  // L1 blocking: 4 * 256 * 2 bytes = 2KB packed A strip
		Mc: 256,  // L2 blocking: 256 * 256 * 2 bytes = 128KB packed A panel
		Nc: 1024, // L3 blocking: 256 * 1024 * 2 bytes = 512KB packed B panel
	}
}

// PackedASize returns the buffer size needed for packed A matrix.
// Packed A layout: ceil(Mc/Mr) micro-panels, each Mr × Kc elements.
func (p CacheParams) PackedASize() int {
	numPanels := (p.Mc + p.Mr - 1) / p.Mr
	return numPanels * p.Mr * p.Kc
}

// PackedBSize returns the buffer size needed for packed B matrix.
// Packed B layout: ceil(Nc/Nr) micro-panels, each Kc × Nr elements.
func (p CacheParams) PackedBSize() int {
	numPanels := (p.Nc + p.Nr - 1) / p.Nr
	return numPanels * p.Kc * p.Nr
}

// PackedOutputSize returns the buffer size needed for packed output.
// Used as intermediate buffer between micro-kernel and final output.
// Layout: Mc × Nc elements (one panel's worth of output).
func (p CacheParams) PackedOutputSize() int {
	return p.Mc * p.Nc
}

// V2 Cache Parameters
//
// These parameters are optimized for the packed output buffer pattern used in V2.
// Key differences from V1:
//   - Much smaller Mc: Reduces packed output buffer size for better cache locality
//   - Smaller Nc: Further reduces packed output buffer
//   - These match the approach in gomlx's packgemm-simd-large-opt
//
// The packed output pattern benefits from smaller panels because:
//   - Micro-kernel writes to a small contiguous buffer (no bounds checking)
//   - ApplyPackedOutput then copies to final output with SIMD
//   - Smaller buffer = better L1/L2 cache utilization

// CacheParamsV2AVX512 returns V2 blocking parameters for AVX-512.
// Optimized for the packed output buffer pattern.
func CacheParamsV2AVX512() CacheParams {
	return CacheParams{
		Mr: 4,   // 4 rows per micro-tile
		Nr: 32,  // 2 vectors × 16 lanes = 32 columns
		Kc: 256, // L1 blocking: smaller for better reuse
		Mc: 4,   // Very small: matches Jan's approach, tiny packed output
		Nc: 512, // Smaller: 4 * 512 = 2KB packed output buffer
	}
}

// CacheParamsV2AVX2 returns V2 blocking parameters for AVX2.
func CacheParamsV2AVX2() CacheParams {
	return CacheParams{
		Mr: 4,   // 4 rows per micro-tile
		Nr: 16,  // 2 vectors × 8 lanes = 16 columns
		Kc: 256, // L1 blocking
		Mc: 4,   // Very small for packed output pattern
		Nc: 512, // 4 * 512 = 2KB packed output buffer
	}
}

// CacheParamsV2NEON returns V2 blocking parameters for ARM NEON.
func CacheParamsV2NEON() CacheParams {
	return CacheParams{
		Mr: 4,   // 4 rows per micro-tile
		Nr: 8,   // 2 vectors × 4 lanes = 8 columns
		Kc: 256, // L1 blocking
		Mc: 4,   // Very small for packed output pattern
		Nc: 512, // 4 * 512 = 2KB packed output buffer
	}
}

// CacheParamsV2Fallback returns V2 blocking parameters for fallback.
func CacheParamsV2Fallback() CacheParams {
	return CacheParams{
		Mr: 4,   // 4 rows per micro-tile
		Nr: 4,   // Smaller for scalar code
		Kc: 128, // Smaller K-blocking
		Mc: 4,   // Very small for packed output pattern
		Nc: 256, // 4 * 256 = 1KB packed output buffer
	}
}
