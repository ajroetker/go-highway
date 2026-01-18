package vec

import "github.com/ajroetker/go-highway/hwy"

// =============================================================================
// Constants for vector operations
// =============================================================================

// Float16 constants
var (
	vecOne_f16  hwy.Float16 = hwy.Float32ToFloat16(1.0)
	vecZero_f16 hwy.Float16 = hwy.Float32ToFloat16(0.0)
)

// BFloat16 constants
var (
	vecOne_bf16  hwy.BFloat16 = hwy.Float32ToBFloat16(1.0)
	vecZero_bf16 hwy.BFloat16 = hwy.Float32ToBFloat16(0.0)
)

// Float32 constants
var (
	vecOne_f32  float32 = 1.0
	vecZero_f32 float32 = 0.0
)

// Float64 constants
var (
	vecOne_f64  float64 = 1.0
	vecZero_f64 float64 = 0.0
)
