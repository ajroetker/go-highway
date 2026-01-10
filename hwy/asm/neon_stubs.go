//go:build !arm64 || noasm

package asm

// Stub implementations for non-ARM64 or noasm builds.
// These should never be called - the hwy package will use scalar fallbacks.

func AddF32(a, b, result []float32)        { panic("NEON not available") }
func SubF32(a, b, result []float32)        { panic("NEON not available") }
func MulF32(a, b, result []float32)        { panic("NEON not available") }
func DivF32(a, b, result []float32)        { panic("NEON not available") }
func FmaF32(a, b, c, result []float32)     { panic("NEON not available") }
func MinF32(a, b, result []float32)        { panic("NEON not available") }
func MaxF32(a, b, result []float32)        { panic("NEON not available") }
func ReduceSumF32(input []float32) float32 { panic("NEON not available") }
func ReduceMinF32(input []float32) float32 { panic("NEON not available") }
func ReduceMaxF32(input []float32) float32 { panic("NEON not available") }
func SqrtF32(a, result []float32)          { panic("NEON not available") }
func AbsF32(a, result []float32)           { panic("NEON not available") }
func NegF32(a, result []float32)           { panic("NEON not available") }

func AddF64(a, b, result []float64)        { panic("NEON not available") }
func MulF64(a, b, result []float64)        { panic("NEON not available") }
func FmaF64(a, b, c, result []float64)     { panic("NEON not available") }
func ReduceSumF64(input []float64) float64 { panic("NEON not available") }
