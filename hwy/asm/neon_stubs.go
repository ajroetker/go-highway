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

// Type conversions (Phase 5)
func PromoteF32ToF64(input []float32, result []float64) { panic("NEON not available") }
func DemoteF64ToF32(input []float64, result []float32)  { panic("NEON not available") }
func ConvertF32ToI32(input []float32, result []int32)   { panic("NEON not available") }
func ConvertI32ToF32(input []int32, result []float32)   { panic("NEON not available") }
func RoundF32(input, result []float32)                  { panic("NEON not available") }
func TruncF32(input, result []float32)                  { panic("NEON not available") }
func CeilF32(input, result []float32)                   { panic("NEON not available") }
func FloorF32(input, result []float32)                  { panic("NEON not available") }

// Memory operations (Phase 4)
func GatherF32(base []float32, indices []int32, result []float32) { panic("NEON not available") }
func GatherF64(base []float64, indices []int32, result []float64) { panic("NEON not available") }
func GatherI32(base []int32, indices []int32, result []int32)     { panic("NEON not available") }
func ScatterF32(values []float32, indices []int32, base []float32) {
	panic("NEON not available")
}
func ScatterF64(values []float64, indices []int32, base []float64) {
	panic("NEON not available")
}
func ScatterI32(values []int32, indices []int32, base []int32)       { panic("NEON not available") }
func MaskedLoadF32(input []float32, mask []int32, result []float32)  { panic("NEON not available") }
func MaskedStoreF32(input []float32, mask []int32, output []float32) { panic("NEON not available") }

// Shuffle/Permutation operations (Phase 6)
func ReverseF32(input, result []float32)                              { panic("NEON not available") }
func ReverseF64(input, result []float64)                              { panic("NEON not available") }
func Reverse2F32(input, result []float32)                             { panic("NEON not available") }
func Reverse4F32(input, result []float32)                             { panic("NEON not available") }
func BroadcastF32(input []float32, lane int, result []float32)        { panic("NEON not available") }
func GetLaneF32(input []float32, lane int) float32                    { panic("NEON not available") }
func InsertLaneF32(input []float32, lane int, value float32, result []float32) {
	panic("NEON not available")
}
func InterleaveLowerF32(a, b, result []float32)   { panic("NEON not available") }
func InterleaveUpperF32(a, b, result []float32)   { panic("NEON not available") }
func TableLookupBytesU8(tbl, idx, result []uint8) { panic("NEON not available") }

// Comparison operations (Phase 7)
func EqF32(a, b []float32, result []int32) { panic("NEON not available") }
func EqI32(a, b, result []int32)           { panic("NEON not available") }
func NeF32(a, b []float32, result []int32) { panic("NEON not available") }
func NeI32(a, b, result []int32)           { panic("NEON not available") }
func LtF32(a, b []float32, result []int32) { panic("NEON not available") }
func LtI32(a, b, result []int32)           { panic("NEON not available") }
func LeF32(a, b []float32, result []int32) { panic("NEON not available") }
func LeI32(a, b, result []int32)           { panic("NEON not available") }
func GtF32(a, b []float32, result []int32) { panic("NEON not available") }
func GtI32(a, b, result []int32)           { panic("NEON not available") }
func GeF32(a, b []float32, result []int32) { panic("NEON not available") }
func GeI32(a, b, result []int32)           { panic("NEON not available") }
