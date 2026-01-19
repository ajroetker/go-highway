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

//go:build !arm64 || noasm

package asm

import "unsafe"

// Stub implementations for non-ARM64 or noasm builds.
// These should never be called - the hwy package will use scalar fallbacks.

// Float32x4 represents a 128-bit NEON vector of 4 float32 values.
type Float32x4 [16]byte

// Float64x2 represents a 128-bit NEON vector of 2 float64 values.
type Float64x2 [16]byte

// Int32x4 represents a 128-bit NEON vector of 4 int32 values.
type Int32x4 [16]byte

// Int64x2 represents a 128-bit NEON vector of 2 int64 values.
type Int64x2 [16]byte

// Float32x2 represents a 64-bit NEON vector of 2 float32 values.
type Float32x2 [8]byte

// Int32x2 represents a 64-bit vector of 2 int32 values.
type Int32x2 [8]byte

// Uint8x16 represents a 128-bit NEON vector of 16 uint8 values.
type Uint8x16 [16]byte

// Uint16x8 represents a 128-bit NEON vector of 8 uint16 values.
type Uint16x8 [16]byte

// Uint32x4 represents a 128-bit NEON vector of 4 uint32 values.
type Uint32x4 [16]byte

// Uint64x2 represents a 128-bit NEON vector of 2 uint64 values.
type Uint64x2 [16]byte

// BoolMask32x4 represents a 4-element boolean mask.
type BoolMask32x4 [4]bool

// GetBit returns the boolean value at the given index.
func (m BoolMask32x4) GetBit(i int) bool { return m[i] }

// BoolMask32x2 represents a 2-element boolean mask.
type BoolMask32x2 [2]bool

// GetBit returns the boolean value at the given index.
func (m BoolMask32x2) GetBit(i int) bool { return m[i] }

// ===== Uint8x16 stub methods =====

func BroadcastUint8x16(v uint8) Uint8x16 {
	var arr [16]uint8
	for i := range arr {
		arr[i] = v
	}
	return *(*Uint8x16)(unsafe.Pointer(&arr))
}

func LoadUint8x16(s []uint8) Uint8x16       { return *(*Uint8x16)(unsafe.Pointer(&s[0])) }
func LoadUint8x16Slice(s []uint8) Uint8x16  { return LoadUint8x16(s) }
func ZeroUint8x16() Uint8x16                { return Uint8x16{} }
func (v Uint8x16) Get(i int) uint8          { return v[i] }
func (v *Uint8x16) Set(i int, val uint8)    { v[i] = val }
func (v Uint8x16) Data() []uint8            { return v[:] }
func (v Uint8x16) StoreSlice(s []uint8)     { *(*Uint8x16)(unsafe.Pointer(&s[0])) = v }
func (v Uint8x16) GetBit(i int) bool        { return v[i] != 0 }
func (v Uint8x16) Add(other Uint8x16) Uint8x16         { panic("NEON not available") }
func (v Uint8x16) Sub(other Uint8x16) Uint8x16         { panic("NEON not available") }
func (v Uint8x16) AddSaturated(other Uint8x16) Uint8x16 { panic("NEON not available") }
func (v Uint8x16) SubSaturated(other Uint8x16) Uint8x16 { panic("NEON not available") }
func (v Uint8x16) Min(other Uint8x16) Uint8x16         { panic("NEON not available") }
func (v Uint8x16) Max(other Uint8x16) Uint8x16         { panic("NEON not available") }
func (v Uint8x16) LessThan(other Uint8x16) Uint8x16    { panic("NEON not available") }
func (v Uint8x16) GreaterThan(other Uint8x16) Uint8x16 { panic("NEON not available") }
func (v Uint8x16) LessEqual(other Uint8x16) Uint8x16   { panic("NEON not available") }
func (v Uint8x16) GreaterEqual(other Uint8x16) Uint8x16 { panic("NEON not available") }
func (v Uint8x16) Equal(other Uint8x16) Uint8x16       { panic("NEON not available") }
func (v Uint8x16) And(other Uint8x16) Uint8x16         { panic("NEON not available") }
func (v Uint8x16) Or(other Uint8x16) Uint8x16          { panic("NEON not available") }
func (v Uint8x16) Xor(other Uint8x16) Uint8x16         { panic("NEON not available") }
func (v Uint8x16) Not() Uint8x16                       { panic("NEON not available") }
func (v Uint8x16) TableLookupBytes(idx Uint8x16) Uint8x16 {
	// Scalar fallback implementation
	var result [16]uint8
	for i := 0; i < 16; i++ {
		index := idx[i]
		if index < 16 {
			result[i] = v[index]
		}
		// Out-of-range indices produce 0 (already zero from array initialization)
	}
	return *(*Uint8x16)(unsafe.Pointer(&result))
}

// ===== Uint16x8 stub methods =====

func BroadcastUint16x8(v uint16) Uint16x8 {
	arr := [8]uint16{v, v, v, v, v, v, v, v}
	return *(*Uint16x8)(unsafe.Pointer(&arr))
}

func LoadUint16x8(s []uint16) Uint16x8      { return *(*Uint16x8)(unsafe.Pointer(&s[0])) }
func ZeroUint16x8() Uint16x8                { return Uint16x8{} }
func (v Uint16x8) Get(i int) uint16         { return (*[8]uint16)(unsafe.Pointer(&v))[i] }
func (v *Uint16x8) Set(i int, val uint16)   { (*[8]uint16)(unsafe.Pointer(v))[i] = val }
func (v Uint16x8) Data() []uint16           { return (*[8]uint16)(unsafe.Pointer(&v))[:] }
func (v Uint16x8) StoreSlice(s []uint16)    { *(*Uint16x8)(unsafe.Pointer(&s[0])) = v }
func (v Uint16x8) GetBit(i int) bool        { return (*[8]uint16)(unsafe.Pointer(&v))[i] != 0 }
func (v Uint16x8) Add(other Uint16x8) Uint16x8         { panic("NEON not available") }
func (v Uint16x8) Sub(other Uint16x8) Uint16x8         { panic("NEON not available") }
func (v Uint16x8) AddSaturated(other Uint16x8) Uint16x8 { panic("NEON not available") }
func (v Uint16x8) SubSaturated(other Uint16x8) Uint16x8 { panic("NEON not available") }
func (v Uint16x8) Min(other Uint16x8) Uint16x8         { panic("NEON not available") }
func (v Uint16x8) Max(other Uint16x8) Uint16x8         { panic("NEON not available") }
func (v Uint16x8) LessThan(other Uint16x8) Uint16x8    { panic("NEON not available") }
func (v Uint16x8) GreaterThan(other Uint16x8) Uint16x8 { panic("NEON not available") }
func (v Uint16x8) LessEqual(other Uint16x8) Uint16x8   { panic("NEON not available") }
func (v Uint16x8) GreaterEqual(other Uint16x8) Uint16x8 { panic("NEON not available") }
func (v Uint16x8) Equal(other Uint16x8) Uint16x8       { panic("NEON not available") }
func (v Uint16x8) And(other Uint16x8) Uint16x8         { panic("NEON not available") }
func (v Uint16x8) Or(other Uint16x8) Uint16x8          { panic("NEON not available") }
func (v Uint16x8) Xor(other Uint16x8) Uint16x8         { panic("NEON not available") }
func (v Uint16x8) Not() Uint16x8                       { panic("NEON not available") }

// ===== Uint32x4 stub methods =====

func BroadcastUint32x4(v uint32) Uint32x4 {
	arr := [4]uint32{v, v, v, v}
	return *(*Uint32x4)(unsafe.Pointer(&arr))
}

func LoadUint32x4(s []uint32) Uint32x4      { return *(*Uint32x4)(unsafe.Pointer(&s[0])) }
func LoadUint32x4Slice(s []uint32) Uint32x4 { return LoadUint32x4(s) }
func ZeroUint32x4() Uint32x4                { return Uint32x4{} }
func (v Uint32x4) Get(i int) uint32         { return (*[4]uint32)(unsafe.Pointer(&v))[i] }
func (v *Uint32x4) Set(i int, val uint32)   { (*[4]uint32)(unsafe.Pointer(v))[i] = val }
func (v Uint32x4) Data() []uint32           { return (*[4]uint32)(unsafe.Pointer(&v))[:] }
func (v Uint32x4) StoreSlice(s []uint32)    { *(*Uint32x4)(unsafe.Pointer(&s[0])) = v }
func (v Uint32x4) GetBit(i int) bool        { return (*[4]uint32)(unsafe.Pointer(&v))[i] != 0 }
func (v Uint32x4) AsInt32x4() Int32x4       { return Int32x4(v) }
func (v Uint32x4) Add(other Uint32x4) Uint32x4         { panic("NEON not available") }
func (v Uint32x4) Sub(other Uint32x4) Uint32x4         { panic("NEON not available") }
func (v Uint32x4) Mul(other Uint32x4) Uint32x4         { panic("NEON not available") }
func (v Uint32x4) AddSaturated(other Uint32x4) Uint32x4 { panic("NEON not available") }
func (v Uint32x4) SubSaturated(other Uint32x4) Uint32x4 { panic("NEON not available") }
func (v Uint32x4) Min(other Uint32x4) Uint32x4         { panic("NEON not available") }
func (v Uint32x4) Max(other Uint32x4) Uint32x4         { panic("NEON not available") }
func (v Uint32x4) LessThan(other Uint32x4) Uint32x4    { panic("NEON not available") }
func (v Uint32x4) GreaterThan(other Uint32x4) Uint32x4 { panic("NEON not available") }
func (v Uint32x4) LessEqual(other Uint32x4) Uint32x4   { panic("NEON not available") }
func (v Uint32x4) GreaterEqual(other Uint32x4) Uint32x4 { panic("NEON not available") }
func (v Uint32x4) Equal(other Uint32x4) Uint32x4       { panic("NEON not available") }
func (v Uint32x4) And(other Uint32x4) Uint32x4         { panic("NEON not available") }
func (v Uint32x4) Or(other Uint32x4) Uint32x4          { panic("NEON not available") }
func (v Uint32x4) Xor(other Uint32x4) Uint32x4         { panic("NEON not available") }
func (v Uint32x4) Not() Uint32x4                       { panic("NEON not available") }
func (v Uint32x4) AndNot(other Uint32x4) Uint32x4      { panic("NEON not available") }
func (v Uint32x4) ShiftAllLeft(count int) Uint32x4     { panic("NEON not available") }
func (v Uint32x4) ShiftAllRight(count int) Uint32x4    { panic("NEON not available") }
func (v Uint32x4) ReduceSum() uint64                   { panic("NEON not available") }
func (v Uint32x4) ReduceMax() uint32                   { panic("NEON not available") }

// ===== Uint64x2 stub methods =====

func BroadcastUint64x2(v uint64) Uint64x2 {
	arr := [2]uint64{v, v}
	return *(*Uint64x2)(unsafe.Pointer(&arr))
}

func LoadUint64x2(s []uint64) Uint64x2      { return *(*Uint64x2)(unsafe.Pointer(&s[0])) }
func LoadUint64x2Slice(s []uint64) Uint64x2 { return LoadUint64x2(s) }
func ZeroUint64x2() Uint64x2                { return Uint64x2{} }
func (v Uint64x2) Get(i int) uint64         { return (*[2]uint64)(unsafe.Pointer(&v))[i] }
func (v *Uint64x2) Set(i int, val uint64)   { (*[2]uint64)(unsafe.Pointer(v))[i] = val }
func (v Uint64x2) Data() []uint64           { return (*[2]uint64)(unsafe.Pointer(&v))[:] }
func (v Uint64x2) StoreSlice(s []uint64)    { *(*Uint64x2)(unsafe.Pointer(&s[0])) = v }
func (v Uint64x2) GetBit(i int) bool        { return (*[2]uint64)(unsafe.Pointer(&v))[i] != 0 }
func (v Uint64x2) AsInt64x2() Int64x2       { return Int64x2(v) }
func (v Uint64x2) Add(other Uint64x2) Uint64x2         { panic("NEON not available") }
func (v Uint64x2) Sub(other Uint64x2) Uint64x2         { panic("NEON not available") }
func (v Uint64x2) Mul(other Uint64x2) Uint64x2         { panic("NEON not available") }
func (v Uint64x2) AddSaturated(other Uint64x2) Uint64x2 { panic("NEON not available") }
func (v Uint64x2) SubSaturated(other Uint64x2) Uint64x2 { panic("NEON not available") }
func (v Uint64x2) Min(other Uint64x2) Uint64x2         { panic("NEON not available") }
func (v Uint64x2) Max(other Uint64x2) Uint64x2         { panic("NEON not available") }
func (v Uint64x2) LessThan(other Uint64x2) Uint64x2    { panic("NEON not available") }
func (v Uint64x2) GreaterThan(other Uint64x2) Uint64x2 { panic("NEON not available") }
func (v Uint64x2) LessEqual(other Uint64x2) Uint64x2   { panic("NEON not available") }
func (v Uint64x2) GreaterEqual(other Uint64x2) Uint64x2 { panic("NEON not available") }
func (v Uint64x2) Equal(other Uint64x2) Uint64x2       { panic("NEON not available") }
func (v Uint64x2) And(other Uint64x2) Uint64x2         { panic("NEON not available") }
func (v Uint64x2) Or(other Uint64x2) Uint64x2          { panic("NEON not available") }
func (v Uint64x2) Xor(other Uint64x2) Uint64x2         { panic("NEON not available") }
func (v Uint64x2) Not() Uint64x2                       { panic("NEON not available") }
func (v Uint64x2) ShiftAllLeft(count int) Uint64x2     { panic("NEON not available") }
func (v Uint64x2) ShiftAllRight(count int) Uint64x2    { panic("NEON not available") }
func (v Uint64x2) Merge(other Uint64x2, mask Uint64x2) Uint64x2 { panic("NEON not available") }
func (v Uint64x2) ReduceMax() uint64                            { panic("NEON not available") }

// ===== Int32x4 stub methods =====

func BroadcastInt32x4(v int32) Int32x4 {
	arr := [4]int32{v, v, v, v}
	return *(*Int32x4)(unsafe.Pointer(&arr))
}

func LoadInt32x4(s []int32) Int32x4      { return *(*Int32x4)(unsafe.Pointer(&s[0])) }
func LoadInt32x4Slice(s []int32) Int32x4 { return LoadInt32x4(s) }
func ZeroInt32x4() Int32x4               { return Int32x4{} }
func (v Int32x4) Get(i int) int32        { return (*[4]int32)(unsafe.Pointer(&v))[i] }
func (v *Int32x4) Set(i int, val int32)  { (*[4]int32)(unsafe.Pointer(v))[i] = val }
func (v Int32x4) Data() []int32          { return (*[4]int32)(unsafe.Pointer(&v))[:] }
func (v Int32x4) StoreSlice(s []int32)   { *(*Int32x4)(unsafe.Pointer(&s[0])) = v }
func (v Int32x4) GetBit(i int) bool      { return (*[4]int32)(unsafe.Pointer(&v))[i] != 0 }
func (v Int32x4) Add(other Int32x4) Int32x4     { panic("NEON not available") }
func (v Int32x4) Sub(other Int32x4) Int32x4     { panic("NEON not available") }
func (v Int32x4) Mul(other Int32x4) Int32x4     { panic("NEON not available") }
func (v Int32x4) Min(other Int32x4) Int32x4     { panic("NEON not available") }
func (v Int32x4) Max(other Int32x4) Int32x4     { panic("NEON not available") }
func (v Int32x4) Abs() Int32x4                  { panic("NEON not available") }
func (v Int32x4) Neg() Int32x4                  { panic("NEON not available") }
func (v Int32x4) And(other Int32x4) Int32x4     { panic("NEON not available") }
func (v Int32x4) Or(other Int32x4) Int32x4      { panic("NEON not available") }
func (v Int32x4) Xor(other Int32x4) Int32x4     { panic("NEON not available") }
func (v Int32x4) Not() Int32x4                  { panic("NEON not available") }
func (v Int32x4) ReduceSum() int64              { panic("NEON not available") }
func (v Int32x4) ReduceMax() int32              { panic("NEON not available") }
func (v Int32x4) ReduceMin() int32              { panic("NEON not available") }

// ===== Int64x2 stub methods =====

func BroadcastInt64x2(v int64) Int64x2 {
	arr := [2]int64{v, v}
	return *(*Int64x2)(unsafe.Pointer(&arr))
}

func LoadInt64x2(s []int64) Int64x2      { return *(*Int64x2)(unsafe.Pointer(&s[0])) }
func LoadInt64x2Slice(s []int64) Int64x2 { return LoadInt64x2(s) }
func ZeroInt64x2() Int64x2               { return Int64x2{} }
func (v Int64x2) Get(i int) int64        { return (*[2]int64)(unsafe.Pointer(&v))[i] }
func (v *Int64x2) Set(i int, val int64)  { (*[2]int64)(unsafe.Pointer(v))[i] = val }
func (v Int64x2) Data() []int64          { return (*[2]int64)(unsafe.Pointer(&v))[:] }
func (v Int64x2) StoreSlice(s []int64)   { *(*Int64x2)(unsafe.Pointer(&s[0])) = v }
func (v Int64x2) GetBit(i int) bool      { return (*[2]int64)(unsafe.Pointer(&v))[i] != 0 }
func (v Int64x2) Add(other Int64x2) Int64x2     { panic("NEON not available") }
func (v Int64x2) Sub(other Int64x2) Int64x2     { panic("NEON not available") }
func (v Int64x2) Mul(other Int64x2) Int64x2     { panic("NEON not available") }
func (v Int64x2) Min(other Int64x2) Int64x2     { panic("NEON not available") }
func (v Int64x2) Max(other Int64x2) Int64x2     { panic("NEON not available") }
func (v Int64x2) And(other Int64x2) Int64x2     { panic("NEON not available") }
func (v Int64x2) Or(other Int64x2) Int64x2      { panic("NEON not available") }
func (v Int64x2) Xor(other Int64x2) Int64x2     { panic("NEON not available") }
func (v Int64x2) ReduceMax() int64              { panic("NEON not available") }
func (v Int64x2) ReduceMin() int64              { panic("NEON not available") }

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

// Float64 comparison operations
func EqF64(a, b []float64, result []int64) { panic("NEON not available") }
func GtF64(a, b []float64, result []int64) { panic("NEON not available") }
func GeF64(a, b []float64, result []int64) { panic("NEON not available") }
func LtF64(a, b []float64, result []int64) { panic("NEON not available") }
func LeF64(a, b []float64, result []int64) { panic("NEON not available") }

// Power of 2 operations
func Pow2F32(k []int32, result []float32) { panic("NEON not available") }
func Pow2F64(k []int32, result []float64) { panic("NEON not available") }

// Bitwise operations (Phase 8)
func AndI32(a, b, result []int32)                  { panic("NEON not available") }
func OrI32(a, b, result []int32)                   { panic("NEON not available") }
func XorI32(a, b, result []int32)                  { panic("NEON not available") }
func AndNotI32(a, b, result []int32)               { panic("NEON not available") }
func NotI32(a, result []int32)                     { panic("NEON not available") }
func ShiftLeftI32(a []int32, shift int, result []int32)  { panic("NEON not available") }
func ShiftRightI32(a []int32, shift int, result []int32) { panic("NEON not available") }

// Mask operations (Phase 9)
func IfThenElseF32(mask []int32, yes, no, result []float32) { panic("NEON not available") }
func IfThenElseI32(mask, yes, no, result []int32)           { panic("NEON not available") }
func CountTrueI32(mask []int32) int64                       { panic("NEON not available") }
func AllTrueI32(mask []int32) bool                          { panic("NEON not available") }
func AllFalseI32(mask []int32) bool                         { panic("NEON not available") }
func FirstNI32(count int, result []int32)                   { panic("NEON not available") }
func CompressF32(input []float32, mask []int32, result []float32) int64 {
	panic("NEON not available")
}
func ExpandF32(input []float32, mask []int32, result []float32) { panic("NEON not available") }

// Transcendental math operations (Phase 10)
func ExpF32(input, result []float32)     { panic("NEON not available") }
func LogF32(input, result []float32)     { panic("NEON not available") }
func SinF32(input, result []float32)     { panic("NEON not available") }
func CosF32(input, result []float32)     { panic("NEON not available") }
func TanhF32(input, result []float32)    { panic("NEON not available") }
func SigmoidF32(input, result []float32) { panic("NEON not available") }

// Int32 arithmetic operations
func AddI32(a, b, result []int32) { panic("NEON not available") }
func SubI32(a, b, result []int32) { panic("NEON not available") }
func MulI32(a, b, result []int32) { panic("NEON not available") }

// Int64 operations
func AddI64(a, b, result []int64)                       { panic("NEON not available") }
func SubI64(a, b, result []int64)                       { panic("NEON not available") }
func AndI64(a, b, result []int64)                       { panic("NEON not available") }
func OrI64(a, b, result []int64)                        { panic("NEON not available") }
func XorI64(a, b, result []int64)                       { panic("NEON not available") }
func ShiftLeftI64(a []int64, shift int, result []int64) { panic("NEON not available") }
func ShiftRightI64(a []int64, shift int, result []int64) {
	panic("NEON not available")
}
func EqI64(a, b, result []int64)                  { panic("NEON not available") }
func GtI64(a, b, result []int64)                  { panic("NEON not available") }
func GeI64(a, b, result []int64)                  { panic("NEON not available") }
func LtI64(a, b, result []int64)                  { panic("NEON not available") }
func LeI64(a, b, result []int64)                  { panic("NEON not available") }
func IfThenElseI64(mask, yes, no, result []int64) { panic("NEON not available") }

// Single-vector operations (exported)
func FindFirstTrueI32x4(mask *[4]int32) int      { panic("NEON not available") }
func FindFirstTrueI64x2(mask *[2]int64) int      { panic("NEON not available") }
func CountTrueI32x4(mask *[4]int32) int          { panic("NEON not available") }
func CountTrueI64x2(mask *[2]int64) int          { panic("NEON not available") }
func EqF32x4(a, b, result *[4]float32)           { panic("NEON not available") }
func EqI32x4(a, b, result *[4]int32)             { panic("NEON not available") }
func EqF64x2(a, b *[2]float64, result *[2]int64) { panic("NEON not available") }
func EqI64x2(a, b, result *[2]int64)             { panic("NEON not available") }

// Internal asm functions (called by vec_neon.go on arm64)
func allTrueI32x4Asm(mask *[4]int32) bool        { return false }
func allTrueI64x2Asm(mask *[2]int64) bool        { return false }
func allFalseI32x4Asm(mask *[4]int32) bool       { return true }
func allFalseI64x2Asm(mask *[2]int64) bool       { return true }
func firstNI32x4Asm(count int, result *[4]int32) {}
func firstNI64x2Asm(count int, result *[2]int64) {}

// ============================================================================
// Unsigned integer vector stubs
// ============================================================================

// Uint8x16 stubs
func lt_u8x16(a, b [16]byte) [16]byte   { panic("NEON not available") }
func gt_u8x16(a, b [16]byte) [16]byte   { panic("NEON not available") }
func le_u8x16(a, b [16]byte) [16]byte   { panic("NEON not available") }
func ge_u8x16(a, b [16]byte) [16]byte   { panic("NEON not available") }
func eq_u8x16(a, b [16]byte) [16]byte   { panic("NEON not available") }
func min_u8x16(a, b [16]byte) [16]byte  { panic("NEON not available") }
func max_u8x16(a, b [16]byte) [16]byte  { panic("NEON not available") }
func adds_u8x16(a, b [16]byte) [16]byte { panic("NEON not available") }
func subs_u8x16(a, b [16]byte) [16]byte { panic("NEON not available") }
func and_u8x16(a, b [16]byte) [16]byte  { panic("NEON not available") }
func or_u8x16(a, b [16]byte) [16]byte   { panic("NEON not available") }
func xor_u8x16(a, b [16]byte) [16]byte  { panic("NEON not available") }
func not_u8x16(a [16]byte) [16]byte     { panic("NEON not available") }

// Uint16x8 stubs
func lt_u16x8(a, b [16]byte) [16]byte   { panic("NEON not available") }
func gt_u16x8(a, b [16]byte) [16]byte   { panic("NEON not available") }
func le_u16x8(a, b [16]byte) [16]byte   { panic("NEON not available") }
func ge_u16x8(a, b [16]byte) [16]byte   { panic("NEON not available") }
func eq_u16x8(a, b [16]byte) [16]byte   { panic("NEON not available") }
func min_u16x8(a, b [16]byte) [16]byte  { panic("NEON not available") }
func max_u16x8(a, b [16]byte) [16]byte  { panic("NEON not available") }
func adds_u16x8(a, b [16]byte) [16]byte { panic("NEON not available") }
func subs_u16x8(a, b [16]byte) [16]byte { panic("NEON not available") }
func and_u16x8(a, b [16]byte) [16]byte  { panic("NEON not available") }
func or_u16x8(a, b [16]byte) [16]byte   { panic("NEON not available") }
func xor_u16x8(a, b [16]byte) [16]byte  { panic("NEON not available") }
func not_u16x8(a [16]byte) [16]byte     { panic("NEON not available") }

// Uint32x4 stubs
func add_u32x4(a, b [16]byte) [16]byte    { panic("NEON not available") }
func sub_u32x4(a, b [16]byte) [16]byte    { panic("NEON not available") }
func mul_u32x4(a, b [16]byte) [16]byte    { panic("NEON not available") }
func lt_u32x4(a, b [16]byte) [16]byte     { panic("NEON not available") }
func gt_u32x4(a, b [16]byte) [16]byte     { panic("NEON not available") }
func le_u32x4(a, b [16]byte) [16]byte     { panic("NEON not available") }
func ge_u32x4(a, b [16]byte) [16]byte     { panic("NEON not available") }
func eq_u32x4(a, b [16]byte) [16]byte     { panic("NEON not available") }
func min_u32x4(a, b [16]byte) [16]byte    { panic("NEON not available") }
func max_u32x4(a, b [16]byte) [16]byte    { panic("NEON not available") }
func adds_u32x4(a, b [16]byte) [16]byte   { panic("NEON not available") }
func subs_u32x4(a, b [16]byte) [16]byte   { panic("NEON not available") }
func and_u32x4(a, b [16]byte) [16]byte    { panic("NEON not available") }
func or_u32x4(a, b [16]byte) [16]byte     { panic("NEON not available") }
func xor_u32x4(a, b [16]byte) [16]byte    { panic("NEON not available") }
func not_u32x4(a [16]byte) [16]byte       { panic("NEON not available") }
func andnot_u32x4(a, b [16]byte) [16]byte { panic("NEON not available") }
func hsum_u32x4(v [16]byte) int64         { panic("NEON not available") }

// Uint64x2 stubs
func add_u64x2(a, b [16]byte) [16]byte          { panic("NEON not available") }
func sub_u64x2(a, b [16]byte) [16]byte          { panic("NEON not available") }
func lt_u64x2(a, b [16]byte) [16]byte           { panic("NEON not available") }
func gt_u64x2(a, b [16]byte) [16]byte           { panic("NEON not available") }
func le_u64x2(a, b [16]byte) [16]byte           { panic("NEON not available") }
func ge_u64x2(a, b [16]byte) [16]byte           { panic("NEON not available") }
func eq_u64x2(a, b [16]byte) [16]byte           { panic("NEON not available") }
func min_u64x2(a, b [16]byte) [16]byte          { panic("NEON not available") }
func max_u64x2(a, b [16]byte) [16]byte          { panic("NEON not available") }
func adds_u64x2(a, b [16]byte) [16]byte         { panic("NEON not available") }
func subs_u64x2(a, b [16]byte) [16]byte         { panic("NEON not available") }
func and_u64x2(a, b [16]byte) [16]byte          { panic("NEON not available") }
func or_u64x2(a, b [16]byte) [16]byte           { panic("NEON not available") }
func xor_u64x2(a, b [16]byte) [16]byte          { panic("NEON not available") }
func sel_u64x2(mask, yes, no [16]byte) [16]byte { panic("NEON not available") }

// SlideUpLanes stubs
func SlideUpLanesFloat32x4(v Float32x4, offset int) Float32x4 { panic("NEON not available") }
func SlideUpLanesFloat64x2(v Float64x2, offset int) Float64x2 { panic("NEON not available") }
func SlideUpLanesInt32x4(v Int32x4, offset int) Int32x4       { panic("NEON not available") }
func SlideUpLanesInt64x2(v Int64x2, offset int) Int64x2       { panic("NEON not available") }
func SlideUpLanesUint32x4(v Uint32x4, offset int) Uint32x4    { panic("NEON not available") }
func SlideUpLanesUint64x2(v Uint64x2, offset int) Uint64x2    { panic("NEON not available") }

// InsertLane stubs
func InsertLaneFloat32x4(v Float32x4, lane int, val float32) Float32x4 { panic("NEON not available") }
func InsertLaneFloat64x2(v Float64x2, lane int, val float64) Float64x2 { panic("NEON not available") }
func InsertLaneInt32x4(v Int32x4, lane int, val int32) Int32x4         { panic("NEON not available") }
func InsertLaneInt64x2(v Int64x2, lane int, val int64) Int64x2         { panic("NEON not available") }
func InsertLaneUint32x4(v Uint32x4, lane int, val uint32) Uint32x4     { panic("NEON not available") }
func InsertLaneUint64x2(v Uint64x2, lane int, val uint64) Uint64x2     { panic("NEON not available") }
