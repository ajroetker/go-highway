# Implementation Plan: Shuffle, Type Conversions, and Compress/Expand

## Overview

Add three major feature categories to go-highway to close gaps with C++ Highway:
1. **Shuffle/Permutation Operations** - lane reordering and manipulation
2. **Type Conversions** - widening, narrowing, and float/int conversion
3. **Compress/Expand** - mask-based filtering operations

## Key Findings from Exploration

### Available archsimd Operations
- `Reverse()` - reverses lane order (AVX2/AVX-512)
- `Merge(b, mask)` - conditional select (like blend/IfThenElse)
- `Broadcast` - replicate scalar to all lanes
- `ConvertToInt32()`, `ConvertToFloat32()` - type conversion
- `AsFloat32x8()`, `AsInt32x8()` - bitcast (reinterpret bits)
- `ShiftAllLeft()`, `ShiftAllRight()` - bit shifts

### NOT Available in archsimd
- Arbitrary shuffle/permute (no vpermps, vpermd)
- Compress/Expand (no vcompressps, vexpandps)
- Interleave instructions (no vpunpcklps, etc.)

### Implementation Strategy
- Use archsimd intrinsics where available
- Implement scalar fallbacks for everything
- Use hwygen for operations that benefit from code generation
- Add operations directly to hwy package (not hwygen) for core primitives

---

## Phase 1: Shuffle/Permutation Operations

### 1.1 Operations to Implement

| Operation | Description | archsimd? | Approach |
|-----------|-------------|-----------|----------|
| `Reverse` | Reverse all lanes | ✅ Yes | Direct intrinsic |
| `Reverse2` | Reverse pairs | ❌ No | Scalar fallback |
| `Reverse4` | Reverse groups of 4 | ❌ No | Scalar fallback |
| `BroadcastLane` | Broadcast lane N to all | ❌ No | Store + Broadcast |
| `GetLane` | Extract single lane | ❌ No | Store + index |
| `InsertLane` | Insert value at lane | ❌ No | Store + modify + Load |
| `InterleaveLower` | Interleave lower halves | ❌ No | Scalar fallback |
| `InterleaveUpper` | Interleave upper halves | ❌ No | Scalar fallback |
| `ConcatLowerLower` | Concat lower halves | ❌ No | Scalar fallback |
| `OddEven` | Combine odd/even lanes | ❌ No | Merge with mask |

### 1.2 File Changes

**New file: `hwy/shuffle.go`** (scalar fallbacks)
```go
// Reverse reverses lane order
func Reverse[T Lanes](v Vec[T]) Vec[T]

// Reverse2 reverses pairs of lanes [0,1,2,3] -> [1,0,3,2]
func Reverse2[T Lanes](v Vec[T]) Vec[T]

// Reverse4 reverses groups of 4 [0,1,2,3,4,5,6,7] -> [3,2,1,0,7,6,5,4]
func Reverse4[T Lanes](v Vec[T]) Vec[T]

// GetLane extracts lane at index
func GetLane[T Lanes](v Vec[T], idx int) T

// InsertLane inserts value at lane index
func InsertLane[T Lanes](v Vec[T], idx int, val T) Vec[T]

// BroadcastLane broadcasts lane at index to all lanes
func BroadcastLane[T Lanes](v Vec[T], idx int) Vec[T]

// InterleaveLower interleaves lower halves: [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a0,b0,a1,b1]
func InterleaveLower[T Lanes](a, b Vec[T]) Vec[T]

// InterleaveUpper interleaves upper halves: [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a2,b2,a3,b3]
func InterleaveUpper[T Lanes](a, b Vec[T]) Vec[T]

// OddEven combines odd lanes from a with even lanes from b
func OddEven[T Lanes](a, b Vec[T]) Vec[T]
```

**New file: `hwy/shuffle_avx2.go`** (SIMD implementations)
```go
//go:build amd64 && goexperiment.simd

func Reverse_AVX2_F32x8(v archsimd.Float32x8) archsimd.Float32x8
func Reverse_AVX2_F64x4(v archsimd.Float64x4) archsimd.Float64x4
// ... etc
```

**New file: `hwy/shuffle_avx512.go`** (AVX-512 implementations)

**Update: `cmd/hwygen/targets.go`** - Add shuffle ops to OpMap

### 1.3 Test File

**New file: `hwy/shuffle_test.go`**
- Test each operation with known input/output
- Test edge cases (single element, empty)
- Benchmark SIMD vs scalar

---

## Phase 2: Type Conversions

### 2.1 Operations to Implement

| Operation | Description | archsimd? | Approach |
|-----------|-------------|-----------|----------|
| `ConvertTo[U]` | Convert float↔int | ✅ Yes | Direct intrinsic |
| `PromoteTo[U]` | Widen (f32→f64, i16→i32) | ❌ No | Scalar + archsimd partial |
| `DemoteTo[U]` | Narrow (f64→f32, i32→i16) | ❌ No | Scalar + archsimd partial |
| `BitCast[U]` | Reinterpret bits | ✅ Yes | AsFloat32x8, etc. |
| `Round` | Round to nearest int | ❌ No | Scalar math.Round |
| `Trunc` | Truncate toward zero | ❌ No | Scalar math.Trunc |
| `Ceil` | Round up | ❌ No | Scalar math.Ceil |
| `Floor` | Round down | ❌ No | Scalar math.Floor |

### 2.2 File Changes

**New file: `hwy/convert.go`** (scalar fallbacks)
```go
// ConvertTo converts between float and int types
func ConvertTo[U, T Lanes](v Vec[T]) Vec[U]

// PromoteTo widens elements (lower half of input fills output)
func PromoteTo[U, T Lanes](v Vec[T]) Vec[U]

// DemoteTo narrows elements (output is half the lanes)
func DemoteTo[U, T Lanes](v Vec[T]) Vec[U]

// BitCast reinterprets bits as different type (same size required)
func BitCast[U, T Lanes](v Vec[T]) Vec[U]

// Round rounds to nearest integer (float only)
func Round[T Floats](v Vec[T]) Vec[T]

// Trunc truncates toward zero (float only)
func Trunc[T Floats](v Vec[T]) Vec[T]

// Ceil rounds up (float only)
func Ceil[T Floats](v Vec[T]) Vec[T]

// Floor rounds down (float only)
func Floor[T Floats](v Vec[T]) Vec[T]
```

**New file: `hwy/convert_avx2.go`**
```go
//go:build amd64 && goexperiment.simd

// Use archsimd ConvertToInt32, ConvertToFloat32, etc.
func ConvertToInt32_AVX2_F32x8(v archsimd.Float32x8) archsimd.Int32x8
func ConvertToFloat32_AVX2_I32x8(v archsimd.Int32x8) archsimd.Float32x8
func BitCast_AVX2_F32x8_I32x8(v archsimd.Float32x8) archsimd.Int32x8
// ... etc
```

**New file: `hwy/convert_avx512.go`**

**Update: `cmd/hwygen/targets.go`** - Add conversion ops to OpMap

### 2.3 Test File

**New file: `hwy/convert_test.go`**
- Test float→int truncation behavior
- Test int→float precision
- Test PromoteTo/DemoteTo lane counts
- Test BitCast preserves bits

---

## Phase 3: Compress/Expand Operations

### 3.1 Operations to Implement

| Operation | Description | archsimd? | Approach |
|-----------|-------------|-----------|----------|
| `Compress` | Pack lanes where mask=true | ❌ No | Scalar loop |
| `Expand` | Unpack into lanes where mask=true | ❌ No | Scalar loop |
| `CompressStore` | Compress and store to memory | ❌ No | Compress + Store |
| `CountTrue` | Count true lanes in mask | ❌ No | Scalar count |
| `AllTrue` | All lanes true? | ❌ No | Scalar check |
| `AllFalse` | All lanes false? | ❌ No | Scalar check |
| `FindFirstTrue` | Index of first true lane | ❌ No | Scalar search |

### 3.2 File Changes

**New file: `hwy/compress.go`** (scalar implementations)
```go
// Compress packs elements where mask is true to the front
// Returns compressed vector and count of valid elements
func Compress[T Lanes](v Vec[T], mask Mask[T]) (Vec[T], int)

// Expand unpacks elements into positions where mask is true
// Elements from v fill true positions, false positions get zero
func Expand[T Lanes](v Vec[T], mask Mask[T]) Vec[T]

// CompressStore compresses and stores directly to slice
// Returns number of elements stored
func CompressStore[T Lanes](v Vec[T], mask Mask[T], dst []T) int

// CountTrue counts true lanes in mask
func CountTrue[T Lanes](mask Mask[T]) int

// AllTrue returns true if all lanes are true
func AllTrue[T Lanes](mask Mask[T]) bool

// AllFalse returns true if all lanes are false
func AllFalse[T Lanes](mask Mask[T]) bool

// FindFirstTrue returns index of first true lane, or -1 if none
func FindFirstTrue[T Lanes](mask Mask[T]) int

// FindLastTrue returns index of last true lane, or -1 if none
func FindLastTrue[T Lanes](mask Mask[T]) int
```

**New file: `hwy/compress_avx2.go`** (optimized where possible)
```go
//go:build amd64 && goexperiment.simd

// Note: No direct archsimd support for compress/expand
// Implement using store + scalar or lookup tables
```

**New file: `hwy/compress_avx512.go`**
```go
// AVX-512 has native vcompressps/vexpandps but not exposed in archsimd
// Use scalar fallback for now, optimize later if archsimd adds support
```

### 3.3 Test File

**New file: `hwy/compress_test.go`**
- Test Compress with various mask patterns
- Test Expand inverse of Compress
- Test edge cases (all true, all false, alternating)
- Benchmark vs naive filtering

---

## Phase 4: hwygen Integration

### 4.1 Update targets.go OpMap

Add new operations to both AVX2Target() and AVX512Target():

```go
// Shuffle operations
"Reverse":         {Name: "Reverse", IsMethod: true},
"Reverse2":        {Name: "Reverse2", IsMethod: false},
"GetLane":         {Name: "GetLane", IsMethod: false},
"BroadcastLane":   {Name: "BroadcastLane", IsMethod: false},
"InterleaveLower": {Name: "InterleaveLower", IsMethod: false},
"InterleaveUpper": {Name: "InterleaveUpper", IsMethod: false},

// Conversion operations
"ConvertToInt":    {Name: "ConvertToInt32", IsMethod: true},
"ConvertToFloat":  {Name: "ConvertToFloat32", IsMethod: true},
"BitCast":         {Name: "BitCast", IsMethod: false},
"Round":           {Name: "Round", IsMethod: false},
"Floor":           {Name: "Floor", IsMethod: false},
"Ceil":            {Name: "Ceil", IsMethod: false},

// Compress operations
"Compress":        {Name: "Compress", IsMethod: false},
"CountTrue":       {Name: "CountTrue", IsMethod: false},
```

---

## Implementation Order

### Step 1: Shuffle Operations (Core)
1. Create `hwy/shuffle.go` with scalar implementations
2. Create `hwy/shuffle_avx2.go` with Reverse using archsimd
3. Create `hwy/shuffle_avx512.go` with Reverse using archsimd
4. Create `hwy/shuffle_test.go`
5. Run tests: `GOEXPERIMENT=simd go test ./hwy -run Shuffle`

### Step 2: Type Conversions
1. Create `hwy/convert.go` with scalar implementations
2. Create `hwy/convert_avx2.go` using archsimd Convert/BitCast
3. Create `hwy/convert_avx512.go`
4. Create `hwy/convert_test.go`
5. Run tests: `GOEXPERIMENT=simd go test ./hwy -run Convert`

### Step 3: Compress/Expand
1. Create `hwy/compress.go` with scalar implementations
2. Create `hwy/compress_avx2.go` (mostly scalar, mask ops optimized)
3. Create `hwy/compress_avx512.go`
4. Create `hwy/compress_test.go`
5. Run tests: `GOEXPERIMENT=simd go test ./hwy -run Compress`

### Step 4: hwygen Integration
1. Update `cmd/hwygen/targets.go` with new OpMap entries
2. Test with examples/gelu to ensure no regressions
3. Run full test suite: `GOEXPERIMENT=simd go test ./...`

### Step 5: Documentation
1. Update `specs/feature-gaps.md` to mark completed features
2. Add usage examples to README

---

## Verification

### Unit Tests
```bash
GOEXPERIMENT=simd go test ./hwy -v -run "Shuffle|Convert|Compress"
```

### Scalar Fallback Tests
```bash
HWY_NO_SIMD=1 GOEXPERIMENT=simd go test ./hwy -v -run "Shuffle|Convert|Compress"
```

### Full Test Suite
```bash
GOEXPERIMENT=simd go test ./...
```

### Benchmarks
```bash
GOEXPERIMENT=simd go test ./hwy -bench="Shuffle|Convert|Compress" -benchmem
```

---

## Files to Create/Modify

### New Files (12)
- `hwy/shuffle.go` - Scalar shuffle implementations
- `hwy/shuffle_avx2.go` - AVX2 shuffle implementations
- `hwy/shuffle_avx512.go` - AVX-512 shuffle implementations
- `hwy/shuffle_test.go` - Shuffle tests
- `hwy/convert.go` - Scalar conversion implementations
- `hwy/convert_avx2.go` - AVX2 conversion implementations
- `hwy/convert_avx512.go` - AVX-512 conversion implementations
- `hwy/convert_test.go` - Conversion tests
- `hwy/compress.go` - Scalar compress/expand implementations
- `hwy/compress_avx2.go` - AVX2 compress implementations
- `hwy/compress_avx512.go` - AVX-512 compress implementations
- `hwy/compress_test.go` - Compress tests

### Modified Files (2)
- `cmd/hwygen/targets.go` - Add new operations to OpMap
- `specs/feature-gaps.md` - Update with completed features
