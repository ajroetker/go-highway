# Feature Parity: go-highway vs C++ Highway

This document tracks feature parity between go-highway and Google's C++ Highway SIMD library.

## Architecture Support

### Current Status

| Architecture | C++ Highway | go-highway | Status |
|-------------|-------------|------------|--------|
| SSE2/SSE4 | ✅ | ⚠️ Baseline only | Limited |
| SSSE3 | ✅ | ❌ | Missing |
| AVX2 | ✅ | ✅ | Implemented |
| AVX-512 (AVX3) | ✅ | ✅ | Implemented |
| AVX3_DL, ZEN4, SPR | ✅ | ❌ | Missing |
| AVX10.2 | ✅ | ❌ | Missing |
| ARM NEON | ✅ | ✅ | Implemented |
| ARM SVE/SVE2 | ✅ | ❌ Planned | Missing |
| RISC-V (RVV) | ✅ | ❌ | Missing |
| WebAssembly SIMD | ✅ | ❌ Planned | Missing |
| PowerPC (PPC8-10) | ✅ | ❌ | Missing |
| IBM Z (Z14/Z15) | ✅ | ❌ | Missing |
| LoongArch (LSX/LASX) | ✅ | ❌ | Missing |

**C++ Highway: 27 targets | go-highway: 4 (Scalar, AVX2, AVX-512, NEON)**

### Priority for New Architectures

1. ~~**ARM NEON**~~ ✅ **IMPLEMENTED** (mobile, Apple Silicon, AWS Graviton)
2. **ARM SVE/SVE2** - High priority (modern ARM servers)
3. **WebAssembly SIMD** - Medium priority (browser deployment)
4. **RISC-V RVV** - Low priority (emerging)

---

## Operations

### Arithmetic Operations

| Operation | C++ Highway | go-highway | Notes |
|-----------|-------------|------------|-------|
| Add | ✅ | ✅ | |
| Sub | ✅ | ✅ | |
| Mul | ✅ | ✅ | |
| Div | ✅ | ✅ | Float only |
| Neg | ✅ | ✅ | |
| Abs | ✅ | ✅ | |
| Min | ✅ | ✅ | |
| Max | ✅ | ✅ | |
| FMA | ✅ | ✅ | |
| Sqrt | ✅ | ✅ | |
| MulHigh | ✅ | ✅ | High bits of widening multiply |
| SaturatedAdd | ✅ | ✅ | Clamp to type range |
| SaturatedSub | ✅ | ✅ | Clamp to type range |
| Avg | ✅ | ✅ | Rounded average |
| AbsDiff | ✅ | ✅ | Absolute difference |
| Clamp | ✅ | ✅ | Clamp to range |

### Comparison Operations ✅ **Complete**

| Operation | C++ Highway | go-highway | Notes |
|-----------|-------------|------------|-------|
| Equal | ✅ | ✅ | |
| NotEqual | ✅ | ✅ | |
| LessThan | ✅ | ✅ | |
| LessEqual | ✅ | ✅ | |
| GreaterThan | ✅ | ✅ | |
| GreaterEqual | ✅ | ✅ | |
| TestBit | ✅ | ✅ | Test if bit is set |
| IsNaN | ✅ | ✅ | Float NaN check |
| IsInf | ✅ | ✅ | Float infinity check |
| IsFinite | ✅ | ✅ | Float finite check |

### Memory Operations ✅ **Complete**

| Operation | C++ Highway | go-highway | Notes |
|-----------|-------------|------------|-------|
| Load | ✅ | ✅ | |
| Store | ✅ | ✅ | |
| LoadU (unaligned) | ✅ | ✅ | Go handles alignment |
| StoreU (unaligned) | ✅ | ✅ | Go handles alignment |
| MaskedLoad | ✅ | ✅ | |
| MaskedStore | ✅ | ✅ | |
| BlendedStore | ✅ | ✅ | Conditional store |
| Set (broadcast) | ✅ | ✅ | |
| Zero | ✅ | ✅ | |
| Undefined | ✅ | ✅ | Uninitialized vector |
| GatherIndex | ✅ | ✅ | Indexed load |
| ScatterIndex | ✅ | ✅ | Indexed store |
| GatherIndexMasked | ✅ | ✅ | Masked indexed load |
| ScatterIndexMasked | ✅ | ✅ | Masked indexed store |
| LoadDup128 | ✅ | ✅ | Load and duplicate |
| LoadInterleaved | ✅ | ✅ | AoS to SoA |
| StoreInterleaved | ✅ | ✅ | SoA to AoS |

### Shuffle/Permutation Operations ✅ **Complete**

| Operation | C++ Highway | go-highway | Notes |
|-----------|-------------|------------|-------|
| Reverse | ✅ | ✅ | Reverse lane order |
| Reverse2/4/8 | ✅ | ✅ | Reverse in groups |
| Shuffle0123 | ✅ | ✅ | 4-element shuffle |
| TableLookupBytes | ✅ | ✅ | Byte-level shuffle |
| TableLookupLanes | ✅ | ✅ | Lane-level shuffle |
| InterleaveLower | ✅ | ✅ | Interleave lower halves |
| InterleaveUpper | ✅ | ✅ | Interleave upper halves |
| ZipLower | ✅ | ✅ | Zip lower halves |
| ZipUpper | ✅ | ✅ | Zip upper halves |
| OddEven | ✅ | ✅ | Combine odd/even lanes |
| ConcatLowerLower | ✅ | ✅ | Concatenate halves |
| ConcatUpperUpper | ✅ | ✅ | Concatenate halves |
| ConcatLowerUpper | ✅ | ✅ | Concatenate halves |
| ConcatUpperLower | ✅ | ✅ | Concatenate halves |
| Broadcast<N> | ✅ | ✅ | Broadcast lane N |
| DupEven | ✅ | ✅ | Duplicate even lanes |
| DupOdd | ✅ | ✅ | Duplicate odd lanes |
| SwapAdjacentBlocks | ✅ | ✅ | Swap 128-bit blocks |
| Per4LaneBlockShuffle | ✅ | ✅ | Complex shuffle |
| SlideUpLanes | ✅ | ✅ | Shift lanes up |
| SlideDownLanes | ✅ | ✅ | Shift lanes down |
| Slide1Up | ✅ | ✅ | Shift by 1 lane |
| Slide1Down | ✅ | ✅ | Shift by 1 lane |

### Reduction Operations

| Operation | C++ Highway | go-highway | Notes |
|-----------|-------------|------------|-------|
| ReduceSum | ✅ | ✅ | |
| ReduceMin | ✅ | ✅ | |
| ReduceMax | ✅ | ✅ | |
| SumOfLanes | ✅ | ✅ | Same as ReduceSum |
| MinOfLanes | ✅ | ✅ | Same as ReduceMin |
| MaxOfLanes | ✅ | ✅ | Same as ReduceMax |
| GetLane | ✅ | ✅ | Extract single lane |
| ExtractLane | ✅ | ✅ | Extract lane by index (same as GetLane) |
| InsertLane | ✅ | ✅ | Insert into lane |

### Bitwise Operations ✅ **Bit Manipulation Implemented**

| Operation | C++ Highway | go-highway | Notes |
|-----------|-------------|------------|-------|
| And | ✅ | ✅ | |
| Or | ✅ | ✅ | |
| Xor | ✅ | ✅ | |
| AndNot | ✅ | ✅ | |
| Not | ✅ | ✅ | Bitwise NOT |
| ShiftLeft | ✅ | ✅ | Per-lane shift |
| ShiftRight | ✅ | ✅ | Per-lane shift |
| ShiftLeftSame | ✅ | ✅ | All lanes same shift (same as ShiftLeft) |
| ShiftRightSame | ✅ | ✅ | All lanes same shift (same as ShiftRight) |
| RotateRight | ✅ | ✅ | Bit rotation |
| ReverseBits | ✅ | ✅ | Reverse bit order |
| PopCount | ✅ | ✅ | Count set bits |
| LeadingZeroCount | ✅ | ✅ | CLZ |
| TrailingZeroCount | ✅ | ✅ | CTZ |
| HighestSetBitIndex | ✅ | ✅ | BSR equivalent |

### Type Conversion Operations ✅ **Promote/Demote Implemented**

| Operation | C++ Highway | go-highway | Notes |
|-----------|-------------|------------|-------|
| ConvertTo | ✅ | ✅ | Float <-> Int conversion |
| PromoteTo | ✅ | ✅ | Widen (e.g., f32 -> f64) |
| DemoteTo | ✅ | ✅ | Narrow (e.g., f64 -> f32, saturating) |
| PromoteUpperTo | ✅ | ✅ | Promote upper half |
| PromoteLowerTo | ✅ | ✅ | Promote lower half |
| ReorderDemote2To | ✅ | ✅ | Demote two vectors (DemoteTwo*) |
| BitCast | ✅ | ✅ | Reinterpret bits |
| TruncateTo | ✅ | ✅ | Truncate to narrower (non-saturating) |
| Round | ✅ | ✅ | Round to nearest int |
| Trunc | ✅ | ✅ | Truncate toward zero |
| Ceil | ✅ | ✅ | Round up |
| Floor | ✅ | ✅ | Round down |
| NearestInt | ✅ | ✅ | Nearest integer (banker's rounding) |

### Mask Operations ✅ **Complete**

| Operation | C++ Highway | go-highway | Notes |
|-----------|-------------|------------|-------|
| FirstN | ✅ | ✅ | Mask for first N lanes |
| LastN | ✅ | ✅ | Mask for last N lanes |
| IfThenElse | ✅ | ✅ | |
| IfThenElseZero | ✅ | ✅ | If-then-else with zero |
| IfThenZeroElse | ✅ | ✅ | If-then-zero-else |
| ZeroIfNegative | ✅ | ✅ | |
| CountTrue | ✅ | ✅ | Count true lanes |
| AllTrue | ✅ | ✅ | All lanes true? |
| AllFalse | ✅ | ✅ | All lanes false? |
| FindFirstTrue | ✅ | ✅ | Index of first true |
| FindLastTrue | ✅ | ✅ | Index of last true |
| MaskFromBits | ✅ | ✅ | Create mask from bitmask |
| BitsFromMask | ✅ | ✅ | Convert mask to bitmask |
| MaskAnd | ✅ | ✅ | Mask AND |
| MaskOr | ✅ | ✅ | Mask OR |
| MaskXor | ✅ | ✅ | Mask XOR |
| MaskNot | ✅ | ✅ | Mask NOT |
| MaskAndNot | ✅ | ✅ | Mask AND NOT |

### Compress/Expand Operations ✅ **Implemented**

| Operation | C++ Highway | go-highway | Notes |
|-----------|-------------|------------|-------|
| Compress | ✅ | ✅ | Pack true lanes |
| Expand | ✅ | ✅ | Unpack to true lanes |
| CompressStore | ✅ | ✅ | Compress and store |
| CompressBlendedStore | ✅ | ✅ | Compress with blend |

---

## Math Functions (contrib/math) ✅ **Complete**

| Function | C++ Highway | go-highway | Notes |
|----------|-------------|------------|-------|
| Exp | ✅ | ✅ | |
| Exp2 | ✅ | ✅ | |
| Exp10 | ✅ | ✅ | |
| Expm1 | ✅ | ✅ | exp(x) - 1 |
| Log | ✅ | ✅ | |
| Log2 | ✅ | ✅ | |
| Log10 | ✅ | ✅ | |
| Log1p | ✅ | ✅ | log(1 + x) |
| Pow | ✅ | ✅ | x^y |
| Cbrt | ✅ | ✅ | Cube root |
| Sin | ✅ | ✅ | |
| Cos | ✅ | ✅ | |
| SinCos | ✅ | ✅ | |
| Tan | ✅ | ✅ | |
| Asin | ✅ | ✅ | |
| Acos | ✅ | ✅ | |
| Atan | ✅ | ✅ | |
| Atan2 | ✅ | ✅ | |
| Sinh | ✅ | ✅ | |
| Cosh | ✅ | ✅ | |
| Tanh | ✅ | ✅ | |
| Asinh | ✅ | ✅ | |
| Acosh | ✅ | ✅ | |
| Atanh | ✅ | ✅ | |
| Erf | ✅ | ✅ | |
| Sigmoid | ✅ | ✅ | |
| Hypot | ✅ | ✅ | sqrt(x^2 + y^2) |

---

## Data Type Support

| Type | C++ Highway | go-highway | Notes |
|------|-------------|------------|-------|
| int8 | ✅ | ✅ | |
| int16 | ✅ | ✅ | |
| int32 | ✅ | ✅ | |
| int64 | ✅ | ✅ | |
| uint8 | ✅ | ✅ | |
| uint16 | ✅ | ✅ | |
| uint32 | ✅ | ✅ | |
| uint64 | ✅ | ✅ | |
| float16 | ✅ | ❌ | Half precision |
| bfloat16 | ✅ | ❌ | Brain float |
| float32 | ✅ | ✅ | |
| float64 | ✅ | ✅ | |

---

## Recent Additions (January 2026)

The following features were added to close major gaps:

### Shuffle/Permutation Operations
- `Reverse`, `Reverse2`, `Reverse4`, `Reverse8` - Lane reversal
- `Broadcast` - Broadcast single lane to all lanes
- `GetLane`, `InsertLane` - Lane extraction/insertion
- `InterleaveLower`, `InterleaveUpper` - Interleave operations
- `ConcatLowerLower`, `ConcatUpperUpper`, `ConcatLowerUpper`, `ConcatUpperLower` - Concatenation
- `OddEven`, `DupEven`, `DupOdd` - Even/odd lane operations
- `SwapAdjacentBlocks` - 128-bit block swap
- `TableLookupBytes` - Byte-level table lookup

### Type Conversions
- `ConvertToInt32`, `ConvertToInt64` - Float to int conversion
- `ConvertToFloat32`, `ConvertToFloat64` - Int to float conversion
- `Round`, `Trunc`, `Ceil`, `Floor`, `NearestInt` - Rounding operations
- `BitCastF32ToI32`, `BitCastI32ToF32`, etc. - Bit-level reinterpretation

### Compress/Expand Operations
- `Compress` - Pack elements where mask is true
- `Expand` - Unpack elements to mask positions
- `CompressStore`, `CompressBlendedStore` - Compress with store

### Mask Operations
- `CountTrue`, `AllTrue`, `AllFalse` - Mask predicates
- `FindFirstTrue`, `FindLastTrue` - Find operations
- `FirstN`, `LastN` - Create masks for N lanes
- `MaskFromBits`, `BitsFromMask` - Bit conversion
- `MaskAnd`, `MaskOr`, `MaskXor`, `MaskNot`, `MaskAndNot` - Logical operations

### Gather/Scatter Operations (January 2026)
- `GatherIndex` - Load elements from non-contiguous memory using indices
- `GatherIndexMasked` - Masked gather with selective loading
- `ScatterIndex` - Store elements to non-contiguous memory using indices
- `ScatterIndexMasked` - Masked scatter with selective storing
- `GatherIndexOffset` - Gather with base + index*scale addressing
- `IndicesIota`, `IndicesStride`, `IndicesFromFunc` - Index vector creation utilities

### Type Width Conversion Operations (January 2026)
- **Float Promotion**: `PromoteF32ToF64`, `PromoteLowerF32ToF64`, `PromoteUpperF32ToF64`
- **Float Demotion**: `DemoteF64ToF32`, `DemoteTwoF64ToF32`
- **Signed Integer Promotion**: `PromoteI8ToI16`, `PromoteI16ToI32`, `PromoteI32ToI64` (with Lower/Upper variants)
- **Unsigned Integer Promotion**: `PromoteU8ToU16`, `PromoteU16ToU32`, `PromoteU32ToU64` (with Lower/Upper variants)
- **Signed Integer Demotion (Saturating)**: `DemoteI16ToI8`, `DemoteI32ToI16`, `DemoteI64ToI32`, `DemoteTwoI*` variants
- **Unsigned Integer Demotion (Saturating)**: `DemoteU16ToU8`, `DemoteU32ToU16`, `DemoteU64ToU32`, `DemoteTwoU*` variants
- **Truncating Demotion (Non-saturating)**: `TruncateI16ToI8`, `TruncateI32ToI16`, `TruncateI64ToI32`, and unsigned variants

### Bit Manipulation Operations (January 2026)
- `PopCount` - Count set bits per lane
- `LeadingZeroCount` - Count leading zeros (CLZ)
- `TrailingZeroCount` - Count trailing zeros (CTZ)
- `RotateRight` - Bit rotation right by constant
- `ReverseBits` - Reverse bit order per lane
- `HighestSetBitIndex` - Index of highest set bit (BSR equivalent)

### Slide Operations (January 2026)
- `SlideUpLanes` - Shift lanes up by N, fill low with zeros
- `SlideDownLanes` - Shift lanes down by N, fill high with zeros
- `Slide1Up` - Shift lanes up by 1
- `Slide1Down` - Shift lanes down by 1

### Saturated Arithmetic Operations (January 2026)
- `SaturatedAdd` - Addition with saturation (clamp to type range)
- `SaturatedSub` - Subtraction with saturation
- `Clamp` - Clamp each element to [lo, hi] range
- `AbsDiff` - Absolute difference |a - b|
- `Avg` - Rounded average (a + b + 1) / 2
- `MulHigh` - High bits of widening multiplication

### Comparison Operations (January 2026)
- `NotEqual` - Element-wise not-equal comparison
- `IsNaN` - Check for NaN values (float)
- `IsInf` - Check for infinity values (float)
- `IsFinite` - Check for finite values (float)
- `TestBit` - Test if specific bit is set (integer)

### Additional Mask Operations (January 2026)
- `IfThenElseZero` - Returns a where mask is true, zero otherwise
- `IfThenZeroElse` - Returns zero where mask is true, b otherwise
- `ZeroIfNegative` - Zero for negative lanes, original value otherwise

### Additional Shuffle Operations (January 2026)
- `Shuffle0123` - 4-element shuffle with explicit indices
- `TableLookupLanes` - Lane-level table lookup
- `TableLookupLanesOr` - Lane-level table lookup with fallback for out-of-bounds
- `ZipLower` - Interleave lower halves (alias for InterleaveLower)
- `ZipUpper` - Interleave upper halves (alias for InterleaveUpper)
- `Per4LaneBlockShuffle` - Complex shuffle within 4-lane blocks

### Math Functions (January 2026)
- `Cbrt` - Cube root
- `Hypot` - sqrt(x² + y²) with numerical stability

### Memory Operations (January 2026)
- `BlendedStore` - Conditional store preserving existing values where mask is false
- `Undefined` - Returns uninitialized vector (zero-initialized in Go for safety)
- `LoadDup128` - Load 128-bit block and duplicate to fill wider vectors
- `LoadInterleaved2/3/4` - Deinterleave AoS data into SoA vectors
- `StoreInterleaved2/3/4` - Interleave SoA vectors into AoS format

---

## Priority Implementation Recommendations

### High Priority (Most Impact)

1. ~~**ARM NEON** - Architecture support~~ ✅ **IMPLEMENTED**
   - Enables Apple Silicon, mobile, AWS Graviton

2. ~~**Gather/Scatter** - Indexed memory access~~ ✅ **IMPLEMENTED**
   - Important for sparse operations

3. ~~**Promote/Demote** - Width conversion~~ ✅ **IMPLEMENTED**
   - `PromoteTo`, `DemoteTo` for mixed-precision

### Medium Priority

4. ~~**Bit Operations** - `PopCount`, `LeadingZeroCount`, `RotateRight`~~ ✅ **IMPLEMENTED**
   - Important for bit manipulation algorithms

5. ~~**Missing Math** - `Tan`, `Atan`, `Atan2`, `Pow`~~ ✅ **Previously Implemented**
   - Complete transcendental coverage

6. ~~**Slide Operations** - `SlideUpLanes`, `SlideDownLanes`~~ ✅ **IMPLEMENTED**
   - Lane shifting operations

### Lower Priority

7. **Half-Precision Floats** - float16, bfloat16
   - ML inference optimization

8. **WebAssembly SIMD** - Browser deployment

9. **Additional x86 Variants** - AVX3_DL, ZEN4
    - Marginal improvements over AVX-512

---

## What go-highway Does Well

- Core arithmetic operations (Add, Sub, Mul, Div, FMA, Sqrt)
- **Saturated arithmetic** (SaturatedAdd, SaturatedSub, Clamp, AbsDiff, Avg, MulHigh)
- Transcendental math (Exp, Log, Sin, Cos, Tan, Tanh, Sigmoid, Erf, Atan, Atan2, Pow, Cbrt, Hypot)
- Basic reductions (Sum, Min, Max)
- Dot product and matrix-vector operations
- **Complete shuffle/permutation operations** (Reverse, Interleave, Concat, OddEven, Slide, TableLookupLanes, Zip, Per4LaneBlockShuffle)
- **Bit manipulation** (PopCount, LeadingZeroCount, TrailingZeroCount, RotateRight, ReverseBits)
- **Complete comparison operations** (Equal, NotEqual, LessThan, GreaterThan, IsNaN, IsInf, IsFinite, TestBit)
- **Type conversions** (ConvertTo, Round, Trunc, Ceil, Floor, BitCast)
- **Type width conversions** (PromoteTo, DemoteTo for all integer/float types)
- **Compress/Expand operations** (Compress, Expand, CompressStore)
- **Gather/Scatter operations** (GatherIndex, ScatterIndex, masked variants)
- **Complete memory operations** (BlendedStore, Undefined, LoadDup128, LoadInterleaved, StoreInterleaved)
- **Comprehensive mask operations** (CountTrue, AllTrue, FindFirstTrue, MaskFromBits, IfThenElseZero, ZeroIfNegative)
- Code generation tool (hwygen) for multi-target dispatch
- Clean API leveraging Go generics
- Automatic tail handling for non-aligned sizes
