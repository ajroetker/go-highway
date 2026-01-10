# ARM NEON Support

**Status: 🚧 In Progress** (2026-01-09)

Add ARM NEON (Advanced SIMD) implementations for go-highway using GOAT code generation.

## Overview

Unlike AVX2/AVX-512 which use Go's experimental `simd/archsimd` package, NEON support uses:
- **GOAT** (Go Assembly Transpiler) to generate assembly from C with NEON intrinsics
- Direct assembly calls via `//go:noescape` FFI
- Slice-based API (no archsimd vector types)

NEON is 128-bit SIMD:
- Float32: 4 lanes per vector
- Float64: 2 lanes per vector
- Int32: 4 lanes per vector

## Prerequisites

- [x] Install GOAT: `go1.26rc1 get -tool github.com/gorse-io/goat@latest`
- [x] ARM64 hardware (Apple Silicon, AWS Graviton, etc.)
- [x] Clang with ARM64 support

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Public API (hwy package)                  │
│         Load, Store, Add, Mul, Transform, etc.              │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
      ┌───────▼─────┐  ┌──────▼──────┐  ┌────▼──────────┐
      │   Scalar    │  │    NEON     │  │  AVX2/AVX-512 │
      │  (Pure Go)  │  │   (GOAT)    │  │  (archsimd)   │
      └─────────────┘  └──────┬──────┘  └───────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
      ┌───────▼─────┐  ┌──────▼──────┐  ┌────▼──────────┐
      │  hwy/c/*.c  │  │ hwy/asm/*.s │  │ hwy/asm/*.go  │
      │ C + NEON    │→ │  Generated  │ + │  Wrappers    │
      │ intrinsics  │  │  Assembly   │   │              │
      └─────────────┘  └─────────────┘  └───────────────┘
```

## GOAT Workflow

1. Write C code with NEON intrinsics in `hwy/c/`
2. Run `go generate` to produce assembly via GOAT
3. GOAT generates both `.go` declarations and `.s` assembly
4. Go wrappers in `hwy/asm/` provide safe slice-based APIs

See [GOAT.md](../GOAT.md) for complete GOAT documentation.

### go:generate Directive

```go
//go:generate go tool goat ../c/ops_neon_arm64.c -O3 -e="--target=arm64" -e="-march=armv8-a+simd+fp"
```

### Build Tags

```go
//go:build arm64 && !noasm    // NEON implementations
//go:build !arm64 || noasm    // Stubs/fallbacks
```

---

## GOAT Implementation Guide

### C Function Requirements

GOAT has strict requirements for C functions:

```c
// ✅ CORRECT: void return, pointer args, long for length
void add_f32_neon(float *a, float *b, float *result, long *len) {
    long n = *len;
    // ... implementation
}

// ❌ WRONG: non-void return type
float sum_f32_neon(float *a, long *len) { ... }

// ❌ WRONG: uint64_t not supported
void foo(uint64_t *data, long *len) { ... }

// ❌ WRONG: function calls (except inline)
void bar(float *a, long *len) {
    some_other_function(a);  // Not allowed!
}
```

### Supported Argument Types

| Type | Supported | Notes |
|------|-----------|-------|
| `float *` | ✅ | Float32 pointer |
| `double *` | ✅ | Float64 pointer |
| `int32_t *` | ✅ | Signed 32-bit int |
| `int64_t *` | ✅ | Signed 64-bit int |
| `long *` | ✅ | Use for lengths |
| `_Bool` | ✅ | Boolean |
| `uint64_t *` | ❌ | Not supported |
| Return values | ❌ | Must be void |

### C Code Pattern: Vectorized Loop

```c
#include <arm_neon.h>

void operation_f32_neon(float *a, float *b, float *result, long *len) {
    long n = *len;
    long i = 0;

    // Process 16 floats at a time (4 vectors × 4 lanes)
    // Using multiple accumulators reduces loop-carried dependencies
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(result + i,      vaddq_f32(a0, b0));
        vst1q_f32(result + i + 4,  vaddq_f32(a1, b1));
        vst1q_f32(result + i + 8,  vaddq_f32(a2, b2));
        vst1q_f32(result + i + 12, vaddq_f32(a3, b3));
    }

    // Process 4 floats at a time (single vector)
    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        vst1q_f32(result + i, vaddq_f32(av, bv));
    }

    // Scalar remainder (0-3 elements)
    for (; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}
```

### C Code Pattern: Horizontal Reduction

```c
void reduce_sum_f32_neon(float *input, float *result, long *len) {
    long n = *len;
    long i = 0;
    float sum = 0.0f;

    // Process with 4 accumulators to hide latency
    if (n >= 16) {
        float32x4_t sum0 = vdupq_n_f32(0);
        float32x4_t sum1 = vdupq_n_f32(0);
        float32x4_t sum2 = vdupq_n_f32(0);
        float32x4_t sum3 = vdupq_n_f32(0);

        for (; i + 15 < n; i += 16) {
            sum0 = vaddq_f32(sum0, vld1q_f32(input + i));
            sum1 = vaddq_f32(sum1, vld1q_f32(input + i + 4));
            sum2 = vaddq_f32(sum2, vld1q_f32(input + i + 8));
            sum3 = vaddq_f32(sum3, vld1q_f32(input + i + 12));
        }

        // Combine accumulators
        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);

        // Horizontal sum (ARMv8.2+)
        sum = vaddvq_f32(sum0);
    }

    // Scalar remainder
    for (; i < n; i++) {
        sum += input[i];
    }

    *result = sum;
}
```

### Go Wrapper Pattern

```go
//go:build !noasm && arm64

package asm

import "unsafe"

// Generated by GOAT - DO NOT EDIT
//go:noescape
func add_f32_neon(a, b, result, len unsafe.Pointer)

// Safe Go wrapper with slice API
func AddF32(a, b, result []float32) {
    if len(a) == 0 {
        return
    }
    n := int64(len(a))
    add_f32_neon(
        unsafe.Pointer(&a[0]),
        unsafe.Pointer(&b[0]),
        unsafe.Pointer(&result[0]),
        unsafe.Pointer(&n),
    )
}
```

### File Naming Convention

| File | Purpose |
|------|---------|
| `hwy/c/<category>_neon_arm64.c` | C source with NEON intrinsics |
| `hwy/asm/<category>_neon_arm64.go` | Generated Go declarations |
| `hwy/asm/<category>_neon_arm64.s` | Generated ARM64 assembly |
| `hwy/asm/neon_wrappers.go` | Safe slice-based Go wrappers |
| `hwy/asm/neon_stubs.go` | Stubs for non-ARM64 builds |

### Generated Assembly Structure

GOAT produces assembly with `WORD` directives containing ARM64 machine code:

```asm
//go:build !noasm && arm64
// Code generated by GoAT. DO NOT EDIT.

TEXT ·add_f32_neon(SB), $0-32
    MOVD a+0(FP), R0
    MOVD b+8(FP), R1
    MOVD result+16(FP), R2
    MOVD len+24(FP), R3
    WORD $0xf9400068       // ldr x8, [x3]
    WORD $0x4e20d420       // fadd v0.4s, v1.4s, v0.4s
    // ... more instructions
    RET
```

### Debugging GOAT Issues

1. **Compilation errors**: Check C syntax, ensure `#include <arm_neon.h>`
2. **Link errors**: Verify function signatures match between C and Go
3. **Runtime crashes**: Check pointer alignment, array bounds
4. **Wrong results**: Verify scalar fallback matches, check loop bounds

### Performance Tips

1. **Use multiple accumulators** - Reduces loop-carried dependencies
2. **Process 16+ elements per iteration** - Better instruction pipelining
3. **Align to 16 bytes when possible** - Faster loads/stores
4. **Minimize branches in hot loops** - Use predication where possible
5. **Let compiler auto-vectorize simple loops** - Sometimes `-O3` is enough

---

## Implementation Status

### Phase 1: Infrastructure ✅ Complete

| Item | Status | File |
|------|--------|------|
| ARM64 dispatch detection | ✅ | `hwy/dispatch_arm64.go` |
| GOAT integration | ✅ | `hwy/asm/ops_neon_arm64.go` |
| Build system | ✅ | `go.mod` (tool dependency) |
| Test infrastructure | ✅ | `hwy/asm/neon_test.go` |

### Phase 2: Core Arithmetic ✅ Complete

| Operation | F32 | F64 | C Function | Notes |
|-----------|-----|-----|------------|-------|
| Add | ✅ | ✅ | `add_f32_neon`, `add_f64_neon` | Element-wise |
| Sub | ✅ | ❌ | `sub_f32_neon` | |
| Mul | ✅ | ✅ | `mul_f32_neon`, `mul_f64_neon` | |
| Div | ✅ | ❌ | `div_f32_neon` | |
| FMA | ✅ | ✅ | `fma_f32_neon`, `fma_f64_neon` | a*b + c |
| Min | ✅ | ❌ | `min_f32_neon` | Element-wise |
| Max | ✅ | ❌ | `max_f32_neon` | Element-wise |
| Sqrt | ✅ | ❌ | `sqrt_f32_neon` | |
| Abs | ✅ | ❌ | `abs_f32_neon` | |
| Neg | ✅ | ❌ | `neg_f32_neon` | |

### Phase 3: Reductions ✅ Complete

| Operation | F32 | F64 | C Function | Notes |
|-----------|-----|-----|------------|-------|
| ReduceSum | ✅ | ✅ | `reduce_sum_f32_neon`, `reduce_sum_f64_neon` | Horizontal sum |
| ReduceMin | ✅ | ❌ | `reduce_min_f32_neon` | Horizontal min |
| ReduceMax | ✅ | ❌ | `reduce_max_f32_neon` | Horizontal max |

### Phase 4: Memory Operations 🔲 Not Started

| Operation | F32 | F64 | I32 | I64 | C Function | Priority |
|-----------|-----|-----|-----|-----|------------|----------|
| GatherIndex | ❌ | ❌ | ❌ | ❌ | `gather_*_neon` | High |
| ScatterIndex | ❌ | ❌ | ❌ | ❌ | `scatter_*_neon` | High |
| MaskedLoad | ❌ | ❌ | ❌ | ❌ | `masked_load_*_neon` | Medium |
| MaskedStore | ❌ | ❌ | ❌ | ❌ | `masked_store_*_neon` | Medium |

**Notes:**
- NEON doesn't have native gather/scatter instructions
- Must implement as scalar loop with NEON load/store
- Consider using `vld1q_lane_*` for partial vectorization

### Phase 5: Type Conversions 🔲 Not Started

| Operation | Status | C Function | Notes |
|-----------|--------|------------|-------|
| PromoteF32ToF64 | ❌ | `promote_f32_f64_neon` | `vcvt_f64_f32` |
| DemoteF64ToF32 | ❌ | `demote_f64_f32_neon` | `vcvt_f32_f64` |
| ConvertF32ToI32 | ❌ | `convert_f32_i32_neon` | `vcvtq_s32_f32` |
| ConvertI32ToF32 | ❌ | `convert_i32_f32_neon` | `vcvtq_f32_s32` |
| Round | ❌ | `round_f32_neon` | `vrndnq_f32` |
| Trunc | ❌ | `trunc_f32_neon` | `vrndq_f32` |
| Ceil | ❌ | `ceil_f32_neon` | `vrndpq_f32` |
| Floor | ❌ | `floor_f32_neon` | `vrndmq_f32` |

### Phase 6: Shuffle/Permutation 🔲 Not Started

| Operation | Status | C Function | NEON Intrinsic |
|-----------|--------|------------|----------------|
| Reverse | ❌ | `reverse_*_neon` | `vrev64q_*` |
| Reverse2 | ❌ | `reverse2_*_neon` | `vrev64q_*` |
| Reverse4 | ❌ | `reverse4_*_neon` | `vextq_*` |
| Broadcast | ❌ | `broadcast_*_neon` | `vdupq_laneq_*` |
| GetLane | ❌ | `getlane_*_neon` | `vgetq_lane_*` |
| InsertLane | ❌ | `insertlane_*_neon` | `vsetq_lane_*` |
| InterleaveLower | ❌ | `interleave_lo_*_neon` | `vzip1q_*` |
| InterleaveUpper | ❌ | `interleave_hi_*_neon` | `vzip2q_*` |
| TableLookupBytes | ❌ | `tbl_*_neon` | `vqtbl1q_u8` |

### Phase 7: Comparisons 🔲 Not Started

| Operation | F32 | I32 | C Function | NEON Intrinsic |
|-----------|-----|-----|------------|----------------|
| Equal | ❌ | ❌ | `eq_*_neon` | `vceqq_*` |
| NotEqual | ❌ | ❌ | `ne_*_neon` | `vmvnq_u32(vceqq_*)` |
| LessThan | ❌ | ❌ | `lt_*_neon` | `vcltq_*` |
| LessEqual | ❌ | ❌ | `le_*_neon` | `vcleq_*` |
| GreaterThan | ❌ | ❌ | `gt_*_neon` | `vcgtq_*` |
| GreaterEqual | ❌ | ❌ | `ge_*_neon` | `vcgeq_*` |

### Phase 8: Bitwise Operations 🔲 Not Started

| Operation | Status | C Function | NEON Intrinsic |
|-----------|--------|------------|----------------|
| And | ❌ | `and_*_neon` | `vandq_*` |
| Or | ❌ | `or_*_neon` | `vorrq_*` |
| Xor | ❌ | `xor_*_neon` | `veorq_*` |
| AndNot | ❌ | `andnot_*_neon` | `vbicq_*` |
| Not | ❌ | `not_*_neon` | `vmvnq_*` |
| ShiftLeft | ❌ | `shl_*_neon` | `vshlq_*` |
| ShiftRight | ❌ | `shr_*_neon` | `vshrq_n_*` |

### Phase 9: Mask Operations 🔲 Not Started

| Operation | Status | C Function | Notes |
|-----------|--------|------------|-------|
| IfThenElse | ❌ | `ifthenelse_*_neon` | `vbslq_*` |
| CountTrue | ❌ | `count_true_neon` | Horizontal popcount |
| AllTrue | ❌ | `all_true_neon` | `vmaxvq_u32` == 0xFFFFFFFF |
| AllFalse | ❌ | `all_false_neon` | `vmaxvq_u32` == 0 |
| FirstN | ❌ | `firstn_*_neon` | Generate mask |
| Compress | ❌ | `compress_*_neon` | No native support |
| Expand | ❌ | `expand_*_neon` | No native support |

**Notes:**
- NEON doesn't have native compress/expand
- Must implement via scalar or lookup tables
- Consider `vqtbl1q_u8` for byte-level compress

### Phase 10: Transcendental Math 🔲 Not Started

| Function | F32 | F64 | Priority | Algorithm |
|----------|-----|-----|----------|-----------|
| Exp | ❌ | ❌ | High | Range reduction + polynomial |
| Log | ❌ | ❌ | High | Range reduction + polynomial |
| Exp2 | ❌ | ❌ | Medium | Similar to Exp |
| Log2 | ❌ | ❌ | Medium | Similar to Log |
| Log10 | ❌ | ❌ | Low | Log(x) / Log(10) |
| Exp10 | ❌ | ❌ | Low | Exp(x * Log(10)) |
| Sin | ❌ | ❌ | High | Range reduction + Chebyshev |
| Cos | ❌ | ❌ | High | Range reduction + Chebyshev |
| SinCos | ❌ | ❌ | Medium | Combined sin/cos |
| Tan | ❌ | ❌ | Low | Sin/Cos |
| Tanh | ❌ | ❌ | High | Exp-based or rational approx |
| Sigmoid | ❌ | ❌ | High | 1/(1+exp(-x)) |
| Erf | ❌ | ❌ | Medium | Polynomial approximation |
| Atan | ❌ | ❌ | Low | Polynomial approximation |
| Atan2 | ❌ | ❌ | Low | Atan with quadrant handling |
| Pow | ❌ | ❌ | Low | Exp(y * Log(x)) |

**Implementation Strategy:**
- Port polynomial coefficients from AVX2 implementations
- Use NEON FMA (`vfmaq_f32`) for Horner's method
- Consider accuracy vs performance tradeoffs

---

## File Structure

### Existing Files

| File | Purpose |
|------|---------|
| `hwy/dispatch_arm64.go` | ARM64 CPU detection |
| `hwy/c/ops_neon_arm64.c` | Core arithmetic C source |
| `hwy/asm/ops_neon_arm64.go` | Generated Go declarations |
| `hwy/asm/ops_neon_arm64.s` | Generated assembly (56KB) |
| `hwy/asm/neon_wrappers.go` | Safe slice-based Go wrappers |
| `hwy/asm/neon_stubs.go` | Stubs for non-ARM64 |
| `hwy/asm/neon_test.go` | Tests and benchmarks |
| `hwy/contrib/algo/transform_neon.go` | Transform integration |

### New Files Needed

| File | Purpose | Phase |
|------|---------|-------|
| `hwy/c/gather_neon_arm64.c` | Gather/scatter operations | 4 |
| `hwy/c/convert_neon_arm64.c` | Type conversions | 5 |
| `hwy/c/shuffle_neon_arm64.c` | Shuffle/permutation | 6 |
| `hwy/c/compare_neon_arm64.c` | Comparison operations | 7 |
| `hwy/c/bitwise_neon_arm64.c` | Bitwise operations | 8 |
| `hwy/c/mask_neon_arm64.c` | Mask operations | 9 |
| `hwy/c/math_neon_arm64.c` | Transcendental math | 10 |

---

## Performance Results

### Phase 2 Benchmarks (Apple M4 Max)

| Operation | NEON | Scalar | Speedup |
|-----------|------|--------|---------|
| AddF32 (1024) | 51.74 ns | 247.2 ns | **4.8x** |
| MulF32 (1024) | 51.54 ns | - | - |
| ReduceSumF32 (1024) | 31.98 ns | 246.8 ns | **7.7x** |
| SqrtF32 (1024) | 120.3 ns | - | - |

### Expected Performance Gains

| Category | Expected Speedup | Notes |
|----------|------------------|-------|
| Element-wise ops | 4-8x | Memory bandwidth limited |
| Reductions | 6-10x | Fewer loop iterations |
| Transcendentals | 3-6x | Polynomial evaluation |
| Shuffle/permute | 2-4x | Limited by latency |

---

## NEON Intrinsic Reference

### Common Float32 Intrinsics

```c
// Load/Store
float32x4_t vld1q_f32(const float *ptr);
void vst1q_f32(float *ptr, float32x4_t v);

// Arithmetic
float32x4_t vaddq_f32(float32x4_t a, float32x4_t b);
float32x4_t vsubq_f32(float32x4_t a, float32x4_t b);
float32x4_t vmulq_f32(float32x4_t a, float32x4_t b);
float32x4_t vdivq_f32(float32x4_t a, float32x4_t b);
float32x4_t vfmaq_f32(float32x4_t acc, float32x4_t a, float32x4_t b); // acc + a*b

// Min/Max
float32x4_t vminq_f32(float32x4_t a, float32x4_t b);
float32x4_t vmaxq_f32(float32x4_t a, float32x4_t b);

// Unary
float32x4_t vsqrtq_f32(float32x4_t a);
float32x4_t vabsq_f32(float32x4_t a);
float32x4_t vnegq_f32(float32x4_t a);

// Horizontal reductions (ARMv8.2+)
float vaddvq_f32(float32x4_t a);  // Sum all lanes
float vmaxvq_f32(float32x4_t a);  // Max of all lanes
float vminvq_f32(float32x4_t a);  // Min of all lanes

// Broadcast
float32x4_t vdupq_n_f32(float value);

// Lane operations
float vgetq_lane_f32(float32x4_t v, int lane);
float32x4_t vsetq_lane_f32(float value, float32x4_t v, int lane);

// Rounding (ARMv8+)
float32x4_t vrndnq_f32(float32x4_t a);  // Round to nearest
float32x4_t vrndmq_f32(float32x4_t a);  // Floor
float32x4_t vrndpq_f32(float32x4_t a);  // Ceil
float32x4_t vrndq_f32(float32x4_t a);   // Truncate

// Comparison (returns uint32x4_t mask)
uint32x4_t vceqq_f32(float32x4_t a, float32x4_t b);
uint32x4_t vcltq_f32(float32x4_t a, float32x4_t b);
uint32x4_t vcleq_f32(float32x4_t a, float32x4_t b);
uint32x4_t vcgtq_f32(float32x4_t a, float32x4_t b);
uint32x4_t vcgeq_f32(float32x4_t a, float32x4_t b);

// Select (if-then-else)
float32x4_t vbslq_f32(uint32x4_t mask, float32x4_t a, float32x4_t b);

// Type conversion
int32x4_t vcvtq_s32_f32(float32x4_t a);      // Float to int
float32x4_t vcvtq_f32_s32(int32x4_t a);      // Int to float
float64x2_t vcvt_f64_f32(float32x2_t a);     // F32 to F64 (2 lanes)
float32x2_t vcvt_f32_f64(float64x2_t a);     // F64 to F32 (2 lanes)

// Reinterpret (bitcast)
int32x4_t vreinterpretq_s32_f32(float32x4_t a);
float32x4_t vreinterpretq_f32_s32(int32x4_t a);
```

### Common Float64 Intrinsics

```c
// Load/Store
float64x2_t vld1q_f64(const double *ptr);
void vst1q_f64(double *ptr, float64x2_t v);

// Arithmetic
float64x2_t vaddq_f64(float64x2_t a, float64x2_t b);
float64x2_t vsubq_f64(float64x2_t a, float64x2_t b);
float64x2_t vmulq_f64(float64x2_t a, float64x2_t b);
float64x2_t vdivq_f64(float64x2_t a, float64x2_t b);
float64x2_t vfmaq_f64(float64x2_t acc, float64x2_t a, float64x2_t b);

// Horizontal reduction
double vaddvq_f64(float64x2_t a);

// Broadcast
float64x2_t vdupq_n_f64(double value);
```

---

## Testing Strategy

### Unit Tests

Each C function should have corresponding Go tests:

```go
func TestOperationName(t *testing.T) {
    // Test aligned sizes (multiple of 16)
    // Test unaligned sizes (remainder handling)
    // Test edge cases (empty, single element)
    // Test special values (NaN, Inf, denormals)
}
```

### Benchmarks

Compare NEON vs scalar for various sizes:

```go
func BenchmarkOperation_NEON(b *testing.B) { ... }
func BenchmarkOperation_Scalar(b *testing.B) { ... }
```

### Cross-validation

Verify NEON results match scalar baseline within tolerance.

---

## Build & Test Commands

```bash
# Generate assembly from C
cd hwy/asm && go1.26rc1 generate ./...

# Build all
go1.26rc1 build ./...

# Run tests
go1.26rc1 test -v ./hwy/asm/...

# Run benchmarks
go1.26rc1 test -bench=. -benchmem ./hwy/asm/...

# Force scalar fallback (for comparison)
HWY_NO_SIMD=1 go1.26rc1 test -v ./hwy/...

# Build with noasm tag (use stubs)
go1.26rc1 build -tags=noasm ./...
```

---

## Known Limitations

### GOAT Limitations

1. **No function calls** - Only inline functions allowed in C
2. **Void return types** - Functions must return void
3. **Limited argument types** - Only pointers, int64_t, long, float, double, _Bool
4. **No uint64_t** - Use int64_t or long instead

### NEON Limitations

1. **No native gather/scatter** - Must emulate with scalar or lane ops
2. **No native compress/expand** - Must use lookup tables or scalar
3. **128-bit only** - No 256-bit or 512-bit NEON (SVE is separate)
4. **Limited horizontal ops** - Some require ARMv8.2+

### Platform Notes

1. **macOS/Apple Silicon** - Full NEON support, avoid SVE for now
2. **Linux/Graviton** - Full NEON support
3. **iOS** - Should work but untested

---

## Future: SVE Support

ARM SVE (Scalable Vector Extension) support can be added later:

- Variable vector length (128-2048 bits)
- Native gather/scatter
- Native compress/expand (compact)
- Predicate registers

SVE requires different C code and GOAT flags:
```
//go:generate go tool goat ../c/ops_sve_arm64.c -O3 -e="--target=arm64" -e="-march=armv9-a+sve"
```

**Note:** GOAT-generated SVE code on macOS is slower than hand-written due to streaming mode overhead. See `GOAT.md` for details.

---

## References

- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [GOAT Documentation](https://github.com/gorse-io/goat)
- [Highway C++ Library](https://github.com/google/highway)
- [go-highway feature gaps](./feature-gaps.md)
