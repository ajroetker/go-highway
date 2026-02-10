# Darwin SME Matrix Multiplication Support

This document specifies the requirements for adding ARM SME (Scalable Matrix Extension) support for matrix multiplication operations in go-highway, targeting Apple Silicon M4+ chips on macOS (Darwin).

## Overview

ARM SME provides dedicated matrix multiplication hardware through:
- **ZA tile registers**: 2D matrix accumulators (SVL × SVL bits, where SVL is 128-2048 bits)
- **Outer product instructions**: FMOPA/FMOPS that compute O(N²) results per instruction
- **Streaming mode**: Special processor mode required for SME/SVE instructions

### Why SME for MatMul?

Traditional SIMD approaches (NEON, AVX) compute matrix multiplication using dot products:
- Load row of A, column of B
- Multiply-accumulate into scalar result
- O(N) operations per instruction

SME's outer product approach:
- Load vector from A, vector from B
- `FMOPA` accumulates entire N×N tile in one instruction
- O(N²) operations per instruction
- Theoretical 10-20x speedup for large matrices

## Hardware Availability

| Chip | SME Support | SVL (Streaming Vector Length) |
|------|-------------|-------------------------------|
| Apple M4 | Yes | 512 bits (16 × float32) |
| Apple M4 Pro/Max | Yes | 512 bits |
| ARM Neoverse V2 | Yes | 128-512 bits (configurable) |
| Earlier Apple Silicon | No | N/A |

## Required Operations

### 1. Tile Types and Management

```go
// New tile type representing ZA matrix accumulator
// Size is runtime-determined based on SVL
type Tile[T Floats] struct {
    // Implementation opaque - backed by ZA register file
}

// Tile dimensions (runtime query)
func TileDim[T Floats]() int  // Returns SVL / sizeof(T)

// Tile management
func ZeroTile[T Floats]() Tile[T]                              // ZERO {za}
func LoadTileRow[T Floats](tile *Tile[T], row int, src []T)    // LD1W horizontal
func StoreTileRow[T Floats](tile Tile[T], row int, dst []T)    // ST1W horizontal
func LoadTileCol[T Floats](tile *Tile[T], col int, src []T)    // LD1W vertical
func StoreTileCol[T Floats](tile Tile[T], col int, dst []T)    // ST1W vertical
```

### 2. Outer Product Operations (Core MatMul)

```go
// FMOPA - Floating-point outer product and accumulate
// tile[i,j] += a[i] * b[j] for all i,j
func OuterProductAccumulate[T Floats](tile *Tile[T], a, b Vec[T])

// FMOPS - Outer product and subtract
func OuterProductSubtract[T Floats](tile *Tile[T], a, b Vec[T])

// Widening variants (BF16 → FP32)
func OuterProductAccumulateBF16(tile *Tile[float32], a, b Vec[bfloat16])

// Integer variants (INT8 → INT32)
func OuterProductAccumulateInt8(tile *Tile[int32], a, b Vec[int8])   // SMOPA/UMOPA
func OuterProductAccumulateInt16(tile *Tile[int32], a, b Vec[int16])
```

### 3. Predicated Operations

SME uses two predicates (row and column) for handling edge cases where matrix dimensions aren't multiples of SVL:

```go
// Predicated outer product for non-multiple-of-SVL dimensions
func OuterProductAccumulateMasked[T Floats](
    tile *Tile[T],
    rowMask, colMask Mask[T],  // Pn/M and Pm/M predicates
    a, b Vec[T],
)
```

### 4. Tile-Vector Movement

```go
// MOVA - Move between tiles and vectors
func TileRowToVec[T Floats](tile Tile[T], row int) Vec[T]
func TileColToVec[T Floats](tile Tile[T], col int) Vec[T]
func VecToTileRow[T Floats](tile *Tile[T], row int, v Vec[T])
func VecToTileCol[T Floats](tile *Tile[T], col int, v Vec[T])
```

### 5. Horizontal/Vertical Accumulation

```go
// ADDHA/ADDVA - Add vector to all rows/columns of tile
func AddToTileRows[T Floats](tile *Tile[T], v Vec[T])   // ADDHA
func AddToTileCols[T Floats](tile *Tile[T], v Vec[T])   // ADDVA
```

### 6. Streaming Mode Control

```go
// Runtime mode switching (SMSTART/SMSTOP)
func EnterStreamingMode()
func ExitStreamingMode()
func InStreamingMode() bool
```

### 7. Feature Detection

Update `dispatch_arm64.go`:

```go
// Add to dispatch levels
const (
    DispatchSME DispatchLevel = iota + 100
)

func init() {
    // ... existing NEON detection ...

    if cpu.ARM64.HasSME {
        currentLevel = DispatchSME
        streamingVectorLength = queryStreamingSVL()
        currentName = "sme"
    }
}
```

## Assembly Implementation Patterns

### Reference: antfly RaBitQ SME Implementation

From `/Users/ajroetker/go/src/github.com/antflydb/antfly/lib/num32/asm/rabitq_sme_arm64.s`:

#### Build Constraints
```go
//go:build !noasm && darwin && arm64
```

Darwin-only because:
1. SME currently only available on Apple M4+
2. macOS has specific restrictions on streaming mode instructions

#### Function Declaration (Go)
```go
//go:noescape
func rabitq_bit_product_sme(code, q1, q2, q3, q4, res, len unsafe.Pointer)
```

#### Assembly Structure
```assembly
TEXT ·rabitq_bit_product_sme(SB), NOSPLIT, $0-56
    // Load parameters
    MOVD    code+0(FP), R0
    MOVD    q1+8(FP), R1
    // ...

    // Threshold check - fall back to NEON for small inputs
    CMP     $8, R8
    BLT     scalar_fallback

    // Enter streaming mode
    WORD    $0xd503477f        // smstart

    // Set up predicate (all lanes active)
    WORD    $0x2598e3e0        // ptrue p0.s

    // Zero accumulator registers
    WORD    $0x25b8c010        // mov z16.s, #0

vector_loop:
    // SVE loads (scalable width)
    WORD    $0xa540a000        // ld1w {z0.s}, p0/z, [x0]
    ADD     $64, R0, R0        // Advance pointer (64 bytes = 512-bit SVL)

    // Compute operations
    WORD    $0x04213005        // and z5.d, z0.d, z1.d
    WORD    $0x04a50210        // add z16.s, z16.s, z5.s

    SUBS    $1, R9, R9
    BNE     vector_loop

    // Horizontal reduction
    WORD    $0x04812210        // uaddv d16, p0, z16.s
    FMOVD   F16, R12

sme_done:
    // Exit streaming mode
    WORD    $0xd503467f        // smstop

    // Handle remainder with scalar loop
    // ...

    RET
```

### SME MatMul Kernel Pattern

```assembly
TEXT ·matmul_sme_f32(SB), NOSPLIT, $0-64
    // Parameters: A, B, C matrices, M, N, K dimensions
    MOVD    a+0(FP), R0
    MOVD    b+8(FP), R1
    MOVD    c+16(FP), R2
    MOVD    m+24(FP), R3
    MOVD    n+32(FP), R4
    MOVD    k+40(FP), R5

    // Enter streaming mode
    WORD    $0xd503477f          // smstart

    // Set up predicates
    WORD    $0x2598e3e0          // ptrue p0.s

    // Zero ZA tile (matrix accumulator)
    WORD    $0xc00800ff          // zero {za}

    // Outer loop over tile blocks
tile_loop:
    // Load column vector from A into z0
    WORD    $0xa540a000          // ld1w {z0.s}, p0/z, [x0]

    // Load row vector from B into z1
    WORD    $0xa540a021          // ld1w {z1.s}, p0/z, [x1]

    // FMOPA - outer product accumulate into ZA tile
    // za0.s += z0.s ⊗ z1.s (outer product)
    WORD    $0x80800000          // fmopa za0.s, p0/m, p0/m, z0.s, z1.s

    // Advance K dimension
    ADD     $64, R0, R0
    ADD     $64, R1, R1
    SUBS    $1, R6, R6
    BNE     tile_loop

    // Store ZA tile rows to C matrix
    MOVD    $0, R7               // row counter
store_loop:
    // st1w {za0h.s[w12]}, p0, [x2]
    WORD    $0xe0004040          // Store horizontal slice
    ADD     $64, R2, R2          // Next row of C
    ADD     $1, R7, R7
    CMP     R7, R8               // R8 = tile dimension
    BLT     store_loop

    // Exit streaming mode
    WORD    $0xd503467f          // smstop
    RET
```

## Key SME Instruction Encodings

| Instruction | Encoding | Description |
|-------------|----------|-------------|
| `smstart` | `0xd503477f` | Enter streaming mode |
| `smstop` | `0xd503467f` | Exit streaming mode |
| `ptrue p0.s` | `0x2598e3e0` | Set predicate all-true (32-bit elements) |
| `zero {za}` | `0xc00800ff` | Zero entire ZA tile array |
| `mov z0.s, #0` | `0x25b8c000` | Zero SVE register |
| `ld1w {z0.s}, p0/z, [x0]` | `0xa540a000` | Load 32-bit elements |
| `st1w {z0.s}, p0, [x0]` | `0xe540e000` | Store 32-bit elements |
| `fmopa za0.s, p0/m, p0/m, z0.s, z1.s` | `0x80800000` | FP32 outer product accumulate |
| `uaddv d0, p0, z0.s` | `0x04812000` | Unsigned add reduction |

## macOS-Specific Restrictions

### Instructions That Fail in Streaming Mode on macOS

| Instruction | Issue | Workaround |
|-------------|-------|------------|
| `movi d0, #0` | SIGILL | Use `fmov s0, wzr` (`0x1e2703e0`) |
| `fadda` (ordered reduction) | SIGILL | Use `fadd` + `faddv` |
| SVE setup before streaming | SIGILL | Must enter streaming mode first |

### Streaming Mode Transition Cost

Each `smstart`/`smstop` pair has significant overhead (~100-1000 cycles). Design APIs to:
1. Process entire matrix blocks per call
2. Minimize mode transitions
3. Use threshold to fall back to NEON for small inputs

From benchmarks (4096×4096 matrix-vector):
- 1 transition (hand-written): 2.5ms
- 1024 transitions (per-row calls): 23ms (9x slower)

## Implementation Strategy

### Phase 1: Infrastructure
1. Add `DispatchSME` level to dispatch system
2. Add SME feature detection (requires darwin/arm64 runtime check)
3. Add streaming mode entry/exit helpers

### Phase 2: Core Operations
1. Implement tile zero (`ZeroTile`)
2. Implement outer product (`OuterProductAccumulate`) using FMOPA
3. Implement tile load/store for rows and columns

### Phase 3: MatMul Kernel
1. Implement blocked MatMul using tile operations
2. Add predicated variants for edge handling
3. Optimize tile blocking strategy

### Phase 4: Integration
1. Add `hwy/contrib/matmul/` package
2. Dispatch to SME/NEON/scalar based on hardware and size
3. Benchmark and tune thresholds

## Testing Strategy

### Hardware Requirements
- Apple M4 Mac (MacBook Pro 14"/16" 2024, Mac Mini 2024, iMac 2024)
- macOS 15.0+ (Sequoia)

### Test Cases
1. **Unit tests**: Individual operations (zero, load, store, FMOPA)
2. **Correctness tests**: Compare SME results against scalar reference
3. **Edge cases**: Non-multiple-of-SVL dimensions, single row/column
4. **Performance tests**: Benchmark against NEON and scalar baselines

### Build Tags
```go
//go:build !noasm && darwin && arm64
```

Run with:
```bash
# SME path
GOEXPERIMENT=simd go test ./hwy/contrib/matmul/...

# Force NEON fallback
HWY_NO_SME=1 GOEXPERIMENT=simd go test ./hwy/contrib/matmul/...
```

## Performance Expectations

Based on SME hardware characteristics:

| Matrix Size | NEON | SME (Expected) | Speedup |
|-------------|------|----------------|---------|
| 64×64 | Baseline | ~1.5x | Mode transition overhead |
| 256×256 | Baseline | ~4x | Tile blocking benefits |
| 1024×1024 | Baseline | ~8-12x | Full tile utilization |
| 4096×4096 | Baseline | ~10-15x | Memory bandwidth limited |

Actual performance depends on:
- Memory access patterns
- Tile blocking efficiency
- Mode transition amortization

## References

- [ARM SME Introduction Part 1](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction)
- [ARM SME Introduction Part 2](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction-p2)
- [SME2 Outer Product Tutorial](https://learn.arm.com/learning-paths/cross-platform/multiplying-matrices-with-sme2/5-outer-product/)
- [ARM SME Architecture Reference](https://developer.arm.com/documentation/ddi0616/latest/)
- [Go SIMD Proposal #73787](https://github.com/golang/go/issues/73787)
- [antfly RaBitQ SME Implementation](https://github.com/antflydb/antfly/tree/main/lib/num32/asm)
