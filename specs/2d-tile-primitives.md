# Plan: Tile-Aware Abstraction Layer for SME and Future Matrix Extensions

## Context

The go-highway library currently has `hwy.Vec[T]` for 1D SIMD vectors with operations
that hwygen transforms into target-specific code. ARM's SME (Scalable Matrix Extension)
on Apple M4+ provides dedicated 2D tile registers (ZA) with single-instruction outer
products (`FMOPA`), but these can only be used today by hand-writing C files for the
`:asm` path. There is no way to write a Go base function using tile operations and have
hwygen generate correct code for all targets.

This plan adds `hwy.Tile[T]` as a 2D matrix accumulator alongside `hwy.Vec[T]`, with
operations that lower naturally to:
- **SME**: single FMOPA instructions (via `:asm` C generation)
- **AVX2/AVX512/NEON (GoSimd)**: broadcast + FMA loops on arrays of vectors
- **Fallback**: nested scalar loops

The abstraction is designed to extend to ARM ACE and future matrix accelerators by
adding new targets without changing the base function code.

## Design

### Core Abstraction

A `Tile[T]` is a `TileDim × TileDim` square matrix accumulator, where `TileDim` equals
the vector lane count for the target (same as `MaxLanes[T]()`):

| Target | float32 TileDim | Tile shape | Hardware backing |
|--------|----------------|------------|------------------|
| AVX2 | 8 | 8×8 | 8 YMM registers |
| AVX512 | 16 | 16×16 | 16 ZMM registers |
| NEON | 4 | 4×4 | 4 Q registers |
| SVE_DARWIN (SME) | 16 | 16×16 | ZA tile register |
| Fallback | 4 | 4×4 | flat slice |

### Operations

| Operation | Scalar semantics | SME instruction |
|-----------|-----------------|-----------------|
| `TileZero(&tile)` | zero all elements | `svzero_za()` |
| `OuterProductAdd(&tile, row, col)` | tile[i][j] += row[i] * col[j] | `svfmopa_za32_f32` |
| `OuterProductSub(&tile, row, col)` | tile[i][j] -= row[i] * col[j] | `svfmops_za32_f32` |
| `TileStoreRow(&tile, idx, dst)` | copy row to dst slice | `svread_hor` + `svst1` |
| `TileReadRow(&tile, idx)` → Vec | read row as vector | `svread_hor` |
| `TileLoadCol(&tile, idx, src)` | load column from src | `svld1` + `svwrite_ver` |
| `TileDim[T]()` → int | compile-time constant | N/A |

### Lowering Strategy

**GoSimd targets (AVX2, AVX512, NEON)**: Tile types are structs with `Rows [N]VecType`.
Operations are methods on these structs. `OuterProductAdd` internally does N broadcast+FMA
operations. The transformer treats tile ops as normal method calls (same `IsMethod: true`
pattern as existing Vec ops). Note: NEON GoSimd calls `hwy/asm` package methods (not
`archsimd`) since Go's `simd/archsimd` does not yet support ARM64.

**ASM targets (neon:asm, sve_darwin:asm)**: The C generator translates tile ops to
target-specific intrinsics in `CIntrinsicProfile`. For NEON:asm, `OuterProductAdd`
emits `vfmaq_laneq_f32` loops. For SVE_DARWIN:asm, it emits `svfmopa_za32_f32`.
The entire function compiles to assembly via GoAT. **NEON:asm is the primary ARM64
performance target** since Go's `simd/archsimd` does not support NEON.

**Fallback**: Uses the scalar `hwy.Tile[T]` type with nested loop implementations.

### Usage Example (base function)

```go
//go:generate go run ../../../cmd/hwygen -input kernel_base.go -output . \
//    -targets avx2,avx512,neon,fallback -dispatch kernel

//hwy:gen T={float32, float64}
func BaseOuterProductKernel[T hwy.Floats](aPacked, b, c []T, m, n, k int) {
    dim := hwy.TileDim[T]()
    for i := 0; i < m; i += dim {
        for j := 0; j < n; j += dim {
            var tile hwy.Tile[T]
            hwy.TileZero(&tile)
            for p := 0; p < k; p++ {
                aCol := hwy.Load[T](aPacked[p*m+i:])
                bRow := hwy.Load[T](b[p*n+j:])
                hwy.OuterProductAdd(&tile, aCol, bRow)
            }
            for r := 0; r < dim; r++ {
                existing := hwy.Load[T](c[(i+r)*n+j:])
                row := hwy.TileReadRow(&tile, r)
                hwy.Store(hwy.Add(existing, row), c[(i+r)*n+j:])
            }
        }
    }
}
```

hwygen generates:
- `kernel_avx2.gen.go`: `tile` is `asm.TileFloat32x8`, `OuterProductAdd` → `tile.OuterProductAdd(aCol, bRow)`
- `kernel_arm64.gen.go`: `tile` is `asm.TileFloat32x4`, same method pattern
- `kernel_fallback.gen.go`: `tile` is `hwy.Tile[float32]`, scalar loops

SME specialization via existing `//hwy:specializes` + `//hwy:targets sve_darwin`:
```go
//hwy:specializes OuterProductKernel
//hwy:targets sve_darwin
func BaseOuterProductKernelSME[T hwy.Floats](...) {
    // Different algorithm using SME tiles directly, compiled via :asm
}
```

## Implementation Steps

### Step 1: Scalar Tile Type (`hwy/tile.go`) — new file

Define `Tile[T]`, `TileDim[T]()`, and all tile operations as pure Go functions.
Follow the patterns in `hwy/ops_base.go`. The `Tile[T]` struct stores a flat `[]T`
slice and a `dim int`. Operations use nested loops over `dim`.

Functions to implement:
- `TileDim[T]() int` — returns `MaxLanes[T]()` in scalar mode
- `NewTile[T]() Tile[T]` — zero-initialized tile
- `TileZero[T](tile *Tile[T])`
- `OuterProductAdd[T Floats](tile *Tile[T], row, col Vec[T])`
- `OuterProductSub[T Floats](tile *Tile[T], row, col Vec[T])`
- `TileStoreRow[T](tile *Tile[T], rowIdx int, dst []T)`
- `TileReadRow[T](tile *Tile[T], rowIdx int) Vec[T]`
- `TileLoadCol[T](tile *Tile[T], colIdx int, src []T)`

Add `hwy/tile_test.go` with tests verifying outer product correctness against
naive triple-nested loop.

### Step 2: NEON Tile Types (`hwy/asm/tile_neon.go`) — new file

Build tag: `!noasm && arm64`

Define concrete tile types for NEON:
- `TileFloat32x4` — struct with `Rows [4]Float32x4`
- `TileFloat64x2` — struct with `Rows [2]Float64x2`

Methods (pointer receivers):
- `Zero()` — zero all row vectors
- `OuterProductAdd(row, col Float32x4)` — 4× broadcast+FMA
- `OuterProductSub(row, col Float32x4)` — 4× broadcast+FMS
- `StoreRow(idx int, dst []float32)` — delegate to row's StoreSlice
- `ReadRow(idx int) Float32x4` — return row copy
- `LoadCol(idx int, src []float32)` — set element `idx` in each row from src

For `OuterProductAdd` on NEON, implement a `MulAddLane` assembly helper that maps
to `vfmaq_laneq_f32` (avoids explicit broadcast). This goes in
`hwy/asm/vec_neon.go` as a new method on `Float32x4`:

```go
// MulAddLaneAcc: acc += col * row[lane] using vfmaq_laneq_f32
func MulAddLaneAcc(col, row Float32x4, lane int, acc *Float32x4)
```

With corresponding assembly stubs in `hwy/asm/neon_stubs.go` and GoAT-generated
`.s` file. If the assembly is complex to land initially, fall back to
`BroadcastFloat32x4(row.Get(i))` + `MulAddAcc` pattern.

### Step 3: AVX2/AVX512 Tile Types (`hwy/asm/tile_avx2.go`, `tile_avx512.go`) — new files

Build tags: `amd64 && goexperiment.simd`

The `hwy/asm` package already imports `simd/archsimd` for amd64 builds
(see `hwy/asm/f16_avx2_promoted.go:24`).

**`tile_avx2.go`:**
- `TileFloat32x8` — struct with `Rows [8]archsimd.Float32x8`
- `TileFloat64x4` — struct with `Rows [4]archsimd.Float64x4`

`OuterProductAdd` implementation:
```go
func (t *TileFloat32x8) OuterProductAdd(row, col archsimd.Float32x8) {
    var rowArr [8]float32
    row.Store(&rowArr)
    for i := 0; i < 8; i++ {
        bcast := archsimd.BroadcastFloat32x8(rowArr[i])
        t.Rows[i] = bcast.MulAdd(col, t.Rows[i])
    }
}
```

**`tile_avx512.go`:**
- `TileFloat32x16` — struct with `Rows [16]archsimd.Float32x16`
- `TileFloat64x8` — struct with `Rows [8]archsimd.Float64x8`

Same method pattern, unrolled to 16 or 8 iterations.

### Step 4: hwygen Target Changes (`cmd/hwygen/targets.go`)

**4a. Add tile types to TypeMap for each target:**

```go
// AVX2Target TypeMap additions:
"tile_float32": "TileFloat32x8",
"tile_float64": "TileFloat64x4",

// AVX512Target TypeMap additions:
"tile_float32": "TileFloat32x16",
"tile_float64": "TileFloat64x8",

// NEONTarget TypeMap additions:
"tile_float32": "TileFloat32x4",
"tile_float64": "TileFloat64x2",

// FallbackTarget TypeMap additions:
// (empty — uses hwy.Tile[T] directly, which is already in hwy package)
```

**4b. Add tile operations to OpMap for SIMD targets (AVX2, AVX512, NEON):**

```go
"TileZero":         {Name: "Zero", IsMethod: true},
"OuterProductAdd":  {Name: "OuterProductAdd", IsMethod: true},
"OuterProductSub":  {Name: "OuterProductSub", IsMethod: true},
"TileStoreRow":     {Name: "StoreRow", IsMethod: true},
"TileReadRow":      {Name: "ReadRow", IsMethod: true},
"TileLoadCol":      {Name: "LoadCol", IsMethod: true},
"NewTile":          {Name: "New", IsMethod: false},  // constructor
```

**4c. Add tile operations to Fallback OpMap:**

```go
"TileZero":         {Package: "hwy", Name: "TileZero", IsMethod: false},
"OuterProductAdd":  {Package: "hwy", Name: "OuterProductAdd", IsMethod: false},
"OuterProductSub":  {Package: "hwy", Name: "OuterProductSub", IsMethod: false},
"TileStoreRow":     {Package: "hwy", Name: "TileStoreRow", IsMethod: false},
"TileReadRow":      {Package: "hwy", Name: "TileReadRow", IsMethod: false},
"TileLoadCol":      {Package: "hwy", Name: "TileLoadCol", IsMethod: false},
"NewTile":          {Package: "hwy", Name: "NewTile", IsMethod: false},
```

**4d. Add `TileDim` as a special-case constant (same as `MaxLanes`):**

```go
"TileDim": {Package: "special", Name: "TileDim", IsMethod: false},
```

**4e. Add `TileDim()` method to `Target` struct:**

```go
func (t Target) TileDim(elemType string) int {
    return t.LanesFor(elemType)  // TileDim == vector lane count
}
```

### Step 5: Transformer Changes (`cmd/hwygen/transformer.go`)

**5a. Handle `hwy.Tile[T]` type replacement in `specializeVecType` or similar:**

When the transformer encounters `hwy.Tile[float32]`, look up `tile_float32` in
the target's TypeMap and replace. For fallback, keep as `hwy.Tile[float32]`.

**5b. Handle `TileDim` constant substitution:**

Add `"TileDim"` to the same switch case that handles `"MaxLanes"` / `"NumLanes"`.
Replace `hwy.TileDim[T]()` with the target's `TileDim(elemType)` integer literal.

**5c. Handle tile method calls:**

The existing `IsMethod: true` transform already handles the conversion:
- `hwy.OuterProductAdd(&tile, row, col)` → `(&tile).OuterProductAdd(row, col)`
- `hwy.TileZero(&tile)` → `(&tile).Zero()`
- `hwy.TileStoreRow(&tile, i, dst)` → `(&tile).StoreRow(i, dst)`

The `&tile` first argument becomes the method receiver. Since tile methods use
pointer receivers, `(&tile).Method()` is valid Go. No new transformer logic needed
beyond the OpMap entries — the existing `IsMethod` transform handles it.

**5d. Track tile variable types:**

Add tile type tracking to `transformContext.varTypes` so the transformer knows
which variables are tiles (needed for type specialization of variable declarations
like `var tile hwy.Tile[T]`).

### Step 6: C Generator — Tile Intrinsics for :asm Targets

**6a. NEON:asm tile intrinsics (`cmd/hwygen/c_profiles.go`)**

Add tile operation mappings for NEON C profiles. `OuterProductAdd` emits a loop
over 4 lanes using `vfmaq_laneq_f32` (broadcast a lane of `row` and FMA with `col`):

```c
// Generated C for NEON:asm OuterProductAdd
tile.rows[0] = vfmaq_laneq_f32(tile.rows[0], col, row, 0);
tile.rows[1] = vfmaq_laneq_f32(tile.rows[1], col, row, 1);
tile.rows[2] = vfmaq_laneq_f32(tile.rows[2], col, row, 2);
tile.rows[3] = vfmaq_laneq_f32(tile.rows[3], col, row, 3);
```

The tile type in C is `struct { float32x4_t rows[4]; }`.

**6b. SVE_DARWIN:asm tile intrinsics (SME)**

Add SME tile operation mappings:

```go
TileZeroFn:        "svzero_za()",
OuterProductAddFn: "svfmopa_za32_f32_m(0, pg, pg, %s, %s)",
OuterProductSubFn: "svfmops_za32_f32_m(0, pg, pg, %s, %s)",
TileStoreRowFn:    "svst1_f32(pg, %s, svread_hor_za32_f32_m(svundef_f32(), pg, 0, %d))",
TileLoadColFn:     "svwrite_ver_za32_f32_m(0, %d, pg, svld1_f32(pg, %s))",
```

The tile type for SME is implicit (ZA register) — no struct needed. Functions using
tile ops must have `__arm_streaming __arm_out("za")` attributes.

**6c. Update `c_ast_translator.go`**

Recognize tile operations in the AST and emit the corresponding C intrinsics for
both NEON:asm and SVE_DARWIN:asm. Map `hwy.Tile[T]` to the appropriate C type
(struct-of-vectors for NEON, implicit ZA for SME).

### Step 7: Example + Tests

**7a. Unit test:** `hwy/tile_test.go` — verify scalar tile operations match naive
matmul for small sizes.

**7b. hwygen test:** `cmd/hwygen/hwygen_test.go` — add a test case with a base
function using tile ops, verify generated code for each target has correct type
replacements and method calls.

**7c. Integration example:** `hwy/contrib/matmul/outerproduct_base.go` — a simple
outer-product kernel using tile ops, generated for all targets. Run with
`GOEXPERIMENT=simd go test ./hwy/contrib/matmul/...` to verify correctness on
hardware.

## Verification

```bash
# Build hwygen
go build ./cmd/hwygen

# Run hwygen tests
go test -v ./cmd/hwygen/...

# Build everything with SIMD
GOEXPERIMENT=simd go build ./...

# Run all tests with SIMD
GOEXPERIMENT=simd go test ./...

# Run tile-specific tests
go test -v -run TestTile ./hwy/...
GOEXPERIMENT=simd go test -v -run TestTile ./hwy/...

# Fallback path
HWY_NO_SIMD=1 GOEXPERIMENT=simd go test -v -run TestTile ./hwy/...
```

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `hwy/tile.go` | **Create** | Scalar `Tile[T]` type + all tile operations |
| `hwy/tile_test.go` | **Create** | Unit tests for scalar tile ops |
| `hwy/asm/tile_neon.go` | **Create** | NEON tile types + methods (arm64) |
| `hwy/asm/tile_avx2.go` | **Create** | AVX2 tile types + methods (amd64+simd) |
| `hwy/asm/tile_avx512.go` | **Create** | AVX512 tile types + methods (amd64+simd) |
| `hwy/asm/vec_neon.go` | **Modify** | Add `MulAddLaneAcc` for NEON outer product |
| `cmd/hwygen/targets.go` | **Modify** | TypeMap + OpMap + TileDim for all targets |
| `cmd/hwygen/transformer.go` | **Modify** | Tile type replacement + TileDim constant |
| `cmd/hwygen/c_profiles.go` | **Modify** | NEON:asm + SME tile intrinsic mappings |
| `cmd/hwygen/c_ast_translator.go` | **Modify** | Tile op translation to C (NEON + SME) |
| `cmd/hwygen/hwygen_test.go` | **Modify** | Tile generation tests |
