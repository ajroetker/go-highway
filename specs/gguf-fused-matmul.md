# SMOPA Integer Path for GGUF K-quant MatMul

## Context

The K-quant fused matmul is complete with NEON assembly (vecdot-based). The current `GGUFMatMul` loop is row-by-row: quantize activations to Q8_K, then for each output element compute a vecdot. This works but doesn't leverage SME's outer product accelerators.

llama.cpp/KleidiAI uses SMOPA (integer outer product) to compute matmul directly on quantized data — keeping weights in INT4/INT8, quantizing activations to INT8, accumulating in INT32 via SMOPA, then applying scales. This avoids dequantizing to float32 entirely.

go-highway already has UMOPA infrastructure (`int8x8MatMulSME` in `z_matmul_arm64.go:3271-3416`, `TileUMOPAU8` kernel in `asm/block_kernel_umopa_wrappers.go`). We extend this to GGUF K-quant types using **signed SMOPA** (`svmopa_za32_s8_m`) since Q8_K activations are signed int8.

### Key Challenge: Per-Sub-Block Scales

Unlike simple int8 matmul (per-tensor zero point), GGUF K-quant has **per-sub-block scales**:
- Q4_K: 8 sub-blocks of 32 values, each with its own `scale` and `min`
- Q6_K: 16 sub-blocks of 16 values, each with its own `scale`
- Q2_K/Q3_K/Q5_K: similar sub-block structure

SMOPA gives raw `int32 = sum(int8_a * int8_b)` accumulation. To apply per-sub-block scales, we split the K dimension by sub-block boundaries and run SMOPA independently per sub-block, then scale-weight and float-accumulate the per-sub-block int32 results.

### Algorithm

For each M-tile (16 rows) × N-tile (16 columns):
```
for each GGUF super-block (256 values along K):
  for each sub-block j (e.g. 32 values for Q4_K):
    1. Extract weight quants for 16 N-rows → int8 [16 × subBlockSize]
    2. Extract activation int8s for 16 M-rows → int8 [16 × subBlockSize]
    3. Pack both into SMOPA panel format (interleaved groups of 4)
    4. SMOPA → int32 [16 × 16] tile
    5. float32 += int32_tile * (d_w * sc[j] * d_a)   // per sub-block
  Apply min correction: float32 -= dmin_w * mn[j] * d_a * bsums  // for Q4_K/Q5_K/Q2_K
```

For Q4_K (8 sub-blocks of 32): 8 SMOPA calls per super-block, each with kGroups=8 (32/4).
For Q6_K (16 sub-blocks of 16): 16 SMOPA calls per super-block, each with kGroups=4 (16/4).

## Files to Create/Modify

### 1. `hwy/contrib/matmul/c/block_kernel_smopa_arm64.c` (NEW) — Signed SMOPA tile kernel

New C kernel for signed int8×int8→int32 outer product (the existing `tile_umopa_u8` is unsigned).

```c
// Signed SMOPA: int8×int8→int32 outer product
void tile_smopa_s8(signed char * restrict aPanel, signed char * restrict bPanel,
                   int * restrict c, long kGroups)
    __arm_streaming __arm_out("za") {
    svbool_t pg8 = svptrue_b8();
    svbool_t pg32 = svptrue_b32();
    svzero_za();
    for (long k4 = 0; k4 < kGroups; k4++) {
        svint8_t av = svld1_s8(pg8, aPanel + k4 * 64);
        svint8_t bv = svld1_s8(pg8, bPanel + k4 * 64);
        svmopa_za32_s8_m(0, pg8, pg8, av, bv);  // SIGNED SMOPA
    }
    int *c_ptr = c;
    for (int row = 0; row < 16; row++) {
        svint32_t za_row = svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row);
        svst1_s32(pg32, c_ptr, za_row);
        c_ptr += 16;
    }
}
```

### 2. `hwy/contrib/matmul/asm/block_kernel_smopa_wrappers.go` (NEW) — Go wrapper

```go
//go:build !noasm && arm64

//go:generate go tool goat ../c/block_kernel_smopa_arm64.c -O3 --target arm64 --target-os darwin -e="-march=armv9-a+sme+sme-i16i64"

// TileSMOPAS8 computes a single 16×16 int32 output tile using signed SMOPA.
// Same panel format as TileUMOPAU8 but with signed int8 inputs.
func TileSMOPAS8(aPanel, bPanel []int8, c []int32, kGroups int) {
    tile_smopa_s8(...)
}
```

### 3. `hwy/contrib/gguf/matmul_smopa.go` (NEW) — SMOPA-based GGUF matmul

Core SMOPA matmul implementation. Lives in the gguf package, imports `matmul/asm` for the SMOPA kernel (no cycle — matmul doesn't import gguf).

```go
//go:build !noasm && arm64

package gguf

// SMEGGUFMatMul is the SME-accelerated GGUF matmul using SMOPA integer outer product.
// Set to non-nil by init() on SME-capable hardware.
var SMEGGUFMatMul func(input []float32, weights []uint8, output []float32,
    M, K, N int, qt QuantType)
var SMEParallelGGUFMatMul func(pool workerpool.Executor, input []float32,
    weights []uint8, output []float32, M, K, N int, qt QuantType)
```

**smeGGUFMatMul implementation:**

```
func smeGGUFMatMul(input, weights, output, M, K, N, qt):
  if !hwy.HasSME() || K < 256 || N < 16:
    GGUFMatMul(input, weights, output, M, K, N, qt)  // fallback to vecdot
    return

  // Step 1: Quantize ALL activations to Q8_K (reuse existing QuantizeQ8_K)
  //   M rows × K values → M × (K/256) Q8_K blocks
  aData = make([]uint8, M * nblocks * BlockSizeQ8K)
  for m := range M:
    QuantizeQ8_K(input[m*K:(m+1)*K], aData[m*aRowBytes:(m+1)*aRowBytes])

  // Step 2: Pad M to multiple of 16
  paddedM = AlignUp(M, 16)

  // Step 3: For each N-tile of 16 weight rows:
  defer hwy.SMEGuard()()
  for nTile := 0; nTile < N; nTile += 16:
    // Zero float32 accumulator tile [paddedM × 16]
    clear(accTile)

    // For each K-block (256 values = 1 GGUF super-block):
    for kb := range nblocks:
      // Extract per-sub-block scales, mins, and quants for 16 weight rows
      // Run SMOPA per sub-block, accumulate with scale into float32

      processKBlock(qt, weights, aData, accTile, nTile, kb, paddedM, N, nblocks)

    // Write accTile to output[m, nTile:nTile+16]
    for m := range M:
      copy(output[m*N+nTile:], accTile[m*16:(m+1)*16][:min(16, N-nTile)])
```

**processKBlock** — the core per-super-block SMOPA loop:

```
func processKBlock(qt, weights, aData, accTile, nTile, kb, paddedM, N, nblocks):
  // For each of the 16 (or fewer) weight rows in this N-tile:
  //   Parse the GGUF block: extract d, dmin, scales, mins, quant bytes
  // For each sub-block j:
  //   Extract int8 quants from weight block for all 16 rows → bPanel
  //   Extract int8 values from Q8_K activation block for all paddedM rows → aPanel
  //   Pack panels in SMOPA interleaved format
  //   TileSMOPAS8(aPanel, bPanel, tileI32, kGroups)
  //   accTile += float32(tileI32) * (d_w[row] * sc[row][j] * d_a[row])
  //
  // Apply min correction via bsums (Q4_K, Q5_K, Q2_K only):
  //   accTile -= dmin_w[row] * mn[row][j] * d_a[row] * bsums[row][j]
```

The quant extraction mirrors existing dequantize functions in `gguf_base.go`:
- Q4_K (`gguf_base.go:401-455`): low/high nibble extraction → int8 [0,15]
- Q6_K (`gguf_base.go:341-372`): 6-bit extraction from ql+qh → int8 [0,63], subtract 32
- Q2_K (`gguf_base.go:459-507`): 2-bit extraction → int8 [0,3]
- Q3_K (`gguf_base.go:509-563`): 3-bit extraction from qs+hmask → int8 [0,7], subtract 4
- Q5_K (`gguf_base.go:565-630`): 5-bit extraction from qs+qh → int8 [0,31]

### 4. `hwy/contrib/gguf/smopa_extract.go` (NEW) — Per-type quant extraction to int8

Helper functions that extract quants from each GGUF block format into contiguous int8 slices for SMOPA panel packing.

```go
// extractQ4KSubBlock extracts 32 int8 quants from Q4_K sub-block j.
// Returns quants in [0, 15] range (unsigned nibbles).
func extractQ4KSubBlock(block []uint8, j int, dst []int8) {
    // qs at offset 12 in Q4_K block, 128 nibble bytes total
    qs := block[12:]
    base := j * 16 // 32 values = 16 nibble bytes
    for i := range 16 {
        lo := int8(qs[base+i] & 0x0F)
        hi := int8(qs[base+i] >> 4)
        dst[2*i] = lo
        dst[2*i+1] = hi
    }
}

// Similar functions for Q6_K, Q2_K, Q3_K, Q5_K sub-blocks
```

### 5. `hwy/contrib/gguf/z_gguf_sme_arm64.go` (NEW) — SME dispatch override

```go
//go:build !noasm && arm64

package gguf

import "github.com/ajroetker/go-highway/hwy"

func init() {
    if hwy.NoSimdEnv() { return }
    if hwy.HasSME() {
        SMEGGUFMatMul = smeGGUFMatMul
        SMEParallelGGUFMatMul = parallelSMEGGUFMatMul
    }
}
```

### 6. `hwy/contrib/gguf/matmul.go` — Update to check SME dispatch first

```go
func GGUFMatMul(...) {
    if SMEGGUFMatMul != nil {
        SMEGGUFMatMul(input, weights, output, M, K, N, qt)
        return
    }
    // existing vecdot loop (unchanged)
}

func ParallelGGUFMatMul(...) {
    if SMEParallelGGUFMatMul != nil {
        SMEParallelGGUFMatMul(pool, input, weights, output, M, K, N, qt)
        return
    }
    // existing parallel vecdot loop (unchanged)
}
```

### 7. `hwy/contrib/gguf/matmul_smopa_parallel.go` (NEW) — Parallel SMOPA matmul

```go
//go:build !noasm && arm64

// parallelSMEGGUFMatMul distributes N-tiles across workers.
// Activations are quantized once (shared), each worker gets its own SMEGuard.
func parallelSMEGGUFMatMul(pool, input, weights, output, M, K, N, qt):
  // Quantize all activations to Q8_K (shared)
  // ParallelFor over N-tiles (each 16 columns):
  //   defer hwy.SMEGuard()()
  //   process tiles for this worker's column range
```

## Sub-Block SMOPA Mechanics

### Q4_K Example (8 sub-blocks of 32 values)

For one GGUF super-block (256 values along K), weight row n, activation row m:

```
Weight block [n]: d_w (fp16), dmin_w (fp16), scales[12 bytes packed], qs[128 bytes]
  → 8 (scale, min) pairs unpacked from 12 bytes
  → 8 sub-blocks of 32 nibble values each

Activation block [m]: d_a (f32), qs[256 bytes int8], bsums[16 x int16]

For sub-block j (32 values):
  Weight quants: extract 32 nibbles → int8 [0, 15]
  Activation quants: aqs[j*32 : j*32+32] → int8 [-128, 127]

  SMOPA panels (kGroups = 32/4 = 8):
    aPanel[k4*64 + row*4 + g] = activation_qs[row_m, kb*256 + j*32 + k4*4 + g]
    bPanel[k4*64 + col*4 + g] = weight_nibble[col_n, j*32 + k4*4 + g]

  TileSMOPAS8 → tileI32[16 × 16]

  For each (m, n) in tile:
    accTile[m][n] += float32(tileI32[m][n]) * d_w[n] * sc[n][j] * d_a[m]

  Min correction (via bsums, j maps to bsums pair):
    accTile[m][n] -= dmin_w[n] * mn[n][j] * d_a[m] * float32(bsums_pair[m][j])
```

### Signedness

- Q8_K activations: signed int8 [-128, 127] → use directly
- Q4_K quants: unsigned [0, 15] → can use SUMOPA (signed×unsigned) or offset to signed
- Q6_K quants: [0, 63] centered at 32 → subtract 32 → signed [-32, 31] → SMOPA
- Q2_K quants: [0, 3] → use SUMOPA or offset
- Q3_K quants: [0, 7] centered at 4 → subtract 4 → signed [-4, 3] → SMOPA
- Q5_K quants: [0, 31] → use SUMOPA or offset

For types with unsigned quants (Q4_K, Q2_K, Q5_K), two approaches:
1. **SUMOPA** (`svmopa_za32_s8_u8_m`): signed A × unsigned B → int32. No offset needed.
2. **Offset to signed**: subtract midpoint, use SMOPA, apply correction via bsums.

Approach 1 (SUMOPA) is cleaner — we need a `tile_sumopa_s8u8` kernel variant.

### Kernel Variants Needed

| Kernel | A type | B type | Used by |
|--------|--------|--------|---------|
| `tile_smopa_s8` | int8 | int8 | Q6_K, Q3_K (quants already signed after bias subtraction) |
| `tile_sumopa_s8u8` | int8 | uint8 | Q4_K, Q2_K, Q5_K (signed activation × unsigned quants) |

Both kernels have the same structure as the existing `tile_umopa_u8`, just different intrinsics:
- `svmopa_za32_s8_m` for signed×signed
- `svmopa_za32_s8_u8_m` for signed×unsigned (SUMOPA)

## Tests

**`hwy/contrib/matmul/asm/smopa_test.go`** (NEW):
- `TestTileSMOPAS8_Identity`: simple known-input test
- `TestTileSUMOPAS8U8_Identity`: same for signed×unsigned
- Compare against scalar reference

**`hwy/contrib/gguf/matmul_smopa_test.go`** (NEW):
- `TestSMEGGUFMatMul_Q4_K`: M=32, K=256, N=64 — compare SMOPA result vs existing vecdot-based `GGUFMatMul` (with `HWY_NO_SME=1`)
- Same for Q6_K, Q2_K, Q3_K, Q5_K
- `TestSMEGGUFMatMul_LargerDims`: M=64, K=1024, N=256
- `TestSMEGGUFMatMul_NonAligned`: N not multiple of 16
- `TestSMEGGUFMatMul_SmallFallback`: verify small dims fall back to vecdot
- `TestParallelSMEGGUFMatMul_Q4_K`: parallel matches serial
- `BenchmarkSMEGGUFMatMul_Q4_K`: {M=1,4,16,32,64} × {K=256,1024,4096} × {N=256,1024,4096}, report GFLOPS
- All tests gated with `if !hwy.HasSME() { t.Skip("SME not available") }`

**`hwy/contrib/gguf/smopa_extract_test.go`** (NEW):
- `TestExtractQ4KSubBlock`: verify nibble extraction matches dequantize
- Same for all other types

## Implementation Order

1. Create signed SMOPA tile kernel (`tile_smopa_s8`) + GoAT + wrappers
2. Create signed×unsigned SUMOPA tile kernel (`tile_sumopa_s8u8`) + GoAT + wrappers
3. Write SMOPA/SUMOPA kernel unit tests
4. Implement `extractQ*SubBlock` functions for all K-quant types
5. Write extraction unit tests
6. Implement `smeGGUFMatMul` (serial) with `processKBlock` for Q4_K
7. Wire dispatch in `matmul.go` + create `z_gguf_sme_arm64.go`
8. Write Q4_K SMOPA matmul correctness tests
9. Extend to Q6_K, Q2_K, Q3_K, Q5_K
10. Implement `parallelSMEGGUFMatMul`
11. Write parallel tests
12. Benchmarks + performance tuning (min dim thresholds, buffer pools)

## Key Design Decisions

**Signed SMOPA kernels**: The existing `tile_umopa_u8` is unsigned. Q8_K activations are signed int8, so we need `tile_smopa_s8` (signed×signed) and `tile_sumopa_s8u8` (signed×unsigned). These go in `hwy/contrib/matmul/c/` alongside the existing UMOPA kernel.

**Per-sub-block accumulation**: We run SMOPA once per sub-block (not once per super-block). For Q4_K this means 8 SMOPA calls per K-block with kGroups=8 each. This is granular but necessary because each sub-block has a different scale. The int32→float32 conversion and scale multiplication happen after each SMOPA call.

**No weight repacking**: Unlike KleidiAI's `qsi4c32p` format which pre-packs weights for optimal tile access, we extract quants on-the-fly from the native GGUF format. This avoids a separate packing step and works with weights as stored. Future optimization could pre-pack weights.

**SUMOPA for unsigned quant types**: Q4_K, Q2_K, Q5_K have unsigned quants [0, N]. Using SUMOPA (signed×unsigned) avoids needing to offset quants to signed range and correcting with bsums. This is cleaner than the offset approach.

**Min correction via bsums**: For types with min values (Q4_K, Q5_K, Q2_K), the bsums pre-computed during Q8_K quantization provide O(1) min correction per sub-block pair, same as the vecdot path.

**Activation quantization once**: All M rows are quantized to Q8_K upfront (not per-tile). The Q8_K data is shared across all N-tiles. For parallel matmul, the quantization happens before `ParallelFor`.

## Verification

```bash
# Build
GOEXPERIMENT=simd go build ./hwy/contrib/gguf/...
GOEXPERIMENT=simd go build ./hwy/contrib/matmul/...

# Test SMOPA kernel
GOEXPERIMENT=simd go test -v -run 'SMOPA|SUMOPA' ./hwy/contrib/matmul/...

# Test SME GGUF matmul (Apple Silicon M4+)
GOEXPERIMENT=simd go test -v -run SMEGGUFMatMul ./hwy/contrib/gguf/...

# Test with SME disabled (falls back to NEON vecdot)
HWY_NO_SME=1 GOEXPERIMENT=simd go test -v ./hwy/contrib/gguf/...

# Test pure fallback
HWY_NO_SIMD=1 GOEXPERIMENT=simd go test -v ./hwy/contrib/gguf/...

# Benchmark SMOPA vs NEON vecdot
GOEXPERIMENT=simd go test -bench=SMEGGUFMatMul -benchmem ./hwy/contrib/gguf/...
HWY_NO_SME=1 GOEXPERIMENT=simd go test -bench=GGUFMatMul -benchmem ./hwy/contrib/gguf/...

# Existing tests still pass
GOEXPERIMENT=simd go test ./cmd/hwygen/...
GOEXPERIMENT=simd go test ./hwy/contrib/matmul/...
GOEXPERIMENT=simd go test ./hwy/contrib/gguf/...
```
