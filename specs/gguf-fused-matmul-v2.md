# Fused SMOPA C Kernels for GGUF K-quant MatMul

## Context

The SMOPA matmul is functional but only 1.15x faster than NEON vecdot at M=32, K=4096. The bottleneck is Go overhead wrapping tiny SMOPA calls: for Q4_K, the Go code makes 8 separate SMOPA kernel calls per K-block with expensive scalar Go extraction, panel packing, and scale accumulation between each call. The actual SMOPA compute is a tiny fraction of the runtime.

The fix is to **fuse** the entire per-K-block inner loop into a single C function per quant type. One C function call replaces: quant extraction + panel packing + all sub-block SMOPA calls + scale accumulation. Target: 3-5x over NEON vecdot at M≥32.

## Design

### Kernel Granularity

One fused C kernel per quant type, processing **one K-block (256 values) for one 16M × 16N tile**. The outer loops (N-tiles, M-tiles, K-blocks) and activation quantization remain in Go.

### Files in `hwy/contrib/gguf/c/` (NEW C sources)

| File | Quant | SMOPA variant | Sub-blocks |
|------|-------|--------------|------------|
| `fused_smopa_q4k_arm64.c` | Q4_K | SUMOPA (signed×unsigned) | 8 × 32 |
| `fused_smopa_q5k_arm64.c` | Q5_K | SUMOPA | 8 × 32 |
| `fused_smopa_q6k_arm64.c` | Q6_K | SMOPA (signed×signed) | 16 × 16 |
| `fused_smopa_q2k_arm64.c` | Q2_K | SUMOPA | 16 × 16 |
| `fused_smopa_q3k_arm64.c` | Q3_K | SMOPA | 16 × 16 |

### Kernel Signatures

All kernels use 8 arguments (fits ARM64 register ABI, proven to work with GoAT):

**Unsigned types (Q4_K, Q5_K, Q2_K) — SUMOPA:**
```c
void fused_sumopa_q4k(
    unsigned char * restrict wBlocks,  // weight block base for this N-tile + K-block
    long wRowBytes,                     // stride between weight rows
    unsigned char * restrict aBlocks,  // Q8_K block base for this M-tile + K-block
    long aRowBytes,                     // stride between activation rows
    float * restrict acc,               // 16×16 float32 accumulator (add-to)
    float * restrict meta,              // packed float metadata (see layout below)
    long * restrict aBsums,            // pre-computed bsums [16 × numSubBlocks]
    long nCols                          // valid N columns (1-16)
) __arm_streaming __arm_out("za");
```

**Signed types (Q6_K, Q3_K) — SMOPA:**
```c
void fused_smopa_q6k(
    unsigned char * restrict wBlocks,
    long wRowBytes,
    unsigned char * restrict aBlocks,
    long aRowBytes,
    float * restrict acc,
    float * restrict meta,              // packed: wD[16] + wScales[16*numSubBlocks]
    long nCols,
    long mRows                          // valid M rows (1-16)
) __arm_streaming __arm_out("za");
```

### Meta Buffer Layout

Pre-parsed in Go, passed as flat float32 array:

**Q4_K / Q5_K (8 sub-blocks):** 288 floats
- `[0:16]` wD — 16 block-level d values (pre-converted from fp16)
- `[16:32]` wDmin — 16 block-level dmin values
- `[32:160]` wScales — 16 × 8 sub-block scales
- `[160:288]` wMins — 16 × 8 sub-block mins

**Q6_K / Q3_K (16 sub-blocks, no min):** 272 floats
- `[0:16]` wD — 16 d values
- `[16:272]` wScales — 16 × 16 sub-block scales

**Q2_K (16 sub-blocks):** 544 floats
- `[0:16]` wD, `[16:32]` wDmin, `[32:288]` wScales (16×16), `[288:544]` wMins (16×16)

### Internal Kernel Structure (Q4_K pseudocode)

```c
void fused_sumopa_q4k(...) __arm_streaming __arm_out("za") {
    svbool_t pg8 = svptrue_b8();
    svbool_t pg32 = svptrue_b32();
    svint8_t zeros_s8 = svdup_s8(0);

    // Stack panels: 512 bytes each (kGroups=8, 64 bytes per group)
    signed char aPanel[512];
    unsigned char bPanel[512];

    for (int j = 0; j < 8; j++) {  // 8 sub-blocks
        // 1. Zero bPanel via SVE stores (8 × 64-byte stores)
        for (int k4 = 0; k4 < 8; k4++) {
            svst1_s8(pg8, (signed char *)bPanel + k4 * 64, zeros_s8);
        }

        // 2. Extract Q4_K nibbles + pack into B panel (inline, no helper)
        int chunk = j / 2;
        int isHigh = j % 2;
        int qOff = chunk * 32;
        for (int col = 0; col < nCols; col++) {
            unsigned char *qs = wBlocks + col * wRowBytes + 16;  // skip header
            for (int k4 = 0; k4 < 8; k4++) {
                for (int g = 0; g < 4; g++) {
                    int idx = qOff + k4 * 4 + g;
                    unsigned char val;
                    if (isHigh == 0) {
                        val = qs[idx] & 0x0F;
                    } else {
                        val = qs[idx] >> 4;
                    }
                    bPanel[k4 * 64 + col * 4 + g] = val;
                }
            }
        }

        // 3. Pack A panel from Q8_K qs (activations)
        for (int row = 0; row < 16; row++) {
            signed char *aQs = (signed char *)aBlocks + row * aRowBytes + 4 + j * 32;
            for (int k4 = 0; k4 < 8; k4++) {
                for (int g = 0; g < 4; g++) {
                    aPanel[k4 * 64 + row * 4 + g] = aQs[k4 * 4 + g];
                }
            }
        }

        // 4. SUMOPA: signed(activations) × unsigned(weights) → int32
        svzero_za();
        for (int k4 = 0; k4 < 8; k4++) {
            svint8_t av = svld1_s8(pg8, aPanel + k4 * 64);
            svuint8_t bv = svld1_u8(pg8, bPanel + k4 * 64);
            svsumopa_za32_s8_m(0, pg8, pg8, av, bv);
        }

        // 5. Extract ZA tile + apply scales + accumulate to float32
        for (int row = 0; row < 16; row++) {
            svint32_t za_row = svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row);
            // Read activation d_a from Q8_K block header
            float dA = *(float *)(aBlocks + row * aRowBytes);  // little-endian ARM64
            for (int col = 0; col < 16; col++) {
                // ... extract za_row[col], apply meta scales, accumulate to acc
            }
        }
    }
}
```

**Key optimization**: The extraction/packing/SMOPA loops will be compiled by clang -O3 which can auto-vectorize the extraction loops. All work runs in compiled C — zero Go overhead in the hot path.

**Scale accumulation detail (step 5)**: After extracting the ZA int32 tile, for each valid (row, col):
```
raw = za_tile[row][col]   // int32 from SMOPA
d_w = meta[col]           // wD[col]
sc_j = meta[32 + col*8 + j]  // wScales[col*8 + j]
d_a = *(float*)(aBlocks + row * aRowBytes)
acc[row*16 + col] += (float)raw * d_w * sc_j * d_a
// Min correction (Q4_K, Q5_K, Q2_K only):
dmin_w = meta[16 + col]   // wDmin[col]
mn_j = meta[160 + col*8 + j]  // wMins[col*8 + j]
acc[row*16 + col] -= dmin_w * mn_j * d_a * (float)aBsums[row*8 + j]
```

**ZA tile column extraction**: SVE doesn't provide per-element indexing. Use `svlastb` with a predicate that has exactly one active lane, or store the ZA row to a temp buffer and index. The simplest approach: store ZA row to a stack `int[16]` array, then loop over cols.

### Go Integration

**Modified file: `hwy/contrib/gguf/matmul_smopa.go`**

Replace the inner loop of `smeGGUFMatMul`. The new structure:

```
for nTile (16 cols):
  for mTile (16 rows):
    clear(accTile)
    for kb (K-blocks):
      // Pre-parse metadata in Go (cheap: ~16 fp16 conversions + scale unpacking)
      fillMeta(qt, weights, nTile, kb, nCols, &meta)
      // Pre-compute aBsums in Go
      fillBsums(aData, mTile, kb, aRowBytes, info, &aBsums)
      // ONE fused kernel call per K-block
      switch qt {
      case TypeQ4_K: fused_sumopa_q4k(wBase, wRowBytes, aBase, aRowBytes, acc, meta, aBsums, nCols)
      ...
      }
    write accTile to output
```

**New file: `hwy/contrib/gguf/fused_smopa_wrappers.go`** (`//go:build !noasm && arm64`)

Contains:
- `//go:generate go tool goat c/fused_smopa_q4k_arm64.c -O3 --target arm64 --target-os darwin -e="-march=armv9-a+sme+sme-i16i64" -o asm/`
- (similar for q5k, q6k, q2k, q3k)
- Go wrapper functions `FusedSUMOPAQ4K(...)`, `FusedSMOPAQ6K(...)`, etc.
- `fillMeta()` and `fillBsums()` helpers for pre-parsing metadata
- Pool for meta buffer: `var fusedMetaPool = sync.Pool{...}`

### GoAT Compilation

Each C file compiled separately:
```bash
go tool goat c/fused_smopa_q4k_arm64.c -O3 --target arm64 --target-os darwin \
  -e="-march=armv9-a+sme+sme-i16i64" -o asm/
```

Output goes to `hwy/contrib/gguf/asm/`:
- `fused_smopa_q4k_arm64.s` — assembly
- `fused_smopa_q4k_arm64.go` — `//go:noescape` declarations

### GoAT Constraints Respected

- No `static inline` helpers — all extraction inlined in the function body
- No `__builtin_*` — fp16 conversion done in Go, passed as float32 in meta
- No `union` — float reading uses `*(float *)ptr` (valid on little-endian ARM64)
- No single-line `if` with braces — multi-line format throughout
- Panel zeroing via SVE stores (avoids memset optimization)
- Post-function SME attributes: `__arm_streaming __arm_out("za")`
- All args are pointer or `long` (supported types)
- Max 8 arguments per kernel (fits register ABI)

## Implementation Order

1. **Q4_K fused kernel** — write C, compile with GoAT, write Go wrapper
2. **Q4_K test** — compare fused vs unfused vs vecdot output
3. **Q4_K benchmark** — verify speedup
4. **Remaining unsigned types** — Q5_K (similar to Q4_K), Q2_K (16 sub-blocks)
5. **Signed types** — Q6_K, Q3_K (SMOPA instead of SUMOPA)
6. **Wire into matmul_smopa.go** — replace inner loop, keep unfused as fallback
7. **Parallel matmul** — update `parallelSMEGGUFMatMul` with fused path
8. **Full benchmark suite** — all quant types, multiple dimensions

## Verification

```bash
# Build
GOEXPERIMENT=simd go build ./hwy/contrib/gguf/...

# Test fused vs vecdot correctness
GOEXPERIMENT=simd go test -v -run 'SMEGGUFMatMul|FusedSMOPA' ./hwy/contrib/gguf/...

# Test SME disabled fallback
HWY_NO_SME=1 GOEXPERIMENT=simd go test ./hwy/contrib/gguf/...

# Benchmark fused vs unfused vs vecdot
GOEXPERIMENT=simd go test -bench='BenchmarkSMEGGUFMatMul' -benchmem ./hwy/contrib/gguf/...
```
