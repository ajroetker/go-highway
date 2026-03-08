# K-quant Fused Dequant+MatMul (llama.cpp approach)

## Context

The integer dot product feature (DotInt with sdot/udot NEON assembly) is complete. The next step is implementing K-quant fused matmul — the same approach llama.cpp uses for inference. Instead of dequantizing K-quant weights to float32, we:

1. Quantize float32 activations to Q8_K (int8 with per-sub-block sums)
2. Compute dot products directly between quantized weights (Q4_K, Q6_K, etc.) and Q8_K activations
3. Use integer accumulation in inner loops, float32 only for final scaling

This avoids the `N*K` float32 intermediate that dequantize+float-matmul requires. The existing Tier 1 matmul (`Q4_0`/`Q8_0`) already uses this pattern — we extend it to K-quant types.

## Q8_K Block Format

292 bytes per block (256 values):
- `d` (4 bytes): float32 scale (little-endian) — NOT fp16 like Q8_0
- `qs` (256 bytes): int8 quantized values
- `bsums` (32 bytes): 16 x int16 sub-block sums (LE), where `bsums[j] = sum(qs[j*16..j*16+15])`

The `bsums` optimization: K-quant types like Q4_K have per-sub-block min values. The correction term `dmin * m * sum(activation_quants)` can be computed in O(16) using pre-computed bsums instead of O(256) per block.

## Files to Create/Modify

### 1. `hwy/contrib/gguf/gguf_base.go` — Add helper functions

Add `f32LE` and `i16LE` alongside existing `fp16LE`:

```go
// f32LE decodes a little-endian float32 from 4 bytes.
func f32LE(b []uint8) float32 {
    bits := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
    return math.Float32frombits(bits)
}

// i16LE decodes a little-endian int16 from 2 bytes.
func i16LE(lo, hi uint8) int16 {
    return int16(uint16(lo) | uint16(hi)<<8)
}
```

### 2. `hwy/contrib/gguf/quantize_kquant_base.go` (NEW) — Q8_K Quantization

```go
//go:generate go run ../../../cmd/hwygen -input quantize_kquant_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch ggufkqquant
```

`BaseQuantizeQ8_K` follows the same pattern as `BaseQuantizeQ8_0` (`quantize_base.go:35-130`):
- Find amax over 256 values
- Compute `d = amax / 127.0`, `id = 127.0 / amax`
- Encode d as **float32 LE** (4 bytes, not fp16 like Q8_0)
- Quantize values: `qs[i] = round(input[i] * id)` clamped to [-128, 127]
- Compute 16 bsums: `bsums[j] = sum(qs[j*16..j*16+15])` as int16 LE

SIMD vectorization: same as `BaseQuantizeQ8_0` — Load+Abs for amax, Load+Mul+Round+Clamp for quantization, scalar bsums accumulation.

### 3. `hwy/contrib/gguf/vecdot_kquant_base.go` (NEW) — K-quant VecDot Functions

```go
//go:generate go run ../../../cmd/hwygen -input vecdot_kquant_base.go -output . -targets avx2,avx512,neon:asm,fallback -dispatch ggufkqvecdot
```

Separate dispatch group from Tier 1 (`-dispatch ggufkqvecdot`) because hwygen processes one input file per invocation.

#### `BaseVecDotQ4_KQ8_K` (Priority 1)

Algorithm (from llama.cpp `ggml_vec_dot_q4_K_q8_K`):

For each super-block pair (256 values):
1. Read weight block: `d_w` (fp16), `dmin_w` (fp16), scales (12 bytes packed), qs (128 nibble bytes)
2. Read activation block: `d_a` (float32), activation qs (256 bytes), bsums (16 x int16)
3. Unpack 8 (scale, min) pairs from 12-byte packed format (same as `BaseDequantizeQ4K` at `gguf_base.go:401-410`)
4. **Min correction via bsums**: For each sub-block pair `j`, accumulate `mns[j] * (bsums[2*j] + bsums[2*j+1])` as int32. Multiply by `dmin_w * d_a` at the end.
5. **Integer dot product**: For each of 8 sub-block pairs (32 values each), extract low/high nibbles, compute `sum(nibble * activation_qs)` as int32, multiply by `scs[j]`.
6. Final: `sumf += d_w * d_a * (sum of scaled sub-block dots) - dmin_w * d_a * (bsums correction)`

The inner loop uses int32 scalar accumulation (not float32 SIMD like Tier 1). The integer dot product primitive (`DotInt`) is NOT used here because the weight data needs nibble extraction before dot product — it's not a direct int8×int8 operation.

#### `BaseVecDotQ6_KQ8_K` (Priority 1)

Algorithm (from llama.cpp `ggml_vec_dot_q6_K_q8_K`):

For each super-block pair:
1. Read weight block: ql (128 bytes), qh (64 bytes), scales (16 x int8), d_w (fp16)
2. Read activation block: d_a (float32), qs (256 bytes), bsums (not needed — Q6_K has no min)
3. For each of 16 sub-blocks of 16 values: extract 6-bit quants (same as `BaseDequantizeQ6K` at `gguf_base.go:341-372`), compute `sum((q6-32) * activation_qs)` as int32, multiply by `sc[j]`.
4. Final: `sumf += d_w * d_a * (sum of scaled sub-block dots)`

#### `BaseVecDotQ2_KQ8_K`, `BaseVecDotQ3_KQ8_K`, `BaseVecDotQ5_KQ8_K` (Priority 2)

Same pattern — extract weight quants per sub-block, integer accumulate against Q8_K activation quants, apply scales. The quant extraction logic mirrors the corresponding dequantize functions already in `gguf_base.go`.

### 4. `hwy/contrib/gguf/matmul.go` — Wire K-quant dispatch

Extend `getVecDot` (`matmul.go:112-120`) to return K-quant vecdot functions:
```go
case TypeQ4_K:
    return VecDotQ4_KQ8_K
case TypeQ6_K:
    return VecDotQ6_KQ8_K
// ... etc for Q2_K, Q3_K, Q5_K
```

Extend `getQuantize` (`matmul.go:123-130`) to return Q8_K quantization for K-quant types:
```go
case TypeQ2_K, TypeQ3_K, TypeQ4_K, TypeQ5_K, TypeQ6_K:
    return QuantizeQ8_K
```

No changes needed to the matmul loop itself — `GGUFMatMul` and `ParallelGGUFMatMul` already handle variable block sizes via `ValuesPerBlock(qt)`, `BytesPerBlock(qt)`, and `ActivationBlockSize(qt)`.

### 5. Tests

**`hwy/contrib/gguf/quantize_kquant_test.go`** (NEW):
- `TestQuantizeQ8_K_RoundTrip`: quantize float32 → Q8_K, verify `d * qs[i]` approximates input
- `TestQuantizeQ8_K_Bsums`: verify bsums match sum of corresponding qs sub-blocks
- `TestQuantizeQ8_K_Zeros`: all-zero input produces zero d and zero qs
- `TestQuantizeQ8_K_FallbackMatchesDispatch`: compare SIMD vs fallback paths
- `BenchmarkQuantizeQ8_K`: {256, 1024, 4096} values

**`hwy/contrib/gguf/vecdot_kquant_test.go`** (NEW):
- `TestVecDotQ4_KQ8_K_MatchesDequant`: quantize weights to Q4_K, quantize activations to Q8_K, compute vecdot, compare against dequantize-then-float-dot result
- Same for Q6_K, Q2_K, Q3_K, Q5_K
- `TestVecDotQ4_KQ8_K_FallbackMatchesDispatch`
- `BenchmarkVecDotQ4_KQ8_K`: {1, 4, 16, 64} blocks

**`hwy/contrib/gguf/matmul_kquant_test.go`** (NEW):
- `TestGGUFMatMul_Q4_K`: end-to-end M=2, K=256, N=3 — quantize random weights to Q4_K, run matmul, compare against float reference
- Same for Q6_K
- `TestParallelGGUFMatMul_Q4_K`: verify parallel matches serial

## Implementation Order

1. Add `f32LE` and `i16LE` helpers to `gguf_base.go`
2. Create `quantize_kquant_base.go` with `BaseQuantizeQ8_K` + hwygen generate
3. Write quantization tests (`quantize_kquant_test.go`)
4. Create `vecdot_kquant_base.go` with `BaseVecDotQ4_KQ8_K` and `BaseVecDotQ6_KQ8_K` + hwygen generate
5. Write vecdot tests (`vecdot_kquant_test.go`)
6. Wire dispatch in `matmul.go`
7. Write matmul integration tests (`matmul_kquant_test.go`)
8. Add remaining vecdot functions (Q2_K, Q3_K, Q5_K)
9. Benchmarks

## Key Design Decisions

**Separate dispatch group**: K-quant vecdot uses `-dispatch ggufkqvecdot` (not the existing `ggufvecdot`) because hwygen processes one input file per invocation. Keeps Tier 1 and K-quant code independent.

**Integer accumulation, not DotInt**: Despite having the DotInt primitive, K-quant vecdot doesn't use it directly. The weight data requires nibble/bit extraction before multiplication — it's not a raw int8×int8 dot. The inner loops accumulate int32 scalars. Future NEON optimization can use `vdotq_s32` on the extracted quants, but the base function uses scalar arithmetic for correctness and GoAT compatibility.

**float32 scale in Q8_K (not fp16)**: Q8_K uses float32 `d` (4 bytes) unlike Q8_0's fp16 `d` (2 bytes). This matches llama.cpp's `block_q8_K` format and provides better precision for the larger 256-value blocks.

**bsums optimization**: Only used for types with min values (Q4_K, Q5_K, Q2_K). Q6_K and Q3_K have no min, so bsums are unused in their vecdot. The bsums are still computed during Q8_K quantization (they're part of the format).

**Quant extraction reuse**: The sub-block quant extraction in each vecdot mirrors the corresponding `BaseDequantize*` function in `gguf_base.go`. The difference is that instead of `output[i] = d * scale * (quant - bias)`, we compute `sum += (quant - bias) * activation_qs[i]` as integer.

## Verification

```bash
# Build
GOEXPERIMENT=simd go build ./hwy/contrib/gguf/...

# Test all (SIMD path on Apple Silicon)
GOEXPERIMENT=simd go test -v ./hwy/contrib/gguf/...

# Test fallback
HWY_NO_SIMD=1 GOEXPERIMENT=simd go test -v ./hwy/contrib/gguf/...

# hwygen tests still pass
GOEXPERIMENT=simd go test -v ./cmd/hwygen/...

# Benchmark
GOEXPERIMENT=simd go test -bench=Q4_K -benchmem ./hwy/contrib/gguf/...
GOEXPERIMENT=simd go test -bench=Q6_K -benchmem ./hwy/contrib/gguf/...
GOEXPERIMENT=simd go test -bench=QuantizeQ8_K -benchmem ./hwy/contrib/gguf/...
```
