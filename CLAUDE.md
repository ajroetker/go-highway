# Go Highway

A portable SIMD abstraction library for Go, inspired by Google's Highway C++ library.

## Go Version

Use `go` for all Go commands in this repository:

```bash
go build ./...
go test ./...
go run ./cmd/hwygen
```

## SIMD Acceleration

Enable hardware SIMD with the `GOEXPERIMENT=simd` environment variable:

```bash
# Build with SIMD
GOEXPERIMENT=simd go build ./...

# Test with SIMD
GOEXPERIMENT=simd go test ./...

# Force fallback path (for testing pure Go implementation)
HWY_NO_SIMD=1 GOEXPERIMENT=simd go test ./...

# Disable SME dispatch on ARM64 (falls back to NEON)
HWY_NO_SME=1 go test ./...

# Disable SVE dispatch on ARM64 Linux (falls back to NEON)
HWY_NO_SVE=1 go test ./...

# Run benchmarks
GOEXPERIMENT=simd go test -bench=. -benchmem ./hwy/contrib/algo/...
GOEXPERIMENT=simd go test -bench=. -benchmem ./hwy/contrib/math/...
```

## Project Structure

- `hwy/` - Core SIMD operations (Load, Store, Add, Mul, etc.)
- `hwy/contrib/algo/` - Algorithm transforms (ExpTransform, SinTransform, etc.)
- `hwy/contrib/math/` - Low-level math functions (Exp_AVX2_F32x8, etc.)
- `hwy/asm/` - Assembly implementations
- `hwy/c/` - C source files for GoAT transpilation
- `cmd/hwygen/` - Code generator for target-specific implementations
- `examples/` - Example usage (gelu, softmax)
- `specs/` - Specification files

## Code Generator (hwygen)

Generate optimized target-specific code:

```bash
go build -o bin/hwygen ./cmd/hwygen
./bin/hwygen -input mycode.go -target avx2 -output mycode_avx2.go
```

### hwygen Workflow for New Features

When implementing a new SIMD feature:

1. **Create the base file first** (`*_base.go`) with the `//go:generate` directive:
   ```go
   //go:generate go run ../../../cmd/hwygen -input transpose_base.go -output . -targets avx2,avx512,neon,fallback -dispatch transpose

   func BaseTranspose2D[T hwy.Floats](src []T, m, k int, dst []T) { ... }
   ```

2. **Run hwygen generation before creating any dispatch overrides**:
   ```bash
   cd hwy/contrib/matmul && go generate transpose_base.go
   ```
   This creates `*_arm64.gen.go`, `*_amd64.gen.go`, `*_fallback.gen.go`, etc.

3. **Generated variable names** follow the pattern `Transpose2DFloat32` (function name + type), not `TransposeFloat32`.

4. **Dispatch overrides are only needed** for functionality not yet in Go's simd package:
   - SME assembly (via GOAT C files)
   - NEON assembly for operations Go simd doesn't support
   - float16/bfloat16 types (not native to Go simd)

5. **Function signatures** should match: `func(src []T, m, k int, dst []T)` (src first, dst last).

### hwygen Directives

Directive comments placed within 5 lines above a `Base*` function:

- `//hwy:gen T={float32, float64}` — explicit type expansion (cross-product for multi-param)
- `//hwy:specializes <GroupName>` — joins a different `Base*` function's dispatch group (different body, same interface)
- `//hwy:targets neon,sme` — restricts a function to specific SIMD targets

Specializations are auto-discovered from sibling `*_base.go` files. See `specs/multi-dispatch.md` for details.

### Target Modes: `neon` vs `neon:asm` vs `neon:goat`

hwygen supports these generation modes, selected with a colon suffix on the target name:

| Suffix | Mode | What it generates |
|--------|------|-------------------|
| *(none)* | GoSimd | TWO variants: archsimd intrinsics under `arm64 && goexperiment.simd`, plus `hwy/asm`-backed Go under `arm64 && !goexperiment.simd` |
| `:asm` | Assembly | C source → GoAT → Go assembly + wrappers |
| `:goat` | GoSimd (legacy) | `hwy/asm`-backed Go only, plain `arm64` tag (for ops with no archsimd arm64 mapping yet, e.g. Compress) |
| `:c` | C only | C source for inspection (not compiled) |

**Use plain `neon`** (GoSimd mode) when the ops map to Go 1.27's native arm64 archsimd support (arithmetic, comparisons, conversions, FMA — see `neonArchsimdOps` in cmd/hwygen/targets.go). Both variants define identical symbols under mutually exclusive build tags, so dispatch files need no changes. The archsimd variant inlines as compiler intrinsics and is ~20x faster than the per-op-call asm path on math kernels. Ops without archsimd arm64 equivalents route to `hwy.X_NEON_SIMD_*` wrappers in `hwy/ops_neon_simd.go` — a missing wrapper is a compile error flagging the package for `neon:goat`.

**Use `neon:asm`** when you need bulk assembly — the entire function is compiled from C to Go assembly via GoAT, eliminating per-vector call overhead. This is best for:
- Compute-heavy kernels (matmul, cross-entropy loss, fused quantized ops)
- Functions where per-vector function call overhead dominates
- Operations that benefit from compiler auto-vectorization at `-O3`

Examples from the codebase:
```go
// GoSimd mode — calls asm package methods per-vector
//go:generate go run ../../../cmd/hwygen -input dense_base.go -output . -targets avx2,avx512,neon,fallback

// Assembly mode — compiles entire function to NEON assembly via GoAT
//go:generate go run ../../../cmd/hwygen -input matmul_fused_int8.go -dispatch fusedint8matmul -output . -targets avx2,avx512,neon:asm,fallback
```

The `:asm` suffix generates:
- C source files (in `asm/c/` or kept with `-keepc`)
- Go assembly (`.s`) via GoAT transpilation
- `//go:noescape` wrapper functions for slice-to-pointer conversion
- Dispatch override files (`z_c_slices_*_neon_arm64.gen.go`)

SVE targets (`sve_darwin`, `sve_linux`) are always assembly-only — they have no GoSimd mode since Go's simd package does not support SVE.

### Portable target (`portable`)

The `portable` target emits Go 1.27's size-agnostic `simd` package
(`simd.Float32s`, `v.Len()`, slice-based loads/stores) under a plain
`goexperiment.simd` build tag, giving wasm and other non-amd64/arm64
architectures a vectorized tier. Code is generated for the 128-bit minimum
width; the generated dispatcher (`*_portable.gen.go`, tag
`goexperiment.simd && !amd64 && !arm64`) wires it only when
`simd.VectorBitSize() == 128` at runtime (true on wasm) and falls back to
scalar otherwise. Add it to element-wise packages as
`-targets avx2,avx512,neon:asm,portable,fallback`. The portable package has
no reductions, gathers, or shuffles — packages needing those stay off this
target for now.

## Supported Architectures

| Architecture | SIMD Width | Backend | Status |
|--------------|------------|---------|--------|
| AMD64 AVX2 | 256-bit | Go 1.26 `simd/archsimd` | Supported |
| AMD64 AVX-512 | 512-bit | Go 1.26 `simd/archsimd` | Supported |
| ARM64 NEON | 128-bit | Go 1.27 `simd/archsimd` (+`hwy/asm` fallback) | Supported |
| ARM64 SVE (Darwin) | 512-bit (fixed) | `hwy/asm` (GoAT assembly) | Supported |
| ARM64 SVE (Linux) | Scalable | `hwy/asm` (GoAT assembly) | Supported |
| Pure Go | Scalar | — | Supported (fallback) |

ARM64 NEON uses native `simd/archsimd` (Go 1.27+) under `GOEXPERIMENT=simd`, with the `hwy/asm` GoAT path for non-experiment builds. SVE/SME remain GoAT-only (Go simd has no scalable-vector support).

## GoAT Transpiler (C to Go Assembly)

See [GOAT.md](GOAT.md) for the C-to-Go assembly transpiler documentation.

Key limitations to be aware of:
- No `__builtin_*` functions - use polynomial approximations instead
- No `static inline` helper functions - inline code directly
- No `union` type punning
- Supported return types: `void`, `long`, `float`, `double`, `_Bool`
- Arguments must be `int64_t`, `long`, `float`, `double`, `_Bool`, or pointer

For ARM64 NEON code generation, see the GOAT.md section on SME/SVE support and macOS compatibility issues.

### SME C Function Attributes

For SME functions, use post-function attributes (not pre-function):
```c
// Correct:
void my_sme_func(float *a, float *b) __arm_streaming __arm_out("za") { ... }

// Wrong (won't work with GOAT):
__arm_locally_streaming
void my_sme_func(float *a, float *b) { ... }
```

## Testing

Always run tests with SIMD enabled to verify hardware paths:

```bash
GOEXPERIMENT=simd go test ./...
```

Tests automatically skip AVX-512 on unsupported hardware.
