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

## Supported Architectures

| Architecture | SIMD Width | Status |
|--------------|------------|--------|
| AMD64 AVX2 | 256-bit | Supported |
| AMD64 AVX-512 | 512-bit | Supported |
| ARM64 NEON | 128-bit | Supported |
| Pure Go | Scalar | Supported (fallback) |

## GoAT Transpiler (C to Go Assembly)

See [GOAT.md](GOAT.md) for the C-to-Go assembly transpiler documentation.

Key limitations to be aware of:
- No `else` clauses - rewrite using multiple `if` statements
- No `__builtin_*` functions - use polynomial approximations instead
- No `static inline` helper functions - inline code directly
- No `union` type punning
- C functions must have `void` return types
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
