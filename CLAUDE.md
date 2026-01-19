# Go Highway

A portable SIMD abstraction library for Go, inspired by Google's Highway C++ library.

## Go Version

Use `go1.26rc2` for all Go commands in this repository:

```bash
go1.26rc2 build ./...
go1.26rc2 test ./...
go1.26rc2 run ./cmd/hwygen
```

## SIMD Acceleration

Enable hardware SIMD with the `GOEXPERIMENT=simd` environment variable:

```bash
# Build with SIMD
GOEXPERIMENT=simd go1.26rc2 build ./...

# Test with SIMD
GOEXPERIMENT=simd go1.26rc2 test ./...

# Force fallback path (for testing pure Go implementation)
HWY_NO_SIMD=1 GOEXPERIMENT=simd go1.26rc2 test ./...

# Run benchmarks
GOEXPERIMENT=simd go1.26rc2 test -bench=. -benchmem ./hwy/contrib/algo/...
GOEXPERIMENT=simd go1.26rc2 test -bench=. -benchmem ./hwy/contrib/math/...
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
go1.26rc2 build -o bin/hwygen ./cmd/hwygen
./bin/hwygen -input mycode.go -target avx2 -output mycode_avx2.go
```

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

## Testing

Always run tests with SIMD enabled to verify hardware paths:

```bash
GOEXPERIMENT=simd go1.26rc2 test ./...
```

Tests automatically skip AVX-512 on unsupported hardware.
