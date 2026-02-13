# GoAT

Go assembly transpiler for C programming languages. It helps utilize optimization from C compiler in Go projects. For example, generate SIMD vectorized functions for Go (refer to [How to Use AVX512 in Golang](https://gorse.io/posts/avx512-in-golang.html)).

In go-highway, GoAT is the assembly backend for all ARM64 targets (NEON, SVE, SME). Go 1.26's `simd/archsimd` package supports AVX2 and AVX-512 but does not yet support ARM NEON or SVE, so ARM64 code generation relies on GoAT to transpile C into Go assembly. The hwygen code generator integrates GoAT via the `neon:asm`, `sve_darwin`, and `sve_linux` target modes.

## Installation

```bash
go install github.com/ajroetker/goat@latest
```

## Example

Suppose you have a C function that adds two arrays of floats in `src/add.c`:

```c
void add(float *a, float *b, float *result, long n) {
    for (long i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}
```

You can use GoAT to transpile this C function to Go assembly code:

```bash
goat src/add.c -o ./asm -O3
```

Finally, the add function can be used by:

```go
func Add(a, b, c []float32) {
	if len(a) != len(b) || len(a) != len(c) {
		panic("floats: slice lengths do not match")
	}
	add(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), int64(len(a)))
}
```

## Command Line Options

```
Usage: goat source [-o output_directory]

Flags:
  -o, --output string          Output directory for generated files (default: current directory)
  -O, --optimize-level int     Optimization level for clang (0-3, default: 0)
  -m, --machine-option strings Machine options for clang (e.g., -m avx2, -m sse4.2)
  -e, --extra-option strings   Extra options for clang (passed directly)
  -t, --target string          Target architecture: amd64, arm64, loong64, riscv64 (default: host)
      --target-os string       Target operating system: darwin, linux, windows (default: host)
  -I, --include-path strings   Additional include paths for C parser (for cross-compilation)
  -v, --verbose                Enable verbose output
```

### Examples

```bash
# Basic transpilation with optimization
goat src/add.c -o ./asm -O3

# ARM64 with NEON
goat src/vector.c -o ./asm -O3 -t arm64

# ARM64 with SME (Scalable Matrix Extension)
goat src/matmul.c -o ./asm -O3 -t arm64 -m arch=armv9-a+sme

# x86-64 with AVX2
goat src/simd.c -o ./asm -O3 -t amd64 -m avx2

# x86-64 with AVX-512
goat src/simd.c -o ./asm -O3 -t amd64 -m avx512f -m avx512vl

# Cross-compilation with custom include paths
goat src/code.c -o ./asm -O3 -t arm64 --target-os linux -I /path/to/arm64/includes
```

# Supported Architectures

| Architecture | Target Flag | SIMD Support | Notes |
|--------------|-------------|--------------|-------|
| x86-64 | `-t amd64` | SSE, AVX, AVX2, AVX-512 | Use `-m avx2`, `-m avx512f`, etc. |
| ARM64 | `-t arm64` | NEON, SVE, SME | Use `-m arch=armv9-a+sme` for SME |
| LoongArch64 | `-t loong64` | LSX, LASX | China's LoongArch architecture |
| RISC-V 64 | `-t riscv64` | V extension | Experimental |

# Limitations

- No call statements except for inline functions.
- Arguments must be int64_t, long, float, double, _Bool or pointer.
- Potentially BUGGY code generation.
- C functions must have void return types
- `uint64_t` from the header <stdint.h> is not supported
- C source file names should not begin with `_`.
- **`else` clauses in conditionals are not supported** - The parser fails with "expected `}`" errors when encountering `else`. Rewrite code to avoid `else` by using multiple `if` statements or initializing values before conditionally updating them.
- **Single-line `if` statements with braces are not supported** - `if (x) { y; }` on one line causes parser errors. Use multi-line format instead:
  ```c
  // BAD: causes "expected }" error
  if (lane0) { *result = 0; }

  // GOOD: works correctly
  if (lane0) {
      *result = 0;
  }
  ```
- **No `__builtin_*` functions** - Calls to `__builtin_expf`, `__builtin_sqrtf`, etc. generate `bl` (branch-link) instructions to C library functions (`expf`, `sqrtf`, etc.) which don't exist in Go assembly context. Use polynomial approximations or other manual implementations instead.
- **No `static inline` helper functions** - The parser fails on `static inline` function definitions. Inline all helper code directly where needed.
- **No `union` type punning** - Union types for float/int bit reinterpretation cause parsing errors. Use alternative approaches like loop-based 2^k computation.
- **No array initializers with variables** - `int arr[4] = {m0, m1, m2, m3}` causes parsing errors. Use explicit stores instead.
- **Scalar loops may be optimized to `memset`** - Simple loops like `for (i = 0; i < n; i++) result[i] = 0;` may be optimized by the compiler into `memset` calls, which don't exist in Go assembly. Break the pattern by using NEON stores for the scalar remainder or add complexity to prevent the optimization.
- **Double constants cause `.rodata` sections → parser panic** - When using `double` constants, clang may place them in `.section .rodata.cst8` and load with `adrp` + `ldr d_reg, [x_reg, :lo12:.LCPI...]`. GOAT's ARM64 parser panics with "index out of range [-1]" when encountering rodata sections. This happens in two scenarios:

  1. **Scalar loops with double literals** - Constants like `0.6931471805599453` that can't be represented with `fmov` immediate
  2. **Multi-loop code with shared constants** - When the same constant is used across multiple loops (e.g., main SIMD loop + scalar remainder), clang may "hoist" the constant to rodata for efficiency, even if constructed inline

  **Workarounds:**
  - Use `volatile` keyword on f64 vector constants to prevent clang from hoisting to rodata:
    ```c
    // BAD: clang may hoist to rodata when used in multiple loops
    float64x2_t invSqrt2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE6A09E667F3BCDLL));

    // GOOD: volatile prevents rodata hoisting
    volatile float64x2_t invSqrt2 = vreinterpretq_f64_s64(vdupq_n_s64(0x3FE6A09E667F3BCDLL));
    ```
  - Use hex bit patterns with `vreinterpretq_f64_s64(vdupq_n_s64(0x...LL))` to construct double constants from integer immediates
  - Use SIMD-only code without scalar remainder loops for F64 (process only multiples of 2 elements)
  - Float32 constants work fine because clang uses `mov w_reg, #imm` + `fmov s_reg, w_reg` which GOAT can parse

# ARM SME/SVE Support

GoAT supports ARM SVE (Scalable Vector Extension) and SME (Scalable Matrix Extension) code generation when using `-m arch=armv9-a+sme`. GoAT automatically handles streaming mode requirements and macOS compatibility.

## Automatic Streaming Mode Injection

GoAT automatically detects SVE/SME instructions in compiler output and injects the necessary `smstart`/`smstop` instructions. You no longer need to manually add streaming mode entry/exit.

**What GoAT does automatically:**

1. **Detects SVE/SME instructions** - Recognizes Z registers, predicate registers, SVE loads/stores, SME outer products, and ZA tile operations
2. **Injects `smstart sm`** - Placed before the first SVE instruction, with setup instructions (like `ptrue`, `cntw`) moved inside the streaming section
3. **Injects `smstop sm`** - Placed before ALL `ret` instructions to ensure safe function exit
4. **Handles complex control flow** - For functions with branches, uses a conservative approach that adds streaming mode at all necessary points

Example generated code structure:
```assembly
TEXT ·my_function(SB), $0-48
    MOVD a+0(FP), R0
    MOVD b+8(FP), R1

    WORD $0xd503477f  // smstart sm (auto-injected)

    // ptrue, cntw, etc. moved inside streaming section
    // ... SVE/SME code ...

    WORD $0xd503467f  // smstop sm (auto-injected)
    RET
```

## macOS Compatibility (Automatic)

GoAT automatically handles macOS-specific restrictions on Apple Silicon (M4+):

1. **`movi d0, #0` replacement** - Automatically replaced with `fmov s0, wzr` which works in streaming mode
2. **MOVA encoding fix** - ZA→Z tile reads have bit 17 corrected (0xc080 → 0xc082) for M4 compatibility

These transformations happen automatically during transpilation.

## Tips for SME Code

1. **Keep C code simple** - Complex loops with nested conditions generate harder-to-maintain assembly
2. **Let compiler auto-vectorize** - Don't use SVE intrinsics (`#include <arm_sve.h>`) unless necessary; simple loops often vectorize well
3. **Use `-O3` for best vectorization** - Higher optimization levels produce better SVE code
4. **Test on macOS** - macOS has stricter restrictions; code that works there will work on Linux

## SVE Type Support

GoAT supports SVE scalable vector types in function parameters:

```c
#include <arm_sve.h>

void process_vectors(svfloat32_t *input, svfloat32_t *output, long n);
```

Since SVE vector sizes are determined at runtime, these are passed as `unsafe.Pointer` in Go.

## Performance Notes

For best SME performance:

- **Batch operations** - Design functions to process multiple rows/elements to amortize streaming mode entry/exit overhead
- **Avoid small functions** - Streaming mode transitions have overhead; larger compute-to-overhead ratio is better
- **Use NEON for small data** - For small arrays, NEON may outperform SME due to lower overhead

**Performance Comparison** (4096x4096 matrix-vector multiplication on M4):
- Pure Go: 3.8ms
- NEON (GoAT): 2.9ms (1.3x faster than Go)
- SME (GoAT with auto-streaming): Competitive with NEON, no manual work needed
