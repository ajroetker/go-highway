# hwygen

`hwygen` generates target-specific implementations and dispatch glue from plain Go functions that use `hwy.*`.

The language is intentionally small:

- write normal top-level `Base...` or `base...` functions
- use ordinary Go control flow and `hwy.*` operations
- add directives only when inference is not enough

## Install

```bash
cd cmd/hwygen
go build -o ../../bin/hwygen
```

## Quick Start

```go
//go:generate go run ../../../cmd/hwygen -check -input $GOFILE -targets avx2,fallback
//go:generate go run ../../../cmd/hwygen -input $GOFILE -output . -targets avx2,fallback

package mypkg

import "github.com/ajroetker/go-highway/hwy"

func BaseAdd[T hwy.Floats](dst, a, b []T) {
	n := min(len(dst), min(len(a), len(b)))
	for i := 0; i < n; i += hwy.Lanes[T]() {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		hwy.Store(hwy.Add(va, vb), dst[i:])
	}
}
```

Then run:

```bash
go generate ./...
```

## Author Model

`hwygen` looks for top-level functions whose names start with `Base` or `base`.

- `BaseFoo` generates an exported dispatch entry `Foo`
- `baseFoo` generates an unexported dispatch entry `foo`
- the dispatch group name comes from stripping the `Base` or `base` prefix

Examples:

- `BaseMatMul` belongs to dispatch group `MatMul`
- `baseNormalize` belongs to dispatch group `Normalize`

If a function does not start with `Base` or `base`, it is not a dispatch entrypoint.

## Accepted Code Shapes

The generator is built around a small set of normal Go idioms. Prefer these shapes:

### Shared slice bounds

```go
n := min(len(dst), min(len(a), len(b)))
if n == 0 {
	return
}
```

or:

```go
size := min(len(src), len(dst))
```

### Canonical vector loop

```go
for i := 0; i < n; i += hwy.Lanes[T]() {
	v := hwy.Load(src[i:])
	hwy.Store(v, dst[i:])
}
```

Also accepted:

- `lanes := hwy.Lanes[T]()` and then `i += lanes`
- `v.NumLanes()` / `v.NumElements()` when the lane count comes from a vector value
- standard scalar tails after the vector loop

### Generic type parameters

```go
func BaseOp[T hwy.Floats](...)
func BaseOp[T hwy.Integers](...)
func BaseOp[T hwy.FloatsNative](...)
```

Non-generic functions are also supported.

## Directives

Keep directives as escape hatches, not the default.

### `//hwy:gen`

Restricts which concrete type combinations to generate.

```go
//hwy:gen T={float32, float64}
func BaseDot[T hwy.Floats](...) { ... }
```

### `//hwy:specializes`

Marks a function as a specialization for another dispatch group.

```go
func BaseMatMul[T hwy.Floats](...) { ... }

//hwy:specializes MatMul
func BaseMatMulHalf[T hwy.HalfFloats](...) { ... }
```

`//hwy:specializes` only applies to top-level `Base...` or `base...` functions.

### `//hwy:targets`

Restricts a function or specialization to specific targets.

```go
//hwy:targets neon,avx512
func BaseOpHalf[T hwy.Floats](...) { ... }
```

Mode-qualified targets are supported:

```go
//hwy:targets neon:asm,avx2:asm
```

Rules:

- bare targets like `neon` match any generation mode for that target
- `:asm` and `:c` restrict matching to that mode only
- valid target names are `avx2`, `avx512`, `fallback`, `neon`, `sve_darwin`, `sve_linux`

### `//hwy:elemtype`

Overrides the SIMD element type inferred from parameters.

Useful for packed or byte-oriented APIs whose logical SIMD type differs from the slice element type.

## Command-Line Usage

Validate only:

```bash
hwygen -check -input add_base.go -targets avx2,fallback
```

Generate Go SIMD output:

```bash
hwygen -input add_base.go -output . -targets avx2,fallback
```

Generate C output only:

```bash
hwygen -c -input add_base.go -output . -targets neon
```

Generate C plus GOAT-compiled Go assembly:

```bash
hwygen -asm -input add_base.go -output . -targets neon
```

Per-target mode suffixes override global `-c` / `-asm` flags:

```bash
hwygen -input add_base.go -output . -targets fallback,neon:asm,avx2
```

## `-check`

`hwygen -check` parses and validates input without writing generated files.

It is meant to catch:

- invalid target selectors like `neon:weird`
- misplaced `//hwy:specializes`
- unknown specialization groups
- ambiguous specializations
- target-mode mismatches in specialization selection
- files that do not define any eligible `Base...` or `base...` entrypoints

This is the fastest feedback loop for authors and fits well in `go:generate`.

## Specialization Example

```go
func BaseMatMul[T hwy.FloatsNative](a, b, c []T) {
	n := min(len(a), min(len(b), len(c)))
	for i := 0; i < n; i += hwy.Lanes[T]() {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		hwy.Store(hwy.Add(va, vb), c[i:])
	}
}

//hwy:gen T={hwy.Float16, hwy.BFloat16}
//hwy:specializes MatMul
//hwy:targets neon:asm
func BaseMatMulHalf[T hwy.HalfFloats](a, b, c []T) {
	n := min(len(a), min(len(b), len(c)))
	for i := 0; i < n; i += hwy.Lanes[T]() {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		hwy.Store(hwy.Mul(va, vb), c[i:])
	}
}
```

This means:

- `BaseMatMul` is the primary dispatch source for group `MatMul`
- `BaseMatMulHalf` overrides the `MatMul` group for half-precision combos
- that override only applies when the selected target is `neon:asm`
- generated implementation names still use the primary name `BaseMatMul`

## Guidelines

- Prefer naming conventions over directives.
- Prefer ordinary Go loop and bounds patterns over clever annotation.
- Use `-check` before generation when editing the source language.
- Keep specialized functions signature-compatible with their primary function.
