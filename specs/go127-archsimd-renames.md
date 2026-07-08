# Go 1.27 `simd/archsimd` API Revision — Rename Table

Source: diff of `go doc -all simd/archsimd` between go1.26.4 and go1.27rc2
(GOEXPERIMENT=simd, GOARCH=amd64). This table drives the Phase 1 migration of
hwygen OpMaps and handwritten call sites.

## Renames the repo uses

| Go 1.26 | Go 1.27 | Notes |
|---------|---------|-------|
| `archsimd.Load<T>x<N>Slice(s []T)` | `archsimd.Load<T>x<N>(s []T)` | Slice form is now the primary name |
| `archsimd.Load<T>x<N>(p *[N]T)` | `archsimd.Load<T>x<N>Array(p *[N]T)` | Pointer-to-array form renamed |
| `archsimd.Load<T>x<N>SlicePart(s []T)` | `archsimd.Load<T>x<N>Part(s []T)` | |
| `v.StoreSlice(s []T)` | `v.Store(s []T)` | Slice form is now the primary name |
| `v.Store(p *[N]T)` | `v.StoreArray(p *[N]T)` | Pointer-to-array form renamed |
| `v.StoreSlicePart(s []T)` | `v.StorePart(s []T) int` | Now returns count stored |
| `v.StoreMasked(p *[N]T, m)` | `v.StoreArrayMasked(p *[N]T, m)` | |
| `v.RoundToEven()` | `v.Round()` | |
| `v.RoundToEvenScaled(prec)` | `v.RoundScaled(prec)` | |
| `v.RoundToEvenScaledResidue(prec)` | `v.RoundScaledResidue(prec)` | |

## Renames the repo does not currently use (for reference)

| Go 1.26 | Go 1.27 |
|---------|---------|
| `TruncateToInt8/16/32`, `TruncateToUint8/16/32` | `TruncToInt8/16/32`, `TruncToUint8/16/32` |
| `MulEvenWiden` | `MulWidenEven` |
| `AddSub` | `AddOddSubEven` |
| `MulAddSub` / `MulSubAdd` | `MulAddEvenSubOdd` / `MulAddOddSubEven` |
| `AddPairs` / `SubPairs` (+`Grouped`) | `ConcatAddPairs` / `ConcatSubPairs` (+`Grouped`) |
| `SelectFromPair` / `Select128FromPair` / `SelectFromPairGrouped` | `ConcatPermuteScalars` / `ConcatPermute128Scalars` / `ConcatPermuteScalarsGrouped` |
| `Shift{,All}{Left,Right}Concat` | `Shift{,All}{Left,Right}ConcatMod{16,32,64}` |
| `CopySign` | `MulSign` |
| `SumAbsDiff` | `SumOf8AbsDiff` |
| `Broadcast1To<N>` (method) | removed (use package-level `Broadcast<T>x<N>`) |
| `CarrylessMultiplyGrouped` | see `CarrylessMultiply*` variants |

## Unchanged (verified)

- `archsimd.X86.AVX()/AVX2()/AVX512()` feature detection — identical.
- `Broadcast<T>x<N>` package functions — identical.
- Comparison methods (`Equal`, `Less`, `Greater`, ...) — identical.
- Mask types — only gained `String()`.
- `Merge(y, mask)` — kept (new `IfElse(mask, y)` added as alternative).
- Arithmetic (`Add/Sub/Mul/Div/Min/Max/Sqrt/MulAdd/...`) — identical.

## New in 1.27 (opportunities, not required for migration)

- `Abs()`, `Neg()` on float vectors (previously emulated).
- `IfElse(mask, y)`, `ToBits()`, `ReshapeToUint*`, `RotateAllLeft/Right`,
  `Round()` (round-to-even), `String()` on all vectors.
- arm64 NEON 128-bit types + ops (see Phase 2; dump captured from
  `GOARCH=arm64 go doc -all simd/archsimd`).
- Portable `simd` package (size-agnostic `Float32s`/`Int8s`/... with
  `Len()`, `LoadPart`/`StorePart`).

## Migration mechanics

Handwritten files: mechanical rewrite —

```
gofmt -r 'archsimd.LoadFloat32x8Slice(a) -> archsimd.LoadFloat32x8(a)'   # per type/width
gofmt -r 'v.StoreSlice(s) -> v.Store(s)'                                  # after fixing ptr-array Store calls
```

Pointer-to-array call sites (e.g. `hwy/bitops_avx2.go` `v.Store(&data)`,
`archsimd.LoadInt32x8(&data)`) must be rewritten to `StoreArray`/`Load*Array`
FIRST, before the Slice-suffix drop, to avoid conflating the two forms.

Generated files: fix `cmd/hwygen/targets.go` OpMaps (`avxBaseOps`,
`AVX512Target` overrides) + any hardcoded names in `transformer_ops.go`,
then `go generate ./...`.

## Phase 3 gate decision: int8/quantized NEON kernels stay on GoAT (`neon:asm`)

Measured on go1.27rc2, Apple M4 Max (2026-07): the GoAT C kernels show no
regression under the 1.27 toolchain (geomean ~7% faster than the 1.26
baseline; `Int8x8MatMul/64x256x512` 586µs → 498µs).

An archsimd rewrite was evaluated and rejected for now because rc2's arm64
archsimd exposes:
- no SDOT/UDOT or i8mm bindings,
- only low-half widening (`ExtendLo8To*`, `MulWidenLo`) — no high-half
  variants without an extra `HiToLo()` shuffle,
- and hwy's base-op vocabulary has no widening ops, so the base kernels
  cannot express a competitive dequant+widen+accumulate pipeline portably.

Revisit when archsimd grows dot-product bindings (track golang/go#73787).
The NEONSimd TypeMap already carries int8/int16/uint8/uint16 entries so
base code using those element types generates as soon as the ops exist.

## Selector semantics (final)

Plain `neon` = the native archsimd target only (`arm64 && goexperiment.simd`);
non-experiment arm64 builds get scalar fallback dispatch, matching amd64.
`neon:goat` = legacy hwy/asm-backed GoSimd on plain `arm64` (sort uses this).
`neon:asm` = GoAT kernels, tag-independent. The earlier dual-variant
expansion of `neon` was dropped: one selector, one implementation.
