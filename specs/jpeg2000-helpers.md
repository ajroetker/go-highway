# Plan: SIMD Color Transforms and Wavelet Lifting

Add color space transforms to `go-highway/hwy/contrib/image/` and create a new `go-highway/hwy/contrib/wavelet/` package, then wire them into `go-jpeg2000`.

## Part 1: Color Transforms in `contrib/image`

### New file: `color_base.go`

```
//go:generate go run ../../../cmd/hwygen -input color_base.go -output . -targets avx2,avx512,neon,fallback -dispatch color
```

Four functions, following the exact pattern from `point_ops_base.go` (nil checks, `hwy.MaxLanes`, row loop, vector inner loop, buffer tail):

**`BaseForwardRCT[T hwy.SignedInts](r, g, b, outY, outCb, outCr *Image[T])`**
- `Y = (R + 2G + B) >> 2` — use `hwy.Add(vg, vg)` for 2G, then `hwy.Add` twice, then `hwy.ShiftRight(sum, 2)`
- `Cb = B - G` — `hwy.Sub(vb, vg)`
- `Cr = R - G` — `hwy.Sub(vr, vg)`

**`BaseInverseRCT[T hwy.SignedInts](y, cb, cr, outR, outG, outB *Image[T])`**
- `G = Y - ((Cb + Cr) >> 2)` — `hwy.Add` then `hwy.ShiftRight` then `hwy.Sub`
- `R = Cr + G` — `hwy.Add`
- `B = Cb + G` — `hwy.Add`

**`BaseForwardICT[T hwy.Floats](r, g, b, outY, outCb, outCr *Image[T])`**
- 3x3 matrix multiply using `hwy.FMA` chains (9 coefficients broadcast to vectors at init)
- `Y = FMA(rToY, vr, FMA(gToY, vg, Mul(bToY, vb)))`
- Similarly for Cb, Cr

**`BaseInverseICT[T hwy.Floats](y, cb, cr, outR, outG, outB *Image[T])`**
- `R = Y + 1.402*Cr` — `hwy.FMA(crToRVec, vcr, vy)`
- `G = Y - 0.344136*Cb - 0.714136*Cr` — two FMA ops chained
- `B = Y + 1.772*Cb` — `hwy.FMA(cbToBVec, vcb, vy)`

### New file: `color_test.go`

- Table-driven with widths: 1, 7, 8, 15, 16, 17, 100, 1024 (test aligned, unaligned, tail)
- Round-trip tests: ForwardRCT → InverseRCT = identity (exact for int32)
- Round-trip tests: ForwardICT → InverseICT ≈ identity (tolerance ~1e-5 for float32)
- Nil image safety tests
- Both float32 and float64 for ICT

### New file: `color_bench_test.go`

- Sizes: 64x64, 256x256, 1920x1080, 3840x2160
- Bench all four functions

### New file: `color_constants.go`

Exported untyped `const` values as single source of truth for ITU-T T.800 color transform coefficients, plus unexported typed `var` values derived from them for hwygen.

```go
// Exported public API — ICT forward coefficients (ITU-T T.800 Table G.2)
const (
    ICT_RtoY  =  0.299
    ICT_GtoY  =  0.587
    ICT_BtoY  =  0.114
    ICT_RtoCb = -0.16875
    ICT_GtoCb = -0.33126
    ICT_BtoCb =  0.5
    ICT_RtoCr =  0.5
    ICT_GtoCr = -0.41869
    ICT_BtoCr = -0.08131
)

// Exported public API — ICT inverse coefficients
const (
    ICT_CrtoR =  1.402
    ICT_CbtoG = -0.344136
    ICT_CrtoG = -0.714136
    ICT_CbtoB =  1.772
)

// Typed vars for hwygen (derived from consts)
// Float32
var (
    ictRtoY_f32  float32 = ICT_RtoY
    ictGtoY_f32  float32 = ICT_GtoY
    // ... etc
)
// + Float64, Float16, BFloat16 variants
```

Base functions reference via `hwy.Const[T](ictRtoY_f32)`. The code generator handles precision for float64 targets per the comment on `hwy.Const` (line 128 of `ops_base.go`).

RCT uses no constants file — the only hardcoded value is `2`, used inline as `hwy.Const[T](2.0)` for the `2*G` term.

### Update: `doc.go`

Add color transform section to package doc comment.

### Generated files (hwygen output):

- `color_base_{avx2,avx512,neon,fallback}.gen.go`
- `color_{amd64,arm64,other}.gen.go`

---

## Part 2: Wavelet Package `contrib/wavelet`

### New file: `doc.go`

Package documentation covering CDF 5/3 and 9/7 lifting wavelets, the `phase` parameter, and 2D multi-level decomposition.

### New file: `lifting_base.go`

```
//go:generate go run ../../../cmd/hwygen -input lifting_base.go -output . -targets avx2,avx512,neon,fallback -dispatch lifting
```

Low-level SIMD lifting step primitives operating on **raw slices** (not Image[T]):

**`BaseLiftSym53[T hwy.SignedInts](target []T, tLen int, neighbor []T, nLen int, phase int)`**
- Applies the 5/3 update step: `target[i] -= (neighbor[i+off1] + neighbor[i+off2] + 2) >> 2`
- SIMD inner loop: handle boundary element(s) scalar, then process bulk with shifted loads
  - Load `hwy.Load(neighbor[i:])` and `hwy.Load(neighbor[i-1:])` (or `[i+1:]` depending on step direction)
  - `hwy.Add` the two, `hwy.Add` with twoVec, `hwy.ShiftRight` by 2, `hwy.Sub` from target
- Separate code paths for `phase=0` and `phase=1` (different neighbor offsets)

**`BaseLiftPredict53[T hwy.SignedInts](target []T, tLen int, neighbor []T, nLen int, phase int)`**
- Applies the 5/3 predict step: `target[i] += (neighbor[i+off1] + neighbor[i+off2]) >> 1`
- Same SIMD pattern with shifted loads

**`BaseLiftStep97[T hwy.Floats](target []T, tLen int, neighbor []T, nLen int, coeff T, phase int)`**
- Generic 9/7 lifting step: `target[i] -= coeff * (neighbor[i+off1] + neighbor[i+off2])`
- Uses `hwy.FMA`: `target = FMA(negCoeffVec, hwy.Add(nPrev, nCurr), targetVec)`
- Same shifted-load pattern, boundary elements handled scalar

**`BaseScaleSlice[T hwy.Floats](data []T, n int, scale T)`**
- `data[i] *= scale` — straightforward `hwy.Mul` loop

**`BaseInterleave[T hwy.Lanes](dst []T, low []T, sn int, high []T, dn int, phase int)`**
- Phase 0: `dst[2i]=low[i], dst[2i+1]=high[i]`
- Phase 1: `dst[2i]=high[i], dst[2i+1]=low[i]`
- Can use `hwy.StoreInterleaved2` if available, otherwise scalar

**`BaseDeinterleave[T hwy.Lanes](src []T, low []T, sn int, high []T, dn int, phase int)`**
- Inverse of interleave — extract even/odd into separate arrays

### New file: `constants.go`

Exported untyped `const` values as the single source of truth for standard CDF 9/7 coefficients, plus unexported typed `var` values derived from them for hwygen code generation.

```go
// Exported public API — full precision, usable as any float type
const (
    Alpha97 = -1.586134342059924  // predict step 1
    Beta97  = -0.052980118572961  // update step 2
    Gamma97 =  0.882911075530934  // predict step 3
    Delta97 =  0.443506852043971  // update step 4
    K97     =  1.230174104914001  // normalization
    InvK97  =  1.0 / K97
)

// Typed vars for hwygen (derived from consts, not duplicated literals)
// Float32
var (
    liftAlpha97_f32 float32 = Alpha97
    liftBeta97_f32  float32 = Beta97
    liftGamma97_f32 float32 = Gamma97
    liftDelta97_f32 float32 = Delta97
    liftK97_f32     float32 = K97
    liftInvK97_f32  float32 = InvK97
)
// + Float64, Float16, BFloat16 variants (same pattern, narrowed from const)
```

Base lifting functions reference via `hwy.Const[T](liftAlpha97_f32)`.

5/3 lifting uses no float constants — it's pure integer arithmetic with `hwy.Const[T](2.0)` inline for the rounding bias.

### New file: `wavelet.go`

Public API wrappers (manually written, no hwygen). These compose the Base* lifting steps from `lifting_base.go` and use the coefficients from `constants.go`:

```go
// 1D transforms (in-place, operate on [low...|high...] layout)
func Synthesize53[T hwy.SignedInts](data []T, phase int)
func Analyze53[T hwy.SignedInts](data []T, phase int)
func Synthesize97[T hwy.Floats](data []T, phase int)
func Analyze97[T hwy.Floats](data []T, phase int)
```

Each composes the Base* lifting steps + interleave/deinterleave:
- Synthesize53: deinterleave → update step → predict step → interleave
- Synthesize97: deinterleave → scale (K, InvK from `constants.go`) → 4 lifting steps (Alpha97..Delta97) → interleave
- Analyze53/97: reverse order

**Standard K/InvK scaling** (NOT JPEG2000's 2/K). The codec handles the difference.

### New file: `wavelet2d.go`

2D transforms using `Image[T]` for aligned row access:

```go
func Synthesize2D_53(img *image.Image[int32], levels int, phaseFn func(level int) (phaseH, phaseV int))
func Analyze2D_53(img *image.Image[int32], levels int, phaseFn func(level int) (phaseH, phaseV int))
func Synthesize2D_97[T hwy.Floats](img *image.Image[T], levels int, phaseFn func(level int) (phaseH, phaseV int))
func Analyze2D_97[T hwy.Floats](img *image.Image[T], levels int, phaseFn func(level int) (phaseH, phaseV int))
```

- Horizontal pass: call 1D transform on `img.Row(y)[:levelWidth]`
- Vertical pass: extract column to temp buffer, transform, write back
- Process coarsest→finest (synthesis) or finest→coarsest (analysis)

### New files: `lifting_test.go`, `wavelet2d_test.go`, `bench_test.go`

- 1D round-trip: Analyze → Synthesize = identity (exact for 5/3, ~1e-4 tolerance for 9/7 float32)
- Test both phase=0 and phase=1
- Multiple sizes: 1, 2, 3, 4, 7, 8, 15, 16, 17, 32, 63, 64, 100
- 2D round-trip with multiple decomposition levels
- Benchmarks: 1D at sizes 64/256/1024/4096, 2D at 256x256/1080p/4K

### Generated files:

- `lifting_base_{avx2,avx512,neon,fallback}.gen.go`
- `lifting_{amd64,arm64,other}.gen.go`

---

## Part 3: Integration in go-jpeg2000

### go.mod change

Add `require github.com/ajroetker/go-highway` dependency. Update `go` directive from 1.25 to 1.26 (matching go-highway).

### Adapter helpers (new file: `simd.go`)

Convert between JPEG2000's `[][]int32`/`[][]float64` and `Image[T]`:

```go
func slicesToImage[T hwy.Lanes](data [][]T, width, height int) *image.Image[T]
func imageToSlices[T hwy.Lanes](img *image.Image[T]) [][]T
```

### Changes to `color.go`

- `applyRCT`: Convert int32 slices → Image[int32], call `image.InverseRCT`, convert back
- `applyICT`: Convert float64 slices → Image[float64], call `image.InverseICT`, convert back
- `forwardRCT`/`forwardICT`: Same pattern with forward transforms

### Changes to `dwt.go`

- `synthesize1D_53_cas`: Call `wavelet.Synthesize53(data, cas)`
- `synthesize1D_97_cas`: Pre-scale high-pass by 2x (for JPEG2000's non-standard 2/K), then call `wavelet.Synthesize97(data, cas)`
- 2D functions: Convert `[][]T` to `Image[T]`, call `wavelet.Synthesize2D_*`, convert back

### Changes to `dwt_encode.go`

- `analyze1D_53`/`analyze1D_97`: Similar adapter pattern
- 2D analyze functions: delegate to wavelet package

### Validation

Run go-jpeg2000's existing conformance test suite to verify bit-exact (5/3) and tolerance (9/7) matching against reference images.

---

## Implementation Order

1. **Color transforms** in `contrib/image` (simplest, validates hwygen workflow with integer types)
2. **1D wavelet lifting** in `contrib/wavelet` (core SIMD work)
3. **2D wavelet orchestration** (composition of 1D primitives)
4. **go-jpeg2000 integration** (adapters + wiring)
5. **Benchmarks and validation** (conformance tests)

## Key files to modify

| Repository | File | Change |
|---|---|---|
| go-highway | `hwy/contrib/image/color_base.go` | New — 4 color transform functions |
| go-highway | `hwy/contrib/image/color_constants.go` | New — typed ICT coefficients (_f16/_bf16/_f32/_f64) |
| go-highway | `hwy/contrib/image/color_test.go` | New — tests |
| go-highway | `hwy/contrib/image/color_bench_test.go` | New — benchmarks |
| go-highway | `hwy/contrib/image/doc.go` | Update — add color docs |
| go-highway | `hwy/contrib/wavelet/doc.go` | New — package docs |
| go-highway | `hwy/contrib/wavelet/constants.go` | New — typed 9/7 lifting coefficients (_f16/_bf16/_f32/_f64) |
| go-highway | `hwy/contrib/wavelet/lifting_base.go` | New — SIMD lifting steps |
| go-highway | `hwy/contrib/wavelet/wavelet.go` | New — 1D public API |
| go-highway | `hwy/contrib/wavelet/wavelet2d.go` | New — 2D transforms |
| go-highway | `hwy/contrib/wavelet/lifting_test.go` | New — 1D tests |
| go-highway | `hwy/contrib/wavelet/wavelet2d_test.go` | New — 2D tests |
| go-highway | `hwy/contrib/wavelet/bench_test.go` | New — benchmarks |
| go-jpeg2000 | `go.mod` | Update — add go-highway dep, go 1.26 |
| go-jpeg2000 | `simd.go` | New — slice↔Image adapters |
| go-jpeg2000 | `color.go` | Update — delegate to highway |
| go-jpeg2000 | `dwt.go` | Update — delegate to highway |
| go-jpeg2000 | `dwt_encode.go` | Update — delegate to highway |
