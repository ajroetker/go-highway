# Add Integer Dot Product API to hwy (vdotq_s32 / VPDPBSSD)

## Context

GGUF K-quant dequantization is complete. The next step toward fused dequant+matmul is integer dot product support. llama.cpp never dequantizes to float — it quantizes activations to Q8_K (int8) and uses `vdotq_s32` (16 int8×int8 MACs per instruction, 4× throughput vs float).

Go's `simd/archsimd` will add integer dot products in Go 1.27:
```
func (Int8x16) DotProductQuadrupleSigned(Int8x16) Int32x4      // VPDPBSSD
func (Int8x16) DotProductQuadrupleUnsigned(Uint8x16) Int32x4   // VPDPBUSD
func (Uint8x16) DotProductQuadrupleSigned(Int8x16) Int32x4     // VPDPBUSD/VPDPBSUD
func (Uint8x16) DotProductQuadrupleUnsigned(Uint8x16) Uint32x4 // VPDPBUUD
```

The addend is NOT part of Go's API — the compiler fuses `x.DotProduct(y).Add(z)`.

## API Design — `hwy/ops_int8dot.go`

Three levels: explicit SS/UU primitives + generic `DotProduct[T]` convenience wrapper.

```go
// DotProductSS: signed×signed int8 dot product → int32 (groups of 4).
// Maps to vdotq_s32 (NEON) / VPDPBSSD (AVX-VNNI Go 1.27).
func DotProductSS(a, b Vec[int8]) Vec[int32]

// DotProductUU: unsigned×unsigned uint8 dot product → uint32.
// Maps to vdotq_u32 (NEON) / VPDPBUUD (AVX-VNNI Go 1.27).
func DotProductUU(a, b Vec[uint8]) Vec[uint32]

// DotProduct: generic dot product, dispatches to SS or UU based on T.
// Returns int32 for both (pragmatically fine — max per-group value is
// 4×255×255 = 260100, well within int32 range even after accumulation).
// This is the primary API for use in generic base functions.
func DotProduct[T int8 | uint8](a, b Vec[T]) Vec[int32]
```

US/SU variants need FEAT_I8MM (ARMv8.6-A, different from DOTPROD). Defer to a follow-up with `+i8mm` flag and separate feature gate.

## Generic Base Function Design

The base function is **generic over `int8 | uint8`**, matching highway's pattern of writing one function that generates type-specialized SIMD:

```go
//go:generate go run ../../../cmd/hwygen -input dot_int_base.go -output . -targets neon:asm,fallback -dispatch dotint

//hwy:gen T={int8, uint8}
func BaseDotInt[T int8 | uint8](a, b []T) int32 {
    n := min(len(a), len(b))
    lanes := hwy.NumLanes[T]()
    acc := hwy.Zero[int32]()

    var i int
    for i = 0; i+lanes <= n; i += lanes {
        va := hwy.Load(a[i:])
        vb := hwy.Load(b[i:])
        acc = hwy.Add(acc, hwy.DotProduct(va, vb))
    }

    result := hwy.ReduceSum(acc)
    for ; i < n; i++ {
        result += int32(a[i]) * int32(b[i])
    }
    return result
}
```

### Profile selection per type combo

hwygen expands `//hwy:gen T={int8, uint8}` into two combos. For each, `comboPrimaryType` returns the Go type, and `GetCProfile(target, elemType)` selects the profile:

| Combo | elemType | NEON profile | Result |
|-------|----------|-------------|--------|
| T=int8 | `"int8"` | New int8 dotprod profile (DotAccFn=vdotq_s32) | NEON asm generated ✓ |
| T=uint8 | `"uint8"` | Existing uint8 profile (no DotAccFn) | **Skipped** — translator returns "unsupported op" error, generator skips this combo |

**Why uint8 NEON is skipped**: The existing `("NEON", "uint8")` profile lacks DotAccFn and uses `-march=armv8-a` (no +dotprod). Replacing it would affect all uint8 NEON code (gguf dequantize). Creating a separate `"uint8dp"` profile would require hwygen changes to map Go types to different profile names per function.

**What works today**: int8 gets full NEON asm. Both int8 and uint8 get scalar fallback. The generic `DotInt[T]` dispatcher works for both types.

**Path to uint8 NEON**: Add per-combo profile overrides to hwygen (e.g., `//hwy:profilemap uint8=uint8dp`), or write a non-generic `BaseDotIntUint8` specialization. Either can be done as a follow-up.

## Files to Modify/Create

### 1. `hwy/ops_int8dot.go` — Scalar primitives (NEW)

Scalar implementations used by hwygen-generated fallback code and the generic wrapper.

```go
package hwy

// DotProductSS computes widening 4-group dot product for signed int8.
func DotProductSS(a, b Vec[int8]) Vec[int32] {
    n := len(a.data) / 4
    result := make([]int32, n)
    for i := range n {
        base := i * 4
        result[i] = int32(a.data[base])*int32(b.data[base]) +
            int32(a.data[base+1])*int32(b.data[base+1]) +
            int32(a.data[base+2])*int32(b.data[base+2]) +
            int32(a.data[base+3])*int32(b.data[base+3])
    }
    return Vec[int32]{data: result}
}

// DotProductUU computes widening 4-group dot product for unsigned uint8.
func DotProductUU(a, b Vec[uint8]) Vec[uint32] {
    n := len(a.data) / 4
    result := make([]uint32, n)
    for i := range n {
        base := i * 4
        result[i] = uint32(a.data[base])*uint32(b.data[base]) +
            uint32(a.data[base+1])*uint32(b.data[base+1]) +
            uint32(a.data[base+2])*uint32(b.data[base+2]) +
            uint32(a.data[base+3])*uint32(b.data[base+3])
    }
    return Vec[uint32]{data: result}
}

// DotProduct is the generic wrapper used in base functions.
// Returns int32 for both signed and unsigned inputs.
func DotProduct[T int8 | uint8](a, b Vec[T]) Vec[int32] {
    switch any(a).(type) {
    case Vec[int8]:
        ss := DotProductSS(any(a).(Vec[int8]), any(b).(Vec[int8]))
        return ss
    case Vec[uint8]:
        uu := DotProductUU(any(a).(Vec[uint8]), any(b).(Vec[uint8]))
        // Convert uint32 results to int32 (safe for dot product magnitudes)
        result := make([]int32, len(uu.data))
        for i, v := range uu.data {
            result[i] = int32(v)
        }
        return Vec[int32]{data: result}
    }
    panic("unreachable")
}
```

### 2. Runtime detection files (NEW)

**`hwy/dotprod_detect_darwin.go`** (build: `darwin && arm64`):
```go
var hasDotProdDarwin = detectDotProd()
func detectDotProd() bool {
    val, err := syscall.Sysctl("hw.optional.arm.FEAT_DotProd")
    if err != nil { return false }
    return len(val) > 0 && val[0] == 1
}
```

**`hwy/dotprod_detect_other.go`** (build: `!darwin || !arm64`):
```go
var hasDotProdDarwin = false
```

### 3. `hwy/dispatch_arm64.go` — HasARMDotProd()

Add to existing detection (follows HasARMFP16/HasARMBF16 pattern):
```go
var hasARMDotProd bool

// In detectARMFP16BF16Features (or new function):
hasARMDotProd = cpu.ARM64.HasASIMDDP
if !hasARMDotProd && hasDotProdDarwin {
    hasARMDotProd = true
}

func HasARMDotProd() bool { return hasARMDotProd }
```

Stubs in `dispatch_amd64.go`, `dispatch_other.go`:
```go
func HasARMDotProd() bool { return false }
```

### 4. `cmd/hwygen/c_profiles.go` — int8 dotprod profile (NEW)

New profile centered on **int8** with DotAcc mapping to int32:

```go
func neonInt8DotProdProfile() *CIntrinsicProfile {
    return &CIntrinsicProfile{
        ElemType:   "int8",
        TargetName: "NEON",
        Include:    "#include <arm_neon.h>",
        CType:      "signed char",
        VecTypes:   map[string]string{"q": "int8x16_t"},
        Tiers:      []CLoopTier{
            {Name: "q", Lanes: 16, Unroll: 1},
            {Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
        },
        LoadFn:     map[string]string{"q": "vld1q_s8"},
        StoreFn:    map[string]string{"q": "vst1q_s8"},
        DupFn:      map[string]string{"q": "vdupq_n_s8"},
        AddFn:      map[string]string{"q": "vaddq_s8"},
        // Dot product: int8×int8 → int32
        DotAccFn:   map[string]string{"q": "vdotq_s32"},
        DotAccType: map[string]string{"q": "int32x4_t"},
        GoatTarget:     "arm64",
        GoatExtraFlags: []string{"-march=armv8.2-a+dotprod+simd+fp"},
    }
}
```

Register in `init()`. No uint8 dotprod profile needed yet (see "Profile selection per type combo" above).

### 5. `cmd/hwygen/c_ast_translator.go` — Mixed-type support for DotProduct

The translator currently uses a single profile for all operations. We need it to use the correct profile for operations on DotAccType variables (int32 operations when the primary profile is int8).

**Changes:**

**a) Add accumulator profile tracking:**
```go
type CASTTranslator struct {
    // ... existing fields ...
    accProfile  *CIntrinsicProfile  // Profile for DotAccType operations (e.g., int32)
    accVarNames map[string]bool     // Set of variable names that hold DotAccType values
}
```

Initialize `accProfile` from DotAccType: if `profile.DotAccType["q"] == "int32x4_t"`, load `GetCProfile("NEON", "int32")`.

**b) New operation: `DotProduct` (generic) / `DotProductSS` / `DotProductUU`:**

The translator handles all three names. For the generic `DotProduct`, it uses the profile's DotAccFn (which encodes whether it's signed or unsigned):

```go
case "DotProduct", "DotProductSS", "DotProductUU":
    if t.profile.DotAccFn == nil || t.profile.DotAccFn[t.tier] == "" {
        return "", fmt.Errorf("unsupported: DotProduct requires DotAccFn in profile %s:%s",
            t.profile.TargetName, t.profile.ElemType)
    }
    return t.emitHwyDotProduct(args)
```

Standalone (no accumulator) emits with zero accumulator:
```go
func (t *CASTTranslator) emitHwyDotProduct(args []ast.Expr) string {
    dotFn := t.profile.DotAccFn[t.tier]        // "vdotq_s32" or "vdotq_u32"
    accType := t.profile.DotAccType[t.tier]     // "int32x4_t" or "uint32x4_t"
    zeroAcc := t.accProfile.DupFn[t.tier] + "(0)"  // "vdupq_n_s32(0)"
    a := t.translateExpr(args[0])
    b := t.translateExpr(args[1])
    return fmt.Sprintf("%s(%s, %s, %s)", dotFn, zeroAcc, a, b)
}
```

**b2) Add+DotProduct fusion:**

When `hwy.Add(acc, hwy.DotProduct(a, b))` appears, fuse to a single 3-operand instruction `vdotq_s32(acc, a, b)` instead of the two-instruction sequence (dot to temp + add).

In the `"Add"` case, before falling through to the normal binary op:
```go
case "Add":
    // Fuse Add(acc, DotProduct(a,b)) → vdotq_s32(acc, a, b)
    if fused := t.tryFuseDotAccumulate(args); fused != "" {
        return fused
    }
    // ... existing Add handling with accProfile-aware dispatch
```

```go
func (t *CASTTranslator) tryFuseDotAccumulate(args []ast.Expr) string {
    if len(args) != 2 { return "" }
    if t.profile.DotAccFn == nil { return "" }
    dotFn := t.profile.DotAccFn[t.tier]
    if dotFn == "" { return "" }

    for i, arg := range args {
        call, ok := arg.(*ast.CallExpr)
        if !ok { continue }
        fnName := t.getHwyFuncName(call)
        switch fnName {
        case "DotProduct", "DotProductSS", "DotProductUU":
        default:
            continue
        }
        acc := t.translateExpr(args[1-i])
        a := t.translateExpr(call.Args[0])
        b := t.translateExpr(call.Args[1])
        return fmt.Sprintf("%s(%s, %s, %s)", dotFn, acc, a, b)
    }
    return ""
}
```

This means:
```go
acc = hwy.Add(acc, hwy.DotProduct(va, vb))
// Emits: acc = vdotq_s32(acc, va, vb)  ← single instruction, optimal
```

**c) Track DotAccType variables:**
When assigning `x := hwy.DotProduct(a, b)` or `x = hwy.Add(x, dot)` where x is DotAccType, add `x` to `accVarNames`.

**d) Profile-switch for operations on DotAccType vars:**
When emitting `hwy.Add(x, y)`, `hwy.ReduceSum(x)`, `hwy.Store(x, ...)`, or `hwy.Zero[int32]()`:
- Check if operand is in `accVarNames`
- If yes, use `accProfile` instead of `profile` for the operation's intrinsic lookup

Specific intrinsic mappings when using accProfile (int32):
- `hwy.Add(acc, dot)` → `vaddq_s32(acc, dot)` (from int32 profile's AddFn)
- `hwy.ReduceSum(acc)` → `vaddvq_s32(acc)` (from int32 profile's ReduceSumFn)
- `hwy.Zero[int32]()` → `vdupq_n_s32(0)` (from int32 profile's DupFn)
- `hwy.Store(acc, buf)` → `vst1q_s32(...)` (from int32 profile's StoreFn)

**e) Update `inferType()` for DotProduct/DotProductSS/DotProductUU:**
```go
case "DotProduct", "DotProductSS", "DotProductUU":
    if accType, ok := t.profile.DotAccType[t.tier]; ok {
        return cVarInfo{cType: accType, isVector: true}
    }
```

**f) Graceful error for unsupported combos:**

When the translator encounters DotProduct but the profile has no DotAccFn, return a typed error (e.g., `ErrUnsupportedOp`). The C generator catches this and skips the combo, allowing fallback to handle it.

### 6. `cmd/hwygen/c_generator.go` — Feature guard + graceful skip

**Feature guard** — extend `elemTypeFeatureGuard()`:
```go
func elemTypeFeatureGuard(elemType string, profile *CIntrinsicProfile) string {
    if isFloat16Type(elemType) { return "hwy.HasARMFP16()" }
    if isBFloat16Type(elemType) { return "hwy.HasARMBF16()" }
    if profile != nil && len(profile.DotAccFn) > 0 {
        return "hwy.HasARMDotProd()"
    }
    return ""
}
```

Add to guard ordering in `emitGuardedDispatchAssignments()`:
```go
for _, guard := range []string{"hwy.HasARMFP16()", "hwy.HasARMBF16()", "hwy.HasARMDotProd()"} {
```

**Graceful skip** — in the C generation loop, catch translator errors:
```go
cFile, cerr := emitter.EmitASTTranslatedC(&pf, cOutputDir)
if cerr != nil {
    if errors.Is(cerr, ErrUnsupportedOp) {
        // Profile doesn't support required operation — skip this combo
        continue
    }
    return cerr  // real error
}
```

**getCElemTypes** — add int8 handling:
```go
case "int8":
    return "int8"
```

### 7. `cmd/hwygen/parser.go` — GetConcreteTypes for int8|uint8

Add to `GetConcreteTypes()`:
```go
if strings.Contains(constraint, "int8") && strings.Contains(constraint, "uint8") {
    return []string{"int8", "uint8"}
}
```

This allows the constraint `int8 | uint8` to auto-expand. However, since we use `//hwy:gen T={int8, uint8}` explicitly, this is optional (the gen directive bypasses GetConcreteTypes).

### 8. `hwy/contrib/vec/dot_int_base.go` — Generic int dot product (NEW)

```go
//go:generate go run ../../../cmd/hwygen -input dot_int_base.go -output . -targets neon:asm,fallback -dispatch dotint

//hwy:gen T={int8, uint8}
func BaseDotInt[T int8 | uint8](a, b []T) int32 {
    n := min(len(a), len(b))
    lanes := hwy.NumLanes[T]()
    acc := hwy.Zero[int32]()       // DotAccType → int32x4_t via accProfile

    var i int
    for i = 0; i+lanes <= n; i += lanes {
        va := hwy.Load(a[i:])      // int8x16_t (or uint8x16_t)
        vb := hwy.Load(b[i:])
        // Fused by translator to: acc = vdotq_s32(acc, va, vb)
        acc = hwy.Add(acc, hwy.DotProduct(va, vb))
    }

    result := hwy.ReduceSum(acc)   // int32 ReduceSum via accProfile

    for ; i < n; i++ {
        result += int32(a[i]) * int32(b[i])
    }
    return result
}
```

hwygen generates:
- T=int8, neon:asm → C with `vld1q_s8`, `vdotq_s32`, `vaddvq_s32` → GoAT → assembly ✓
- T=uint8, neon:asm → **skipped** (uint8 profile lacks DotAccFn)
- T=int8, fallback → scalar Go using `hwy.DotProduct[int8]` ✓
- T=uint8, fallback → scalar Go using `hwy.DotProduct[uint8]` ✓

Dispatch variables:
- `DotIntInt8 func(a, b []int8) int32` — NEON asm on arm64, fallback elsewhere
- `DotIntUint8 func(a, b []uint8) int32` — fallback everywhere (for now)

Generic dispatcher (auto-generated by hwygen):
```go
func DotInt[T int8 | uint8](a, b []T) int32 {
    switch any(a).(type) {
    case []int8:
        return DotIntInt8(any(a).([]int8), any(b).([]int8))
    case []uint8:
        return DotIntUint8(any(a).([]uint8), any(b).([]uint8))
    }
    panic("unreachable")
}
```

### 9. Tests — `hwy/contrib/vec/dot_int_test.go`

- Known values: `DotInt([1,2,3,...16], [1,1,1,...])` == 136
- Test both int8 and uint8 via generic `DotInt[T]`
- Empty / short inputs
- Large arrays (10000+ elements)
- Dispatch vs fallback comparison
- Negative values (int8), overflow edge cases
- Benchmark: {256, 1024, 4096, 16384} elements for both types

## Implementation Order

1. Runtime detection: `HasARMDotProd()` + detect files + stubs
2. Scalar primitives: `hwy/ops_int8dot.go` (DotProductSS, DotProductUU, DotProduct[T])
3. C profile: `neonInt8DotProdProfile` in `c_profiles.go`
4. C translator: mixed-type support (accProfile, accVarNames, DotProduct ops, Add+DotProduct fusion, graceful error for missing DotAccFn)
5. C generator: feature guard for dotprod profiles, graceful skip on ErrUnsupportedOp, int8 in getCElemTypes
6. Parser: GetConcreteTypes for int8|uint8 constraint (optional, //hwy:gen bypasses this)
7. Base function: `hwy/contrib/vec/dot_int_base.go`
8. `go generate ./hwy/contrib/vec/...`
9. Tests: `dot_int_test.go`
10. Verify: SIMD path + fallback path + assembly inspection

## Key Design Decisions

**Generic base function**: `BaseDotInt[T int8|uint8]` generates type-specialized code for both types. This is the highway pattern — write once, generate SIMD for each type.

**Profile centered on int8 (not int32)**: The primary type is int8 because Load operations are on int8 data. DotAccFn/DotAccType handle the int32 accumulator as a widening operation — same pattern as BFloat16 DotAccumulate.

**Graceful skip for uint8 NEON**: Rather than creating a conflicting uint8 profile or modifying the existing one (which would affect gguf dequantize code), the C generator skips (uint8, neon:asm) combos when the translator reports the profile lacks DotAccFn. uint8 still works via scalar fallback. NEON asm for uint8 can be added later with per-combo profile overrides.

**Mixed-type translator via accProfile**: Rather than full type inference, track a set of "accumulator variables" and switch to accProfile for operations on them. Minimal change to the translator.

**No accumulator in API**: Matches Go's archsimd design. Users compose `hwy.Add(acc, hwy.DotProduct(a, b))`. The C translator fuses this pattern into a single `vdotq_s32(acc, a, b)` instruction via the `tryFuseDotAccumulate` peephole in the `Add` handler.

**DOTPROD vs I8MM**: DOTPROD (ARMv8.2-A) gives SS and UU variants. I8MM (ARMv8.6-A) adds US mixed-sign. We implement DOTPROD variants now; I8MM variants come later with a separate feature flag.

**Feature guard via profile introspection**: `elemTypeFeatureGuard` checks `profile.DotAccFn` rather than hard-coding element type names. This is forward-compatible — any future profile with DotAccFn automatically gets the dotprod guard.

## Verification

```bash
# Build
GOEXPERIMENT=simd go build ./hwy/... ./hwy/contrib/vec/...

# Test SIMD (vdotq_s32 on Apple Silicon)
GOEXPERIMENT=simd go test -v -run DotInt ./hwy/contrib/vec/...

# Test fallback
HWY_NO_SIMD=1 GOEXPERIMENT=simd go test -v -run DotInt ./hwy/contrib/vec/...

# Verify assembly contains sdot
grep -r "sdot" hwy/contrib/vec/asm/*.s

# hwygen tests
GOEXPERIMENT=simd go test -v ./cmd/hwygen/...

# Benchmark
GOEXPERIMENT=simd go test -bench=DotInt -benchmem ./hwy/contrib/vec/...
```
