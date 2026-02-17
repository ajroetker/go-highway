# Plan: BF16 Accumulator Widening in C Emitter

## Context

Level 1 (register-accumulator restructuring) is complete. All matmul kernels now use j-outer/p-inner loops with register accumulators. The BF16 C code currently emits `bf16_fma_q(acc, a, b)` which promotes all 3 args, does 2 FMAs, then demotes+combines — ~11 ops per FMA call. With 4 accumulators per K iteration, that's 44 ops/K.

With widened f32 accumulators, we promote only the 2 inputs per FMA (4 ops) + 2 native f32 FMAs = 6 ops per accumulator, ~18 ops/K total (after CSE of vA promotes). The demote+combine happens once at store time. **~2.4x reduction** in inner-loop instruction count for BF16.

## Files to Modify

1. `cmd/hwygen/c_profiles.go` — Add widening fields to `CIntrinsicProfile` and `neonBF16Profile()`
2. `cmd/hwygen/c_ast_translator.go` — Add widened accumulator tracking and emission
3. `cmd/hwygen/hwygen_test.go` — Update BF16 matmul test assertions

## Step 1: Add Profile Fields

**File:** `cmd/hwygen/c_profiles.go`

Add to `CIntrinsicProfile` struct (after existing `DotAccType` fields, ~line 78):

```go
// Accumulator widening: keep accumulators in a wider type (e.g., f32 for BF16)
// to avoid promote/demote round-trips on every FMA iteration.
WidenAccumulators bool   // Enable widened accumulator optimization
WidenedAccZero    string // Zero init expr: "vdupq_n_f32(0.0f)"
WidenedAccType    string // Widened type: "float32x4_t"
WidenedFmaFn      string // Native FMA on widened type: "vfmaq_f32"
WidenedAddFn      string // Native Add on widened type: "vaddq_f32"
```

Uses existing profile fields for promote/demote/combine:
- `SplitPromoteLo` → `"bf16_promote_lo(%s)"` (already exists)
- `SplitPromoteHi` → `"bf16_promote_hi(%s)"` (already exists)
- `DemoteFn` → `"bf16_demote_half(%s)"` (already exists)
- `CombineFn` → `"bf16_combine(%s, %s)"` (already exists)

Set these in `neonBF16Profile()` (~line 479):

```go
WidenAccumulators: true,
WidenedAccZero:    "vdupq_n_f32(0.0f)",
WidenedAccType:    "float32x4_t",
WidenedFmaFn:      "vfmaq_f32",
WidenedAddFn:      "vaddq_f32",
```

## Step 2: Add Widened Variable Tracking to Translator

**File:** `cmd/hwygen/c_ast_translator.go`

Add field to `CASTTranslator` struct (~line 40):
```go
widenedVars map[string]bool  // Variables that are widened accumulators (lo/hi pair)
```

Add `isWidened` field to `cVarInfo` struct (~line 95):
```go
isWidened bool  // true if widened accumulator (varname_lo/varname_hi pair)
```

Initialize `widenedVars` in the constructor (same place `vars` is initialized).

Add helper to check if an AST expr is a widened var:
```go
func (t *CASTTranslator) isWidenedVar(expr ast.Expr) (string, bool) {
    if id, ok := expr.(*ast.Ident); ok {
        return id.Name, t.widenedVars[id.Name]
    }
    return "", false
}
```

Add helper to check for `hwy.X()` calls by name:
```go
func isHwyCall(expr ast.Expr, fnName string) (*ast.CallExpr, bool) {
    call, ok := expr.(*ast.CallExpr)
    if !ok { return nil, false }
    fun := call.Fun
    if idx, ok := fun.(*ast.IndexExpr); ok { fun = idx.X }
    sel := extractSelectorExpr(fun)
    if sel == nil { return nil, false }
    pkg, ok := sel.X.(*ast.Ident)
    if !ok || pkg.Name != "hwy" { return nil, false }
    return call, sel.Sel.Name == fnName
}
```

## Step 3: Intercept Patterns in Statement Translation

Five patterns need special handling. All intercepts go in `translateAssignStmt` and `emitHwyStore`.

### Pattern 1: `acc0 := hwy.Zero[T]()` → two f32 zeros

In `translateAssignStmt`, DEFINE case (~line 1193), before the existing `t.inferType` call:

```go
if t.profile.WidenAccumulators {
    if _, ok := isHwyCall(rhs, "Zero"); ok {
        wideType := t.profile.WidenedAccType
        zeroExpr := t.profile.WidenedAccZero
        t.widenedVars[lhsName] = true
        t.vars[lhsName] = cVarInfo{cType: wideType, isVector: true, isWidened: true}
        t.writef("%s %s_lo = %s;\n", wideType, lhsName, zeroExpr)
        t.writef("%s %s_hi = %s;\n", wideType, lhsName, zeroExpr)
        return
    }
}
```

### Pattern 2: `acc0 = hwy.MulAdd(vA, vB, acc0)` → two f32 FMAs

In `translateAssignStmt`, ASSIGN case (~line 1229), before the existing `translateExpr` call:

```go
if t.widenedVars[lhsName] {
    if call, ok := isHwyCall(rhs, "MulAdd"); ok && len(call.Args) >= 3 {
        a := t.translateExpr(call.Args[0])
        b := t.translateExpr(call.Args[1])
        fmaFn := t.profile.WidenedFmaFn
        proLo := t.profile.SplitPromoteLo
        proHi := t.profile.SplitPromoteHi
        t.writef("%s_lo = %s(%s_lo, %s, %s);\n", lhsName, fmaFn, lhsName,
            fmt.Sprintf(proLo, a), fmt.Sprintf(proLo, b))
        t.writef("%s_hi = %s(%s_hi, %s, %s);\n", lhsName, fmaFn, lhsName,
            fmt.Sprintf(proHi, a), fmt.Sprintf(proHi, b))
        return
    }
}
```

### Pattern 3: `hwy.Store(acc0, dst)` → combine+demote+store

In `emitHwyStore` (~line 2379), before the existing translation:

```go
if name, ok := t.isWidenedVar(args[0]); ok {
    ptr := t.translateExpr(args[1])
    if t.profile.CastExpr != "" {
        ptr = fmt.Sprintf("%s(%s)", t.profile.CastExpr, ptr)
    }
    lo := fmt.Sprintf(t.profile.DemoteFn, name+"_lo")
    hi := fmt.Sprintf(t.profile.DemoteFn, name+"_hi")
    combined := fmt.Sprintf(t.profile.CombineFn, lo, hi)
    storeFn := t.profile.StoreFn[t.tier]
    t.writef("%s(%s, %s);\n", storeFn, ptr, combined)
    return
}
```

### Pattern 4: `hwy.Store(hwy.Add(vC, acc0), dst)` → promote+add+store

In `emitHwyStore`, check if vec arg is `hwy.Add` with a widened operand:

```go
if addCall, ok := isHwyCall(args[0], "Add"); ok && len(addCall.Args) >= 2 {
    wName, narrowIdx := "", -1
    for i, arg := range addCall.Args {
        if n, isW := t.isWidenedVar(arg); isW {
            wName = n
        } else {
            narrowIdx = i
        }
    }
    if wName != "" && narrowIdx >= 0 {
        narrow := t.translateExpr(addCall.Args[narrowIdx])
        ptr := t.translateExpr(args[1])
        if t.profile.CastExpr != "" {
            ptr = fmt.Sprintf("%s(%s)", t.profile.CastExpr, ptr)
        }
        addFn := t.profile.WidenedAddFn
        proLo, proHi := t.profile.SplitPromoteLo, t.profile.SplitPromoteHi
        lo := fmt.Sprintf("%s(%s, %s_lo)", addFn, fmt.Sprintf(proLo, narrow), wName)
        hi := fmt.Sprintf("%s(%s, %s_hi)", addFn, fmt.Sprintf(proHi, narrow), wName)
        dLo := fmt.Sprintf(t.profile.DemoteFn, lo)
        dHi := fmt.Sprintf(t.profile.DemoteFn, hi)
        combined := fmt.Sprintf(t.profile.CombineFn, dLo, dHi)
        storeFn := t.profile.StoreFn[t.tier]
        t.writef("%s(%s, %s);\n", storeFn, ptr, combined)
        return
    }
}
```

### Pattern 5: `vC = hwy.Add(vC, acc0)` → promote+add+combine

In `translateAssignStmt`, ASSIGN case, check for Add with widened arg:

```go
if addCall, ok := isHwyCall(rhs, "Add"); ok && len(addCall.Args) >= 2 {
    wName, narrowIdx := "", -1
    for i, arg := range addCall.Args {
        if n, isW := t.isWidenedVar(arg); isW {
            wName = n
        } else {
            narrowIdx = i
        }
    }
    if wName != "" && narrowIdx >= 0 {
        narrow := t.translateExpr(addCall.Args[narrowIdx])
        addFn := t.profile.WidenedAddFn
        proLo, proHi := t.profile.SplitPromoteLo, t.profile.SplitPromoteHi
        lo := fmt.Sprintf("%s(%s, %s_lo)", addFn, fmt.Sprintf(proLo, narrow), wName)
        hi := fmt.Sprintf("%s(%s, %s_hi)", addFn, fmt.Sprintf(proHi, narrow), wName)
        dLo := fmt.Sprintf(t.profile.DemoteFn, lo)
        dHi := fmt.Sprintf(t.profile.DemoteFn, hi)
        combined := fmt.Sprintf(t.profile.CombineFn, dLo, dHi)
        t.writef("%s = %s;\n", lhsName, combined)
        return
    }
}
```

## Step 4: Fallback Materialization

In `translateExpr`, Ident case (~line 1831), add materialization for widened vars used in unrecognized contexts:

```go
case *ast.Ident:
    if e.Name == "nil" { return "0" }
    if t.widenedVars[e.Name] {
        lo := fmt.Sprintf(t.profile.DemoteFn, e.Name+"_lo")
        hi := fmt.Sprintf(t.profile.DemoteFn, e.Name+"_hi")
        return fmt.Sprintf(t.profile.CombineFn, lo, hi)
    }
    return e.Name
```

This ensures correctness for any pattern not explicitly optimized.

## Step 5: Update Tests

**File:** `cmd/hwygen/hwygen_test.go`

Update BF16 matmul test assertions to expect widened pattern:
- `bf16_zero_q()` → `vdupq_n_f32(0.0f)` with `_lo`/`_hi` suffixes
- `bf16_fma_q(acc0, vA, vB)` → `vfmaq_f32(acc0_lo, bf16_promote_lo(vA), ...)`
- `vst1q_bf16(ptr, acc0)` → `vst1q_bf16(ptr, bf16_combine(bf16_demote_half(acc0_lo), ...))`

## Step 6: Regenerate and Verify

```bash
# Build hwygen with changes
GOEXPERIMENT=simd go build ./cmd/hwygen

# Run hwygen tests
GOEXPERIMENT=simd go test ./cmd/hwygen/...

# Regenerate matmul and block_kernel assembly (BF16 output changes)
cd hwy/contrib/matmul
GOEXPERIMENT=simd go generate matmul_base.go
GOEXPERIMENT=simd go generate block_kernel.go

# Build everything
GOEXPERIMENT=simd go build ./...

# Run matmul tests (SIMD + fallback)
GOEXPERIMENT=simd go test ./hwy/contrib/matmul/...
HWY_NO_SIMD=1 GOEXPERIMENT=simd go test ./hwy/contrib/matmul/...

# Verify generated BF16 C compiles
clang -S -O3 -target arm64-apple-macos -march=armv8.6-a+bf16+simd \
  hwy/contrib/matmul/asm/c/basematmul_c_bf16_neon_arm64.c -o /dev/null
```

## Expected Generated Output

**Before (current):**
```c
bfloat16x8_t acc0 = bf16_zero_q();
// per K:
acc0 = bf16_fma_q(acc0, vA, vld1q_bf16(...));  // ~11 ops each
// store:
vst1q_bf16(ptr, acc0);
```

**After (widened):**
```c
float32x4_t acc0_lo = vdupq_n_f32(0.0f);
float32x4_t acc0_hi = vdupq_n_f32(0.0f);
// per K:
acc0_lo = vfmaq_f32(acc0_lo, bf16_promote_lo(vA), bf16_promote_lo(vld1q_bf16(...)));
acc0_hi = vfmaq_f32(acc0_hi, bf16_promote_hi(vA), bf16_promote_hi(vld1q_bf16(...)));
// store (once):
vst1q_bf16(ptr, bf16_combine(bf16_demote_half(acc0_lo), bf16_demote_half(acc0_hi)));
```
