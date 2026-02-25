# Plan: `//hwy:specializes` and `//hwy:targets` Directives

## Context

PR #50 added `//hwy:gen` directives for multi-type SIMD dispatch with cross-product expansion. The next step is C++-style template specialization: different function bodies for different types and/or architectures, unified under a single dispatch group. This lets half-precision matmul use widening loads while float32 matmul uses direct FMA, or SME use outer-product tiles while NEON uses dot-product reduction — all behind a single `MatMul[T]()` dispatch.

## Directive Syntax

```go
// matmul_base.go — default implementation
//hwy:gen T={float32, float64}
func BaseMatMul[T hwy.Floats](a, b, c []T, m, n, k int) {
    // FMA-based pipelined implementation
}

// matmul_half_base.go — type + arch specialization
//hwy:gen T={hwy.Float16, hwy.BFloat16}
//hwy:specializes MatMul
//hwy:targets neon
func BaseMatMulHalf[T hwy.HalfFloats](a, b, c []T, m, n, k int) {
    // Widening load into f32 accumulators, narrow store
}
```

- `//hwy:specializes <Name>` — joins this function into the `<Name>` dispatch group
- `//hwy:targets <t1,t2,...>` — restricts this function to specific targets (case-insensitive, e.g. `neon`, `avx2`, `sme`)
- Both directives use the same 5-line proximity rule as `//hwy:gen`
- Specialization functions are auto-discovered from sibling `*_base.go` files (no CLI change)

## Files to Modify

| File | Changes |
|------|---------|
| `cmd/hwygen/parser.go` | New fields on `ParsedFunc`, directive parsers, sibling file scanning |
| `cmd/hwygen/generator.go` | `DispatchGroup` type, `buildDispatchGroups`, `selectSourceFunc`, modified generation loop |
| `cmd/hwygen/emitter.go` | No structural change — receives synthetic merged `ParsedFunc` per group |
| `cmd/hwygen/c_generator.go` | Same selection logic for C/ASM path |
| `cmd/hwygen/hwygen_test.go` | Tests for parsing, group building, selection, end-to-end |

## Step 1: Parser — New Fields and Directive Parsing (`parser.go`)

### 1a. Add fields to `ParsedFunc` (line 30)

```go
type ParsedFunc struct {
    // ... existing fields ...
    SpecializesGroup string   // from //hwy:specializes; empty = primary function
    AllowedTargets   []string // from //hwy:targets; empty = all targets
    SourceFile       string   // which file this function came from (for error messages)
}
```

### 1b. Add directive types (near `TypeCombination`, line 43)

```go
type SpecializesDirective struct {
    Line      int
    GroupName string
}

type TargetsDirective struct {
    Line    int
    Targets []string
}
```

### 1c. Implement `parseSpecializesDirectives` and `parseTargetsDirectives`

Same pattern as `parseGenDirectives` (line 553): iterate `file.Comments`, match prefix, extract line number.

- `//hwy:specializes MatMul` → `SpecializesDirective{Line: N, GroupName: "MatMul"}`
- `//hwy:targets neon,avx512` → `TargetsDirective{Line: N, Targets: ["neon", "avx512"]}`

### 1d. Wire into `Parse()` (after line 247)

```go
specializesDirectives := parseSpecializesDirectives(file, fset)
targetsDirectives := parseTargetsDirectives(file, fset)
```

In the function loop (after line 312, where `//hwy:gen` matching ends), match these to functions using the same 5-line proximity rule:

```go
for _, sd := range specializesDirectives {
    if sd.Line >= funcLine-5 && sd.Line < funcLine {
        pf.SpecializesGroup = sd.GroupName
    }
}
for _, td := range targetsDirectives {
    if td.Line >= funcLine-5 && td.Line < funcLine {
        pf.AllowedTargets = td.Targets
    }
}
pf.SourceFile = filename
```

### 1e. Implement `scanSpecializations` (new function, after `scanPackageFuncs`)

Extends the `scanPackageFuncs` pattern (line 1262) but for specialization discovery:

1. Scan sibling `*_base.go` files (same dir, skip self, skip test/gen files)
2. Parse each file with `parser.ParseComments`
3. Scan comments for `//hwy:specializes` directives
4. For functions with a specializes directive: fully parse them (type params, params, returns, hwy calls, `//hwy:gen`, `//hwy:targets`) and append to `result.Funcs`
5. Set `SourceFile` on each discovered function

Call from `Parse()` after `scanPackageFuncs` (line 367):
```go
scanSpecializations(filename, result)
```

## Step 2: Dispatch Group Building (`generator.go`)

### 2a. `DispatchGroup` type

```go
type DispatchGroup struct {
    GroupName       string
    Primary         *ParsedFunc   // default function (widest constraint)
    Specializations []*ParsedFunc // functions with //hwy:specializes
    AllCombos       []TypeCombination // union of all members' combos
    AllTypeParams   []TypeParam   // from Primary (widest)
    Private         bool
}
```

### 2b. `buildDispatchGroups(funcs []ParsedFunc) ([]DispatchGroup, error)`

1. Separate functions: `SpecializesGroup == ""` → primary (keyed by derived group name), otherwise → specialization
2. Validate each specialization references an existing primary
3. Validate signature compatibility (same param count, structurally compatible types)
4. Compute `AllCombos` = union of all members' `TypeCombinations`
5. Widen the primary's constraint if specializations add types outside it (reuse `GetConcreteTypes` hierarchy)
6. Return sorted `[]DispatchGroup`

For functions without any specializations, they form single-member groups (no behavioral change).

### 2c. `selectSourceFunc(group *DispatchGroup, target Target, combo TypeCombination) *ParsedFunc`

For each (target, combo):
1. Filter specializations: must cover this combo (`comboMatchesFunc`) AND allow this target (`targetAllowed`)
2. Score: target-restricted function scores higher than unrestricted
3. Pick highest score; error on tie between equally-specific specializations
4. Fall through to primary if no specialization matches
5. Return `nil` if no function covers this combo on this target (combo only from a target-restricted specialization, and this isn't that target)

Helper: `targetAllowed(pf, target)` — if `AllowedTargets` is empty, return true; otherwise check if target name (lowercased) matches any entry.

### 2d. Modify `Generator.Run()` Go SIMD loop (lines 220-287)

Replace:
```go
for _, pf := range result.Funcs {
    combos := ...
    for _, combo := range combos {
        // transform pf
    }
}
```

With:
```go
groups, err := buildDispatchGroups(result.Funcs)
// ...
for _, group := range groups {
    for _, combo := range group.AllCombos {
        sourcePF, err := selectSourceFunc(&group, target, combo)
        if sourcePF == nil { continue } // no coverage on this target

        // Transform sourcePF body for this combo
        // ...existing transform logic...

        // NAME NORMALIZATION: use Primary's name for output
        transformResult.FuncDecl.Name.Name = group.Primary.Name + target.Suffix()
        suffix := comboTypeSuffix(combo, group.AllTypeParams)
        if suffix != "" && suffix != "Float32" && len(group.AllTypeParams) > 0 {
            transformResult.FuncDecl.Name.Name += "_" + suffix
        }
        if group.Private {
            transformResult.FuncDecl.Name.Name = makeUnexported(transformResult.FuncDecl.Name.Name)
        }
    }
}
```

Name normalization ensures `BaseMatMulHalf` on NEON for Float16 produces `BaseMatMul_neon_Float16`, matching what the emitter expects.

### 2e. Build synthetic funcs for emitter (before `EmitDispatcher` call, line 341)

```go
synthFuncs := make([]ParsedFunc, 0, len(groups))
for _, group := range groups {
    synth := synthPrimaryForDispatch(&group)
    synthFuncs = append(synthFuncs, synth)
}
EmitDispatcher(synthFuncs, allTargets, ...)
```

`synthPrimaryForDispatch` copies the primary, sets `TypeCombinations = group.AllCombos`, and widens the constraint if needed. The emitter receives what looks like a single function with all combos — **no emitter code changes required**.

## Step 3: C Generator (`c_generator.go`)

Same pattern as Step 2d: build dispatch groups, use `selectSourceFunc` to pick the source function body for each (target, combo). Override the function name to use the primary's name before passing to the C emitter. The dispatch wiring in `emitGuardedDispatchAssignments` derives names from `pf.Name`, so name normalization keeps it working.

## Step 4: Tests (`hwygen_test.go`)

### Unit tests
- `TestParseSpecializesDirective` — parse `//hwy:specializes X` from comment text
- `TestParseTargetsDirective` — parse `//hwy:targets neon,avx2` from comment text
- `TestBuildDispatchGroups` — table-driven: primary + specializations → correct groups, combos, errors
- `TestSelectSourceFunc` — table-driven covering:
  - No specialization → returns primary
  - Type specialization matches → returns specialization
  - Target specialization matches → returns specialization
  - Target doesn't match → falls through to primary
  - No coverage (target-only combo, wrong target) → returns nil
  - Ambiguous tie → returns error

### Integration tests
- `TestSpecializesEndToEnd` — write two Go source strings (primary + specialization), run generator, verify:
  - Dispatch vars cover union of all combos
  - Generic wrapper uses widest constraint
  - Init functions wire correct impl names
  - Specialization body is used for matching (target, combo)
  - Primary body is used for non-matching (target, combo)

### Error tests
- Specialization referencing nonexistent group → error
- Two specializations claiming same (target, combo) → error
- Signature mismatch (different param count) → error

## Verification

```bash
# Build hwygen
go build ./cmd/hwygen

# Run hwygen tests (includes all new tests)
go test -v ./cmd/hwygen/...

# Full project build with SIMD
GOEXPERIMENT=simd go build ./...

# Full project tests
GOEXPERIMENT=simd go test ./...
```
