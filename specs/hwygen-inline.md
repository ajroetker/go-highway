# hwygen: Helper Function Inlining

## Problem

Currently hwygen transforms calls like `transposeBlockSIMD(...)` to `transposeBlockSIMD_avx2(...)` but doesn't inline the helper's body. This causes issues when:

1. The helper uses `hwy.LoadFull()` which loads `MaxLanes[T]()` elements at runtime
2. The generated code passes a target-specific `lanes` constant (e.g., `lanes := 8` for AVX2)
3. If runtime `MaxLanes` differs from the generated `lanes`, the code breaks

**Example Problem:**
```go
// Generated AVX2 code
func BaseTranspose2D_avx2(src []float32, m, k int, dst []float32) {
    lanes := 8  // hwygen knows AVX2 has 8 lanes
    for i := 0; i <= m-lanes; i += lanes {
        transposeBlockSIMD(src, dst, i, j, m, k, lanes)  // NOT inlined!
    }
}

// Helper function (not specialized)
func transposeBlockSIMD[T hwy.Floats](..., lanes int) {
    rows[r] = hwy.LoadFull(src[...])  // Loads MaxLanes[T]() elements at RUNTIME
    // If MaxLanes != 8, this breaks!
}
```

## Solution

Always inline local helper functions (Base*/base* functions called from the main generated function). This ensures the entire code path gets specialized for each target.

## Implementation Plan

### Phase 1: Parse All Local Helpers

**File: `parser.go`**

1. Parse ALL Base*/base* functions in the file, not just those with hwy calls:
```go
// Current (line ~180):
if len(pf.HwyCalls) > 0 {
    result.Funcs = append(result.Funcs, pf)
}

// Change to:
result.Funcs = append(result.Funcs, pf)  // Always include Base* functions
```

2. Track which functions call which local helpers:
```go
type ParsedFile struct {
    // ... existing fields ...
    LocalHelperCalls map[string][]string  // funcName -> list of local Base*/base* helpers it calls
}
```

3. In `findHwyCalls()`, already detects local Base* calls - just need to store them:
```go
// Already exists around line 280-309:
if ident, ok := expr.Fun.(*ast.Ident); ok {
    if hasBasePrefix(ident.Name) {
        calls = append(calls, HwyCall{
            Package:  "local",
            FuncName: ident.Name,
        })
    }
}
```

### Phase 2: Transform Helper Functions First

**File: `generator.go`**

Modify `Run()` to transform in dependency order:

```go
func (g *Generator) Run() error {
    // ... existing parsing ...

    // Identify which functions are helpers (called by other Base* functions)
    helpers := findLocalHelpers(parsed)

    for _, target := range targets {
        // 1. First pass: transform all helper functions
        transformedHelpers := make(map[string]*ast.FuncDecl)
        for _, fn := range parsed.Funcs {
            if helpers[fn.Name] {
                transformed := g.transformer.TransformHelper(fn, target)
                transformedHelpers[fn.Name] = transformed
            }
        }

        // 2. Second pass: transform main functions with inlining
        ctx := &transformContext{
            inlinedHelpers: transformedHelpers,
        }
        for _, fn := range parsed.Funcs {
            if !helpers[fn.Name] {
                g.transformer.TransformWithInlining(fn, target, ctx)
            }
        }
    }
}

// findLocalHelpers returns set of function names that are called by other Base* functions
func findLocalHelpers(parsed *ParsedFile) map[string]bool {
    helpers := make(map[string]bool)
    for _, fn := range parsed.Funcs {
        for _, call := range fn.HwyCalls {
            if call.Package == "local" {
                helpers[call.FuncName] = true
            }
        }
    }
    return helpers
}
```

### Phase 3: Inline During Transformation

**File: `transformer.go`**

Add inlining logic to `transformCallExpr()`:

```go
func (t *Transformer) transformCallExpr(call *ast.CallExpr, ctx *transformContext) {
    if ident, ok := call.Fun.(*ast.Ident); ok {
        if strings.HasPrefix(ident.Name, "Base") || strings.HasPrefix(ident.Name, "base") {
            // Check if this helper should be inlined
            if helper, ok := ctx.inlinedHelpers[ident.Name]; ok {
                inlinedStmts := t.inlineHelper(call, helper, ctx)
                ctx.replaceWithStatements = inlinedStmts
                return
            }
            // Otherwise, add target suffix as before
            ident.Name = ident.Name + ctx.target.Suffix()
        }
    }
}

func (t *Transformer) inlineHelper(call *ast.CallExpr, helper *ast.FuncDecl, ctx *transformContext) []ast.Stmt {
    // 1. Create parameter -> argument mapping
    paramMap := make(map[string]ast.Expr)
    for i, param := range helper.Type.Params.List {
        for _, name := range param.Names {
            paramMap[name.Name] = call.Args[i]
        }
    }

    // 2. Clone and substitute helper body
    clonedBody := t.cloneStmtList(helper.Body.List)
    t.substituteParams(clonedBody, paramMap)

    // 3. Handle return values (if any)
    // For void helpers, just return statements
    // For value-returning helpers, need to assign to temp var

    return clonedBody
}
```

### Phase 4: Handle Statement Replacement

**File: `transformer.go`**

The tricky part is replacing a single call expression with multiple statements:

```go
type transformContext struct {
    // ... existing fields ...
    replaceWithStatements []ast.Stmt  // If set, replace current stmt with these
}

func (t *Transformer) transformStmt(stmt ast.Stmt, ctx *transformContext) ast.Stmt {
    ctx.replaceWithStatements = nil

    result := t.doTransformStmt(stmt, ctx)

    if ctx.replaceWithStatements != nil {
        // Wrap in block statement if needed
        return &ast.BlockStmt{List: ctx.replaceWithStatements}
    }
    return result
}
```

### Phase 5: Variable Renaming to Avoid Conflicts

When inlining multiple times, local variables may conflict:

```go
func (t *Transformer) inlineHelper(...) []ast.Stmt {
    // Generate unique suffix for this inline site
    suffix := fmt.Sprintf("_%d", ctx.inlineCounter)
    ctx.inlineCounter++

    // Rename all local variables in cloned body
    t.renameLocals(clonedBody, suffix)

    return clonedBody
}
```

## Example: Before and After

**Input (`transpose_base.go`):**
```go
// Helper function - will be automatically inlined
func transposeBlockSIMD[T hwy.Floats](src, dst []T, startI, startJ, m, k, lanes int) {
    rows := make([]hwy.Vec[T], lanes)
    for r := 0; r < lanes; r++ {
        rows[r] = hwy.LoadFull(src[(startI+r)*k+startJ:])
    }
    // ... interleave logic ...
}

func BaseTranspose2D[T hwy.Floats](src []T, m, k int, dst []T) {
    lanes := hwy.MaxLanes[T]()
    for i := 0; i <= m-lanes; i += lanes {
        for j := 0; j <= k-lanes; j += lanes {
            transposeBlockSIMD(src, dst, i, j, m, k, lanes)
        }
    }
}
```

**Generated AVX2 (with inlining):**
```go
func BaseTranspose2D_avx2(src []float32, m int, k int, dst []float32) {
    lanes := 8
    for i := 0; i <= m-lanes; i += lanes {
        for j := 0; j <= k-lanes; j += lanes {
            // INLINED from transposeBlockSIMD:
            {
                rows_0 := make([]archsimd.Float32x8, 8)
                for r_0 := 0; r_0 < 8; r_0++ {
                    rows_0[r_0] = archsimd.LoadFloat32x8(src[(i+r_0)*k+j:])
                }
                // ... specialized interleave logic ...
            }
        }
    }
}
```

**Generated Fallback (scalar, with inlining):**
```go
func BaseTranspose2D_fallback(src []float32, m int, k int, dst []float32) {
    lanes := 1  // Fallback has 1 lane
    for i := 0; i <= m-lanes; i += lanes {
        for j := 0; j <= k-lanes; j += lanes {
            // INLINED from transposeBlockSIMD:
            {
                rows_0 := make([]float32, 1)
                for r_0 := 0; r_0 < 1; r_0++ {
                    rows_0[r_0] = src[(i+r_0)*k+j]  // Scalar load
                }
                // ... scalar operations (loop unrolled to single element) ...
            }
        }
    }
}
```

## Benefits

1. **Correctness**: `hwy.LoadFull` gets specialized to target-specific load (8 elements for AVX2, 1 for scalar)
2. **Scalar just works**: Fallback target inlines with `lanes := 1`, loops become single iterations
3. **Performance**: No function call overhead, better optimization opportunities
4. **No z_* overrides needed**: Can remove `z_transpose_amd64.go` and `z_transpose_other.go` since generated code will be correct
5. **Simpler user code**: Users can write helper functions without worrying about specialization

## Testing

1. Add test case with local helper function
2. Verify generated code has no function calls to local helpers (they're inlined)
3. Verify variable substitution works correctly (parameters replaced with arguments)
4. Verify variable renaming avoids conflicts when helper called multiple times
5. Verify nested inlining works (helper calls helper)
6. Run transpose tests on all targets (should pass without z_* overrides)

## Complexity Estimate

- Parser changes: ~50 lines
- Generator changes: ~100 lines
- Transformer changes: ~200 lines (inlining, substitution, renaming)
- Tests: ~100 lines

Total: ~450 lines of Go code
