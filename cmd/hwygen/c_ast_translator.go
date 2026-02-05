package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"sort"
	"strings"
)

// CASTTranslator walks a ParsedFunc's Go AST and emits GOAT-compatible C code
// with target-specific SIMD intrinsics. Unlike the template-based CEmitter
// (which pattern-matches math function names), this translator handles
// arbitrary control flow: matmul, transpose, dot product, etc.
type CASTTranslator struct {
	profile  *CIntrinsicProfile
	tier     string // primary SIMD tier: "q" for NEON, "ymm" for AVX2, "zmm" for AVX-512
	lanes    int    // number of elements per vector at primary tier
	elemType string // "float32", "float64"

	// Variable tracking
	vars   map[string]cVarInfo   // declared local variables
	params map[string]cParamInfo // function parameters

	// Slice length tracking: maps slice param name → C length variable name.
	// e.g. "code" → "len_code". Used to translate len(code) → len_code in C.
	sliceLenVars map[string]string

	// Deferred popcount accumulation: active only inside for-loops that
	// contain Load4 calls and sum += uint64(ReduceSum(PopCount(And(...)))) patterns.
	// Maps scalar variable name → accumulator info.
	deferredAccums map[string]*deferredAccum

	buf      *bytes.Buffer
	indent   int
	tmpCount int // counter for unique temporary variable names
}

// deferredAccum tracks a scalar variable being replaced by a vector accumulator
// for deferred horizontal reduction of popcount results.
type deferredAccum struct {
	scalarVar string // Go scalar variable name, e.g. "sum1_0"
	accVar    string // C vector accumulator name, e.g. "_pacc_0"
}

// deferredAccumsOrdered returns the active deferred accumulators in a stable
// order (sorted by accVar name) for deterministic C output.
func (t *CASTTranslator) deferredAccumsOrdered() []deferredAccum {
	accums := make([]deferredAccum, 0, len(t.deferredAccums))
	for _, acc := range t.deferredAccums {
		accums = append(accums, *acc)
	}
	sort.Slice(accums, func(i, j int) bool {
		return accums[i].accVar < accums[j].accVar
	})
	return accums
}

// cVarInfo tracks the C type of a local variable.
type cVarInfo struct {
	cType    string // "float32x4_t", "float", "long", "float *"
	isVector bool
	isPtr    bool
}

// cParamInfo tracks function parameter translation details.
type cParamInfo struct {
	goName  string // "a", "b", "c", "m", "n", "k"
	goType  string // "[]T", "int"
	cName   string // "a", "b", "c", "pm", "pn", "pk"
	cType   string // "float *", "long *"
	isSlice bool
	isInt   bool
}

// NewCASTTranslator creates a translator for the given profile and element type.
func NewCASTTranslator(profile *CIntrinsicProfile, elemType string) *CASTTranslator {
	tier, lanes := primaryTier(profile)
	return &CASTTranslator{
		profile:      profile,
		tier:         tier,
		lanes:        lanes,
		elemType:     elemType,
		vars:         make(map[string]cVarInfo),
		params:       make(map[string]cParamInfo),
		sliceLenVars: make(map[string]string),
		buf:          &bytes.Buffer{},
	}
}

// primaryTier returns the first non-scalar tier name and its lane count.
func primaryTier(p *CIntrinsicProfile) (string, int) {
	for _, t := range p.Tiers {
		if !t.IsScalar {
			return t.Name, t.Lanes
		}
	}
	// Fallback
	return "q", 4
}

// TranslateToC translates a ParsedFunc to GOAT-compatible C source code.
func (t *CASTTranslator) TranslateToC(pf *ParsedFunc) (string, error) {
	t.buf.Reset()
	t.vars = make(map[string]cVarInfo)
	t.params = make(map[string]cParamInfo)
	t.indent = 0

	// Build parameter map
	t.buildParamMap(pf)

	// Emit function signature
	t.emitFuncSignature(pf)

	// Emit int parameter dereferences
	t.indent = 1
	t.emitParamDerefs()

	// Translate function body
	if pf.Body != nil {
		t.translateBlockStmtContents(pf.Body)
	}

	// Close function
	t.indent = 0
	t.writef("}\n")

	return t.buf.String(), nil
}

// buildParamMap creates cParamInfo entries for each Go parameter.
func (t *CASTTranslator) buildParamMap(pf *ParsedFunc) {
	for _, p := range pf.Params {
		info := cParamInfo{
			goName: p.Name,
			goType: p.Type,
		}
		if strings.HasPrefix(p.Type, "[]") {
			// Slice param → pointer. Determine element type from the slice type.
			info.isSlice = true
			info.cName = p.Name
			elemType := strings.TrimPrefix(p.Type, "[]")
			info.cType = goSliceElemToCType(elemType, t.profile) + " *"
		} else if p.Type == "int" || p.Type == "int64" {
			// Int param → long pointer (GOAT convention)
			info.isInt = true
			info.cName = "p" + p.Name
			info.cType = "long *"
		} else if p.Type == "float32" || p.Type == "float64" {
			// Scalar float param → passed as pointer in GOAT
			info.isInt = true // reuse isInt to get pointer + dereference treatment
			info.cName = "p" + p.Name
			if p.Type == "float32" {
				info.cType = "float *"
			} else {
				info.cType = "double *"
			}
		} else {
			info.cName = p.Name
			info.cType = t.profile.CType
		}
		t.params[p.Name] = info
	}

	// For functions without explicit int size params (e.g. BaseBitProduct uses len(code)),
	// add a length parameter for the first slice. All slices are assumed same length.
	hasExplicitSize := false
	for _, p := range pf.Params {
		if p.Type == "int" || p.Type == "int64" {
			hasExplicitSize = true
			break
		}
	}
	if !hasExplicitSize {
		var firstSlice string
		for _, p := range pf.Params {
			if strings.HasPrefix(p.Type, "[]") {
				if firstSlice == "" {
					firstSlice = p.Name
				}
				// Map all slice params' len() to the same length variable
				t.sliceLenVars[p.Name] = "len_" + firstSlice
			}
		}
		if firstSlice != "" {
			lenCName := "plen_" + firstSlice
			lenVarName := "len_" + firstSlice
			info := cParamInfo{
				goName: lenVarName,
				goType: "int",
				cName:  lenCName,
				cType:  "long *",
				isInt:  true,
			}
			t.params["__len_"+firstSlice] = info
		}
	}

	// Handle return values as output pointers.
	// GOAT requires all pointer dereferences to be 64-bit (long/unsigned long),
	// so we always use "long *" for output pointers. The Go wrapper handles
	// narrowing (e.g., int64 → uint32).
	for _, ret := range pf.Returns {
		name := ret.Name
		if name == "" {
			name = "result"
		}
		info := cParamInfo{
			goName: name,
			goType: ret.Type,
			cName:  "pout_" + name,
			cType:  "long *",
			isInt:  false, // not dereferenced at top - handled by return stmt
		}
		t.params["__return_"+name] = info
	}
}

// goSliceElemToCType maps a Go slice element type to its C type equivalent.
func goSliceElemToCType(elemType string, profile *CIntrinsicProfile) string {
	switch elemType {
	case "float32":
		return "float"
	case "float64":
		return "double"
	case "uint64":
		return "unsigned long"
	case "uint32":
		return "unsigned int"
	case "uint8", "byte":
		return "unsigned char"
	case "int64":
		return "long"
	case "int32":
		return "int"
	case "T":
		return profile.CType
	default:
		return profile.CType
	}
}

// isScalarCType returns true for C scalar types that should be zero-initialized.
func isScalarCType(cType string) bool {
	switch cType {
	case "long", "int", "unsigned long", "unsigned int", "unsigned char",
		"float", "double", "short", "unsigned short":
		return true
	default:
		return false
	}
}

// emitFuncSignature emits: void funcname(float *a, float *b, ..., long *pm, long *pn, ...)
// If the Go function has return values, they are appended as output pointers.
func (t *CASTTranslator) emitFuncSignature(pf *ParsedFunc) {
	funcName := t.cFuncName(pf.Name)
	var params []string
	for _, p := range pf.Params {
		info := t.params[p.Name]
		if strings.HasSuffix(info.cType, "*") {
			params = append(params, info.cType+info.cName)
		} else {
			params = append(params, info.cType+" "+info.cName)
		}
	}
	// Append hidden length parameters (for functions without explicit int size params)
	for key, info := range t.params {
		if strings.HasPrefix(key, "__len_") {
			params = append(params, info.cType+info.cName)
		}
	}
	// Append output pointers for return values
	for _, ret := range pf.Returns {
		name := ret.Name
		if name == "" {
			name = "result"
		}
		info := t.params["__return_"+name]
		if strings.HasSuffix(info.cType, "*") {
			params = append(params, info.cType+info.cName)
		} else {
			params = append(params, info.cType+" "+info.cName)
		}
	}
	t.writef("void %s(%s) {\n", funcName, strings.Join(params, ", "))
}

// cFuncName builds the C function name from the Go function name.
// BaseMatMul → basematmul_c_f32_neon
func (t *CASTTranslator) cFuncName(baseName string) string {
	name := strings.ToLower(baseName)
	name = strings.TrimPrefix(name, "base")
	targetSuffix := strings.ToLower(t.profile.TargetName)
	return name + "_c_" + cTypeSuffix(t.elemType) + "_" + targetSuffix
}

// emitParamDerefs emits: long m = *pm; for each pointer-passed parameter.
func (t *CASTTranslator) emitParamDerefs() {
	for _, info := range sortedParams(t.params) {
		if !info.isInt {
			continue
		}
		// Skip return value output pointers
		if strings.HasPrefix(info.cName, "pout_") {
			continue
		}
		// Determine the dereferenced C type from the pointer type
		derefType := strings.TrimSuffix(strings.TrimSpace(info.cType), "*")
		derefType = strings.TrimSpace(derefType)
		t.writef("%s %s = *%s;\n", derefType, info.goName, info.cName)
		t.vars[info.goName] = cVarInfo{cType: derefType}
	}
}

// sortedParams returns params in stable order (by original param order in the function).
// We iterate over t.params, but need deterministic order. We reconstruct from params.
func sortedParams(params map[string]cParamInfo) []cParamInfo {
	// Since we need order, collect values sorted by goName which gives deterministic output.
	// In practice the caller should use the original pf.Params order, but since we
	// only have the map here, we need a stable approach.
	var result []cParamInfo
	for _, v := range params {
		result = append(result, v)
	}
	// Sort by cName for stable output
	for i := range result {
		for j := i + 1; j < len(result); j++ {
			if result[i].goName > result[j].goName {
				result[i], result[j] = result[j], result[i]
			}
		}
	}
	return result
}

// cDeclVar formats a C variable declaration, e.g. "float *x" or "long y".
// For pointer types ending in "*", the name is placed directly after the star.
func cDeclVar(cType, name string) string {
	if strings.HasSuffix(cType, "*") {
		return cType + name
	}
	return cType + " " + name
}

// ---------------------------------------------------------------------------
// Statement translators
// ---------------------------------------------------------------------------

// translateBlockStmtContents translates each statement in a block.
func (t *CASTTranslator) translateBlockStmtContents(block *ast.BlockStmt) {
	if block == nil {
		return
	}

	// Pre-scan for consecutive for-loops that share popcount accum variables.
	// When found, declare shared vector accumulators before the first loop
	// and reduce them after the last loop, avoiding intermediate reductions.
	var sharedAccums []deferredAccum
	firstLoopIdx, lastLoopIdx := -1, -1
	if t.profile.PopCountPartialFn[t.tier] != "" {
		sharedAccums, firstLoopIdx, lastLoopIdx = t.scanSharedPopCountLoops(block)
	}

	for i, stmt := range block.List {
		if len(sharedAccums) > 0 && i == firstLoopIdx {
			// Emit shared vector accumulators BEFORE the first loop
			for _, acc := range sharedAccums {
				t.writef("%s %s = vdupq_n_u32(0);\n",
					t.profile.AccVecType[t.tier], acc.accVar)
			}
			// Activate shared deferred accumulation
			t.deferredAccums = make(map[string]*deferredAccum, len(sharedAccums))
			for j := range sharedAccums {
				t.deferredAccums[sharedAccums[j].scalarVar] = &sharedAccums[j]
			}
		}

		t.translateStmt(stmt)

		if len(sharedAccums) > 0 && i == lastLoopIdx {
			// Emit finalization AFTER the last loop
			for _, acc := range sharedAccums {
				t.writef("%s += (unsigned long)(%s(%s));\n",
					acc.scalarVar, t.profile.AccReduceFn[t.tier], acc.accVar)
			}
			t.deferredAccums = nil
		}
	}
}

// scanSharedPopCountLoops finds consecutive for-loops in a block that share
// popcount accumulation variables. Returns the shared accumulators and the
// indices of the first and last for-loop in the run.
func (t *CASTTranslator) scanSharedPopCountLoops(block *ast.BlockStmt) (accums []deferredAccum, firstIdx, lastIdx int) {
	firstIdx, lastIdx = -1, -1

	// Collect all for-loops and their popcount accum variables
	type loopInfo struct {
		idx      int
		forStmt  *ast.ForStmt
		scalarVars map[string]bool
	}
	var loops []loopInfo
	for i, stmt := range block.List {
		fs, ok := stmt.(*ast.ForStmt)
		if !ok {
			continue
		}
		vars := t.scanPopCountScalarVars(fs.Body)
		if len(vars) == 0 {
			continue
		}
		loops = append(loops, loopInfo{idx: i, forStmt: fs, scalarVars: vars})
	}

	if len(loops) < 2 {
		return nil, -1, -1
	}

	// Find the longest run of consecutive loops that share at least one variable
	bestStart, bestEnd := 0, 0
	for start := 0; start < len(loops); start++ {
		shared := make(map[string]bool)
		for v := range loops[start].scalarVars {
			shared[v] = true
		}
		end := start
		for j := start + 1; j < len(loops); j++ {
			overlap := false
			for v := range loops[j].scalarVars {
				if shared[v] {
					overlap = true
					break
				}
			}
			if !overlap {
				break
			}
			// Merge variables
			for v := range loops[j].scalarVars {
				shared[v] = true
			}
			end = j
		}
		if end-start > bestEnd-bestStart {
			bestStart, bestEnd = start, end
		}
	}

	if bestStart == bestEnd {
		return nil, -1, -1 // no consecutive shared loops found
	}

	// Build unified accumulator list from the run
	seen := make(map[string]bool)
	for i := bestStart; i <= bestEnd; i++ {
		for _, stmt := range loops[i].forStmt.Body.List {
			assign, ok := stmt.(*ast.AssignStmt)
			if !ok {
				continue
			}
			scalarVar, ok := isPopCountAccumPattern(assign)
			if !ok || seen[scalarVar] {
				continue
			}
			seen[scalarVar] = true
			accums = append(accums, deferredAccum{
				scalarVar: scalarVar,
				accVar:    fmt.Sprintf("_pacc_%d", t.tmpCount),
			})
			t.tmpCount++
		}
	}

	return accums, loops[bestStart].idx, loops[bestEnd].idx
}

// translateStmt dispatches to the appropriate statement handler.
func (t *CASTTranslator) translateStmt(stmt ast.Stmt) {
	switch s := stmt.(type) {
	case *ast.AssignStmt:
		t.translateAssignStmt(s)
	case *ast.ForStmt:
		t.translateForStmt(s)
	case *ast.RangeStmt:
		t.translateRangeStmt(s)
	case *ast.ExprStmt:
		t.translateExprStmt(s)
	case *ast.DeclStmt:
		t.translateDeclStmt(s)
	case *ast.IfStmt:
		t.translateIfStmt(s)
	case *ast.BlockStmt:
		t.writef("{\n")
		t.indent++
		t.translateBlockStmtContents(s)
		t.indent--
		t.writef("}\n")
	case *ast.IncDecStmt:
		t.translateIncDecStmt(s)
	case *ast.ReturnStmt:
		t.translateReturnStmt(s)
	default:
		// Skip unsupported statements
	}
}

// translateAssignStmt handles := and = assignments.
func (t *CASTTranslator) translateAssignStmt(s *ast.AssignStmt) {
	// Handle 4-way multi-assign from hwy.Load4
	if len(s.Lhs) == 4 && len(s.Rhs) == 1 {
		if call, ok := s.Rhs[0].(*ast.CallExpr); ok {
			if sel := extractSelectorExpr(call.Fun); sel != nil {
				if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" && sel.Sel.Name == "Load4" {
					t.translateLoad4Assign(s.Lhs, call.Args, s.Tok)
					return
				}
			}
		}
	}

	if len(s.Lhs) != 1 || len(s.Rhs) != 1 {
		// Multi-assign not supported
		return
	}

	lhs := s.Lhs[0]
	rhs := s.Rhs[0]

	// Check for hwy.GetLane with variable index — requires store-to-stack pattern
	if call, ok := rhs.(*ast.CallExpr); ok {
		if sel := extractSelectorExpr(call.Fun); sel != nil {
			if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" && sel.Sel.Name == "GetLane" {
				if len(call.Args) >= 2 {
					if _, isLit := call.Args[1].(*ast.BasicLit); !isLit {
						lhsName := t.translateExpr(lhs)
						t.translateGetLaneVarIndex(lhsName, call.Args, s.Tok)
						return
					}
				}
			}
		}
	}

	lhsName := t.translateExpr(lhs)

	switch s.Tok {
	case token.DEFINE: // :=
		varInfo := t.inferType(rhs)
		rhsStr := t.translateExpr(rhs)

		// Handle make([]hwy.Vec[T], N) → stack-allocated C array
		if strings.HasPrefix(rhsStr, "/* VEC_ARRAY:") {
			// Parse: /* VEC_ARRAY:float32x4_t:4 */
			inner := strings.TrimPrefix(rhsStr, "/* VEC_ARRAY:")
			inner = strings.TrimSuffix(inner, " */")
			parts := strings.SplitN(inner, ":", 2)
			if len(parts) == 2 {
				vecType := parts[0]
				arrLen := parts[1]
				t.vars[lhsName] = cVarInfo{cType: vecType, isVector: true, isPtr: true}
				t.writef("%s %s[%s];\n", vecType, lhsName, arrLen)
				return
			}
		}

		t.vars[lhsName] = varInfo
		t.writef("%s = %s;\n", cDeclVar(varInfo.cType, lhsName), rhsStr)

	case token.ASSIGN: // =
		rhsStr := t.translateExpr(rhs)
		t.writef("%s = %s;\n", lhsName, rhsStr)

	case token.ADD_ASSIGN: // +=
		// Check for deferred popcount accumulation rewrite
		if t.deferredAccums != nil {
			if acc, ok := t.deferredAccums[lhsName]; ok {
				if andCall := extractPopCountAndExpr(rhs); andCall != nil {
					// Emit: _pacc_N = vaddq_u32(_pacc_N, neon_popcnt_u64_to_u32(vandq_u64(...)))
					andExpr := t.translateExpr(andCall)
					t.writef("%s = %s(%s, %s(%s));\n",
						acc.accVar,
						t.profile.AccAddFn[t.tier], acc.accVar,
						t.profile.PopCountPartialFn[t.tier], andExpr)
					return
				}
			}
		}
		rhsStr := t.translateExpr(rhs)
		t.writef("%s += %s;\n", lhsName, rhsStr)

	case token.SUB_ASSIGN: // -=
		rhsStr := t.translateExpr(rhs)
		t.writef("%s -= %s;\n", lhsName, rhsStr)

	case token.MUL_ASSIGN: // *=
		rhsStr := t.translateExpr(rhs)
		t.writef("%s *= %s;\n", lhsName, rhsStr)

	case token.OR_ASSIGN: // |=
		rhsStr := t.translateExpr(rhs)
		t.writef("%s |= %s;\n", lhsName, rhsStr)

	case token.AND_ASSIGN: // &=
		rhsStr := t.translateExpr(rhs)
		t.writef("%s &= %s;\n", lhsName, rhsStr)

	case token.SHL_ASSIGN: // <<=
		rhsStr := t.translateExpr(rhs)
		t.writef("%s <<= %s;\n", lhsName, rhsStr)

	case token.SHR_ASSIGN: // >>=
		rhsStr := t.translateExpr(rhs)
		t.writef("%s >>= %s;\n", lhsName, rhsStr)

	case token.XOR_ASSIGN: // ^=
		rhsStr := t.translateExpr(rhs)
		t.writef("%s ^= %s;\n", lhsName, rhsStr)
	}
}

// ---------------------------------------------------------------------------
// Deferred PopCount Accumulation
// ---------------------------------------------------------------------------

// isPopCountAccumPattern checks if an AssignStmt matches:
//
//	scalarVar += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(...))))
//
// or the variant without the uint64() cast.
// Returns the LHS scalar variable name and true if matched.
func isPopCountAccumPattern(s *ast.AssignStmt) (scalarVar string, ok bool) {
	if s.Tok != token.ADD_ASSIGN || len(s.Lhs) != 1 || len(s.Rhs) != 1 {
		return "", false
	}
	ident, ok := s.Lhs[0].(*ast.Ident)
	if !ok {
		return "", false
	}
	if extractPopCountAndExpr(s.Rhs[0]) != nil {
		return ident.Name, true
	}
	return "", false
}

// extractPopCountAndExpr unwraps the AST chain:
//
//	uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(a, b)))) → returns the And call
//	hwy.ReduceSum(hwy.PopCount(hwy.And(a, b)))          → returns the And call
//
// Returns nil if the expression doesn't match the pattern.
func extractPopCountAndExpr(expr ast.Expr) *ast.CallExpr {
	// Unwrap optional uint64() cast
	inner := expr
	if call, ok := inner.(*ast.CallExpr); ok {
		if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "uint64" {
			if len(call.Args) == 1 {
				inner = call.Args[0]
			}
		}
	}

	// Match hwy.ReduceSum(...)
	rsCall, ok := inner.(*ast.CallExpr)
	if !ok {
		return nil
	}
	rsSel := extractSelectorExpr(rsCall.Fun)
	if rsSel == nil {
		return nil
	}
	rsPkg, ok := rsSel.X.(*ast.Ident)
	if !ok || rsPkg.Name != "hwy" || rsSel.Sel.Name != "ReduceSum" {
		return nil
	}
	if len(rsCall.Args) != 1 {
		return nil
	}

	// Match hwy.PopCount(...)
	pcCall, ok := rsCall.Args[0].(*ast.CallExpr)
	if !ok {
		return nil
	}
	pcSel := extractSelectorExpr(pcCall.Fun)
	if pcSel == nil {
		return nil
	}
	pcPkg, ok := pcSel.X.(*ast.Ident)
	if !ok || pcPkg.Name != "hwy" || pcSel.Sel.Name != "PopCount" {
		return nil
	}
	if len(pcCall.Args) != 1 {
		return nil
	}

	// Match hwy.And(...) — the inner expression we want to keep
	andCall, ok := pcCall.Args[0].(*ast.CallExpr)
	if !ok {
		return nil
	}
	andSel := extractSelectorExpr(andCall.Fun)
	if andSel == nil {
		return nil
	}
	andPkg, ok := andSel.X.(*ast.Ident)
	if !ok || andPkg.Name != "hwy" || andSel.Sel.Name != "And" {
		return nil
	}

	return andCall
}

// scanPopCountScalarVars returns the set of scalar variable names used
// in popcount accumulation patterns within a for-loop body.
func (t *CASTTranslator) scanPopCountScalarVars(block *ast.BlockStmt) map[string]bool {
	if block == nil {
		return nil
	}
	vars := make(map[string]bool)
	for _, stmt := range block.List {
		assign, ok := stmt.(*ast.AssignStmt)
		if !ok {
			continue
		}
		if scalarVar, ok := isPopCountAccumPattern(assign); ok {
			vars[scalarVar] = true
		}
	}
	if len(vars) == 0 {
		return nil
	}
	return vars
}

// scanForPopCountAccumPattern scans a for-loop body for statements matching
// the popcount accumulation pattern and returns accumulator descriptors.
// The pattern (ReduceSum(PopCount(And(...)))) is specific enough that no
// additional gating (e.g. Load4 presence) is needed.
func (t *CASTTranslator) scanForPopCountAccumPattern(block *ast.BlockStmt) []deferredAccum {
	if block == nil {
		return nil
	}

	var accums []deferredAccum
	seen := make(map[string]bool)
	for _, stmt := range block.List {
		assign, ok := stmt.(*ast.AssignStmt)
		if !ok {
			continue
		}
		scalarVar, ok := isPopCountAccumPattern(assign)
		if !ok || seen[scalarVar] {
			continue
		}
		seen[scalarVar] = true
		accums = append(accums, deferredAccum{
			scalarVar: scalarVar,
			accVar:    fmt.Sprintf("_pacc_%d", t.tmpCount),
		})
		t.tmpCount++
	}
	return accums
}

// translateForStmt handles C-style for loops.
func (t *CASTTranslator) translateForStmt(s *ast.ForStmt) {
	initStr := ""
	condStr := ""
	postStr := ""

	// Init
	if s.Init != nil {
		initStr = t.translateForInit(s.Init)
	}

	// Condition
	if s.Cond != nil {
		condStr = t.translateExpr(s.Cond)
	}

	// Post
	if s.Post != nil {
		postStr = t.translateForPost(s.Post)
	}

	// Deferred popcount accumulation. If deferredAccums is already set,
	// a parent block is managing the lifecycle (cross-loop sharing).
	// Otherwise, this loop manages its own accumulators.
	externalAccums := t.deferredAccums != nil
	if !externalAccums && t.profile.PopCountPartialFn[t.tier] != "" {
		accums := t.scanForPopCountAccumPattern(s.Body)
		if len(accums) > 0 {
			// Emit vector accumulators BEFORE the loop
			for _, acc := range accums {
				t.writef("%s %s = vdupq_n_u32(0);\n",
					t.profile.AccVecType[t.tier], acc.accVar)
			}
			// Activate deferred accumulation for the loop body
			t.deferredAccums = make(map[string]*deferredAccum, len(accums))
			for i := range accums {
				t.deferredAccums[accums[i].scalarVar] = &accums[i]
			}
		}
	}

	// Prevent clang from auto-vectorizing scalar loops into NEON code
	// with constant pool references (adrp+ldr from .rodata), which GOAT
	// cannot relocate in Go assembly.
	t.writef("#pragma clang loop vectorize(disable) interleave(disable)\n")
	t.writef("for (%s; %s; %s) {\n", initStr, condStr, postStr)
	t.indent++
	t.translateBlockStmtContents(s.Body)
	t.indent--
	t.writef("}\n")

	// Only reduce and clear if this loop owns the accumulators
	if !externalAccums && t.deferredAccums != nil {
		for _, acc := range t.deferredAccumsOrdered() {
			t.writef("%s += (unsigned long)(%s(%s));\n",
				acc.scalarVar, t.profile.AccReduceFn[t.tier], acc.accVar)
		}
		t.deferredAccums = nil
	}
}

// translateForInit translates a for-loop init statement.
func (t *CASTTranslator) translateForInit(stmt ast.Stmt) string {
	switch s := stmt.(type) {
	case *ast.AssignStmt:
		if len(s.Lhs) == 1 && len(s.Rhs) == 1 {
			lhs := t.translateExpr(s.Lhs[0])
			rhs := t.translateExpr(s.Rhs[0])
			if s.Tok == token.DEFINE {
				// Declare variable with type
				varInfo := t.inferType(s.Rhs[0])
				t.vars[lhs] = varInfo
				return fmt.Sprintf("%s = %s", cDeclVar(varInfo.cType, lhs), rhs)
			}
			return fmt.Sprintf("%s = %s", lhs, rhs)
		}
	}
	return ""
}

// translateForPost translates a for-loop post statement.
func (t *CASTTranslator) translateForPost(stmt ast.Stmt) string {
	switch s := stmt.(type) {
	case *ast.AssignStmt:
		if len(s.Lhs) == 1 && len(s.Rhs) == 1 {
			lhs := t.translateExpr(s.Lhs[0])
			rhs := t.translateExpr(s.Rhs[0])
			switch s.Tok {
			case token.ADD_ASSIGN:
				return fmt.Sprintf("%s += %s", lhs, rhs)
			case token.ASSIGN:
				return fmt.Sprintf("%s = %s", lhs, rhs)
			}
		}
	case *ast.IncDecStmt:
		name := t.translateExpr(s.X)
		if s.Tok == token.INC {
			return name + "++"
		}
		return name + "--"
	}
	return ""
}

// translateRangeStmt translates `for i := range m` to `for (long i = 0; i < m; i++)`.
func (t *CASTTranslator) translateRangeStmt(s *ast.RangeStmt) {
	// `for i := range m` → key is i, X is m
	if s.Key == nil {
		return
	}

	iter := t.translateExpr(s.Key)
	rangeOver := t.translateExpr(s.X)

	// Register the iterator variable
	t.vars[iter] = cVarInfo{cType: "long"}

	// Prevent clang from auto-vectorizing scalar loops into NEON code
	// with constant pool references (adrp+ldr from .rodata), which GOAT
	// cannot relocate in Go assembly.
	t.writef("#pragma clang loop vectorize(disable) interleave(disable)\n")
	t.writef("for (long %s = 0; %s < %s; %s++) {\n", iter, iter, rangeOver, iter)
	t.indent++
	t.translateBlockStmtContents(s.Body)
	t.indent--
	t.writef("}\n")
}

// translateExprStmt handles standalone expression statements (function calls).
func (t *CASTTranslator) translateExprStmt(s *ast.ExprStmt) {
	call, ok := s.X.(*ast.CallExpr)
	if !ok {
		return
	}

	// Check for panic() calls - skip them
	if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "panic" {
		return
	}

	// Check for hwy.Store calls
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
			if sel.Sel.Name == "Store" {
				t.emitHwyStore(call.Args)
				return
			}
		}
	}

	// Generic function call
	expr := t.translateExpr(s.X)
	t.writef("%s;\n", expr)
}

// translateDeclStmt handles `var j int`.
func (t *CASTTranslator) translateDeclStmt(s *ast.DeclStmt) {
	genDecl, ok := s.Decl.(*ast.GenDecl)
	if !ok || genDecl.Tok != token.VAR {
		return
	}

	for _, spec := range genDecl.Specs {
		valueSpec, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}

		cType := t.goTypeToCType(exprToString(valueSpec.Type))
		for _, name := range valueSpec.Names {
			t.vars[name.Name] = cVarInfo{cType: cType}
			// Go zero-initializes all declared variables; emit = 0 for scalar types
			if isScalarCType(cType) {
				t.writef("%s = 0;\n", cDeclVar(cType, name.Name))
			} else {
				t.writef("%s;\n", cDeclVar(cType, name.Name))
			}
		}
	}
}

// translateIfStmt handles if statements. Skips panic guards (bounds checks).
func (t *CASTTranslator) translateIfStmt(s *ast.IfStmt) {
	// Skip if the body contains only a panic call (bounds check)
	if isPanicGuard(s) {
		return
	}

	condStr := t.translateExpr(s.Cond)
	t.writef("if (%s) {\n", condStr)
	t.indent++
	t.translateBlockStmtContents(s.Body)
	t.indent--
	if s.Else != nil {
		t.writef("} else ")
		// Don't add newline - let the else clause handle it
		switch e := s.Else.(type) {
		case *ast.BlockStmt:
			t.writefRaw("{\n")
			t.indent++
			t.translateBlockStmtContents(e)
			t.indent--
			t.writef("}\n")
		case *ast.IfStmt:
			// else if
			t.translateIfStmt(e)
		}
	} else {
		t.writef("}\n")
	}
}

// translateIncDecStmt handles i++ and i--.
func (t *CASTTranslator) translateIncDecStmt(s *ast.IncDecStmt) {
	name := t.translateExpr(s.X)
	if s.Tok == token.INC {
		t.writef("%s++;\n", name)
	} else {
		t.writef("%s--;\n", name)
	}
}

// translateReturnStmt handles `return expr` → `*pout_name = expr; return;`
func (t *CASTTranslator) translateReturnStmt(s *ast.ReturnStmt) {
	if len(s.Results) == 0 {
		t.writef("return;\n")
		return
	}

	// Collect return parameter names from the params map
	var retNames []string
	for key := range t.params {
		if strings.HasPrefix(key, "__return_") {
			retNames = append(retNames, key)
		}
	}
	// Sort for stable output
	for i := range retNames {
		for j := i + 1; j < len(retNames); j++ {
			if retNames[i] > retNames[j] {
				retNames[i], retNames[j] = retNames[j], retNames[i]
			}
		}
	}

	for i, result := range s.Results {
		expr := t.translateExpr(result)
		if i < len(retNames) {
			info := t.params[retNames[i]]
			t.writef("*%s = %s;\n", info.cName, expr)
		}
	}
	t.writef("return;\n")
}

// isPanicGuard returns true if the if statement's body only contains panic().
func isPanicGuard(s *ast.IfStmt) bool {
	if s.Body == nil || len(s.Body.List) == 0 {
		return false
	}
	for _, stmt := range s.Body.List {
		exprStmt, ok := stmt.(*ast.ExprStmt)
		if !ok {
			return false
		}
		call, ok := exprStmt.X.(*ast.CallExpr)
		if !ok {
			return false
		}
		ident, ok := call.Fun.(*ast.Ident)
		if !ok || ident.Name != "panic" {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// Expression translators
// ---------------------------------------------------------------------------

// translateExpr converts a Go AST expression to a C expression string.
func (t *CASTTranslator) translateExpr(expr ast.Expr) string {
	if expr == nil {
		return ""
	}

	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.BasicLit:
		return t.translateBasicLit(e)
	case *ast.BinaryExpr:
		return t.translateBinaryExpr(e)
	case *ast.CallExpr:
		return t.translateCallExpr(e)
	case *ast.IndexExpr:
		return t.translateIndexExpr(e)
	case *ast.SliceExpr:
		return t.translateSliceExpr(e)
	case *ast.SelectorExpr:
		return t.translateSelectorExpr(e)
	case *ast.ParenExpr:
		return "(" + t.translateExpr(e.X) + ")"
	case *ast.UnaryExpr:
		return t.translateUnaryExpr(e)
	case *ast.StarExpr:
		return "*" + t.translateExpr(e.X)
	default:
		return exprToString(expr)
	}
}

// translateBasicLit translates a literal (int, float, string).
func (t *CASTTranslator) translateBasicLit(lit *ast.BasicLit) string {
	switch lit.Kind {
	case token.INT:
		return lit.Value
	case token.FLOAT:
		if t.elemType == "float32" {
			return lit.Value + "f"
		}
		return lit.Value
	case token.CHAR:
		return lit.Value
	default:
		return lit.Value
	}
}

// translateBinaryExpr translates binary operations.
func (t *CASTTranslator) translateBinaryExpr(e *ast.BinaryExpr) string {
	left := t.translateExpr(e.X)
	right := t.translateExpr(e.Y)
	return left + " " + e.Op.String() + " " + right
}

// translateCallExpr translates function calls, dispatching hwy.* calls to intrinsics.
func (t *CASTTranslator) translateCallExpr(e *ast.CallExpr) string {
	// Check for hwy.Func(...) or hwy.Func[T](...)
	if sel := extractSelectorExpr(e.Fun); sel != nil {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
			return t.translateHwyCall(sel.Sel.Name, e.Args)
		}
	}

	// Check for bits.OnesCount* → __builtin_popcount*
	if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "bits" {
			if builtinFn := bitsOnesCountToBuiltin(sel.Sel.Name); builtinFn != "" {
				if len(e.Args) == 1 {
					arg := t.translateExpr(e.Args[0])
					return fmt.Sprintf("%s(%s)", builtinFn, arg)
				}
			}
		}
	}

	// Check for math.Float32bits → float_to_bits, math.Float32frombits → bits_to_float
	if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "math" {
			switch sel.Sel.Name {
			case "Float32bits":
				if len(e.Args) == 1 {
					arg := t.translateExpr(e.Args[0])
					return fmt.Sprintf("float_to_bits(%s)", arg)
				}
			case "Float32frombits":
				if len(e.Args) == 1 {
					arg := t.translateExpr(e.Args[0])
					return fmt.Sprintf("bits_to_float(%s)", arg)
				}
			}
		}
	}

	// Check for v.NumLanes() method calls
	if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
		if sel.Sel.Name == "NumLanes" || sel.Sel.Name == "NumElements" {
			return fmt.Sprintf("%d", t.lanes)
		}
	}

	// Check for len() calls → use the length parameter variable if known,
	// otherwise emit a comment placeholder.
	if ident, ok := e.Fun.(*ast.Ident); ok && ident.Name == "len" {
		if len(e.Args) == 1 {
			// Get the raw name of the slice argument
			if argIdent, ok := e.Args[0].(*ast.Ident); ok {
				if lenVar, ok := t.sliceLenVars[argIdent.Name]; ok {
					return lenVar
				}
			}
			arg := t.translateExpr(e.Args[0])
			return fmt.Sprintf("/* len(%s) */", arg)
		}
	}

	// Check for min() calls → ternary or built-in
	if ident, ok := e.Fun.(*ast.Ident); ok && ident.Name == "min" {
		if len(e.Args) == 2 {
			a := t.translateExpr(e.Args[0])
			b := t.translateExpr(e.Args[1])
			return fmt.Sprintf("((%s) < (%s) ? (%s) : (%s))", a, b, a, b)
		}
	}

	// Check for make() calls → stack-allocated C arrays
	if ident, ok := e.Fun.(*ast.Ident); ok && ident.Name == "make" {
		return t.translateMakeExpr(e)
	}

	// Check for getSignBit(x) → (float_to_bits(x) >> 31)
	if ident, ok := e.Fun.(*ast.Ident); ok && ident.Name == "getSignBit" {
		if len(e.Args) == 1 {
			arg := t.translateExpr(e.Args[0])
			return fmt.Sprintf("(float_to_bits(%s) >> 31)", arg)
		}
	}

	// Check for type conversions: uint64(x), float64(x), uint32(x), int(x), etc.
	if ident, ok := e.Fun.(*ast.Ident); ok {
		if cType := t.goTypeConvToCType(ident.Name); cType != "" {
			if len(e.Args) == 1 {
				arg := t.translateExpr(e.Args[0])
				return fmt.Sprintf("(%s)(%s)", cType, arg)
			}
		}
	}

	// Generic call - just translate arguments
	fun := t.translateExpr(e.Fun)
	var args []string
	for _, arg := range e.Args {
		args = append(args, t.translateExpr(arg))
	}
	return fmt.Sprintf("%s(%s)", fun, strings.Join(args, ", "))
}

// extractSelectorExpr extracts the SelectorExpr from a call expression's Fun,
// handling both direct calls (hwy.Load) and generic calls (hwy.Zero[T]).
func extractSelectorExpr(fun ast.Expr) *ast.SelectorExpr {
	switch f := fun.(type) {
	case *ast.SelectorExpr:
		return f
	case *ast.IndexExpr:
		// hwy.Zero[T]() → IndexExpr{X: SelectorExpr{hwy, Zero}, Index: T}
		if sel, ok := f.X.(*ast.SelectorExpr); ok {
			return sel
		}
	case *ast.IndexListExpr:
		if sel, ok := f.X.(*ast.SelectorExpr); ok {
			return sel
		}
	}
	return nil
}

// translateIndexExpr translates array indexing: a[i*k+p].
func (t *CASTTranslator) translateIndexExpr(e *ast.IndexExpr) string {
	x := t.translateExpr(e.X)
	idx := t.translateExpr(e.Index)
	return fmt.Sprintf("%s[%s]", x, idx)
}

// translateSliceExpr translates slice expressions to C pointer arithmetic.
// c[i*n : (i+1)*n] → c + i*n   (as a pointer alias)
// bRow[j:]          → bRow + j  (as a pointer argument)
func (t *CASTTranslator) translateSliceExpr(e *ast.SliceExpr) string {
	x := t.translateExpr(e.X)

	if e.Low != nil {
		low := t.translateExpr(e.Low)
		return x + " + " + low
	}
	return x
}

// translateSelectorExpr translates field access / method references.
func (t *CASTTranslator) translateSelectorExpr(e *ast.SelectorExpr) string {
	x := t.translateExpr(e.X)
	return x + "." + e.Sel.Name
}

// translateUnaryExpr translates unary expressions.
func (t *CASTTranslator) translateUnaryExpr(e *ast.UnaryExpr) string {
	operand := t.translateExpr(e.X)
	return e.Op.String() + operand
}

// ---------------------------------------------------------------------------
// hwy.* call mapping
// ---------------------------------------------------------------------------

// translateHwyCall maps hwy.FuncName to the appropriate C intrinsic.
func (t *CASTTranslator) translateHwyCall(funcName string, args []ast.Expr) string {
	switch funcName {
	case "Load":
		return t.emitHwyLoad(args)
	case "Store":
		// Store is typically a statement, not expression — but handle both
		return t.emitHwyStoreExpr(args)
	case "Set":
		return t.emitHwySet(args)
	case "Zero":
		return t.emitHwyZero()
	case "MulAdd":
		return t.emitHwyMulAdd(args)
	case "Add":
		return t.emitHwyBinaryOp(t.profile.AddFn, args)
	case "Sub":
		return t.emitHwyBinaryOp(t.profile.SubFn, args)
	case "Mul":
		return t.emitHwyBinaryOp(t.profile.MulFn, args)
	case "Div":
		return t.emitHwyBinaryOp(t.profile.DivFn, args)
	case "Min":
		return t.emitHwyBinaryOp(t.profile.MinFn, args)
	case "Max":
		return t.emitHwyBinaryOp(t.profile.MaxFn, args)
	case "Neg":
		return t.emitHwyUnaryOp(t.profile.NegFn, args)
	case "Abs":
		return t.emitHwyUnaryOp(t.profile.AbsFn, args)
	case "Sqrt":
		return t.emitHwyUnaryOp(t.profile.SqrtFn, args)
	case "ReduceSum":
		return t.emitHwyReduceSum(args)
	case "InterleaveLower":
		return t.emitHwyBinaryOp(t.profile.InterleaveLowerFn, args)
	case "InterleaveUpper":
		return t.emitHwyBinaryOp(t.profile.InterleaveUpperFn, args)
	case "And":
		return t.emitHwyBinaryOp(t.profile.AndFn, args)
	case "Or":
		return t.emitHwyBinaryOp(t.profile.OrFn, args)
	case "Xor":
		return t.emitHwyBinaryOp(t.profile.XorFn, args)
	case "PopCount":
		return t.emitHwyUnaryOp(t.profile.PopCountFn, args)
	case "LessThan":
		return t.emitHwyBinaryOp(t.profile.LessThanFn, args)
	case "IfThenElse":
		return t.emitHwyIfThenElse(args)
	case "BitsFromMask":
		return t.emitHwyUnaryOp(t.profile.BitsFromMaskFn, args)
	case "TableLookupBytes":
		return t.emitHwyBinaryOp(t.profile.TableLookupBytesFn, args)
	case "Load4":
		// Load4 is typically handled as a multi-assign statement; if it appears
		// as an expression, treat it like a single load (the multi-value handling
		// is in translateAssignStmt).
		return t.emitHwyLoad(args)
	case "LoadSlice":
		return t.emitHwyLoad(args) // same semantics as Load for C
	case "StoreSlice":
		return t.emitHwyStoreExpr(args) // same semantics as Store for C
	case "MaxLanes":
		return fmt.Sprintf("%d", t.lanes)
	case "GetLane":
		return t.emitHwyGetLane(args)
	default:
		// Unknown hwy call — emit as-is
		var argStrs []string
		for _, a := range args {
			argStrs = append(argStrs, t.translateExpr(a))
		}
		return fmt.Sprintf("hwy_%s(%s)", strings.ToLower(funcName), strings.Join(argStrs, ", "))
	}
}

// emitHwyLoad: hwy.Load(slice[off:]) → vld1q_f32(ptr + off)
func (t *CASTTranslator) emitHwyLoad(args []ast.Expr) string {
	if len(args) < 1 {
		return "/* Load: missing args */"
	}
	loadFn := t.profile.LoadFn[t.tier]
	ptr := t.translateExpr(args[0])
	return fmt.Sprintf("%s(%s)", loadFn, ptr)
}

// emitHwyStore: hwy.Store(vec, slice[off:]) → vst1q_f32(ptr + off, vec)
func (t *CASTTranslator) emitHwyStore(args []ast.Expr) {
	if len(args) < 2 {
		t.writef("/* Store: missing args */\n")
		return
	}
	storeFn := t.profile.StoreFn[t.tier]
	vec := t.translateExpr(args[0])
	ptr := t.translateExpr(args[1])
	// NEON Store: vst1q_f32(ptr, vec) — pointer first, vector second
	// AVX Store: _mm256_storeu_ps(ptr, vec) — same convention
	t.writef("%s(%s, %s);\n", storeFn, ptr, vec)
}

// emitHwyStoreExpr returns Store as a string expression (for edge cases).
func (t *CASTTranslator) emitHwyStoreExpr(args []ast.Expr) string {
	if len(args) < 2 {
		return "/* Store: missing args */"
	}
	storeFn := t.profile.StoreFn[t.tier]
	vec := t.translateExpr(args[0])
	ptr := t.translateExpr(args[1])
	return fmt.Sprintf("%s(%s, %s)", storeFn, ptr, vec)
}

// emitHwySet: hwy.Set(val) → vdupq_n_f32(val)
func (t *CASTTranslator) emitHwySet(args []ast.Expr) string {
	if len(args) < 1 {
		return "/* Set: missing args */"
	}
	dupFn := t.profile.DupFn[t.tier]
	val := t.translateExpr(args[0])
	return fmt.Sprintf("%s(%s)", dupFn, val)
}

// emitHwyZero: hwy.Zero[T]() → vdupq_n_f32(0.0f) or vdupq_n_u64(0) for integers
func (t *CASTTranslator) emitHwyZero() string {
	dupFn := t.profile.DupFn[t.tier]
	var zero string
	switch t.elemType {
	case "float64":
		zero = "0.0"
	case "float32":
		zero = "0.0f"
	default:
		// Integer types: use plain 0
		zero = "0"
	}
	return fmt.Sprintf("%s(%s)", dupFn, zero)
}

// emitHwyMulAdd: hwy.MulAdd(a, b, acc) → target-specific FMA.
// Go convention (like AVX): MulAdd(a, b, acc) = a*b + acc
// NEON: vfmaq_f32(acc, a, b) — accumulator first
// AVX: _mm256_fmadd_ps(a, b, acc) — accumulator last
func (t *CASTTranslator) emitHwyMulAdd(args []ast.Expr) string {
	if len(args) < 3 {
		return "/* MulAdd: missing args */"
	}
	fmaFn := t.profile.FmaFn[t.tier]
	a := t.translateExpr(args[0])
	b := t.translateExpr(args[1])
	acc := t.translateExpr(args[2])

	if t.profile.FmaArgOrder == "acc_first" {
		// NEON: FMA(acc, a, b)
		return fmt.Sprintf("%s(%s, %s, %s)", fmaFn, acc, a, b)
	}
	// AVX: FMA(a, b, acc)
	return fmt.Sprintf("%s(%s, %s, %s)", fmaFn, a, b, acc)
}

// emitHwyBinaryOp: hwy.Add(a, b) → vaddq_f32(a, b)
func (t *CASTTranslator) emitHwyBinaryOp(fnMap map[string]string, args []ast.Expr) string {
	if len(args) < 2 {
		return "/* binary op: missing args */"
	}
	fn := fnMap[t.tier]
	a := t.translateExpr(args[0])
	b := t.translateExpr(args[1])
	return fmt.Sprintf("%s(%s, %s)", fn, a, b)
}

// emitHwyUnaryOp: hwy.Neg(x) → vnegq_f32(x)
func (t *CASTTranslator) emitHwyUnaryOp(fnMap map[string]string, args []ast.Expr) string {
	if len(args) < 1 {
		return "/* unary op: missing args */"
	}
	fn := fnMap[t.tier]
	x := t.translateExpr(args[0])
	return fmt.Sprintf("%s(%s)", fn, x)
}

// emitHwyReduceSum: hwy.ReduceSum(v) → vaddvq_f32(v) (returns scalar, not vector)
func (t *CASTTranslator) emitHwyReduceSum(args []ast.Expr) string {
	if len(args) < 1 {
		return "/* ReduceSum: missing args */"
	}
	fn := t.profile.ReduceSumFn[t.tier]
	v := t.translateExpr(args[0])
	return fmt.Sprintf("%s(%s)", fn, v)
}

// emitHwyIfThenElse: hwy.IfThenElse(mask, yes, no) → target-specific select.
// NEON: vbslq_f32(mask, yes, no) — mask first
// AVX: _mm256_blendv_ps(no, yes, mask) — mask last, false first
func (t *CASTTranslator) emitHwyIfThenElse(args []ast.Expr) string {
	if len(args) < 3 {
		return "/* IfThenElse: missing args */"
	}
	fn := t.profile.IfThenElseFn[t.tier]
	mask := t.translateExpr(args[0])
	yes := t.translateExpr(args[1])
	no := t.translateExpr(args[2])

	if t.profile.FmaArgOrder == "acc_last" {
		// AVX convention: blendv(no, yes, mask)
		return fmt.Sprintf("%s(%s, %s, %s)", fn, no, yes, mask)
	}
	// NEON convention: vbsl(mask, yes, no)
	return fmt.Sprintf("%s(%s, %s, %s)", fn, mask, yes, no)
}

// emitHwyGetLane: hwy.GetLane(v, idx) → vgetq_lane_f32(v, idx)
// For variable (non-literal) indices, emits a store-to-stack pattern.
func (t *CASTTranslator) emitHwyGetLane(args []ast.Expr) string {
	if len(args) < 2 {
		return "/* GetLane: missing args */"
	}
	fn := t.profile.GetLaneFn[t.tier]
	v := t.translateExpr(args[0])
	idx := t.translateExpr(args[1])
	return fmt.Sprintf("%s(%s, %s)", fn, v, idx)
}

// translateLoad4Assign handles: a, b, c, d := hwy.Load4(slice[off:])
// On NEON (VecX4Type populated): emits vld1q_u64_x4 + .val[i] destructuring.
// On AVX (VecX4Type nil): emits 4 individual loads with ptr + i*lanes offsets.
func (t *CASTTranslator) translateLoad4Assign(lhs []ast.Expr, args []ast.Expr, tok token.Token) {
	if len(args) < 1 {
		t.writef("/* Load4: missing args */\n")
		return
	}
	ptr := t.translateExpr(args[0])
	vecType := t.profile.VecTypes[t.tier]

	// Get LHS variable names
	var names [4]string
	for i := 0; i < 4; i++ {
		names[i] = t.translateExpr(lhs[i])
	}

	if x4Type, ok := t.profile.VecX4Type[t.tier]; ok && x4Type != "" {
		// NEON path: use vld1q_*_x4 multi-load
		load4Fn := t.profile.Load4Fn[t.tier]
		tmpName := fmt.Sprintf("_load4_%d", t.tmpCount)
		t.tmpCount++
		t.writef("%s %s = %s(%s);\n", x4Type, tmpName, load4Fn, ptr)
		for i := 0; i < 4; i++ {
			if tok == token.DEFINE {
				t.vars[names[i]] = cVarInfo{cType: vecType, isVector: true}
				t.writef("%s %s = %s.val[%d];\n", vecType, names[i], tmpName, i)
			} else {
				t.writef("%s = %s.val[%d];\n", names[i], tmpName, i)
			}
		}
	} else {
		// AVX fallback: 4 individual loads
		loadFn := t.profile.LoadFn[t.tier]
		for i := 0; i < 4; i++ {
			var loadExpr string
			if i == 0 {
				loadExpr = fmt.Sprintf("%s(%s)", loadFn, ptr)
			} else {
				loadExpr = fmt.Sprintf("%s(%s + %d)", loadFn, ptr, i*t.lanes)
			}
			if tok == token.DEFINE {
				t.vars[names[i]] = cVarInfo{cType: vecType, isVector: true}
				t.writef("%s %s = %s;\n", vecType, names[i], loadExpr)
			} else {
				t.writef("%s = %s;\n", names[i], loadExpr)
			}
		}
	}
}

// translateGetLaneVarIndex emits the store-to-stack pattern for variable-index GetLane.
// Produces:
//
//	volatile float _getlane_buf[4];
//	vst1q_f32((float *)_getlane_buf, vecData);
//	float element = _getlane_buf[j];
func (t *CASTTranslator) translateGetLaneVarIndex(lhsName string, args []ast.Expr, tok token.Token) {
	if len(args) < 2 {
		t.writef("/* GetLane var index: missing args */\n")
		return
	}
	vec := t.translateExpr(args[0])
	idx := t.translateExpr(args[1])
	storeFn := t.profile.StoreFn[t.tier]
	cType := t.profile.CType

	t.writef("volatile %s _getlane_buf[%d];\n", cType, t.lanes)
	t.writef("%s((%s *)_getlane_buf, %s);\n", storeFn, cType, vec)

	varInfo := cVarInfo{cType: cType}
	if tok == token.DEFINE {
		t.vars[lhsName] = varInfo
		t.writef("%s %s = _getlane_buf[%s];\n", cType, lhsName, idx)
	} else {
		t.writef("%s = _getlane_buf[%s];\n", lhsName, idx)
	}
}

// translateMakeExpr handles make([]hwy.Vec[T], N) → declares a C stack array.
// Returns a placeholder expression; the variable name is set by the caller's := assignment.
func (t *CASTTranslator) translateMakeExpr(e *ast.CallExpr) string {
	if len(e.Args) < 2 {
		return "/* make: insufficient args */"
	}

	// Second arg is the length
	length := t.translateExpr(e.Args[1])

	// Check if the first arg is a slice of hwy.Vec[T]
	typeStr := exprToString(e.Args[0])
	if strings.Contains(typeStr, "hwy.Vec") || strings.Contains(typeStr, "Vec[") {
		// This will be used with := assignment. The CASTTranslator will see this
		// is a make call and handle it specially via inferType.
		// Return a special token that the assignment handler uses.
		return fmt.Sprintf("/* VEC_ARRAY:%s:%s */", t.profile.VecTypes[t.tier], length)
	}

	return fmt.Sprintf("/* make: unsupported type %s */", typeStr)
}

// goTypeConvToCType returns the C type for a Go type conversion function name,
// or empty string if it's not a type conversion.
func (t *CASTTranslator) goTypeConvToCType(name string) string {
	switch name {
	case "uint64":
		return "unsigned long"
	case "uint32":
		return "unsigned int"
	case "uint8", "byte":
		return "unsigned char"
	case "int64":
		return "long"
	case "int32":
		return "int"
	case "int":
		return "long"
	case "uint":
		return "unsigned long"
	case "float32":
		return "float"
	case "float64":
		return "double"
	default:
		return ""
	}
}

// bitsOnesCountToBuiltin maps Go math/bits popcount functions to GCC builtins.
func bitsOnesCountToBuiltin(funcName string) string {
	switch funcName {
	case "OnesCount64":
		return "__builtin_popcountll"
	case "OnesCount32":
		return "__builtin_popcount"
	case "OnesCount16":
		return "__builtin_popcount"
	case "OnesCount8":
		return "__builtin_popcount"
	case "OnesCount":
		return "__builtin_popcountll"
	default:
		return ""
	}
}

// ---------------------------------------------------------------------------
// Type inference
// ---------------------------------------------------------------------------

// inferType infers the C type from the RHS of an assignment.
func (t *CASTTranslator) inferType(expr ast.Expr) cVarInfo {
	switch e := expr.(type) {
	case *ast.CallExpr:
		return t.inferCallType(e)
	case *ast.SliceExpr:
		// Infer pointer type from the base expression (e.g., codes[i*w:(i+1)*w]
		// where codes is unsigned long * should yield unsigned long *, not float *).
		if baseType := t.inferPtrType(e.X); baseType != "" {
			return cVarInfo{cType: baseType, isPtr: true}
		}
		return cVarInfo{cType: t.profile.CType + " *", isPtr: true}
	case *ast.IndexExpr:
		// Infer element type from the base expression (e.g., codes[i]
		// where codes is unsigned long * should yield unsigned long).
		if baseType := t.inferPtrType(e.X); baseType != "" {
			elemType := strings.TrimSuffix(strings.TrimSpace(baseType), "*")
			return cVarInfo{cType: strings.TrimSpace(elemType)}
		}
		return cVarInfo{cType: t.profile.CType}
	case *ast.BinaryExpr:
		// Infer from left operand to propagate type through expressions
		left := t.inferType(e.X)
		return left
	case *ast.BasicLit:
		if e.Kind == token.INT {
			return cVarInfo{cType: "long"}
		}
		return cVarInfo{cType: t.profile.CType}
	case *ast.Ident:
		// Variable reference — look up its type
		if info, ok := t.vars[e.Name]; ok {
			return info
		}
		if info, ok := t.params[e.Name]; ok {
			if info.isSlice {
				return cVarInfo{cType: info.cType, isPtr: true}
			}
			return cVarInfo{cType: t.profile.CType}
		}
		return cVarInfo{cType: t.profile.CType}
	default:
		return cVarInfo{cType: t.profile.CType}
	}
}

// inferPtrType returns the C pointer type for an expression that represents
// a slice or pointer, or "" if the type cannot be determined from context.
// This is used to correctly type slice expressions and index expressions
// when the base has a different element type than the profile's CType
// (e.g., []uint64 param in a float32 profile).
func (t *CASTTranslator) inferPtrType(expr ast.Expr) string {
	ident, ok := expr.(*ast.Ident)
	if !ok {
		return ""
	}
	if info, ok := t.vars[ident.Name]; ok && info.isPtr {
		return info.cType
	}
	if info, ok := t.params[ident.Name]; ok && info.isSlice {
		return info.cType
	}
	return ""
}

// inferCallType infers the return type of a function call.
func (t *CASTTranslator) inferCallType(e *ast.CallExpr) cVarInfo {
	vecType := t.profile.VecTypes[t.tier]

	// Check for hwy.Func calls
	if sel := extractSelectorExpr(e.Fun); sel != nil {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
			switch sel.Sel.Name {
			case "Load", "Load4", "Zero", "Set", "MulAdd", "Add", "Sub", "Mul", "Div",
				"Min", "Max", "Neg", "Abs", "Sqrt",
				"LoadSlice", "InterleaveLower", "InterleaveUpper",
				"And", "Or", "Xor", "PopCount", "TableLookupBytes":
				return cVarInfo{cType: vecType, isVector: true}
			case "ReduceSum":
				// ReduceSum returns a scalar, not a vector
				return cVarInfo{cType: t.profile.CType}
			case "BitsFromMask":
				// BitsFromMask returns a scalar unsigned integer
				return cVarInfo{cType: "unsigned int"}
			case "LessThan":
				// LessThan returns a mask vector
				maskType := vecType // default to same vector type
				if mt, ok := t.profile.MaskType[t.tier]; ok {
					maskType = mt
				}
				return cVarInfo{cType: maskType, isVector: true}
			case "IfThenElse":
				return cVarInfo{cType: vecType, isVector: true}
			case "MaxLanes", "GetLane":
				return cVarInfo{cType: "long"}
			}
		}
	}

	// Check for v.NumLanes() method calls
	if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
		if sel.Sel.Name == "NumLanes" || sel.Sel.Name == "NumElements" {
			return cVarInfo{cType: "long"}
		}
	}

	// Check for math.Float32bits → unsigned int, math.Float32frombits → float
	if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "math" {
			switch sel.Sel.Name {
			case "Float32bits":
				return cVarInfo{cType: "unsigned int"}
			case "Float32frombits":
				return cVarInfo{cType: "float"}
			}
		}
	}

	// Check for built-in functions and type conversions
	if ident, ok := e.Fun.(*ast.Ident); ok {
		// len() returns an integer
		if ident.Name == "len" {
			return cVarInfo{cType: "long"}
		}
		// getSignBit() → unsigned int
		if ident.Name == "getSignBit" {
			return cVarInfo{cType: "unsigned int"}
		}
		// Type conversions: uint32(x) → unsigned int, etc.
		if cType := t.goTypeConvToCType(ident.Name); cType != "" {
			return cVarInfo{cType: cType}
		}
		// min/max: infer from first argument
		if (ident.Name == "min" || ident.Name == "max") && len(e.Args) > 0 {
			return t.inferType(e.Args[0])
		}
	}

	return cVarInfo{cType: t.profile.CType}
}

// goTypeToCType converts Go type names to C type names.
func (t *CASTTranslator) goTypeToCType(goType string) string {
	switch goType {
	case "int", "int64":
		return "long"
	case "int32":
		return "int"
	case "float32":
		return "float"
	case "float64":
		return "double"
	case "uint64":
		return "unsigned long"
	case "uint32":
		return "unsigned int"
	case "uint8", "byte":
		return "unsigned char"
	case "T":
		return t.profile.CType
	default:
		if strings.HasPrefix(goType, "[]") {
			elemType := strings.TrimPrefix(goType, "[]")
			return goSliceElemToCType(elemType, t.profile) + " *"
		}
		return "long" // default for unknown types
	}
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

// writef writes indented formatted output.
func (t *CASTTranslator) writef(format string, args ...any) {
	for range t.indent {
		t.buf.WriteString("    ")
	}
	fmt.Fprintf(t.buf, format, args...)
}

// writefRaw writes formatted output without indentation (for continuing same line).
func (t *CASTTranslator) writefRaw(format string, args ...any) {
	fmt.Fprintf(t.buf, format, args...)
}

// ---------------------------------------------------------------------------
// Eligibility check
// ---------------------------------------------------------------------------

// IsASTCEligible returns true if a function should be translated using the
// AST-walking translator rather than the template-based CEmitter.
// Eligible functions have slice parameters, use hwy.* operations, and are
// NOT composite math functions handled by the template path.
func IsASTCEligible(pf *ParsedFunc) bool {
	// Must have slice params (not Vec→Vec)
	hasSlice := false
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			hasSlice = true
			break
		}
	}
	if !hasSlice {
		return false
	}

	// Must NOT have Vec in signature (those go through IsCEligible)
	if hasVecInSignature(*pf) {
		return false
	}

	// Must use hwy.* operations
	hasHwyOps := false
	for _, call := range pf.HwyCalls {
		if call.Package == "hwy" {
			hasHwyOps = true
			break
		}
	}
	if !hasHwyOps {
		return false
	}

	// Must NOT be a composite math function (those use the template path)
	if mathOpFromFuncName(pf.Name) != "" {
		return false
	}

	// Must have int params (indicates a multi-dimensional function, not a simple
	// array transform). This distinguishes matmul/transpose from GELU/softmax.
	// OR must use integer SIMD ops (And, PopCount, BitsFromMask, LessThan, etc.)
	hasIntParam := false
	for _, p := range pf.Params {
		if p.Type == "int" || p.Type == "int64" {
			hasIntParam = true
			break
		}
	}

	if hasIntParam {
		return true
	}

	// Also eligible if the function uses integer-specific SIMD operations
	return hasIntegerSIMDOps(pf)
}

// hasIntegerSIMDOps returns true if the function uses SIMD operations that
// indicate integer/bitwise processing (RaBitQ, varint, etc.)
func hasIntegerSIMDOps(pf *ParsedFunc) bool {
	intOps := map[string]bool{
		"And": true, "Or": true, "Xor": true,
		"PopCount": true, "BitsFromMask": true,
		"LessThan": true, "TableLookupBytes": true,
		"IfThenElse": true, "LoadSlice": true,
		"Load4": true,
	}
	for _, call := range pf.HwyCalls {
		if call.Package == "hwy" && intOps[call.FuncName] {
			return true
		}
	}
	return false
}
