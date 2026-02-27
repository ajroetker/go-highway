package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"slices"
	"sort"
	"strconv"
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

	// Struct type tracking: maps C struct type name → struct info.
	// e.g. "ImageF32" → {goType: "*Image[T]", elemCType: "float"}
	// Used to emit struct typedefs for any generic struct pointer parameters.
	requiredStructTypes map[string]structTypeInfo

	// Widened accumulator tracking: variables that are kept in a wider type
	// (e.g., f32 for BF16) to avoid promote/demote round-trips per FMA.
	// Maps variable name → true. Each widened var "x" becomes "x_lo"/"x_hi" in C.
	widenedVars map[string]bool

	// Generic type parameter names (e.g., {"T": true}).
	// Used to resolve type conversions like T(2) to the concrete C type.
	typeParamNames map[string]bool

	// Per-type-param concrete types (from //hwy:gen); nil for single-type.
	// When set, resolveTypeParam uses this map instead of elemType.
	typeMap map[string]string

	// Package-level array globals (e.g., nf4LookupTable).
	// Set via SetPackageGlobals before TranslateToC.
	packageGlobals    map[string]*PackageGlobal  // name → global
	referencedGlobals map[string]bool            // globals actually used in function body
	packageStructs    map[string]*PackageStruct  // struct type name → def (from struct-typed globals)

	// Package-level integer constants (e.g., BlockSize = 48).
	// Set via SetPackageConsts before TranslateToC.
	packageConsts    map[string]string // name → value
	referencedConsts map[string]bool   // consts actually used in function body

	buf      *bytes.Buffer
	indent   int
	tmpCount int // counter for unique temporary variable names
	errors   []string // translation errors collected during buildParamMap

	// returnOrder stores return parameter map keys in declaration order
	// (matching pf.Returns), so translateReturnStmt assigns results correctly.
	returnOrder  []string // e.g., ["__return_values", "__return_consumed"]
	returnParams []Param  // original Go return params for type inspection

	// helperMode generates a C function with a simple calling convention:
	// ints/floats passed by value (not pointers), direct return values,
	// no hidden slice length params, and the original Go function name.
	// Used for sibling helper functions called from the main GOAT function.
	helperMode bool

	// helperSliceParams maps helper function name → list of param indices
	// that are slices. When calling a helper, the translator appends the
	// corresponding len_<name> variables for each slice argument.
	helperSliceParams map[string][]int

	// helperReturnVec tracks helper functions that return hwy.Vec[T].
	// inferCallType uses this to assign the correct vector type to results
	// of helper function calls (instead of defaulting to scalar).
	helperReturnVec map[string]bool

	// constVars maps variable names to their known constant integer values.
	// Populated when translating hwy.NumLanes/MaxLanes assignments (e.g.,
	// "lanes" → 4). Used by tryEvalConstInt to constant-fold expressions
	// like lanes-1 in GetLane index arguments.
	constVars map[string]int
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

// isWidenedVar checks if an AST expression refers to a widened accumulator variable.
func (t *CASTTranslator) isWidenedVar(expr ast.Expr) (string, bool) {
	if id, ok := expr.(*ast.Ident); ok {
		return id.Name, t.widenedVars[id.Name]
	}
	return "", false
}

// isHwyCall checks if an expression is a call to hwy.<fnName> (with optional
// type parameters) and returns the call expression if so.
func isHwyCall(expr ast.Expr, fnName string) (*ast.CallExpr, bool) {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return nil, false
	}
	fun := call.Fun
	if idx, ok := fun.(*ast.IndexExpr); ok {
		fun = idx.X
	}
	if idx, ok := fun.(*ast.IndexListExpr); ok {
		fun = idx.X
	}
	sel := extractSelectorExpr(fun)
	if sel == nil {
		return nil, false
	}
	pkg, ok := sel.X.(*ast.Ident)
	if !ok || pkg.Name != "hwy" {
		return nil, false
	}
	return call, sel.Sel.Name == fnName
}

// structTypeInfo tracks information about a generic struct type used as a parameter.
type structTypeInfo struct {
	goType    string        // Original Go type, e.g. "*Image[T]"
	elemCType string        // C element type, e.g. "float", "double"
	fields    []StructField // Discovered fields from method calls
}

// cVarInfo tracks the C type of a local variable.
type cVarInfo struct {
	cType         string // "float32x4_t", "float", "long", "float *"
	isVector      bool
	isPtr         bool
	isArray       bool   // true for fixed-size C arrays (e.g., unsigned char result[16])
	arrayLen      string // for vector arrays: the C length expression (e.g. "lanes")
	isWidened     bool   // true if widened accumulator (varname_lo/varname_hi pair in C)
	isStructPtr   bool   // true for local variables pointing to struct (e.g., &structArray[idx])
	structPtrType string // struct type name in C (e.g., "maskedVByte12Lookup")
}

// cParamInfo tracks function parameter translation details.
type cParamInfo struct {
	goName          string // "a", "b", "c", "m", "n", "k"
	goType          string // "[]T", "int", "*Image[T]"
	cName           string // "a", "b", "c", "pm", "pn", "pk"
	cType           string // "float *", "long *", "ImageF32 *"
	isSlice         bool
	isInt           bool
	isVector        bool   // true for hwy.Vec[T] parameters (helper functions only)
	isStructPtr     bool   // true for generic struct pointer parameters (e.g., *Image[T])
	structElemCType string // "float", "double" - element type for struct's data field
}

// NewCASTTranslator creates a translator for the given profile and element type.
func NewCASTTranslator(profile *CIntrinsicProfile, elemType string) *CASTTranslator {
	tier, lanes := primaryTier(profile)
	return &CASTTranslator{
		profile:             profile,
		tier:                tier,
		lanes:               lanes,
		elemType:            elemType,
		vars:                make(map[string]cVarInfo),
		params:              make(map[string]cParamInfo),
		sliceLenVars:        make(map[string]string),
		widenedVars:         make(map[string]bool),
		requiredStructTypes: make(map[string]structTypeInfo),
		helperSliceParams:   make(map[string][]int),
		helperReturnVec:     make(map[string]bool),
		constVars:           make(map[string]int),
		buf:                 &bytes.Buffer{},
	}
}

// SetPackageGlobals provides the translator with package-level array globals
// that may be referenced in the function body. Referenced globals are emitted
// as static const arrays in the generated C file.
func (t *CASTTranslator) SetPackageGlobals(globals []PackageGlobal) {
	t.packageGlobals = make(map[string]*PackageGlobal, len(globals))
	t.packageStructs = make(map[string]*PackageStruct)
	for i := range globals {
		t.packageGlobals[globals[i].Name] = &globals[i]
		// Collect struct type definitions from struct-typed globals
		if globals[i].IsStruct && globals[i].StructDef != nil {
			t.packageStructs[globals[i].StructDef.Name] = globals[i].StructDef
		}
	}
}

// SetPackageConsts provides the translator with package-level integer constants
// (e.g., BlockSize = 48). Referenced constants are emitted as #define macros.
func (t *CASTTranslator) SetPackageConsts(consts []PackageConst) {
	t.packageConsts = make(map[string]string, len(consts))
	for _, c := range consts {
		t.packageConsts[c.Name] = c.Value
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

// TranslateToCHelper translates a ParsedFunc to C as an inline helper function.
// Unlike TranslateToC, helper mode uses a simple calling convention: ints/floats
// by value (not pointers), direct return values, no hidden slice length params,
// and the original Go function name. This matches how callers emit calls to
// sibling functions in the same C file.
func (t *CASTTranslator) TranslateToCHelper(pf *ParsedFunc) (string, error) {
	t.helperMode = true
	defer func() { t.helperMode = false }()
	return t.TranslateToC(pf)
}

// TranslateToC translates a ParsedFunc to GOAT-compatible C source code.
func (t *CASTTranslator) TranslateToC(pf *ParsedFunc) (string, error) {
	t.buf.Reset()
	t.vars = make(map[string]cVarInfo)
	t.params = make(map[string]cParamInfo)
	t.widenedVars = make(map[string]bool)
	t.requiredStructTypes = make(map[string]structTypeInfo)
	t.typeParamNames = make(map[string]bool)
	for _, tp := range pf.TypeParams {
		t.typeParamNames[tp.Name] = true
	}
	t.indent = 0

	// Build parameter map (this also collects required struct types)
	t.errors = nil
	t.buildParamMap(pf)
	if len(t.errors) > 0 {
		return "", fmt.Errorf("parameter mapping errors:\n  %s", strings.Join(t.errors, "\n  "))
	}

	// Discover struct fields by analyzing method calls in the function body
	// This must happen before emitting typedefs
	if pf.Body != nil {
		t.discoverStructFields(pf.Body)
	}

	// Discover referenced package-level globals and constants
	t.referencedGlobals = make(map[string]bool)
	if pf.Body != nil && len(t.packageGlobals) > 0 {
		t.discoverReferencedGlobals(pf.Body)
	}
	t.referencedConsts = make(map[string]bool)
	if pf.Body != nil && len(t.packageConsts) > 0 {
		t.discoverReferencedConsts(pf.Body)
	}
	t.emitPackageLevelDefines()
	t.emitStaticConstGlobals()

	// Emit struct typedefs if needed
	t.emitStructTypedefs()

	// Emit function signature
	t.emitFuncSignature(pf)

	// Emit int parameter dereferences
	t.indent = 1
	t.emitParamDerefs()

	// Emit local variable declarations for named return values.
	// In Go, named return values are like pre-declared local variables initialized to zero.
	// In C, the return statement writes them to output pointer parameters.
	if !t.helperMode {
		t.emitNamedReturnDecls(pf)
	}

	// Translate function body
	if pf.Body != nil {
		t.translateBlockStmtContents(pf.Body)
	}

	// Close function
	t.indent = 0
	t.writef("}\n")

	return t.buf.String(), nil
}

// discoverStructFields walks the function body to discover struct fields
// from field accesses and method calls. This uses a convention-based approach:
//   - Direct field accesses (e.g., img.height, img.width) → scalar fields (long)
//   - Methods with 0 args (e.g., Width(), Height()) → scalar fields (long)
//   - Methods with 1 arg (e.g., Row(y)) → data + stride pattern
func (t *CASTTranslator) discoverStructFields(body *ast.BlockStmt) {
	// Track discovered fields per struct type
	discovered := make(map[string]map[string]StructField) // cTypeName -> fieldName -> field
	// Track method names per struct type to avoid treating them as fields
	methodNames := make(map[string]map[string]bool) // cTypeName -> methodName -> true

	// Initialize maps for each struct param
	for cTypeName := range t.requiredStructTypes {
		discovered[cTypeName] = make(map[string]StructField)
		methodNames[cTypeName] = make(map[string]bool)
	}

	// First pass: find all method calls to identify method names
	ast.Inspect(body, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if ident, ok := sel.X.(*ast.Ident); ok {
					if paramInfo, exists := t.params[ident.Name]; exists && paramInfo.isStructPtr {
						structBaseName := extractStructBaseName(paramInfo.goType)
						cTypeName := structBaseName + cTypeShortSuffix(paramInfo.structElemCType)
						methodNames[cTypeName][sel.Sel.Name] = true
					}
				}
			}
		}
		return true
	})

	// Second pass: discover fields from method calls and field accesses
	ast.Inspect(body, func(n ast.Node) bool {
		// Handle method calls: param.Method(...)
		if call, ok := n.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if ident, ok := sel.X.(*ast.Ident); ok {
					if paramInfo, exists := t.params[ident.Name]; exists && paramInfo.isStructPtr {
						structBaseName := extractStructBaseName(paramInfo.goType)
						cTypeName := structBaseName + cTypeShortSuffix(paramInfo.structElemCType)

						methodName := sel.Sel.Name
						fieldName := strings.ToLower(methodName)
						argCount := len(call.Args)

						if argCount == 0 {
							// Simple getter: Width() → width field of type long
							if _, exists := discovered[cTypeName][fieldName]; !exists {
								discovered[cTypeName][fieldName] = StructField{
									Name:     fieldName,
									GoType:   "int64",
									CType:    "long",
									GoGetter: "." + methodName + "()",
									IsPtr:    false,
								}
							}
						} else if argCount == 1 {
							// Row-like accessor: Row(y) → data pointer field
							if _, exists := discovered[cTypeName]["data"]; !exists {
								discovered[cTypeName]["data"] = StructField{
									Name:     "data",
									GoType:   "unsafe.Pointer",
									CType:    paramInfo.structElemCType + " *",
									GoGetter: "." + methodName + "(0)[0]",
									IsPtr:    true,
									IsData:   true,
								}
							}
							// Add stride if not already present
							if _, exists := discovered[cTypeName]["stride"]; !exists {
								discovered[cTypeName]["stride"] = StructField{
									Name:     "stride",
									GoType:   "int64",
									CType:    "long",
									GoGetter: ".Stride()",
									IsPtr:    false,
								}
							}
						}
					}
				}
			}
		}

		// Handle direct field accesses: param.field
		// Skip if this is a method name (to avoid treating Row as a field when it's a method)
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if ident, ok := sel.X.(*ast.Ident); ok {
				if paramInfo, exists := t.params[ident.Name]; exists && paramInfo.isStructPtr {
					structBaseName := extractStructBaseName(paramInfo.goType)
					cTypeName := structBaseName + cTypeShortSuffix(paramInfo.structElemCType)

					fieldName := sel.Sel.Name

					// Skip method names - they're not fields
					if methodNames[cTypeName][fieldName] {
						return true
					}

					if _, exists := discovered[cTypeName][fieldName]; !exists {
						// Check if this is the data field
						if fieldName == "data" {
							discovered[cTypeName][fieldName] = StructField{
								Name:     fieldName,
								GoType:   "unsafe.Pointer",
								CType:    paramInfo.structElemCType + " *",
								GoGetter: ".data",
								IsPtr:    true,
								IsData:   true,
							}
						} else {
							// Assume other fields are long (int64) - common for dimensions
							discovered[cTypeName][fieldName] = StructField{
								Name:     fieldName,
								GoType:   "int64",
								CType:    "long",
								GoGetter: "." + fieldName,
								IsPtr:    false,
							}
						}
					}
				}
			}
		}

		return true
	})

	// Update requiredStructTypes with discovered fields
	for cTypeName, fields := range discovered {
		info := t.requiredStructTypes[cTypeName]
		// Convert map to slice in a deterministic order
		info.fields = nil
		// Data field first (if present)
		if f, ok := fields["data"]; ok {
			info.fields = append(info.fields, f)
			delete(fields, "data")
		}
		// Then other fields in sorted order
		var names []string
		for name := range fields {
			names = append(names, name)
		}
		slices.Sort(names)
		for _, name := range names {
			info.fields = append(info.fields, fields[name])
		}
		t.requiredStructTypes[cTypeName] = info
	}
}

// emitStructTypedefs emits C struct typedefs for generic struct types used as parameters.
// The struct layout is discovered by analyzing method calls in the function body.
// Example: typedef struct { float *data; long width; long height; long stride; } ImageF32;
func (t *CASTTranslator) emitStructTypedefs() {
	if len(t.requiredStructTypes) == 0 {
		return
	}

	t.writef("// Struct typedefs for C-compatible parameter passing\n")
	for cTypeName, info := range t.requiredStructTypes {
		if len(info.fields) == 0 {
			continue
		}

		t.writef("typedef struct {\n")
		for _, field := range info.fields {
			if field.IsPtr {
				// Data pointer field - use the element type
				t.writef("    %s *%s;\n", info.elemCType, field.Name)
			} else {
				// Dimension fields
				t.writef("    %s %s;\n", field.CType, field.Name)
			}
		}
		t.writef("} %s;\n\n", cTypeName)
	}
}

// discoverReferencedGlobals walks the function body to find identifiers
// matching known package-level globals (e.g., nf4LookupTable).
func (t *CASTTranslator) discoverReferencedGlobals(body *ast.BlockStmt) {
	ast.Inspect(body, func(n ast.Node) bool {
		ident, ok := n.(*ast.Ident)
		if !ok {
			return true
		}
		if _, known := t.packageGlobals[ident.Name]; !known {
			return true
		}
		// Skip if it's a local variable or parameter
		if _, isVar := t.vars[ident.Name]; isVar {
			return true
		}
		if _, isParam := t.params[ident.Name]; isParam {
			return true
		}
		t.referencedGlobals[ident.Name] = true
		return true
	})
}

// emitStaticConstGlobals emits static const array declarations for any
// package-level globals referenced in the function body.
func (t *CASTTranslator) emitStaticConstGlobals() {
	if len(t.referencedGlobals) == 0 {
		return
	}

	// Sort for deterministic output
	names := make([]string, 0, len(t.referencedGlobals))
	for name := range t.referencedGlobals {
		names = append(names, name)
	}
	sort.Strings(names)

	// Emit struct typedefs for any struct-typed globals (before the arrays)
	emittedStructs := make(map[string]bool)
	for _, name := range names {
		pg := t.packageGlobals[name]
		if pg.IsStruct && pg.StructDef != nil && !emittedStructs[pg.StructDef.Name] {
			t.emitPackageStructTypedef(pg.StructDef)
			emittedStructs[pg.StructDef.Name] = true
		}
	}

	for _, name := range names {
		pg := t.packageGlobals[name]
		if len(pg.Values) == 0 {
			// No values computed — skip (init() evaluator didn't produce values)
			continue
		}

		if pg.IsStruct && pg.StructDef != nil {
			t.emitStructArrayGlobal(pg)
			continue
		}

		cElemType := goPkgGlobalElemToCType(pg.ElemType)
		suffix := ""
		if pg.ElemType == "float32" {
			suffix = "f"
		}

		if pg.InnerSize > 0 {
			// 2D array: static const T name[N][M] = { {v0, v1, ...}, {vM, ...}, ... };
			t.writefRaw("static const %s %s[%d][%d] = {\n", cElemType, pg.Name, pg.Size, pg.InnerSize)
			for i := range pg.Size {
				t.buf.WriteString("    { ")
				for j := range pg.InnerSize {
					if j > 0 {
						t.buf.WriteString(", ")
					}
					idx := i*pg.InnerSize + j
					if idx < len(pg.Values) {
						t.buf.WriteString(pg.Values[idx])
					} else {
						t.buf.WriteString("0")
					}
					if suffix != "" {
						t.buf.WriteString(suffix)
					}
				}
				if i < pg.Size-1 {
					t.buf.WriteString(" },\n")
				} else {
					t.buf.WriteString(" }\n")
				}
			}
			t.buf.WriteString("};\n\n")
		} else {
			// 1D array: static const T name[N] = { v0, v1, ... };
			t.writefRaw("static const %s %s[%d] = { ", cElemType, pg.Name, pg.Size)
			for i, val := range pg.Values {
				if i > 0 {
					t.buf.WriteString(", ")
				}
				t.buf.WriteString(val)
				if suffix != "" && !strings.HasSuffix(val, suffix) {
					t.buf.WriteString(suffix)
				}
			}
			t.buf.WriteString(" };\n\n")
		}
	}
}

// emitPackageStructTypedef emits a C typedef for a package-level struct type.
// Example: typedef struct { unsigned char numValues; unsigned char bytesConsumed; unsigned char valueEnds[4]; } maskedVByte12Lookup;
func (t *CASTTranslator) emitPackageStructTypedef(sd *PackageStruct) {
	t.writefRaw("typedef struct {\n")
	for _, f := range sd.Fields {
		cType := goPkgGlobalElemToCType(f.ElemType)
		if f.IsArray {
			t.writefRaw("    %s %s[%d];\n", cType, f.Name, f.ArraySize)
		} else {
			t.writefRaw("    %s %s;\n", cType, f.Name)
		}
	}
	t.writefRaw("} %s;\n\n", sd.Name)
}

// emitStructArrayGlobal emits a static const struct array global.
// Values are stored flattened: for each element, fields in declaration order,
// with array fields expanded inline.
func (t *CASTTranslator) emitStructArrayGlobal(pg *PackageGlobal) {
	sd := pg.StructDef
	flatSize := sd.FlatSize()

	t.writefRaw("static const %s %s[%d] = {\n", sd.Name, pg.Name, pg.Size)
	for i := range pg.Size {
		base := i * flatSize
		t.buf.WriteString("    { ")
		first := true
		for _, f := range sd.Fields {
			if f.IsArray {
				if !first {
					t.buf.WriteString(", ")
				}
				t.buf.WriteString("{ ")
				for j := range f.ArraySize {
					if j > 0 {
						t.buf.WriteString(", ")
					}
					idx := base
					if idx < len(pg.Values) {
						t.buf.WriteString(pg.Values[idx])
					} else {
						t.buf.WriteString("0")
					}
					base++
				}
				t.buf.WriteString(" }")
				first = false
			} else {
				if !first {
					t.buf.WriteString(", ")
				}
				if base < len(pg.Values) {
					t.buf.WriteString(pg.Values[base])
				} else {
					t.buf.WriteString("0")
				}
				base++
				first = false
			}
		}
		if i < pg.Size-1 {
			t.buf.WriteString(" },\n")
		} else {
			t.buf.WriteString(" }\n")
		}
	}
	t.buf.WriteString("};\n\n")
}

// discoverReferencedConsts walks the function body to find identifiers
// matching known package-level constants (e.g., BlockSize).
func (t *CASTTranslator) discoverReferencedConsts(body *ast.BlockStmt) {
	ast.Inspect(body, func(n ast.Node) bool {
		ident, ok := n.(*ast.Ident)
		if !ok {
			return true
		}
		if _, known := t.packageConsts[ident.Name]; !known {
			return true
		}
		// Skip if shadowed by local variable or parameter
		if _, isVar := t.vars[ident.Name]; isVar {
			return true
		}
		if _, isParam := t.params[ident.Name]; isParam {
			return true
		}
		t.referencedConsts[ident.Name] = true
		return true
	})
}

// emitPackageLevelDefines emits #define macros for referenced package-level constants.
func (t *CASTTranslator) emitPackageLevelDefines() {
	if len(t.referencedConsts) == 0 {
		return
	}

	names := make([]string, 0, len(t.referencedConsts))
	for name := range t.referencedConsts {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		t.writefRaw("#define %s %s\n", name, t.packageConsts[name])
	}
	t.buf.WriteString("\n")
}

// goPkgGlobalElemToCType maps Go element types to C types for static const arrays.
func goPkgGlobalElemToCType(goType string) string {
	switch goType {
	case "float32":
		return "float"
	case "float64":
		return "double"
	case "int32":
		return "int"
	case "int64", "int":
		return "long"
	case "uint32":
		return "unsigned int"
	case "uint64":
		return "unsigned long"
	case "uint8", "byte":
		return "unsigned char"
	case "int8":
		return "signed char"
	case "int16":
		return "short"
	case "uint16":
		return "unsigned short"
	default:
		return "float"
	}
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
		} else if isGenericStructPtr(p.Type) {
			// Generic struct pointer param (e.g., *Image[T]) → C struct pointer
			info.isStructPtr = true
			info.cName = p.Name
			elemType := extractStructElemType(p.Type)
			info.structElemCType = goSliceElemToCType(elemType, t.profile)
			structBaseName := extractStructBaseName(p.Type)
			cTypeName := structBaseName + cTypeShortSuffix(info.structElemCType)
			info.cType = cTypeName + " *"
			// Register this struct type for typedef generation
			t.requiredStructTypes[cTypeName] = structTypeInfo{
				goType:    p.Type,
				elemCType: info.structElemCType,
			}
		} else if isGoScalarIntType(p.Type) {
			if t.helperMode {
				// Helper mode: pass ints by value
				info.isInt = false
				info.cName = p.Name
				info.cType = "long"
			} else {
				// Scalar integer param → long pointer (GOAT convention).
				// GOAT only supports int64_t/long, float, double, _Bool, or pointer
				// as function arguments, so all integer types are passed as long*.
				info.isInt = true
				info.cName = "p" + p.Name
				info.cType = "long *"
			}
		} else if p.Type == "float32" || p.Type == "float64" {
			if t.helperMode {
				// Helper mode: pass floats by value
				info.isInt = false
				info.cName = p.Name
				if p.Type == "float32" {
					info.cType = "float"
				} else {
					info.cType = "double"
				}
			} else {
				// Scalar float param → passed as pointer in GOAT
				info.isInt = true // reuse isInt to get pointer + dereference treatment
				info.cName = "p" + p.Name
				if p.Type == "float32" {
					info.cType = "float *"
				} else {
					info.cType = "double *"
				}
			}
		} else if t.isTypeParam(p.Type) {
			// Generic type parameter (e.g., "T" in func F[T hwy.Floats](..., coeff T)).
			// Resolve to the concrete element type for this instantiation.
			resolvedType := t.resolveTypeParam(p.Type)
			if t.helperMode {
				// Helper mode: pass by value
				info.isInt = false
				info.cName = p.Name
				if resolvedType == "float32" {
					info.cType = "float"
				} else if resolvedType == "float64" {
					info.cType = "double"
				} else if isGoScalarIntType(resolvedType) {
					info.cType = "long"
				} else {
					scalarCType := goSliceElemToCType(p.Type, t.profile)
					info.cType = scalarCType
				}
			} else {
				info.isInt = true // reuse isInt for pointer + dereference treatment
				info.cName = "p" + p.Name
				if resolvedType == "float32" {
					info.cType = "float *"
				} else if resolvedType == "float64" {
					info.cType = "double *"
				} else if isGoScalarIntType(resolvedType) {
					info.cType = "long *"
				} else {
					// Exotic types (e.g., hwy.Float16 → float16_t) — use profile's
					// scalar arithmetic type if available, else CType.
					scalarCType := goSliceElemToCType(p.Type, t.profile)
					info.cType = scalarCType + " *"
				}
			}
		} else if strings.HasPrefix(p.Type, "hwy.Vec[") {
			// Vector param — used by helper functions inlined in the caller's C file.
			// These never cross the Go/assembly boundary, so passing vector register
			// types (float32x4_t etc.) is safe within a single compilation unit.
			info.cName = p.Name
			info.cType = t.profile.VecTypes[t.tier]
			info.isVector = true
		} else if strings.HasPrefix(p.Type, "*") && isGoScalarIntType(p.Type[1:]) {
			// Pointer-to-scalar-int param (e.g. *int, *int64) — used by helper
			// functions that modify counters by reference.
			info.cName = p.Name
			info.cType = goPkgGlobalElemToCType(p.Type[1:]) + " *"
		} else {
			t.errors = append(t.errors, fmt.Sprintf("unsupported parameter type %q for param %q; "+
				"neon:asm supports slices ([]T), generic struct pointers (*Struct[T]), "+
				"scalar ints (int, int64, uint64, ...), float32, and float64", p.Type, p.Name))
			info.cName = p.Name
			info.cType = "long" // placeholder to avoid cascading errors
		}
		t.params[p.Name] = info
	}

	// For functions without explicit int size params (e.g. BaseBitProduct uses len(code)),
	// add a length parameter for the first slice. All slices are assumed same length.
	hasExplicitSize := false
	for _, p := range pf.Params {
		if isGoScalarIntType(p.Type) {
			hasExplicitSize = true
			break
		}
	}
	// Always register sliceLenVars so len(slice) resolves to a C variable.
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			t.sliceLenVars[p.Name] = "len_" + p.Name
		}
	}

	// In helper mode, add explicit length params for slices (so len(s) resolves)
	// but skip output pointer handling — helpers return values directly.
	if t.helperMode {
		for _, p := range pf.Params {
			if strings.HasPrefix(p.Type, "[]") {
				lenVarName := "len_" + p.Name
				info := cParamInfo{
					goName: lenVarName,
					goType: "int",
					cName:  lenVarName,
					cType:  "long",
				}
				t.params["__len_"+p.Name] = info
			}
		}
		return
	}

	if !hasExplicitSize && !hasMixedSliceTypes(pf) {
		// No explicit int params and all slices have the same element type:
		// add a single shared length parameter.
		var firstSlice string
		for _, p := range pf.Params {
			if strings.HasPrefix(p.Type, "[]") {
				if firstSlice == "" {
					firstSlice = p.Name
				}
				// Override per-slice mapping: all map to the shared variable.
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
	} else {
		// Explicit int params exist, but len(slice) calls may still appear.
		// Add per-slice hidden length parameters so len() resolves correctly.
		for _, p := range pf.Params {
			if strings.HasPrefix(p.Type, "[]") {
				lenVarName := "len_" + p.Name
				lenCName := "plen_" + p.Name
				info := cParamInfo{
					goName: lenVarName,
					goType: "int",
					cName:  lenCName,
					cType:  "long *",
					isInt:  true,
				}
				t.params["__len_"+p.Name] = info
			}
		}
	}

	// Handle return values as output pointers.
	// Use the appropriate C pointer type so that GOAT generates correct
	// load/store instructions (e.g., float * for float32 results, not long *
	// which would emit fcvtzs and truncate special values like Inf).
	t.returnOrder = nil
	t.returnParams = nil
	for _, ret := range pf.Returns {
		name := ret.Name
		if name == "" {
			name = "result"
		}
		// Resolve type parameter to concrete type (uses typeMap if set)
		resolvedRetType := ret.Type
		if t.isTypeParam(ret.Type) {
			resolvedRetType = t.resolveTypeParam(ret.Type)
		}
		cType := goReturnTypeToCPtrType(resolvedRetType, t.elemType)
		info := cParamInfo{
			goName: name,
			goType: ret.Type,
			cName:  "pout_" + name,
			cType:  cType,
			isInt:  false, // not dereferenced at top - handled by return stmt
		}
		key := "__return_" + name
		t.params[key] = info
		t.returnOrder = append(t.returnOrder, key)
		t.returnParams = append(t.returnParams, ret)
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
	case "int16":
		return "short"
	case "int8":
		return "signed char"
	case "uint16":
		return "unsigned short"
	case "T":
		if profile.PointerElemType != "" {
			return profile.PointerElemType
		}
		if profile.ScalarArithType != "" {
			return profile.ScalarArithType
		}
		return profile.CType
	default:
		return profile.CType
	}
}

// goReturnTypeToCPtrType maps a Go return type to the C pointer type for the
// output parameter. Float types use float */double * so GOAT generates correct
// FP load/store instructions instead of integer conversion (fcvtzs).
// When goType is "T", elemType is used to resolve to the concrete type.
func goReturnTypeToCPtrType(goType, elemType string) string {
	resolved := goType
	if resolved == "T" {
		resolved = elemType
	}
	switch resolved {
	case "float32":
		return "float *"
	case "float64":
		return "double *"
	default:
		return "long *"
	}
}

// isScalarCType returns true for C scalar types that should be zero-initialized.
func isScalarCType(cType string) bool {
	switch cType {
	case "long", "int", "unsigned long", "unsigned int", "unsigned char",
		"float", "double", "short", "unsigned short", "signed char":
		return true
	default:
		return false
	}
}

// extractStructElemType extracts the element type from a generic struct pointer type.
// E.g., "*Image[T]" → "T", "*Image[float32]" → "float32"
func extractStructElemType(goType string) string {
	start := strings.Index(goType, "[")
	end := strings.LastIndex(goType, "]")
	if start == -1 || end == -1 || start >= end {
		return "T"
	}
	return goType[start+1 : end]
}

// Note: extractStructBaseName is defined in c_generator.go as the single source of truth

// cTypeShortSuffix returns a short suffix for C struct type names.
// "float" → "F32", "double" → "F64", "int" → "I32", etc.
func cTypeShortSuffix(cType string) string {
	switch cType {
	case "float":
		return "F32"
	case "double":
		return "F64"
	case "int":
		return "I32"
	case "long":
		return "I64"
	case "unsigned int":
		return "U32"
	case "unsigned long":
		return "U64"
	case "unsigned short":
		return "F16" // Used for hwy.Float16
	case "short":
		return "BF16" // Used for hwy.BFloat16
	case "unsigned char":
		return "U8"
	default:
		return "T"
	}
}

// emitFuncSignature emits: void funcname(float *a, float *b, ..., long *pm, long *pn, ...)
// If the Go function has return values, they are appended as output pointers.
// In helperMode, uses the original Go function name, passes ints by value,
// and emits a return type instead of void + output pointers.
func (t *CASTTranslator) emitFuncSignature(pf *ParsedFunc) {
	funcName := t.cFuncName(pf.Name)
	if t.helperMode {
		funcName = pf.Name
	}
	var params []string
	for _, p := range pf.Params {
		info := t.params[p.Name]
		if strings.HasSuffix(info.cType, "*") {
			params = append(params, info.cType+info.cName)
		} else {
			params = append(params, info.cType+" "+info.cName)
		}
	}

	// Append hidden length parameters in deterministic order (matching pf.Params slice order).
	// In helper mode, length params are plain values; otherwise they are pointers.
	for _, p := range pf.Params {
		if key := "__len_" + p.Name; strings.HasPrefix(p.Type, "[]") {
			if info, ok := t.params[key]; ok {
				if strings.HasSuffix(info.cType, "*") {
					params = append(params, info.cType+info.cName)
				} else {
					params = append(params, info.cType+" "+info.cName)
				}
			}
		}
	}
	if !t.helperMode {
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
	}

	// Determine return type
	retType := "void"
	if t.helperMode && len(pf.Returns) > 0 {
		ret := pf.Returns[0]
		if strings.HasPrefix(ret.Type, "hwy.Vec[") {
			retType = t.profile.VecTypes[t.tier]
		} else if isGoScalarIntType(ret.Type) {
			retType = "long"
		} else if ret.Type == "float32" {
			retType = "float"
		} else if ret.Type == "float64" {
			retType = "double"
		} else {
			retType = t.goTypeToCType(ret.Type)
		}
	}

	if t.profile.FuncAttrs != "" {
		t.writef("%s %s(%s) %s {\n", retType, funcName, strings.Join(params, ", "), t.profile.FuncAttrs)
	} else {
		t.writef("%s %s(%s) {\n", retType, funcName, strings.Join(params, ", "))
	}
}

// cFuncName builds the C function name from the Go function name.
// BaseMatMul → basematmul_c_f32_neon
func (t *CASTTranslator) cFuncName(baseName string) string {
	name := strings.ToLower(baseName)
	name = strings.TrimPrefix(name, "base")
	targetSuffix := strings.ToLower(t.profile.TargetName)
	return name + "_c_" + cTypeSuffix(t.elemType) + "_" + targetSuffix
}

// lanesExpr returns the lanes count as a string. For profiles with dynamic
// lanes (e.g. SVE Linux), this returns the runtime expression like "svcntw()".
// Otherwise it returns the literal lane count.
func (t *CASTTranslator) lanesExpr() string {
	for _, tier := range t.profile.Tiers {
		if tier.Name == t.tier && tier.DynamicLanes != "" {
			return tier.DynamicLanes
		}
	}
	return fmt.Sprintf("%d", t.lanes)
}

// tryEvalConstInt tries to evaluate an integer expression to a constant value.
// It recognizes:
//   - Integer literals (ast.BasicLit with token.INT)
//   - Known identifiers (e.g., "lanes" mapped to t.lanes via t.constVars)
//   - Binary expressions with constant operands (e.g., lanes - 1)
//   - Parenthesized expressions
func (t *CASTTranslator) tryEvalConstInt(expr ast.Expr) (int, bool) {
	switch e := expr.(type) {
	case *ast.BasicLit:
		if e.Kind == token.INT {
			val, err := strconv.Atoi(e.Value)
			if err == nil {
				return val, true
			}
		}
	case *ast.Ident:
		if val, ok := t.constVars[e.Name]; ok {
			return val, true
		}
	case *ast.BinaryExpr:
		left, lok := t.tryEvalConstInt(e.X)
		right, rok := t.tryEvalConstInt(e.Y)
		if lok && rok {
			switch e.Op {
			case token.ADD:
				return left + right, true
			case token.SUB:
				return left - right, true
			case token.MUL:
				return left * right, true
			}
		}
	case *ast.ParenExpr:
		return t.tryEvalConstInt(e.X)
	}
	return 0, false
}

// isNumLanesCall reports whether expr is a call to hwy.NumLanes[T](),
// hwy.MaxLanes[T](), v.NumLanes(), or v.NumElements().
func (t *CASTTranslator) isNumLanesCall(expr ast.Expr) bool {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return false
	}
	// hwy.NumLanes[T]() or hwy.MaxLanes[T]()
	if sel := extractSelectorExpr(call.Fun); sel != nil {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
			return sel.Sel.Name == "NumLanes" || sel.Sel.Name == "MaxLanes"
		}
	}
	// v.NumLanes() or v.NumElements() method call
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		return sel.Sel.Name == "NumLanes" || sel.Sel.Name == "NumElements"
	}
	return false
}

// numLanesCallLanes returns the correct lane count for a NumLanes/MaxLanes call,
// taking into account the type parameter (e.g., hwy.NumLanes[uint8]() → 16 on NEON).
// For method calls like v.NumLanes(), returns t.lanes (the dispatch type's lane count).
func (t *CASTTranslator) numLanesCallLanes(expr ast.Expr) int {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return t.lanes
	}
	// Check for hwy.NumLanes[T]() with explicit type parameter via IndexExpr
	if idx, ok := call.Fun.(*ast.IndexExpr); ok {
		if sel, ok := idx.X.(*ast.SelectorExpr); ok {
			if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
				if sel.Sel.Name == "NumLanes" || sel.Sel.Name == "MaxLanes" {
					if typeName := typeExprToString(idx.Index); typeName != "" {
						if lanes := t.lanesForType(typeName); lanes > 0 {
							return lanes
						}
					}
				}
			}
		}
	}
	return t.lanes
}

// typeExprToString extracts a Go type name from an AST expression.
// Handles both plain idents (uint8) and qualified names (hwy.Float16).
func typeExprToString(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.SelectorExpr:
		if pkg, ok := e.X.(*ast.Ident); ok {
			return pkg.Name + "." + e.Sel.Name
		}
	}
	return ""
}

// lanesForType computes the number of SIMD lanes for an arbitrary element type,
// based on the vector width implied by the primary tier's dispatch type.
// Returns 0 if the type is unknown.
func (t *CASTTranslator) lanesForType(elemType string) int {
	// Compute vector width from dispatch type
	vecWidth := t.lanes * goTypeSize(t.elemType)
	if vecWidth == 0 {
		return 0
	}
	elemSize := goTypeSize(elemType)
	if elemSize == 0 {
		return 0
	}
	return vecWidth / elemSize
}

// goTypeSize returns the byte size for a Go type name.
func goTypeSize(typeName string) int {
	switch typeName {
	case "float32", "int32", "uint32":
		return 4
	case "float64", "int64", "uint64":
		return 8
	case "int16", "uint16", "hwy.Float16", "hwy.BFloat16", "Float16", "BFloat16":
		return 2
	case "int8", "uint8", "byte":
		return 1
	default:
		return 0
	}
}

// emitNamedReturnDecls emits local variable declarations for named return values.
// In Go, named return values are pre-declared and zero-initialized. In C, we emit
// them as local variables and the return statement writes them to output pointers.
// Array-typed returns (e.g., [4]uint32) are aliased to the output pointer.
func (t *CASTTranslator) emitNamedReturnDecls(pf *ParsedFunc) {
	for _, ret := range pf.Returns {
		if ret.Name == "" {
			continue
		}
		// Check for fixed-size array return types like [4]uint32
		if elemType, size := parseGoArrayType(ret.Type); size > 0 {
			cElem := t.goTypeToCType(elemType)
			poutName := "pout_" + ret.Name
			// Alias the output pointer as the array variable
			t.writef("%s *%s = (%s *)%s;\n", cElem, ret.Name, cElem, poutName)
			t.vars[ret.Name] = cVarInfo{cType: cElem + " *", isPtr: true}
			continue
		}
		cType := t.goTypeToCType(ret.Type)
		t.writef("%s %s = 0;\n", cType, ret.Name)
		t.vars[ret.Name] = cVarInfo{cType: cType}
	}
}

// parseGoArrayType parses Go array type strings like "[4]uint32" and returns
// the element type and size. Returns ("", 0) if not an array type.
func parseGoArrayType(goType string) (elemType string, size int) {
	if len(goType) < 3 || goType[0] != '[' {
		return "", 0
	}
	closeBracket := strings.Index(goType, "]")
	if closeBracket < 2 {
		return "", 0
	}
	sizeStr := goType[1:closeBracket]
	n, err := strconv.Atoi(sizeStr)
	if err != nil || n <= 0 {
		return "", 0
	}
	return goType[closeBracket+1:], n
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
	// SVE predicate declaration
	if t.profile.NeedsPredicate {
		t.writef("svbool_t pg = %s;\n", t.profile.PredicateDecl)
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
		idx        int
		forStmt    *ast.ForStmt
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
	case *ast.BranchStmt:
		if s.Tok == token.CONTINUE {
			t.writef("continue;\n")
		} else if s.Tok == token.BREAK {
			t.writef("break;\n")
		}
	default:
		// Skip unsupported statements
	}
}

// translateAssignStmt handles := and = assignments.
func (t *CASTTranslator) translateAssignStmt(s *ast.AssignStmt) {
	// Handle blank identifier discards: _ = expr (no-op, skip entirely)
	if len(s.Lhs) == 1 {
		if ident, ok := s.Lhs[0].(*ast.Ident); ok && ident.Name == "_" {
			return
		}
	}

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

	// Handle multi-value assignments from ictCoeffs[T]()
	if len(s.Lhs) > 1 && len(s.Rhs) == 1 {
		if call, ok := s.Rhs[0].(*ast.CallExpr); ok {
			if t.translateICTCoeffsAssign(s.Lhs, call, s.Tok) {
				return
			}
		}
		// Other single-RHS multi-assigns not supported
		return
	}

	// Handle parallel multi-assign: off1, off2 := -1, 0
	// where len(Lhs) == len(Rhs) and each pair is an independent assignment.
	if len(s.Lhs) > 1 && len(s.Lhs) == len(s.Rhs) {
		for i := range s.Lhs {
			// Skip blank identifiers
			if ident, ok := s.Lhs[i].(*ast.Ident); ok && ident.Name == "_" {
				continue
			}
			sub := &ast.AssignStmt{
				Lhs:    []ast.Expr{s.Lhs[i]},
				TokPos: s.TokPos,
				Tok:    s.Tok,
				Rhs:    []ast.Expr{s.Rhs[i]},
			}
			t.translateAssignStmt(sub)
		}
		return
	}

	if len(s.Lhs) != 1 || len(s.Rhs) != 1 {
		// Unsupported multi-assign shape
		return
	}

	lhs := s.Lhs[0]
	rhs := s.Rhs[0]

	// Check for vec.Data() — store vector to stack buffer for element access.
	// Pattern: valsData := maxVals.Data()
	// Emits: float valsData[4]; vst1q_f32(valsData, maxVals);
	if call, ok := rhs.(*ast.CallExpr); ok {
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Data" && len(call.Args) == 0 {
			if vecIdent, ok := sel.X.(*ast.Ident); ok {
				vecInfo := t.inferType(sel.X)
				if vecInfo.isVector {
					lhsName := t.translateExpr(lhs)
					vecType := t.profile.VecTypes[t.tier]
					storeFn := t.profile.StoreFn[t.tier]
					lanes := t.lanes
					cType := t.profile.CType
					if t.profile.ScalarArithType != "" {
						cType = t.profile.ScalarArithType
					}
					// Declare a stack buffer and store the vector into it
					t.vars[lhsName] = cVarInfo{cType: cType, isPtr: true}
					_ = vecType
					t.writef("%s %s[%d];\n", cType, lhsName, lanes)
					ptr := lhsName
					if t.profile.CastExpr != "" {
						ptr = fmt.Sprintf("%s(%s)", t.profile.CastExpr, ptr)
					}
					t.writef("%s(%s, %s);\n", storeFn, ptr, vecIdent.Name)
					return
				}
			}
		}
	}

	// Check for hwy.GetLane with non-literal index. Try constant-folding
	// first (e.g., lanes-1 → 3); fall back to store-to-stack for truly
	// variable indices (loop counters, etc.).
	if call, ok := rhs.(*ast.CallExpr); ok {
		if sel := extractSelectorExpr(call.Fun); sel != nil {
			if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" && sel.Sel.Name == "GetLane" {
				if len(call.Args) >= 2 {
					if _, isLit := call.Args[1].(*ast.BasicLit); !isLit {
						// Try constant-folding the index expression
						if val, ok := t.tryEvalConstInt(call.Args[1]); ok {
							lhsName := t.translateExpr(lhs)
							fn := t.profile.GetLaneFn[t.tier]
							vec := t.translateExpr(call.Args[0])
							rhsExpr := fmt.Sprintf("%s(%s, %d)", fn, vec, val)
							varInfo := cVarInfo{cType: t.profile.CType}
							if s.Tok == token.DEFINE {
								t.vars[lhsName] = varInfo
								t.writef("%s = %s;\n", cDeclVar(varInfo.cType, lhsName), rhsExpr)
							} else {
								t.writef("%s = %s;\n", lhsName, rhsExpr)
							}
							return
						}
						// Truly variable index — use store-to-stack
						lhsName := t.translateExpr(lhs)
						t.translateGetLaneVarIndex(lhsName, call.Args, s.Tok)
						return
					}
				}
			}
		}
	}

	// Extract raw LHS identifier for widened accumulator handling.
	// Must happen before translateExpr(lhs) which would materialize widened vars.
	var lhsIdent string
	if id, ok := lhs.(*ast.Ident); ok {
		lhsIdent = id.Name
	}

	// Widened accumulator: acc := hwy.Zero[T]() → two f32 halves
	if lhsIdent != "" && s.Tok == token.DEFINE && t.profile.WidenAccumulators {
		if _, ok := isHwyCall(rhs, "Zero"); ok {
			wideType := t.profile.WidenedAccType
			zeroExpr := t.profile.WidenedAccZero
			t.widenedVars[lhsIdent] = true
			t.vars[lhsIdent] = cVarInfo{cType: wideType, isVector: true, isWidened: true}
			t.writef("%s %s_lo = %s;\n", wideType, lhsIdent, zeroExpr)
			t.writef("%s %s_hi = %s;\n", wideType, lhsIdent, zeroExpr)
			return
		}
	}

	// Widened accumulator re-init: acc = hwy.Zero[T]() → reset both halves
	if lhsIdent != "" && s.Tok == token.ASSIGN && t.widenedVars[lhsIdent] {
		if _, ok := isHwyCall(rhs, "Zero"); ok {
			zeroExpr := t.profile.WidenedAccZero
			t.writef("%s_lo = %s;\n", lhsIdent, zeroExpr)
			t.writef("%s_hi = %s;\n", lhsIdent, zeroExpr)
			return
		}
	}

	// Widened accumulator: acc = hwy.MulAdd(a, b, acc) → two f32 FMAs
	if lhsIdent != "" && s.Tok == token.ASSIGN && t.widenedVars[lhsIdent] {
		if call, ok := isHwyCall(rhs, "MulAdd"); ok && len(call.Args) >= 3 {
			a := t.translateExpr(call.Args[0])
			b := t.translateExpr(call.Args[1])
			fmaFn := t.profile.WidenedFmaFn
			proLo := t.profile.SplitPromoteLo
			proHi := t.profile.SplitPromoteHi
			t.writef("%s_lo = %s(%s_lo, %s, %s);\n", lhsIdent, fmaFn, lhsIdent,
				fmt.Sprintf(proLo, a), fmt.Sprintf(proLo, b))
			t.writef("%s_hi = %s(%s_hi, %s, %s);\n", lhsIdent, fmaFn, lhsIdent,
				fmt.Sprintf(proHi, a), fmt.Sprintf(proHi, b))
			return
		}
	}

	// Widened accumulator: v = hwy.Add(v, acc) → promote+add on widened halves
	if lhsIdent != "" && s.Tok == token.ASSIGN {
		if call, ok := isHwyCall(rhs, "Add"); ok && len(call.Args) >= 2 {
			wName0, isW0 := t.isWidenedVar(call.Args[0])
			wName1, isW1 := t.isWidenedVar(call.Args[1])

			if isW0 && isW1 && t.widenedVars[lhsIdent] {
				// Both args are widened accumulators: add halves directly
				addFn := t.profile.WidenedAddFn
				t.writef("%s_lo = %s(%s_lo, %s_lo);\n", lhsIdent, addFn, wName0, wName1)
				t.writef("%s_hi = %s(%s_hi, %s_hi);\n", lhsIdent, addFn, wName0, wName1)
				return
			}

			wName, narrowIdx := "", -1
			if isW0 {
				wName = wName0
			}
			if isW1 {
				wName = wName1
			}
			if !isW0 {
				narrowIdx = 0
			} else if !isW1 {
				narrowIdx = 1
			}
			if wName != "" && narrowIdx >= 0 {
				narrow := t.translateExpr(call.Args[narrowIdx])
				addFn := t.profile.WidenedAddFn
				proLo, proHi := t.profile.SplitPromoteLo, t.profile.SplitPromoteHi
				if t.widenedVars[lhsIdent] {
					// LHS is also widened: update halves directly (avoid demote+combine round-trip)
					t.writef("%s_lo = %s(%s, %s_lo);\n", lhsIdent, addFn, fmt.Sprintf(proLo, narrow), wName)
					t.writef("%s_hi = %s(%s, %s_hi);\n", lhsIdent, addFn, fmt.Sprintf(proHi, narrow), wName)
				} else {
					// LHS is a normal (narrow) variable: demote+combine
					lo := fmt.Sprintf("%s(%s, %s_lo)", addFn, fmt.Sprintf(proLo, narrow), wName)
					hi := fmt.Sprintf("%s(%s, %s_hi)", addFn, fmt.Sprintf(proHi, narrow), wName)
					dLo := fmt.Sprintf(t.profile.DemoteFn, lo)
					dHi := fmt.Sprintf(t.profile.DemoteFn, hi)
					combined := fmt.Sprintf(t.profile.CombineFn, dLo, dHi)
					t.writef("%s = %s;\n", lhsIdent, combined)
				}
				return
			}
		}
	}

	lhsName := t.translateExpr(lhs)

	switch s.Tok {
	case token.DEFINE: // :=
		varInfo := t.inferType(rhs)
		rhsStr := t.translateExpr(rhs)

		// Handle make([]hwy.Vec[T], N) → stack-allocated C array
		if after, ok := strings.CutPrefix(rhsStr, "/* VEC_ARRAY:"); ok {
			// Parse: /* VEC_ARRAY:float32x4_t:4 */
			inner := after
			inner = strings.TrimSuffix(inner, " */")
			parts := strings.SplitN(inner, ":", 2)
			if len(parts) == 2 {
				vecType := parts[0]
				arrLen := parts[1]
				t.vars[lhsName] = cVarInfo{cType: vecType, isVector: true, isPtr: true, arrayLen: arrLen}
				t.writef("%s %s[%s];\n", vecType, lhsName, arrLen)
				return
			}
		}

		// Handle make([]T, size) → C99 VLA for scalar slices
		if after, ok := strings.CutPrefix(rhsStr, "/* SCALAR_ARRAY:"); ok {
			inner := after
			inner = strings.TrimSuffix(inner, " */")
			parts := strings.SplitN(inner, ":", 2)
			if len(parts) == 2 {
				cType := parts[0]
				arrLen := parts[1]
				t.vars[lhsName] = cVarInfo{cType: cType + " *", isPtr: true}
				t.sliceLenVars[lhsName] = arrLen
				t.writef("%s %s[%s];\n", cType, lhsName, arrLen)
				return
			}
		}

		t.vars[lhsName] = varInfo
		t.writef("%s = %s;\n", cDeclVar(varInfo.cType, lhsName), rhsStr)

		// Record constant value for NumLanes/MaxLanes assignments to enable
		// constant-folding in GetLane index expressions (e.g., lanes-1).
		if t.isNumLanesCall(rhs) {
			t.constVars[lhsName] = t.numLanesCallLanes(rhs)
		}

	case token.ASSIGN: // =
		rhsStr := t.translateExpr(rhs)
		// Promoted-type array element writes need demotion (e.g., BF16
		// unsigned short ← float). The RHS is a scalar value that must
		// be converted back to the storage type.
		if t.profile.ScalarDemote != "" && t.isPromotedArrayIndexExpr(lhs) {
			t.writef("%s = %s(%s);\n", lhsName, t.profile.ScalarDemote, rhsStr)
		} else if vi, ok := t.vars[lhsName]; ok && vi.isVector && vi.isPtr && vi.arrayLen != "" {
			// Vector arrays (e.g. float32x4_t rows[lanes]) can't be directly
			// assigned in C. Emit an element-wise copy loop instead.
			t.writef("for (int _ci = 0; _ci < %s; _ci++) %s[_ci] = %s[_ci];\n", vi.arrayLen, lhsName, rhsStr)
		} else {
			t.writef("%s = %s;\n", lhsName, rhsStr)
		}

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
		// Promoted-type array compound assignment: decompose into
		// promote → compute → demote. E.g., BF16 scalar tail:
		//   cRow[j] += aip * bRow[j]
		// becomes:
		//   cRow[j] = f32_scalar_to_bf16(bf16_scalar_to_f32(cRow[j]) + bf16_scalar_to_f32(aip) * bf16_scalar_to_f32(bRow[j]))
		if t.profile.ScalarDemote != "" && t.isPromotedArrayIndexExpr(lhs) {
			rhsStr := t.translatePromotedExpr(rhs)
			promotedLhs := fmt.Sprintf("%s(%s)", t.profile.ScalarPromote, lhsName)
			t.writef("%s = %s(%s + %s);\n", lhsName, t.profile.ScalarDemote, promotedLhs, rhsStr)
		} else {
			rhsStr := t.translateExpr(rhs)
			t.writef("%s += %s;\n", lhsName, rhsStr)
		}

	case token.SUB_ASSIGN: // -=
		if t.profile.ScalarDemote != "" && t.isPromotedArrayIndexExpr(lhs) {
			rhsStr := t.translatePromotedExpr(rhs)
			promotedLhs := fmt.Sprintf("%s(%s)", t.profile.ScalarPromote, lhsName)
			t.writef("%s = %s(%s - %s);\n", lhsName, t.profile.ScalarDemote, promotedLhs, rhsStr)
		} else {
			rhsStr := t.translateExpr(rhs)
			t.writef("%s -= %s;\n", lhsName, rhsStr)
		}

	case token.MUL_ASSIGN: // *=
		if t.profile.ScalarDemote != "" && t.isPromotedArrayIndexExpr(lhs) {
			rhsStr := t.translatePromotedExpr(rhs)
			promotedLhs := fmt.Sprintf("%s(%s)", t.profile.ScalarPromote, lhsName)
			t.writef("%s = %s(%s * %s);\n", lhsName, t.profile.ScalarDemote, promotedLhs, rhsStr)
		} else {
			rhsStr := t.translateExpr(rhs)
			t.writef("%s *= %s;\n", lhsName, rhsStr)
		}

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
			case token.ASSIGN:
				return fmt.Sprintf("%s = %s", lhs, rhs)
			case token.ADD_ASSIGN:
				return fmt.Sprintf("%s += %s", lhs, rhs)
			case token.SUB_ASSIGN:
				return fmt.Sprintf("%s -= %s", lhs, rhs)
			case token.MUL_ASSIGN:
				return fmt.Sprintf("%s *= %s", lhs, rhs)
			case token.QUO_ASSIGN:
				return fmt.Sprintf("%s /= %s", lhs, rhs)
			case token.REM_ASSIGN:
				return fmt.Sprintf("%s %%= %s", lhs, rhs)
			case token.AND_ASSIGN:
				return fmt.Sprintf("%s &= %s", lhs, rhs)
			case token.OR_ASSIGN:
				return fmt.Sprintf("%s |= %s", lhs, rhs)
			case token.XOR_ASSIGN:
				return fmt.Sprintf("%s ^= %s", lhs, rhs)
			case token.SHL_ASSIGN:
				return fmt.Sprintf("%s <<= %s", lhs, rhs)
			case token.SHR_ASSIGN:
				return fmt.Sprintf("%s >>= %s", lhs, rhs)
			}
		}
	case *ast.IncDecStmt:
		name := t.translateExpr(s.X)
		if _, isStar := s.X.(*ast.StarExpr); isStar {
			name = "(" + name + ")"
		}
		if s.Tok == token.INC {
			return name + "++"
		}
		return name + "--"
	}
	return ""
}

// translateRangeStmt translates `for i := range m` to `for (long i = 0; i < m; i++)`.
// Also handles `for i := range s[:n]` → `for (i = 0; i < n; i++)`.
func (t *CASTTranslator) translateRangeStmt(s *ast.RangeStmt) {
	// `for range m` (Go 1.22+) — no loop variable, just iterate m times
	if s.Key == nil {
		rangeOver := t.translateExpr(s.X)
		// When ranging over a slice parameter, use the length variable.
		if ident, ok := s.X.(*ast.Ident); ok {
			if lenVar, ok := t.sliceLenVars[ident.Name]; ok {
				rangeOver = lenVar
			}
		}
		t.writef("#pragma clang loop vectorize(disable) interleave(disable)\n")
		t.writef("for (long _range_i = 0; _range_i < %s; _range_i++) {\n", rangeOver)
		t.indent++
		t.translateBlockStmtContents(s.Body)
		t.indent--
		t.writef("}\n")
		return
	}

	// `for i := range m` → key is i, X is m
	iter := t.translateExpr(s.Key)

	// For `for i := range s[:high]` or `for i := range s[low:high]`,
	// use the slice length (high - low) as the range limit.
	var rangeOver string
	if se, ok := s.X.(*ast.SliceExpr); ok && se.High != nil {
		high := t.translateExpr(se.High)
		if se.Low != nil {
			low := t.translateExpr(se.Low)
			rangeOver = fmt.Sprintf("(%s) - (%s)", high, low)
		} else {
			rangeOver = high
		}
	} else {
		rangeOver = t.translateExpr(s.X)
		// When ranging over a slice parameter, use the length variable
		// instead of the pointer. E.g., `for i := range v` where v is
		// a slice parameter should use `len_v`, not `v` (which is a pointer in C).
		if ident, ok := s.X.(*ast.Ident); ok {
			if lenVar, ok := t.sliceLenVars[ident.Name]; ok {
				rangeOver = lenVar
			}
		}
	}

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

	// Check for copy() calls → memcpy
	if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "copy" {
		if len(call.Args) >= 2 {
			t.emitCopy(call.Args)
			return
		}
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

	// Check for BaseApply(in, out, mathFunc) → inline the load-transform-store loop
	// Matches both unqualified BaseApply(...) and qualified algo.BaseApply(...)
	isBaseApply := false
	if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "BaseApply" {
		isBaseApply = true
	}
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "BaseApply" {
		isBaseApply = true
	}
	if isBaseApply && len(call.Args) == 3 {
		t.emitInlinedBaseApply(call.Args)
		return
	}

	// Generic function call
	expr := t.translateExpr(s.X)
	t.writef("%s;\n", expr)
}

// translateDeclStmt handles `var j int` and `const blockSize = 64`.
func (t *CASTTranslator) translateDeclStmt(s *ast.DeclStmt) {
	genDecl, ok := s.Decl.(*ast.GenDecl)
	if !ok {
		return
	}

	switch genDecl.Tok {
	case token.CONST:
		// Local const declarations → C const or #define
		for _, spec := range genDecl.Specs {
			valueSpec, ok := spec.(*ast.ValueSpec)
			if !ok || len(valueSpec.Values) == 0 {
				continue
			}
			for i, name := range valueSpec.Names {
				if i >= len(valueSpec.Values) {
					break
				}
				val := t.translateExpr(valueSpec.Values[i])
				// Infer type from value
				varInfo := t.inferType(valueSpec.Values[i])
				t.vars[name.Name] = varInfo
				t.writef("const %s = %s;\n", cDeclVar(varInfo.cType, name.Name), val)
			}
		}
	case token.VAR:
		for _, spec := range genDecl.Specs {
			valueSpec, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}

			// Check for fixed-size array types: var result [16]uint8
			if arrType, ok := valueSpec.Type.(*ast.ArrayType); ok && arrType.Len != nil {
				elemGoType := exprToString(arrType.Elt)
				elemCType := t.goTypeToCType(elemGoType)
				size := exprToString(arrType.Len)
				for _, name := range valueSpec.Names {
					t.vars[name.Name] = cVarInfo{cType: elemCType, isArray: true, arrayLen: size}
					// Go zero-initializes arrays; emit = {0} in C
					t.writef("%s %s[%s] = {0};\n", elemCType, name.Name, size)
				}
				continue
			}

			goType := exprToString(valueSpec.Type)
			cType := t.goTypeToCType(goType)
			isVec := strings.HasPrefix(goType, "hwy.Vec[") || strings.HasPrefix(goType, "hwy.Mask[")
			for _, name := range valueSpec.Names {
				// Clear widened status if this var declaration shadows a
				// previously widened variable (e.g., scalar tail loop reusing
				// the same name as a vector loop accumulator).
				delete(t.widenedVars, name.Name)
				t.vars[name.Name] = cVarInfo{cType: cType, isVector: isVec}
				// Go zero-initializes all declared variables; emit = 0 for scalar types
				if isScalarCType(cType) {
					t.writef("%s = 0;\n", cDeclVar(cType, name.Name))
				} else {
					t.writef("%s;\n", cDeclVar(cType, name.Name))
				}
			}
		}
	}
}

// translateIfStmt handles if statements. Skips panic guards (bounds checks).
// Handles init statements like: if remaining := width - i; remaining > 0 { ... }
func (t *CASTTranslator) translateIfStmt(s *ast.IfStmt) {
	// Skip if the body contains only a panic call (bounds check)
	if isPanicGuard(s) {
		return
	}

	// Handle init statement (e.g., if remaining := width - i; remaining > 0)
	if s.Init != nil {
		t.translateStmt(s.Init)
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
// For pointer dereferences (*p), wraps in parens: (*p)++ instead of *p++,
// because C's postfix ++ binds tighter than unary *.
func (t *CASTTranslator) translateIncDecStmt(s *ast.IncDecStmt) {
	name := t.translateExpr(s.X)
	if _, isStar := s.X.(*ast.StarExpr); isStar {
		name = "(" + name + ")"
	}
	if s.Tok == token.INC {
		t.writef("%s++;\n", name)
	} else {
		t.writef("%s--;\n", name)
	}
}

// translateReturnStmt handles `return expr` → `*pout_name = expr; return;`
// In helperMode, emits `return expr;` directly.
func (t *CASTTranslator) translateReturnStmt(s *ast.ReturnStmt) {
	if len(s.Results) == 0 {
		t.writef("return;\n")
		return
	}

	// Helper mode: direct return
	if t.helperMode {
		if len(s.Results) == 1 {
			expr := t.translateExpr(s.Results[0])
			t.writef("return %s;\n", expr)
		} else {
			// Multiple returns not supported in helper mode — emit first
			expr := t.translateExpr(s.Results[0])
			t.writef("return %s;\n", expr)
		}
		return
	}

	// Use declaration-order return names (not alphabetical) so that
	// s.Results[i] maps to the correct output pointer.
	for i, result := range s.Results {
		if i >= len(t.returnOrder) {
			break
		}
		retKey := t.returnOrder[i]
		info := t.params[retKey]

		// Check if this return is an array type (e.g., [4]uint32).
		// Array returns are aliased to the output pointer in emitNamedReturnDecls,
		// so we handle them specially.
		if i < len(t.returnParams) {
			if _, arrSize := parseGoArrayType(t.returnParams[i].Type); arrSize > 0 {
				// If the return expression is a zero-valued composite literal,
				// zero the output buffer. If it's the named return variable
				// itself, the data is already written through the pointer alias.
				if compLit, ok := result.(*ast.CompositeLit); ok && len(compLit.Elts) == 0 {
					for j := range arrSize {
						t.writef("((%s)[%d]) = 0;\n", info.goName, j)
					}
				}
				// Named return variable (e.g., "return values, totalLen"):
				// skip — data already in output buffer via pointer alias.
				continue
			}
		}

		expr := t.translateExpr(result)
		t.writef("*%s = %s;\n", info.cName, expr)
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
		if e.Name == "nil" || e.Name == "false" {
			return "0"
		}
		if e.Name == "true" {
			return "1"
		}
		// Widened accumulator fallback: materialize as demote+combine
		// for any context not explicitly optimized (e.g., passing to unknown func).
		if t.widenedVars[e.Name] {
			lo := fmt.Sprintf(t.profile.DemoteFn, e.Name+"_lo")
			hi := fmt.Sprintf(t.profile.DemoteFn, e.Name+"_hi")
			return fmt.Sprintf(t.profile.CombineFn, lo, hi)
		}
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
	case *ast.CompositeLit:
		// Handle composite literals like [4]uint32{}, [16]uint8{0}
		if len(e.Elts) == 0 {
			// Zero-valued composite literal — emit 0
			return "0"
		}
		// Non-empty composite literal: emit as C array initializer
		var elts []string
		for _, elt := range e.Elts {
			elts = append(elts, t.translateExpr(elt))
		}
		return "{ " + strings.Join(elts, ", ") + " }"
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
		if t.profile.ScalarArithType != "" {
			// Half-precision with native arithmetic: use bare literal
			// (float16_t arithmetic uses native half-precision)
			return lit.Value
		}
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

	// In Go, integer literals are 64-bit on arm64. In C, integer literals
	// like `1` are 32-bit int. `1 << N` for N >= 32 is undefined behavior
	// in C. Suffix with L to force 64-bit when the left operand of a shift
	// is a small integer literal.
	if e.Op == token.SHL || e.Op == token.SHR {
		if lit, ok := e.X.(*ast.BasicLit); ok && lit.Kind == token.INT {
			left = left + "L"
		}
	}

	return left + " " + e.Op.String() + " " + right
}

// translateCallExpr translates function calls, dispatching hwy.* calls to intrinsics.
func (t *CASTTranslator) translateCallExpr(e *ast.CallExpr) string {
	// Check for hwy.Func(...) or hwy.Func[T](...)
	if sel := extractSelectorExpr(e.Fun); sel != nil {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "hwy" {
			// Extract explicit type parameter from generic calls like hwy.LoadSlice[uint8](...)
			var hwyTypeParam string
			if idx, ok := e.Fun.(*ast.IndexExpr); ok {
				hwyTypeParam = exprToString(idx.Index)
			}
			return t.translateHwyCall(sel.Sel.Name, e.Args, hwyTypeParam)
		}
	}

	// Check for struct method calls: obj.Row(y), obj.Width(), obj.Height()
	if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
		if ident, ok := sel.X.(*ast.Ident); ok {
			if info, exists := t.params[ident.Name]; exists && info.isStructPtr {
				return t.translateStructMethodCall(ident.Name, sel.Sel.Name, e.Args)
			}
		}
	}

	// Check for bits.OnesCount* → __builtin_popcount*
	// Check for bits.Len32/Len64 → bit-length via __builtin_clz
	if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "bits" {
			if builtinFn := bitsOnesCountToBuiltin(sel.Sel.Name); builtinFn != "" {
				if len(e.Args) == 1 {
					arg := t.translateExpr(e.Args[0])
					return fmt.Sprintf("%s(%s)", builtinFn, arg)
				}
			}
			if builtinExpr := bitsLenToBuiltin(sel.Sel.Name); builtinExpr != "" {
				if len(e.Args) == 1 {
					arg := t.translateExpr(e.Args[0])
					return fmt.Sprintf(builtinExpr, arg, arg)
				}
			}
		}
	}

	// Check for unsafe.Slice((*T)(unsafe.Pointer(&arr[0])), N) → (C_T *)arr
	// This pattern reinterprets a slice's memory as a different element type.
	if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "unsafe" && sel.Sel.Name == "Slice" {
			if len(e.Args) == 2 {
				return t.translateUnsafeSlice(e.Args[0], e.Args[1])
			}
		}
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "unsafe" && sel.Sel.Name == "Pointer" {
			// unsafe.Pointer(&x) → &x (the C address-of is the same concept)
			if len(e.Args) == 1 {
				return t.translateExpr(e.Args[0])
			}
		}
	}

	// Check for math/stdmath function calls (handles both plain and generic calls)
	if sel := extractSelectorExpr(e.Fun); sel != nil {
		if pkg, ok := sel.X.(*ast.Ident); ok && (pkg.Name == "math" || pkg.Name == "stdmath") {
			name := sel.Sel.Name

			// Single-arg functions that map directly to C stdlib: f64 name / f32 namef.
			// Covers stdmath (Sqrt, Exp, Log, Erf, …) and contrib/math Vec wrappers.
			if cName, ok := mathFuncToC[name]; ok {
				if len(e.Args) == 1 {
					arg := t.translateExpr(e.Args[0])
					// Use GOAT-safe inline helpers when available.
					// expf/erff/sqrtf are C library calls that GOAT can't link,
					// so both contrib/math Base*Vec and stdmath (Exp, Erf, Sqrt)
					// route through _v_/_s_ helpers. Abs/Max/Min map to HW
					// instructions and are not in goatSafeMathHelper.
					if goatSafeMathHelper[cName] {
						// stdmath (Go's math package) always operates on float64,
						// so force _f64 suffix for precision. contrib/math Vec
						// functions use the element type (f16/bf16 promote to f32).
						suffix := goatMathSuffix(t.elemType)
						if pkg.Name == "stdmath" && suffix != "" {
							suffix = "_f64"
						}
						if suffix != "" {
							argInfo := t.inferType(e.Args[0])
							// Promoted vector: split narrow → two f32 halves → compute → recombine
							if t.profile.MathStrategy == "promoted" && argInfo.isVector && t.profile.SplitPromoteLo != "" {
								lo := fmt.Sprintf("_v_%s_f32(%s)", cName, fmt.Sprintf(t.profile.SplitPromoteLo, arg))
								hi := fmt.Sprintf("_v_%s_f32(%s)", cName, fmt.Sprintf(t.profile.SplitPromoteHi, arg))
								return fmt.Sprintf(t.profile.CombineFn, fmt.Sprintf(t.profile.DemoteFn, lo), fmt.Sprintf(t.profile.DemoteFn, hi))
							}
							// Promoted scalar: promote → compute → demote
							if t.profile.MathStrategy == "promoted" && !argInfo.isVector && t.profile.ScalarPromote != "" {
								result := fmt.Sprintf("_s_%s_f32(%s(%s))", cName, t.profile.ScalarPromote, arg)
								if t.profile.ScalarDemote != "" {
									return fmt.Sprintf("%s(%s)", t.profile.ScalarDemote, result)
								}
								return result
							}
							if argInfo.isVector {
								// For promoted types (f16/bf16) calling contrib/math
								// Vec functions: the vector arg is narrow but the
								// helper expects f32. Split-promote, compute, demote.
								if t.profile.MathStrategy == "promoted" && pkg.Name != "stdmath" && t.profile.SplitPromoteLo != "" {
									return t.emitPromotedMathCall(cName, arg)
								}
								return fmt.Sprintf("_v_%s%s(%s)", cName, suffix, arg)
							}
							return fmt.Sprintf("_s_%s%s(%s)", cName, suffix, arg)
						}
					}
					// Use __builtin_ prefix to avoid <math.h> dependency in GOAT C code.
					// At -O3, clang maps __builtin_sqrt to hardware fsqrt on ARM64.
					if t.elemType == "float64" {
						return fmt.Sprintf("__builtin_%s(%s)", cName, arg)
					}
					return fmt.Sprintf("__builtin_%sf(%s)", cName, arg)
				}
			}

			// Special cases that don't fit the single-arg pattern.
			switch name {
			case "Float32bits":
				if len(e.Args) == 1 {
					return fmt.Sprintf("float_to_bits(%s)", t.translateExpr(e.Args[0]))
				}
			case "Float32frombits":
				if len(e.Args) == 1 {
					return fmt.Sprintf("bits_to_float(%s)", t.translateExpr(e.Args[0]))
				}
			case "Abs":
				if len(e.Args) == 1 {
					arg := t.translateExpr(e.Args[0])
					// Use ternary to avoid <math.h> dependency (GOAT-safe).
					return fmt.Sprintf("((%s) < 0 ? -(%s) : (%s))", arg, arg, arg)
				}
			case "Max":
				if len(e.Args) == 2 {
					a, b := t.translateExpr(e.Args[0]), t.translateExpr(e.Args[1])
					// Use ternary to avoid <math.h> dependency (GOAT-safe).
					return fmt.Sprintf("((%s) > (%s) ? (%s) : (%s))", a, b, a, b)
				}
			case "Min":
				if len(e.Args) == 2 {
					a, b := t.translateExpr(e.Args[0]), t.translateExpr(e.Args[1])
					// Use ternary to avoid <math.h> dependency (GOAT-safe).
					return fmt.Sprintf("((%s) < (%s) ? (%s) : (%s))", a, b, a, b)
				}
			case "Inf":
				if len(e.Args) == 1 {
					arg := t.translateExpr(e.Args[0])
					// Use constant expressions to avoid <math.h> dependency (GOAT-safe)
					if strings.HasPrefix(arg, "-") {
						return "(-1.0f/0.0f)"
					}
					return "(1.0f/0.0f)"
				}
			case "BaseSigmoidVec":
				// sigmoid(x) = 1 / (1 + exp(-x))
				// Note: normally handled by the generic mathFuncToC path above.
				// This fallback handles cases where the generic path didn't match.
				if len(e.Args) == 1 {
					arg := t.translateExpr(e.Args[0])
					// Use GOAT-safe inline helpers for all precisions.
					if suffix := goatMathSuffix(t.elemType); suffix != "" {
						argInfo := t.inferType(e.Args[0])
						// Promoted vector: split narrow → two f32 halves → compute → recombine
						if t.profile.MathStrategy == "promoted" && argInfo.isVector && t.profile.SplitPromoteLo != "" {
							lo := fmt.Sprintf("_v_sigmoid_f32(%s)", fmt.Sprintf(t.profile.SplitPromoteLo, arg))
							hi := fmt.Sprintf("_v_sigmoid_f32(%s)", fmt.Sprintf(t.profile.SplitPromoteHi, arg))
							return fmt.Sprintf(t.profile.CombineFn, fmt.Sprintf(t.profile.DemoteFn, lo), fmt.Sprintf(t.profile.DemoteFn, hi))
						}
						// Promoted scalar: promote → compute → demote
						if t.profile.MathStrategy == "promoted" && !argInfo.isVector && t.profile.ScalarPromote != "" {
							result := fmt.Sprintf("_s_sigmoid_f32(%s(%s))", t.profile.ScalarPromote, arg)
							if t.profile.ScalarDemote != "" {
								return fmt.Sprintf("%s(%s)", t.profile.ScalarDemote, result)
							}
							return result
						}
						if argInfo.isVector {
							if t.profile.MathStrategy == "promoted" && t.profile.SplitPromoteLo != "" {
								return t.emitPromotedMathCall("sigmoid", arg)
							}
							return fmt.Sprintf("_v_sigmoid%s(%s)", suffix, arg)
						}
						return fmt.Sprintf("_s_sigmoid%s(%s)", suffix, arg)
					}
					// Fallback for unsupported types.
					if t.elemType == "float64" {
						return fmt.Sprintf("(1.0 / (1.0 + exp(-(%s))))", arg)
					}
					return fmt.Sprintf("(1.0f / (1.0f + expf(-(%s))))", arg)
				}
			case "BaseExp10Vec":
				if len(e.Args) == 1 {
					arg := t.translateExpr(e.Args[0])
					if t.elemType == "float64" {
						return fmt.Sprintf("pow(10.0, %s)", arg)
					}
					return fmt.Sprintf("powf(10.0f, %s)", arg)
				}
			}
		}
	}

	// Check for v.NumLanes() method calls
	if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
		if sel.Sel.Name == "NumLanes" || sel.Sel.Name == "NumElements" {
			return t.lanesExpr()
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
			// len() on a non-slice argument that has no known length mapping.
			// Emit a C compilation error instead of silently producing broken code.
			return fmt.Sprintf("_UNSUPPORTED_LEN_%s /* len(%s) has no C mapping */", arg, arg)
		}
	}

	// Check for min()/max() calls → ternary
	if ident, ok := e.Fun.(*ast.Ident); ok && ident.Name == "min" {
		if len(e.Args) == 2 {
			a := t.translateExpr(e.Args[0])
			b := t.translateExpr(e.Args[1])
			return fmt.Sprintf("((%s) < (%s) ? (%s) : (%s))", a, b, a, b)
		}
	}
	if ident, ok := e.Fun.(*ast.Ident); ok && ident.Name == "max" {
		if len(e.Args) == 2 {
			a := t.translateExpr(e.Args[0])
			b := t.translateExpr(e.Args[1])
			return fmt.Sprintf("((%s) > (%s) ? (%s) : (%s))", a, b, a, b)
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

	// Generic call - just translate arguments.
	// For known helper functions with slice params, append len_<name> args.
	fun := t.translateExpr(e.Fun)
	var args []string
	for _, arg := range e.Args {
		args = append(args, t.translateExpr(arg))
	}
	if ident, ok := e.Fun.(*ast.Ident); ok {
		if sliceIndices, ok := t.helperSliceParams[ident.Name]; ok {
			for _, idx := range sliceIndices {
				if idx < len(e.Args) {
					sliceName := sliceArgBaseName(e.Args[idx])
					if sliceName != "" {
						if lenVar, ok := t.sliceLenVars[sliceName]; ok {
							args = append(args, lenVar)
						}
					}
				}
			}
		}
	}
	return fmt.Sprintf("%s(%s)", fun, strings.Join(args, ", "))
}

// sliceArgBaseName extracts the underlying slice identifier name from an
// argument expression. Handles both simple identifiers (src) and slice
// expressions (src[:n], src[i:]). Returns "" if the base name can't be
// determined.
func sliceArgBaseName(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.SliceExpr:
		if ident, ok := e.X.(*ast.Ident); ok {
			return ident.Name
		}
	}
	return ""
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

// isPromotedArray returns true if expr is an identifier for a slice/pointer
// whose element type is PointerElemType and the profile has scalar
// promote/demote functions. This indicates array elements need explicit
// promote/demote for scalar arithmetic (e.g., BF16 unsigned short ↔ float).
func (t *CASTTranslator) isPromotedArray(expr ast.Expr) bool {
	if t.profile.ScalarPromote == "" || t.profile.PointerElemType == "" {
		return false
	}
	ident, ok := expr.(*ast.Ident)
	if !ok {
		return false
	}
	name := ident.Name
	if info, ok := t.vars[name]; ok && info.isPtr {
		elemType := strings.TrimSuffix(strings.TrimSpace(info.cType), "*")
		return strings.TrimSpace(elemType) == t.profile.PointerElemType
	}
	if info, ok := t.params[name]; ok && info.isSlice {
		elemType := strings.TrimSuffix(strings.TrimSpace(info.cType), "*")
		return strings.TrimSpace(elemType) == t.profile.PointerElemType
	}
	return false
}

// isPromotedArrayIndexExpr returns true if expr is an IndexExpr into a
// promoted-type array (see isPromotedArray).
func (t *CASTTranslator) isPromotedArrayIndexExpr(expr ast.Expr) bool {
	ie, ok := expr.(*ast.IndexExpr)
	if !ok {
		return false
	}
	return t.isPromotedArray(ie.X)
}

// isPromotedScalarVar returns true if the identifier refers to a local variable
// whose C type matches the profile's PointerElemType (i.e., it holds a raw
// promoted-type value like an unsigned short BF16 bit pattern).
func (t *CASTTranslator) isPromotedScalarVar(expr ast.Expr) bool {
	if t.profile.ScalarPromote == "" || t.profile.PointerElemType == "" {
		return false
	}
	ident, ok := expr.(*ast.Ident)
	if !ok {
		return false
	}
	if info, ok := t.vars[ident.Name]; ok {
		return info.cType == t.profile.PointerElemType
	}
	return false
}

// translatePromotedExpr translates an expression while wrapping promoted-type
// leaf values (array elements and scalar variables) with ScalarPromote.
// This is used for scalar tail arithmetic where values stored as unsigned short
// (BF16 bit patterns) need to be promoted to float for computation.
func (t *CASTTranslator) translatePromotedExpr(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.BinaryExpr:
		left := t.translatePromotedExpr(e.X)
		right := t.translatePromotedExpr(e.Y)
		return left + " " + e.Op.String() + " " + right
	case *ast.IndexExpr:
		x := t.translateExpr(e.X)
		idx := t.translateExpr(e.Index)
		result := fmt.Sprintf("%s[%s]", x, idx)
		if t.isPromotedArray(e.X) {
			return fmt.Sprintf("%s(%s)", t.profile.ScalarPromote, result)
		}
		return result
	case *ast.Ident:
		if t.isPromotedScalarVar(e) {
			return fmt.Sprintf("%s(%s)", t.profile.ScalarPromote, e.Name)
		}
		return t.translateExpr(e)
	case *ast.ParenExpr:
		return "(" + t.translatePromotedExpr(e.X) + ")"
	default:
		return t.translateExpr(expr)
	}
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

	// Check if X refers to a struct pointer parameter
	if ident, ok := e.X.(*ast.Ident); ok {
		if info, exists := t.params[ident.Name]; exists && info.isStructPtr {
			// Use -> for struct pointer field access
			return x + "->" + e.Sel.Name
		}
		// Check if X refers to a local struct pointer variable
		if vi, exists := t.vars[ident.Name]; exists && vi.isStructPtr {
			return x + "->" + e.Sel.Name
		}
	}

	return x + "." + e.Sel.Name
}

// translateStructMethodCall translates generic struct method calls to C.
// Convention-based translation:
//   - Methods with 0 args: obj.Method() → obj->methodname (lowercase)
//   - Methods with 1 arg: obj.Method(x) → (obj->data + x * obj->stride)
//
// This is fully generic and works with any struct type following these conventions.
func (t *CASTTranslator) translateStructMethodCall(structName, methodName string, args []ast.Expr) string {
	fieldName := strings.ToLower(methodName)

	if len(args) == 0 {
		// Simple getter: Width() → ->width
		return structName + "->" + fieldName
	}

	if len(args) == 1 {
		// Row-like accessor: Row(y) → (data + y * stride)
		// This assumes the standard 2D array layout pattern
		arg := t.translateExpr(args[0])
		return fmt.Sprintf("(%s->data + %s * %s->stride)", structName, arg, structName)
	}

	// Fallback: emit as method call (will likely cause C compile error)
	var argStrs []string
	for _, arg := range args {
		argStrs = append(argStrs, t.translateExpr(arg))
	}
	return fmt.Sprintf("%s->%s(%s)", structName, fieldName, strings.Join(argStrs, ", "))
}

// translateUnaryExpr translates unary expressions.
func (t *CASTTranslator) translateUnaryExpr(e *ast.UnaryExpr) string {
	operand := t.translateExpr(e.X)
	// In Go, unary ^ is bitwise complement (NOT). In C, the complement
	// operator is ~ (C's ^ is binary XOR only).
	if e.Op == token.XOR {
		return "~" + operand
	}
	return e.Op.String() + operand
}

// ---------------------------------------------------------------------------
// hwy.* call mapping
// ---------------------------------------------------------------------------

// translateHwyCall maps hwy.FuncName to the appropriate C intrinsic.
func (t *CASTTranslator) translateHwyCall(funcName string, args []ast.Expr, typeParam string) string {
	// When an explicit type parameter differs from the profile's element type,
	// use byte-level NEON intrinsics. This handles patterns like
	// hwy.LoadSlice[uint8](...) on a uint32 profile.
	if typeParam != "" && typeParam != t.profile.ElemType {
		if byteExpr := t.emitHwyByteOverride(funcName, args, typeParam); byteExpr != "" {
			return byteExpr
		}
	}

	switch funcName {
	case "Load":
		return t.emitHwyLoad(args)
	case "Store":
		// Store is typically a statement, not expression — but handle both
		return t.emitHwyStoreExpr(args)
	case "Set":
		return t.emitHwySet(args)
	case "Const":
		// hwy.Const[T](val) is semantically Set(ConstValue[T](val)).
		// In C the float32→element-type conversion is implicit, so we
		// just broadcast the literal like Set.
		return t.emitHwySet(args)
	case "Zero":
		return t.emitHwyZero()
	case "MulAdd", "FMA":
		return t.emitHwyMulAdd(args)
	case "ShiftRight":
		return t.emitHwyShiftRight(args)
	case "Add":
		return t.emitHwyBinaryOp(t.profile.AddFn, "+", args)
	case "Sub":
		return t.emitHwyBinaryOp(t.profile.SubFn, "-", args)
	case "Mul":
		return t.emitHwyBinaryOp(t.profile.MulFn, "*", args)
	case "Div":
		return t.emitHwyBinaryOp(t.profile.DivFn, "/", args)
	case "Min":
		return t.emitHwyBinaryOp(t.profile.MinFn, "", args)
	case "Max":
		return t.emitHwyBinaryOp(t.profile.MaxFn, "", args)
	case "Neg":
		return t.emitHwyUnaryOp(t.profile.NegFn, args)
	case "Abs":
		return t.emitHwyUnaryOp(t.profile.AbsFn, args)
	case "Sqrt":
		return t.emitHwyUnaryOp(t.profile.SqrtFn, args)
	case "RSqrt", "InvSqrt":
		return t.emitHwyUnaryOp(t.profile.RSqrtFn, args)
	case "Round":
		return t.emitHwyUnaryOp(t.profile.RoundFn, args)
	case "ConvertToFloat32":
		return t.emitHwyUnaryOp(t.profile.ConvertToFloat32Fn, args)
	case "ConvertToInt32":
		return t.emitHwyUnaryOp(t.profile.ConvertToInt32Fn, args)
	case "Clamp":
		return t.emitHwyClamp(args)
	case "ReduceSum":
		return t.emitHwyReduceSum(args)
	case "InterleaveLower":
		return t.emitHwyBinaryOp(t.profile.InterleaveLowerFn, "", args)
	case "InterleaveUpper":
		return t.emitHwyBinaryOp(t.profile.InterleaveUpperFn, "", args)
	case "And":
		return t.emitHwyBinaryOp(t.profile.AndFn, "&", args)
	case "Or":
		return t.emitHwyBinaryOp(t.profile.OrFn, "|", args)
	case "Xor":
		return t.emitHwyBinaryOp(t.profile.XorFn, "^", args)
	case "PopCount":
		return t.emitHwyUnaryOp(t.profile.PopCountFn, args)
	case "LessThan":
		return t.emitHwyBinaryOp(t.profile.LessThanFn, "<", args)
	case "Equal":
		return t.emitHwyBinaryOp(t.profile.EqualFn, "==", args)
	case "GreaterThan":
		return t.emitHwyBinaryOp(t.profile.GreaterThanFn, ">", args)
	case "GreaterEqual":
		return t.emitHwyBinaryOp(t.profile.GreaterEqualFn, ">=", args)
	case "ReduceMin":
		return t.emitHwyUnaryOp(t.profile.ReduceMinFn, args)
	case "ReduceMax":
		return t.emitHwyUnaryOp(t.profile.ReduceMaxFn, args)
	case "IfThenElse":
		return t.emitHwyIfThenElse(args)
	case "BitsFromMask":
		return t.emitHwyUnaryOp(t.profile.BitsFromMaskFn, args)
	case "TableLookupBytes":
		return t.emitHwyBinaryOp(t.profile.TableLookupBytesFn, "", args)
	case "Load4":
		// Load4 is typically handled as a multi-assign statement; if it appears
		// as an expression, treat it like a single load (the multi-value handling
		// is in translateAssignStmt).
		return t.emitHwyLoad(args)
	case "SlideUpLanes":
		return t.emitHwySlideUpLanes(args)
	case "LoadSlice":
		return t.emitHwyLoad(args) // same semantics as Load for C
	case "StoreSlice":
		// Check if the vector argument is a byte vector (uint8x16_t) and use
		// the byte store intrinsic instead of the profile's store.
		if len(args) >= 1 {
			vecInfo := t.inferType(args[0])
			if vecInfo.cType == "uint8x16_t" {
				vec := t.translateExpr(args[0])
				ptr := t.translateExpr(args[1])
				return fmt.Sprintf("vst1q_u8(%s, %s)", ptr, vec)
			}
		}
		return t.emitHwyStoreExpr(args) // same semantics as Store for C
	case "MaxLanes", "NumLanes":
		if typeParam != "" {
			if lanes := t.lanesForType(typeParam); lanes > 0 {
				return fmt.Sprintf("%d", lanes)
			}
		}
		return t.lanesExpr()
	case "GetLane":
		return t.emitHwyGetLane(args)
	case "DotAccumulate":
		return t.emitHwyDotAccumulate(args)
	case "Pow":
		return t.emitHwyPow(args)
	case "Iota":
		return t.emitHwyIota()
	case "FindFirstTrue":
		return t.emitHwyUnaryScalarOp(t.profile.FindFirstTrueFn, args)
	case "CountTrue":
		return t.emitHwyUnaryScalarOp(t.profile.CountTrueFn, args)
	case "AllTrue":
		return t.emitHwyUnaryScalarOp(t.profile.AllTrueFn, args)
	case "AllFalse":
		return t.emitHwyUnaryScalarOp(t.profile.AllFalseFn, args)
	case "FirstN":
		return t.emitHwyFirstN(args)
	case "MaskAnd":
		return t.emitHwyBinaryOp(t.profile.MaskAndFn, "&", args)
	case "MaskOr":
		return t.emitHwyBinaryOp(t.profile.MaskOrFn, "|", args)
	case "MaskAndNot":
		return t.emitHwyMaskAndNot(args)
	case "CompressStore":
		return t.emitHwyCompressStore(args)
	case "Greater":
		// Greater is an alias for GreaterThan
		return t.emitHwyBinaryOp(t.profile.GreaterThanFn, ">", args)
	case "Merge":
		// Merge(a, b, mask) → IfThenElse(mask, a, b) with reordered args
		if len(args) >= 3 {
			return t.emitHwyIfThenElse([]ast.Expr{args[2], args[0], args[1]})
		}
		return "/* Merge: missing args */"

	// Tile operations
	case "TileZero":
		return t.emitTileZero(args)
	case "OuterProductAdd":
		return t.emitTileOuterProduct(args, true)
	case "OuterProductSub":
		return t.emitTileOuterProduct(args, false)
	case "TileStoreRow":
		return t.emitTileStoreRow(args)
	case "TileReadRow":
		return t.emitTileReadRow(args)
	case "TileLoadCol":
		return t.emitTileLoadCol(args)
	case "NewTile":
		return t.emitTileNew()
	case "TileDim":
		return t.lanesExpr()

	default:
		// Unknown hwy call — emit as-is
		var argStrs []string
		for _, a := range args {
			argStrs = append(argStrs, t.translateExpr(a))
		}
		return fmt.Sprintf("hwy_%s(%s)", strings.ToLower(funcName), strings.Join(argStrs, ", "))
	}
}

// emitHwyByteOverride handles hwy calls with an explicit type parameter that
// differs from the profile's element type (e.g., hwy.LoadSlice[uint8] on a
// uint32 profile). It uses byte-level NEON intrinsics and returns the C
// expression, or "" if this override doesn't apply.
func (t *CASTTranslator) emitHwyByteOverride(funcName string, args []ast.Expr, typeParam string) string {
	// Map Go type → NEON load/store/vec type
	type byteIntrinsics struct {
		vecType string
		loadFn  string
		storeFn string
	}
	byteMap := map[string]byteIntrinsics{
		"uint8": {"uint8x16_t", "vld1q_u8", "vst1q_u8"},
		"byte":  {"uint8x16_t", "vld1q_u8", "vst1q_u8"},
	}
	bi, ok := byteMap[typeParam]
	if !ok {
		return ""
	}

	switch funcName {
	case "LoadSlice", "Load":
		if len(args) < 1 {
			return "/* LoadSlice: missing args */"
		}
		ptr := t.translateExpr(args[0])
		return fmt.Sprintf("%s(%s)", bi.loadFn, ptr)
	case "StoreSlice", "Store":
		if len(args) < 2 {
			return "/* StoreSlice: missing args */"
		}
		vec := t.translateExpr(args[0])
		ptr := t.translateExpr(args[1])
		return fmt.Sprintf("%s(%s, %s)", bi.storeFn, ptr, vec)
	}
	return ""
}

// emitHwyLoad: hwy.Load(slice[off:]) → vld1q_f32(ptr + off)
// SVE: svld1_f32(pg, ptr + off) — predicate is first arg
func (t *CASTTranslator) emitHwyLoad(args []ast.Expr) string {
	if len(args) < 1 {
		return "/* Load: missing args */"
	}
	loadFn := t.profile.LoadFn[t.tier]
	ptr := t.translateExpr(args[0])
	if t.profile.CastExpr != "" {
		ptr = fmt.Sprintf("%s(%s)", t.profile.CastExpr, ptr)
	}
	if t.profile.NeedsPredicate {
		return fmt.Sprintf("%s(pg, %s)", loadFn, ptr)
	}
	return fmt.Sprintf("%s(%s)", loadFn, ptr)
}

// emitHwyStore: hwy.Store(vec, slice[off:]) → vst1q_f32(ptr + off, vec)
// SVE: svst1_f32(pg, ptr + off, vec) — predicate first, then ptr, then vec
func (t *CASTTranslator) emitHwyStore(args []ast.Expr) {
	if len(args) < 2 {
		t.writef("/* Store: missing args */\n")
		return
	}

	// Pattern 4: hwy.Store(hwy.Add(vC, acc), dst) with widened acc → promote+add+store
	if addCall, ok := isHwyCall(args[0], "Add"); ok && len(addCall.Args) >= 2 {
		wName, narrowIdx := "", -1
		for i, arg := range addCall.Args {
			if n, isW := t.isWidenedVar(arg); isW {
				wName = n
				_ = i
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

	// Pattern 3: hwy.Store(acc, dst) where acc is widened → combine+demote+store
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

	storeFn := t.profile.StoreFn[t.tier]
	vec := t.translateExpr(args[0])
	ptr := t.translateExpr(args[1])
	if t.profile.CastExpr != "" {
		ptr = fmt.Sprintf("%s(%s)", t.profile.CastExpr, ptr)
	}
	if t.profile.NeedsPredicate {
		// SVE Store: svst1_f32(pg, ptr, vec)
		t.writef("%s(pg, %s, %s);\n", storeFn, ptr, vec)
	} else {
		// NEON/AVX Store: vst1q_f32(ptr, vec)
		t.writef("%s(%s, %s);\n", storeFn, ptr, vec)
	}
}

// emitHwyStoreExpr returns Store as a string expression (for edge cases).
func (t *CASTTranslator) emitHwyStoreExpr(args []ast.Expr) string {
	if len(args) < 2 {
		return "/* Store: missing args */"
	}
	storeFn := t.profile.StoreFn[t.tier]
	vec := t.translateExpr(args[0])
	ptr := t.translateExpr(args[1])
	if t.profile.CastExpr != "" {
		ptr = fmt.Sprintf("%s(%s)", t.profile.CastExpr, ptr)
	}
	if t.profile.NeedsPredicate {
		return fmt.Sprintf("%s(pg, %s, %s)", storeFn, ptr, vec)
	}
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
// BF16: uses dedicated bf16_zero_q() / avx512_bf16_zero() helpers.
func (t *CASTTranslator) emitHwyZero() string {
	// BF16 has dedicated zero helpers that avoid type mismatch with DupFn.
	if t.elemType == "hwy.BFloat16" || t.elemType == "bfloat16" {
		switch t.profile.TargetName {
		case "AVX512":
			return "avx512_bf16_zero()"
		default:
			return "bf16_zero_q()"
		}
	}

	dupFn := t.profile.DupFn[t.tier]
	var zero string
	switch t.elemType {
	case "float64":
		zero = "0.0"
	case "float32":
		zero = "0.0f"
	default:
		if t.profile.ScalarArithType != "" {
			// Half-precision with native arithmetic (e.g., NEON f16):
			// use 0.0 as the zero literal for the dup intrinsic
			zero = "0.0"
		} else {
			// Integer types: use plain 0
			zero = "0"
		}
	}
	return fmt.Sprintf("%s(%s)", dupFn, zero)
}

// emitHwyMulAdd: hwy.MulAdd(a, b, acc) → target-specific FMA.
// Go convention (like AVX): MulAdd(a, b, acc) = a*b + acc
// NEON: vfmaq_f32(acc, a, b) — accumulator first
// AVX: _mm256_fmadd_ps(a, b, acc) — accumulator last
// SVE: svmla_f32_x(pg, acc, a, b) — predicate first, accumulator second
func (t *CASTTranslator) emitHwyMulAdd(args []ast.Expr) string {
	if len(args) < 3 {
		return "/* MulAdd: missing args */"
	}
	fmaFn := t.profile.FmaFn[t.tier]
	a := t.translateExpr(args[0])
	b := t.translateExpr(args[1])
	acc := t.translateExpr(args[2])

	if t.profile.NeedsPredicate {
		// SVE: svmla_f32_x(pg, acc, a, b) — predicate first, then acc_first order
		return fmt.Sprintf("%s(pg, %s, %s, %s)", fmaFn, acc, a, b)
	}
	if t.profile.FmaArgOrder == "acc_first" {
		// NEON: FMA(acc, a, b)
		return fmt.Sprintf("%s(%s, %s, %s)", fmaFn, acc, a, b)
	}
	// AVX: FMA(a, b, acc)
	return fmt.Sprintf("%s(%s, %s, %s)", fmaFn, a, b, acc)
}

// emitHwyShiftRight: hwy.ShiftRight(v, n) → vshrq_n_s32(v, n) for NEON
// The shift amount must be a compile-time constant.
func (t *CASTTranslator) emitHwyShiftRight(args []ast.Expr) string {
	if len(args) < 2 {
		return "/* ShiftRight: missing args */"
	}
	v := t.translateExpr(args[0])
	n := t.translateExpr(args[1])

	// Select intrinsic based on target and element type
	var fn string
	needsPg := false
	switch t.profile.TargetName {
	case "NEON":
		switch t.elemType {
		case "int32":
			fn = "vshrq_n_s32"
		case "int64":
			fn = "vshrq_n_s64"
		case "uint32":
			fn = "vshrq_n_u32"
		case "uint64":
			fn = "vshrq_n_u64"
		default:
			fn = "vshrq_n_s32" // fallback
		}
	case "SVE_DARWIN", "SVE_LINUX":
		needsPg = true
		switch t.elemType {
		case "int32":
			fn = "svasr_n_s32_x"
		case "int64":
			fn = "svasr_n_s64_x"
		case "uint32":
			fn = "svlsr_n_u32_x"
		case "uint64":
			fn = "svlsr_n_u64_x"
		default:
			fn = "svasr_n_s32_x" // fallback
		}
	case "AVX2":
		switch t.elemType {
		case "int32":
			fn = "_mm256_srai_epi32"
		case "int64":
			fn = "_mm256_srai_epi64" // AVX-512 only for 64-bit
		default:
			fn = "_mm256_srai_epi32"
		}
	case "AVX512":
		switch t.elemType {
		case "int32":
			fn = "_mm512_srai_epi32"
		case "int64":
			fn = "_mm512_srai_epi64"
		default:
			fn = "_mm512_srai_epi32"
		}
	default:
		fn = "vshrq_n_s32" // fallback to NEON
	}

	if needsPg {
		return fmt.Sprintf("%s(pg, %s, %s)", fn, v, n)
	}
	return fmt.Sprintf("%s(%s, %s)", fn, v, n)
}

// emitCopy emits a for-loop to copy array elements.
// copy(dst, src) in Go copies min(len(dst), len(src)) elements.
// We emit a for-loop since GOAT doesn't support memcpy calls directly.
//
// When the destination is a slice expression with explicit bounds (e.g.
// output[m*N:(m+1)*N]), the copy length is computed from the bounds.
// Otherwise, it defaults to `lanes` elements, which is safe for
// SIMD-width buffers.
func (t *CASTTranslator) emitCopy(args []ast.Expr) {
	if len(args) < 2 {
		t.writef("/* copy: missing args */\n")
		return
	}
	dst := t.translateExpr(args[0])
	src := t.translateExpr(args[1])

	// Wrap pointer expressions with parentheses to ensure correct indexing
	// e.g., "rRow + i" → "(rRow + i)[_ci]"
	if strings.Contains(dst, "+") || strings.Contains(dst, "-") {
		dst = "(" + dst + ")"
	}
	if strings.Contains(src, "+") || strings.Contains(src, "-") {
		src = "(" + src + ")"
	}

	// Determine copy length: use slice bounds if available, else lanes.
	copyLen := fmt.Sprintf("%d", t.lanes)
	if se, ok := args[0].(*ast.SliceExpr); ok && se.Low != nil && se.High != nil {
		lo := t.translateExpr(se.Low)
		hi := t.translateExpr(se.High)
		copyLen = fmt.Sprintf("(%s) - (%s)", hi, lo)
	} else if se, ok := args[1].(*ast.SliceExpr); ok && se.Low != nil && se.High != nil {
		lo := t.translateExpr(se.Low)
		hi := t.translateExpr(se.High)
		copyLen = fmt.Sprintf("(%s) - (%s)", hi, lo)
	}

	t.writef("for (long _ci = 0; _ci < %s; _ci++) { %s[_ci] = %s[_ci]; }\n",
		copyLen, dst, src)
}

// emitHwyBinaryOp: hwy.Add(a, b) → vaddq_f32(a, b)
// SVE: svadd_f32_x(pg, a, b) — predicate first
// fallbackOp is the C operator to use when no SIMD intrinsic is available
// (e.g. "+" for Add, "*" for Mul). Empty string means no fallback.
func (t *CASTTranslator) emitHwyBinaryOp(fnMap map[string]string, fallbackOp string, args []ast.Expr) string {
	if len(args) < 2 {
		return "/* binary op: missing args */"
	}
	a := t.translateExpr(args[0])
	b := t.translateExpr(args[1])
	fn := fnMap[t.tier]
	if fn == "" {
		// No SIMD intrinsic available — fall back to C operator.
		// This handles cases like int64 multiply on NEON (no vmulq_s64).
		if fallbackOp == "" {
			fallbackOp = "*"
		}
		return fmt.Sprintf("(%s) %s (%s)", a, fallbackOp, b)
	}
	if t.profile.NeedsPredicate {
		return fmt.Sprintf("%s(pg, %s, %s)", fn, a, b)
	}
	return fmt.Sprintf("%s(%s, %s)", fn, a, b)
}

// emitHwyUnaryOp: hwy.Neg(x) → vnegq_f32(x)
// SVE: svneg_f32_x(pg, x) — predicate first
func (t *CASTTranslator) emitHwyUnaryOp(fnMap map[string]string, args []ast.Expr) string {
	if len(args) < 1 {
		return "/* unary op: missing args */"
	}
	fn := fnMap[t.tier]
	x := t.translateExpr(args[0])
	if t.profile.NeedsPredicate {
		return fmt.Sprintf("%s(pg, %s)", fn, x)
	}
	return fmt.Sprintf("%s(%s)", fn, x)
}

// emitHwyReduceSum: hwy.ReduceSum(v) → vaddvq_f32(v) (returns scalar, not vector)
// SVE: svaddv_f32(pg, v) — predicate is first arg
func (t *CASTTranslator) emitHwyReduceSum(args []ast.Expr) string {
	if len(args) < 1 {
		return "/* ReduceSum: missing args */"
	}
	fn := t.profile.ReduceSumFn[t.tier]
	v := t.translateExpr(args[0])
	if t.profile.NeedsPredicate {
		return fmt.Sprintf("%s(pg, %s)", fn, v)
	}
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

// emitHwyClamp: hwy.Clamp(v, lo, hi) → max(min(v, hi), lo)
// Composed from Min and Max intrinsics.
func (t *CASTTranslator) emitHwyClamp(args []ast.Expr) string {
	if len(args) < 3 {
		return "/* Clamp: missing args */"
	}
	minFn := t.profile.MinFn[t.tier]
	maxFn := t.profile.MaxFn[t.tier]
	v := t.translateExpr(args[0])
	lo := t.translateExpr(args[1])
	hi := t.translateExpr(args[2])

	if minFn == "" || maxFn == "" {
		// Scalar fallback
		return fmt.Sprintf("((%s) < (%s) ? (%s) : ((%s) > (%s) ? (%s) : (%s)))", v, lo, lo, v, hi, hi, v)
	}
	if t.profile.NeedsPredicate {
		inner := fmt.Sprintf("%s(pg, %s, %s)", minFn, v, hi)
		return fmt.Sprintf("%s(pg, %s, %s)", maxFn, inner, lo)
	}
	inner := fmt.Sprintf("%s(%s, %s)", minFn, v, hi)
	return fmt.Sprintf("%s(%s, %s)", maxFn, inner, lo)
}

// emitHwyGetLane: hwy.GetLane(v, idx) → vgetq_lane_f32(v, idx)
// SVE: svlasta_f32(pg, v) — predicate is first arg, no index (extracts first active element)
// For variable (non-literal) indices, emits a store-to-stack pattern.
func (t *CASTTranslator) emitHwyGetLane(args []ast.Expr) string {
	if len(args) < 2 {
		return "/* GetLane: missing args */"
	}
	fn := t.profile.GetLaneFn[t.tier]
	v := t.translateExpr(args[0])
	if t.profile.NeedsPredicate {
		// SVE: svlasta_f32(pg, v) — index is implicit (first active element)
		return fmt.Sprintf("%s(pg, %s)", fn, v)
	}
	idx := t.translateExpr(args[1])
	return fmt.Sprintf("%s(%s, %s)", fn, v, idx)
}

// emitHwySlideUpLanes: hwy.SlideUpLanes(v, offset) → vextq_f32(zero, v, NumLanes-offset)
// NEON semantics: vextq extracts elements from two vectors. Using (zero, v, N-offset)
// effectively shifts lanes up by offset, filling low lanes with zeros.
// The third argument must be a compile-time constant.
func (t *CASTTranslator) emitHwySlideUpLanes(args []ast.Expr) string {
	if len(args) < 2 {
		return "/* SlideUpLanes: missing args */"
	}
	vec := t.translateExpr(args[0])
	offsetExpr := args[1]

	extFn := ""
	if t.profile.SlideUpExtFn != nil {
		extFn = t.profile.SlideUpExtFn[t.tier]
	}

	// NEON path: literal offset → direct vextq
	if extFn != "" {
		if lit, ok := offsetExpr.(*ast.BasicLit); ok && lit.Kind == token.INT {
			offsetInt := 0
			fmt.Sscanf(lit.Value, "%d", &offsetInt)
			if offsetInt <= 0 {
				return vec
			}
			if offsetInt >= t.lanes {
				return t.emitHwyZero()
			}
			complement := t.lanes - offsetInt
			zero := t.emitHwyZero()
			return fmt.Sprintf("%s(%s, %s, %d)", extFn, zero, vec, complement)
		}
	}

	// Fallback placeholder for AVX / non-literal offsets
	offset := t.translateExpr(offsetExpr)
	return fmt.Sprintf("/* SlideUpLanes: fallback for offset=%s */", offset)
}

// emitHwyDotAccumulate: hwy.DotAccumulate(a, b, acc) → vbfdotq_f32(acc, a, b)
// BFDOT/VDPBF16PS: pairwise dot-product accumulation of BF16 into F32.
// Both NEON and AVX-512 use (acc, a, b) argument order.
func (t *CASTTranslator) emitHwyDotAccumulate(args []ast.Expr) string {
	if len(args) < 3 {
		return "/* DotAccumulate: missing args */"
	}
	fn := t.profile.DotAccFn[t.tier]
	if fn == "" {
		return "/* DotAccumulate: no intrinsic for this target */"
	}
	a := t.translateExpr(args[0])
	b := t.translateExpr(args[1])
	acc := t.translateExpr(args[2])
	// Both NEON (vbfdotq_f32) and AVX-512 (_mm512_dpbf16_ps) use (acc, a, b)
	return fmt.Sprintf("%s(%s, %s, %s)", fn, acc, a, b)
}

// emitHwyPow handles hwy.Pow(base, exp) → _v_pow_f32/f64 or promoted split.
// For native types (f32, f64), emits _v_pow_<prec>(base, exp) using the
// GOAT-safe inline helpers (exp(exp * log(base))).
// For promoted types (f16, bf16), emits split-promote → _v_pow_f32 → combine.
func (t *CASTTranslator) emitHwyPow(args []ast.Expr) string {
	if len(args) < 2 {
		return "/* Pow: missing args */"
	}
	base := t.translateExpr(args[0])
	exp := t.translateExpr(args[1])
	baseInfo := t.inferType(args[0])

	// For promoted types (f16, bf16), split-promote → compute in f32 → combine
	if t.profile.MathStrategy == "promoted" && baseInfo.isVector && t.profile.SplitPromoteLo != "" {
		proLo := t.profile.SplitPromoteLo
		proHi := t.profile.SplitPromoteHi
		demoteFn := t.profile.DemoteFn
		combineFn := t.profile.CombineFn

		baseLo := fmt.Sprintf(proLo, base)
		baseHi := fmt.Sprintf(proHi, base)
		expLo := fmt.Sprintf(proLo, exp)
		expHi := fmt.Sprintf(proHi, exp)

		lo := fmt.Sprintf("_v_pow_f32(%s, %s)", baseLo, expLo)
		hi := fmt.Sprintf("_v_pow_f32(%s, %s)", baseHi, expHi)

		dLo := fmt.Sprintf(demoteFn, lo)
		dHi := fmt.Sprintf(demoteFn, hi)

		return fmt.Sprintf(combineFn, dLo, dHi)
	}

	// For promoted scalar (e.g., bf16 scalar tail)
	if t.profile.MathStrategy == "promoted" && !baseInfo.isVector && t.profile.ScalarPromote != "" {
		pBase := fmt.Sprintf("%s(%s)", t.profile.ScalarPromote, base)
		pExp := fmt.Sprintf("%s(%s)", t.profile.ScalarPromote, exp)
		result := fmt.Sprintf("_s_pow_f32(%s, %s)", pBase, pExp)
		if t.profile.ScalarDemote != "" {
			return fmt.Sprintf("%s(%s)", t.profile.ScalarDemote, result)
		}
		return result
	}

	// For native types (f32, f64), use direct GOAT-safe helpers
	suffix := goatMathSuffix(t.elemType)
	if suffix != "" {
		if baseInfo.isVector {
			return fmt.Sprintf("_v_pow%s(%s, %s)", suffix, base, exp)
		}
		return fmt.Sprintf("_s_pow%s(%s, %s)", suffix, base, exp)
	}

	// Fallback
	return fmt.Sprintf("pow(%s, %s)", base, exp)
}

// emitPromotedMathCall handles a math Vec→Vec function call when the current
// profile uses MathStrategy=="promoted" (f16/bf16). The narrow vector argument
// is split-promoted to two f32 halves, the math function is applied to each,
// and the results are demoted and combined back to the narrow type.
//
// This emits pre-statements via t.writef() and returns the name of the result
// variable. The caller's assignment (e.g., erfX := ...) will then assign from
// this variable.
func (t *CASTTranslator) emitPromotedMathCall(mathCName, arg string) string {
	tc := t.tmpCount
	t.tmpCount++

	proLo := t.profile.SplitPromoteLo
	proHi := t.profile.SplitPromoteHi
	demoteFn := t.profile.DemoteFn
	combineFn := t.profile.CombineFn
	vecType := t.profile.VecTypes[t.tier]

	t.writef("float32x4_t _pm_lo_%d = %s;\n", tc, fmt.Sprintf(proLo, arg))
	t.writef("float32x4_t _pm_hi_%d = %s;\n", tc, fmt.Sprintf(proHi, arg))
	t.writef("_pm_lo_%d = _v_%s_f32(_pm_lo_%d);\n", tc, mathCName, tc)
	t.writef("_pm_hi_%d = _v_%s_f32(_pm_hi_%d);\n", tc, mathCName, tc)
	dLo := fmt.Sprintf(demoteFn, fmt.Sprintf("_pm_lo_%d", tc))
	dHi := fmt.Sprintf(demoteFn, fmt.Sprintf("_pm_hi_%d", tc))
	t.writef("%s _pm_result_%d = %s;\n", vecType, tc, fmt.Sprintf(combineFn, dLo, dHi))

	return fmt.Sprintf("_pm_result_%d", tc)
}

// emitInlinedBaseApply inlines a BaseApply(in, out, math.FooVec) call as a
// load→math→store C loop. This eliminates per-vector function call overhead
// by compiling the entire loop into GOAT assembly.
//
// For native types (f32, f64): straightforward load → math → store.
// For promoted types (f16, bf16): split-promote → f32 math → demote → combine → store.
func (t *CASTTranslator) emitInlinedBaseApply(args []ast.Expr) {
	// Extract math function name from args[2] (e.g., math.BaseExpVec → "exp")
	mathCName := ""
	if sel, ok := args[2].(*ast.SelectorExpr); ok {
		if cName, ok := mathFuncToC[sel.Sel.Name]; ok {
			mathCName = cName
		}
	}
	// Also handle generic form: math.BaseExpVec[T]
	if indexExpr, ok := args[2].(*ast.IndexExpr); ok {
		if sel, ok := indexExpr.X.(*ast.SelectorExpr); ok {
			if cName, ok := mathFuncToC[sel.Sel.Name]; ok {
				mathCName = cName
			}
		}
	}
	if mathCName == "" {
		t.writef("/* emitInlinedBaseApply: could not resolve math function */\n")
		return
	}

	// Get param names for in, out
	inName := t.translateExpr(args[0])
	outName := t.translateExpr(args[1])

	// Get length variable names from sliceLenVars (handles shared length optimization
	// where all slices map to a single length variable like "len_in").
	inLen := t.sliceLenVars[inName]
	outLen := t.sliceLenVars[outName]
	if inLen == "" {
		inLen = "len_" + inName
	}
	if outLen == "" {
		outLen = "len_" + outName
	}

	lanes := t.lanes
	vecType := t.profile.VecTypes[t.tier]
	loadFn := t.profile.LoadFn[t.tier]
	storeFn := t.profile.StoreFn[t.tier]
	suffix := goatMathSuffix(t.elemType)
	castExpr := t.profile.CastExpr

	t.writef("{\n")
	t.indent++
	t.writef("long n = (%s < %s) ? %s : %s;\n", inLen, outLen, inLen, outLen)
	t.writef("long i = 0;\n")

	if t.profile.MathStrategy == "promoted" && t.profile.SplitPromoteLo != "" {
		// Promoted path (f16, bf16): split-promote → f32 math → demote → combine
		proLo := t.profile.SplitPromoteLo
		proHi := t.profile.SplitPromoteHi
		demoteFn := t.profile.DemoteFn
		combineFn := t.profile.CombineFn

		t.writef("for (; i + %d <= n; i += %d) {\n", lanes, lanes)
		t.indent++
		loadPtr := fmt.Sprintf("%s + i", inName)
		if castExpr != "" {
			loadPtr = fmt.Sprintf("%s(%s + i)", castExpr, inName)
		}
		t.writef("%s narrow = %s(%s);\n", vecType, loadFn, loadPtr)
		t.writef("float32x4_t lo = %s;\n", fmt.Sprintf(proLo, "narrow"))
		t.writef("float32x4_t hi = %s;\n", fmt.Sprintf(proHi, "narrow"))
		t.writef("lo = _v_%s_f32(lo);\n", mathCName)
		t.writef("hi = _v_%s_f32(hi);\n", mathCName)
		dLo := fmt.Sprintf(demoteFn, "lo")
		dHi := fmt.Sprintf(demoteFn, "hi")
		t.writef("%s result = %s;\n", vecType, fmt.Sprintf(combineFn, dLo, dHi))
		storePtr := fmt.Sprintf("%s + i", outName)
		if castExpr != "" {
			storePtr = fmt.Sprintf("%s(%s + i)", castExpr, outName)
		}
		t.writef("%s(%s, result);\n", storeFn, storePtr)
		t.indent--
		t.writef("}\n")
		// Scalar tail with promotion/demotion
		scalarPromote := t.profile.ScalarPromote
		scalarDemote := t.profile.ScalarDemote
		t.writef("for (; i < n; i++) {\n")
		t.indent++
		if scalarPromote != "" && scalarDemote != "" {
			t.writef("%s[i] = %s(_s_%s_f32(%s(%s[i])));\n",
				outName, scalarDemote, mathCName, scalarPromote, inName)
		} else {
			// Fallback: direct scalar (shouldn't happen for promoted types)
			t.writef("%s[i] = _s_%s%s(%s[i]);\n", outName, mathCName, suffix, inName)
		}
		t.indent--
		t.writef("}\n")
	} else {
		// Native path (f32, f64): direct load → math → store
		t.writef("for (; i + %d <= n; i += %d) {\n", lanes, lanes)
		t.indent++
		loadPtr := fmt.Sprintf("%s + i", inName)
		if castExpr != "" {
			loadPtr = fmt.Sprintf("%s(%s + i)", castExpr, inName)
		}
		t.writef("%s x = %s(%s);\n", vecType, loadFn, loadPtr)
		t.writef("x = _v_%s%s(x);\n", mathCName, suffix)
		storePtr := fmt.Sprintf("%s + i", outName)
		if castExpr != "" {
			storePtr = fmt.Sprintf("%s(%s + i)", castExpr, outName)
		}
		t.writef("%s(%s, x);\n", storeFn, storePtr)
		t.indent--
		t.writef("}\n")
		// Scalar tail
		t.writef("for (; i < n; i++) {\n")
		t.indent++
		t.writef("%s[i] = _s_%s%s(%s[i]);\n", outName, mathCName, suffix, inName)
		t.indent--
		t.writef("}\n")
	}

	t.indent--
	t.writef("}\n")
}

// emitHwyIota: hwy.Iota[T]() → hwy_iota_f32() or hwy_iota_f64()
func (t *CASTTranslator) emitHwyIota() string {
	iotaFn := t.profile.IotaFn[t.tier]
	if iotaFn == "" {
		return "/* Iota: not supported for this profile */"
	}
	return fmt.Sprintf("%s()", iotaFn)
}

// emitHwyUnaryScalarOp emits a unary operation that returns a scalar (long).
// Used for AllTrue, AllFalse, FindFirstTrue, CountTrue.
func (t *CASTTranslator) emitHwyUnaryScalarOp(fnMap map[string]string, args []ast.Expr) string {
	if len(args) < 1 {
		return "/* unary scalar op: missing args */"
	}
	fn := fnMap[t.tier]
	if fn == "" {
		return fmt.Sprintf("/* unary scalar op: not supported for tier %s */", t.tier)
	}
	arg := t.translateExpr(args[0])
	return fmt.Sprintf("%s(%s)", fn, arg)
}

// emitHwyFirstN: hwy.FirstN[T](n) → hwy_first_n_u32(n)
func (t *CASTTranslator) emitHwyFirstN(args []ast.Expr) string {
	if len(args) < 1 {
		return "/* FirstN: missing args */"
	}
	fn := t.profile.FirstNFn[t.tier]
	if fn == "" {
		return "/* FirstN: not supported for this profile */"
	}
	n := t.translateExpr(args[0])
	return fmt.Sprintf("%s(%s)", fn, n)
}

// emitHwyMaskAndNot: hwy.MaskAndNot(a, b) → vbicq_u32(b, a)
// Note: hwy.MaskAndNot(mask, notMask) returns mask AND (NOT notMask),
// but NEON vbic(a, b) = a AND (NOT b), so args are swapped.
func (t *CASTTranslator) emitHwyMaskAndNot(args []ast.Expr) string {
	if len(args) < 2 {
		return "/* MaskAndNot: missing args */"
	}
	fn := t.profile.MaskAndNotFn[t.tier]
	if fn == "" {
		return "/* MaskAndNot: not supported for this profile */"
	}
	a := t.translateExpr(args[0])
	b := t.translateExpr(args[1])
	// hwy.MaskAndNot(mask, notMask) = notMask AND NOT mask
	// vbicq(a, b) = a AND NOT b
	// So: vbicq(notMask, mask)
	return fmt.Sprintf("%s(%s, %s)", fn, b, a)
}

// emitHwyCompressStore: hwy.CompressStore(vec, mask, slice[off:]) → hwy_compress_store_f32(vec, mask, ptr)
func (t *CASTTranslator) emitHwyCompressStore(args []ast.Expr) string {
	if len(args) < 3 {
		return "/* CompressStore: missing args */"
	}
	fn := t.profile.CompressStoreFn[t.tier]
	if fn == "" {
		return "/* CompressStore: not supported for this profile */"
	}
	vec := t.translateExpr(args[0])
	mask := t.translateExpr(args[1])
	ptr := t.translateExpr(args[2])
	if t.profile.CastExpr != "" {
		ptr = fmt.Sprintf("%s(%s)", t.profile.CastExpr, ptr)
	}
	return fmt.Sprintf("%s(%s, %s, %s)", fn, vec, mask, ptr)
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
	for i := range 4 {
		names[i] = t.translateExpr(lhs[i])
	}

	if x4Type, ok := t.profile.VecX4Type[t.tier]; ok && x4Type != "" {
		// NEON path: use vld1q_*_x4 multi-load
		load4Fn := t.profile.Load4Fn[t.tier]
		tmpName := fmt.Sprintf("_load4_%d", t.tmpCount)
		t.tmpCount++
		t.writef("%s %s = %s(%s);\n", x4Type, tmpName, load4Fn, ptr)
		for i := range 4 {
			if tok == token.DEFINE {
				t.vars[names[i]] = cVarInfo{cType: vecType, isVector: true}
				t.writef("%s %s = %s.val[%d];\n", vecType, names[i], tmpName, i)
			} else {
				t.writef("%s = %s.val[%d];\n", names[i], tmpName, i)
			}
		}
	} else {
		// AVX/SVE fallback: 4 individual loads
		loadFn := t.profile.LoadFn[t.tier]
		for i := range 4 {
			var loadExpr string
			ptrExpr := ptr
			if i > 0 {
				ptrExpr = fmt.Sprintf("%s + %d", ptr, i*t.lanes)
			}
			if t.profile.NeedsPredicate {
				loadExpr = fmt.Sprintf("%s(pg, %s)", loadFn, ptrExpr)
			} else {
				loadExpr = fmt.Sprintf("%s(%s)", loadFn, ptrExpr)
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

// translateICTCoeffsAssign handles multi-value assignments from ictCoeffs[T]().
// These are the ICT (Irreversible Color Transform) coefficients for JPEG 2000.
// The function inlines the coefficient values based on the element type.
func (t *CASTTranslator) translateICTCoeffsAssign(lhs []ast.Expr, call *ast.CallExpr, tok token.Token) bool {
	// Check if this is an ictCoeffs call
	funcExpr := call.Fun
	// Handle IndexExpr for ictCoeffs[T]
	if idx, ok := funcExpr.(*ast.IndexExpr); ok {
		if ident, ok := idx.X.(*ast.Ident); ok && ident.Name == "ictCoeffs" {
			// This is ictCoeffs[T]() - inline the coefficients
			return t.emitICTCoeffsAssign(lhs, tok)
		}
	}
	// Handle Ident for ictCoeffs (without type param)
	if ident, ok := funcExpr.(*ast.Ident); ok && ident.Name == "ictCoeffs" {
		return t.emitICTCoeffsAssign(lhs, tok)
	}
	return false
}

// emitICTCoeffsAssign emits constant assignments for ICT coefficients.
// ICT coefficients from ITU-T T.800:
//
//	Forward: rToY=0.299, gToY=0.587, bToY=0.114, rToCb=-0.16875, gToCb=-0.33126, bToCb=0.5,
//	         rToCr=0.5, gToCr=-0.41869, bToCr=-0.08131
//	Inverse: crToR=1.402, cbToG=-0.344136, crToG=-0.714136, cbToB=1.772
func (t *CASTTranslator) emitICTCoeffsAssign(lhs []ast.Expr, tok token.Token) bool {
	if len(lhs) < 13 {
		return false
	}

	coeffs := []float64{
		0.299,     // rToY
		0.587,     // gToY
		0.114,     // bToY
		-0.16875,  // rToCb
		-0.33126,  // gToCb
		0.5,       // bToCb
		0.5,       // rToCr
		-0.41869,  // gToCr
		-0.08131,  // bToCr
		1.402,     // crToR
		-0.344136, // cbToG
		-0.714136, // crToG
		1.772,     // cbToB
	}

	// Use ScalarArithType if available (e.g., float16_t for NEON f16),
	// otherwise fall back to CType.
	cType := t.profile.CType
	if t.profile.ScalarArithType != "" {
		cType = t.profile.ScalarArithType
	}

	for i := range 13 {
		// Handle blank identifier _
		ident, ok := lhs[i].(*ast.Ident)
		if !ok || ident.Name == "_" {
			continue
		}
		name := ident.Name
		var valStr string
		switch t.elemType {
		case "float32":
			valStr = fmt.Sprintf("%.7gf", coeffs[i])
		case "float64":
			valStr = fmt.Sprintf("%.15g", coeffs[i])
		default:
			// Half-precision types: cast from double to the scalar type
			if t.profile.ScalarArithType != "" {
				valStr = fmt.Sprintf("(%s)%.15g", t.profile.ScalarArithType, coeffs[i])
			} else {
				valStr = fmt.Sprintf("%.15g", coeffs[i])
			}
		}
		if tok == token.DEFINE {
			t.vars[name] = cVarInfo{cType: cType}
			t.writef("%s %s = %s;\n", cType, name, valStr)
		} else {
			t.writef("%s = %s;\n", name, valStr)
		}
	}
	return true
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
	if t.profile.CastExpr != "" {
		t.writef("%s(%s_getlane_buf, %s);\n", storeFn, t.profile.CastExpr, vec)
	} else {
		t.writef("%s((%s *)_getlane_buf, %s);\n", storeFn, cType, vec)
	}

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

	// Second arg is the length. Try constant-folding first so that
	// make([]T, lanes) produces a fixed-size C array (e.g. int buf[4])
	// instead of a VLA (int buf[lanes]). Clang -O3 can merge VLAs to
	// the same stack address when it thinks lifetimes don't overlap,
	// producing incorrect results for functions with multiple buffers.
	var length string
	if val, ok := t.tryEvalConstInt(e.Args[1]); ok {
		length = strconv.Itoa(val)
	} else {
		length = t.translateExpr(e.Args[1])
	}

	// Check if the first arg is a slice of hwy.Vec[T]
	typeStr := exprToString(e.Args[0])
	if strings.Contains(typeStr, "hwy.Vec") || strings.Contains(typeStr, "Vec[") {
		// This will be used with := assignment. The CASTTranslator will see this
		// is a make call and handle it specially via inferType.
		// Return a special token that the assignment handler uses.
		return fmt.Sprintf("/* VEC_ARRAY:%s:%s */", t.profile.VecTypes[t.tier], length)
	}

	// Scalar slice types: []float32, []T, etc.
	if after, ok := strings.CutPrefix(typeStr, "[]"); ok {
		elemGoType := after
		cType := t.goTypeToCType(elemGoType)
		return fmt.Sprintf("/* SCALAR_ARRAY:%s:%s */", cType, length)
	}

	return fmt.Sprintf("/* make: unsupported type %s */", typeStr)
}

// isTypeParam returns true if the name is a generic type parameter of the
// function being translated (e.g., "T" in BaseForwardRCT[T hwy.SignedInts]).
func (t *CASTTranslator) isTypeParam(name string) bool {
	return t.typeParamNames[name]
}

// resolveTypeParam resolves a type parameter name to its concrete Go type.
// When typeMap is set and contains the parameter name, uses the mapped type.
// Otherwise falls back to the translator's primary elemType.
func (t *CASTTranslator) resolveTypeParam(name string) string {
	if t.typeMap != nil {
		if ct, ok := t.typeMap[name]; ok {
			return ct
		}
	}
	return t.elemType
}

// goTypeConvToCType returns the C type for a Go type conversion function name,
// or empty string if it's not a type conversion.
func (t *CASTTranslator) goTypeConvToCType(name string) string {
	switch name {
	case "uint64":
		return "unsigned long"
	case "uint32":
		return "unsigned int"
	case "uint16":
		return "unsigned short"
	case "uint8", "byte":
		return "unsigned char"
	case "int64":
		return "long"
	case "int32":
		return "int"
	case "int16":
		return "short"
	case "int8":
		return "signed char"
	case "int":
		return "long"
	case "uint":
		return "unsigned long"
	case "float32":
		return "float"
	case "float64":
		return "double"
	default:
		// Check if this is a generic type parameter (e.g., "T") and resolve
		// it to the concrete C type from the profile.
		if t.profile != nil && t.isTypeParam(name) {
			return t.profile.CType
		}
		return ""
	}
}

// mathFuncToC maps single-arg Go math/stdmath functions and contrib/math Vec
// functions to their C stdlib equivalents. The f32 variant is formed by appending "f".
// Special cases (multi-arg, composite, non-standard naming) are handled in the switch.
var mathFuncToC = map[string]string{
	// stdmath
	"Sqrt":  "sqrt",
	"RSqrt": "rsqrt",
	"Exp":   "exp",
	"Log":   "log",
	"Erf":   "erf",
	"Tanh":  "tanh",
	// contrib/math Vec wrappers
	"BaseExpVec":   "exp",
	"BaseExp2Vec":  "exp2",
	"BaseLogVec":   "log",
	"BaseLog2Vec":  "log2",
	"BaseLog10Vec": "log10",
	"BaseSinVec":   "sin",
	"BaseCosVec":   "cos",
	"BaseTanhVec":  "tanh",
	"BaseSinhVec":  "sinh",
	"BaseCoshVec":  "cosh",
	"BaseAsinhVec": "asinh",
	"BaseAcoshVec": "acosh",
	"BaseAtanhVec":    "atanh",
	"BaseSigmoidVec": "sigmoid",
	"BaseErfVec":     "erf",
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

// bitsLenToBuiltin maps Go math/bits Len* functions to GCC builtin expressions.
// The returned format string has two %s placeholders for the argument (used twice:
// once for the zero check, once in the clz call).
// bits.Len32(x) = number of bits required to represent x (0 for x==0).
// In C: x == 0 ? 0 : (32 - __builtin_clz(x))
func bitsLenToBuiltin(funcName string) string {
	switch funcName {
	case "Len64":
		return "((%s) == 0 ? 0 : (64 - __builtin_clzll(%s)))"
	case "Len32":
		return "((%s) == 0 ? 0 : (32 - __builtin_clz(%s)))"
	case "Len16":
		return "((%s) == 0 ? 0 : (32 - __builtin_clz((unsigned int)(%s))))"
	case "Len8":
		return "((%s) == 0 ? 0 : (32 - __builtin_clz((unsigned int)(%s))))"
	case "Len":
		return "((%s) == 0 ? 0 : (64 - __builtin_clzll(%s)))"
	default:
		return ""
	}
}

// translateUnsafeSlice handles unsafe.Slice((*T)(unsafe.Pointer(&arr[0])), N).
// This common Go pattern reinterprets a slice's backing memory as a different type.
// In C, this is just a pointer cast: (C_T *)arr.
func (t *CASTTranslator) translateUnsafeSlice(ptrArg, _ ast.Expr) string {
	// The pointer argument is typically (*T)(unsafe.Pointer(&arr[0])):
	//   CallExpr{ Fun: (*T), Args: [unsafe.Pointer(&arr[0])] }
	// We need to extract the target type T and the source array name.

	// Extract C type from the type-conversion expression.
	// Pattern: (*uint8)(unsafe.Pointer(&values[0]))
	// The outer call is a type conversion: (*uint8)(...)
	if outerCall, ok := ptrArg.(*ast.CallExpr); ok {
		// Get the target pointer type from the Fun expression
		if starExpr, ok := outerCall.Fun.(*ast.ParenExpr); ok {
			if star, ok := starExpr.X.(*ast.StarExpr); ok {
				targetType := exprToString(star.X)
				cType := t.goTypeToCType(targetType)

				// Extract the source pointer from inside unsafe.Pointer(&arr[0])
				if len(outerCall.Args) == 1 {
					srcPtr := t.extractUnsafePointerArg(outerCall.Args[0])
					if srcPtr != "" {
						return fmt.Sprintf("(%s *)(%s)", cType, srcPtr)
					}
				}
			}
		}
	}

	// Fallback: just translate both args
	ptr := t.translateExpr(ptrArg)
	return ptr
}

// extractUnsafePointerArg extracts the underlying pointer from unsafe.Pointer(&arr[0]).
// Returns the C expression for the base pointer, or "" if the pattern is unrecognized.
func (t *CASTTranslator) extractUnsafePointerArg(expr ast.Expr) string {
	// unsafe.Pointer(&arr[0]) → the arg is a CallExpr calling unsafe.Pointer
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return ""
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return ""
	}
	pkg, ok := sel.X.(*ast.Ident)
	if !ok || pkg.Name != "unsafe" || sel.Sel.Name != "Pointer" {
		return ""
	}
	if len(call.Args) != 1 {
		return ""
	}

	// The arg should be &arr[0]
	unary, ok := call.Args[0].(*ast.UnaryExpr)
	if !ok || unary.Op != token.AND {
		return ""
	}

	// arr[0]
	idx, ok := unary.X.(*ast.IndexExpr)
	if !ok {
		return ""
	}

	// Return just the array/slice name (arr)
	return t.translateExpr(idx.X)
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
		// Check if base is a package global array — for 2D arrays like [256][4]uint8,
		// indexing by the first dimension yields a pointer to the inner element type.
		if ident, ok := e.X.(*ast.Ident); ok && t.packageGlobals != nil {
			if pg, ok := t.packageGlobals[ident.Name]; ok {
				cElem := t.goTypeToCType(pg.ElemType)
				if pg.InnerSize > 0 {
					// 2D array: first index yields pointer to inner row
					return cVarInfo{cType: cElem + " *", isPtr: true}
				}
				// 1D array: index yields scalar element
				return cVarInfo{cType: cElem}
			}
		}
		return cVarInfo{cType: t.profile.CType}
	case *ast.ParenExpr:
		return t.inferType(e.X)
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
		// Boolean literals → long
		if e.Name == "true" || e.Name == "false" {
			return cVarInfo{cType: "long"}
		}
		// Variable reference — look up its type
		if info, ok := t.vars[e.Name]; ok {
			return info
		}
		if info, ok := t.params[e.Name]; ok {
			if info.isSlice {
				return cVarInfo{cType: info.cType, isPtr: true}
			}
			if info.isInt {
				// GOAT pointer-wrapped param: dereferenced value is in t.vars.
				// If we reach here, use long as the dereferenced type.
				return cVarInfo{cType: "long"}
			}
			// Helper mode: param is by value with its actual C type.
			// Non-pointer types (long, float, double) are returned directly.
			if !strings.HasSuffix(info.cType, "*") {
				return cVarInfo{cType: info.cType}
			}
			return cVarInfo{cType: t.profile.CType}
		}
		return cVarInfo{cType: t.profile.CType}
	case *ast.SelectorExpr:
		// Field access — check for Image struct fields (parameters)
		if ident, ok := e.X.(*ast.Ident); ok {
			if info, ok := t.params[ident.Name]; ok && info.isStructPtr {
				switch e.Sel.Name {
				case "width", "height", "stride":
					return cVarInfo{cType: "long"}
				case "data":
					return cVarInfo{cType: info.structElemCType + " *", isPtr: true}
				}
			}
			// Check for local struct pointer variables
			if vi, ok := t.vars[ident.Name]; ok && vi.isStructPtr {
				if sd, ok := t.packageStructs[vi.structPtrType]; ok {
					for _, f := range sd.Fields {
						if f.Name == e.Sel.Name {
							cType := goPkgGlobalElemToCType(f.ElemType)
							return cVarInfo{cType: cType}
						}
					}
				}
			}
		}
		return cVarInfo{cType: t.profile.CType}
	case *ast.UnaryExpr:
		if e.Op == token.AND {
			// &structArray[idx] → const StructType *
			if idxExpr, ok := e.X.(*ast.IndexExpr); ok {
				if ident, ok := idxExpr.X.(*ast.Ident); ok && t.packageGlobals != nil {
					if pg, ok := t.packageGlobals[ident.Name]; ok && pg.IsStruct && pg.StructDef != nil {
						return cVarInfo{
							cType:         "const " + pg.StructDef.Name + " *",
							isPtr:         true,
							isStructPtr:   true,
							structPtrType: pg.StructDef.Name,
						}
					}
				}
			}
		}
		return t.inferType(e.X)
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
			// Scalar-returning functions must be checked before the type parameter
			// override below, which would incorrectly return a vector type for
			// calls like hwy.NumLanes[uint8]().
			switch sel.Sel.Name {
			case "MaxLanes", "NumLanes", "GetLane", "TileDim":
				return cVarInfo{cType: "long"}
			case "NewTile":
				return cVarInfo{cType: tileStructName(t.profile)}
			case "TileReadRow":
				return cVarInfo{cType: vecType, isVector: true}
			}

			// Check for explicit type parameter that overrides the profile's vector type.
			// e.g., hwy.LoadSlice[uint8](...) on a uint32 profile → uint8x16_t
			if idx, ok := e.Fun.(*ast.IndexExpr); ok {
				typeParam := exprToString(idx.Index)
				if typeParam != "" && typeParam != t.profile.ElemType {
					switch typeParam {
					case "uint8", "byte":
						return cVarInfo{cType: "uint8x16_t", isVector: true}
					}
				}
			}

			switch sel.Sel.Name {
			case "Load", "Load4", "Zero", "Set", "Const", "MulAdd", "FMA", "Add", "Sub", "Mul", "Div",
				"Min", "Max", "Neg", "Abs", "Sqrt", "RSqrt", "InvSqrt", "ShiftRight",
				"LoadSlice", "InterleaveLower", "InterleaveUpper",
				"And", "Or", "Xor", "PopCount",
				"IfThenElse", "Merge", "SlideUpLanes", "Pow",
				"Iota", "Round", "Clamp", "ConvertToFloat32":
				return cVarInfo{cType: vecType, isVector: true}
			case "ConvertToInt32":
				if it, ok := t.profile.Int32VecType[t.tier]; ok {
					return cVarInfo{cType: it, isVector: true}
				}
				return cVarInfo{cType: vecType, isVector: true}
			case "MaskAnd", "MaskOr", "MaskAndNot", "FirstN":
				// Mask operations return a mask vector
				maskType := vecType
				if mt, ok := t.profile.MaskType[t.tier]; ok {
					maskType = mt
				}
				return cVarInfo{cType: maskType, isVector: true}
			case "FindFirstTrue", "CountTrue":
				return cVarInfo{cType: "long"}
			case "AllTrue", "AllFalse":
				return cVarInfo{cType: "long"}
			case "CompressStore":
				return cVarInfo{cType: "long"}
			case "TableLookupBytes":
				// TableLookupBytes always operates on bytes, returns uint8x16_t
				return cVarInfo{cType: "uint8x16_t", isVector: true}
			case "ReduceMin", "ReduceMax":
				// ReduceMin/Max return a scalar
				if t.profile.ScalarArithType != "" {
					return cVarInfo{cType: t.profile.ScalarArithType}
				}
				return cVarInfo{cType: t.profile.CType}
			case "ReduceSum":
				// ReduceSum returns a scalar, not a vector
				if t.profile.ScalarArithType != "" {
					return cVarInfo{cType: t.profile.ScalarArithType}
				}
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
			case "Equal", "Greater", "GreaterThan", "GreaterEqual":
				// Comparison ops return a mask vector
				maskType := vecType
				if mt, ok := t.profile.MaskType[t.tier]; ok {
					maskType = mt
				}
				return cVarInfo{cType: maskType, isVector: true}
			case "DotAccumulate":
				// DotAccumulate returns the wide accumulator type (float32x4_t / __m512)
				if accType, ok := t.profile.DotAccType[t.tier]; ok {
					return cVarInfo{cType: accType, isVector: true}
				}
				return cVarInfo{cType: vecType, isVector: true}
			}
		}
	}

	// Check for v.NumLanes() method calls
	if sel, ok := e.Fun.(*ast.SelectorExpr); ok {
		if sel.Sel.Name == "NumLanes" || sel.Sel.Name == "NumElements" {
			return cVarInfo{cType: "long"}
		}
		// Check for unsafe.Slice((*T)(unsafe.Pointer(&arr[0])), N) → T *
		if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "unsafe" && sel.Sel.Name == "Slice" {
			if len(e.Args) >= 1 {
				if outerCall, ok := e.Args[0].(*ast.CallExpr); ok {
					if parenExpr, ok := outerCall.Fun.(*ast.ParenExpr); ok {
						if star, ok := parenExpr.X.(*ast.StarExpr); ok {
							targetType := exprToString(star.X)
							cType := t.goTypeToCType(targetType)
							return cVarInfo{cType: cType + " *", isPtr: true}
						}
					}
				}
			}
		}
		// Check for struct method calls (e.g., Row(), Width(), etc.)
		if ident, ok := sel.X.(*ast.Ident); ok {
			if info, ok := t.params[ident.Name]; ok && info.isStructPtr {
				switch sel.Sel.Name {
				case "Row":
					// img.Row(y) returns a pointer to the element type
					return cVarInfo{cType: info.structElemCType + " *", isPtr: true}
				case "Width", "Height", "Stride":
					return cVarInfo{cType: "long"}
				}
			}
		}
	}

	// Check for math/stdmath functions (use extractSelectorExpr to handle
	// IndexExpr wrappers like math.BaseSigmoidVec[float32](...)).
	if sel := extractSelectorExpr(e.Fun); sel != nil {
		if pkg, ok := sel.X.(*ast.Ident); ok && (pkg.Name == "math" || pkg.Name == "stdmath") {
			switch sel.Sel.Name {
			case "Float32bits":
				return cVarInfo{cType: "unsigned int"}
			case "Float32frombits":
				return cVarInfo{cType: "float"}
			case "Sqrt", "RSqrt", "Exp", "Log", "Erf", "Abs":
				if t.elemType == "float64" {
					return cVarInfo{cType: "double"}
				}
				return cVarInfo{cType: "float"}
			case "Max", "Min":
				if t.elemType == "float64" {
					return cVarInfo{cType: "double"}
				}
				return cVarInfo{cType: "float"}
			case "Inf":
				if t.elemType == "float64" {
					return cVarInfo{cType: "double"}
				}
				return cVarInfo{cType: "float"}
			default:
				// Base*Vec functions (BaseSigmoidVec, BaseExpVec, BaseErfVec, etc.)
				// return vector types when given vector arguments.
				if strings.HasPrefix(sel.Sel.Name, "Base") && strings.HasSuffix(sel.Sel.Name, "Vec") {
					if len(e.Args) > 0 && t.inferType(e.Args[0]).isVector {
						return cVarInfo{cType: t.profile.VecTypes[t.tier], isVector: true}
					}
					if t.elemType == "float64" {
						return cVarInfo{cType: "double"}
					}
					return cVarInfo{cType: "float"}
				}
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
		// make() → infer from the type argument
		if ident.Name == "make" && len(e.Args) >= 1 {
			typeStr := exprToString(e.Args[0])
			if strings.Contains(typeStr, "hwy.Vec") || strings.Contains(typeStr, "Vec[") {
				return cVarInfo{cType: t.profile.VecTypes[t.tier], isVector: true, isPtr: true}
			}
			if after, ok0 := strings.CutPrefix(typeStr, "[]"); ok0 {
				elemGoType := after
				cType := t.goTypeToCType(elemGoType)
				return cVarInfo{cType: cType + " *", isPtr: true}
			}
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

	// Check if this is a call to a helper function that returns Vec[T].
	if ident, ok := e.Fun.(*ast.Ident); ok {
		if t.helperReturnVec[ident.Name] {
			return cVarInfo{cType: t.profile.VecTypes[t.tier], isVector: true}
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
	case "uint16":
		return "unsigned short"
	case "uint8", "byte":
		return "unsigned char"
	case "int16":
		return "short"
	case "int8":
		return "signed char"
	case "bool":
		return "long"
	case "T":
		if t.profile.ScalarArithType != "" {
			return t.profile.ScalarArithType
		}
		return t.profile.CType
	default:
		if after, ok := strings.CutPrefix(goType, "[]"); ok {
			elemType := after
			return goSliceElemToCType(elemType, t.profile) + " *"
		}
		// Fixed-size array types like [4]uint32 → element C type
		if elemType, _ := parseGoArrayType(goType); elemType != "" {
			return t.goTypeToCType(elemType)
		}
		// hwy.Vec[T] → the profile's primary vector type (e.g., int32x4_t, float32x4_t)
		if goType == "hwy.Vec[T]" || strings.HasPrefix(goType, "hwy.Vec[") {
			return t.profile.VecTypes[t.tier]
		}
		// hwy.Mask[T] → the profile's primary vector type (masks are same width)
		if goType == "hwy.Mask[T]" || strings.HasPrefix(goType, "hwy.Mask[") {
			return t.profile.VecTypes[t.tier]
		}
		// hwy.Tile[T] → tile struct type for the current profile
		if goType == "hwy.Tile[T]" || strings.HasPrefix(goType, "hwy.Tile[") {
			return tileStructName(t.profile)
		}
		return "long" // default for unknown types
	}
}

// ---------------------------------------------------------------------------
// Tile operation helpers
// ---------------------------------------------------------------------------

// tileStructName returns the C typedef name for the tile type.
// For NEON float32: "HwyTileF32x4", for SVE float32: "HwyTileF32x16", etc.
func tileStructName(profile *CIntrinsicProfile) string {
	suffix := "F32"
	switch profile.ElemType {
	case "float64":
		suffix = "F64"
	}
	lanes := profile.Tiers[0].Lanes
	return fmt.Sprintf("HwyTile%sx%d", suffix, lanes)
}

// tileLanes returns the number of tile rows (= vector lane count).
func tileLanes(profile *CIntrinsicProfile) int {
	return profile.Tiers[0].Lanes
}

// translateTileArg translates a tile pointer argument (e.g., &tile from
// hwy.TileZero(&tile)) to the bare C variable name. In Go, tile ops take
// *Tile via &tile, but in C the tile is a local struct and we use . access.
func (t *CASTTranslator) translateTileArg(expr ast.Expr) string {
	// Strip the & from &tile to get the bare identifier
	if unary, ok := expr.(*ast.UnaryExpr); ok && unary.Op == token.AND {
		return t.translateExpr(unary.X)
	}
	// Fallback: translate as-is (shouldn't normally happen for tile ops)
	return t.translateExpr(expr)
}

// emitTileNew returns a zero-initialized tile struct literal.
func (t *CASTTranslator) emitTileNew() string {
	return fmt.Sprintf("(%s){{0}}", tileStructName(t.profile))
}

// emitTileZero emits code to zero all rows of a tile.
// hwy.TileZero(&tile) → tile zero loop or unrolled zero
func (t *CASTTranslator) emitTileZero(args []ast.Expr) string {
	if len(args) < 1 {
		return "/* TileZero: missing args */"
	}
	tileName := t.translateTileArg(args[0])
	n := tileLanes(t.profile)
	dupFn := t.profile.DupFn[t.tier]

	var parts []string
	for i := 0; i < n; i++ {
		if t.profile.NeedsPredicate {
			parts = append(parts, fmt.Sprintf("%s.rows[%d] = %s(0)", tileName, i, dupFn))
		} else {
			parts = append(parts, fmt.Sprintf("%s.rows[%d] = %s(0.0f)", tileName, i, dupFn))
		}
	}
	return strings.Join(parts, ", ")
}

// emitTileOuterProduct emits code for OuterProductAdd or OuterProductSub.
// hwy.OuterProductAdd(&tile, row, col) → broadcast+FMA per row
// On NEON: uses vfmaq_laneq_f32 for lane broadcast+FMA (avoids explicit broadcast).
// On SVE: uses explicit broadcast + svmla.
func (t *CASTTranslator) emitTileOuterProduct(args []ast.Expr, isAdd bool) string {
	if len(args) < 3 {
		return "/* OuterProduct: missing args */"
	}
	tileName := t.translateTileArg(args[0])
	rowVec := t.translateExpr(args[1])
	colVec := t.translateExpr(args[2])
	n := tileLanes(t.profile)

	var parts []string
	for i := 0; i < n; i++ {
		if isAdd {
			if t.profile.TargetName == "NEON" {
				// NEON: vfmaq_laneq_f32(acc, col, row, lane)
				fmaLaneFn := "vfmaq_laneq_f32"
				if t.profile.ElemType == "float64" {
					fmaLaneFn = "vfmaq_laneq_f64"
				}
				parts = append(parts, fmt.Sprintf(
					"%s.rows[%d] = %s(%s.rows[%d], %s, %s, %d)",
					tileName, i, fmaLaneFn, tileName, i, colVec, rowVec, i))
			} else {
				// SVE: broadcast lane, then FMA
				dupFn := t.profile.DupFn[t.tier]
				getLaneFn := t.profile.GetLaneFn[t.tier]
				fmaFn := t.profile.FmaFn[t.tier]
				parts = append(parts, fmt.Sprintf(
					"%s.rows[%d] = %s(pg, %s.rows[%d], %s(%s(pg, %s, %d)), %s)",
					tileName, i, fmaFn, tileName, i, dupFn, getLaneFn, rowVec, i, colVec))
			}
		} else {
			// OuterProductSub: acc -= broadcast(row[i]) * col
			if t.profile.TargetName == "NEON" {
				mulLaneFn := "vmulq_laneq_f32"
				subFn := "vsubq_f32"
				if t.profile.ElemType == "float64" {
					mulLaneFn = "vmulq_laneq_f64"
					subFn = "vsubq_f64"
				}
				parts = append(parts, fmt.Sprintf(
					"%s.rows[%d] = %s(%s.rows[%d], %s(%s, %s, %d))",
					tileName, i, subFn, tileName, i, mulLaneFn, colVec, rowVec, i))
			} else {
				// SVE: broadcast, mul, sub
				dupFn := t.profile.DupFn[t.tier]
				getLaneFn := t.profile.GetLaneFn[t.tier]
				mulFn := t.profile.MulFn[t.tier]
				subFn := t.profile.SubFn[t.tier]
				parts = append(parts, fmt.Sprintf(
					"%s.rows[%d] = %s(pg, %s.rows[%d], %s(pg, %s(%s(pg, %s, %d)), %s))",
					tileName, i, subFn, tileName, i, mulFn, dupFn, getLaneFn, rowVec, i, colVec))
			}
		}
	}
	return strings.Join(parts, ", ")
}

// emitTileStoreRow emits code to store a tile row to a destination slice.
// hwy.TileStoreRow(&tile, idx, dst) → store(dst, tile.rows[idx])
func (t *CASTTranslator) emitTileStoreRow(args []ast.Expr) string {
	if len(args) < 3 {
		return "/* TileStoreRow: missing args */"
	}
	tileName := t.translateTileArg(args[0])
	idx := t.translateExpr(args[1])
	dst := t.translateExpr(args[2])
	storeFn := t.profile.StoreFn[t.tier]
	if t.profile.NeedsPredicate {
		return fmt.Sprintf("%s(pg, %s, %s.rows[%s])", storeFn, dst, tileName, idx)
	}
	return fmt.Sprintf("%s(%s, %s.rows[%s])", storeFn, dst, tileName, idx)
}

// emitTileReadRow emits code to read a tile row as a vector.
// hwy.TileReadRow(&tile, idx) → tile.rows[idx]
func (t *CASTTranslator) emitTileReadRow(args []ast.Expr) string {
	if len(args) < 2 {
		return "/* TileReadRow: missing args */"
	}
	tileName := t.translateTileArg(args[0])
	idx := t.translateExpr(args[1])
	return fmt.Sprintf("%s.rows[%s]", tileName, idx)
}

// emitTileLoadCol emits code to load a source slice into a tile column.
// hwy.TileLoadCol(&tile, idx, src) → tile.rows[i] = insert(tile.rows[i], src[i], idx)
func (t *CASTTranslator) emitTileLoadCol(args []ast.Expr) string {
	if len(args) < 3 {
		return "/* TileLoadCol: missing args */"
	}
	tileName := t.translateTileArg(args[0])
	colIdx := t.translateExpr(args[1])
	src := t.translateExpr(args[2])
	n := tileLanes(t.profile)

	var parts []string
	for i := 0; i < n; i++ {
		if t.profile.TargetName == "NEON" {
			// NEON: vsetq_lane_f32(src[i], tile.rows[i], colIdx)
			setLaneFn := "vsetq_lane_f32"
			if t.profile.ElemType == "float64" {
				setLaneFn = "vsetq_lane_f64"
			}
			parts = append(parts, fmt.Sprintf(
				"%s.rows[%d] = %s(%s[%d], %s.rows[%d], %s)",
				tileName, i, setLaneFn, src, i, tileName, i, colIdx))
		} else {
			// SVE: svdup + svsel to set one lane. For simplicity, use
			// store-to-stack, modify, reload approach.
			// TODO: implement proper SVE lane-set
			parts = append(parts, fmt.Sprintf(
				"/* TileLoadCol SVE row %d */ (void)0", i))
		}
	}
	return strings.Join(parts, ", ")
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
// Eligible functions have slice or *Image[T] parameters, use hwy.* operations,
// and are NOT composite math functions handled by the template path.
func IsASTCEligible(pf *ParsedFunc) bool {
	// Must have slice or *Image[T] params (not Vec→Vec)
	hasSliceOrImage := false
	hasImagePtr := false
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "[]") {
			hasSliceOrImage = true
			break
		}
		if isGenericStructPtr(p.Type) {
			hasSliceOrImage = true
			hasImagePtr = true
			break
		}
	}
	if !hasSliceOrImage {
		return false
	}

	// If we have Image pointers, that's sufficient (they contain width/height)
	if hasImagePtr {
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
		// Must NOT have Vec in signature (those go through IsCEligible)
		if hasVecInSignature(*pf) {
			return false
		}
		// Must NOT be a composite math function (those use the template path)
		if mathOpFromFuncName(pf.Name) != "" {
			return false
		}
		return true
	}

	// Must NOT have Vec in signature (those go through IsCEligible)
	if hasVecInSignature(*pf) {
		return false
	}

	// Must NOT be a composite math function (those use the template path)
	if mathOpFromFuncName(pf.Name) != "" {
		return false
	}

	// BaseApply wrappers (e.g., BaseExpTransform) are eligible —
	// the BaseApply call will be inlined at translation time.
	// Check this before hasHwyOps because these wrappers don't contain
	// direct hwy.* calls (the SIMD ops are inside BaseApply itself).
	if isBaseApplyWrapper(pf) {
		return true
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

	// Reject functions with function, interface, or generic type parameters — these
	// can't be translated to C (no vtable/dispatch).
	for _, p := range pf.Params {
		if strings.HasPrefix(p.Type, "func(") {
			return false
		}
		// Reject single-letter uppercase type parameters (e.g., P Predicate[T])
		// which indicate generic interface constraints that can't be translated.
		if len(p.Type) == 1 && p.Type[0] >= 'A' && p.Type[0] <= 'Z' && p.Type != "T" {
			return false
		}
	}

	// All remaining slice-based functions with hwy ops are eligible.
	// This includes activation functions (GELU, ReLU, etc.), matmul, transpose,
	// softmax, and bitwise operations.
	return true
}

// isBaseApplyWrapper returns true if the function's HwyCalls include a call to
// BaseApply. These are thin wrappers like BaseExpTransform that delegate all
// SIMD work to BaseApply(in, out, math.FooVec). They don't contain hwy.* ops
// directly, but are eligible for AST translation because the translator will
// inline the BaseApply call into a load→math→store C loop.
func isBaseApplyWrapper(pf *ParsedFunc) bool {
	for _, call := range pf.HwyCalls {
		if call.FuncName == "BaseApply" {
			return true
		}
	}
	return false
}

// hasMathVecCalls returns true if the function calls math Vec→Vec functions
// like BaseErfVec, BaseSigmoidVec, BaseExpVec, etc. These functions are
// translatable to inline C helpers and indicate the function is eligible for
// AST→C translation even without int params.
func hasMathVecCalls(pf *ParsedFunc) bool {
	for _, call := range pf.HwyCalls {
		if call.Package == "math" || call.Package == "stdmath" {
			if _, ok := mathFuncToC[call.FuncName]; ok {
				return true
			}
		}
	}
	return false
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
