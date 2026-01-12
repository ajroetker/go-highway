package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"strconv"
	"strings"
)

// TransformResult contains the transformed function and any hoisted constants.
type TransformResult struct {
	FuncDecl      *ast.FuncDecl
	HoistedConsts []HoistedConst
}

// HoistedConst represents a constant that was hoisted from a function to package level.
type HoistedConst struct {
	VarName   string // Package-level var name (e.g., "BaseSigmoid_one_f32")
	Value     string // The constant value (e.g., "1.0")
	VecType   string // Vector type (e.g., "Float32x8")
	Broadcast string // Broadcast function (e.g., "archsimd.BroadcastFloat32x8")
}

// TransformOptions contains additional context for transformation.
type TransformOptions struct {
	TypeSpecificConsts map[string]*TypeSpecificConst
	ConditionalBlocks  []ConditionalBlock
	FileSet            *token.FileSet    // For resolving line numbers in conditional blocks
	Imports            map[string]string // map[local_name]import_path for resolving package references
}

// Transform transforms a parsed function for a specific target and element type.
// It clones the AST, specializes generics, and transforms hwy operations.
func Transform(pf *ParsedFunc, target Target, elemType string) *TransformResult {
	return TransformWithOptions(pf, target, elemType, nil)
}

// TransformWithOptions transforms a parsed function with additional options.
func TransformWithOptions(pf *ParsedFunc, target Target, elemType string, opts *TransformOptions) *TransformResult {
	if opts == nil {
		opts = &TransformOptions{}
	}

	// First, filter the original body based on conditional blocks.
	// We need to do this BEFORE cloning because the original AST has valid positions.
	var filteredBody *ast.BlockStmt
	if len(opts.ConditionalBlocks) > 0 && opts.FileSet != nil {
		filteredBody = filterConditionalBlocks(pf.Body, opts.ConditionalBlocks, opts.FileSet, target.Name, elemType)
	} else {
		filteredBody = pf.Body
	}

	// Create new function declaration (don't copy Doc - emitter handles comments)
	funcDecl := &ast.FuncDecl{
		Name: ast.NewIdent(pf.Name + target.Suffix()),
		Type: &ast.FuncType{
			Params:  &ast.FieldList{},
			Results: pf.buildResultsWithTarget(elemType, target),
		},
		Body: cloneBlockStmt(filteredBody),
	}

	// Build parameter list with specialized types
	for _, param := range pf.Params {
		paramType := specializeType(param.Type, pf.TypeParams, elemType)
		// Also transform hwy.Vec[T] to concrete vector types for SIMD targets
		paramType = specializeVecType(paramType, elemType, target)
		field := &ast.Field{
			Names: []*ast.Ident{ast.NewIdent(param.Name)},
			Type:  parseTypeExpr(paramType),
		}
		funcDecl.Type.Params.List = append(funcDecl.Type.Params.List, field)
	}

	// Transform the function body
	ctx := &transformContext{
		target:             target,
		elemType:           elemType,
		typeParams:         pf.TypeParams,
		loopInfo:           pf.LoopInfo,
		lanesVars:          make(map[string]bool),
		localVars:          make(map[string]bool),
		stackArrayVars:     make(map[string]bool),
		hoistedConsts:      make(map[string]HoistedConst),
		funcName:           pf.Name,
		typeSpecificConsts: opts.TypeSpecificConsts,
		conditionalBlocks:  opts.ConditionalBlocks,
		fset:               opts.FileSet,
		imports:            opts.Imports,
	}

	// Collect all locally-defined variable names to avoid hoisting them as constants
	collectLocalVariables(funcDecl.Body, ctx)

	// Resolve type-specific constant references
	// Pattern 1: expC0 -> expC0_f32 (base name lookup)
	// Pattern 2: expC0_f32 -> expC0_f64 (suffix swapping for compilable base files)
	transformIdentifiers(funcDecl.Body, ctx)

	transformNode(funcDecl.Body, ctx)

	// Post-process to replace NumLanes() calls and ReduceSum() calls
	if target.Name != "Fallback" {
		postProcessSIMD(funcDecl.Body, ctx)
	}

	// Post-process to convert stack array usages to slice expressions
	if target.Name != "Fallback" && len(ctx.stackArrayVars) > 0 {
		convertStackArrayUsages(funcDecl.Body, ctx)
	}

	// Insert tail handling if there's a loop and function doesn't return a value
	// (functions that return values have their own tail handling in the template)
	if pf.LoopInfo != nil && len(pf.Returns) == 0 {
		insertTailHandling(funcDecl.Body, pf.LoopInfo, elemType, target, pf.Name, pf.Params)
	}

	// Collect hoisted constants
	var hoisted []HoistedConst
	for _, hc := range ctx.hoistedConsts {
		hoisted = append(hoisted, hc)
	}

	return &TransformResult{
		FuncDecl:      funcDecl,
		HoistedConsts: hoisted,
	}
}

type transformContext struct {
	target             Target
	elemType           string
	typeParams         []TypeParam
	lanesVars          map[string]bool                   // Variables assigned from NumLanes()
	localVars          map[string]bool                   // Variables defined locally in the function
	stackArrayVars     map[string]bool                   // Variables that are stack arrays (need [:] when used as slice)
	loopInfo           *LoopInfo
	hoistedConsts      map[string]HoistedConst           // Hoisted constants (key is local var name)
	funcName           string                            // Current function name for generating unique hoisted names
	typeSpecificConsts map[string]*TypeSpecificConst     // Type-specific constant registry
	conditionalBlocks  []ConditionalBlock                // Conditional blocks to process
	fset               *token.FileSet                    // For resolving line numbers
	imports            map[string]string                 // map[local_name]import_path for resolving package references
}

// collectLocalVariables walks the AST and collects all locally-defined variable names.
// This is used to exclude local variables from constant hoisting.
func collectLocalVariables(node ast.Node, ctx *transformContext) {
	if node == nil {
		return
	}

	ast.Inspect(node, func(n ast.Node) bool {
		switch stmt := n.(type) {
		case *ast.AssignStmt:
			// Collect all LHS identifiers from := and = assignments
			// Only := definitely defines new variables, but we track both to be safe
			if stmt.Tok == token.DEFINE {
				for _, lhs := range stmt.Lhs {
					if ident, ok := lhs.(*ast.Ident); ok {
						ctx.localVars[ident.Name] = true
					}
				}
			}
		case *ast.DeclStmt:
			// var declarations
			if genDecl, ok := stmt.Decl.(*ast.GenDecl); ok {
				if genDecl.Tok == token.VAR {
					for _, spec := range genDecl.Specs {
						if valueSpec, ok := spec.(*ast.ValueSpec); ok {
							for _, name := range valueSpec.Names {
								ctx.localVars[name.Name] = true
							}
						}
					}
				}
			}
		case *ast.RangeStmt:
			// for k, v := range ...
			if stmt.Tok == token.DEFINE {
				if ident, ok := stmt.Key.(*ast.Ident); ok && ident.Name != "_" {
					ctx.localVars[ident.Name] = true
				}
				if ident, ok := stmt.Value.(*ast.Ident); ok && ident.Name != "_" {
					ctx.localVars[ident.Name] = true
				}
			}
		case *ast.ForStmt:
			// for i := 0; ... - the init statement
			if stmt.Init != nil {
				if assign, ok := stmt.Init.(*ast.AssignStmt); ok && assign.Tok == token.DEFINE {
					for _, lhs := range assign.Lhs {
						if ident, ok := lhs.(*ast.Ident); ok {
							ctx.localVars[ident.Name] = true
						}
					}
				}
			}
		}
		return true
	})
}

// transformNode recursively transforms AST nodes.
func transformNode(node ast.Node, ctx *transformContext) {
	if node == nil {
		return
	}

	ast.Inspect(node, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.CallExpr:
			transformCallExpr(node, ctx)
			// Also check for type conversions like T(1)
			transformTypeConversion(node, ctx)
			// Transform function references passed as arguments
			transformFuncRefArgs(node, ctx)
		case *ast.DeclStmt:
			// Transform variable declarations
			if genDecl, ok := node.Decl.(*ast.GenDecl); ok {
				transformGenDecl(genDecl, ctx)
			}
		case *ast.AssignStmt:
			// Transform assignments
			transformAssignStmt(node, ctx)
		case *ast.ForStmt:
			// Transform for loop for SIMD (stride, condition)
			transformForStmt(node, ctx)
		}
		return true
	})
}

// transformForStmt transforms for loops for SIMD targets.
// Changes: for ii := 0; ii < size; ii += v.NumLanes()
// To:      for ii := 0; ii+8 <= size; ii += 8
// Also handles: for ii := 0; ii < size; ii += lanes (where lanes was from NumLanes())
// Only transforms loops that use NumLanes() stride (not scalar tail loops).
func transformForStmt(stmt *ast.ForStmt, ctx *transformContext) {
	if ctx.target.Name == "Fallback" || ctx.loopInfo == nil {
		return
	}

	lanes := ctx.target.LanesFor(ctx.elemType)

	// Check if this loop uses NumLanes() stride - only transform those loops
	isSimdLoop := false
	if assignStmt, ok := stmt.Post.(*ast.AssignStmt); ok {
		if len(assignStmt.Rhs) == 1 {
			// Case 1: ii += v.NumLanes() - direct call
			if call, ok := assignStmt.Rhs[0].(*ast.CallExpr); ok {
				if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
					if sel.Sel.Name == "NumElements" || sel.Sel.Name == "NumLanes" {
						isSimdLoop = true
						// Transform post: ii += v.NumLanes() -> ii += lanes
						assignStmt.Rhs[0] = &ast.BasicLit{
							Kind:  token.INT,
							Value: strconv.Itoa(lanes),
						}
					}
				}
			}
			// Case 2: ii += lanes - variable assigned from NumLanes()
			if ident, ok := assignStmt.Rhs[0].(*ast.Ident); ok {
				if ctx.lanesVars[ident.Name] {
					isSimdLoop = true
					// The variable was already replaced with a constant in transformAssignStmt,
					// but we still need to transform the loop condition
				}
			}
			// Case 3: ii += 8 (or other constant) - already transformed
			if lit, ok := assignStmt.Rhs[0].(*ast.BasicLit); ok {
				if lit.Kind == token.INT {
					// Check if the value matches our lanes - this means it was already transformed
					if lit.Value == strconv.Itoa(lanes) {
						isSimdLoop = true
					}
				}
			}
		}
	}

	// Only transform condition for SIMD loops (not scalar tail loops)
	if isSimdLoop {
		// Transform condition: ii < size -> ii+lanes <= size
		if binExpr, ok := stmt.Cond.(*ast.BinaryExpr); ok {
			if binExpr.Op == token.LSS {
				// Change ii < size to ii+lanes <= size
				binExpr.Op = token.LEQ
				binExpr.X = &ast.BinaryExpr{
					X:  binExpr.X,
					Op: token.ADD,
					Y: &ast.BasicLit{
						Kind:  token.INT,
						Value: strconv.Itoa(lanes),
					},
				}
			}
		}
	}
}

// transformTypeConversion converts T(1) to float32(1) for generic type parameters.
func transformTypeConversion(call *ast.CallExpr, ctx *transformContext) {
	// Check if this is a type conversion T(value) where T is a type parameter
	ident, ok := call.Fun.(*ast.Ident)
	if !ok {
		return
	}

	// Check if the identifier is a type parameter
	for _, tp := range ctx.typeParams {
		if ident.Name == tp.Name {
			// Replace T with the concrete element type
			ident.Name = ctx.elemType
			return
		}
	}
}

// transformCallExpr transforms hwy.* and contrib.* function calls.
func transformCallExpr(call *ast.CallExpr, ctx *transformContext) {
	// First, check for calls to other Base* functions and add target suffix
	// This applies to ALL targets including fallback, since generated functions
	// are always concrete (BaseApply_fallback, not generic BaseApply)
	if ident, ok := call.Fun.(*ast.Ident); ok {
		if strings.HasPrefix(ident.Name, "Base") {
			// Transform BaseFoo to BaseFoo_avx2 (or BaseFoo_fallback, etc.)
			suffix := ctx.target.Suffix()
			if ctx.elemType == "float64" {
				suffix = suffix + "_Float64"
			}
			ident.Name = ident.Name + suffix
		}
	}

	// Transform Vec method calls like .Store() -> .StoreSlice() for SIMD targets
	// This handles cases like fn(x).Store(dst) where fn returns a Vec
	// Skip this for package-level function calls like hwy.Store() which are handled later
	if ctx.target.Name != "Fallback" {
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
			// Don't transform package-level function calls
			if ident, ok := sel.X.(*ast.Ident); ok && ident.Name == "hwy" {
				// Skip - this is a package-level function, handled later
			} else {
				switch sel.Sel.Name {
				case "Store":
					// Transform .Store(dst) -> .StoreSlice(dst)
					sel.Sel.Name = "StoreSlice"
				case "Data":
					transformDataMethod(call, ctx)
					return
				case "GetBit":
					transformGetBitMethod(call, ctx)
					return
				}
			}
		}
	}

	// Also handle generic Base* calls like BaseFoo[T](x)
	// All targets get the suffix since generated functions are concrete
	if indexExpr, ok := call.Fun.(*ast.IndexExpr); ok {
		if ident, ok := indexExpr.X.(*ast.Ident); ok {
			if strings.HasPrefix(ident.Name, "Base") {
				suffix := ctx.target.Suffix()
				if ctx.elemType == "float64" {
					suffix = suffix + "_Float64"
				}
				// Strip the type param and add suffix
				call.Fun = ast.NewIdent(ident.Name + suffix)
			}
		}
	}

	var selExpr *ast.SelectorExpr
	var ok bool

	// Handle both regular calls (hwy.Load) and generic calls (hwy.Zero[T])
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		selExpr = fun
	case *ast.IndexExpr:
		// Generic function call like hwy.Zero[T]()
		// The IndexExpr wraps the SelectorExpr
		selExpr, ok = fun.X.(*ast.SelectorExpr)
		if !ok {
			return
		}
		if ctx.target.Name == "Fallback" {
			// For fallback, replace type param with concrete type
			// hwy.Zero[T]() -> hwy.Zero[float32]()
			for _, tp := range ctx.typeParams {
				if ident, ok := fun.Index.(*ast.Ident); ok && ident.Name == tp.Name {
					ident.Name = ctx.elemType
				}
			}
			// Keep the IndexExpr (with type param), just update the type
		} else {
			// For SIMD targets, strip the type param (will be transformed later)
			call.Fun = selExpr
		}
	case *ast.IndexListExpr:
		// Generic function call with multiple type params like hwy.Func[T, U]()
		selExpr, ok = fun.X.(*ast.SelectorExpr)
		if !ok {
			return
		}
		if ctx.target.Name == "Fallback" {
			// For fallback, replace type params with concrete types
			for i, idx := range fun.Indices {
				if ident, ok := idx.(*ast.Ident); ok {
					for _, tp := range ctx.typeParams {
						if ident.Name == tp.Name {
							fun.Indices[i] = ast.NewIdent(ctx.elemType)
						}
					}
				}
			}
		} else {
			call.Fun = selExpr
		}
	default:
		return
	}

	ident, ok := selExpr.X.(*ast.Ident)
	if !ok {
		return
	}

	// Handle hwy.* and contrib subpackage calls (math.*, dot.*, matvec.*, algo.*)
	switch ident.Name {
	case "hwy", "contrib", "math", "dot", "matvec", "algo":
		// Continue processing
	default:
		return
	}

	funcName := selExpr.Sel.Name

	// Handle cross-package Base* function calls (e.g., algo.BaseApply, math.BaseExpVec)
	// These need target suffix added, similar to same-package Base* calls
	if strings.HasPrefix(funcName, "Base") {
		suffix := ctx.target.Suffix()
		if ctx.elemType == "float64" {
			suffix = suffix + "_Float64"
		}
		selExpr.Sel.Name = funcName + suffix
		return
	}

	opInfo, ok := ctx.target.OpMap[funcName]
	if !ok {
		// Unknown operation, leave as-is
		return
	}

	// Transform based on operation type
	if opInfo.IsMethod {
		transformToMethod(call, funcName, opInfo, ctx)
	} else {
		transformToFunction(call, funcName, opInfo, ctx)
	}
}

// transformDataMethod transforms v.Data() to a temporary slice.
func transformDataMethod(call *ast.CallExpr, ctx *transformContext) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}
	vecExpr := sel.X
	lanes := ctx.target.LanesFor(ctx.elemType)

	// func() []T { var tmp [lanes]T; v.StoreSlice(tmp[:]); return tmp[:] }()

	// var tmp [lanes]T
	decl := &ast.DeclStmt{
		Decl: &ast.GenDecl{
			Tok: token.VAR,
			Specs: []ast.Spec{
				&ast.ValueSpec{
					Names: []*ast.Ident{ast.NewIdent("_simd_tmp")},
					Type: &ast.ArrayType{
						Len: &ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(lanes)},
						Elt: ast.NewIdent(ctx.elemType),
					},
				},
			},
		},
	}

	// v.StoreSlice(tmp[:])
	storeCall := &ast.CallExpr{
		Fun: &ast.SelectorExpr{
			X:   cloneExpr(vecExpr),
			Sel: ast.NewIdent("StoreSlice"),
		},
		Args: []ast.Expr{
			&ast.SliceExpr{
				X: ast.NewIdent("_simd_tmp"),
			},
		},
	}

	// return tmp[:]
	retStmt := &ast.ReturnStmt{
		Results: []ast.Expr{
			&ast.SliceExpr{
				X: ast.NewIdent("_simd_tmp"),
			},
		},
	}

	// Function literal
	funcLit := &ast.FuncLit{
		Type: &ast.FuncType{
			Results: &ast.FieldList{
				List: []*ast.Field{
					{Type: &ast.ArrayType{Elt: ast.NewIdent(ctx.elemType)}},
				},
			},
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				decl,
				&ast.ExprStmt{X: storeCall},
				retStmt,
			},
		},
	}

	// Replace call with invocation
	*call = ast.CallExpr{
		Fun: funcLit,
	}
}

// transformGetBitMethod transforms mask.GetBit(i) to check the i-th element.
func transformGetBitMethod(call *ast.CallExpr, ctx *transformContext) {
	if len(call.Args) != 1 {
		return
	}
	indexExpr := call.Args[0]
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}
	maskExpr := sel.X
	lanes := ctx.target.LanesFor(ctx.elemType)

	// Use Int32 vector for extraction to match most masks used with GetBit
	intVecTypeName := getVectorTypeNameForInt("int32", ctx.elemType, ctx.target)
	pkgName := getVecPackageName(ctx.target)

	// func() bool {
	//   vOne := pkg.BroadcastInt32x4(1)
	//   vZero := pkg.BroadcastInt32x4(0)
	//   vMasked := vOne.Merge(vZero, mask)
	//   var tmp [lanes]int32
	//   vMasked.StoreSlice(tmp[:])
	//   return tmp[i] != 0
	// }()

	// 1. vOne := pkg.BroadcastInt32x*(1)
	vOneDecl := &ast.AssignStmt{
		Lhs: []ast.Expr{ast.NewIdent("_vOne")},
		Tok: token.DEFINE,
		Rhs: []ast.Expr{
			&ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "1"}},
			},
		},
	}

	// 2. vZero := pkg.BroadcastInt32x*(0)
	vZeroDecl := &ast.AssignStmt{
		Lhs: []ast.Expr{ast.NewIdent("_vZero")},
		Tok: token.DEFINE,
		Rhs: []ast.Expr{
			&ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "0"}},
			},
		},
	}

	// 3. vMasked := vOne.Merge(vZero, mask)
	vMaskedDecl := &ast.AssignStmt{
		Lhs: []ast.Expr{ast.NewIdent("_vMasked")},
		Tok: token.DEFINE,
		Rhs: []ast.Expr{
			&ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent("_vOne"),
					Sel: ast.NewIdent("Merge"),
				},
				Args: []ast.Expr{
					ast.NewIdent("_vZero"),
					cloneExpr(maskExpr),
				},
			},
		},
	}

	// 4. var tmp [lanes]int32
	tmpDecl := &ast.DeclStmt{
		Decl: &ast.GenDecl{
			Tok: token.VAR,
			Specs: []ast.Spec{
				&ast.ValueSpec{
					Names: []*ast.Ident{ast.NewIdent("_simd_mask_tmp")},
					Type: &ast.ArrayType{
						Len: &ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(lanes)},
						Elt: ast.NewIdent("int32"),
					},
				},
			},
		},
	}

	// 5. vMasked.StoreSlice(tmp[:])
	storeCall := &ast.CallExpr{
		Fun: &ast.SelectorExpr{
			X:   ast.NewIdent("_vMasked"),
			Sel: ast.NewIdent("StoreSlice"),
		},
		Args: []ast.Expr{
			&ast.SliceExpr{
				X: ast.NewIdent("_simd_mask_tmp"),
			},
		},
	}

	// 6. return tmp[i] != 0
	checkExpr := &ast.BinaryExpr{
		X: &ast.IndexExpr{
			X:     ast.NewIdent("_simd_mask_tmp"),
			Index: cloneExpr(indexExpr),
		},
		Op: token.NEQ,
		Y:  &ast.BasicLit{Kind: token.INT, Value: "0"},
	}

	retStmt := &ast.ReturnStmt{
		Results: []ast.Expr{checkExpr},
	}

	funcLit := &ast.FuncLit{
		Type: &ast.FuncType{
			Results: &ast.FieldList{
				List: []*ast.Field{
					{Type: ast.NewIdent("bool")},
				},
			},
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				vOneDecl,
				vZeroDecl,
				vMaskedDecl,
				tmpDecl,
				&ast.ExprStmt{X: storeCall},
				retStmt,
			},
		},
	}

	*call = ast.CallExpr{
		Fun: funcLit,
	}
}

// transformToMethod converts hwy.Add(a, b) to a.Add(b) for SIMD targets.
// For Fallback, keeps hwy.Add(a, b) as-is.
func transformToMethod(call *ast.CallExpr, funcName string, opInfo OpInfo, ctx *transformContext) {
	if len(call.Args) < 1 {
		return
	}

	// For fallback, keep hwy calls as-is (don't convert to method calls)
	if ctx.target.Name == "Fallback" {
		// Just update the package name if needed
		switch fun := call.Fun.(type) {
		case *ast.SelectorExpr:
			fun.X = ast.NewIdent("hwy")
		case *ast.IndexExpr:
			if sel, ok := fun.X.(*ast.SelectorExpr); ok {
				sel.X = ast.NewIdent("hwy")
			}
		}
		return
	}

	// For SIMD targets, convert to method calls on archsimd types
	switch funcName {
	case "Store":
		// hwy.Store(v, dst) -> v.StoreSlice(dst)
		if len(call.Args) >= 2 {
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent("StoreSlice"),
			}
			call.Args = call.Args[1:]
		}

	case "MaskStore":
		// hwy.MaskStore(mask, v, dst) -> v.MaskStoreSlice(mask, dst)
		if len(call.Args) >= 3 {
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[1],
				Sel: ast.NewIdent("MaskStoreSlice"),
			}
			call.Args = []ast.Expr{call.Args[0], call.Args[2]}
		}

	case "Neg":
		// hwy.Neg(x) -> pkg.BroadcastFloat32x8(0).Sub(x) for SIMD
		// (archsimd/asm types don't have a Neg method, so we use 0 - x)
		if len(call.Args) >= 1 {
			vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
			pkgName := getVecPackageName(ctx.target)
			// Create pkg.BroadcastFloat32x8(0)
			zeroLit := &ast.BasicLit{Kind: token.INT, Value: "0"}
			zeroCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{zeroLit},
			}
			call.Fun = &ast.SelectorExpr{
				X:   zeroCall,
				Sel: ast.NewIdent("Sub"),
			}
			// Args stays as [x]
		}

	case "Pow2":
		// hwy.Pow2[T](kInt) -> kInt.Pow2Float32() or kInt.Pow2Float64()
		// based on context element type
		if len(call.Args) >= 1 {
			var methodName string
			switch ctx.elemType {
			case "float32":
				methodName = "Pow2Float32"
			case "float64":
				methodName = "Pow2Float64"
			default:
				methodName = "Pow2Float32" // fallback
			}
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent(methodName),
			}
			call.Args = nil
		}

	case "GetExponent":
		if len(call.Args) >= 1 {
			x := call.Args[0]
			intVecTypeName := getVectorTypeNameForInt("int32", ctx.elemType, ctx.target)
			if ctx.elemType == "float64" {
				intVecTypeName = getVectorTypeNameForInt("int64", ctx.elemType, ctx.target)
			}
			pkgName := getVecPackageName(ctx.target)

			// 1. x.AsInt32() / x.AsInt64()
			var asIntMethod string
			var shift int
			var mask string
			var bias string
			var convertToFloatMethod string

			if ctx.elemType == "float32" {
				asIntMethod = "AsInt32x8"
				// Check targets.go OpMap["AsInt32"].Name
				if op, ok := ctx.target.OpMap["AsInt32"]; ok {
					asIntMethod = op.Name
				}
				shift = 23
				mask = "255" // 0xFF
				bias = "127"
				convertToFloatMethod = "ConvertToFloat32"
			} else {
				asIntMethod = "AsInt64x4"
				if op, ok := ctx.target.OpMap["AsInt64"]; ok {
					asIntMethod = op.Name
				}
				shift = 52
				mask = "2047" // 0x7FF
				bias = "1023"
				convertToFloatMethod = "ConvertToFloat64"
			}

			// x.AsInt32()
			expr := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   cloneExpr(x),
					Sel: ast.NewIdent(asIntMethod),
				},
			}

			// .ShiftAllRight(shift)
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("ShiftAllRight"),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(shift)}},
			}

			// .And(Broadcast(mask))
			broadcastMask := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: mask}},
			}
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("And"),
				},
				Args: []ast.Expr{broadcastMask},
			}

			// .Sub(Broadcast(bias))
			broadcastBias := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: bias}},
			}
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("Sub"),
				},
				Args: []ast.Expr{broadcastBias},
			}

			// .ConvertToFloat()
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent(convertToFloatMethod),
				},
			}

			*call = *expr
		}

	case "GetMantissa":
		if len(call.Args) >= 1 {
			x := call.Args[0]
			intVecTypeName := getVectorTypeNameForInt("int32", ctx.elemType, ctx.target)
			if ctx.elemType == "float64" {
				intVecTypeName = getVectorTypeNameForInt("int64", ctx.elemType, ctx.target)
			}
			pkgName := getVecPackageName(ctx.target)

			var asIntMethod string
			var mask string
			var one string
			var asFloatMethod string

			if ctx.elemType == "float32" {
				asIntMethod = "AsInt32x8"
				if op, ok := ctx.target.OpMap["AsInt32"]; ok {
					asIntMethod = op.Name
				}
				mask = "8388607" // 0x7FFFFF
				one = "1065353216" // 0x3F800000
				asFloatMethod = "AsFloat32x8"
				if op, ok := ctx.target.OpMap["AsFloat32"]; ok {
					asFloatMethod = op.Name
				}
			} else {
				asIntMethod = "AsInt64x4"
				if op, ok := ctx.target.OpMap["AsInt64"]; ok {
					asIntMethod = op.Name
				}
				mask = "4503599627370495" // 0x000FFFFFFFFFFFFF
				one = "4607182418800017408" // 0x3FF0000000000000
				asFloatMethod = "AsFloat64x4"
				if op, ok := ctx.target.OpMap["AsFloat64"]; ok {
					asFloatMethod = op.Name
				}
			}

			// x.AsInt32()
			expr := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   cloneExpr(x),
					Sel: ast.NewIdent(asIntMethod),
				},
			}

			// .And(Broadcast(mask))
			broadcastMask := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: mask}},
			}
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("And"),
				},
				Args: []ast.Expr{broadcastMask},
			}

			// .Or(Broadcast(one))
			broadcastOne := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: one}},
			}
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("Or"),
				},
				Args: []ast.Expr{broadcastOne},
			}

			// .AsFloat32()
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent(asFloatMethod),
				},
			}

			*call = *expr
		}

	case "Abs":
		// hwy.Abs(x) -> x.Max(negX) where negX = pkg.Broadcast*(0).Sub(x)
		// archsimd doesn't have Abs method, so we implement |x| = max(x, -x)
		if opInfo.Package == "special" && len(call.Args) >= 1 {
			vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
			pkgName := getVecPackageName(ctx.target)
			x := call.Args[0]
			// Create pkg.Broadcast*(0)
			zeroLit := &ast.BasicLit{Kind: token.INT, Value: "0"}
			zeroCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{zeroLit},
			}
			// Create pkg.Broadcast*(0).Sub(x) = -x
			negX := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   zeroCall,
					Sel: ast.NewIdent("Sub"),
				},
				Args: []ast.Expr{cloneExpr(x)},
			}
			// Create x.Max(-x)
			call.Fun = &ast.SelectorExpr{
				X:   x,
				Sel: ast.NewIdent("Max"),
			}
			call.Args = []ast.Expr{negX}
		} else {
			// Normal Abs method call
			if len(call.Args) >= 1 {
				call.Fun = &ast.SelectorExpr{
					X:   call.Args[0],
					Sel: ast.NewIdent(opInfo.Name),
				}
				call.Args = nil
			}
		}

	case "IsNaN":
		// hwy.IsNaN(x) -> x.Equal(x).Xor(one.Equal(one))
		// NaN != NaN, so x.Equal(x) is false (all 0s) for NaN elements
		// We XOR with all-true mask to invert, giving true for NaN
		if opInfo.Package == "special" && len(call.Args) >= 1 {
			vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
			pkgName := getVecPackageName(ctx.target)
			x := call.Args[0]
			// Create pkg.Broadcast*(1.0)
			oneLit := &ast.BasicLit{Kind: token.FLOAT, Value: "1.0"}
			oneCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{oneLit},
			}
			// Create one.Equal(one) to get all-true mask
			allTrue := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   oneCall,
					Sel: ast.NewIdent("Equal"),
				},
				Args: []ast.Expr{cloneExpr(oneCall)},
			}
			// Create x.Equal(x)
			xEqX := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   x,
					Sel: ast.NewIdent("Equal"),
				},
				Args: []ast.Expr{cloneExpr(x)},
			}
			// Create x.Equal(x).Xor(allTrue) to invert
			call.Fun = &ast.SelectorExpr{
				X:   xEqX,
				Sel: ast.NewIdent("Xor"),
			}
			call.Args = []ast.Expr{allTrue}
		}

	case "IsInf":
		// hwy.IsInf(x, sign) -> compare with +Inf and/or -Inf
		// sign=0: either +Inf or -Inf, sign=1: +Inf only, sign=-1: -Inf only
		if opInfo.Package == "special" && len(call.Args) >= 2 {
			vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
			pkgName := getVecPackageName(ctx.target)
			x := call.Args[0]
			signArg := call.Args[1]

			// Determine sign value (0, 1, or -1)
			signVal := 0
			if lit, ok := signArg.(*ast.BasicLit); ok && lit.Kind == token.INT {
				if lit.Value == "1" {
					signVal = 1
				} else if lit.Value == "-1" {
					signVal = -1
				}
			} else if unary, ok := signArg.(*ast.UnaryExpr); ok && unary.Op == token.SUB {
				if lit, ok := unary.X.(*ast.BasicLit); ok && lit.Kind == token.INT && lit.Value == "1" {
					signVal = -1
				}
			}

			// Create math.Inf(1) with type conversion for float32
			posInfExpr := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent("stdmath"),
					Sel: ast.NewIdent("Inf"),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "1"}},
			}
			// For float32, wrap in type conversion
			var posInfArg ast.Expr = posInfExpr
			if ctx.elemType == "float32" {
				posInfArg = &ast.CallExpr{
					Fun:  ast.NewIdent("float32"),
					Args: []ast.Expr{posInfExpr},
				}
			}

			// Create pkg.Broadcast*(posInf) for +Inf
			posInfCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{posInfArg},
			}

			// Create math.Inf(-1) with type conversion for float32
			negInfExpr := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent("stdmath"),
					Sel: ast.NewIdent("Inf"),
				},
				Args: []ast.Expr{
					&ast.UnaryExpr{
						Op: token.SUB,
						X:  &ast.BasicLit{Kind: token.INT, Value: "1"},
					},
				},
			}
			// For float32, wrap in type conversion
			var negInfArg ast.Expr = negInfExpr
			if ctx.elemType == "float32" {
				negInfArg = &ast.CallExpr{
					Fun:  ast.NewIdent("float32"),
					Args: []ast.Expr{negInfExpr},
				}
			}

			// Create pkg.Broadcast*(negInf) for -Inf
			negInfCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{negInfArg},
			}

			switch signVal {
			case 1:
				// Check +Inf only: x.Equal(posInf)
				call.Fun = &ast.SelectorExpr{
					X:   x,
					Sel: ast.NewIdent("Equal"),
				}
				call.Args = []ast.Expr{posInfCall}
			case -1:
				// Check -Inf only: x.Equal(negInf)
				call.Fun = &ast.SelectorExpr{
					X:   x,
					Sel: ast.NewIdent("Equal"),
				}
				call.Args = []ast.Expr{negInfCall}
			default:
				// Check either: x.Equal(posInf).Or(x.Equal(negInf))
				posInfMask := &ast.CallExpr{
					Fun: &ast.SelectorExpr{
						X:   cloneExpr(x),
						Sel: ast.NewIdent("Equal"),
					},
					Args: []ast.Expr{posInfCall},
				}
				negInfMask := &ast.CallExpr{
					Fun: &ast.SelectorExpr{
						X:   x,
						Sel: ast.NewIdent("Equal"),
					},
					Args: []ast.Expr{negInfCall},
				}
				call.Fun = &ast.SelectorExpr{
					X:   posInfMask,
					Sel: ast.NewIdent("Or"),
				}
				call.Args = []ast.Expr{negInfMask}
			}
		}

	default:
		// Binary operations: hwy.Add(a, b) -> a.Add(b)
		if len(call.Args) >= 2 {
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent(opInfo.Name),
			}
			call.Args = call.Args[1:]
		} else if len(call.Args) == 1 {
			// Other unary operations
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent(opInfo.Name),
			}
			call.Args = nil
		}
	}
}

// transformToFunction converts hwy.Load(src) to archsimd.LoadFloat32x8Slice(src).
func transformToFunction(call *ast.CallExpr, funcName string, opInfo OpInfo, ctx *transformContext) {
	// Handle both SelectorExpr (hwy.Load) and IndexExpr (hwy.Zero[float32])
	var selExpr *ast.SelectorExpr
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		selExpr = fun
	case *ast.IndexExpr:
		// For fallback generic functions like hwy.Zero[float32]()
		selExpr = fun.X.(*ast.SelectorExpr)
	default:
		return
	}

	if ctx.target.Name == "Fallback" {
		// For fallback, use the appropriate package
		if opInfo.SubPackage != "" {
			// Contrib functions stay in their subpackage (e.g., math.Exp stays math.Exp)
			selExpr.X = ast.NewIdent(opInfo.SubPackage)
		} else {
			// Core ops use hwy package
			selExpr.X = ast.NewIdent("hwy")
		}
		selExpr.Sel.Name = funcName
		return
	}

	// For SIMD targets, transform to package calls (archsimd for AVX, asm for NEON)
	var fullName string
	vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)
	pkgName := getVecPackageName(ctx.target)

	switch funcName {
	case "Load":
		fullName = fmt.Sprintf("Load%sSlice", vecTypeName)
		selExpr.X = ast.NewIdent(pkgName)
	case "Set":
		fullName = fmt.Sprintf("Broadcast%s", vecTypeName)
		selExpr.X = ast.NewIdent(pkgName)
	case "Zero":
		if opInfo.Package == "special" {
			// archsimd doesn't have Zero*, use Broadcast with 0
			fullName = fmt.Sprintf("Broadcast%s", vecTypeName)
			selExpr.X = ast.NewIdent(pkgName)
			// Add 0 as argument
			call.Args = []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "0"}}
		} else {
			fullName = fmt.Sprintf("Zero%s", vecTypeName)
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "MaskLoad":
		fullName = fmt.Sprintf("MaskLoad%sSlice", vecTypeName)
		selExpr.X = ast.NewIdent(pkgName)
	default:
		// For contrib functions, add target and type suffix (e.g., math.Exp_AVX2_F32x8)
		if opInfo.SubPackage != "" {
			fullName = fmt.Sprintf("%s_%s_%s", opInfo.Name, ctx.target.Name, getShortTypeName(ctx.elemType, ctx.target))
			selExpr.X = ast.NewIdent(opInfo.SubPackage) // math, dot, matvec, algo
		} else if opInfo.Package == "hwy" {
			// Core ops from hwy package (e.g., hwy.Sqrt_AVX2_F32x8)
			fullName = fmt.Sprintf("%s_%s_%s", opInfo.Name, ctx.target.Name, getShortTypeName(ctx.elemType, ctx.target))
			selExpr.X = ast.NewIdent("hwy")
		} else {
			fullName = opInfo.Name
			selExpr.X = ast.NewIdent(pkgName)
		}
	}

	selExpr.Sel.Name = fullName
}

// getVecPackageName returns the package name for vector types based on target.
// Returns "archsimd" for AVX targets, "asm" for NEON.
func getVecPackageName(target Target) string {
	switch target.VecPackage {
	case "archsimd":
		return "archsimd"
	case "asm":
		return "asm"
	default:
		return "archsimd" // default for compatibility
	}
}

// getShortTypeName returns the short type name like F32x8 for contrib functions.
func getShortTypeName(elemType string, target Target) string {
	lanes := target.LanesFor(elemType)
	switch elemType {
	case "float32":
		return fmt.Sprintf("F32x%d", lanes)
	case "float64":
		return fmt.Sprintf("F64x%d", lanes)
	case "int32":
		return fmt.Sprintf("I32x%d", lanes)
	case "int64":
		return fmt.Sprintf("I64x%d", lanes)
	default:
		return "Vec"
	}
}

// transformGenDecl transforms variable declarations with generic types.
func transformGenDecl(decl *ast.GenDecl, ctx *transformContext) {
	if decl.Tok != token.VAR && decl.Tok != token.CONST {
		return
	}

	for _, spec := range decl.Specs {
		valueSpec, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}

		// Transform type if present
		if valueSpec.Type != nil {
			typeStr := exprToString(valueSpec.Type)
			specialized := specializeType(typeStr, ctx.typeParams, ctx.elemType)
			if specialized != typeStr {
				valueSpec.Type = parseTypeExpr(specialized)
			}
		}
	}
}

// transformAssignStmt transforms assignments, particularly for loop stride calculations
// and hoisting hwy.Set calls with constant values.
func transformAssignStmt(stmt *ast.AssignStmt, ctx *transformContext) {
	// For fallback, don't replace NumLanes with a constant - keep it dynamic
	if ctx.target.Name == "Fallback" {
		return
	}

	// Look for v.NumElements(), hwy.Lanes[T](), or similar and replace with constant
	for i, rhs := range stmt.Rhs {
		if call, ok := rhs.(*ast.CallExpr); ok {
			// Check for hwy.Lanes[T]() - IndexExpr wrapping SelectorExpr
			if indexExpr, ok := call.Fun.(*ast.IndexExpr); ok {
				if sel, ok := indexExpr.X.(*ast.SelectorExpr); ok {
					if pkgIdent, ok := sel.X.(*ast.Ident); ok {
						if pkgIdent.Name == "hwy" && (sel.Sel.Name == "Lanes" || sel.Sel.Name == "MaxLanes") {
							// Replace with constant lane count
							lanes := ctx.target.LanesFor(ctx.elemType)
							stmt.Rhs[i] = &ast.BasicLit{
								Kind:  token.INT,
								Value: strconv.Itoa(lanes),
							}
							// Track the variable name
							if len(stmt.Lhs) > i {
								if ident, ok := stmt.Lhs[i].(*ast.Ident); ok {
									ctx.lanesVars[ident.Name] = true
								}
							}
							continue
						}
					}
				}
			}
			// Check for v.NumElements() or v.NumLanes()
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if sel.Sel.Name == "NumElements" || sel.Sel.Name == "NumLanes" {
					// Replace with constant lane count
					lanes := ctx.target.LanesFor(ctx.elemType)
					stmt.Rhs[i] = &ast.BasicLit{
						Kind:  token.INT,
						Value: strconv.Itoa(lanes),
					}
					// Track the variable name so we can recognize it in loop strides
					if len(stmt.Lhs) > i {
						if ident, ok := stmt.Lhs[i].(*ast.Ident); ok {
							ctx.lanesVars[ident.Name] = true
						}
					}
				}
			}
		}

		// Check for make([]T, ...) calls
		if call, ok := rhs.(*ast.CallExpr); ok {
			if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "make" {
				if len(call.Args) >= 2 {
					// Check if first arg is []T (slice type)
					if arrayType, ok := call.Args[0].(*ast.ArrayType); ok {
						if arrayType.Len == nil { // It's a slice, not an array
							// Specialize the element type (T -> float32/float64)
							elemTypeStr := exprToString(arrayType.Elt)
							specializedType := specializeType(elemTypeStr, ctx.typeParams, ctx.elemType)

							// Check if second arg is a lanes variable or literal for stack array optimization
							var lanesCount int
							switch sizeArg := call.Args[1].(type) {
							case *ast.Ident:
								if ctx.lanesVars[sizeArg.Name] {
									lanesCount = ctx.target.LanesFor(ctx.elemType)
								}
							case *ast.BasicLit:
								if sizeArg.Kind == token.INT {
									lanesCount, _ = strconv.Atoi(sizeArg.Value)
								}
							}

							if lanesCount > 0 {
								// Replace make([]T, lanes) with [lanes]T{} (zero-valued array literal)
								stmt.Rhs[i] = &ast.CompositeLit{
									Type: &ast.ArrayType{
										Len: &ast.BasicLit{
											Kind:  token.INT,
											Value: strconv.Itoa(lanesCount),
										},
										Elt: parseTypeExpr(specializedType),
									},
								}
								// Track this variable as a stack array
								if len(stmt.Lhs) > i {
									if ident, ok := stmt.Lhs[i].(*ast.Ident); ok {
										ctx.stackArrayVars[ident.Name] = true
									}
								}
							} else if elemTypeStr != specializedType {
								// Just replace T with concrete type in make call
								arrayType.Elt = parseTypeExpr(specializedType)
							}
						}
					}
				}
			}
		}

		// Check for hwy.Set[T](constant) calls that can be hoisted
		if hoistedName := tryHoistSetCall(stmt, i, rhs, ctx); hoistedName != "" {
			// Replace RHS with reference to hoisted variable
			stmt.Rhs[i] = ast.NewIdent(hoistedName)
		}
	}
}

// tryHoistSetCall checks if an expression is a hwy.Set[T](constant) call
// and if so, registers it for hoisting and returns the hoisted variable name.
func tryHoistSetCall(stmt *ast.AssignStmt, rhsIndex int, rhs ast.Expr, ctx *transformContext) string {
	call, ok := rhs.(*ast.CallExpr)
	if !ok {
		return ""
	}

	// Check for hwy.Set[T](arg) pattern - could be IndexExpr wrapping SelectorExpr
	var selExpr *ast.SelectorExpr
	var typeParam string
	switch fun := call.Fun.(type) {
	case *ast.IndexExpr:
		// hwy.Set[T](arg)
		selExpr, ok = fun.X.(*ast.SelectorExpr)
		if !ok {
			return ""
		}
		// Extract the type parameter
		if typeIdent, ok := fun.Index.(*ast.Ident); ok {
			typeParam = typeIdent.Name
		}
	case *ast.SelectorExpr:
		// hwy.Set(arg) - non-generic, shouldn't happen but handle it
		selExpr = fun
	default:
		return ""
	}

	// Verify it's hwy.Set
	ident, ok := selExpr.X.(*ast.Ident)
	if !ok || ident.Name != "hwy" {
		return ""
	}
	if selExpr.Sel.Name != "Set" {
		return ""
	}

	// Determine the actual element type for this Set call
	// If the type parameter is explicitly "int32", use that instead of ctx.elemType
	actualElemType := ctx.elemType
	if typeParam == "int32" {
		actualElemType = "int32"
	}

	// Check if the argument is a constant (literal or type conversion of constant)
	if len(call.Args) != 1 {
		return ""
	}
	arg := call.Args[0]
	constValue := extractConstantValue(arg, actualElemType, ctx)
	if constValue == "" {
		return ""
	}

	// Get the local variable name being assigned
	if rhsIndex >= len(stmt.Lhs) {
		return ""
	}
	localIdent, ok := stmt.Lhs[rhsIndex].(*ast.Ident)
	if !ok {
		return ""
	}
	localVarName := localIdent.Name

	// Generate unique hoisted variable name (include target to avoid conflicts)
	// For int32 constants, we need separate versions for f32 and f64 functions
	// because they have different lane counts
	elemSuffix := "f32"
	if ctx.elemType == "float64" {
		elemSuffix = "f64"
	}
	if actualElemType == "int32" {
		// Include parent element type in suffix for proper lane matching
		if ctx.elemType == "float64" {
			elemSuffix = "i32_f64"
		} else {
			elemSuffix = "i32_f32"
		}
	}
	hoistedName := fmt.Sprintf("%s_%s_%s_%s", ctx.funcName, ctx.target.Name, localVarName, elemSuffix)

	// Get vector type and broadcast function for this target
	// For int32 types used in float operations, match the lane count of the parent element type
	vecTypeName := getVectorTypeNameForInt(actualElemType, ctx.elemType, ctx.target)
	pkgName := getVecPackageName(ctx.target)
	broadcastFunc := fmt.Sprintf("%s.Broadcast%s", pkgName, vecTypeName)

	// Register the hoisted constant
	ctx.hoistedConsts[localVarName] = HoistedConst{
		VarName:   hoistedName,
		Value:     constValue,
		VecType:   vecTypeName,
		Broadcast: broadcastFunc,
	}

	return hoistedName
}

// extractConstantValue extracts the string representation of a constant value.
// Returns empty string if the expression is not a constant.
// The elemType parameter is used to add type conversion when needed.
// The ctx is used to check if a variable is locally-defined (not a constant).
func extractConstantValue(expr ast.Expr, elemType string, ctx *transformContext) string {
	switch e := expr.(type) {
	case *ast.BasicLit:
		// Literal like 1.0, 0.5, etc.
		return e.Value
	case *ast.UnaryExpr:
		// Handle negative literals like -1.0
		if e.Op == token.SUB {
			if inner := extractConstantValueRaw(e.X, ctx); inner != "" {
				return "-" + inner
			}
		}
	case *ast.CallExpr:
		// Type conversion like T(1.0) or float32(sigmoidC1)
		// Get the inner value without adding another type conversion
		if len(e.Args) == 1 {
			inner := extractConstantValueRaw(e.Args[0], ctx)
			if inner != "" {
				// Add the target type conversion
				return fmt.Sprintf("%s(%s)", elemType, inner)
			}
		}
	case *ast.Ident:
		// Variable reference - only hoist if it's NOT a local variable
		name := e.Name
		if ctx.localVars[name] {
			// This is a locally-defined variable, not a package-level constant
			return ""
		}
		if isLikelyConstant(name) {
			// Add type conversion in case the var type differs from target type
			return fmt.Sprintf("%s(%s)", elemType, name)
		}
	}
	return ""
}

// extractConstantValueRaw extracts the raw constant value without type conversion.
func extractConstantValueRaw(expr ast.Expr, ctx *transformContext) string {
	switch e := expr.(type) {
	case *ast.BasicLit:
		return e.Value
	case *ast.UnaryExpr:
		if e.Op == token.SUB {
			if inner := extractConstantValueRaw(e.X, ctx); inner != "" {
				return "-" + inner
			}
		}
	case *ast.Ident:
		name := e.Name
		// Skip if it's a locally-defined variable
		if ctx != nil && ctx.localVars[name] {
			return ""
		}
		if isLikelyConstant(name) {
			return name
		}
	}
	return ""
}

// isLikelyConstant checks if a name looks like a package-level constant.
// This is a heuristic - constants typically have specific naming patterns.
// Note: This function is only called AFTER checking that the name is not
// in ctx.localVars, so locally-defined variables are already excluded.
func isLikelyConstant(name string) bool {
	// Skip very common short names that are almost never constants
	skipNames := map[string]bool{
		"i": true, "j": true, "k": true, "ii": true, "jj": true,
		"x": true, "y": true, "z": true, "n": true, "m": true,
		"a": true, "b": true, "c": true, "v": true, "w": true,
		"err": true, "ok": true,
	}
	if skipNames[name] {
		return false
	}

	// Constants typically:
	// 1. Contain digits (e.g., sigmoidC1, ln2Hi, exp2bias, c1, c2)
	// 2. Are all uppercase (e.g., PI, MAX_VALUE)
	hasDigit := false
	hasLower := false
	hasUpper := false
	for _, r := range name {
		if r >= '0' && r <= '9' {
			hasDigit = true
		} else if r >= 'a' && r <= 'z' {
			hasLower = true
		} else if r >= 'A' && r <= 'Z' {
			hasUpper = true
		}
	}

	// Accept if it has a digit (like sigmoidC1, ln2Hi)
	if hasDigit {
		return true
	}

	// Accept if all uppercase (like PI, MAX_VALUE)
	if hasUpper && !hasLower {
		return true
	}

	// Reject short lowercase names that look like local variables
	if len(name) <= 4 && hasLower {
		return false
	}

	// Accept longer camelCase names that look like package-level constants
	// (e.g., sigmoidScale, expBias, tanhClamp)
	return len(name) > 4
}

// matchesLoopIterator checks if a for loop uses the given iterator name.
// It checks both the init statement (for ii := 0) and the condition (ii < size).
func matchesLoopIterator(forStmt *ast.ForStmt, iteratorName string) bool {
	// Check init statement: for ii := 0
	if forStmt.Init != nil {
		if assign, ok := forStmt.Init.(*ast.AssignStmt); ok {
			for _, lhs := range assign.Lhs {
				if ident, ok := lhs.(*ast.Ident); ok && ident.Name == iteratorName {
					return true
				}
			}
		}
	}

	// Check condition: ii < size or ii+N <= size
	if forStmt.Cond != nil {
		if binExpr, ok := forStmt.Cond.(*ast.BinaryExpr); ok {
			// Check LHS directly (ii < size)
			if ident, ok := binExpr.X.(*ast.Ident); ok && ident.Name == iteratorName {
				return true
			}
			// Check LHS if it's a binary expression (ii+N <= size)
			if innerBin, ok := binExpr.X.(*ast.BinaryExpr); ok {
				if ident, ok := innerBin.X.(*ast.Ident); ok && ident.Name == iteratorName {
					return true
				}
			}
		}
	}

	return false
}

// insertTailHandling adds scalar tail handling after the vectorized loop.
func insertTailHandling(body *ast.BlockStmt, loopInfo *LoopInfo, elemType string, target Target, funcName string, params []Param) {
	if body == nil || loopInfo == nil {
		return
	}

	// For fallback, no tail handling needed - callers must provide inputs >= vector width
	if target.Name == "Fallback" {
		return
	}

	// Find the SIMD loop that uses loopInfo.Iterator as its iterator.
	// This ensures we don't add tail handling after unrelated loops (e.g., scalar loops).
	var loopIdx int = -1
	var mainLoop *ast.ForStmt
	for i, stmt := range body.List {
		if forStmt, ok := stmt.(*ast.ForStmt); ok {
			// Check if this loop's iterator matches loopInfo.Iterator
			if matchesLoopIterator(forStmt, loopInfo.Iterator) {
				loopIdx = i
				mainLoop = forStmt
				break
			}
		}
	}

	if mainLoop == nil || loopIdx < 0 {
		return
	}

	// Declare the iterator before the loop so it's in scope for the tail
	// Change: for ii := 0; ... to: ii := 0; for ; ...
	var initStmt ast.Stmt
	if mainLoop.Init != nil {
		initStmt = mainLoop.Init
		mainLoop.Init = nil
	}

	// Build tail handling that calls the fallback function for remaining elements
	// if ii < size {
	//     BaseSigmoid_fallback(in[ii:size], out[ii:size])
	// }
	fallbackFuncName := funcName + "_fallback"
	// Add type suffix for non-float32 types (matches how generator.go names functions)
	if elemType != "float32" {
		fallbackFuncName = fallbackFuncName + "_" + strings.Title(elemType)
	}

	// Build arguments for the fallback call
	// For slice parameters: param[ii:size]
	// For non-slice parameters: pass as-is
	var callArgs []ast.Expr
	for _, param := range params {
		if strings.HasPrefix(param.Type, "[]") {
			// Create param[ii:size] for slice parameters
			sliceExpr := &ast.SliceExpr{
				X:    ast.NewIdent(param.Name),
				Low:  ast.NewIdent(loopInfo.Iterator),
				High: ast.NewIdent(loopInfo.End),
			}
			callArgs = append(callArgs, sliceExpr)
		} else {
			// Pass non-slice parameters as-is
			callArgs = append(callArgs, ast.NewIdent(param.Name))
		}
	}

	// Create the fallback call: BasePoly2_fallback(x[ii:size], c0, c1, c2, result[ii:size])
	fallbackCall := &ast.CallExpr{
		Fun:  ast.NewIdent(fallbackFuncName),
		Args: callArgs,
	}

	// Wrap in if statement: if ii < size { ... }
	tailIf := &ast.IfStmt{
		Cond: &ast.BinaryExpr{
			X:  ast.NewIdent(loopInfo.Iterator),
			Op: token.LSS,
			Y:  ast.NewIdent(loopInfo.End),
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				&ast.ExprStmt{X: fallbackCall},
			},
		},
	}

	// Insert init statement, main loop, and tail handling
	newStmts := make([]ast.Stmt, 0, len(body.List)+2)
	newStmts = append(newStmts, body.List[:loopIdx]...)
	if initStmt != nil {
		newStmts = append(newStmts, initStmt)
	}
	newStmts = append(newStmts, mainLoop)
	newStmts = append(newStmts, tailIf)
	newStmts = append(newStmts, body.List[loopIdx+1:]...)
	body.List = newStmts
}


// specializeType replaces generic type parameters with concrete types.
// For SIMD targets, also transforms hwy.Vec[T] to archsimd/asm vector types.
func specializeType(typeStr string, typeParams []TypeParam, elemType string) string {
	for _, tp := range typeParams {
		// Replace hwy.Vec[T] with concrete vector type placeholder
		// This will be further processed by specializeVecType
		typeStr = strings.ReplaceAll(typeStr, "hwy.Vec["+tp.Name+"]", "hwy.Vec["+elemType+"]")
		// Replace []T with []float32, etc.
		typeStr = strings.ReplaceAll(typeStr, "[]"+tp.Name, "[]"+elemType)
		typeStr = strings.ReplaceAll(typeStr, tp.Name, elemType)
	}
	return typeStr
}

// specializeVecType transforms hwy.Vec[elemType] to the concrete archsimd/asm type.
// For example: hwy.Vec[float32] -> archsimd.Float32x8 (for AVX2)
func specializeVecType(typeStr string, elemType string, target Target) string {
	vecPlaceholder := "hwy.Vec[" + elemType + "]"
	if !strings.Contains(typeStr, vecPlaceholder) {
		return typeStr
	}

	if target.Name == "Fallback" {
		// For fallback, keep hwy.Vec[float32] etc.
		return typeStr
	}

	// Get the concrete vector type from target's TypeMap
	vecTypeName, ok := target.TypeMap[elemType]
	if !ok {
		return typeStr
	}

	// Build fully qualified type: archsimd.Float32x8 or asm.Float32x4
	pkgName := target.VecPackage
	if pkgName == "" {
		pkgName = "archsimd" // default
	}
	concreteType := pkgName + "." + vecTypeName

	return strings.ReplaceAll(typeStr, vecPlaceholder, concreteType)
}

// getVectorTypeName returns the vector type name for archsimd functions.
func getVectorTypeName(elemType string, target Target) string {
	lanes := target.LanesFor(elemType)
	switch elemType {
	case "float32":
		return fmt.Sprintf("Float32x%d", lanes)
	case "float64":
		return fmt.Sprintf("Float64x%d", lanes)
	case "int32":
		return fmt.Sprintf("Int32x%d", lanes)
	case "int64":
		return fmt.Sprintf("Int64x%d", lanes)
	default:
		return "Vec"
	}
}

// getVectorTypeNameForInt returns the vector type name for int types used
// in float operations. The lane count matches the parent element type.
// For example, int32 in a float64 function needs Int32x2 (matching Float64x2 lanes).
func getVectorTypeNameForInt(intType, parentElemType string, target Target) string {
	if intType != "int32" && intType != "int64" {
		// Non-integer type, use regular logic
		return getVectorTypeName(intType, target)
	}

	// Match lanes to parent element type
	lanes := target.LanesFor(parentElemType)
	switch intType {
	case "int32":
		return fmt.Sprintf("Int32x%d", lanes)
	case "int64":
		return fmt.Sprintf("Int64x%d", lanes)
	default:
		return getVectorTypeName(intType, target)
	}
}

// parseTypeExpr converts a type string back to an AST expression.
func parseTypeExpr(typeStr string) ast.Expr {
	// Handle slice types
	if strings.HasPrefix(typeStr, "[]") {
		return &ast.ArrayType{
			Elt: parseTypeExpr(typeStr[2:]),
		}
	}

	// Handle pointer types
	if strings.HasPrefix(typeStr, "*") {
		return &ast.StarExpr{
			X: parseTypeExpr(typeStr[1:]),
		}
	}

	// Handle function types like func(archsimd.Float32x8) archsimd.Float32x8
	if strings.HasPrefix(typeStr, "func(") {
		return parseFuncType(typeStr)
	}

	// Handle generic types like hwy.Vec[float32] or Vec[float32]
	if bracketIdx := strings.Index(typeStr, "["); bracketIdx >= 0 {
		closeBracket := strings.LastIndex(typeStr, "]")
		if closeBracket > bracketIdx {
			baseType := typeStr[:bracketIdx]
			typeArg := typeStr[bracketIdx+1 : closeBracket]

			// Parse the base type (could be pkg.Type or just Type)
			baseExpr := parseTypeExpr(baseType)

			// Create IndexExpr for the generic instantiation
			return &ast.IndexExpr{
				X:     baseExpr,
				Index: parseTypeExpr(typeArg),
			}
		}
	}

	// Handle qualified names (pkg.Type)
	if idx := strings.Index(typeStr, "."); idx >= 0 {
		return &ast.SelectorExpr{
			X:   ast.NewIdent(typeStr[:idx]),
			Sel: ast.NewIdent(typeStr[idx+1:]),
		}
	}

	// Simple identifier
	return ast.NewIdent(typeStr)
}

// parseFuncType parses a function type string like "func(archsimd.Float32x8) archsimd.Float32x8"
func parseFuncType(typeStr string) *ast.FuncType {
	// Find the matching closing paren for the params
	parenDepth := 0
	paramsEnd := -1
	for i := 5; i < len(typeStr); i++ { // Start after "func("
		switch typeStr[i] {
		case '(':
			parenDepth++
		case ')':
			if parenDepth == 0 {
				paramsEnd = i
				break
			}
			parenDepth--
		}
		if paramsEnd >= 0 {
			break
		}
	}
	if paramsEnd < 0 {
		// Malformed, return empty func type
		return &ast.FuncType{}
	}

	// Extract params string (between "func(" and ")")
	paramsStr := typeStr[5:paramsEnd]

	// Extract results string (after ")")
	resultsStr := strings.TrimSpace(typeStr[paramsEnd+1:])
	// Remove surrounding parens from results if present
	if strings.HasPrefix(resultsStr, "(") && strings.HasSuffix(resultsStr, ")") {
		resultsStr = resultsStr[1 : len(resultsStr)-1]
	}

	// Parse params
	var params []*ast.Field
	if paramsStr != "" {
		for _, paramType := range splitTypeList(paramsStr) {
			paramType = strings.TrimSpace(paramType)
			if paramType != "" {
				params = append(params, &ast.Field{
					Type: parseTypeExpr(paramType),
				})
			}
		}
	}

	// Parse results
	var results []*ast.Field
	if resultsStr != "" {
		for _, resultType := range splitTypeList(resultsStr) {
			resultType = strings.TrimSpace(resultType)
			if resultType != "" {
				results = append(results, &ast.Field{
					Type: parseTypeExpr(resultType),
				})
			}
		}
	}

	funcType := &ast.FuncType{
		Params: &ast.FieldList{List: params},
	}
	if len(results) > 0 {
		funcType.Results = &ast.FieldList{List: results}
	}
	return funcType
}

// splitTypeList splits a comma-separated type list, respecting nested brackets and parens.
func splitTypeList(s string) []string {
	var parts []string
	depth := 0
	start := 0
	for i, c := range s {
		switch c {
		case '(', '[':
			depth++
		case ')', ']':
			depth--
		case ',':
			if depth == 0 {
				parts = append(parts, s[start:i])
				start = i + 1
			}
		}
	}
	if start < len(s) {
		parts = append(parts, s[start:])
	}
	return parts
}

// cloneBlockStmt creates a deep copy of a block statement.
func cloneBlockStmt(block *ast.BlockStmt) *ast.BlockStmt {
	if block == nil {
		return nil
	}

	newBlock := &ast.BlockStmt{
		List: make([]ast.Stmt, len(block.List)),
	}

	for i, stmt := range block.List {
		newBlock.List[i] = cloneStmt(stmt)
	}

	return newBlock
}

// cloneStmt creates a deep copy of a statement.
func cloneStmt(stmt ast.Stmt) ast.Stmt {
	if stmt == nil {
		return nil
	}

	switch s := stmt.(type) {
	case *ast.ExprStmt:
		return &ast.ExprStmt{X: cloneExpr(s.X)}
	case *ast.AssignStmt:
		return cloneAssignStmt(s)
	case *ast.DeclStmt:
		return &ast.DeclStmt{Decl: cloneDecl(s.Decl)}
	case *ast.ReturnStmt:
		return cloneReturnStmt(s)
	case *ast.ForStmt:
		return cloneForStmt(s)
	case *ast.IfStmt:
		return cloneIfStmt(s)
	case *ast.IncDecStmt:
		return &ast.IncDecStmt{X: cloneExpr(s.X), Tok: s.Tok}
	case *ast.BranchStmt:
		return &ast.BranchStmt{Tok: s.Tok, Label: s.Label}
	case *ast.BlockStmt:
		return cloneBlockStmt(s)
	case *ast.RangeStmt:
		return &ast.RangeStmt{
			Key:   cloneExpr(s.Key),
			Value: cloneExpr(s.Value),
			Tok:   s.Tok,
			X:     cloneExpr(s.X),
			Body:  cloneBlockStmt(s.Body),
		}
	default:
		// For other statement types, return as-is
		return stmt
	}
}

// cloneExpr creates a deep copy of an expression.
func cloneExpr(expr ast.Expr) ast.Expr {
	if expr == nil {
		return nil
	}

	switch e := expr.(type) {
	case *ast.Ident:
		return &ast.Ident{Name: e.Name}
	case *ast.BasicLit:
		return &ast.BasicLit{Kind: e.Kind, Value: e.Value}
	case *ast.SelectorExpr:
		return &ast.SelectorExpr{
			X:   cloneExpr(e.X),
			Sel: ast.NewIdent(e.Sel.Name),
		}
	case *ast.CallExpr:
		args := make([]ast.Expr, len(e.Args))
		for i, arg := range e.Args {
			args[i] = cloneExpr(arg)
		}
		return &ast.CallExpr{
			Fun:  cloneExpr(e.Fun),
			Args: args,
		}
	case *ast.BinaryExpr:
		return &ast.BinaryExpr{
			X:  cloneExpr(e.X),
			Op: e.Op,
			Y:  cloneExpr(e.Y),
		}
	case *ast.UnaryExpr:
		return &ast.UnaryExpr{
			Op: e.Op,
			X:  cloneExpr(e.X),
		}
	case *ast.ParenExpr:
		return &ast.ParenExpr{X: cloneExpr(e.X)}
	case *ast.IndexExpr:
		return &ast.IndexExpr{
			X:     cloneExpr(e.X),
			Index: cloneExpr(e.Index),
		}
	case *ast.SliceExpr:
		return &ast.SliceExpr{
			X:      cloneExpr(e.X),
			Low:    cloneExpr(e.Low),
			High:   cloneExpr(e.High),
			Max:    cloneExpr(e.Max),
			Slice3: e.Slice3,
		}
	case *ast.StarExpr:
		return &ast.StarExpr{X: cloneExpr(e.X)}
	case *ast.ArrayType:
		return &ast.ArrayType{
			Len: cloneExpr(e.Len),
			Elt: cloneExpr(e.Elt),
		}
	case *ast.CompositeLit:
		elts := make([]ast.Expr, len(e.Elts))
		for i, elt := range e.Elts {
			elts[i] = cloneExpr(elt)
		}
		return &ast.CompositeLit{
			Type: cloneExpr(e.Type),
			Elts: elts,
		}
	default:
		// For unsupported types, return as-is (may cause issues for complex expressions)
		return expr
	}
}

// cloneAssignStmt clones an assignment statement.
func cloneAssignStmt(stmt *ast.AssignStmt) *ast.AssignStmt {
	newStmt := &ast.AssignStmt{
		Lhs: make([]ast.Expr, len(stmt.Lhs)),
		Rhs: make([]ast.Expr, len(stmt.Rhs)),
		Tok: stmt.Tok,
	}
	for i, lhs := range stmt.Lhs {
		newStmt.Lhs[i] = cloneExpr(lhs)
	}
	for i, rhs := range stmt.Rhs {
		newStmt.Rhs[i] = cloneExpr(rhs)
	}
	return newStmt
}

// cloneDecl clones a declaration.
func cloneDecl(decl ast.Decl) ast.Decl {
	if decl == nil {
		return nil
	}

	switch d := decl.(type) {
	case *ast.GenDecl:
		newSpecs := make([]ast.Spec, len(d.Specs))
		for i, spec := range d.Specs {
			newSpecs[i] = cloneSpec(spec)
		}
		return &ast.GenDecl{
			Tok:   d.Tok,
			Specs: newSpecs,
		}
	default:
		return decl
	}
}

// cloneSpec clones a declaration spec (e.g., variable declaration).
func cloneSpec(spec ast.Spec) ast.Spec {
	if spec == nil {
		return nil
	}

	switch s := spec.(type) {
	case *ast.ValueSpec:
		var newValues []ast.Expr
		if len(s.Values) > 0 {
			newValues = make([]ast.Expr, len(s.Values))
			for i, v := range s.Values {
				newValues[i] = cloneExpr(v)
			}
		}
		newNames := make([]*ast.Ident, len(s.Names))
		for i, n := range s.Names {
			newNames[i] = &ast.Ident{Name: n.Name}
		}
		return &ast.ValueSpec{
			Names:  newNames,
			Type:   cloneExpr(s.Type),
			Values: newValues,
		}
	default:
		return spec
	}
}

// cloneReturnStmt clones a return statement.
func cloneReturnStmt(stmt *ast.ReturnStmt) *ast.ReturnStmt {
	newStmt := &ast.ReturnStmt{
		Results: make([]ast.Expr, len(stmt.Results)),
	}
	for i, result := range stmt.Results {
		newStmt.Results[i] = cloneExpr(result)
	}
	return newStmt
}

// cloneForStmt clones a for loop.
func cloneForStmt(stmt *ast.ForStmt) *ast.ForStmt {
	return &ast.ForStmt{
		Init: cloneStmt(stmt.Init),
		Cond: cloneExpr(stmt.Cond),
		Post: cloneStmt(stmt.Post),
		Body: cloneBlockStmt(stmt.Body),
	}
}

// cloneIfStmt clones an if statement.
func cloneIfStmt(stmt *ast.IfStmt) *ast.IfStmt {
	return &ast.IfStmt{
		Init: cloneStmt(stmt.Init),
		Cond: cloneExpr(stmt.Cond),
		Body: cloneBlockStmt(stmt.Body),
		Else: cloneStmt(stmt.Else),
	}
}

// buildResults builds the return type list for a function.
func (pf *ParsedFunc) buildResults(elemType string) *ast.FieldList {
	if len(pf.Returns) == 0 {
		return nil
	}

	fieldList := &ast.FieldList{
		List: make([]*ast.Field, 0, len(pf.Returns)),
	}

	for _, ret := range pf.Returns {
		retType := specializeType(ret.Type, pf.TypeParams, elemType)
		field := &ast.Field{
			Type: parseTypeExpr(retType),
		}
		if ret.Name != "" {
			field.Names = []*ast.Ident{ast.NewIdent(ret.Name)}
		}
		fieldList.List = append(fieldList.List, field)
	}

	return fieldList
}

// buildResultsWithTarget builds the return type list with target-specific Vec types.
func (pf *ParsedFunc) buildResultsWithTarget(elemType string, target Target) *ast.FieldList {
	if len(pf.Returns) == 0 {
		return nil
	}

	fieldList := &ast.FieldList{
		List: make([]*ast.Field, 0, len(pf.Returns)),
	}

	for _, ret := range pf.Returns {
		retType := specializeType(ret.Type, pf.TypeParams, elemType)
		// Transform hwy.Vec[T] to concrete vector types for SIMD targets
		retType = specializeVecType(retType, elemType, target)
		field := &ast.Field{
			Type: parseTypeExpr(retType),
		}
		if ret.Name != "" {
			field.Names = []*ast.Ident{ast.NewIdent(ret.Name)}
		}
		fieldList.List = append(fieldList.List, field)
	}

	return fieldList
}

// postProcessSIMD walks the AST and replaces NumLanes() calls with constants
// and transforms ReduceSum() calls to store+sum patterns.
func postProcessSIMD(node ast.Node, ctx *transformContext) {
	if node == nil {
		return
	}

	lanes := ctx.target.LanesFor(ctx.elemType)
	vecTypeName := getVectorTypeName(ctx.elemType, ctx.target)

	// Walk all statements and expressions, replacing as needed
	ast.Inspect(node, func(n ast.Node) bool {
		switch stmt := n.(type) {
		case *ast.IfStmt:
			// Replace comparisons like: remaining >= v.NumLanes()
			if binExpr, ok := stmt.Cond.(*ast.BinaryExpr); ok {
				replaceNumLanesInExpr(binExpr, lanes)
			}
		case *ast.AssignStmt:
			// Replace: sum += v.ReduceSum() or sum += hwy.ReduceSum(v)
			for i, rhs := range stmt.Rhs {
				if call, ok := rhs.(*ast.CallExpr); ok {
					if isReduceSumCall(call) {
						// Transform to store + sum pattern
						stmt.Rhs[i] = createReduceSumExpr(call, lanes, vecTypeName, ctx.elemType)
					}
				}
			}
		case *ast.ExprStmt:
			// Handle standalone expressions if needed
		}
		return true
	})
}

// replaceNumLanesInExpr replaces v.NumLanes() with a constant in a binary expression.
func replaceNumLanesInExpr(binExpr *ast.BinaryExpr, lanes int) {
	// Check RHS
	if call, ok := binExpr.Y.(*ast.CallExpr); ok {
		if isNumLanesCall(call) {
			binExpr.Y = &ast.BasicLit{
				Kind:  token.INT,
				Value: strconv.Itoa(lanes),
			}
		}
	}
	// Check LHS (less common but possible)
	if call, ok := binExpr.X.(*ast.CallExpr); ok {
		if isNumLanesCall(call) {
			binExpr.X = &ast.BasicLit{
				Kind:  token.INT,
				Value: strconv.Itoa(lanes),
			}
		}
	}
}

// isNumLanesCall checks if a call expression is v.NumLanes() or v.NumElements().
func isNumLanesCall(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	return sel.Sel.Name == "NumLanes" || sel.Sel.Name == "NumElements"
}

// isReduceSumCall checks if a call expression is v.ReduceSum() or hwy.ReduceSum(v).
func isReduceSumCall(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	if sel.Sel.Name == "ReduceSum" {
		return true
	}
	return false
}

// createReduceSumExpr creates an expression that stores the vector and sums elements.
// For now, we generate a function call that we'll define in a helper.
// Actually, archsimd vectors don't have a built-in ReduceSum, so we need to
// generate inline code that stores to temp and sums.
// Since we can't inject statements here, we'll generate a compound expression.
func createReduceSumExpr(call *ast.CallExpr, lanes int, vecTypeName, elemType string) ast.Expr {
	// Get the vector argument
	var vecExpr ast.Expr
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		if _, ok := sel.X.(*ast.Ident); ok {
			// It's v.ReduceSum() - the receiver is the vector
			vecExpr = sel.X
		}
	}
	if vecExpr == nil && len(call.Args) > 0 {
		// It's hwy.ReduceSum(v) - first arg is the vector
		vecExpr = call.Args[0]
	}
	if vecExpr == nil {
		return call // Can't transform, leave as-is
	}

	// Generate a function call to a helper we'll need to add
	// For now, generate inline reduction: func() T { var t [N]T; v.StoreSlice(t[:]); return t[0]+t[1]+... }()
	// This is verbose but works without injecting statements

	// Build: t[0] + t[1] + ... + t[lanes-1]
	var sumExpr ast.Expr
	for i := 0; i < lanes; i++ {
		indexExpr := &ast.IndexExpr{
			X: ast.NewIdent("_simd_temp"),
			Index: &ast.BasicLit{
				Kind:  token.INT,
				Value: strconv.Itoa(i),
			},
		}
		if sumExpr == nil {
			sumExpr = indexExpr
		} else {
			sumExpr = &ast.BinaryExpr{
				X:  sumExpr,
				Op: token.ADD,
				Y:  indexExpr,
			}
		}
	}

	// Build the function literal:
	// func() elemType {
	//     var _simd_temp [lanes]elemType
	//     vec.StoreSlice(_simd_temp[:])
	//     return t[0] + t[1] + ...
	// }()
	funcLit := &ast.FuncLit{
		Type: &ast.FuncType{
			Results: &ast.FieldList{
				List: []*ast.Field{
					{Type: ast.NewIdent(elemType)},
				},
			},
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				// var _simd_temp [lanes]elemType
				&ast.DeclStmt{
					Decl: &ast.GenDecl{
						Tok: token.VAR,
						Specs: []ast.Spec{
							&ast.ValueSpec{
								Names: []*ast.Ident{ast.NewIdent("_simd_temp")},
								Type: &ast.ArrayType{
									Len: &ast.BasicLit{
										Kind:  token.INT,
										Value: strconv.Itoa(lanes),
									},
									Elt: ast.NewIdent(elemType),
								},
							},
						},
					},
				},
				// vec.StoreSlice(_simd_temp[:])
				&ast.ExprStmt{
					X: &ast.CallExpr{
						Fun: &ast.SelectorExpr{
							X:   vecExpr,
							Sel: ast.NewIdent("StoreSlice"),
						},
						Args: []ast.Expr{
							&ast.SliceExpr{
								X: ast.NewIdent("_simd_temp"),
							},
						},
					},
				},
				// return sum expression
				&ast.ReturnStmt{
					Results: []ast.Expr{sumExpr},
				},
			},
		},
	}

	// Call the function literal immediately
	return &ast.CallExpr{
		Fun: funcLit,
	}
}

// filterConditionalBlocks filters statements based on //hwy:if, //hwy:else, //hwy:endif directives.
// It returns a new BlockStmt with only the statements that match the current target and element type.
// The original AST is not modified.
func filterConditionalBlocks(body *ast.BlockStmt, blocks []ConditionalBlock, fset *token.FileSet, targetName, elemType string) *ast.BlockStmt {
	if body == nil || len(blocks) == 0 {
		return body
	}

	// Create a new block with filtered statements
	newBody := &ast.BlockStmt{
		Lbrace: body.Lbrace,
		Rbrace: body.Rbrace,
	}

	for _, stmt := range body.List {
		// Get the line number of this statement
		stmtLine := fset.Position(stmt.Pos()).Line

		// Check if this statement is within any conditional block
		included := true
		for _, block := range blocks {
			if stmtLine > block.StartLine && stmtLine < block.EndLine {
				// Statement is within this conditional block
				conditionMatches := block.ParsedCondition.Evaluate(targetName, elemType)

				if block.ElseLine > 0 {
					// Block has an else clause
					if stmtLine < block.ElseLine {
						// Statement is in the "if" part
						included = conditionMatches
					} else {
						// Statement is in the "else" part
						included = !conditionMatches
					}
				} else {
					// No else clause - include only if condition matches
					included = conditionMatches
				}
				break // Found the innermost containing block
			}
		}

		if included {
			// Recursively filter nested blocks (e.g., for statements, if statements)
			filteredStmt := filterNestedConditionalBlocks(stmt, blocks, fset, targetName, elemType)
			newBody.List = append(newBody.List, filteredStmt)
		}
	}

	return newBody
}

// filterNestedConditionalBlocks recursively filters conditional blocks within nested statements.
func filterNestedConditionalBlocks(stmt ast.Stmt, blocks []ConditionalBlock, fset *token.FileSet, targetName, elemType string) ast.Stmt {
	switch s := stmt.(type) {
	case *ast.BlockStmt:
		return filterConditionalBlocks(s, blocks, fset, targetName, elemType)
	case *ast.IfStmt:
		newIf := *s // shallow copy
		if s.Body != nil {
			newIf.Body = filterConditionalBlocks(s.Body, blocks, fset, targetName, elemType)
		}
		if s.Else != nil {
			newIf.Else = filterNestedConditionalBlocks(s.Else, blocks, fset, targetName, elemType)
		}
		return &newIf
	case *ast.ForStmt:
		newFor := *s // shallow copy
		if s.Body != nil {
			newFor.Body = filterConditionalBlocks(s.Body, blocks, fset, targetName, elemType)
		}
		return &newFor
	case *ast.RangeStmt:
		newRange := *s // shallow copy
		if s.Body != nil {
			newRange.Body = filterConditionalBlocks(s.Body, blocks, fset, targetName, elemType)
		}
		return &newRange
	case *ast.SwitchStmt:
		newSwitch := *s // shallow copy
		if s.Body != nil {
			newSwitch.Body = filterConditionalBlocks(s.Body, blocks, fset, targetName, elemType)
		}
		return &newSwitch
	case *ast.TypeSwitchStmt:
		newSwitch := *s // shallow copy
		if s.Body != nil {
			newSwitch.Body = filterConditionalBlocks(s.Body, blocks, fset, targetName, elemType)
		}
		return &newSwitch
	case *ast.SelectStmt:
		newSelect := *s // shallow copy
		if s.Body != nil {
			newSelect.Body = filterConditionalBlocks(s.Body, blocks, fset, targetName, elemType)
		}
		return &newSelect
	default:
		return stmt
	}
}

// resolveTypeSpecificConst resolves type-specific constant references.
// It supports two patterns:
//
// Pattern 1 (base name): "expC0" -> "expC0_f32" or "expC0_f64"
//   - Looks up base name in typeSpecificConsts map
//   - Resolves to variant matching target element type
//
// Pattern 2 (suffix swap): "expC0_f32" -> "expC0_f64"
//   - Detects existing type suffix in the name
//   - Swaps to suffix matching target element type
//   - This allows base files to be compilable while hwygen adjusts for other types
func resolveTypeSpecificConst(name string, ctx *transformContext) string {
	targetSuffix := GetTypeSuffix(ctx.elemType)

	// Pattern 1: Check if this is a base name with type-specific variants
	if ctx.typeSpecificConsts != nil {
		if tsc, ok := ctx.typeSpecificConsts[name]; ok {
			if resolved, exists := tsc.Variants[targetSuffix]; exists {
				return resolved
			}
			// Fallback: if no exact match, try f32 for Float16 (compute type)
			if targetSuffix == "f16" {
				if resolved, exists := tsc.Variants["f32"]; exists {
					return resolved
				}
			}
		}
	}

	// Pattern 2: Check if name already has a type suffix that needs swapping
	for _, suffix := range typeSuffixes {
		if strings.HasSuffix(name, suffix) {
			// Extract base name and swap suffix
			baseName := strings.TrimSuffix(name, suffix)
			newSuffix := "_" + targetSuffix

			// Only swap if target suffix is different
			if suffix != newSuffix {
				return baseName + newSuffix
			}
			return name // Same suffix, no change needed
		}
	}

	return name
}

// transformIdentifiers walks the AST and resolves type-specific constant references
// and type parameter substitutions.
// This handles both Pattern 1 (base name lookup) and Pattern 2 (suffix swapping).
func transformIdentifiers(node ast.Node, ctx *transformContext) {
	if node == nil {
		return
	}

	ast.Inspect(node, func(n ast.Node) bool {
		switch expr := n.(type) {
		case *ast.Ident:
			// First check if it's a type parameter that should be replaced
			for _, tp := range ctx.typeParams {
				if expr.Name == tp.Name {
					expr.Name = ctx.elemType
					return true
				}
			}
			// Otherwise check if it's a constant reference
			resolved := resolveTypeSpecificConst(expr.Name, ctx)
			if resolved != expr.Name {
				expr.Name = resolved
			}
		case *ast.SelectorExpr:
			// Rename math.X to stdmath.X to avoid package name conflict
			// since generated files are in the math package but need stdlib math.
			// Only rename if "math" actually refers to the stdlib "math" import,
			// not a local variable or a different package aliased as "math".
			if ident, ok := expr.X.(*ast.Ident); ok && ident.Name == "math" {
				if importPath, isImport := ctx.imports[ident.Name]; isImport && importPath == "math" {
					ident.Name = "stdmath"
				}
			}
		}
		return true
	})
}

// convertStackArrayUsages converts stack array variable usages to slice expressions.
// For example, if buf is a stack array, convert:
//   - copy(buf, ...) -> copy(buf[:], ...)
//   - archsimd.LoadFloat32x8Slice(buf) -> archsimd.LoadFloat32x8Slice(buf[:])
//   - v.StoreSlice(buf) -> v.StoreSlice(buf[:])
func convertStackArrayUsages(node ast.Node, ctx *transformContext) {
	if node == nil {
		return
	}

	ast.Inspect(node, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}

		// Check each argument
		for i, arg := range call.Args {
			// Skip if it's already a slice expression
			if _, ok := arg.(*ast.SliceExpr); ok {
				continue
			}

			// Check if the argument is a stack array variable
			if ident, ok := arg.(*ast.Ident); ok {
				if ctx.stackArrayVars[ident.Name] {
					// Replace buf with buf[:]
					call.Args[i] = &ast.SliceExpr{
						X: ident,
					}
				}
			}
		}

		return true
	})
}

// transformFuncRefArgs transforms function references passed as arguments.
// For example: BaseApply(in, out, math.BaseExpVec)
// The math.BaseExpVec should become math.BaseExpVec_avx2 for SIMD targets,
// or math.BaseExpVec_fallback for fallback targets.
func transformFuncRefArgs(call *ast.CallExpr, ctx *transformContext) {
	for i, arg := range call.Args {
		// Handle package.BaseFuncName (SelectorExpr)
		if sel, ok := arg.(*ast.SelectorExpr); ok {
			if ident, ok := sel.X.(*ast.Ident); ok {
				// Check if it's a contrib package with a Base* function
				switch ident.Name {
				case "math", "dot", "matvec", "algo":
					if strings.HasPrefix(sel.Sel.Name, "Base") {
						// Transform math.BaseExpVec to math.BaseExpVec_avx2
						suffix := ctx.target.Suffix()
						if ctx.elemType == "float64" {
							suffix = suffix + "_Float64"
						}
						sel.Sel.Name = sel.Sel.Name + suffix
					}
				}
			}
		}

		// Handle package.BaseFuncName[T] (IndexExpr wrapping SelectorExpr)
		if indexExpr, ok := arg.(*ast.IndexExpr); ok {
			if sel, ok := indexExpr.X.(*ast.SelectorExpr); ok {
				if ident, ok := sel.X.(*ast.Ident); ok {
					switch ident.Name {
					case "math", "dot", "matvec", "algo":
						if strings.HasPrefix(sel.Sel.Name, "Base") {
							// Transform to non-generic version with suffix
							suffix := ctx.target.Suffix()
							if ctx.elemType == "float64" {
								suffix = suffix + "_Float64"
							}
							// Replace the IndexExpr with just the SelectorExpr (strip type param)
							sel.Sel.Name = sel.Sel.Name + suffix
							call.Args[i] = sel
						}
					}
				}
			}
		}

		// Handle local BaseFuncName[T] (IndexExpr wrapping Ident)
		if indexExpr, ok := arg.(*ast.IndexExpr); ok {
			if ident, ok := indexExpr.X.(*ast.Ident); ok {
				if strings.HasPrefix(ident.Name, "Base") {
					// Transform BaseFunc[T] to BaseFunc_avx2
					suffix := ctx.target.Suffix()
					if ctx.elemType == "float64" {
						suffix = suffix + "_Float64"
					}
					// Replace the IndexExpr with just the Ident
					call.Args[i] = ast.NewIdent(ident.Name + suffix)
				}
			}
		}
	}
}
