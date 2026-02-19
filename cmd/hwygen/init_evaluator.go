package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// evaluateInitGlobals fills in values for uninitialized PackageGlobals by
// evaluating init() functions in Go source files. This handles the common
// pattern of lookup tables populated by init():
//
//	var table [256]uint8
//	func init() {
//	    for ctrl := range 256 {
//	        table[ctrl] = uint8(expr(ctrl))
//	    }
//	}
func evaluateInitGlobals(filename string, globals []PackageGlobal) []PackageGlobal {
	// Find globals that need evaluation (no values)
	needsEval := make(map[string]int) // name → index in globals
	for i := range globals {
		if len(globals[i].Values) == 0 {
			needsEval[globals[i].Name] = i
		}
	}
	if len(needsEval) == 0 {
		return globals
	}

	// Build struct type map from globals that have struct definitions
	structTypes := make(map[string]*PackageStruct)
	for i := range globals {
		if globals[i].IsStruct && globals[i].StructDef != nil {
			structTypes[globals[i].StructDef.Name] = globals[i].StructDef
		}
	}

	// Parse all Go files in the directory to find init() functions
	dir := filepath.Dir(filename)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return globals
	}

	fset := token.NewFileSet()
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") {
			continue
		}
		if strings.HasSuffix(name, "_test.go") || strings.HasSuffix(name, ".gen.go") {
			continue
		}

		path := filepath.Join(dir, name)
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			continue
		}

		for _, decl := range file.Decls {
			funcDecl, ok := decl.(*ast.FuncDecl)
			if !ok || funcDecl.Name.Name != "init" || funcDecl.Body == nil {
				continue
			}
			evaluateInitBody(funcDecl.Body, globals, needsEval, structTypes)
		}
	}

	return globals
}

// evaluateInitBody evaluates an init() function body to compute values for
// uninitialized globals.
func evaluateInitBody(body *ast.BlockStmt, globals []PackageGlobal, needsEval map[string]int, structTypes map[string]*PackageStruct) {
	for _, stmt := range body.List {
		forStmt, ok := stmt.(*ast.RangeStmt)
		if !ok {
			continue
		}
		evaluateRangeLoop(forStmt, globals, needsEval, structTypes)
	}
}

// evaluateRangeLoop evaluates a for-range loop that populates a global array.
func evaluateRangeLoop(forStmt *ast.RangeStmt, globals []PackageGlobal, needsEval map[string]int, structTypes map[string]*PackageStruct) {
	// Get loop variable name and range value
	var loopVar string
	if forStmt.Key != nil {
		if ident, ok := forStmt.Key.(*ast.Ident); ok {
			loopVar = ident.Name
		}
	}

	// Get range upper bound (for i := range N)
	rangeMax := evalConstInt(forStmt.X)
	if rangeMax <= 0 {
		return
	}

	// Evaluate the loop body for each iteration
	for i := range rangeMax {
		env := &evalEnv{
			vars:        map[string]int64{loopVar: int64(i)},
			structTypes: structTypes,
		}
		evaluateBlock(forStmt.Body, env, globals, needsEval)
	}
}

// evalEnv holds variable values during evaluation.
type evalEnv struct {
	vars        map[string]int64
	structVars  map[string]*PackageStruct // varname → struct def (for struct-typed local vars)
	structTypes map[string]*PackageStruct // struct type name → def (for recognizing declarations)
}

// structFieldKey returns the env var key for a struct field.
// For scalar fields: "lookup__numValues"
// For array field elements: "lookup__valueEnds__arr_0"
func structFieldKey(varName, fieldName string) string {
	return varName + "__" + fieldName
}

func structFieldArrKey(varName, fieldName string, idx int) string {
	return fmt.Sprintf("%s__%s__arr_%d", varName, fieldName, idx)
}

// initStructVar initializes all fields of a struct variable to zero in the env.
func (env *evalEnv) initStructVar(varName string, sd *PackageStruct) {
	if env.structVars == nil {
		env.structVars = make(map[string]*PackageStruct)
	}
	env.structVars[varName] = sd
	for _, f := range sd.Fields {
		if f.IsArray {
			for i := range f.ArraySize {
				env.vars[structFieldArrKey(varName, f.Name, i)] = 0
			}
		} else {
			env.vars[structFieldKey(varName, f.Name)] = 0
		}
	}
}

// copyStructToGlobal copies all fields from a struct variable to a struct array global.
func copyStructToGlobal(env *evalEnv, varName string, sd *PackageStruct, pg *PackageGlobal, outerIdx int) {
	flatSize := sd.FlatSize()
	// Ensure global values are allocated
	totalSize := pg.Size * flatSize
	if len(pg.Values) == 0 {
		pg.Values = make([]string, totalSize)
		for i := range pg.Values {
			pg.Values[i] = "0"
		}
	}

	base := outerIdx * flatSize
	offset := 0
	for _, f := range sd.Fields {
		if f.IsArray {
			for i := range f.ArraySize {
				val := env.vars[structFieldArrKey(varName, f.Name, i)]
				idx := base + offset
				if idx >= 0 && idx < len(pg.Values) {
					pg.Values[idx] = strconv.FormatInt(val, 10)
				}
				offset++
			}
		} else {
			val := env.vars[structFieldKey(varName, f.Name)]
			idx := base + offset
			if idx >= 0 && idx < len(pg.Values) {
				pg.Values[idx] = strconv.FormatInt(val, 10)
			}
			offset++
		}
	}
}

// evaluateBlock evaluates a block of statements in the given environment.
func evaluateBlock(block *ast.BlockStmt, env *evalEnv, globals []PackageGlobal, needsEval map[string]int) {
	for _, stmt := range block.List {
		evaluateStmt(stmt, env, globals, needsEval)
	}
}

// evaluateStmt evaluates a single statement.
func evaluateStmt(stmt ast.Stmt, env *evalEnv, globals []PackageGlobal, needsEval map[string]int) {
	switch s := stmt.(type) {
	case *ast.AssignStmt:
		evaluateAssign(s, env, globals, needsEval)
	case *ast.DeclStmt:
		evaluateDeclStmt(s, env)
	case *ast.ForStmt:
		evaluateForStmt(s, env, globals, needsEval)
	case *ast.RangeStmt:
		evaluateNestedRange(s, env, globals, needsEval)
	case *ast.IfStmt:
		evaluateIfStmt(s, env, globals, needsEval)
	case *ast.IncDecStmt:
		evaluateIncDec(s, env)
	}
}

// evaluateAssign handles assignment statements.
func evaluateAssign(s *ast.AssignStmt, env *evalEnv, globals []PackageGlobal, needsEval map[string]int) {
	for i, lhs := range s.Lhs {
		if i >= len(s.Rhs) {
			break
		}

		// Check for struct field assignment: lookup.field = val
		if selExpr, ok := lhs.(*ast.SelectorExpr); ok {
			if ident, ok := selExpr.X.(*ast.Ident); ok {
				if _, isStruct := env.structVars[ident.Name]; isStruct {
					rhs := evalExpr(s.Rhs[i], env)
					key := structFieldKey(ident.Name, selExpr.Sel.Name)
					env.vars[key] = rhs
					continue
				}
			}
		}

		// Check for struct array field assignment: lookup.field[idx] = val
		if idxExpr, ok := lhs.(*ast.IndexExpr); ok {
			if selExpr, ok := idxExpr.X.(*ast.SelectorExpr); ok {
				if ident, ok := selExpr.X.(*ast.Ident); ok {
					if _, isStruct := env.structVars[ident.Name]; isStruct {
						rhs := evalExpr(s.Rhs[i], env)
						idx := int(evalExpr(idxExpr.Index, env))
						key := structFieldArrKey(ident.Name, selExpr.Sel.Name, idx)
						env.vars[key] = rhs
						continue
					}
				}
			}
		}

		// Check for global array assignment: table[idx] = val  or  table2D[idx] = localArr
		if idxExpr, ok := lhs.(*ast.IndexExpr); ok {
			// Case A: Direct global index — table[idx] or table2D[idx]
			if ident, ok := idxExpr.X.(*ast.Ident); ok {
				if gIdx, found := needsEval[ident.Name]; found {
					pg := &globals[gIdx]
					idx := int(evalExpr(idxExpr.Index, env))

					if pg.IsStruct {
						// Struct array global: whole-struct assignment
						if rhsIdent, ok := s.Rhs[i].(*ast.Ident); ok {
							if sd, isStruct := env.structVars[rhsIdent.Name]; isStruct {
								copyStructToGlobal(env, rhsIdent.Name, sd, pg, idx)
							}
						}
					} else if pg.InnerSize > 0 {
						// 2D global: whole-row assignment table2D[idx] = localArr
						if rhsIdent, ok := s.Rhs[i].(*ast.Ident); ok {
							for j := range pg.InnerSize {
								key := fmt.Sprintf("%s__arr_%d", rhsIdent.Name, j)
								val := env.vars[key]
								setGlobalValue(pg, idx, j, val)
							}
						}
					} else {
						// 1D global: scalar assignment table[idx] = val
						rhs := evalExpr(s.Rhs[i], env)
						setGlobalValue(pg, idx, -1, rhs)
					}
					continue
				}
			}

			// Case B: 2D element access — table[outer][inner] = val
			if innerIdx, ok := idxExpr.X.(*ast.IndexExpr); ok {
				if ident, ok := innerIdx.X.(*ast.Ident); ok {
					if gIdx, found := needsEval[ident.Name]; found {
						outerIdx := int(evalExpr(innerIdx.Index, env))
						innerIdxVal := int(evalExpr(idxExpr.Index, env))
						rhs := evalExpr(s.Rhs[i], env)
						pg := &globals[gIdx]
						setGlobalValue(pg, outerIdx, innerIdxVal, rhs)
						continue
					}
				}
			}
		}

		rhs := evalExpr(s.Rhs[i], env)

		// Local variable assignment
		if ident, ok := lhs.(*ast.Ident); ok {
			env.vars[ident.Name] = rhs
		}
		// Local array element: localArr[idx] = val
		if idxExpr, ok := lhs.(*ast.IndexExpr); ok {
			if ident, ok := idxExpr.X.(*ast.Ident); ok {
				idx := int(evalExpr(idxExpr.Index, env))
				key := fmt.Sprintf("%s__arr_%d", ident.Name, idx)
				env.vars[key] = rhs
			}
		}
	}
}

// evaluateDeclStmt handles var declarations in init() bodies.
func evaluateDeclStmt(s *ast.DeclStmt, env *evalEnv) {
	genDecl, ok := s.Decl.(*ast.GenDecl)
	if !ok {
		return
	}
	for _, spec := range genDecl.Specs {
		vs, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}
		for i, name := range vs.Names {
			// Check for struct type declaration: var lookup maskedVByte12Lookup
			if vs.Type != nil && len(vs.Values) == 0 {
				if typeIdent, ok := vs.Type.(*ast.Ident); ok {
					if sd, found := env.structTypes[typeIdent.Name]; found {
						env.initStructVar(name.Name, sd)
						continue
					}
				}
				// Check for fixed-size array type: var mask [16]uint8
				if arrType, ok := vs.Type.(*ast.ArrayType); ok && arrType.Len != nil {
					if lenLit, ok := arrType.Len.(*ast.BasicLit); ok && lenLit.Kind == token.INT {
						arrSize, _ := strconv.Atoi(lenLit.Value)
						for j := range arrSize {
							env.vars[fmt.Sprintf("%s__arr_%d", name.Name, j)] = 0
						}
					}
				}
			}
			if i < len(vs.Values) {
				env.vars[name.Name] = evalExpr(vs.Values[i], env)
			} else {
				env.vars[name.Name] = 0
			}
		}
	}
}

// evaluateForStmt handles C-style for loops.
func evaluateForStmt(s *ast.ForStmt, env *evalEnv, globals []PackageGlobal, needsEval map[string]int) {
	// Init
	if s.Init != nil {
		evaluateStmt(s.Init, env, globals, needsEval)
	}

	for range 100000 {
		// Check condition
		if s.Cond != nil {
			cond := evalExpr(s.Cond, env)
			if cond == 0 {
				break
			}
		}
		// Body
		evaluateBlock(s.Body, env, globals, needsEval)
		// Post
		if s.Post != nil {
			evaluateStmt(s.Post, env, globals, needsEval)
		}
	}
}

// evaluateNestedRange handles nested range loops.
func evaluateNestedRange(s *ast.RangeStmt, env *evalEnv, globals []PackageGlobal, needsEval map[string]int) {
	var loopVar string
	if s.Key != nil {
		if ident, ok := s.Key.(*ast.Ident); ok {
			loopVar = ident.Name
		}
	}
	rangeMax := evalExprInt(s.X, env)
	if rangeMax <= 0 {
		return
	}
	for i := range rangeMax {
		env.vars[loopVar] = int64(i)
		evaluateBlock(s.Body, env, globals, needsEval)
	}
}

// evaluateIfStmt handles if statements.
func evaluateIfStmt(s *ast.IfStmt, env *evalEnv, globals []PackageGlobal, needsEval map[string]int) {
	if s.Init != nil {
		evaluateStmt(s.Init, env, globals, needsEval)
	}
	cond := evalExpr(s.Cond, env)
	if cond != 0 {
		evaluateBlock(s.Body, env, globals, needsEval)
	} else if s.Else != nil {
		switch e := s.Else.(type) {
		case *ast.BlockStmt:
			evaluateBlock(e, env, globals, needsEval)
		case *ast.IfStmt:
			evaluateIfStmt(e, env, globals, needsEval)
		}
	}
}

// evaluateIncDec handles i++ and i--.
func evaluateIncDec(s *ast.IncDecStmt, env *evalEnv) {
	if ident, ok := s.X.(*ast.Ident); ok {
		if s.Tok == token.INC {
			env.vars[ident.Name]++
		} else {
			env.vars[ident.Name]--
		}
	}
}

// setGlobalValue sets a value in a PackageGlobal, auto-expanding the Values slice.
func setGlobalValue(pg *PackageGlobal, outerIdx, innerIdx int, val int64) {
	var flatIdx int
	if pg.InnerSize > 0 {
		if innerIdx < 0 {
			// Whole row — shouldn't happen with this code path
			return
		}
		flatIdx = outerIdx*pg.InnerSize + innerIdx
	} else {
		flatIdx = outerIdx
	}

	// Expand values if needed
	totalSize := pg.Size
	if pg.InnerSize > 0 {
		totalSize = pg.Size * pg.InnerSize
	}
	if len(pg.Values) == 0 {
		pg.Values = make([]string, totalSize)
		for i := range pg.Values {
			pg.Values[i] = "0"
		}
	}

	if flatIdx >= 0 && flatIdx < len(pg.Values) {
		pg.Values[flatIdx] = strconv.FormatInt(val, 10)
	}
}

// evalExpr evaluates an expression to an int64 value.
func evalExpr(expr ast.Expr, env *evalEnv) int64 {
	switch e := expr.(type) {
	case *ast.BasicLit:
		if e.Kind == token.INT {
			val, _ := strconv.ParseInt(e.Value, 0, 64)
			return val
		}
	case *ast.Ident:
		if val, ok := env.vars[e.Name]; ok {
			return val
		}
	case *ast.BinaryExpr:
		left := evalExpr(e.X, env)
		right := evalExpr(e.Y, env)
		return evalBinaryOp(e.Op, left, right)
	case *ast.ParenExpr:
		return evalExpr(e.X, env)
	case *ast.UnaryExpr:
		val := evalExpr(e.X, env)
		switch e.Op {
		case token.SUB:
			return -val
		case token.XOR:
			return ^val
		case token.NOT:
			if val == 0 {
				return 1
			}
			return 0
		}
	case *ast.CallExpr:
		// Type conversions: uint8(x), int(x), etc.
		if ident, ok := e.Fun.(*ast.Ident); ok {
			switch ident.Name {
			case "uint8", "byte":
				return evalExpr(e.Args[0], env) & 0xFF
			case "uint16":
				return evalExpr(e.Args[0], env) & 0xFFFF
			case "uint32":
				return evalExpr(e.Args[0], env) & 0xFFFFFFFF
			case "uint64":
				return evalExpr(e.Args[0], env)
			case "int", "int64", "int32", "int16", "int8":
				return evalExpr(e.Args[0], env)
			}
		}
	case *ast.SelectorExpr:
		// Struct field access: lookup.numValues
		if ident, ok := e.X.(*ast.Ident); ok {
			if _, isStruct := env.structVars[ident.Name]; isStruct {
				key := structFieldKey(ident.Name, e.Sel.Name)
				if val, ok := env.vars[key]; ok {
					return val
				}
			}
		}
	case *ast.IndexExpr:
		// Struct array field access: lookup.valueEnds[idx]
		if selExpr, ok := e.X.(*ast.SelectorExpr); ok {
			if ident, ok := selExpr.X.(*ast.Ident); ok {
				if _, isStruct := env.structVars[ident.Name]; isStruct {
					idx := int(evalExpr(e.Index, env))
					key := structFieldArrKey(ident.Name, selExpr.Sel.Name, idx)
					if val, ok := env.vars[key]; ok {
						return val
					}
				}
			}
		}
		// Local array access: arr[idx]
		if ident, ok := e.X.(*ast.Ident); ok {
			idx := int(evalExpr(e.Index, env))
			key := fmt.Sprintf("%s__arr_%d", ident.Name, idx)
			if val, ok := env.vars[key]; ok {
				return val
			}
			// Could be a regular variable
			if val, ok := env.vars[ident.Name]; ok {
				return val
			}
		}
	}
	return 0
}

// evalExprInt evaluates an expression, also handling constant expressions.
func evalExprInt(expr ast.Expr, env *evalEnv) int64 {
	if v := evalConstInt(expr); v > 0 {
		return int64(v)
	}
	return evalExpr(expr, env)
}

// evalBinaryOp evaluates a binary operation.
func evalBinaryOp(op token.Token, left, right int64) int64 {
	switch op {
	case token.ADD:
		return left + right
	case token.SUB:
		return left - right
	case token.MUL:
		return left * right
	case token.QUO:
		if right == 0 {
			return 0
		}
		return left / right
	case token.REM:
		if right == 0 {
			return 0
		}
		return left % right
	case token.AND:
		return left & right
	case token.OR:
		return left | right
	case token.XOR:
		return left ^ right
	case token.SHL:
		return left << uint(right)
	case token.SHR:
		return left >> uint(right)
	case token.AND_NOT:
		return left &^ right
	case token.LSS:
		if left < right {
			return 1
		}
		return 0
	case token.GTR:
		if left > right {
			return 1
		}
		return 0
	case token.LEQ:
		if left <= right {
			return 1
		}
		return 0
	case token.GEQ:
		if left >= right {
			return 1
		}
		return 0
	case token.EQL:
		if left == right {
			return 1
		}
		return 0
	case token.NEQ:
		if left != right {
			return 1
		}
		return 0
	case token.LAND:
		if left != 0 && right != 0 {
			return 1
		}
		return 0
	case token.LOR:
		if left != 0 || right != 0 {
			return 1
		}
		return 0
	}
	return 0
}

// evalConstInt evaluates a constant integer expression (e.g., range bound).
func evalConstInt(expr ast.Expr) int {
	switch e := expr.(type) {
	case *ast.BasicLit:
		if e.Kind == token.INT {
			val, _ := strconv.Atoi(e.Value)
			return val
		}
	case *ast.BinaryExpr:
		left := evalConstInt(e.X)
		right := evalConstInt(e.Y)
		return int(evalBinaryOp(e.Op, int64(left), int64(right)))
	}
	return 0
}
