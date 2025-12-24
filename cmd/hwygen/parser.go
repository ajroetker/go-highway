package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
)

// ParsedFunc represents a function that has been parsed from the input file.
type ParsedFunc struct {
	Name       string          // Function name
	TypeParams []TypeParam     // Generic type parameters
	Params     []Param         // Function parameters
	Returns    []Param         // Return values
	Body       *ast.BlockStmt  // Function body
	HwyCalls   []HwyCall       // Detected hwy.* and contrib.* calls
	LoopInfo   *LoopInfo       // Main processing loop info
	Doc        *ast.CommentGroup // Function documentation
}

// TypeParam represents a generic type parameter.
type TypeParam struct {
	Name       string // T
	Constraint string // hwy.Floats
}

// Param represents a function parameter or return value.
type Param struct {
	Name string // parameter name (may be empty for returns)
	Type string // type expression as string
}

// HwyCall represents a call to hwy.* or contrib.* package.
type HwyCall struct {
	Package  string    // "hwy" or "contrib"
	FuncName string    // "Load", "Add", "Exp", etc.
	Position token.Pos // Source position
}

// LoopInfo represents information about the main vectorized loop.
type LoopInfo struct {
	Iterator string // "ii", "i", etc.
	Start    string // "0"
	End      string // "size", "len(data)"
	Stride   string // "vOne.NumElements()", "lanes", etc.
}

// Parse parses a Go source file and extracts functions with hwy operations.
func Parse(filename string) ([]ParsedFunc, string, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
	if err != nil {
		return nil, "", fmt.Errorf("parse file: %w", err)
	}

	var funcs []ParsedFunc
	packageName := file.Name.Name

	for _, decl := range file.Decls {
		funcDecl, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}

		// Only process functions (not methods) that start with "Base"
		if funcDecl.Recv != nil || !strings.HasPrefix(funcDecl.Name.Name, "Base") {
			continue
		}

		pf := ParsedFunc{
			Name: funcDecl.Name.Name,
			Body: funcDecl.Body,
			Doc:  funcDecl.Doc,
		}

		// Extract type parameters
		if funcDecl.Type.TypeParams != nil {
			for _, field := range funcDecl.Type.TypeParams.List {
				for _, name := range field.Names {
					constraint := exprToString(field.Type)
					pf.TypeParams = append(pf.TypeParams, TypeParam{
						Name:       name.Name,
						Constraint: constraint,
					})
				}
			}
		}

		// Extract parameters
		if funcDecl.Type.Params != nil {
			for _, field := range funcDecl.Type.Params.List {
				typeStr := exprToString(field.Type)
				if len(field.Names) == 0 {
					// Unnamed parameter
					pf.Params = append(pf.Params, Param{Name: "", Type: typeStr})
				} else {
					for _, name := range field.Names {
						pf.Params = append(pf.Params, Param{Name: name.Name, Type: typeStr})
					}
				}
			}
		}

		// Extract return values
		if funcDecl.Type.Results != nil {
			for _, field := range funcDecl.Type.Results.List {
				typeStr := exprToString(field.Type)
				if len(field.Names) == 0 {
					pf.Returns = append(pf.Returns, Param{Name: "", Type: typeStr})
				} else {
					for _, name := range field.Names {
						pf.Returns = append(pf.Returns, Param{Name: name.Name, Type: typeStr})
					}
				}
			}
		}

		// Find hwy.* and contrib.* calls
		pf.HwyCalls = findHwyCalls(funcDecl.Body)

		// Detect main vectorized loop
		pf.LoopInfo = detectLoop(funcDecl.Body)

		// Only include functions that use hwy operations
		if len(pf.HwyCalls) > 0 {
			funcs = append(funcs, pf)
		}
	}

	return funcs, packageName, nil
}

// findHwyCalls walks the AST and finds all hwy.* and contrib.* calls.
func findHwyCalls(node ast.Node) []HwyCall {
	var calls []HwyCall

	ast.Inspect(node, func(n ast.Node) bool {
		callExpr, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}

		// Check for hwy.Function or contrib.Function
		selExpr, ok := callExpr.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}

		ident, ok := selExpr.X.(*ast.Ident)
		if !ok {
			return true
		}

		if ident.Name == "hwy" || ident.Name == "contrib" {
			calls = append(calls, HwyCall{
				Package:  ident.Name,
				FuncName: selExpr.Sel.Name,
				Position: callExpr.Pos(),
			})
		}

		return true
	})

	return calls
}

// detectLoop attempts to find the main vectorized loop pattern.
// Looks for: for ii := 0; ii < size; ii += stride
func detectLoop(body *ast.BlockStmt) *LoopInfo {
	if body == nil {
		return nil
	}

	for _, stmt := range body.List {
		forStmt, ok := stmt.(*ast.ForStmt)
		if !ok {
			continue
		}

		info := &LoopInfo{}

		// Parse init: ii := 0
		if forStmt.Init != nil {
			if assignStmt, ok := forStmt.Init.(*ast.AssignStmt); ok {
				if len(assignStmt.Lhs) == 1 && len(assignStmt.Rhs) == 1 {
					if ident, ok := assignStmt.Lhs[0].(*ast.Ident); ok {
						info.Iterator = ident.Name
					}
					info.Start = exprToString(assignStmt.Rhs[0])
				}
			}
		}

		// Parse condition: ii < size
		if forStmt.Cond != nil {
			if binExpr, ok := forStmt.Cond.(*ast.BinaryExpr); ok {
				if binExpr.Op == token.LSS {
					info.End = exprToString(binExpr.Y)
				}
			}
		}

		// Parse post: ii += stride
		if forStmt.Post != nil {
			if assignStmt, ok := forStmt.Post.(*ast.AssignStmt); ok {
				if len(assignStmt.Rhs) == 1 {
					info.Stride = exprToString(assignStmt.Rhs[0])
				}
			}
		}

		// If we found a valid loop, return it
		if info.Iterator != "" && info.End != "" {
			return info
		}
	}

	return nil
}

// exprToString converts an AST expression to a string representation.
func exprToString(expr ast.Expr) string {
	if expr == nil {
		return ""
	}

	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.SelectorExpr:
		return exprToString(e.X) + "." + e.Sel.Name
	case *ast.StarExpr:
		return "*" + exprToString(e.X)
	case *ast.ArrayType:
		if e.Len == nil {
			return "[]" + exprToString(e.Elt)
		}
		return "[" + exprToString(e.Len) + "]" + exprToString(e.Elt)
	case *ast.IndexExpr:
		return exprToString(e.X) + "[" + exprToString(e.Index) + "]"
	case *ast.IndexListExpr:
		// Generic type instantiation T[U, V]
		args := make([]string, len(e.Indices))
		for i, idx := range e.Indices {
			args[i] = exprToString(idx)
		}
		return exprToString(e.X) + "[" + strings.Join(args, ", ") + "]"
	case *ast.CallExpr:
		return exprToString(e.Fun) + "(...)"
	case *ast.BasicLit:
		return e.Value
	case *ast.BinaryExpr:
		return exprToString(e.X) + " " + e.Op.String() + " " + exprToString(e.Y)
	case *ast.ParenExpr:
		return "(" + exprToString(e.X) + ")"
	case *ast.InterfaceType:
		// For constraints like "hwy.Floats | hwy.Integers"
		if len(e.Methods.List) == 0 {
			return "interface{}"
		}
		// Just return a simplified version
		return "constraint"
	default:
		return fmt.Sprintf("%T", expr)
	}
}

// GetConcreteTypes returns the concrete types that should be generated
// from a generic type parameter constraint.
func GetConcreteTypes(constraint string) []string {
	// Map constraint names to concrete types
	switch {
	case strings.Contains(constraint, "Floats"):
		return []string{"float32", "float64"}
	case strings.Contains(constraint, "SignedInts"):
		return []string{"int32", "int64"}
	case strings.Contains(constraint, "UnsignedInts"):
		return []string{"uint32", "uint64"}
	case strings.Contains(constraint, "Integers"):
		return []string{"int32", "int64", "uint32", "uint64"}
	case strings.Contains(constraint, "Lanes"):
		return []string{"float32", "float64", "int32", "int64"}
	default:
		// Unknown constraint, default to common types
		return []string{"float32", "float64"}
	}
}
