package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testProfile(t *testing.T, elemType string) *CIntrinsicProfile {
	t.Helper()
	p := GetCProfile("NEON", elemType)
	if p == nil {
		t.Fatalf("no NEON profile for %s", elemType)
	}
	return p
}

// TestHelperReturnVec verifies that inferCallType returns a vector type for
// helper functions registered in helperReturnVec (e.g., BasePrefixSumVec).
func TestHelperReturnVec(t *testing.T) {
	profile := testProfile(t, "float32")
	translator := NewCASTTranslator(profile, "float32")
	translator.helperReturnVec["BasePrefixSumVec"] = true

	// Parse a call expression: BasePrefixSumVec(v)
	expr, err := parser.ParseExpr("BasePrefixSumVec(v)")
	if err != nil {
		t.Fatalf("parse expr: %v", err)
	}
	callExpr := expr.(*ast.CallExpr)

	info := translator.inferCallType(callExpr)
	if info.cType != "float32x4_t" {
		t.Errorf("inferCallType(BasePrefixSumVec(v)) = %q, want %q", info.cType, "float32x4_t")
	}
	if !info.isVector {
		t.Error("inferCallType(BasePrefixSumVec(v)).isVector = false, want true")
	}
}

// TestHelperReturnVec_Scalar verifies that non-Vec helpers still return scalar.
func TestHelperReturnVec_Scalar(t *testing.T) {
	profile := testProfile(t, "float32")
	translator := NewCASTTranslator(profile, "float32")
	// Don't register anything in helperReturnVec

	expr, err := parser.ParseExpr("someHelper(x)")
	if err != nil {
		t.Fatalf("parse expr: %v", err)
	}
	callExpr := expr.(*ast.CallExpr)

	info := translator.inferCallType(callExpr)
	// Should fall through to default scalar type
	if info.cType != "float" {
		t.Errorf("inferCallType(someHelper(x)) = %q, want %q", info.cType, "float")
	}
	if info.isVector {
		t.Error("inferCallType(someHelper(x)).isVector = true, want false")
	}
}

// TestSliceArgBaseName verifies extraction of the base identifier from various
// argument expression forms used in helper function calls.
func TestSliceArgBaseName(t *testing.T) {
	tests := []struct {
		name string
		expr string
		want string
	}{
		{"simple ident", "src", "src"},
		{"slice full", "src[:]", "src"},
		{"slice low", "src[i:]", "src"},
		{"slice high", "src[:n]", "src"},
		{"slice both", "src[i:n]", "src"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			expr, err := parser.ParseExpr(tt.expr)
			if err != nil {
				t.Fatalf("parse %q: %v", tt.expr, err)
			}
			got := sliceArgBaseName(expr)
			if got != tt.want {
				t.Errorf("sliceArgBaseName(%q) = %q, want %q", tt.expr, got, tt.want)
			}
		})
	}
}

// TestSliceArgBaseName_Unsupported verifies that unrecognized expressions return "".
func TestSliceArgBaseName_Unsupported(t *testing.T) {
	// A function call expression — not an ident or slice
	expr, err := parser.ParseExpr("foo()")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	got := sliceArgBaseName(expr)
	if got != "" {
		t.Errorf("sliceArgBaseName(foo()) = %q, want empty", got)
	}
}

// TestInferCallType_NumLanesWithTypeParam verifies that hwy.NumLanes[uint8]()
// returns "long" (scalar), not "uint8x16_t" (vector). The type parameter
// override must skip scalar-returning functions.
func TestInferCallType_NumLanesWithTypeParam(t *testing.T) {
	profile := testProfile(t, "float32")
	translator := NewCASTTranslator(profile, "float32")

	// Parse: hwy.NumLanes[uint8]()
	src := `package p; var _ = hwy.NumLanes[uint8]()`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	genDecl := file.Decls[0].(*ast.GenDecl)
	valueSpec := genDecl.Specs[0].(*ast.ValueSpec)
	callExpr := valueSpec.Values[0].(*ast.CallExpr)

	info := translator.inferCallType(callExpr)
	if info.cType != "long" {
		t.Errorf("inferCallType(hwy.NumLanes[uint8]()) = %q, want %q", info.cType, "long")
	}
	if info.isVector {
		t.Error("inferCallType(hwy.NumLanes[uint8]()).isVector = true, want false")
	}
}

// TestInferCallType_MaxLanesWithTypeParam verifies hwy.MaxLanes[uint8]() returns scalar.
func TestInferCallType_MaxLanesWithTypeParam(t *testing.T) {
	profile := testProfile(t, "float32")
	translator := NewCASTTranslator(profile, "float32")

	src := `package p; var _ = hwy.MaxLanes[uint8]()`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	genDecl := file.Decls[0].(*ast.GenDecl)
	valueSpec := genDecl.Specs[0].(*ast.ValueSpec)
	callExpr := valueSpec.Values[0].(*ast.CallExpr)

	info := translator.inferCallType(callExpr)
	if info.cType != "long" {
		t.Errorf("inferCallType(hwy.MaxLanes[uint8]()) = %q, want %q", info.cType, "long")
	}
}

// TestInferCallType_LoadSliceUint8 verifies that hwy.LoadSlice[uint8]()
// DOES return the uint8 vector type (the type param override should apply).
func TestInferCallType_LoadSliceUint8(t *testing.T) {
	profile := testProfile(t, "float32")
	translator := NewCASTTranslator(profile, "float32")

	src := `package p; var _ = hwy.LoadSlice[uint8](x)`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	genDecl := file.Decls[0].(*ast.GenDecl)
	valueSpec := genDecl.Specs[0].(*ast.ValueSpec)
	callExpr := valueSpec.Values[0].(*ast.CallExpr)

	info := translator.inferCallType(callExpr)
	if info.cType != "uint8x16_t" {
		t.Errorf("inferCallType(hwy.LoadSlice[uint8](x)) = %q, want %q", info.cType, "uint8x16_t")
	}
	if !info.isVector {
		t.Error("want isVector=true for LoadSlice[uint8]")
	}
}

// TestSlideUpExtFn_SignedProfiles verifies that signed integer NEON profiles
// have SlideUpExtFn set, so SlideUpLanes doesn't emit a fallback comment.
func TestSlideUpExtFn_SignedProfiles(t *testing.T) {
	tests := []struct {
		name    string
		elem    string
		wantFn  string
	}{
		{"int32", "int32", "vextq_s32"},
		{"int64", "int64", "vextq_s64"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			profile := testProfile(t, tt.elem)
			if profile.SlideUpExtFn == nil {
				t.Fatal("SlideUpExtFn is nil")
			}
			fn, ok := profile.SlideUpExtFn["q"]
			if !ok {
				t.Fatal("SlideUpExtFn missing 'q' tier")
			}
			if fn != tt.wantFn {
				t.Errorf("SlideUpExtFn[q] = %q, want %q", fn, tt.wantFn)
			}
		})
	}
}

// TestScanPackageFuncs verifies that scanPackageFuncs discovers functions
// from sibling *_base.go files in the same directory.
func TestScanPackageFuncs(t *testing.T) {
	dir := t.TempDir()

	// Main file — defines BaseNormalize
	mainFile := filepath.Join(dir, "normalize_base.go")
	err := os.WriteFile(mainFile, []byte(`package vec

import "github.com/ajroetker/go-highway/hwy"

func BaseNormalize(dst []float32) {
	_ = hwy.Load(dst)
}
`), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Sibling file — defines BaseDot
	err = os.WriteFile(filepath.Join(dir, "dot_base.go"), []byte(`package vec

import "github.com/ajroetker/go-highway/hwy"

func BaseDot[T hwy.Floats](a, b []T) T {
	v := hwy.Load(a)
	_ = v
	return 0
}
`), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Non-base file should not be scanned
	err = os.WriteFile(filepath.Join(dir, "helpers.go"), []byte(`package vec

func helperFunc() int { return 42 }
`), 0644)
	if err != nil {
		t.Fatal(err)
	}

	result, err := Parse(mainFile)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	if _, ok := result.AllFuncs["BaseDot"]; !ok {
		t.Error("AllFuncs missing BaseDot from sibling dot_base.go")
	}
	if _, ok := result.AllFuncs["BaseNormalize"]; !ok {
		t.Error("AllFuncs missing BaseNormalize from main file")
	}
	if _, ok := result.AllFuncs["helperFunc"]; ok {
		t.Error("AllFuncs should not contain helperFunc from non-base file helpers.go")
	}
}

// TestScanPackageFuncs_NoOverwrite verifies that scanPackageFuncs does not
// overwrite functions already parsed from the input file.
func TestScanPackageFuncs_NoOverwrite(t *testing.T) {
	dir := t.TempDir()

	mainFile := filepath.Join(dir, "a_base.go")
	err := os.WriteFile(mainFile, []byte(`package p

import "github.com/ajroetker/go-highway/hwy"

func SharedFunc(x []float32) {
	_ = hwy.Load(x)
}
`), 0644)
	if err != nil {
		t.Fatal(err)
	}

	err = os.WriteFile(filepath.Join(dir, "b_base.go"), []byte(`package p

func SharedFunc(x []float32) {
}
`), 0644)
	if err != nil {
		t.Fatal(err)
	}

	result, err := Parse(mainFile)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	pf := result.AllFuncs["SharedFunc"]
	if pf == nil {
		t.Fatal("SharedFunc not in AllFuncs")
	}
	if len(pf.HwyCalls) == 0 {
		t.Error("SharedFunc should have hwy calls (from main file), not be overwritten by sibling")
	}
}

// TestBuildParamMap_VecParam verifies that Vec[T] parameters get the correct
// vector type in helper mode.
func TestBuildParamMap_VecParam(t *testing.T) {
	profile := testProfile(t, "float32")
	translator := NewCASTTranslator(profile, "float32")
	translator.helperMode = true

	pf := &ParsedFunc{
		Name: "BasePrefixSumVec",
		Params: []Param{
			{Name: "v", Type: "hwy.Vec[T]"},
		},
		Returns: []Param{
			{Name: "", Type: "hwy.Vec[T]"},
		},
	}

	translator.buildParamMap(pf)

	info, ok := translator.params["v"]
	if !ok {
		t.Fatal("params missing 'v'")
	}
	if !info.isVector {
		t.Error("param 'v' should have isVector=true")
	}
	if info.cType != "float32x4_t" {
		t.Errorf("param 'v' cType = %q, want %q", info.cType, "float32x4_t")
	}
}

// TestEmitFuncSignature_VecReturn verifies that helper functions with Vec[T]
// return type get the correct C return type.
func TestEmitFuncSignature_VecReturn(t *testing.T) {
	profile := testProfile(t, "float32")
	translator := NewCASTTranslator(profile, "float32")
	translator.helperMode = true

	pf := &ParsedFunc{
		Name: "BasePrefixSumVec",
		Params: []Param{
			{Name: "v", Type: "hwy.Vec[T]"},
		},
		Returns: []Param{
			{Name: "", Type: "hwy.Vec[T]"},
		},
	}

	translator.emitFuncSignature(pf)

	output := translator.buf.String()
	if !strings.Contains(output, "float32x4_t BasePrefixSumVec(") {
		t.Errorf("expected Vec return type in signature, got: %s", output)
	}
}

// TestTryEvalConstInt verifies constant-folding of integer expressions.
func TestTryEvalConstInt(t *testing.T) {
	profile := testProfile(t, "float32")
	translator := NewCASTTranslator(profile, "float32")
	translator.constVars["lanes"] = 4

	tests := []struct {
		name string
		expr string
		want int
		ok   bool
	}{
		{"literal 4", "4", 4, true},
		{"literal 0", "0", 0, true},
		{"known var", "lanes", 4, true},
		{"lanes - 1", "lanes - 1", 3, true},
		{"lanes * 2", "lanes * 2", 8, true},
		{"lanes * 2 - 1", "lanes * 2 - 1", 7, true},
		{"parenthesized", "(lanes - 1)", 3, true},
		{"unknown var", "i", 0, false},
		{"mixed unknown", "lanes + i", 0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			expr, err := parser.ParseExpr(tt.expr)
			if err != nil {
				t.Fatalf("parse %q: %v", tt.expr, err)
			}
			got, ok := translator.tryEvalConstInt(expr)
			if ok != tt.ok {
				t.Errorf("tryEvalConstInt(%q) ok = %v, want %v", tt.expr, ok, tt.ok)
			}
			if ok && got != tt.want {
				t.Errorf("tryEvalConstInt(%q) = %d, want %d", tt.expr, got, tt.want)
			}
		})
	}
}

// TestGetLaneLanesMinusOne verifies that hwy.GetLane(v, lanes-1) emits a
// direct intrinsic call (vgetq_lane_f32) instead of the store-to-stack fallback.
func TestGetLaneLanesMinusOne(t *testing.T) {
	profile := testProfile(t, "float32")
	translator := NewCASTTranslator(profile, "float32")
	// Simulate: lanes := hwy.NumLanes[float32]() — record lanes=4
	translator.constVars["lanes"] = 4

	// Parse: last := hwy.GetLane(v, lanes-1)
	src := `package p; func f() { last := hwy.GetLane(v, lanes-1) ; _ = last }`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	funcDecl := file.Decls[0].(*ast.FuncDecl)
	assignStmt := funcDecl.Body.List[0].(*ast.AssignStmt)

	// Register "v" as a vector variable so translateExpr works
	translator.vars["v"] = cVarInfo{cType: "float32x4_t", isVector: true}

	translator.translateAssignStmt(assignStmt)
	output := translator.buf.String()

	if !strings.Contains(output, "vgetq_lane_f32(v, 3)") {
		t.Errorf("expected vgetq_lane_f32(v, 3), got: %s", output)
	}
	if strings.Contains(output, "_getlane_buf") {
		t.Errorf("should not use store-to-stack pattern, got: %s", output)
	}
}

// TestWidenedVarsClearedOnVarDecl verifies that a plain `var total T` declaration
// clears the widened status from a prior vector loop. This prevents the C translator
// from emitting `total_lo`/`total_hi` references in a scalar tail where only a
// single scalar `total` exists (regression from pairwise summation bf16 codegen).
func TestWidenedVarsClearedOnVarDecl(t *testing.T) {
	profile := testProfile(t, "hwy.BFloat16")
	translator := NewCASTTranslator(profile, "hwy.BFloat16")

	// Simulate the vector loop having registered "total" as widened
	translator.widenedVars["total"] = true

	// Parse: var total hwy.BFloat16
	src := `package p; func f() { var total hwy.BFloat16; _ = total }`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	funcDecl := file.Decls[0].(*ast.FuncDecl)
	declStmt := funcDecl.Body.List[0].(*ast.DeclStmt)

	translator.translateDeclStmt(declStmt)

	// After the var declaration, "total" should no longer be widened
	if _, widened := translator.widenedVars["total"]; widened {
		t.Error("var declaration should clear widened status, but total is still widened")
	}
}

// TestGetLaneVariableIndex verifies that GetLane with a truly variable index
// (not constant-foldable) still uses the store-to-stack fallback.
func TestGetLaneVariableIndex(t *testing.T) {
	profile := testProfile(t, "float32")
	translator := NewCASTTranslator(profile, "float32")

	// Parse: elem := hwy.GetLane(v, i)
	src := `package p; func f() { elem := hwy.GetLane(v, i) ; _ = elem }`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	funcDecl := file.Decls[0].(*ast.FuncDecl)
	assignStmt := funcDecl.Body.List[0].(*ast.AssignStmt)

	translator.vars["v"] = cVarInfo{cType: "float32x4_t", isVector: true}

	translator.translateAssignStmt(assignStmt)
	output := translator.buf.String()

	if !strings.Contains(output, "_getlane_buf") {
		t.Errorf("expected store-to-stack pattern for variable index, got: %s", output)
	}
}

// TestEmitFuncSignature_ScalarGenericReturn verifies that helper functions
// returning a generic type T get the profile's scalar type (e.g., "float")
// instead of defaulting to "long". This was the root cause of the NEON
// normalize fcvtzs bug: BaseDot returned T which was emitted as "long",
// causing the compiler to truncate the float result to an integer.
func TestEmitFuncSignature_ScalarGenericReturn(t *testing.T) {
	tests := []struct {
		name       string
		elemType   string
		wantReturn string
	}{
		{"float32", "float32", "float BaseDot("},
		{"float64", "float64", "double BaseDot("},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			profile := testProfile(t, tt.elemType)
			translator := NewCASTTranslator(profile, tt.elemType)
			translator.helperMode = true

			pf := &ParsedFunc{
				Name: "BaseDot",
				Params: []Param{
					{Name: "a", Type: "[]T"},
					{Name: "b", Type: "[]T"},
				},
				Returns: []Param{
					{Name: "", Type: "T"},
				},
			}

			translator.buildParamMap(pf)
			translator.emitFuncSignature(pf)

			output := translator.buf.String()
			if !strings.Contains(output, tt.wantReturn) {
				t.Errorf("expected return type %q in signature, got: %s", tt.wantReturn, output)
			}
			if strings.Contains(output, "long BaseDot(") {
				t.Errorf("BaseDot should not return long (would cause fcvtzs truncation), got: %s", output)
			}
		})
	}
}

// TestEmitFuncSignature_ScalarGenericReturn_BFloat16 verifies that BFloat16
// helpers returning T get the scalar arithmetic type ("float"), not "long".
func TestEmitFuncSignature_ScalarGenericReturn_BFloat16(t *testing.T) {
	profile := testProfile(t, "hwy.BFloat16")
	translator := NewCASTTranslator(profile, "hwy.BFloat16")
	translator.helperMode = true

	pf := &ParsedFunc{
		Name: "BaseDot",
		Params: []Param{
			{Name: "a", Type: "[]T"},
			{Name: "b", Type: "[]T"},
		},
		Returns: []Param{
			{Name: "", Type: "T"},
		},
	}

	translator.buildParamMap(pf)
	translator.emitFuncSignature(pf)

	output := translator.buf.String()
	// BFloat16 arithmetic is promoted to float, so BaseDot should return float
	if strings.Contains(output, "long BaseDot(") {
		t.Errorf("BFloat16 BaseDot should not return long, got: %s", output)
	}
}
