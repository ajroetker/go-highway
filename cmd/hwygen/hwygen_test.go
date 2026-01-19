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

func TestGetTarget(t *testing.T) {
	tests := []struct {
		name    string
		target  string
		wantErr bool
	}{
		{"AVX2", "avx2", false},
		{"AVX512", "avx512", false},
		{"Fallback", "fallback", false},
		{"Unknown", "unknown", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := GetTarget(tt.target)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetTarget(%q) error = %v, wantErr %v", tt.target, err, tt.wantErr)
			}
		})
	}
}

func TestTargetLanesFor(t *testing.T) {
	avx2 := AVX2Target()

	tests := []struct {
		elemType string
		want     int
	}{
		{"float32", 8},  // 32 bytes / 4 = 8
		{"float64", 4},  // 32 bytes / 8 = 4
		{"int32", 8},    // 32 bytes / 4 = 8
		{"int64", 4},    // 32 bytes / 8 = 4
	}

	for _, tt := range tests {
		t.Run(tt.elemType, func(t *testing.T) {
			got := avx2.LanesFor(tt.elemType)
			if got != tt.want {
				t.Errorf("AVX2Target.LanesFor(%q) = %d, want %d", tt.elemType, got, tt.want)
			}
		})
	}
}

func TestGetConcreteTypes(t *testing.T) {
	tests := []struct {
		constraint string
		wantLen    int
		wantFirst  string
	}{
		{"hwy.Floats", 4, "hwy.Float16"},    // Float16, BFloat16, float32, float64
		{"hwy.FloatsNative", 2, "float32"},  // float32, float64 only (no half-precision)
		{"hwy.SignedInts", 2, "int32"},
		{"hwy.Integers", 4, "int32"},
		{"unknown", 2, "float32"}, // Default
	}

	for _, tt := range tests {
		t.Run(tt.constraint, func(t *testing.T) {
			got := GetConcreteTypes(tt.constraint)
			if len(got) != tt.wantLen {
				t.Errorf("GetConcreteTypes(%q) returned %d types, want %d", tt.constraint, len(got), tt.wantLen)
			}
			if len(got) > 0 && got[0] != tt.wantFirst {
				t.Errorf("GetConcreteTypes(%q) first type = %q, want %q", tt.constraint, got[0], tt.wantFirst)
			}
		})
	}
}

func TestParseSimpleFunction(t *testing.T) {
	// Create a temporary test file
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.go")

	content := `package test

import "github.com/ajroetker/go-highway/hwy"

func BaseAdd[T hwy.Floats](a, b, result []T) {
	size := min(len(a), len(b), len(result))
	for i := 0; i < size; i += 8 {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		vr := hwy.Add(va, vb)
		hwy.Store(vr, result[i:])
	}
}
`

	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Parse the file
	result, err := Parse(testFile)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// Verify results
	if result.PackageName != "test" {
		t.Errorf("Package name = %q, want %q", result.PackageName, "test")
	}

	if len(result.Funcs) != 1 {
		t.Fatalf("Got %d functions, want 1", len(result.Funcs))
	}

	f := result.Funcs[0]
	if f.Name != "BaseAdd" {
		t.Errorf("Function name = %q, want %q", f.Name, "BaseAdd")
	}

	if len(f.TypeParams) != 1 {
		t.Errorf("Got %d type params, want 1", len(f.TypeParams))
	}

	if len(f.HwyCalls) < 3 {
		t.Errorf("Got %d hwy calls, want at least 3 (Load, Load, Add, Store)", len(f.HwyCalls))
	}

	// Check for specific operations
	var hasLoad, hasAdd, hasStore bool
	for _, call := range f.HwyCalls {
		switch call.FuncName {
		case "Load":
			hasLoad = true
		case "Add":
			hasAdd = true
		case "Store":
			hasStore = true
		}
	}

	if !hasLoad {
		t.Error("Expected to find Load operation")
	}
	if !hasAdd {
		t.Error("Expected to find Add operation")
	}
	if !hasStore {
		t.Error("Expected to find Store operation")
	}
}

func TestGeneratorEndToEnd(t *testing.T) {
	// Create a temporary directory for test
	tmpDir := t.TempDir()

	// Create a simple test input file
	inputFile := filepath.Join(tmpDir, "add.go")
	content := `package testadd

import "github.com/ajroetker/go-highway/hwy"

func BaseAdd[T hwy.Floats](a, b, result []T) {
	size := min(len(a), len(b), len(result))
	vOne := hwy.Set(T(1))
	for i := 0; i < size; i += vOne.NumElements() {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		vr := hwy.Add(va, vb)
		hwy.Store(vr, result[i:])
	}
}
`

	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	// Create generator
	gen := &Generator{
		InputFile: inputFile,
		OutputDir: tmpDir,
		Targets:   []string{"avx2", "fallback"},
	}

	// Run generation
	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	// Check that expected files were created
	// Note: dispatch files are now architecture-specific and prefixed with function name
	expectedFiles := []string{
		"dispatch_add_amd64.gen.go",
		"dispatch_add_other.gen.go",
		"add_avx2.gen.go",
		"add_fallback.gen.go",
	}

	for _, filename := range expectedFiles {
		path := filepath.Join(tmpDir, filename)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("Expected file %q was not created", filename)
		} else {
			// Read and check basic content
			content, err := os.ReadFile(path)
			if err != nil {
				t.Errorf("Failed to read %q: %v", filename, err)
				continue
			}

			contentStr := string(content)

			// Check for package declaration
			if !strings.Contains(contentStr, "package testadd") {
				t.Errorf("File %q missing package declaration", filename)
			}

			// Check for "Code generated" comment
			if !strings.Contains(contentStr, "Code generated by hwygen") {
				t.Errorf("File %q missing generation comment", filename)
			}
		}
	}

	// Check dispatcher file specifically
	dispatchPath := filepath.Join(tmpDir, "dispatch_add_amd64.gen.go")
	dispatchContent, err := os.ReadFile(dispatchPath)
	if err != nil {
		t.Fatalf("Failed to read dispatcher: %v", err)
	}

	dispatchStr := string(dispatchContent)

	// Should have function variables (now with type suffix)
	if !strings.Contains(dispatchStr, "var AddFloat32 func") {
		t.Error("Dispatcher missing AddFloat32 function variable")
	}

	// Should have init function
	if !strings.Contains(dispatchStr, "func init()") {
		t.Error("Dispatcher missing init function")
	}

	// Should have target-specific init functions
	if !strings.Contains(dispatchStr, "func initAddAVX2()") {
		t.Error("Dispatcher missing initAddAVX2 function")
	}
	if !strings.Contains(dispatchStr, "func initAddFallback()") {
		t.Error("Dispatcher missing initAddFallback function")
	}
}

func TestSpecializeType(t *testing.T) {
	typeParams := []TypeParam{
		{Name: "T", Constraint: "hwy.Floats"},
	}

	tests := []struct {
		input    string
		elemType string
		want     string
	}{
		{"[]T", "float32", "[]float32"},
		{"T", "float64", "float64"},
		{"[][]T", "int32", "[][]int32"},
		{"map[string]T", "float32", "map[string]float32"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := specializeType(tt.input, typeParams, tt.elemType)
			if got != tt.want {
				t.Errorf("specializeType(%q, _, %q) = %q, want %q", tt.input, tt.elemType, got, tt.want)
			}
		})
	}
}

func TestBuildDispatchFuncName(t *testing.T) {
	tests := []struct {
		baseName string
		elemType string
		want     string
	}{
		// All types now get explicit suffixes for consistent generic dispatch
		{"BaseSigmoid", "float32", "SigmoidFloat32"},
		{"BaseSigmoid", "float64", "SigmoidFloat64"},
		{"BaseAdd", "float32", "AddFloat32"},
		{"BaseAdd", "int32", "AddInt32"},
	}

	for _, tt := range tests {
		t.Run(tt.baseName+"_"+tt.elemType, func(t *testing.T) {
			got := buildDispatchFuncName(tt.baseName, tt.elemType)
			if got != tt.want {
				t.Errorf("buildDispatchFuncName(%q, %q) = %q, want %q", tt.baseName, tt.elemType, got, tt.want)
			}
		})
	}
}

func TestContribSubpackageImports(t *testing.T) {
	// Create a temporary directory for test
	tmpDir := t.TempDir()

	// Create a test input file that uses contrib/math functions
	inputFile := filepath.Join(tmpDir, "sigmoid.go")
	content := `package testsigmoid

import (
	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib"
)

func BaseSigmoid[T hwy.Floats](input, output []T) {
	size := min(len(input), len(output))
	vOne := hwy.Set(T(1))
	for i := 0; i < size; i += vOne.NumElements() {
		x := hwy.Load(input[i:])
		y := contrib.Sigmoid(x)
		hwy.Store(y, output[i:])
	}
}
`

	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	// Create generator
	gen := &Generator{
		InputFile: inputFile,
		OutputDir: tmpDir,
		Targets:   []string{"avx2", "fallback"},
	}

	// Run generation
	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	// Check AVX2 output file has contrib/math import
	avx2Path := filepath.Join(tmpDir, "sigmoid_avx2.gen.go")
	avx2Content, err := os.ReadFile(avx2Path)
	if err != nil {
		t.Fatalf("Failed to read AVX2 output: %v", err)
	}

	avx2Str := string(avx2Content)

	// Should have contrib/math import for SIMD target
	if !strings.Contains(avx2Str, `"github.com/ajroetker/go-highway/hwy/contrib/math"`) {
		t.Error("AVX2 output missing contrib/math import")
	}

	// Should use math.BaseSigmoidVec_avx2 style function calls
	if !strings.Contains(avx2Str, "math.BaseSigmoidVec_avx2") {
		t.Error("AVX2 output missing math.BaseSigmoidVec_avx2 function call")
	}

	// Check fallback output uses hwy package
	fallbackPath := filepath.Join(tmpDir, "sigmoid_fallback.gen.go")
	fallbackContent, err := os.ReadFile(fallbackPath)
	if err != nil {
		t.Fatalf("Failed to read fallback output: %v", err)
	}

	fallbackStr := string(fallbackContent)

	// Fallback should use hwy package for core ops
	if !strings.Contains(fallbackStr, `"github.com/ajroetker/go-highway/hwy"`) {
		t.Error("Fallback output missing hwy import")
	}

	// Fallback should use math.BaseSigmoidVec for contrib functions (not hwy.Sigmoid)
	if !strings.Contains(fallbackStr, "math.BaseSigmoidVec_fallback") {
		t.Error("Fallback output missing math.BaseSigmoidVec_fallback function call")
	}
}

func TestDetectContribPackages(t *testing.T) {
	targets := []Target{AVX2Target(), FallbackTarget()}

	tests := []struct {
		name      string
		calls     []HwyCall
		wantMath  bool
		wantVec   bool
		wantAlgo  bool
	}{
		{
			name:     "No contrib",
			calls:    []HwyCall{{Package: "hwy", FuncName: "Add"}},
			wantMath: false, wantVec: false, wantAlgo: false,
		},
		{
			name:     "Math function",
			calls:    []HwyCall{{Package: "contrib", FuncName: "Exp"}},
			wantMath: true, wantVec: false, wantAlgo: false,
		},
		{
			name:     "Dot function",
			calls:    []HwyCall{{Package: "contrib", FuncName: "Dot"}},
			wantMath: false, wantVec: true, wantAlgo: false,
		},
		{
			name:      "Multiple functions",
			calls:     []HwyCall{{Package: "contrib", FuncName: "Sigmoid"}, {Package: "contrib", FuncName: "Dot"}},
			wantMath:  true,
			wantVec:   true,
			wantAlgo:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			funcs := []ParsedFunc{{HwyCalls: tt.calls}}
			pkgs := detectContribPackages(funcs, targets)

			if pkgs.Math != tt.wantMath {
				t.Errorf("Math = %v, want %v", pkgs.Math, tt.wantMath)
			}
			if pkgs.Vec != tt.wantVec {
				t.Errorf("Dot = %v, want %v", pkgs.Vec, tt.wantVec)
			}
			if pkgs.Algo != tt.wantAlgo {
				t.Errorf("Algo = %v, want %v", pkgs.Algo, tt.wantAlgo)
			}
		})
	}
}

func TestParseCondition(t *testing.T) {
	tests := []struct {
		name      string
		condition string
		target    string
		elemType  string
		want      bool
	}{
		// Simple type conditions
		{"f32 matches float32", "f32", "AVX2", "float32", true},
		{"f32 doesn't match float64", "f32", "AVX2", "float64", false},
		{"f64 matches float64", "f64", "AVX2", "float64", true},
		{"f64 doesn't match float32", "f64", "AVX2", "float32", false},

		// Simple target conditions
		{"avx2 matches AVX2", "avx2", "AVX2", "float32", true},
		{"avx2 doesn't match AVX512", "avx2", "AVX512", "float32", false},
		{"avx512 matches AVX512", "avx512", "AVX512", "float32", true},
		{"neon matches NEON", "neon", "NEON", "float32", true},
		{"fallback matches Fallback", "fallback", "Fallback", "float32", true},

		// Compound conditions with AND
		{"f64 && avx2 matches", "f64 && avx2", "AVX2", "float64", true},
		{"f64 && avx2 doesn't match f32", "f64 && avx2", "AVX2", "float32", false},
		{"f64 && avx2 doesn't match avx512", "f64 && avx2", "AVX512", "float64", false},
		{"f32 && neon matches", "f32 && neon", "NEON", "float32", true},

		// Compound conditions with OR
		{"f32 || f64 matches f32", "f32 || f64", "AVX2", "float32", true},
		{"f32 || f64 matches f64", "f32 || f64", "AVX2", "float64", true},
		{"avx2 || avx512 matches avx2", "avx2 || avx512", "AVX2", "float32", true},
		{"avx2 || avx512 matches avx512", "avx2 || avx512", "AVX512", "float32", true},
		{"avx2 || avx512 doesn't match neon", "avx2 || avx512", "NEON", "float32", false},

		// Negation
		{"!avx2 matches avx512", "!avx2", "AVX512", "float32", true},
		{"!avx2 doesn't match avx2", "!avx2", "AVX2", "float32", false},
		{"!f64 matches f32", "!f64", "AVX2", "float32", true},
		{"!f64 doesn't match f64", "!f64", "AVX2", "float64", false},

		// Complex compound with mixed AND/OR
		{"(f64 && avx2) matches", "f64 && avx2", "AVX2", "float64", true},
		{"f64 && !avx512 matches avx2", "f64 && !avx512", "AVX2", "float64", true},
		{"f64 && !avx512 doesn't match avx512", "f64 && !avx512", "AVX512", "float64", false},

		// Category conditions
		{"float matches float32", "float", "AVX2", "float32", true},
		{"float matches float64", "float", "AVX2", "float64", true},
		{"float matches Float16", "float", "NEON", "hwy.Float16", true},
		{"float matches BFloat16", "float", "NEON", "hwy.BFloat16", true},
		{"float doesn't match int32", "float", "AVX2", "int32", false},
		{"float doesn't match uint64", "float", "AVX2", "uint64", false},
		{"int matches int32", "int", "AVX2", "int32", true},
		{"int matches int64", "int", "AVX2", "int64", true},
		{"int doesn't match uint32", "int", "AVX2", "uint32", false},
		{"int doesn't match float32", "int", "AVX2", "float32", false},
		{"uint matches uint32", "uint", "AVX2", "uint32", true},
		{"uint matches uint64", "uint", "AVX2", "uint64", true},
		{"uint doesn't match int32", "uint", "AVX2", "int32", false},
		{"uint doesn't match float64", "uint", "AVX2", "float64", false},

		// Category with negation
		{"!float matches int32", "!float", "AVX2", "int32", true},
		{"!float doesn't match float32", "!float", "AVX2", "float32", false},

		// Category compound conditions
		{"float && avx2 matches", "float && avx2", "AVX2", "float32", true},
		{"float && avx2 doesn't match int32", "float && avx2", "AVX2", "int32", false},
		{"int || uint matches int32", "int || uint", "AVX2", "int32", true},
		{"int || uint matches uint64", "int || uint", "AVX2", "uint64", true},
		{"int || uint doesn't match float32", "int || uint", "AVX2", "float32", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pc := parseCondition(tt.condition)
			got := pc.Evaluate(tt.target, tt.elemType)
			if got != tt.want {
				t.Errorf("parseCondition(%q).Evaluate(%q, %q) = %v, want %v",
					tt.condition, tt.target, tt.elemType, got, tt.want)
			}
		})
	}
}

func TestIsScalarTailLoop(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		iterator string
		end      string
		want     bool
	}{
		{
			name:     "valid scalar tail loop",
			code:     "for ; i < n; i++ { dst[i] += s[i] }",
			iterator: "i",
			end:      "n",
			want:     true,
		},
		{
			name:     "scalar tail loop with different iterator",
			code:     "for ; j < n; j++ { dst[j] += s[j] }",
			iterator: "j",
			end:      "n",
			want:     true,
		},
		{
			name:     "scalar tail loop with different end var",
			code:     "for ; i < size; i++ { dst[i] += s[i] }",
			iterator: "i",
			end:      "size",
			want:     true,
		},
		{
			name:     "not a scalar tail - has init",
			code:     "for i := 0; i < n; i++ { dst[i] += s[i] }",
			iterator: "i",
			end:      "n",
			want:     false,
		},
		{
			name:     "not a scalar tail - wrong iterator in condition",
			code:     "for ; j < n; i++ { dst[i] += s[i] }",
			iterator: "i",
			end:      "n",
			want:     false,
		},
		{
			name:     "not a scalar tail - wrong end variable",
			code:     "for ; i < m; i++ { dst[i] += s[i] }",
			iterator: "i",
			end:      "n",
			want:     false,
		},
		{
			name:     "not a scalar tail - decrement instead of increment",
			code:     "for ; i < n; i-- { dst[i] += s[i] }",
			iterator: "i",
			end:      "n",
			want:     false,
		},
		{
			name:     "not a scalar tail - >= condition",
			code:     "for ; i >= n; i++ { dst[i] += s[i] }",
			iterator: "i",
			end:      "n",
			want:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Parse the for loop
			src := "package test\nfunc f() { " + tt.code + " }"
			fset := token.NewFileSet()
			file, err := parser.ParseFile(fset, "", src, 0)
			if err != nil {
				t.Fatalf("Failed to parse test code: %v", err)
			}

			// Find the for statement
			var forStmt *ast.ForStmt
			ast.Inspect(file, func(n ast.Node) bool {
				if f, ok := n.(*ast.ForStmt); ok {
					forStmt = f
					return false
				}
				return true
			})

			if forStmt == nil {
				t.Fatal("No for statement found in parsed code")
			}

			got := isScalarTailLoop(forStmt, tt.iterator, tt.end)
			if got != tt.want {
				t.Errorf("isScalarTailLoop() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestConditionalBlockFiltering(t *testing.T) {
	// Create a temporary test file with hwy:if directives
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.go")

	content := `package test

import "github.com/ajroetker/go-highway/hwy"

func BaseTest[T hwy.Floats](a, result []T) {
	size := len(a)
	for i := 0; i < size; i++ {
		x := a[i]
		//hwy:if f64 && avx2
		result[i] = x * 2 // scalar fallback for f64 on avx2
		//hwy:else
		result[i] = x * 3 // normal simd path
		//hwy:endif
	}
}
`

	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Parse the file
	result, err := Parse(testFile)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// Verify conditional blocks were parsed
	if len(result.ConditionalBlocks) != 1 {
		t.Fatalf("Got %d conditional blocks, want 1", len(result.ConditionalBlocks))
	}

	block := result.ConditionalBlocks[0]
	if block.Condition != "f64 && avx2" {
		t.Errorf("Block condition = %q, want %q", block.Condition, "f64 && avx2")
	}

	// Verify the parsed condition structure
	if block.ParsedCondition == nil {
		t.Fatal("ParsedCondition is nil")
	}
	if block.ParsedCondition.Op != "&&" {
		t.Errorf("ParsedCondition.Op = %q, want %q", block.ParsedCondition.Op, "&&")
	}

	// Test evaluation
	if !block.ParsedCondition.Evaluate("AVX2", "float64") {
		t.Error("Condition should match f64 && avx2")
	}
	if block.ParsedCondition.Evaluate("AVX2", "float32") {
		t.Error("Condition should not match f32 on avx2")
	}
	if block.ParsedCondition.Evaluate("AVX512", "float64") {
		t.Error("Condition should not match f64 on avx512")
	}
}
