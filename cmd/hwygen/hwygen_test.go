package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// makeTestSpecs creates TargetSpecs from target names and a mode for tests.
func makeTestSpecs(mode TargetMode, names ...string) []TargetSpec {
	var specs []TargetSpec
	for _, name := range names {
		t, err := GetTarget(name)
		if err != nil {
			panic(err)
		}
		specs = append(specs, TargetSpec{Target: t, Mode: mode})
	}
	return specs
}

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
		{"float32", 8}, // 32 bytes / 4 = 8
		{"float64", 4}, // 32 bytes / 8 = 4
		{"int32", 8},   // 32 bytes / 4 = 8
		{"int64", 4},   // 32 bytes / 8 = 4
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
		{"hwy.Floats", 4, "hwy.Float16"},   // Float16, BFloat16, float32, float64
		{"hwy.FloatsNative", 2, "float32"}, // float32, float64 only (no half-precision)
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
		InputFile:   inputFile,
		OutputDir:   tmpDir,
		TargetSpecs: makeTestSpecs(TargetModeGoSimd, "avx2", "fallback"),
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
			if !strings.Contains(contentStr, HeaderNote) {
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
		baseName  string
		elemType  string
		isGeneric bool
		private   bool
		want      string
	}{
		// Generic functions get type suffixes
		{"BaseSigmoid", "float32", true, false, "SigmoidFloat32"},
		{"BaseSigmoid", "float64", true, false, "SigmoidFloat64"},
		{"BaseAdd", "float32", true, false, "AddFloat32"},
		{"BaseAdd", "int32", true, false, "AddInt32"},
		// Non-generic functions don't get type suffixes
		{"BaseDecodeStreamVByte32Into", "uint8", false, false, "DecodeStreamVByte32Into"},
		{"BasePack32", "uint32", false, false, "Pack32"},
		// Private (lowercase "base" prefix) functions produce unexported names
		{"baseSigmoid", "float32", true, true, "sigmoidFloat32"},
		{"baseSigmoid", "float64", true, true, "sigmoidFloat64"},
		{"baseAdd", "float32", true, true, "addFloat32"},
		{"basePack32", "uint32", false, true, "pack32"},
	}

	for _, tt := range tests {
		t.Run(tt.baseName+"_"+tt.elemType, func(t *testing.T) {
			got := buildDispatchFuncName(tt.baseName, tt.elemType, tt.isGeneric, tt.private)
			if got != tt.want {
				t.Errorf("buildDispatchFuncName(%q, %q, %v, %v) = %q, want %q", tt.baseName, tt.elemType, tt.isGeneric, tt.private, got, tt.want)
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
		InputFile:   inputFile,
		OutputDir:   tmpDir,
		TargetSpecs: makeTestSpecs(TargetModeGoSimd, "avx2", "fallback"),
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
		name     string
		calls    []HwyCall
		wantMath bool
		wantVec  bool
		wantAlgo bool
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
			name:     "Multiple functions",
			calls:    []HwyCall{{Package: "contrib", FuncName: "Sigmoid"}, {Package: "contrib", FuncName: "Dot"}},
			wantMath: true,
			wantVec:  true,
			wantAlgo: false,
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
		{
			name:     "scalar tail loop with len(dst) end - simple copy",
			code:     "for ; i < len(dst); i++ { dst[i] = src[i] }",
			iterator: "i",
			end:      "len(dst)",
			want:     true,
		},
		{
			name:     "scalar tail loop with len(src) end - simple copy",
			code:     "for ; i < len(src); i++ { dst[i] = src[i] }",
			iterator: "i",
			end:      "len(src)",
			want:     true,
		},
		{
			name:     "not a scalar tail - uses external variable scale (multiply)",
			code:     "for ; i < len(dst); i++ { dst[i] *= scale }",
			iterator: "i",
			end:      "len(dst)",
			want:     false,
		},
		{
			name:     "not a scalar tail - uses external variable scale (expression)",
			code:     "for ; i < len(src); i++ { dst[i] = src[i] * scale }",
			iterator: "i",
			end:      "len(src)",
			want:     false,
		},
		{
			name:     "not a scalar tail - assigns to local variable",
			code:     "for ; i < len(src); i++ { dst[i] = src[i] - prev; prev = src[i] }",
			iterator: "i",
			end:      "len(src)",
			want:     false,
		},
		{
			name:     "not a scalar tail - assigns to local variable only",
			code:     "for ; i < n; i++ { sum += src[i] }",
			iterator: "i",
			end:      "n",
			want:     false,
		},
		{
			name:     "scalar tail with type conversion via selector",
			code:     "for ; i < len(dst); i++ { dst[i] = hwy.Float32ToFloat16(src[i].Float32()) }",
			iterator: "i",
			end:      "len(dst)",
			want:     true,
		},
		{
			name:     "scalar tail with method call on indexed element",
			code:     "for ; i < len(dst); i++ { dst[i] = pkg.Convert(dst[i].Value() + s[i].Value()) }",
			iterator: "i",
			end:      "len(dst)",
			want:     true,
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

func TestOutputPrefix(t *testing.T) {
	// Create a temporary directory for test
	tmpDir := t.TempDir()

	// Create a simple test input file
	inputFile := filepath.Join(tmpDir, "prefix_test.go")
	content := `package testprefix

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

	// Create generator with OutputPrefix
	gen := &Generator{
		InputFile:    inputFile,
		OutputDir:    tmpDir,
		OutputPrefix: "custom_prefix",
		TargetSpecs:  makeTestSpecs(TargetModeGoSimd, "avx2", "fallback"),
	}

	// Run generation
	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	// Check that files with custom prefix were created
	expectedFiles := []string{
		"custom_prefix_avx2.gen.go",
		"custom_prefix_fallback.gen.go",
	}

	for _, filename := range expectedFiles {
		path := filepath.Join(tmpDir, filename)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("Expected file %q was not created", filename)
		} else {
			content, err := os.ReadFile(path)
			if err != nil {
				t.Errorf("Failed to read %q: %v", filename, err)
				continue
			}
			if !strings.Contains(string(content), HeaderNote) {
				t.Errorf("File %q missing generation comment", filename)
			}
		}
	}

	// Ensure default files are NOT created
	defaultFiles := []string{
		"prefix_test_avx2.gen.go",
		"prefix_test_fallback.gen.go",
	}
	for _, filename := range defaultFiles {
		path := filepath.Join(tmpDir, filename)
		if _, err := os.Stat(path); err == nil {
			t.Errorf("Unexpected default file %q was created", filename)
		}
	}
}

func TestDispatchPrefix(t *testing.T) {
	// Create a temporary directory for test
	tmpDir := t.TempDir()

	// Create a simple test input file
	inputFile := filepath.Join(tmpDir, "dispatch_prefix.go")
	content := `package testdispatch

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

	// Create generator with DispatchPrefix
	gen := &Generator{
		InputFile:      inputFile,
		OutputDir:      tmpDir,
		DispatchPrefix: "custom_dispatch",
		TargetSpecs:    makeTestSpecs(TargetModeGoSimd, "avx2", "fallback"),
	}

	// Run generation
	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	// Check that dispatch files with custom prefix were created WITHOUT "dispatch_" prepended
	expectedFiles := []string{
		"custom_dispatch_amd64.gen.go",
		"custom_dispatch_other.gen.go",
	}

	for _, filename := range expectedFiles {
		path := filepath.Join(tmpDir, filename)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("Expected file %q was not created", filename)
		}
	}

	// Check that default dispatch file was NOT created
	defaultDispatch := "dispatch_custom_dispatch_amd64.gen.go"
	if _, err := os.Stat(filepath.Join(tmpDir, defaultDispatch)); err == nil {
		t.Errorf("Unexpected default dispatch file %q was created", defaultDispatch)
	}
}

func TestLoadStoreTransformation(t *testing.T) {
	// Create a temporary directory for test
	tmpDir := t.TempDir()

	// Create a test input file using Load and Store
	// Note: Function must start with "Base" to be processed by hwygen
	inputFile := filepath.Join(tmpDir, "load.go")
	content := `package testload

import "github.com/ajroetker/go-highway/hwy"

func BaseCopy[T hwy.FloatsNative](src, dst []T) {
	n := hwy.NumLanes[T]()
	for i := 0; i <= len(src)-n; i += n {
		v := hwy.Load(src[i:])
		hwy.Store(v, dst[i:])
	}
}
`

	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	// Create generator for AVX2
	gen := &Generator{
		InputFile:   inputFile,
		OutputDir:   tmpDir,
		TargetSpecs: makeTestSpecs(TargetModeGoSimd, "avx2"),
	}

	// Run generation
	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	// Read the generated AVX2 file
	avx2Path := filepath.Join(tmpDir, "load_avx2.gen.go")
	avx2Content, err := os.ReadFile(avx2Path)
	if err != nil {
		t.Fatalf("Failed to read AVX2 output: %v", err)
	}

	contentStr := string(avx2Content)

	// Verify Load was transformed to use unsafe.Pointer
	if !strings.Contains(contentStr, "unsafe.Pointer") {
		t.Error("Generated code missing unsafe.Pointer for Load/Store")
	}

	// Verify the pointer-to-array cast pattern exists
	if !strings.Contains(contentStr, "(*[8]float32)") && !strings.Contains(contentStr, "(*[4]float64)") {
		t.Error("Generated code missing pointer-to-array cast pattern")
	}

	// Verify unsafe import was added
	if !strings.Contains(contentStr, `"unsafe"`) {
		t.Error("Generated code missing unsafe import")
	}

	// Verify archsimd.LoadFloat32x8 is called (not LoadFloat32x8Slice)
	if !strings.Contains(contentStr, "archsimd.LoadFloat32x8(") {
		t.Error("Generated code should use archsimd.LoadFloat32x8 (pointer-based), not LoadFloat32x8Slice")
	}

	// Verify v.Store is called with the cast (not StoreSlice)
	if strings.Contains(contentStr, "StoreSlice") {
		t.Error("Generated code should use v.Store (pointer-based), not StoreSlice")
	}

	// Verify the slice-to-pointer optimization is applied:
	// Should generate &src[i] instead of &src[i:][0]
	// The pattern [i:][0] should NOT appear in the output
	if strings.Contains(contentStr, ":][0]") {
		t.Error("Generated code has unoptimized slice access &slice[i:][0], should be &slice[i]")
	}

	// Verify the optimized pattern &src[i]) or &dst[i]) appears
	if !strings.Contains(contentStr, "&src[i])") && !strings.Contains(contentStr, "&dst[i])") {
		t.Error("Generated code missing optimized pointer pattern &slice[i]")
	}
}

// TestNumLanesTypeParameter verifies that hwy.NumLanes[T]() uses the explicit type parameter T
// for lane count calculation, not the function's first slice parameter type.
// This is a regression test for a bug where functions like:
//
//	func Decode(dst []float32, src []byte) { lanes := hwy.NumLanes[uint8]() ... }
//
// would incorrectly get lanes=8 (float32 AVX2 lanes) instead of lanes=32 (uint8 AVX2 lanes).
func TestNumLanesTypeParameter(t *testing.T) {
	tmpDir := t.TempDir()

	// Create test input that has a first parameter type different from the Load type
	// This triggers the bug: function signature has float32, but operations are on uint8
	inputFile := filepath.Join(tmpDir, "numlanes_type.go")
	content := `package testnumlanes

import (
	"unsafe"
	"github.com/ajroetker/go-highway/hwy"
)

// BaseDecodeFloat32s has float32 as first slice parameter, but uses uint8 operations.
// The lanes variable should be computed from uint8 (32 lanes for AVX2), not float32 (8 lanes).
func BaseDecodeFloat32s(dst []float32, src []byte) {
	if len(dst) == 0 {
		return
	}
	totalBytes := len(dst) * 4
	if len(src) < totalBytes {
		return
	}
	dstBytes := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), totalBytes)

	// This is the key: hwy.NumLanes[uint8]() should give 32 for AVX2, not 8
	lanes := hwy.NumLanes[uint8]()
	i := 0
	for ; i+lanes <= totalBytes; i += lanes {
		v := hwy.Load[uint8](src[i:])
		hwy.Store(v, dstBytes[i:])
	}
	for ; i < totalBytes; i++ {
		dstBytes[i] = src[i]
	}
}
`

	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	// Test AVX2 target (should get lanes=32 for uint8, not lanes=8 for float32)
	gen := &Generator{
		InputFile:   inputFile,
		OutputDir:   tmpDir,
		TargetSpecs: makeTestSpecs(TargetModeGoSimd, "avx2"),
	}

	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	avx2Path := filepath.Join(tmpDir, "numlanes_type_avx2.gen.go")
	avx2Content, err := os.ReadFile(avx2Path)
	if err != nil {
		t.Fatalf("Failed to read AVX2 output: %v", err)
	}

	contentStr := string(avx2Content)

	// The critical check: lanes should be 32 (uint8 AVX2 = 32 bytes = 256 bits)
	// NOT 8 (float32 AVX2 = 8 floats = 256 bits)
	if strings.Contains(contentStr, "lanes := 8") {
		t.Error("Bug: hwy.NumLanes[uint8]() incorrectly generated lanes=8 (float32 lanes) instead of lanes=32 (uint8 lanes)")
	}
	if !strings.Contains(contentStr, "lanes := 32") {
		t.Error("Expected lanes := 32 for uint8 AVX2, but not found in generated code")
	}

	// Also verify the SIMD operations use uint8 vectors
	// hwy.Load generates pointer-based LoadUint8x32 (not LoadUint8x32Slice)
	if !strings.Contains(contentStr, "LoadUint8x32") {
		t.Error("Expected LoadUint8x32 for uint8 AVX2 operations")
	}

	// Test NEON target (should get lanes=16 for uint8, not lanes=4 for float32)
	gen2 := &Generator{
		InputFile:   inputFile,
		OutputDir:   tmpDir,
		TargetSpecs: makeTestSpecs(TargetModeGoSimd, "neon"),
	}

	if err := gen2.Run(); err != nil {
		t.Fatalf("Generator.Run() for NEON failed: %v", err)
	}

	neonPath := filepath.Join(tmpDir, "numlanes_type_neon.gen.go")
	neonContent, err := os.ReadFile(neonPath)
	if err != nil {
		t.Fatalf("Failed to read NEON output: %v", err)
	}

	neonStr := string(neonContent)

	// For NEON: lanes should be 16 (uint8 NEON = 16 bytes = 128 bits)
	// NOT 4 (float32 NEON = 4 floats = 128 bits)
	if strings.Contains(neonStr, "lanes := 4") {
		t.Error("Bug: hwy.NumLanes[uint8]() incorrectly generated lanes=4 (float32 lanes) instead of lanes=16 (uint8 lanes)")
	}
	if !strings.Contains(neonStr, "lanes := 16") {
		t.Error("Expected lanes := 16 for uint8 NEON, but not found in generated code")
	}
}

func TestNEONHalfPrecisionStoreSlice(t *testing.T) {
	// Test that half-precision StoreSlice on NEON properly casts to []uint16
	tmpDir := t.TempDir()

	inputFile := filepath.Join(tmpDir, "halfprec_store.go")
	content := `package testhalf

import "github.com/ajroetker/go-highway/hwy"

func BaseScale[T hwy.Floats](data []T, scale T) {
	lanes := hwy.MaxLanes[T]()
	scaleVec := hwy.Set(scale)
	for i := 0; i < len(data); i += lanes {
		v := hwy.LoadSlice(data[i:])
		result := hwy.Mul(v, scaleVec)
		hwy.StoreSlice(result, data[i:])
	}
}
`

	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	gen := &Generator{
		InputFile:   inputFile,
		OutputDir:   tmpDir,
		TargetSpecs: makeTestSpecs(TargetModeGoSimd, "neon"),
	}

	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	neonPath := filepath.Join(tmpDir, "halfprec_store_neon.gen.go")
	neonContent, err := os.ReadFile(neonPath)
	if err != nil {
		t.Fatalf("Failed to read NEON output: %v", err)
	}

	neonStr := string(neonContent)

	// For Float16 function, StoreSlice should cast slice to []uint16
	// Look for the unsafe.Slice conversion pattern
	if !strings.Contains(neonStr, "unsafe.Slice((*uint16)") {
		t.Error("NEON Float16 StoreSlice should cast slice to []uint16 using unsafe.Slice")
	}

	// Extract just the Float16 function to check for proper conversion
	float16Start := strings.Index(neonStr, "func BaseScale_neon_Float16")
	float16End := strings.Index(neonStr, "func BaseScale_neon_BFloat16")
	if float16Start == -1 || float16End == -1 {
		t.Fatal("Could not find Float16 function in generated code")
	}
	float16Func := neonStr[float16Start:float16End]

	// In the Float16 function, all StoreSlice calls should use unsafe.Slice
	// Count StoreSlice calls and ensure they all have unsafe.Slice
	storeSliceCalls := strings.Count(float16Func, ".StoreSlice(")
	unsafeSliceCalls := strings.Count(float16Func, ".StoreSlice(unsafe.Slice(")

	if storeSliceCalls != unsafeSliceCalls {
		t.Errorf("Float16 function has %d StoreSlice calls but only %d use unsafe.Slice conversion",
			storeSliceCalls, unsafeSliceCalls)
	}

	if storeSliceCalls == 0 {
		t.Error("Float16 function should have StoreSlice calls")
	}
}

// TestCModeExpGeneration verifies that -c mode generates correct C code for the
// exp function across element types. This validates:
// - f32: native NEON path with correct intrinsics and hex constants
// - f16: promoted math path (load f16, promote to f32, compute, demote back)
// - bf16: promoted math path with bfloat16 load/store intrinsics
//
// These were previously hand-written in hwy/c/ files; hwygen now generates them.
func TestCModeExpGeneration(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a minimal Vec→Vec function named BaseExpVec.
	// The C emitter dispatches on function name, not body content.
	inputFile := filepath.Join(tmpDir, "exp_base.go")
	content := `package testexp

import "github.com/ajroetker/go-highway/hwy"

func BaseExpVec[T hwy.Floats](x hwy.Vec[T]) hwy.Vec[T] {
	return hwy.Mul(x, x)
}
`
	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	gen := &Generator{
		InputFile:   inputFile,
		OutputDir:   tmpDir,
		TargetSpecs: makeTestSpecs(TargetModeC, "neon"),
	}

	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() in CMode failed: %v", err)
	}

	// ---- f32: native NEON path ----
	f32Path := filepath.Join(tmpDir, "baseexpvec_c_f32_neon_arm64.c")
	f32Content, err := os.ReadFile(f32Path)
	if err != nil {
		t.Fatalf("Failed to read f32 C file: %v", err)
	}
	f32 := string(f32Content)

	// Verify function signature uses float pointers
	if !strings.Contains(f32, "void exp_c_f32_neon(float *input, float *result, long *len)") {
		t.Error("f32: expected float pointer signature")
	}

	// Verify NEON f32 intrinsics
	for _, intrinsic := range []string{
		"vld1q_f32",  // vector load
		"vst1q_f32",  // vector store
		"vfmaq_f32",  // fused multiply-add (Horner polynomial)
		"vrndnq_f32", // round to nearest (range reduction)
		"vfmsq_f32",  // fused multiply-subtract
	} {
		if !strings.Contains(f32, intrinsic) {
			t.Errorf("f32: missing NEON intrinsic %q", intrinsic)
		}
	}

	// Verify hex constant for invLn2 (1/ln(2) = 1.4426950408889634)
	if !strings.Contains(f32, "0x3FB8AA3B") {
		t.Error("f32: missing hex constant 0x3FB8AA3B (invLn2)")
	}

	// Verify 3-tier loop structure: main (4 vectors), single, scalar
	if !strings.Contains(f32, "i + 15 < n; i += 16") {
		t.Error("f32: missing main loop (16 elements = 4 vectors of 4)")
	}
	if !strings.Contains(f32, "i + 3 < n; i += 4") {
		t.Error("f32: missing single vector loop (4 elements)")
	}
	if !strings.Contains(f32, "for (; i < n; i++)") {
		t.Error("f32: missing scalar tail loop")
	}

	// ---- f16: promoted math path (load f16, compute in f32, store f16) ----
	f16Path := filepath.Join(tmpDir, "baseexpvec_c_f16_neon_arm64.c")
	f16Content, err := os.ReadFile(f16Path)
	if err != nil {
		t.Fatalf("Failed to read f16 C file: %v", err)
	}
	f16 := string(f16Content)

	// Verify function signature uses unsigned short (f16 storage type)
	if !strings.Contains(f16, "void exp_c_f16_neon(unsigned short *input, unsigned short *result, long *len)") {
		t.Error("f16: expected unsigned short pointer signature")
	}

	// Verify promoted math comment
	if !strings.Contains(f16, "promoted Exp via f32 polynomial") {
		t.Error("f16: missing promoted math comment")
	}

	// Verify f16 load intrinsic
	if !strings.Contains(f16, "vld1q_f16") {
		t.Error("f16: missing vld1q_f16 load intrinsic")
	}

	// Verify f16→f32 promotion intrinsics
	if !strings.Contains(f16, "vcvt_f32_f16") {
		t.Error("f16: missing vcvt_f32_f16 promote intrinsic")
	}

	// Verify computation happens in f32 (not f16)
	if !strings.Contains(f16, "float32x4_t") {
		t.Error("f16: computation should use float32x4_t vectors")
	}
	if !strings.Contains(f16, "vfmaq_f32") {
		t.Error("f16: Horner polynomial should use f32 FMA")
	}

	// Verify f32→f16 demotion and store
	if !strings.Contains(f16, "vcvt_f16_f32") {
		t.Error("f16: missing vcvt_f16_f32 demote intrinsic")
	}
	if !strings.Contains(f16, "vst1q_f16") {
		t.Error("f16: missing vst1q_f16 store intrinsic")
	}

	// Verify constants are f32 (not f64)
	if !strings.Contains(f16, "float32x4_t invLn2") {
		t.Error("f16: promoted constants should be float32x4_t")
	}
	if strings.Contains(f16, "float64x2_t") {
		t.Error("f16: should NOT contain float64x2_t (was using wrong legacy path)")
	}

	// ---- bf16: promoted math with bfloat16 intrinsics ----
	bf16Path := filepath.Join(tmpDir, "baseexpvec_c_bf16_neon_arm64.c")
	bf16Content, err := os.ReadFile(bf16Path)
	if err != nil {
		t.Fatalf("Failed to read bf16 C file: %v", err)
	}
	bf16 := string(bf16Content)

	// Verify bf16 function signature
	if !strings.Contains(bf16, "void exp_c_bf16_neon(unsigned short *input, unsigned short *result, long *len)") {
		t.Error("bf16: expected unsigned short pointer signature")
	}

	// Verify bf16 load intrinsic
	if !strings.Contains(bf16, "vld1q_bf16") {
		t.Error("bf16: missing vld1q_bf16 load intrinsic")
	}

	// Verify bf16→f32 promotion (bit-shift pattern for bfloat16)
	if !strings.Contains(bf16, "vshll_n_u16") {
		t.Error("bf16: missing vshll_n_u16 for bf16→f32 promotion")
	}

	// Verify computation in f32
	if !strings.Contains(bf16, "float32x4_t") {
		t.Error("bf16: computation should use float32x4_t vectors")
	}

	// Verify f32→bf16 demotion
	if !strings.Contains(bf16, "vst1q_bf16") {
		t.Error("bf16: missing vst1q_bf16 store intrinsic")
	}
}

func TestIsASTCEligible(t *testing.T) {
	tests := []struct {
		name string
		pf   ParsedFunc
		want bool
	}{
		{
			name: "matmul with slices and int params",
			pf: ParsedFunc{
				Name: "BaseMatMul",
				Params: []Param{
					{Name: "a", Type: "[]T"},
					{Name: "b", Type: "[]T"},
					{Name: "c", Type: "[]T"},
					{Name: "m", Type: "int"},
					{Name: "n", Type: "int"},
					{Name: "k", Type: "int"},
				},
				HwyCalls: []HwyCall{{Package: "hwy", FuncName: "Load"}},
			},
			want: true,
		},
		{
			name: "Vec→Vec function (not eligible)",
			pf: ParsedFunc{
				Name: "BaseExpVec",
				Params: []Param{
					{Name: "v", Type: "hwy.Vec[T]"},
				},
				Returns: []Param{
					{Name: "", Type: "hwy.Vec[T]"},
				},
				HwyCalls: []HwyCall{{Package: "hwy", FuncName: "Add"}},
			},
			want: false,
		},
		{
			name: "GELU composite (not eligible - has math op name)",
			pf: ParsedFunc{
				Name: "BaseGELUVec",
				Params: []Param{
					{Name: "input", Type: "[]T"},
					{Name: "output", Type: "[]T"},
				},
				HwyCalls: []HwyCall{{Package: "hwy", FuncName: "Load"}},
			},
			want: false,
		},
		{
			name: "slice func without int params (not eligible)",
			pf: ParsedFunc{
				Name: "BaseSoftmax",
				Params: []Param{
					{Name: "input", Type: "[]T"},
					{Name: "output", Type: "[]T"},
				},
				HwyCalls: []HwyCall{{Package: "hwy", FuncName: "Load"}},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsASTCEligible(&tt.pf)
			if got != tt.want {
				t.Errorf("IsASTCEligible() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestASTTranslatorMatMul(t *testing.T) {
	// Find the matmul_base.go file
	matmulPath := filepath.Join("..", "..", "hwy", "contrib", "matmul", "matmul_base.go")
	if _, err := os.Stat(matmulPath); err != nil {
		t.Skipf("matmul_base.go not found at %s: %v", matmulPath, err)
	}

	// Parse the file
	result, err := Parse(matmulPath)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// Find BaseMatMul
	var matmulFunc *ParsedFunc
	for i, pf := range result.Funcs {
		if pf.Name == "BaseMatMul" {
			matmulFunc = &result.Funcs[i]
			break
		}
	}
	if matmulFunc == nil {
		t.Fatal("BaseMatMul not found in parsed functions")
	}

	// Verify it's AST-eligible
	if !IsASTCEligible(matmulFunc) {
		t.Fatal("BaseMatMul should be AST-C-eligible")
	}

	// Translate for NEON f32
	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	translator := NewCASTTranslator(profile, "float32")
	cCode, err := translator.TranslateToC(matmulFunc)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	// Verify function signature with GOAT conventions
	if !strings.Contains(cCode, "void matmul_c_f32_neon(") {
		t.Error("missing function name: matmul_c_f32_neon")
	}
	if !strings.Contains(cCode, "float *a") {
		t.Error("missing float *a parameter")
	}
	if !strings.Contains(cCode, "float *b") {
		t.Error("missing float *b parameter")
	}
	if !strings.Contains(cCode, "float *c") {
		t.Error("missing float *c parameter")
	}
	if !strings.Contains(cCode, "long *pm") {
		t.Error("missing long *pm parameter")
	}
	if !strings.Contains(cCode, "long *pn") {
		t.Error("missing long *pn parameter")
	}
	if !strings.Contains(cCode, "long *pk") {
		t.Error("missing long *pk parameter")
	}

	// Verify int param dereferences
	if !strings.Contains(cCode, "long k = *pk;") {
		t.Error("missing int param dereference: long k = *pk;")
	}
	if !strings.Contains(cCode, "long m = *pm;") {
		t.Error("missing int param dereference: long m = *pm;")
	}
	if !strings.Contains(cCode, "long n = *pn;") {
		t.Error("missing int param dereference: long n = *pn;")
	}

	// Verify hwy.Zero → vdupq_n_f32(0.0f)
	if !strings.Contains(cCode, "vdupq_n_f32(0.0f)") {
		t.Error("missing vdupq_n_f32(0.0f) for hwy.Zero")
	}

	// Verify hwy.Set → vdupq_n_f32
	if !strings.Contains(cCode, "vdupq_n_f32(aip)") {
		t.Error("missing vdupq_n_f32(aip) for hwy.Set")
	}

	// Verify hwy.Load → vld1q_f32
	if !strings.Contains(cCode, "vld1q_f32(") {
		t.Error("missing vld1q_f32 for hwy.Load")
	}

	// Verify hwy.Store → vst1q_f32
	if !strings.Contains(cCode, "vst1q_f32(") {
		t.Error("missing vst1q_f32 for hwy.Store")
	}

	// Verify hwy.MulAdd → vfmaq_f32 with accumulator-first arg order (NEON)
	if !strings.Contains(cCode, "vfmaq_f32(vC, vA, vB)") {
		t.Errorf("missing vfmaq_f32(vC, vA, vB) for hwy.MulAdd with NEON acc-first order\n\nGenerated C:\n%s", cCode)
	}

	// Verify nested for loops are preserved
	forCount := strings.Count(cCode, "for (")
	if forCount < 5 {
		t.Errorf("expected at least 5 for loops (got %d) — matmul needs nested loops", forCount)
	}

	// Verify scalar tail loop is preserved
	if !strings.Contains(cCode, "cRow[j] = 0") {
		t.Error("missing scalar zeroing tail: cRow[j] = 0")
	}
	if !strings.Contains(cCode, "cRow[j] += aip * bRow[j]") {
		t.Error("missing scalar FMA tail: cRow[j] += aip * bRow[j]")
	}

	// Verify pointer alias: cRow := c[i*n : (i+1)*n] → float *cRow = c + i * n;
	if !strings.Contains(cCode, "float *") && !strings.Contains(cCode, "cRow") {
		t.Error("missing pointer alias for cRow")
	}

	// Verify NumLanes → constant 4
	if !strings.Contains(cCode, "= 4") {
		t.Error("missing NumLanes constant (= 4 for NEON f32)")
	}

	t.Logf("Generated C code:\n%s", cCode)
}

func TestASTTranslatorMatMulF64(t *testing.T) {
	// Find the matmul_base.go file
	matmulPath := filepath.Join("..", "..", "hwy", "contrib", "matmul", "matmul_base.go")
	if _, err := os.Stat(matmulPath); err != nil {
		t.Skipf("matmul_base.go not found at %s: %v", matmulPath, err)
	}

	result, err := Parse(matmulPath)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	var matmulFunc *ParsedFunc
	for i, pf := range result.Funcs {
		if pf.Name == "BaseMatMul" {
			matmulFunc = &result.Funcs[i]
			break
		}
	}
	if matmulFunc == nil {
		t.Fatal("BaseMatMul not found")
	}

	// Translate for NEON f64
	profile := GetCProfile("NEON", "float64")
	if profile == nil {
		t.Fatal("NEON float64 profile not found")
	}

	translator := NewCASTTranslator(profile, "float64")
	cCode, err := translator.TranslateToC(matmulFunc)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	// Verify f64-specific intrinsics
	if !strings.Contains(cCode, "matmul_c_f64_neon") {
		t.Error("missing f64 function name")
	}
	if !strings.Contains(cCode, "double *a") {
		t.Error("missing double *a parameter")
	}
	if !strings.Contains(cCode, "vld1q_f64") {
		t.Error("missing vld1q_f64")
	}
	if !strings.Contains(cCode, "vst1q_f64") {
		t.Error("missing vst1q_f64")
	}
	if !strings.Contains(cCode, "vfmaq_f64") {
		t.Error("missing vfmaq_f64")
	}
	if !strings.Contains(cCode, "vdupq_n_f64(0.0)") {
		t.Error("missing vdupq_n_f64(0.0) for hwy.Zero")
	}
	// NEON f64 has 2 lanes
	if !strings.Contains(cCode, "= 2") {
		t.Error("missing NumLanes constant (= 2 for NEON f64)")
	}

	t.Logf("Generated C code:\n%s", cCode)
}

func TestASTTranslatorFmaArgOrder(t *testing.T) {
	// NEON profiles should use acc_first
	neonF32 := GetCProfile("NEON", "float32")
	if neonF32 == nil {
		t.Fatal("NEON float32 profile not found")
	}
	if neonF32.FmaArgOrder != "acc_first" {
		t.Errorf("NEON f32 FmaArgOrder = %q, want %q", neonF32.FmaArgOrder, "acc_first")
	}

	// AVX2 F16 profile should use acc_last
	avx2F16 := GetCProfile("AVX2", "hwy.Float16")
	if avx2F16 == nil {
		t.Fatal("AVX2 Float16 profile not found")
	}
	if avx2F16.FmaArgOrder != "acc_last" {
		t.Errorf("AVX2 f16 FmaArgOrder = %q, want %q", avx2F16.FmaArgOrder, "acc_last")
	}

	// AVX512 F16 profile should use acc_last
	avx512F16 := GetCProfile("AVX512", "hwy.Float16")
	if avx512F16 == nil {
		t.Fatal("AVX512 Float16 profile not found")
	}
	if avx512F16.FmaArgOrder != "acc_last" {
		t.Errorf("AVX512 f16 FmaArgOrder = %q, want %q", avx512F16.FmaArgOrder, "acc_last")
	}
}

// TestCModeMatMulNeonGeneration is an end-to-end test that runs the full
// hwygen -c pipeline on matmul_base.go targeting NEON and verifies the
// generated C files match the expected NEON GOAT-compatible style.
func TestCModeMatMulNeonGeneration(t *testing.T) {
	// Use the real matmul_base.go as input
	matmulPath := filepath.Join("..", "..", "hwy", "contrib", "matmul", "matmul_base.go")
	if _, err := os.Stat(matmulPath); err != nil {
		t.Skipf("matmul_base.go not found at %s: %v", matmulPath, err)
	}

	tmpDir := t.TempDir()

	gen := &Generator{
		InputFile:   matmulPath,
		OutputDir:   tmpDir,
		TargetSpecs: makeTestSpecs(TargetModeC, "neon"),
	}

	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() in CMode failed: %v", err)
	}

	// ---- f32: NEON matmul ----
	f32Path := filepath.Join(tmpDir, "basematmul_c_f32_neon_arm64.c")
	f32Content, err := os.ReadFile(f32Path)
	if err != nil {
		t.Fatalf("Failed to read f32 C file: %v", err)
	}
	f32 := string(f32Content)

	// Verify GOAT header structure
	if !strings.Contains(f32, "#ifndef GOAT_PARSER") {
		t.Error("f32: missing GOAT_PARSER guard")
	}
	if !strings.Contains(f32, "#include <arm_neon.h>") {
		t.Error("f32: missing arm_neon.h include")
	}
	if !strings.Contains(f32, "#endif") {
		t.Error("f32: missing #endif for GOAT_PARSER guard")
	}
	if !strings.Contains(f32, "Generated by hwygen") {
		t.Error("f32: missing hwygen generation comment")
	}

	// Verify function signature matches GOAT conventions:
	// void funcname(float *a, float *b, float *c, long *pm, long *pn, long *pk)
	if !strings.Contains(f32, "void matmul_c_f32_neon(float *a, float *b, float *c, long *pk, long *pm, long *pn)") &&
		!strings.Contains(f32, "void matmul_c_f32_neon(float *a, float *b, float *c, long *pm, long *pn, long *pk)") {
		// Extract the actual signature line for debugging
		for line := range strings.SplitSeq(f32, "\n") {
			if strings.Contains(line, "void matmul_c_f32_neon") {
				t.Errorf("f32: unexpected function signature: %s", strings.TrimSpace(line))
				break
			}
		}
	}

	// Verify int params are dereferenced from pointers (GOAT convention)
	for _, deref := range []string{"long k = *pk;", "long m = *pm;", "long n = *pn;"} {
		if !strings.Contains(f32, deref) {
			t.Errorf("f32: missing param dereference: %s", deref)
		}
	}

	// Verify NEON f32 SIMD intrinsics
	for _, intrinsic := range []string{
		"vld1q_f32",   // vector load
		"vst1q_f32",   // vector store
		"vfmaq_f32",   // fused multiply-add
		"vdupq_n_f32", // broadcast scalar
		"float32x4_t", // vector type
	} {
		if !strings.Contains(f32, intrinsic) {
			t.Errorf("f32: missing NEON intrinsic %q", intrinsic)
		}
	}

	// Verify FMA uses NEON accumulator-first order: vfmaq_f32(acc, a, b)
	if !strings.Contains(f32, "vfmaq_f32(vC, vA, vB)") {
		t.Error("f32: FMA should use NEON acc-first order: vfmaq_f32(vC, vA, vB)")
	}

	// Verify hwy.Zero[T]() → vdupq_n_f32(0.0f) (zero vector broadcast)
	if !strings.Contains(f32, "vdupq_n_f32(0.0f)") {
		t.Error("f32: missing vdupq_n_f32(0.0f) for hwy.Zero")
	}

	// Verify hwy.Set(aip) → vdupq_n_f32(aip) (scalar broadcast)
	if !strings.Contains(f32, "vdupq_n_f32(aip)") {
		t.Error("f32: missing vdupq_n_f32(aip) for hwy.Set")
	}

	// Verify .NumLanes() was replaced with the constant 4 (NEON f32 lane count)
	if !strings.Contains(f32, "long lanes = 4;") {
		t.Error("f32: missing constant lane count: long lanes = 4;")
	}

	// Verify structure: nested for loops (outer rows, zero loop, k-loop, inner SIMD)
	forCount := strings.Count(f32, "for (")
	if forCount < 5 {
		t.Errorf("f32: expected at least 5 for loops for matmul, got %d", forCount)
	}

	// Verify scalar tail loops are preserved
	if !strings.Contains(f32, "cRow[j] = 0;") {
		t.Error("f32: missing scalar zeroing tail: cRow[j] = 0;")
	}
	if !strings.Contains(f32, "cRow[j] += aip * bRow[j];") {
		t.Error("f32: missing scalar FMA tail: cRow[j] += aip * bRow[j];")
	}

	// Verify slice aliases become pointer arithmetic
	if !strings.Contains(f32, "float *cRow = c + i * n;") {
		t.Error("f32: missing pointer alias: float *cRow = c + i * n;")
	}
	if !strings.Contains(f32, "float *bRow = b + p * n;") {
		t.Error("f32: missing pointer alias: float *bRow = b + p * n;")
	}

	// Verify no Go-isms leaked into the C output
	for _, goism := range []string{"range ", "hwy.", "panic(", ":=", "[]"} {
		if strings.Contains(f32, goism) {
			t.Errorf("f32: Go syntax leaked into C output: %q", goism)
		}
	}

	// Verify bounds check panic guards were stripped
	if strings.Contains(f32, "panic") {
		t.Error("f32: panic calls should be stripped from C output")
	}

	// ---- f64: NEON matmul ----
	f64Path := filepath.Join(tmpDir, "basematmul_c_f64_neon_arm64.c")
	f64Content, err := os.ReadFile(f64Path)
	if err != nil {
		t.Fatalf("Failed to read f64 C file: %v", err)
	}
	f64 := string(f64Content)

	// Verify function signature with double pointers
	if !strings.Contains(f64, "void matmul_c_f64_neon(") {
		t.Error("f64: missing function name")
	}
	if !strings.Contains(f64, "double *a") {
		t.Error("f64: missing double *a parameter")
	}

	// Verify NEON f64 intrinsics
	for _, intrinsic := range []string{
		"vld1q_f64",
		"vst1q_f64",
		"vfmaq_f64",
		"vdupq_n_f64",
		"float64x2_t",
	} {
		if !strings.Contains(f64, intrinsic) {
			t.Errorf("f64: missing NEON intrinsic %q", intrinsic)
		}
	}

	// Verify f64 lane count (NEON 128-bit holds 2 doubles)
	if !strings.Contains(f64, "long lanes = 2;") {
		t.Error("f64: missing constant lane count: long lanes = 2;")
	}

	// Verify f64 zero uses 0.0 (not 0.0f)
	if !strings.Contains(f64, "vdupq_n_f64(0.0)") {
		t.Error("f64: missing vdupq_n_f64(0.0) for hwy.Zero")
	}

	// Log both files for inspection
	t.Logf("f32 C file:\n%s", f32)
	t.Logf("f64 C file:\n%s", f64)
}

// TestASTCWrapperGeneration verifies that the wrapper generation for
// AST-translated multi-param functions (like matmul) produces proper
// Go wrappers that follow the GOAT calling convention.
func TestASTCWrapperGeneration(t *testing.T) {
	matmulPath := filepath.Join("..", "..", "hwy", "contrib", "matmul", "matmul_base.go")
	if _, err := os.Stat(matmulPath); err != nil {
		t.Skipf("matmul_base.go not found: %v", err)
	}

	tmpDir := t.TempDir()

	// Parse the matmul_base.go to get the parsed functions
	result, err := Parse(matmulPath)
	if err != nil {
		t.Fatalf("Parse matmul_base.go: %v", err)
	}

	// Find AST-eligible functions
	var astFuncs []ParsedFunc
	for _, pf := range result.Funcs {
		if IsASTCEligible(&pf) {
			astFuncs = append(astFuncs, pf)
		}
	}
	if len(astFuncs) == 0 {
		t.Fatal("no AST-eligible functions found in matmul_base.go")
	}

	// Generate wrappers using the emitCWrappers path
	neonTarget, err := GetTarget("neon")
	if err != nil {
		t.Fatalf("GetTarget(neon): %v", err)
	}
	gen2 := &Generator{
		OutputDir: tmpDir,
	}
	if err := gen2.emitCWrappers(astFuncs, neonTarget, tmpDir); err != nil {
		t.Fatalf("emitCWrappers: %v", err)
	}

	// Read the generated wrapper file
	wrapperPath := filepath.Join(tmpDir, "c_wrappers_neon_arm64.gen.go")
	wrapperContent, err := os.ReadFile(wrapperPath)
	if err != nil {
		t.Fatalf("read wrapper file: %v", err)
	}
	wrapper := string(wrapperContent)
	t.Logf("Generated wrapper:\n%s", wrapper)

	// Verify file structure
	if !strings.Contains(wrapper, "//go:build") {
		t.Error("wrapper: missing build tag")
	}
	if !strings.Contains(wrapper, "import") {
		t.Error("wrapper: missing import")
	}
	if !strings.Contains(wrapper, "\"unsafe\"") {
		t.Error("wrapper: missing unsafe import")
	}

	// Verify f32 wrapper function
	if !strings.Contains(wrapper, "func MatMulCF32(a, b, c []float32, m, n, k int)") {
		t.Error("wrapper: missing MatMulCF32 function signature")
	}
	// Verify zero-dimension guard
	if !strings.Contains(wrapper, "m == 0 || n == 0 || k == 0") {
		t.Error("wrapper: missing zero-dimension guard")
	}
	// Verify int64 conversion
	if !strings.Contains(wrapper, "mVal := int64(m)") {
		t.Error("wrapper: missing mVal int64 conversion")
	}
	if !strings.Contains(wrapper, "nVal := int64(n)") {
		t.Error("wrapper: missing nVal int64 conversion")
	}
	if !strings.Contains(wrapper, "kVal := int64(k)") {
		t.Error("wrapper: missing kVal int64 conversion")
	}
	// Verify unsafe.Pointer calls
	if !strings.Contains(wrapper, "unsafe.Pointer(&a[0])") {
		t.Error("wrapper: missing unsafe.Pointer(&a[0])")
	}
	if !strings.Contains(wrapper, "unsafe.Pointer(&mVal)") {
		t.Error("wrapper: missing unsafe.Pointer(&mVal)")
	}
	// Verify asm function name
	if !strings.Contains(wrapper, "matmul_c_f32_neon(") {
		t.Error("wrapper: missing matmul_c_f32_neon asm call")
	}

	// Verify f64 wrapper
	if !strings.Contains(wrapper, "func MatMulCF64(a, b, c []float64, m, n, k int)") {
		t.Error("wrapper: missing MatMulCF64 function signature")
	}
	if !strings.Contains(wrapper, "matmul_c_f64_neon(") {
		t.Error("wrapper: missing matmul_c_f64_neon asm call")
	}

	// NEON f16 has NativeArithmetic, so its wrapper SHOULD be generated
	if !strings.Contains(wrapper, "MatMulCF16") {
		t.Error("wrapper: f16 wrapper should be generated for NEON f16 (NativeArithmetic=true)")
	}
	// bf16 still uses promoted math without native arithmetic, so no wrapper
	if strings.Contains(wrapper, "MatMulCBF16") {
		t.Error("wrapper: bf16 wrapper should not be generated for promoted-math types")
	}
}

// TestCModeAsmPipeline tests the full -c -asm pipeline: generate C,
// compile with GOAT, and verify the output files.
func TestCModeAsmPipeline(t *testing.T) {
	matmulPath := filepath.Join("..", "..", "hwy", "contrib", "matmul", "matmul_base.go")
	if _, err := os.Stat(matmulPath); err != nil {
		t.Skipf("matmul_base.go not found: %v", err)
	}

	tmpDir := t.TempDir()
	// resolveAsmImportPath walks up from OutputDir looking for go.mod.
	if err := os.WriteFile(filepath.Join(tmpDir, "go.mod"),
		[]byte("module testpkg\n\ngo 1.26\n"), 0644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}

	gen := &Generator{
		InputFile:   matmulPath,
		OutputDir:   tmpDir,
		TargetSpecs: makeTestSpecs(TargetModeAsm, "neon"),
	}

	err := gen.Run()
	if err != nil {
		// GOAT might not be available in all environments
		if strings.Contains(err.Error(), "GOAT") || strings.Contains(err.Error(), "goat") ||
			strings.Contains(err.Error(), "exec:") || strings.Contains(err.Error(), "go tool") {
			t.Skipf("GOAT not available: %v", err)
		}
		t.Fatalf("Generator.Run() with -asm failed: %v", err)
	}

	// Verify assembly files were generated in asm/ subdirectory
	asmDir := filepath.Join(tmpDir, "asm")
	entries, err := os.ReadDir(asmDir)
	if err != nil {
		t.Fatalf("ReadDir(asm/): %v", err)
	}

	var sFiles, goFiles, wrapperFiles []string
	for _, e := range entries {
		name := e.Name()
		switch {
		case strings.HasSuffix(name, ".s"):
			sFiles = append(sFiles, name)
		case (strings.HasSuffix(name, ".go") || strings.HasSuffix(name, ".gen.go")) && strings.Contains(name, "c_wrappers"):
			wrapperFiles = append(wrapperFiles, name)
		case strings.HasSuffix(name, ".go") || strings.HasSuffix(name, ".gen.go"):
			goFiles = append(goFiles, name)
		}
	}

	t.Logf("Generated files: .s=%v, .go=%v, wrappers=%v", sFiles, goFiles, wrapperFiles)

	if len(sFiles) == 0 {
		t.Error("no .s assembly files generated")
	}
	if len(goFiles) == 0 {
		t.Error("no GOAT .go declaration files generated")
	}
	if len(wrapperFiles) == 0 {
		t.Error("no wrapper .go files generated")
	}

	// Verify wrapper file content
	if len(wrapperFiles) > 0 {
		content, err := os.ReadFile(filepath.Join(asmDir, wrapperFiles[0]))
		if err != nil {
			t.Fatalf("read wrapper: %v", err)
		}
		wrapper := string(content)
		if !strings.Contains(wrapper, "MatMulCF32") {
			t.Error("wrapper missing MatMulCF32 function")
		}
		if !strings.Contains(wrapper, "unsafe.Pointer") {
			t.Error("wrapper missing unsafe.Pointer calls")
		}
		t.Logf("Wrapper content:\n%s", wrapper)
	}

	// Verify no C files left behind (cleaned up after GOAT)
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".c") {
			t.Errorf("C file not cleaned up: %s", e.Name())
		}
	}
}

// TestCModeAsmCorrectnessF32 is a full end-to-end correctness test that:
// 1. Generates C from matmul_base.go
// 2. Compiles with GOAT to Go assembly
// 3. Creates a complete Go test package with the assembly
// 4. Runs `go test` to verify the SIMD assembly matches a scalar reference.
func TestCModeAsmCorrectnessF32(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skip("correctness test requires arm64 to execute generated NEON assembly")
	}
	matmulPath := filepath.Join("..", "..", "hwy", "contrib", "matmul", "matmul_base.go")
	if _, err := os.Stat(matmulPath); err != nil {
		t.Skipf("matmul_base.go not found: %v", err)
	}

	// Create temp package directory
	tmpRoot := t.TempDir()
	tmpDir := filepath.Join(tmpRoot, "matmulgen")
	if err := os.MkdirAll(tmpDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// resolveAsmImportPath walks up from OutputDir looking for go.mod.
	if err := os.WriteFile(filepath.Join(tmpRoot, "go.mod"),
		[]byte("module testpkg\n\ngo 1.26\n"), 0644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}

	// Step 1+2+3: Generate C, compile with GOAT, generate wrappers
	gen := &Generator{
		InputFile:   matmulPath,
		OutputDir:   tmpDir,
		TargetSpecs: makeTestSpecs(TargetModeAsm, "neon"),
	}
	if err := gen.Run(); err != nil {
		if strings.Contains(err.Error(), "GOAT") || strings.Contains(err.Error(), "goat") ||
			strings.Contains(err.Error(), "exec:") || strings.Contains(err.Error(), "go tool") {
			t.Skipf("GOAT not available: %v", err)
		}
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	// List generated files for debugging
	asmDir := filepath.Join(tmpDir, "asm")
	entries, _ := os.ReadDir(tmpDir)
	var rootFiles []string
	for _, e := range entries {
		rootFiles = append(rootFiles, e.Name())
	}
	asmEntries, _ := os.ReadDir(asmDir)
	var asmFiles []string
	for _, e := range asmEntries {
		asmFiles = append(asmFiles, e.Name())
	}
	t.Logf("Root files: %v", rootFiles)
	t.Logf("Asm files: %v", asmFiles)

	// Step 4: Write go.mod for the temp package
	// Get the absolute path to go-highway for the replace directive
	hwyRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("get go-highway root: %v", err)
	}
	goModContent := fmt.Sprintf(`module matmulgen

go 1.26

require github.com/ajroetker/go-highway v0.0.0

replace github.com/ajroetker/go-highway => %s
`, hwyRoot)
	if err := os.WriteFile(filepath.Join(tmpDir, "go.mod"), []byte(goModContent), 0644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}

	// Step 5: Write a correctness test file in the asm/ subpackage
	// (since the generator now puts assembly + wrappers in asm/)
	testContent := `package asm

import (
	"fmt"
	"math"
	"testing"
)

func matmulScalar(a, b, c []float32, m, n, k int) {
	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += a[i*k+p] * b[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

func TestMatMulCF32Correctness(t *testing.T) {
	sizes := [][3]int{
		{2, 3, 4},
		{4, 4, 4},
		{8, 8, 8},
		{16, 16, 16},
		{7, 9, 5},
		{33, 17, 11},
	}

	for _, sz := range sizes {
		m, n, k := sz[0], sz[1], sz[2]
		t.Run(fmt.Sprintf("%dx%dx%d", m, n, k), func(t *testing.T) {
			a := make([]float32, m*k)
			b := make([]float32, k*n)
			cRef := make([]float32, m*n)
			cAsm := make([]float32, m*n)

			for i := range a {
				a[i] = float32(i%7) * 0.1
			}
			for i := range b {
				b[i] = float32(i%5) * 0.2
			}

			matmulScalar(a, b, cRef, m, n, k)
			MatMulCF32(a, b, cAsm, m, n, k)

			tol := float32(1e-4) * float32(k)
			for i := range cRef {
				diff := float32(math.Abs(float64(cAsm[i] - cRef[i])))
				if diff > tol {
					t.Errorf("index %d: got %v, want %v (diff=%v, tol=%v)",
						i, cAsm[i], cRef[i], diff, tol)
				}
			}
		})
	}
}
`
	if err := os.WriteFile(filepath.Join(asmDir, "matmul_test.go"), []byte(testContent), 0644); err != nil {
		t.Fatalf("write test file: %v", err)
	}

	// Step 6: Run go mod tidy to generate go.sum
	goBin := filepath.Join(goRoot(), "bin", "go")
	tidyCmd := exec.Command(goBin, "mod", "tidy")
	tidyCmd.Dir = tmpDir
	tidyCmd.Env = append(os.Environ(), "GOWORK=off")
	if tidyOutput, err := tidyCmd.CombinedOutput(); err != nil {
		t.Fatalf("go mod tidy failed: %v\n%s", err, string(tidyOutput))
	}

	// Step 7: Run go test in the asm/ subpackage
	cmd := exec.Command(goBin, "test", "-v", "-count=1", "./asm/...")
	cmd.Dir = tmpDir
	cmd.Env = append(os.Environ(), "GOWORK=off")
	output, err := cmd.CombinedOutput()
	t.Logf("go test output:\n%s", string(output))
	if err != nil {
		t.Fatalf("go test failed: %v\n%s", err, string(output))
	}
}

// goRoot returns the GOROOT for go1.26.
func goRoot() string {
	// Try the same approach as runGOAT: use runtime.GOROOT()
	return runtime.GOROOT()
}

// TestTranslateMatMulKLast verifies that matmul_klast_base.go is correctly
// translated with ReduceSum → vaddvq_f32 and FMA order.
func TestTranslateMatMulKLast(t *testing.T) {
	klastPath := filepath.Join("..", "..", "hwy", "contrib", "matmul", "matmul_klast_base.go")
	if _, err := os.Stat(klastPath); err != nil {
		t.Skipf("matmul_klast_base.go not found: %v", err)
	}

	result, err := Parse(klastPath)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	var klastFunc *ParsedFunc
	for i, pf := range result.Funcs {
		if pf.Name == "BaseMatMulKLast" {
			klastFunc = &result.Funcs[i]
			break
		}
	}
	if klastFunc == nil {
		t.Fatal("BaseMatMulKLast not found")
	}

	if !IsASTCEligible(klastFunc) {
		t.Fatal("BaseMatMulKLast should be AST-C-eligible")
	}

	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	translator := NewCASTTranslator(profile, "float32")
	cCode, err := translator.TranslateToC(klastFunc)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C code:\n%s", cCode)

	// Verify function signature
	if !strings.Contains(cCode, "void matmulklast_c_f32_neon(") {
		t.Error("missing function name: matmulklast_c_f32_neon")
	}

	// Verify ReduceSum → vaddvq_f32
	if !strings.Contains(cCode, "vaddvq_f32(") {
		t.Error("missing vaddvq_f32 for hwy.ReduceSum")
	}

	// Verify FMA uses NEON accumulator-first: vfmaq_f32(acc, a, b)
	if !strings.Contains(cCode, "vfmaq_f32(") {
		t.Error("missing vfmaq_f32 for hwy.MulAdd")
	}

	// Verify hwy.Zero → vdupq_n_f32(0.0f)
	if !strings.Contains(cCode, "vdupq_n_f32(0.0f)") {
		t.Error("missing vdupq_n_f32(0.0f) for hwy.Zero")
	}

	// Verify hwy.Load → vld1q_f32
	if !strings.Contains(cCode, "vld1q_f32(") {
		t.Error("missing vld1q_f32 for hwy.Load")
	}

	// Verify NumLanes → 4
	if !strings.Contains(cCode, "= 4") {
		t.Error("missing NumLanes constant (= 4)")
	}

	// Verify nested loop structure (at least 4 for loops)
	forCount := strings.Count(cCode, "for (")
	if forCount < 4 {
		t.Errorf("expected at least 4 for loops, got %d", forCount)
	}

	// Verify scalar tail: sum0 += a[...] * b[...]
	if !strings.Contains(cCode, "sum0 +=") {
		t.Error("missing scalar tail accumulation for sum0")
	}
}

// TestTranslateTranspose2D verifies that transpose_base.go is correctly
// translated with InterleaveLower/Upper → vzip1q_f32/vzip2q_f32.
func TestTranslateTranspose2D(t *testing.T) {
	transposePath := filepath.Join("..", "..", "hwy", "contrib", "matmul", "transpose_base.go")
	if _, err := os.Stat(transposePath); err != nil {
		t.Skipf("transpose_base.go not found: %v", err)
	}

	result, err := Parse(transposePath)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// Find BaseTranspose2D
	var transposeFunc *ParsedFunc
	for i, pf := range result.Funcs {
		if pf.Name == "BaseTranspose2D" {
			transposeFunc = &result.Funcs[i]
			break
		}
	}
	if transposeFunc == nil {
		t.Fatal("BaseTranspose2D not found")
	}

	if !IsASTCEligible(transposeFunc) {
		t.Fatal("BaseTranspose2D should be AST-C-eligible")
	}

	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	translator := NewCASTTranslator(profile, "float32")
	cCode, err := translator.TranslateToC(transposeFunc)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C code:\n%s", cCode)

	// Verify function signature
	if !strings.Contains(cCode, "void transpose2d_c_f32_neon(") {
		t.Error("missing function name: transpose2d_c_f32_neon")
	}

	// Verify MaxLanes → 4
	if !strings.Contains(cCode, "= 4") {
		t.Error("missing MaxLanes constant (= 4)")
	}

	// BaseTranspose2D calls helper functions (transposeBlockSIMD, transposeEdgesScalar)
	// which are translated as separate function calls. The main function itself
	// doesn't directly contain load/store intrinsics — those are in the helpers.
	// Verify the helper calls are present.
	if !strings.Contains(cCode, "transposeBlockSIMD(") {
		t.Error("missing call to transposeBlockSIMD helper")
	}
	if !strings.Contains(cCode, "transposeEdgesScalar(") {
		t.Error("missing call to transposeEdgesScalar helper")
	}

	// Verify loop structure: nested for loops for block processing
	forCount := strings.Count(cCode, "for (")
	if forCount < 2 {
		t.Errorf("expected at least 2 for loops, got %d", forCount)
	}
}

// TestTranslateIntegerProfiles verifies that NEON integer profiles are correctly
// registered and accessible.
func TestTranslateIntegerProfiles(t *testing.T) {
	tests := []struct {
		elemType string
		wantType string
		wantVec  string
		wantLoad string
	}{
		{"uint64", "unsigned long", "uint64x2_t", "vld1q_u64"},
		{"uint8", "unsigned char", "uint8x16_t", "vld1q_u8"},
		{"uint32", "unsigned int", "uint32x4_t", "vld1q_u32"},
	}

	for _, tt := range tests {
		t.Run(tt.elemType, func(t *testing.T) {
			profile := GetCProfile("NEON", tt.elemType)
			if profile == nil {
				t.Fatalf("NEON %s profile not found", tt.elemType)
			}
			if profile.CType != tt.wantType {
				t.Errorf("CType = %q, want %q", profile.CType, tt.wantType)
			}
			if profile.VecTypes["q"] != tt.wantVec {
				t.Errorf("VecTypes[q] = %q, want %q", profile.VecTypes["q"], tt.wantVec)
			}
			if profile.LoadFn["q"] != tt.wantLoad {
				t.Errorf("LoadFn[q] = %q, want %q", profile.LoadFn["q"], tt.wantLoad)
			}
		})
	}

	// Verify uint64 has popcount helper
	u64Profile := GetCProfile("NEON", "uint64")
	if len(u64Profile.InlineHelpers) == 0 {
		t.Error("uint64 profile should have inline helpers for popcount")
	}
	if !strings.Contains(u64Profile.InlineHelpers[0], "neon_popcnt_u64") {
		t.Error("uint64 popcount helper should contain neon_popcnt_u64")
	}
	if !strings.Contains(u64Profile.InlineHelpers[0], "vcntq_u8") {
		t.Error("uint64 popcount helper should use vcntq_u8")
	}

	// Verify uint8 has BitsFromMask helper
	u8Profile := GetCProfile("NEON", "uint8")
	if len(u8Profile.InlineHelpers) == 0 {
		t.Error("uint8 profile should have inline helpers for BitsFromMask")
	}
	if !strings.Contains(u8Profile.InlineHelpers[0], "neon_bits_from_mask_u8") {
		t.Error("uint8 BitsFromMask helper should contain neon_bits_from_mask_u8")
	}
}

// TestIsASTCEligibleIntegerOps verifies that functions using integer SIMD
// ops are eligible even without int params.
func TestIsASTCEligibleIntegerOps(t *testing.T) {
	tests := []struct {
		name string
		pf   ParsedFunc
		want bool
	}{
		{
			name: "function with And/PopCount (RaBitQ-like)",
			pf: ParsedFunc{
				Name: "BaseBitProduct",
				Params: []Param{
					{Name: "code", Type: "[]uint64"},
					{Name: "q1", Type: "[]uint64"},
				},
				Returns: []Param{
					{Name: "", Type: "uint32"},
				},
				HwyCalls: []HwyCall{
					{Package: "hwy", FuncName: "LoadSlice"},
					{Package: "hwy", FuncName: "And"},
					{Package: "hwy", FuncName: "PopCount"},
					{Package: "hwy", FuncName: "ReduceSum"},
				},
			},
			want: true,
		},
		{
			name: "function with LessThan/BitsFromMask (varint-like)",
			pf: ParsedFunc{
				Name: "BaseFindVarintEnds",
				Params: []Param{
					{Name: "src", Type: "[]byte"},
				},
				Returns: []Param{
					{Name: "", Type: "uint32"},
				},
				HwyCalls: []HwyCall{
					{Package: "hwy", FuncName: "LoadSlice"},
					{Package: "hwy", FuncName: "LessThan"},
					{Package: "hwy", FuncName: "BitsFromMask"},
				},
			},
			want: true,
		},
		{
			name: "slice func without hwy ops (not eligible)",
			pf: ParsedFunc{
				Name: "BaseSimpleSum",
				Params: []Param{
					{Name: "input", Type: "[]float32"},
				},
				HwyCalls: []HwyCall{},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsASTCEligible(&tt.pf)
			if got != tt.want {
				t.Errorf("IsASTCEligible() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestTranslateMixedTypeParams verifies that functions with mixed slice types
// (e.g., []float32 and []uint64) get correct C pointer types.
func TestTranslateMixedTypeParams(t *testing.T) {
	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	// Build a synthetic ParsedFunc with mixed types
	pf := &ParsedFunc{
		Name: "BaseQuantize",
		Params: []Param{
			{Name: "vectors", Type: "[]float32"},
			{Name: "codes", Type: "[]uint64"},
			{Name: "counts", Type: "[]uint32"},
			{Name: "n", Type: "int"},
		},
		HwyCalls: []HwyCall{{Package: "hwy", FuncName: "Load"}},
	}

	translator := NewCASTTranslator(profile, "float32")
	cCode, err := translator.TranslateToC(pf)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C:\n%s", cCode)

	// Verify parameter types
	if !strings.Contains(cCode, "float *vectors") {
		t.Error("missing float *vectors")
	}
	if !strings.Contains(cCode, "unsigned long *codes") {
		t.Error("missing unsigned long *codes")
	}
	if !strings.Contains(cCode, "unsigned int *counts") {
		t.Error("missing unsigned int *counts")
	}
	if !strings.Contains(cCode, "long *pn") {
		t.Error("missing long *pn")
	}
}

// TestTranslateReturnValueAsOutputPointer verifies that Go return values
// are translated to C output pointer parameters.
func TestTranslateReturnValueAsOutputPointer(t *testing.T) {
	profile := GetCProfile("NEON", "uint64")
	if profile == nil {
		t.Fatal("NEON uint64 profile not found")
	}

	// Synthetic function with a return value
	fset := token.NewFileSet()
	src := `package test
import "github.com/ajroetker/go-highway/hwy"
func BaseSum(input []uint64) uint32 {
	acc := hwy.Zero[uint64]()
	return uint32(hwy.ReduceSum(acc))
}
`
	file, err := parser.ParseFile(fset, "test.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	// Extract the function
	var funcDecl *ast.FuncDecl
	for _, decl := range file.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok {
			funcDecl = fd
			break
		}
	}
	if funcDecl == nil {
		t.Fatal("no function found")
	}

	pf := &ParsedFunc{
		Name: "BaseSum",
		Params: []Param{
			{Name: "input", Type: "[]uint64"},
		},
		Returns: []Param{
			{Name: "result", Type: "uint32"},
		},
		Body:     funcDecl.Body,
		HwyCalls: []HwyCall{{Package: "hwy", FuncName: "ReduceSum"}},
	}

	translator := NewCASTTranslator(profile, "uint64")
	cCode, err := translator.TranslateToC(pf)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C:\n%s", cCode)

	// Verify output pointer parameter
	if !strings.Contains(cCode, "long *pout_result") {
		t.Error("missing output pointer: long *pout_result")
	}

	// Verify return statement becomes assignment to output pointer
	if !strings.Contains(cCode, "*pout_result =") {
		t.Error("missing *pout_result = assignment from return statement")
	}
}

// TestTranslateNewHwyOps verifies all newly added hwy operations translate correctly.
func TestTranslateNewHwyOps(t *testing.T) {
	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	tests := []struct {
		name    string
		goCode  string
		wantInC []string
	}{
		{
			name: "ReduceSum",
			goCode: `package test
import "github.com/ajroetker/go-highway/hwy"
func BaseTest(a []float32, n int) {
	v := hwy.Load(a[:])
	s := hwy.ReduceSum(v)
	_ = s
}`,
			wantInC: []string{"vaddvq_f32("},
		},
		{
			name: "InterleaveLower_Upper",
			goCode: `package test
import "github.com/ajroetker/go-highway/hwy"
func BaseTest(a []float32, n int) {
	v1 := hwy.Load(a[:])
	v2 := hwy.Load(a[:])
	lo := hwy.InterleaveLower(v1, v2)
	hi := hwy.InterleaveUpper(v1, v2)
	_ = lo
	_ = hi
}`,
			wantInC: []string{"vzip1q_f32(", "vzip2q_f32("},
		},
		{
			name: "LessThan_IfThenElse",
			goCode: `package test
import "github.com/ajroetker/go-highway/hwy"
func BaseTest(a []float32, n int) {
	v := hwy.Load(a[:])
	zero := hwy.Zero[float32]()
	mask := hwy.LessThan(v, zero)
	result := hwy.IfThenElse(mask, v, zero)
	_ = result
}`,
			wantInC: []string{"vcltq_f32(", "vbslq_f32("},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Parse the Go code
			fset := token.NewFileSet()
			file, err := parser.ParseFile(fset, "test.go", tt.goCode, parser.SkipObjectResolution)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}

			var funcDecl *ast.FuncDecl
			for _, decl := range file.Decls {
				if fd, ok := decl.(*ast.FuncDecl); ok {
					funcDecl = fd
					break
				}
			}
			if funcDecl == nil {
				t.Fatal("no function found")
			}

			// Build a ParsedFunc
			pf := &ParsedFunc{
				Name: "BaseTest",
				Params: []Param{
					{Name: "a", Type: "[]float32"},
					{Name: "n", Type: "int"},
				},
				Body:     funcDecl.Body,
				HwyCalls: []HwyCall{{Package: "hwy", FuncName: "Load"}},
			}

			translator := NewCASTTranslator(profile, "float32")
			cCode, err := translator.TranslateToC(pf)
			if err != nil {
				t.Fatalf("TranslateToC failed: %v", err)
			}

			t.Logf("Generated C:\n%s", cCode)

			for _, want := range tt.wantInC {
				if !strings.Contains(cCode, want) {
					t.Errorf("missing expected intrinsic %q in generated C", want)
				}
			}
		})
	}
}

// TestTranslateTypeConversions verifies that Go type conversions like
// uint64(x) and float64(x) are translated to C casts.
func TestTranslateTypeConversions(t *testing.T) {
	profile := GetCProfile("NEON", "uint64")
	if profile == nil {
		t.Fatal("NEON uint64 profile not found")
	}

	fset := token.NewFileSet()
	src := `package test
import "github.com/ajroetker/go-highway/hwy"
func BaseTest(a []uint64, n int) {
	v := hwy.Load(a[:])
	s := hwy.ReduceSum(v)
	x := uint64(s)
	_ = x
}
`
	file, err := parser.ParseFile(fset, "test.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	var funcDecl *ast.FuncDecl
	for _, decl := range file.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok {
			funcDecl = fd
			break
		}
	}

	pf := &ParsedFunc{
		Name: "BaseTest",
		Params: []Param{
			{Name: "a", Type: "[]uint64"},
			{Name: "n", Type: "int"},
		},
		Body:     funcDecl.Body,
		HwyCalls: []HwyCall{{Package: "hwy", FuncName: "Load"}},
	}

	translator := NewCASTTranslator(profile, "uint64")
	cCode, err := translator.TranslateToC(pf)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C:\n%s", cCode)

	// Verify type conversion: uint64(s) → (unsigned long)(s)
	if !strings.Contains(cCode, "(unsigned long)(") {
		t.Error("missing C cast for uint64() type conversion")
	}
}

// TestTranslateInlineHelpers verifies that inline helpers from the profile
// are emitted in the generated C file.
func TestTranslateInlineHelpers(t *testing.T) {
	profile := GetCProfile("NEON", "uint64")
	if profile == nil {
		t.Fatal("NEON uint64 profile not found")
	}

	pf := &ParsedFunc{
		Name: "BaseTest",
		Params: []Param{
			{Name: "a", Type: "[]uint64"},
			{Name: "n", Type: "int"},
		},
		HwyCalls: []HwyCall{{Package: "hwy", FuncName: "PopCount"}},
	}

	emitter := NewCEmitter("test", "uint64", NEONTarget())
	emitter.profile = profile
	tmpDir := t.TempDir()
	cFile, err := emitter.EmitASTTranslatedC(pf, tmpDir)
	if err != nil {
		t.Fatalf("EmitASTTranslatedC failed: %v", err)
	}

	content, err := os.ReadFile(cFile)
	if err != nil {
		t.Fatalf("read C file: %v", err)
	}

	cContent := string(content)
	t.Logf("Generated C file:\n%s", cContent)

	// Verify inline helper is emitted before the main function
	if !strings.Contains(cContent, "static inline uint64x2_t neon_popcnt_u64") {
		t.Error("missing inline helper: neon_popcnt_u64")
	}
	if !strings.Contains(cContent, "vcntq_u8") {
		t.Error("missing vcntq_u8 in popcount helper")
	}
	if !strings.Contains(cContent, "vpaddlq_u8") {
		t.Error("missing vpaddlq_u8 in popcount helper")
	}

	// Verify the helper appears before the main function
	helperIdx := strings.Index(cContent, "neon_popcnt_u64")
	funcIdx := strings.Index(cContent, "void ")
	if helperIdx > funcIdx {
		t.Error("inline helper should appear before the main function")
	}
}

// TestBenchmarkASTvsHandwritten benchmarks the AST-generated NEON matmul
// assembly against the existing hand-written NEON matmul assembly.
//
// The hand-written version uses a j-outer/k-inner loop (accumulates in
// register, stores C once). The AST-translated version faithfully mirrors
// BaseMatMul's k-outer/j-inner loop (loads/stores C every k iteration).
func TestBenchmarkASTvsHandwritten(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skip("benchmark test requires arm64 to execute generated NEON assembly")
	}
	matmulPath := filepath.Join("..", "..", "hwy", "contrib", "matmul", "matmul_base.go")
	if _, err := os.Stat(matmulPath); err != nil {
		t.Skipf("matmul_base.go not found: %v", err)
	}

	tmpRoot := t.TempDir()
	tmpDir := filepath.Join(tmpRoot, "matmulbench")
	if err := os.MkdirAll(tmpDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// resolveAsmImportPath walks up from OutputDir looking for go.mod.
	if err := os.WriteFile(filepath.Join(tmpRoot, "go.mod"),
		[]byte("module testpkg\n\ngo 1.26\n"), 0644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}

	// Step 1: Generate AST-translated C and compile with GOAT
	gen := &Generator{
		InputFile:   matmulPath,
		OutputDir:   tmpDir,
		TargetSpecs: makeTestSpecs(TargetModeAsm, "neon"),
	}
	if err := gen.Run(); err != nil {
		if strings.Contains(err.Error(), "GOAT") || strings.Contains(err.Error(), "goat") ||
			strings.Contains(err.Error(), "exec:") || strings.Contains(err.Error(), "go tool") {
			t.Skipf("GOAT not available: %v", err)
		}
		t.Fatalf("AST generation failed: %v", err)
	}

	// Step 2: Copy the pre-compiled hand-written NEON assembly into the asm/
	// subdirectory alongside the AST-generated files.
	// matmul_neon_f16_arm64.{s,go} contains matmul_neon_f32 alongside f16/f64.
	genAsmDir := filepath.Join(tmpDir, "asm")
	srcAsmDir := filepath.Join("..", "..", "hwy", "contrib", "matmul", "asm")
	for _, suffix := range []string{".s", ".go"} {
		src := filepath.Join(srcAsmDir, "matmul_neon_f16_arm64"+suffix)
		dst := filepath.Join(genAsmDir, "matmul_neon_f16_arm64"+suffix)
		data, err := os.ReadFile(src)
		if err != nil {
			t.Fatalf("read %s: %v", src, err)
		}
		if err := os.WriteFile(dst, data, 0644); err != nil {
			t.Fatalf("write %s: %v", dst, err)
		}
	}

	// Step 3: Write a wrapper for the hand-written asm function in the asm/ subdir
	handwrittenWrapper := `package asm

import "unsafe"

// MatMulHandwrittenF32 wraps the hand-written NEON matmul assembly.
func MatMulHandwrittenF32(a, b, c []float32, m, n, k int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	mVal := int64(m)
	nVal := int64(n)
	kVal := int64(k)
	matmul_neon_f32(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		unsafe.Pointer(&c[0]),
		unsafe.Pointer(&mVal),
		unsafe.Pointer(&nVal),
		unsafe.Pointer(&kVal),
	)
}
`
	if err := os.WriteFile(filepath.Join(genAsmDir, "handwritten_wrapper.go"), []byte(handwrittenWrapper), 0644); err != nil {
		t.Fatalf("write handwritten wrapper: %v", err)
	}

	// Step 4: Write go.mod
	// Get the absolute path to go-highway for the replace directive
	hwyRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("get go-highway root: %v", err)
	}
	goModContent := fmt.Sprintf(`module matmulbench

go 1.26

require github.com/ajroetker/go-highway v0.0.0

replace github.com/ajroetker/go-highway => %s
`, hwyRoot)
	if err := os.WriteFile(filepath.Join(tmpDir, "go.mod"), []byte(goModContent), 0644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}

	// Step 5: Write the benchmark test in the asm/ subpackage
	benchContent := `package asm

import (
	"fmt"
	"math/rand"
	"testing"
)

func BenchmarkMatMulF32(b *testing.B) {
	sizes := []int{64, 128, 256}

	for _, size := range sizes {
		m, n, k := size, size, size
		a := make([]float32, m*k)
		bMat := make([]float32, k*n)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		flops := float64(2*m*n*k) / 1e9

		b.Run(fmt.Sprintf("ASTGenerated/%d", size), func(b *testing.B) {
			c := make([]float32, m*n)
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulCF32(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(fmt.Sprintf("Handwritten/%d", size), func(b *testing.B) {
			c := make([]float32, m*n)
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulHandwrittenF32(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})

		b.Run(fmt.Sprintf("Scalar/%d", size), func(b *testing.B) {
			c := make([]float32, m*n)
			b.SetBytes(int64((m*k + k*n + m*n) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matmulScalar(a, bMat, c, m, n, k)
			}
			b.StopTimer()
			elapsed := b.Elapsed().Seconds()
			gflops := flops * float64(b.N) / elapsed
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

func matmulScalar(a, b, c []float32, m, n, k int) {
	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += a[i*k+p] * b[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
}
`
	if err := os.WriteFile(filepath.Join(genAsmDir, "bench_test.go"), []byte(benchContent), 0644); err != nil {
		t.Fatalf("write bench test: %v", err)
	}

	// List all files for debugging
	entries, _ := os.ReadDir(tmpDir)
	var rootFiles []string
	for _, e := range entries {
		rootFiles = append(rootFiles, e.Name())
	}
	asmEntries, _ := os.ReadDir(genAsmDir)
	var asmFiles []string
	for _, e := range asmEntries {
		asmFiles = append(asmFiles, e.Name())
	}
	t.Logf("Root files: %v", rootFiles)
	t.Logf("Asm files: %v", asmFiles)

	// Step 6a: Run go mod tidy to generate go.sum
	goBin := filepath.Join(goRoot(), "bin", "go")
	tidyCmd := exec.Command(goBin, "mod", "tidy")
	tidyCmd.Dir = tmpDir
	tidyCmd.Env = append(os.Environ(), "GOWORK=off")
	if tidyOut, err := tidyCmd.CombinedOutput(); err != nil {
		t.Fatalf("go mod tidy failed: %v\n%s", err, string(tidyOut))
	}

	// Step 6b: Run the benchmark in asm/ subpackage
	cmd := exec.Command(goBin, "test", "-bench=BenchmarkMatMulF32", "-benchmem", "-count=1", "./asm/...")
	cmd.Dir = tmpDir
	cmd.Env = append(os.Environ(), "GOWORK=off")
	output, err := cmd.CombinedOutput()
	t.Logf("Benchmark output:\n%s", string(output))
	if err != nil {
		t.Fatalf("benchmark failed: %v\n%s", err, string(output))
	}
}

// TestTranslateRaBitQBitProduct verifies that rabitq_base.go's BaseBitProduct
// translates correctly to NEON C using the uint64 profile.
func TestTranslateRaBitQBitProduct(t *testing.T) {
	rabitqPath := filepath.Join("..", "..", "hwy", "contrib", "rabitq", "rabitq_base.go")
	if _, err := os.Stat(rabitqPath); err != nil {
		t.Skipf("rabitq_base.go not found: %v", err)
	}

	result, err := Parse(rabitqPath)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	var bitProductFunc *ParsedFunc
	for i, pf := range result.Funcs {
		if pf.Name == "BaseBitProduct" {
			bitProductFunc = &result.Funcs[i]
			break
		}
	}
	if bitProductFunc == nil {
		t.Fatal("BaseBitProduct not found")
	}

	if !IsASTCEligible(bitProductFunc) {
		t.Fatal("BaseBitProduct should be AST-C-eligible")
	}

	profile := GetCProfile("NEON", "uint64")
	if profile == nil {
		t.Fatal("NEON uint64 profile not found")
	}

	translator := NewCASTTranslator(profile, "uint64")
	cCode, err := translator.TranslateToC(bitProductFunc)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C code:\n%s", cCode)

	// Verify function signature has the uint64 slice params and output pointer
	if !strings.Contains(cCode, "void bitproduct_c_u64_neon(") {
		t.Error("missing function name: bitproduct_c_u64_neon")
	}
	if !strings.Contains(cCode, "unsigned long *code") {
		t.Error("missing 'unsigned long *code' param")
	}
	if !strings.Contains(cCode, "long *pout_result") {
		t.Error("missing output pointer 'long *pout_result'")
	}
	if !strings.Contains(cCode, "long *plen_code") {
		t.Error("missing length param 'long *plen_code'")
	}

	// Verify vld1q_u64_x4 multi-load from Load4 unrolling
	if !strings.Contains(cCode, "vld1q_u64_x4(") {
		t.Error("missing vld1q_u64_x4 for hwy.Load4 (4x unrolling)")
	}

	// Verify NEON intrinsics for And (used in both main loop and remainder)
	if !strings.Contains(cCode, "vandq_u64(") {
		t.Error("missing vandq_u64 for hwy.And")
	}

	// Verify deferred popcount accumulation in the main (Load4) loop:
	// - neon_popcnt_u64_to_u32: partial popcount returning uint32x4_t
	// - vaddq_u32: vector accumulation instead of horizontal reduction
	// - _pacc_: vector accumulator variables
	// - vaddvq_u32: deferred horizontal reduction AFTER the loop
	if !strings.Contains(cCode, "neon_popcnt_u64_to_u32(") {
		t.Error("missing neon_popcnt_u64_to_u32 for deferred popcount accumulation")
	}
	if !strings.Contains(cCode, "vaddq_u32(") {
		t.Error("missing vaddq_u32 for vector accumulation in main loop")
	}
	if !strings.Contains(cCode, "_pacc_") {
		t.Error("missing _pacc_ vector accumulator variables")
	}
	if !strings.Contains(cCode, "vaddvq_u32(") {
		t.Error("missing vaddvq_u32 for deferred horizontal reduction after loop")
	}

	// Both loops share accumulators: 16 in main loop + 4 in remainder = 20
	partialCount := strings.Count(cCode, "neon_popcnt_u64_to_u32(")
	if partialCount != 20 {
		t.Errorf("expected 20 neon_popcnt_u64_to_u32 calls (16 main + 4 remainder), got %d", partialCount)
	}

	// Shared accumulators: only ONE set of declarations and ONE set of reductions
	declCount := strings.Count(cCode, "uint32x4_t _pacc_")
	reduceCount := strings.Count(cCode, "vaddvq_u32(_pacc_")
	if declCount != reduceCount {
		t.Errorf("shared accums: %d declarations vs %d reductions (should match)", declCount, reduceCount)
	}

	// Verify scalar tail uses __builtin_popcountll
	if !strings.Contains(cCode, "__builtin_popcountll(") {
		t.Error("missing __builtin_popcountll for bits.OnesCount64")
	}

	// Verify len(code) is mapped to len_code
	if !strings.Contains(cCode, "len_code") {
		t.Error("missing len_code variable for len(code)")
	}

	// Verify return value is written to output pointer
	if !strings.Contains(cCode, "*pout_result =") {
		t.Error("missing return value output: *pout_result =")
	}
}

// TestTranslateDeferredPopCountAccum verifies the deferred popcount accumulation
// optimization with a synthetic function containing the target pattern.
func TestTranslateDeferredPopCountAccum(t *testing.T) {
	// Synthetic Go source with sum += uint64(ReduceSum(PopCount(And(...)))) in a Load4 loop
	src := `package testpkg

import "github.com/ajroetker/go-highway/hwy"

func BaseTestPopCount(a, b []uint64) uint64 {
	var sum0, sum1, sum2, sum3 uint64
	lanes := hwy.Zero[uint64]().NumLanes()
	n := len(a)
	stride := lanes * 4
	var i int
	for i = 0; i+stride <= n; i += stride {
		aVec0, aVec1, aVec2, aVec3 := hwy.Load4(a[i:])
		bVec0, bVec1, bVec2, bVec3 := hwy.Load4(b[i:])
		sum0 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(aVec0, bVec0))))
		sum1 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(aVec1, bVec1))))
		sum2 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(aVec2, bVec2))))
		sum3 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(aVec3, bVec3))))
	}
	for i+lanes <= n {
		aVec := hwy.LoadSlice(a[i:])
		bVec := hwy.LoadSlice(b[i:])
		sum0 += uint64(hwy.ReduceSum(hwy.PopCount(hwy.And(aVec, bVec))))
		i += lanes
	}
	return sum0 + sum1 + sum2 + sum3
}
`
	tmpFile := filepath.Join(t.TempDir(), "test_popcount.go")
	if err := os.WriteFile(tmpFile, []byte(src), 0644); err != nil {
		t.Fatalf("write test file: %v", err)
	}

	result, err := Parse(tmpFile)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	var fn *ParsedFunc
	for i, pf := range result.Funcs {
		if pf.Name == "BaseTestPopCount" {
			fn = &result.Funcs[i]
			break
		}
	}
	if fn == nil {
		t.Fatal("BaseTestPopCount not found")
	}

	profile := GetCProfile("NEON", "uint64")
	if profile == nil {
		t.Fatal("NEON uint64 profile not found")
	}

	translator := NewCASTTranslator(profile, "uint64")
	cCode, err := translator.TranslateToC(fn)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C code:\n%s", cCode)

	// Main loop (Load4) should use deferred accumulation
	if !strings.Contains(cCode, "uint32x4_t _pacc_") {
		t.Error("missing uint32x4_t _pacc_ accumulator declarations")
	}
	if !strings.Contains(cCode, "neon_popcnt_u64_to_u32(") {
		t.Error("missing neon_popcnt_u64_to_u32 partial popcount in main loop")
	}
	if !strings.Contains(cCode, "vaddq_u32(_pacc_") {
		t.Error("missing vaddq_u32 vector accumulation in main loop")
	}

	// Post-loop finalization should use vaddvq_u32
	if !strings.Contains(cCode, "vaddvq_u32(_pacc_") {
		t.Error("missing vaddvq_u32 post-loop reduction")
	}

	// Both loops should use deferred accumulation — no vaddvq_u64 in either
	partialCount := strings.Count(cCode, "neon_popcnt_u64_to_u32(")
	if partialCount < 5 {
		t.Errorf("expected >= 5 neon_popcnt_u64_to_u32 calls (4 main + 1 remainder), got %d", partialCount)
	}
}

// TestTranslateRaBitQQuantizeVectors verifies that rabitq_base.go's BaseQuantizeVectors
// translates correctly to NEON C using the float32 profile with mixed-type params.
func TestTranslateRaBitQQuantizeVectors(t *testing.T) {
	rabitqPath := filepath.Join("..", "..", "hwy", "contrib", "rabitq", "rabitq_base.go")
	if _, err := os.Stat(rabitqPath); err != nil {
		t.Skipf("rabitq_base.go not found: %v", err)
	}

	result, err := Parse(rabitqPath)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	var fn *ParsedFunc
	for i, pf := range result.Funcs {
		if pf.Name == "BaseQuantizeVectors" {
			fn = &result.Funcs[i]
			break
		}
	}
	if fn == nil {
		t.Fatal("BaseQuantizeVectors not found")
	}

	if !IsASTCEligible(fn) {
		t.Fatal("BaseQuantizeVectors should be AST-C-eligible")
	}

	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	translator := NewCASTTranslator(profile, "float32")
	cCode, err := translator.TranslateToC(fn)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C code:\n%s", cCode)

	// Verify function signature with mixed-type params
	if !strings.Contains(cCode, "void quantizevectors_c_f32_neon(") {
		t.Error("missing function name")
	}
	if !strings.Contains(cCode, "float *unitVectors") {
		t.Error("missing float *unitVectors param")
	}
	if !strings.Contains(cCode, "unsigned long *codes") {
		t.Error("missing unsigned long *codes param")
	}
	if !strings.Contains(cCode, "unsigned int *codeCounts") {
		t.Error("missing unsigned int *codeCounts param")
	}

	// Verify mixed-type slice reslicing: codes → unsigned long *, not float *
	if !strings.Contains(cCode, "unsigned long *code = codes") {
		t.Error("code slice should be typed as unsigned long *, not float *")
	}
	if !strings.Contains(cCode, "float *vec = unitVectors") {
		t.Error("vec slice should be typed as float *")
	}

	// Verify GetLane with variable index uses store-to-stack pattern
	if !strings.Contains(cCode, "volatile float _getlane_buf") {
		t.Error("missing volatile store-to-stack buffer for variable-index GetLane")
	}
	if !strings.Contains(cCode, "_getlane_buf[j]") {
		t.Error("missing variable-index access to _getlane_buf")
	}

	// Verify getSignBit is inlined as float_to_bits >> 31
	if !strings.Contains(cCode, "float_to_bits(") {
		t.Error("missing float_to_bits for getSignBit inlining")
	}
	if !strings.Contains(cCode, ">> 31") {
		t.Error("missing >> 31 for sign bit extraction")
	}

	// Verify NEON SIMD intrinsics for the vectorized path
	if !strings.Contains(cCode, "vld1q_f32(") {
		t.Error("missing vld1q_f32 for float32 SIMD load")
	}
	if !strings.Contains(cCode, "vcltq_f32(") {
		t.Error("missing vcltq_f32 for LessThan")
	}
	if !strings.Contains(cCode, "vbslq_f32(") {
		t.Error("missing vbslq_f32 for IfThenElse")
	}
	if !strings.Contains(cCode, "vmulq_f32(") {
		t.Error("missing vmulq_f32 for Mul")
	}
	if !strings.Contains(cCode, "vaddvq_f32(") {
		t.Error("missing vaddvq_f32 for ReduceSum")
	}

	// Verify scalar popcount for bit counting
	if !strings.Contains(cCode, "__builtin_popcountll(") {
		t.Error("missing __builtin_popcountll for bits.OnesCount64")
	}
}

// TestTranslateVarintFindEnds verifies that varint_base.go's BaseFindVarintEnds
// translates correctly to NEON C using the uint8 profile.
func TestTranslateVarintFindEnds(t *testing.T) {
	varintPath := filepath.Join("..", "..", "hwy", "contrib", "varint", "varint_base.go")
	if _, err := os.Stat(varintPath); err != nil {
		t.Skipf("varint_base.go not found: %v", err)
	}

	result, err := Parse(varintPath)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	var findEndsFunc *ParsedFunc
	for i, pf := range result.Funcs {
		if pf.Name == "BaseFindVarintEnds" {
			findEndsFunc = &result.Funcs[i]
			break
		}
	}
	if findEndsFunc == nil {
		t.Fatal("BaseFindVarintEnds not found")
	}

	if !IsASTCEligible(findEndsFunc) {
		t.Fatal("BaseFindVarintEnds should be AST-C-eligible")
	}

	profile := GetCProfile("NEON", "uint8")
	if profile == nil {
		t.Fatal("NEON uint8 profile not found")
	}

	translator := NewCASTTranslator(profile, "uint8")
	cCode, err := translator.TranslateToC(findEndsFunc)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C code:\n%s", cCode)

	// Verify function signature
	if !strings.Contains(cCode, "void findvarintends_c_u8_neon(") {
		t.Error("missing function name: findvarintends_c_u8_neon")
	}
	if !strings.Contains(cCode, "unsigned char *src") {
		t.Error("missing 'unsigned char *src' param")
	}

	// Verify NEON intrinsics for LessThan and BitsFromMask
	if !strings.Contains(cCode, "vcltq_u8(") {
		t.Error("missing vcltq_u8 for hwy.LessThan")
	}
	if !strings.Contains(cCode, "neon_bits_from_mask_u8(") {
		t.Error("missing neon_bits_from_mask_u8 for hwy.BitsFromMask")
	}

	// Verify hwy.Set → vdupq_n_u8
	if !strings.Contains(cCode, "vdupq_n_u8(") {
		t.Error("missing vdupq_n_u8 for hwy.Set")
	}

	// Verify hwy.LoadSlice → vld1q_u8
	if !strings.Contains(cCode, "vld1q_u8(") {
		t.Error("missing vld1q_u8 for hwy.LoadSlice")
	}

	// Verify return value output pointer
	if !strings.Contains(cCode, "*pout_result =") || !strings.Contains(cCode, "long *pout_result") {
		t.Error("missing return value output pointer")
	}
}

// TestBenchmarkRaBitQASTvsHandwritten benchmarks the AST-generated NEON rabitq
// assembly against the existing hand-written NEON rabitq assembly.
func TestBenchmarkRaBitQASTvsHandwritten(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skip("correctness test requires arm64 to execute generated NEON assembly")
	}
	rabitqPath := filepath.Join("..", "..", "hwy", "contrib", "rabitq", "rabitq_base.go")
	if _, err := os.Stat(rabitqPath); err != nil {
		t.Skipf("rabitq_base.go not found: %v", err)
	}

	tmpDir := filepath.Join(t.TempDir(), "rabitqbench")
	if err := os.MkdirAll(tmpDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	// Step 1: Generate AST-translated C for BaseBitProduct only, then compile with GOAT
	result, err := Parse(rabitqPath)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}
	var bitProductFunc *ParsedFunc
	for i, pf := range result.Funcs {
		if pf.Name == "BaseBitProduct" {
			bitProductFunc = &result.Funcs[i]
			break
		}
	}
	if bitProductFunc == nil {
		t.Fatal("BaseBitProduct not found in rabitq_base.go")
	}

	target, err := GetTarget("neon")
	if err != nil {
		t.Fatalf("GetTarget: %v", err)
	}
	profile := GetCProfile(target.Name, "uint64")
	if profile == nil {
		t.Fatal("NEON uint64 profile not found")
	}
	emitter := NewCEmitter(result.PackageName, "uint64", target)
	emitter.profile = profile
	cFile, err := emitter.EmitASTTranslatedC(bitProductFunc, tmpDir)
	if err != nil {
		t.Fatalf("EmitASTTranslatedC failed: %v", err)
	}

	// Compile with GOAT
	if err := runGOAT(cFile, profile); err != nil {
		if strings.Contains(err.Error(), "exec:") || strings.Contains(err.Error(), "go tool") {
			t.Skipf("GOAT not available: %v", err)
		}
		t.Fatalf("GOAT compile failed: %v", err)
	}
	// Clean up C file (Go build doesn't like it)
	os.Remove(cFile)
	os.Remove(strings.TrimSuffix(cFile, ".c") + ".o")

	// Generate wrapper for AST-translated function
	{
		var buf bytes.Buffer
		fmt.Fprintf(&buf, "package rabitqbench\n\nimport \"unsafe\"\n\n")
		emitASTCWrapperFunc(&buf, bitProductFunc, "uint64", "neon")
		if err := os.WriteFile(filepath.Join(tmpDir, "ast_wrapper.go"), buf.Bytes(), 0644); err != nil {
			t.Fatalf("write ast wrapper: %v", err)
		}
	}

	// Step 2: Copy the hand-written NEON assembly from asm/ directory
	asmDir := filepath.Join("..", "..", "hwy", "contrib", "rabitq", "asm")
	for _, suffix := range []string{".s", ".go"} {
		src := filepath.Join(asmDir, "rabitq_neon_arm64"+suffix)
		dst := filepath.Join(tmpDir, "rabitq_neon_arm64"+suffix)
		data, err := os.ReadFile(src)
		if err != nil {
			t.Fatalf("read %s: %v", src, err)
		}
		if suffix == ".go" {
			data = []byte(strings.Replace(string(data), "package asm", "package rabitqbench", 1))
		}
		if err := os.WriteFile(dst, data, 0644); err != nil {
			t.Fatalf("write %s: %v", dst, err)
		}
	}

	// Step 3: Write a wrapper for the hand-written asm function
	handwrittenWrapper := `package rabitqbench

import "unsafe"

// BitProductHandwritten wraps the hand-written NEON rabitq assembly.
func BitProductHandwritten(code, q1, q2, q3, q4 []uint64) uint32 {
	if len(code) == 0 {
		return 0
	}
	l := int64(len(code))
	var sum uint64
	rabitq_bit_product_neon(
		unsafe.Pointer(&code[0]),
		unsafe.Pointer(&q1[0]),
		unsafe.Pointer(&q2[0]),
		unsafe.Pointer(&q3[0]),
		unsafe.Pointer(&q4[0]),
		unsafe.Pointer(&sum),
		unsafe.Pointer(&l),
	)
	return uint32(sum)
}
`
	if err := os.WriteFile(filepath.Join(tmpDir, "handwritten_wrapper.go"), []byte(handwrittenWrapper), 0644); err != nil {
		t.Fatalf("write handwritten wrapper: %v", err)
	}

	// Step 4: Write go.mod
	goModContent := `module rabitqbench

go 1.26
`
	if err := os.WriteFile(filepath.Join(tmpDir, "go.mod"), []byte(goModContent), 0644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}

	// Step 5: Write the benchmark test
	benchContent := `package rabitqbench

import (
	"fmt"
	"math/bits"
	"math/rand"
	"testing"
)

// bitProductScalar is the pure Go reference implementation.
func bitProductScalar(code, q1, q2, q3, q4 []uint64) uint32 {
	var sum1, sum2, sum4, sum8 uint64
	for i := range code {
		sum1 += uint64(bits.OnesCount64(code[i] & q1[i]))
		sum2 += uint64(bits.OnesCount64(code[i] & q2[i]))
		sum4 += uint64(bits.OnesCount64(code[i] & q3[i]))
		sum8 += uint64(bits.OnesCount64(code[i] & q4[i]))
	}
	return uint32(sum1 + (sum2 << 1) + (sum4 << 2) + (sum8 << 3))
}

func TestBitProductCorrectness(t *testing.T) {
	sizes := []int{1, 2, 4, 8, 16, 32, 64}
	rng := rand.New(rand.NewSource(42))
	for _, n := range sizes {
		code := make([]uint64, n)
		q1 := make([]uint64, n)
		q2 := make([]uint64, n)
		q3 := make([]uint64, n)
		q4 := make([]uint64, n)
		for i := range code {
			code[i] = rng.Uint64()
			q1[i] = rng.Uint64()
			q2[i] = rng.Uint64()
			q3[i] = rng.Uint64()
			q4[i] = rng.Uint64()
		}
		expected := bitProductScalar(code, q1, q2, q3, q4)
		gotAST := BitProductCU64(code, q1, q2, q3, q4)
		gotHW := BitProductHandwritten(code, q1, q2, q3, q4)
		if gotAST != expected {
			t.Errorf("AST n=%d: got %d, want %d", n, gotAST, expected)
		}
		if gotHW != expected {
			t.Errorf("Handwritten n=%d: got %d, want %d", n, gotHW, expected)
		}
	}
}

func BenchmarkBitProduct(b *testing.B) {
	sizes := []int{4, 16, 64, 256}

	for _, n := range sizes {
		rng := rand.New(rand.NewSource(42))
		code := make([]uint64, n)
		q1 := make([]uint64, n)
		q2 := make([]uint64, n)
		q3 := make([]uint64, n)
		q4 := make([]uint64, n)
		for i := range code {
			code[i] = rng.Uint64()
			q1[i] = rng.Uint64()
			q2[i] = rng.Uint64()
			q3[i] = rng.Uint64()
			q4[i] = rng.Uint64()
		}

		b.Run(fmt.Sprintf("ASTGenerated/%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 5 * 8))
			for i := 0; i < b.N; i++ {
				BitProductCU64(code, q1, q2, q3, q4)
			}
		})

		b.Run(fmt.Sprintf("Handwritten/%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 5 * 8))
			for i := 0; i < b.N; i++ {
				BitProductHandwritten(code, q1, q2, q3, q4)
			}
		})

		b.Run(fmt.Sprintf("Scalar/%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 5 * 8))
			for i := 0; i < b.N; i++ {
				bitProductScalar(code, q1, q2, q3, q4)
			}
		})
	}
}
`
	if err := os.WriteFile(filepath.Join(tmpDir, "bench_test.go"), []byte(benchContent), 0644); err != nil {
		t.Fatalf("write bench test: %v", err)
	}

	// List files for debugging
	entries, _ := os.ReadDir(tmpDir)
	var files []string
	for _, e := range entries {
		files = append(files, e.Name())
	}
	t.Logf("Package files: %v", files)

	// Step 6: Run correctness test first
	goBin := filepath.Join(goRoot(), "bin", "go")
	cmd := exec.Command(goBin, "test", "-v", "-run=TestBitProductCorrectness", "./...")
	cmd.Dir = tmpDir
	cmd.Env = append(os.Environ(), "GOWORK=off")
	output, err := cmd.CombinedOutput()
	t.Logf("Correctness test output:\n%s", string(output))
	if err != nil {
		t.Fatalf("correctness test failed: %v\n%s", err, string(output))
	}

	// Step 7: Run benchmarks
	cmd = exec.Command(goBin, "test", "-bench=BenchmarkBitProduct", "-benchmem", "-count=1", "./...")
	cmd.Dir = tmpDir
	cmd.Env = append(os.Environ(), "GOWORK=off")
	output, err = cmd.CombinedOutput()
	t.Logf("Benchmark output:\n%s", string(output))
	if err != nil {
		t.Fatalf("benchmark failed: %v\n%s", err, string(output))
	}
}

// TestBenchmarkVarintASTvsHandwritten benchmarks the AST-generated NEON varint
// assembly against the existing hand-written NEON varint assembly.
func TestBenchmarkVarintASTvsHandwritten(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skip("correctness test requires arm64 to execute generated NEON assembly")
	}
	varintPath := filepath.Join("..", "..", "hwy", "contrib", "varint", "varint_base.go")
	if _, err := os.Stat(varintPath); err != nil {
		t.Skipf("varint_base.go not found: %v", err)
	}

	tmpDir := filepath.Join(t.TempDir(), "varintbench")
	if err := os.MkdirAll(tmpDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	// Step 1: Generate AST-translated C for BaseFindVarintEnds only, then compile with GOAT
	result, err := Parse(varintPath)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}
	var findEndsFunc *ParsedFunc
	for i, pf := range result.Funcs {
		if pf.Name == "BaseFindVarintEnds" {
			findEndsFunc = &result.Funcs[i]
			break
		}
	}
	if findEndsFunc == nil {
		t.Fatal("BaseFindVarintEnds not found in varint_base.go")
	}

	target, err := GetTarget("neon")
	if err != nil {
		t.Fatalf("GetTarget: %v", err)
	}
	profile := GetCProfile(target.Name, "uint8")
	if profile == nil {
		t.Fatal("NEON uint8 profile not found")
	}
	emitter := NewCEmitter(result.PackageName, "uint8", target)
	emitter.profile = profile
	cFile, err := emitter.EmitASTTranslatedC(findEndsFunc, tmpDir)
	if err != nil {
		t.Fatalf("EmitASTTranslatedC failed: %v", err)
	}

	// Compile with GOAT
	if err := runGOAT(cFile, profile); err != nil {
		if strings.Contains(err.Error(), "exec:") || strings.Contains(err.Error(), "go tool") {
			t.Skipf("GOAT not available: %v", err)
		}
		t.Fatalf("GOAT compile failed: %v", err)
	}
	os.Remove(cFile)
	os.Remove(strings.TrimSuffix(cFile, ".c") + ".o")

	// Generate wrapper for AST-translated function
	{
		var buf bytes.Buffer
		fmt.Fprintf(&buf, "package varintbench\n\nimport \"unsafe\"\n\n")
		emitASTCWrapperFunc(&buf, findEndsFunc, "uint8", "neon")
		if err := os.WriteFile(filepath.Join(tmpDir, "ast_wrapper.go"), buf.Bytes(), 0644); err != nil {
			t.Fatalf("write ast wrapper: %v", err)
		}
	}

	// Step 2: Copy the hand-written NEON assembly from asm/ directory
	asmDir := filepath.Join("..", "..", "hwy", "contrib", "varint", "asm")
	for _, suffix := range []string{".s", ".go"} {
		src := filepath.Join(asmDir, "varint_neon_arm64"+suffix)
		dst := filepath.Join(tmpDir, "varint_neon_arm64"+suffix)
		data, err := os.ReadFile(src)
		if err != nil {
			t.Fatalf("read %s: %v", src, err)
		}
		if suffix == ".go" {
			data = []byte(strings.Replace(string(data), "package asm", "package varintbench", 1))
		}
		if err := os.WriteFile(dst, data, 0644); err != nil {
			t.Fatalf("write %s: %v", dst, err)
		}
	}

	// Step 3: Write a wrapper for the hand-written asm function
	handwrittenWrapper := `package varintbench

import "unsafe"

// FindVarintEndsHandwritten wraps the hand-written NEON varint assembly.
// The hand-written version returns uint64 (handles up to 64 bytes),
// but we only compare the lower 32 bits for inputs <= 32 bytes.
func FindVarintEndsHandwritten(src []byte) uint32 {
	if len(src) == 0 {
		return 0
	}
	var result int64
	n := int64(len(src))
	find_varint_ends_u8(unsafe.Pointer(&src[0]), n, unsafe.Pointer(&result))
	return uint32(result)
}
`
	if err := os.WriteFile(filepath.Join(tmpDir, "handwritten_wrapper.go"), []byte(handwrittenWrapper), 0644); err != nil {
		t.Fatalf("write handwritten wrapper: %v", err)
	}

	// Step 4: Write go.mod
	goModContent := `module varintbench

go 1.26
`
	if err := os.WriteFile(filepath.Join(tmpDir, "go.mod"), []byte(goModContent), 0644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}

	// Step 5: Write the benchmark test
	benchContent := `package varintbench

import (
	"fmt"
	"testing"
)

// findVarintEndsScalar is the pure Go reference implementation.
func findVarintEndsScalar(src []byte) uint32 {
	if len(src) == 0 {
		return 0
	}
	n := min(len(src), 32)
	var mask uint32
	for i := range n {
		if src[i] < 0x80 {
			mask |= 1 << uint(i)
		}
	}
	return mask
}

func TestFindVarintEndsCorrectness(t *testing.T) {
	// Test with 32 bytes (full SIMD path)
	src := make([]byte, 32)
	for i := range src {
		if i%3 == 0 {
			src[i] = 0x42 // < 0x80, terminator
		} else {
			src[i] = 0x82 // >= 0x80, continuation
		}
	}
	expected := findVarintEndsScalar(src)
	gotAST := FindVarintEndsCU8(src)
	gotHW := FindVarintEndsHandwritten(src)
	if gotAST != expected {
		t.Errorf("AST 32-byte: got 0x%08x, want 0x%08x", gotAST, expected)
	}
	if gotHW != expected {
		t.Errorf("Handwritten 32-byte: got 0x%08x, want 0x%08x", gotHW, expected)
	}

	// Test with shorter buffers
	for n := 1; n <= 16; n++ {
		data := make([]byte, n)
		for i := range data {
			if i%2 == 0 {
				data[i] = 0x7F
			} else {
				data[i] = 0x80
			}
		}
		expected := findVarintEndsScalar(data)
		gotAST := FindVarintEndsCU8(data)
		gotHW := FindVarintEndsHandwritten(data)
		if gotAST != expected {
			t.Errorf("AST n=%d: got 0x%08x, want 0x%08x", n, gotAST, expected)
		}
		if gotHW != expected {
			t.Errorf("Handwritten n=%d: got 0x%08x, want 0x%08x", n, gotHW, expected)
		}
	}
}

func BenchmarkFindVarintEnds(b *testing.B) {
	sizes := []int{16, 32}

	for _, n := range sizes {
		src := make([]byte, n)
		for i := range src {
			if i%3 == 0 {
				src[i] = 0x42
			} else {
				src[i] = 0x82
			}
		}

		b.Run(fmt.Sprintf("ASTGenerated/%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				FindVarintEndsCU8(src)
			}
		})

		b.Run(fmt.Sprintf("Handwritten/%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				FindVarintEndsHandwritten(src)
			}
		})

		b.Run(fmt.Sprintf("Scalar/%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				findVarintEndsScalar(src)
			}
		})
	}
}
`
	if err := os.WriteFile(filepath.Join(tmpDir, "bench_test.go"), []byte(benchContent), 0644); err != nil {
		t.Fatalf("write bench test: %v", err)
	}

	// List files for debugging
	entries, _ := os.ReadDir(tmpDir)
	var files []string
	for _, e := range entries {
		files = append(files, e.Name())
	}
	t.Logf("Package files: %v", files)

	// Step 6: Run correctness test first
	goBin := filepath.Join(goRoot(), "bin", "go")
	cmd := exec.Command(goBin, "test", "-v", "-run=TestFindVarintEndsCorrectness", "./...")
	cmd.Dir = tmpDir
	cmd.Env = append(os.Environ(), "GOWORK=off")
	output, err := cmd.CombinedOutput()
	t.Logf("Correctness test output:\n%s", string(output))
	if err != nil {
		t.Fatalf("correctness test failed: %v\n%s", err, string(output))
	}

	// Step 7: Run benchmarks
	cmd = exec.Command(goBin, "test", "-bench=BenchmarkFindVarintEnds", "-benchmem", "-count=1", "./...")
	cmd.Dir = tmpDir
	cmd.Env = append(os.Environ(), "GOWORK=off")
	output, err = cmd.CombinedOutput()
	t.Logf("Benchmark output:\n%s", string(output))
	if err != nil {
		t.Fatalf("benchmark failed: %v\n%s", err, string(output))
	}
}

// TestTranslateLoad4NEON verifies that hwy.Load4 multi-assign is translated to
// vld1q_u64_x4 + .val[N] destructuring on NEON.
func TestTranslateLoad4NEON(t *testing.T) {
	profile := GetCProfile("NEON", "uint64")
	if profile == nil {
		t.Fatal("NEON uint64 profile not found")
	}

	fset := token.NewFileSet()
	src := `package test
import "github.com/ajroetker/go-highway/hwy"
func BaseLoad4Test(data []uint64, n int) {
	lanes := hwy.Zero[uint64]().NumLanes()
	stride := lanes * 4
	for i := 0; i + stride <= n; i += stride {
		a, b, c, d := hwy.Load4(data[i:])
		_ = hwy.ReduceSum(hwy.PopCount(hwy.And(a, b)))
		_ = hwy.ReduceSum(hwy.PopCount(hwy.And(c, d)))
	}
}
`
	file, err := parser.ParseFile(fset, "test.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	var funcDecl *ast.FuncDecl
	for _, decl := range file.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok {
			funcDecl = fd
			break
		}
	}
	if funcDecl == nil {
		t.Fatal("no function found")
	}

	pf := &ParsedFunc{
		Name: "BaseLoad4Test",
		Params: []Param{
			{Name: "data", Type: "[]uint64"},
			{Name: "n", Type: "int"},
		},
		Body: funcDecl.Body,
		HwyCalls: []HwyCall{
			{Package: "hwy", FuncName: "Load4"},
			{Package: "hwy", FuncName: "And"},
			{Package: "hwy", FuncName: "PopCount"},
			{Package: "hwy", FuncName: "ReduceSum"},
		},
	}

	translator := NewCASTTranslator(profile, "uint64")
	cCode, err := translator.TranslateToC(pf)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C:\n%s", cCode)

	// Verify vld1q_u64_x4 multi-load
	if !strings.Contains(cCode, "vld1q_u64_x4(") {
		t.Error("missing vld1q_u64_x4 for hwy.Load4")
	}

	// Verify uint64x2x4_t struct type with unique temp name
	if !strings.Contains(cCode, "uint64x2x4_t _load4_0") {
		t.Error("missing uint64x2x4_t _load4_0 type for Load4 result")
	}

	// Verify .val[0] through .val[3] destructuring
	for i := range 4 {
		pattern := fmt.Sprintf(".val[%d]", i)
		if !strings.Contains(cCode, pattern) {
			t.Errorf("missing %s destructuring", pattern)
		}
	}
}

// TestTranslateGetLaneVariableIndex verifies that hwy.GetLane with a variable index
// emits the store-to-stack pattern (volatile buffer + vst1q + array index).
func TestTranslateGetLaneVariableIndex(t *testing.T) {
	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	fset := token.NewFileSet()
	src := `package test
import "github.com/ajroetker/go-highway/hwy"
func BaseGetLaneTest(data []float32, n int) {
	v := hwy.Load(data[0:])
	for j := range 4 {
		element := hwy.GetLane(v, j)
		_ = element
	}
}
`
	file, err := parser.ParseFile(fset, "test.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	var funcDecl *ast.FuncDecl
	for _, decl := range file.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok {
			funcDecl = fd
			break
		}
	}
	if funcDecl == nil {
		t.Fatal("no function found")
	}

	pf := &ParsedFunc{
		Name: "BaseGetLaneTest",
		Params: []Param{
			{Name: "data", Type: "[]float32"},
			{Name: "n", Type: "int"},
		},
		Body: funcDecl.Body,
		HwyCalls: []HwyCall{
			{Package: "hwy", FuncName: "Load"},
			{Package: "hwy", FuncName: "GetLane"},
		},
	}

	translator := NewCASTTranslator(profile, "float32")
	cCode, err := translator.TranslateToC(pf)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C:\n%s", cCode)

	// Verify volatile buffer for store-to-stack
	if !strings.Contains(cCode, "volatile float _getlane_buf[4]") {
		t.Error("missing volatile float _getlane_buf[4] for variable-index GetLane")
	}

	// Verify store instruction
	if !strings.Contains(cCode, "vst1q_f32(") {
		t.Error("missing vst1q_f32 for store-to-stack pattern")
	}

	// Verify array index access
	if !strings.Contains(cCode, "_getlane_buf[j]") {
		t.Error("missing _getlane_buf[j] for variable-index access")
	}
}

// TestTranslateMathFloat32bits verifies that math.Float32bits and math.Float32frombits
// are translated to float_to_bits() and bits_to_float() helper calls.
func TestTranslateMathFloat32bits(t *testing.T) {
	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	fset := token.NewFileSet()
	src := `package test
import "math"
func BaseSignTest(data []float32, n int) {
	for i := 0; i < n; i++ {
		bits := math.Float32bits(data[i])
		sign := bits >> 31
		result := math.Float32frombits(sign)
		_ = result
	}
}
`
	file, err := parser.ParseFile(fset, "test.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	var funcDecl *ast.FuncDecl
	for _, decl := range file.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok {
			funcDecl = fd
			break
		}
	}
	if funcDecl == nil {
		t.Fatal("no function found")
	}

	pf := &ParsedFunc{
		Name: "BaseSignTest",
		Params: []Param{
			{Name: "data", Type: "[]float32"},
			{Name: "n", Type: "int"},
		},
		Body: funcDecl.Body,
		HwyCalls: []HwyCall{
			{Package: "hwy", FuncName: "Load"}, // dummy to pass eligibility
		},
	}

	translator := NewCASTTranslator(profile, "float32")
	cCode, err := translator.TranslateToC(pf)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C:\n%s", cCode)

	// Verify float_to_bits mapping
	if !strings.Contains(cCode, "float_to_bits(") {
		t.Error("missing float_to_bits() for math.Float32bits")
	}

	// Verify bits_to_float mapping
	if !strings.Contains(cCode, "bits_to_float(") {
		t.Error("missing bits_to_float() for math.Float32frombits")
	}
}

// TestTranslateGetSignBit verifies that the getSignBit helper is inlined as
// (float_to_bits(x) >> 31) in the translator.
func TestTranslateGetSignBit(t *testing.T) {
	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	fset := token.NewFileSet()
	src := `package test
import "math"
func getSignBit(f float32) uint32 {
	return math.Float32bits(f) >> 31
}
func BaseSignBitTest(data []float32, n int) {
	for i := 0; i < n; i++ {
		sign := getSignBit(data[i])
		_ = sign
	}
}
`
	file, err := parser.ParseFile(fset, "test.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	// Find the BaseSignBitTest function
	var funcDecl *ast.FuncDecl
	for _, decl := range file.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok && fd.Name.Name == "BaseSignBitTest" {
			funcDecl = fd
			break
		}
	}
	if funcDecl == nil {
		t.Fatal("BaseSignBitTest not found")
	}

	pf := &ParsedFunc{
		Name: "BaseSignBitTest",
		Params: []Param{
			{Name: "data", Type: "[]float32"},
			{Name: "n", Type: "int"},
		},
		Body: funcDecl.Body,
		HwyCalls: []HwyCall{
			{Package: "hwy", FuncName: "Load"}, // dummy
		},
	}

	translator := NewCASTTranslator(profile, "float32")
	cCode, err := translator.TranslateToC(pf)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C:\n%s", cCode)

	// Verify getSignBit is inlined as (float_to_bits(x) >> 31)
	if !strings.Contains(cCode, "float_to_bits(") {
		t.Error("missing float_to_bits() for inlined getSignBit")
	}
	if !strings.Contains(cCode, ">> 31)") {
		t.Error("missing >> 31 for inlined getSignBit")
	}
}

func TestASTTranslatorF16NEON(t *testing.T) {
	// Find the matmul_base.go file
	matmulPath := filepath.Join("..", "..", "hwy", "contrib", "matmul", "matmul_base.go")
	if _, err := os.Stat(matmulPath); err != nil {
		t.Skipf("matmul_base.go not found at %s: %v", matmulPath, err)
	}

	result, err := Parse(matmulPath)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	var matmulFunc *ParsedFunc
	for i, pf := range result.Funcs {
		if pf.Name == "BaseMatMul" {
			matmulFunc = &result.Funcs[i]
			break
		}
	}
	if matmulFunc == nil {
		t.Fatal("BaseMatMul not found in parsed functions")
	}

	// Translate for NEON f16
	profile := GetCProfile("NEON", "hwy.Float16")
	if profile == nil {
		t.Fatal("NEON Float16 profile not found")
	}

	// Verify profile has NativeArithmetic set
	if !profile.NativeArithmetic {
		t.Error("NEON f16 profile should have NativeArithmetic = true")
	}
	if profile.ScalarArithType != "float16_t" {
		t.Errorf("NEON f16 ScalarArithType = %q, want %q", profile.ScalarArithType, "float16_t")
	}

	translator := NewCASTTranslator(profile, "hwy.Float16")
	cCode, err := translator.TranslateToC(matmulFunc)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C code:\n%s", cCode)

	// Verify Load with CastExpr: vld1q_f16((float16_t*)(...))
	if !strings.Contains(cCode, "vld1q_f16((float16_t*)(") {
		t.Error("missing vld1q_f16((float16_t*)(...)) for hwy.Load with CastExpr")
	}

	// Verify Store with CastExpr: vst1q_f16((float16_t*)(...), ...)
	if !strings.Contains(cCode, "vst1q_f16((float16_t*)(") {
		t.Error("missing vst1q_f16((float16_t*)(...), ...) for hwy.Store with CastExpr")
	}

	// Verify FMA: vfmaq_f16(acc, a, b) — NEON acc-first
	if !strings.Contains(cCode, "vfmaq_f16(vC, vA, vB)") {
		t.Error("missing vfmaq_f16(vC, vA, vB) for hwy.MulAdd with NEON f16")
	}

	// Verify Zero uses dup with 0.0 (not 0.0f for f16)
	if !strings.Contains(cCode, "vdupq_n_f16(0.0)") {
		t.Error("missing vdupq_n_f16(0.0) for hwy.Zero with f16")
	}

	// Verify lanes = 8 for NEON f16
	if !strings.Contains(cCode, "= 8") {
		t.Error("missing NumLanes constant (= 8 for NEON f16)")
	}

	// Verify float literals don't have 'f' suffix for f16
	if strings.Contains(cCode, "0.0f") {
		t.Error("f16 should not use 0.0f suffix — should use bare 0.0 or 0")
	}

	// Verify function name includes f16
	if !strings.Contains(cCode, "matmul_c_f16_neon") {
		t.Error("missing function name with f16 suffix")
	}

	// Verify pointer type uses float16_t (ScalarArithType) for native f16 arithmetic
	if !strings.Contains(cCode, "float16_t *a") {
		t.Error("missing float16_t *a parameter for f16")
	}
}

func TestASTTranslatorContinueBreak(t *testing.T) {
	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	fset := token.NewFileSet()
	src := `package test

import "github.com/ajroetker/go-highway/hwy"

func BaseContinueBreak(data []float32, n int) {
	for i := 0; i < n; i++ {
		if data[i] == 0 {
			continue
		}
		v := hwy.Load(data[i:])
		_ = v
		if i > 10 {
			break
		}
	}
}
`
	file, err := parser.ParseFile(fset, "test.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	var funcDecl *ast.FuncDecl
	for _, decl := range file.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok && fd.Name.Name == "BaseContinueBreak" {
			funcDecl = fd
			break
		}
	}
	if funcDecl == nil {
		t.Fatal("no function found")
	}

	pf := &ParsedFunc{
		Name: "BaseContinueBreak",
		Params: []Param{
			{Name: "data", Type: "[]float32"},
			{Name: "n", Type: "int"},
		},
		Body:     funcDecl.Body,
		HwyCalls: []HwyCall{{Package: "hwy", FuncName: "Load"}},
	}

	translator := NewCASTTranslator(profile, "float32")
	cCode, err := translator.TranslateToC(pf)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C code:\n%s", cCode)

	if !strings.Contains(cCode, "continue;") {
		t.Error("missing 'continue;' in C output")
	}
	if !strings.Contains(cCode, "break;") {
		t.Error("missing 'break;' in C output")
	}
}

func TestASTTranslatorNilComparison(t *testing.T) {
	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	fset := token.NewFileSet()
	src := `package test

import "github.com/ajroetker/go-highway/hwy"

func BaseNilCheck(data, bias []float32, n int) {
	for i := 0; i < n; i++ {
		v := hwy.Load(data[i:])
		if bias != nil {
			b := hwy.Load(bias[i:])
			v = hwy.Add(v, b)
		}
		hwy.Store(v, data[i:])
	}
}
`
	file, err := parser.ParseFile(fset, "test.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	var funcDecl *ast.FuncDecl
	for _, decl := range file.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok && fd.Name.Name == "BaseNilCheck" {
			funcDecl = fd
			break
		}
	}
	if funcDecl == nil {
		t.Fatal("no function found")
	}

	pf := &ParsedFunc{
		Name: "BaseNilCheck",
		Params: []Param{
			{Name: "data", Type: "[]float32"},
			{Name: "bias", Type: "[]float32"},
			{Name: "n", Type: "int"},
		},
		Body:     funcDecl.Body,
		HwyCalls: []HwyCall{{Package: "hwy", FuncName: "Load"}},
	}

	translator := NewCASTTranslator(profile, "float32")
	cCode, err := translator.TranslateToC(pf)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C code:\n%s", cCode)

	// nil should be translated to 0 (C NULL)
	if !strings.Contains(cCode, "!= 0") {
		t.Error("missing '!= 0' for nil comparison — nil should translate to 0")
	}
	// Should NOT contain " nil " or "nil)" in C output (as Go keyword)
	if strings.Contains(cCode, " nil ") || strings.Contains(cCode, " nil)") {
		t.Error("Go 'nil' keyword should not appear in C output")
	}
}

func TestASTTranslatorStdMathCalls(t *testing.T) {
	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	fset := token.NewFileSet()
	src := `package test

import stdmath "math"

import "github.com/ajroetker/go-highway/hwy"

func BaseMathOps(data []float32, n int) {
	for i := 0; i < n; i++ {
		v := hwy.Load(data[i:])
		_ = v
		x := data[i]
		y := stdmath.Sqrt(x)
		z := stdmath.Exp(y)
		w := stdmath.Log(z)
		inf := stdmath.Inf(1)
		_ = w
		_ = inf
	}
}
`
	file, err := parser.ParseFile(fset, "test.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	var funcDecl *ast.FuncDecl
	for _, decl := range file.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok && fd.Name.Name == "BaseMathOps" {
			funcDecl = fd
			break
		}
	}
	if funcDecl == nil {
		t.Fatal("no function found")
	}

	pf := &ParsedFunc{
		Name: "BaseMathOps",
		Params: []Param{
			{Name: "data", Type: "[]float32"},
			{Name: "n", Type: "int"},
		},
		Body:     funcDecl.Body,
		HwyCalls: []HwyCall{{Package: "hwy", FuncName: "Load"}},
	}

	translator := NewCASTTranslator(profile, "float32")
	cCode, err := translator.TranslateToC(pf)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C code:\n%s", cCode)

	// Verify sqrtf for float32
	if !strings.Contains(cCode, "sqrtf(") {
		t.Error("missing sqrtf() for stdmath.Sqrt with float32")
	}

	// Verify expf for float32
	if !strings.Contains(cCode, "expf(") {
		t.Error("missing expf() for stdmath.Exp with float32")
	}

	// Verify logf for float32
	if !strings.Contains(cCode, "logf(") {
		t.Error("missing logf() for stdmath.Log with float32")
	}

	// Verify Inf → (1.0f/0.0f)
	if !strings.Contains(cCode, "(1.0f/0.0f)") {
		t.Error("missing (1.0f/0.0f) for stdmath.Inf(1)")
	}

	// Should NOT contain "stdmath." in C output
	if strings.Contains(cCode, "stdmath.") {
		t.Error("Go 'stdmath.' package prefix should not appear in C output")
	}
}

func TestASTTranslatorScalarMake(t *testing.T) {
	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	fset := token.NewFileSet()
	src := `package test

import "github.com/ajroetker/go-highway/hwy"

func BaseScalarMake(data []float32, n int) {
	shifted := make([]float32, n)
	for i := 0; i < n; i++ {
		v := hwy.Load(data[i:])
		hwy.Store(v, shifted[i:])
	}
	_ = shifted[0]
}
`
	file, err := parser.ParseFile(fset, "test.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	var funcDecl *ast.FuncDecl
	for _, decl := range file.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok && fd.Name.Name == "BaseScalarMake" {
			funcDecl = fd
			break
		}
	}
	if funcDecl == nil {
		t.Fatal("no function found")
	}

	pf := &ParsedFunc{
		Name: "BaseScalarMake",
		Params: []Param{
			{Name: "data", Type: "[]float32"},
			{Name: "n", Type: "int"},
		},
		Body:     funcDecl.Body,
		HwyCalls: []HwyCall{{Package: "hwy", FuncName: "Load"}},
	}

	translator := NewCASTTranslator(profile, "float32")
	cCode, err := translator.TranslateToC(pf)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C code:\n%s", cCode)

	// Verify C99 VLA declaration: float shifted[n];
	if !strings.Contains(cCode, "float shifted[n];") {
		t.Error("missing 'float shifted[n];' for make([]float32, n)")
	}

	// Verify indexing into the scalar array works
	if !strings.Contains(cCode, "shifted[") {
		t.Error("missing indexing into shifted array")
	}

	// Should NOT contain the unsupported make comment
	if strings.Contains(cCode, "make: unsupported type") {
		t.Error("scalar make should not produce 'unsupported type' comment")
	}
}

func TestASTTranslatorSlideUpLanes(t *testing.T) {
	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	fset := token.NewFileSet()
	src := `package test

import "github.com/ajroetker/go-highway/hwy"

func BaseSlideUp(data []float32, n int) {
	for i := 0; i < n; i += 4 {
		v := hwy.Load(data[i:])
		s1 := hwy.SlideUpLanes(v, 1)
		s2 := hwy.SlideUpLanes(v, 2)
		_ = s1
		_ = s2
	}
}
`
	file, err := parser.ParseFile(fset, "test.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	var funcDecl *ast.FuncDecl
	for _, decl := range file.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok && fd.Name.Name == "BaseSlideUp" {
			funcDecl = fd
			break
		}
	}
	if funcDecl == nil {
		t.Fatal("no function found")
	}

	pf := &ParsedFunc{
		Name: "BaseSlideUp",
		Params: []Param{
			{Name: "data", Type: "[]float32"},
			{Name: "n", Type: "int"},
		},
		Body:     funcDecl.Body,
		HwyCalls: []HwyCall{{Package: "hwy", FuncName: "Load"}, {Package: "hwy", FuncName: "SlideUpLanes"}},
	}

	translator := NewCASTTranslator(profile, "float32")
	cCode, err := translator.TranslateToC(pf)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C code:\n%s", cCode)

	// NEON f32: 4 lanes. SlideUpLanes(v, 1) → vextq_f32(zero, v, 3)
	if !strings.Contains(cCode, "vextq_f32(vdupq_n_f32(0.0f), v, 3)") {
		t.Error("missing vextq_f32(vdupq_n_f32(0.0f), v, 3) for SlideUpLanes(v, 1)")
	}

	// SlideUpLanes(v, 2) → vextq_f32(zero, v, 2)
	if !strings.Contains(cCode, "vextq_f32(vdupq_n_f32(0.0f), v, 2)") {
		t.Error("missing vextq_f32(vdupq_n_f32(0.0f), v, 2) for SlideUpLanes(v, 2)")
	}
}

func TestASTTranslatorSlideUpLanesF64(t *testing.T) {
	profile := GetCProfile("NEON", "float64")
	if profile == nil {
		t.Fatal("NEON float64 profile not found")
	}

	fset := token.NewFileSet()
	src := `package test

import "github.com/ajroetker/go-highway/hwy"

func BaseSlideUpF64(data []float64, n int) {
	for i := 0; i < n; i += 2 {
		v := hwy.Load(data[i:])
		s1 := hwy.SlideUpLanes(v, 1)
		_ = s1
	}
}
`
	file, err := parser.ParseFile(fset, "test.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	var funcDecl *ast.FuncDecl
	for _, decl := range file.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok && fd.Name.Name == "BaseSlideUpF64" {
			funcDecl = fd
			break
		}
	}
	if funcDecl == nil {
		t.Fatal("no function found")
	}

	pf := &ParsedFunc{
		Name: "BaseSlideUpF64",
		Params: []Param{
			{Name: "data", Type: "[]float64"},
			{Name: "n", Type: "int"},
		},
		Body:     funcDecl.Body,
		HwyCalls: []HwyCall{{Package: "hwy", FuncName: "Load"}, {Package: "hwy", FuncName: "SlideUpLanes"}},
	}

	translator := NewCASTTranslator(profile, "float64")
	cCode, err := translator.TranslateToC(pf)
	if err != nil {
		t.Fatalf("TranslateToC failed: %v", err)
	}

	t.Logf("Generated C code:\n%s", cCode)

	// NEON f64: 2 lanes. SlideUpLanes(v, 1) → vextq_f64(zero, v, 1)
	if !strings.Contains(cCode, "vextq_f64(vdupq_n_f64(0.0), v, 1)") {
		t.Error("missing vextq_f64(vdupq_n_f64(0.0), v, 1) for SlideUpLanes(v, 1)")
	}
}

func TestGoSliceElemToCType_ScalarArithType(t *testing.T) {
	// When ScalarArithType is set (e.g., float16_t for NEON f16),
	// goSliceElemToCType("T", profile) should return ScalarArithType
	// instead of CType.
	profile := GetCProfile("NEON", "hwy.Float16")
	if profile == nil {
		t.Fatal("NEON Float16 profile not found")
	}

	got := goSliceElemToCType("T", profile)
	if got != "float16_t" {
		t.Errorf("goSliceElemToCType(T, NEON f16) = %q, want %q", got, "float16_t")
	}

	// float32 profile has no ScalarArithType — should return CType
	f32Profile := GetCProfile("NEON", "float32")
	if f32Profile == nil {
		t.Fatal("NEON float32 profile not found")
	}

	got = goSliceElemToCType("T", f32Profile)
	if got != "float" {
		t.Errorf("goSliceElemToCType(T, NEON f32) = %q, want %q", got, "float")
	}
}

func TestGoTypeToCType_ScalarArithType(t *testing.T) {
	// When ScalarArithType is set, goTypeToCType("T") should return it
	profile := GetCProfile("NEON", "hwy.Float16")
	if profile == nil {
		t.Fatal("NEON Float16 profile not found")
	}

	translator := NewCASTTranslator(profile, "hwy.Float16")
	got := translator.goTypeToCType("T")
	if got != "float16_t" {
		t.Errorf("goTypeToCType(T, NEON f16) = %q, want %q", got, "float16_t")
	}

	// Also verify hwy.Vec[T] maps to the vector type
	vecGot := translator.goTypeToCType("hwy.Vec[T]")
	if vecGot == "" {
		t.Error("goTypeToCType(hwy.Vec[T]) returned empty string")
	}
	// NEON f16 primary vector type should be float16x8_t
	if vecGot != "float16x8_t" {
		t.Errorf("goTypeToCType(hwy.Vec[T], NEON f16) = %q, want %q", vecGot, "float16x8_t")
	}
}

func TestASTWrapperGoScalarType(t *testing.T) {
	tests := []struct {
		elemType string
		want     string
	}{
		{"float32", "float32"},
		{"float64", "float64"},
		{"float16", "hwy.Float16"},
		{"hwy.Float16", "hwy.Float16"},
		{"bfloat16", "hwy.BFloat16"},
		{"hwy.BFloat16", "hwy.BFloat16"},
		{"int32", "int32"},
		{"int64", "int64"},
		{"uint32", "uint32"},
		{"uint64", "uint64"},
		{"unknown", "float32"},
	}

	for _, tt := range tests {
		t.Run(tt.elemType, func(t *testing.T) {
			got := astWrapperGoScalarType(tt.elemType)
			if got != tt.want {
				t.Errorf("astWrapperGoScalarType(%q) = %q, want %q", tt.elemType, got, tt.want)
			}
		})
	}
}

func TestGroupGoParams_ScalarT(t *testing.T) {
	// Test that groupGoParams maps T params to concrete Go types
	pf := &ParsedFunc{
		Name: "BaseLiftStep",
		Params: []Param{
			{Name: "target", Type: "[]T"},
			{Name: "tLen", Type: "int"},
			{Name: "neighbor", Type: "[]T"},
			{Name: "nLen", Type: "int"},
			{Name: "coeff", Type: "T"},
			{Name: "phase", Type: "int"},
		},
	}

	sig := groupGoParams(pf, "[]hwy.Float16", "hwy.Float16")
	// coeff should be mapped to hwy.Float16
	if !strings.Contains(sig, "coeff hwy.Float16") {
		t.Errorf("groupGoParams did not map T to hwy.Float16: %q", sig)
	}
	// Slices should use the provided slice type
	if !strings.Contains(sig, "[]hwy.Float16") {
		t.Errorf("groupGoParams did not use []hwy.Float16 for slices: %q", sig)
	}
}
