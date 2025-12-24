package main

import (
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
		{"hwy.Floats", 2, "float32"},
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

import "github.com/go-highway/highway/hwy"

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
	funcs, pkgName, err := Parse(testFile)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// Verify results
	if pkgName != "test" {
		t.Errorf("Package name = %q, want %q", pkgName, "test")
	}

	if len(funcs) != 1 {
		t.Fatalf("Got %d functions, want 1", len(funcs))
	}

	f := funcs[0]
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

import "github.com/go-highway/highway/hwy"

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
	expectedFiles := []string{
		"dispatch.gen.go",
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
	dispatchPath := filepath.Join(tmpDir, "dispatch.gen.go")
	dispatchContent, err := os.ReadFile(dispatchPath)
	if err != nil {
		t.Fatalf("Failed to read dispatcher: %v", err)
	}

	dispatchStr := string(dispatchContent)

	// Should have function variables
	if !strings.Contains(dispatchStr, "var Add func") {
		t.Error("Dispatcher missing Add function variable")
	}

	// Should have init function
	if !strings.Contains(dispatchStr, "func init()") {
		t.Error("Dispatcher missing init function")
	}

	// Should have target-specific init functions
	if !strings.Contains(dispatchStr, "func initAVX2()") {
		t.Error("Dispatcher missing initAVX2 function")
	}
	if !strings.Contains(dispatchStr, "func initFallback()") {
		t.Error("Dispatcher missing initFallback function")
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
		{"BaseSigmoid", "float32", "Sigmoid"},
		{"BaseSigmoid", "float64", "SigmoidFloat64"},
		{"BaseAdd", "float32", "Add"},
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
