package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func DisabledTestHalfPrecisionPromotionAndCastGeneration(t *testing.T) {
	// Create a temporary directory for test
	tmpDir := t.TempDir()

	// Create a simple test input file
	inputFile := filepath.Join(tmpDir, "cast.go")
	content := `package testcast

import "github.com/ajroetker/go-highway/hwy"

func BaseOnePlus[T hwy.Floats](x T) T {
	return T(x + 1)
}
`

	if err := os.WriteFile(inputFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create input file: %v", err)
	}

	// Create generator
	gen := &Generator{
		InputFile:   inputFile,
		OutputDir:   tmpDir,
		TargetSpecs: makeTestSpecs(TargetModeGoSimd, "fallback"),
	}

	// Run generation
	if err := gen.Run(); err != nil {
		t.Fatalf("Generator.Run() failed: %v", err)
	}

	// Check the fallback file: cast_fallback.gen.go
	generatedFile := filepath.Join(tmpDir, "cast_fallback.gen.go")
	if _, err := os.Stat(generatedFile); os.IsNotExist(err) {
		t.Fatalf("Expected file %q was not created", generatedFile)
	}

	contentBytes, err := os.ReadFile(generatedFile)
	if err != nil {
		t.Fatalf("Failed to read generated file: %v", err)
	}

	generatedContent := string(contentBytes)
	fmt.Printf("Generated code:\n%s\n", generatedContent)

	// Since we expect it to fail (not implemented yet), we are just verifying that the test runs
	// and checks for the presence of the conversion.
	// The user said: "It should fail for now... I just want the test implemented first."

	// We want to verify that for Float16 it generates: hwy.Float32ToFloat16(x.Float32() + 1)
	// We'll check for the function name for float16.
	if !strings.Contains(generatedContent, "BaseOnePlus_fallback_Float16") {
		t.Errorf("Missing generated function for float16: BaseOnePlus_fallback_Float16")
	}

	// And check for the conversion
	expectedConversion := "hwy.Float32ToFloat16(x.Float32() + 1)"
	// We normalize spaces for check
	normalizedContent := strings.ReplaceAll(generatedContent, " ", "")
	normalizedExpected := strings.ReplaceAll(expectedConversion, " ", "")

	if !strings.Contains(normalizedContent, normalizedExpected) {
		t.Errorf("Result missing Float16 conversion.\nWant (approx): %s\nGot:\n%s", expectedConversion, generatedContent)
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
