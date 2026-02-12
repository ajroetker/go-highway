package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// checkFallbackGeneration runs the generator with the given content and checks if
// the generated fallback code contains the expected strings.
//
// If the content is not prefixed with "package", this function automatically prefixes it
// with a package name and importing of the "hwy" package.
func checkFallbackGeneration(t *testing.T, content string, expected, mistakes []string) {
	t.Helper()

	if !strings.HasPrefix(content, "package ") {
		content = "package checkfallbackgeneration\n\nimport \"github.com/ajroetker/go-highway/hwy\"\n\n" + content
	}

	// Create a temporary directory for test
	tmpDir := t.TempDir()

	// Create a simple test input file
	inputFile := filepath.Join(tmpDir, "test.go")

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

	// Check the fallback file: test_fallback.gen.go
	generatedFile := filepath.Join(tmpDir, "test_fallback.gen.go")
	if _, err := os.Stat(generatedFile); os.IsNotExist(err) {
		t.Fatalf("Expected file %q was not created", generatedFile)
	}

	contentBytes, err := os.ReadFile(generatedFile)
	if err != nil {
		t.Fatalf("Failed to read generated file: %v", err)
	}

	generatedContent := string(contentBytes)

	// Normalize spaces for robust check
	fmt.Printf("Generated content: %s\n", generatedContent)
	normalizedContent := strings.ReplaceAll(generatedContent, " ", "")

	for _, exp := range expected {
		normalizedExpected := strings.ReplaceAll(exp, " ", "")
		if !strings.Contains(normalizedContent, normalizedExpected) {
			t.Errorf("Result missing expected string.\n- Want (approx): %s\n- Got:\n%s", exp, generatedContent)
		}
	}
	for _, mistake := range mistakes {
		normalizedMistake := strings.ReplaceAll(mistake, " ", "")
		if strings.Contains(normalizedContent, normalizedMistake) {
			t.Errorf("Result should not contain string.\n- Mistake (approx): %s\n- Got:\n%s", mistake, generatedContent)
		}
	}
}

func TestHalfPrecision(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		expected []string
		mistakes []string
	}{
		{
			name: "Return statement with cast",
			content: `
func BaseOnePlus[T hwy.Floats](x T) T {
	return T(x + 1)
}
`,
			expected: []string{
				"hwy.Float32ToFloat16(float32(x.Float32() + 1))",
				"hwy.Float32ToBFloat16(float32(x.Float32() + 1))",
			},
		},
		{
			name: "Conversion and casting",
			content: `
func BaseOnePlus[T hwy.Floats](x T) T {
	y := x+1
	z := T(y)
	return z
}
`,
			expected: []string{
				"y := x.Float32() + 1",
				// T(y) now produces an actual half-precision value at the
				// assignment level via Float32ToFloat16(float32(...)).
				"z := hwy.Float32ToFloat16(float32(y))",
				"z := hwy.Float32ToBFloat16(float32(y))",
				// z is now a half-precision scalar, returned directly.
				"return z",
			},
			mistakes: []string{
				// z is already half-precision, should not be double-wrapped.
				"return hwy.Float32ToFloat16(z.Float32())",
				"return hwy.Float32ToBFloat16(z.Float32())",
				// z holds the correct type, no need to convert on return.
				"return hwy.Float32ToFloat16(z)",
				"return hwy.Float32ToBFloat16(z)",
			},
		},
		{
			name: "Call",
			content: `
func BasePrint[T hwy.Floats](v T) T {
	fmt.Printf("%g\n", float64(v))
}

func BasePrintPlusOne[T hwy.Floats](x T) {
	Print(x)
	Print(T(x+1))
}
`,
			expected: []string{
				`fmt.Printf("%g\n", float64(v.Float32()))`,
				// T(x+1) at a value boundary (function arg) now produces
				// an actual half-precision value inline.
				`Print(hwy.Float32ToFloat16(float32(x.Float32() + 1)))`,
				`Print(hwy.Float32ToBFloat16(float32(x.Float32() + 1)))`,
			},
			mistakes: []string{
				// Without an expression or explicit cast, it shouldn't try to promote the half-precision.
				`Print(x.Float32())`,
				// Should no longer produce float32 intermediate in function args.
				`Print(float32(x.Float32() + 1))`,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			checkFallbackGeneration(t, tt.content, tt.expected, tt.mistakes)
		})
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
