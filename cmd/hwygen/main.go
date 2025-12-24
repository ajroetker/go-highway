// Command hwygen generates architecture-specific SIMD implementations from portable hwy.* code.
//
// Usage:
//
//	hwygen -input sigmoid.go -output . -targets avx2,fallback
//
// Or via go:generate:
//
//	//go:generate hwygen -input $GOFILE -output . -targets avx2,fallback
//
// The generator takes Go source files containing hwy.* calls and produces:
//  1. A dispatcher file with runtime CPU detection
//  2. Target-specific implementation files (AVX2, AVX512, fallback)
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
)

var (
	inputFile  = flag.String("input", "", "Input Go source file (required)")
	outputDir  = flag.String("output", ".", "Output directory (default: current directory)")
	targets    = flag.String("targets", "avx2,fallback", "Comma-separated targets: avx2,avx512,fallback")
	packageOut = flag.String("pkg", "", "Output package name (default: same as input)")
)

func main() {
	flag.Parse()

	if *inputFile == "" {
		fmt.Fprintf(os.Stderr, "Error: -input flag is required\n\n")
		flag.Usage()
		os.Exit(1)
	}

	// Parse target list
	targetList := parseTargets(*targets)
	if len(targetList) == 0 {
		fmt.Fprintf(os.Stderr, "Error: no valid targets specified\n")
		os.Exit(1)
	}

	// Create and run generator
	gen := &Generator{
		InputFile:  *inputFile,
		OutputDir:  *outputDir,
		Targets:    targetList,
		PackageOut: *packageOut,
	}

	if err := gen.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Successfully generated code for targets: %s\n", strings.Join(targetList, ", "))
}

func parseTargets(s string) []string {
	parts := strings.Split(s, ",")
	var result []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			result = append(result, p)
		}
	}
	return result
}
