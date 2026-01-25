// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
	inputFile      = flag.String("input", "", "Input Go source file (required)")
	outputDir      = flag.String("output", ".", "Output directory (default: current directory)")
	outputPrefix   = flag.String("output_prefix", "", "Output file prefix, the default (if empty) is the input file name without .go")
	targets        = flag.String("targets", "avx2,fallback", "Comma-separated targets: avx2,avx512,fallback")
	packageOut     = flag.String("pkg", "", "Output package name (default: same as input)")
	dispatchPrefix = flag.String("dispatch", "", "Dispatch file prefix (default: derived from function name)")
	bulkMode       = flag.Bool("bulk", false, "Generate bulk C code for NEON (for GOAT compilation)")
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
		InputFile:      *inputFile,
		OutputDir:      *outputDir,
		OutputPrefix:   *outputPrefix,
		Targets:        targetList,
		PackageOut:     *packageOut,
		DispatchPrefix: *dispatchPrefix,
		BulkMode:       *bulkMode,
	}

	if err := gen.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	if *bulkMode {
		fmt.Printf("Successfully generated bulk C code for targets: %s\n", strings.Join(targetList, ", "))
	} else {
		fmt.Printf("Successfully generated code for targets: %s\n", strings.Join(targetList, ", "))
	}
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
