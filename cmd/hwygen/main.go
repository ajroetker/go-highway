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
//	hwygen -c -input math.go -output . -targets neon          # C code only
//	hwygen -asm -input math.go -output . -targets neon        # C → Go assembly via GOAT
//
// Or via go:generate:
//
//	//go:generate hwygen -input $GOFILE -output . -targets avx2,fallback
//
// The generator takes Go source files containing hwy.* calls and produces:
//  1. A dispatcher file with runtime CPU detection
//  2. Target-specific implementation files (AVX2, AVX512, fallback)
//
// The -c flag generates GOAT-compatible C files for inspection.
// The -asm flag generates C files, compiles them to Go assembly via GOAT,
// and emits Go wrapper functions.
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
	targets        = flag.String("targets", "avx2,fallback", "Comma-separated targets ("+strings.Join(AvailableTargets(), ",")+") or 'all'")
	packageOut     = flag.String("pkg", "", "Output package name (default: same as input)")
	dispatchPrefix = flag.String("dispatch", "", "Dispatch file prefix (default: derived from function name)")
	cMode          = flag.Bool("c", false, "Generate C code only (supports neon, sve_darwin, sve_linux, avx2, avx512 targets)")
	asmMode        = flag.Bool("asm", false, "Generate C code and compile to Go assembly via GOAT (supports neon, sve_darwin, sve_linux, avx2, avx512 targets)")
	fusionMode     = flag.Bool("fusion", false, "Enable IR-based fusion optimization for cross-package function inlining and loop fusion")
	verboseMode    = flag.Bool("v", false, "Verbose output (show fusion statistics, IR dumps, etc.)")
)

func main() {
	flag.Parse()

	if *inputFile == "" {
		fmt.Fprintf(os.Stderr, "Error: -input flag is required\n\n")
		flag.Usage()
		os.Exit(1)
	}

	// Parse target list with per-target mode suffixes
	targetSpecs, err := parseTargets(*targets, *cMode, *asmMode)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if len(targetSpecs) == 0 {
		fmt.Fprintf(os.Stderr, "Error: no valid targets specified\n")
		os.Exit(1)
	}

	// Create and run generator
	gen := &Generator{
		InputFile:      *inputFile,
		OutputDir:      *outputDir,
		OutputPrefix:   *outputPrefix,
		TargetSpecs:    targetSpecs,
		PackageOut:     *packageOut,
		DispatchPrefix: *dispatchPrefix,
		FusionMode:     *fusionMode,
		Verbose:        *verboseMode,
	}

	if err := gen.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Build display string
	var names []string
	for _, ts := range targetSpecs {
		name := strings.ToLower(ts.Target.Name)
		switch ts.Mode {
		case TargetModeAsm:
			name += ":asm"
		case TargetModeC:
			name += ":c"
		}
		names = append(names, name)
	}
	fmt.Printf("Successfully generated code for targets: %s\n", strings.Join(names, ", "))
}

// globalMode returns the TargetMode implied by the global -c and -asm flags.
func globalMode(globalC, globalAsm bool) TargetMode {
	if globalAsm {
		return TargetModeAsm
	}
	if globalC {
		return TargetModeC
	}
	return TargetModeGoSimd
}

// parseTargets parses the comma-separated target string into TargetSpecs.
// Per-target mode suffixes (e.g., "neon:asm") override the global flags.
func parseTargets(s string, globalC, globalAsm bool) ([]TargetSpec, error) {
	parts := strings.Split(s, ",")
	var result []TargetSpec
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if p == "all" {
			for _, name := range AvailableTargets() {
				t, _ := GetTarget(name)
				result = append(result, TargetSpec{Target: t, Mode: globalMode(globalC, globalAsm)})
			}
			return result, nil
		}
		name, mode := parseTargetSpec(p)
		// Global flags apply when no per-target mode and target isn't fallback
		if mode == TargetModeGoSimd && name != "fallback" {
			mode = globalMode(globalC, globalAsm)
		}
		t, err := GetTarget(name)
		if err != nil {
			return nil, err
		}
		result = append(result, TargetSpec{Target: t, Mode: mode})
	}
	return result, nil
}
