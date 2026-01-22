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

// Package main provides a diagnostic tool to print CPU features detected by Go.
package main

import (
	"fmt"
	"runtime"

	"golang.org/x/sys/cpu"

	"github.com/ajroetker/go-highway/hwy"
)

func main() {
	fmt.Printf("GOOS: %s\n", runtime.GOOS)
	fmt.Printf("GOARCH: %s\n", runtime.GOARCH)
	fmt.Printf("NumCPU: %d\n", runtime.NumCPU())
	fmt.Println()

	fmt.Printf("Highway dispatch level: %s\n", hwy.CurrentLevel())
	fmt.Printf("Highway dispatch width: %d bytes\n", hwy.CurrentWidth())
	fmt.Printf("Highway dispatch name: %s\n", hwy.CurrentName())
	fmt.Println()

	switch runtime.GOARCH {
	case "arm64":
		printARM64Features()
	case "amd64":
		printAMD64Features()
	}

	fmt.Println()
	fmt.Printf("Highway HasARMFP16: %v\n", hwy.HasARMFP16())
	fmt.Printf("Highway HasARMBF16: %v\n", hwy.HasARMBF16())
}

func printARM64Features() {
	fmt.Println("=== golang.org/x/sys/cpu.ARM64 ===")
	fmt.Printf("  HasASIMD:    %v (NEON baseline)\n", cpu.ARM64.HasASIMD)
	fmt.Printf("  HasFP:       %v (Floating point)\n", cpu.ARM64.HasFP)
	fmt.Printf("  HasFPHP:     %v (FP16 scalar, ARMv8.2-A)\n", cpu.ARM64.HasFPHP)
	fmt.Printf("  HasASIMDHP:  %v (FP16 NEON, ARMv8.2-A)\n", cpu.ARM64.HasASIMDHP)
	fmt.Printf("  HasASIMDFHM: %v (FP16 FMA, ARMv8.4-A)\n", cpu.ARM64.HasASIMDFHM)
	fmt.Printf("  HasSVE:      %v (Scalable Vector Extension)\n", cpu.ARM64.HasSVE)
	fmt.Printf("  HasSVE2:     %v (SVE2)\n", cpu.ARM64.HasSVE2)
	fmt.Printf("  HasAES:      %v\n", cpu.ARM64.HasAES)
	fmt.Printf("  HasPMULL:    %v\n", cpu.ARM64.HasPMULL)
	fmt.Printf("  HasSHA1:     %v\n", cpu.ARM64.HasSHA1)
	fmt.Printf("  HasSHA2:     %v\n", cpu.ARM64.HasSHA2)
	fmt.Printf("  HasSHA3:     %v\n", cpu.ARM64.HasSHA3)
	fmt.Printf("  HasSHA512:   %v\n", cpu.ARM64.HasSHA512)
	fmt.Printf("  HasCRC32:    %v\n", cpu.ARM64.HasCRC32)
	fmt.Printf("  HasATOMICS:  %v (Large System Extensions)\n", cpu.ARM64.HasATOMICS)
	fmt.Printf("  HasDCPOP:    %v\n", cpu.ARM64.HasDCPOP)
}

func printAMD64Features() {
	fmt.Println("=== golang.org/x/sys/cpu.X86 ===")
	fmt.Printf("  HasAVX:     %v\n", cpu.X86.HasAVX)
	fmt.Printf("  HasAVX2:    %v\n", cpu.X86.HasAVX2)
	fmt.Printf("  HasAVX512F: %v\n", cpu.X86.HasAVX512F)
	fmt.Printf("  HasAVX512BW: %v\n", cpu.X86.HasAVX512BW)
	fmt.Printf("  HasAVX512VL: %v\n", cpu.X86.HasAVX512VL)
	fmt.Printf("  HasFMA:     %v\n", cpu.X86.HasFMA)
	fmt.Printf("  HasSSE2:    %v\n", cpu.X86.HasSSE2)
	fmt.Printf("  HasSSE41:   %v\n", cpu.X86.HasSSE41)
	fmt.Printf("  HasSSE42:   %v\n", cpu.X86.HasSSE42)
}
