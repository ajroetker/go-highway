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

//go:build evexguard

package evexguard

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestNoEVEXInAVX2ReachableFunctions cross-compiles a binary that blank-imports
// every hwy package (so dispatch vars keep all target functions linked), then
// scans the disassembly for EVEX-prefixed instructions inside any go-highway
// function whose symbol does NOT contain "avx512". A leading 0x62 byte is
// unambiguously an EVEX prefix in 64-bit code (the legacy BOUND opcode is
// invalid there), so any hit is an AVX-512-only instruction that would SIGILL
// on an AVX2-only CPU.
//
// Scoping to go-highway symbols (rather than only "_avx2" names) also catches
// EVEX leaking into generic/fallback helpers and shared functions that an
// AVX2-only CPU could reach, while still excluding our own AVX-512 functions
// (legitimately EVEX, gated by runtime CPU dispatch) and Go's runtime
// (e.g. runtime.asyncPreempt, whose AVX-512 register save is feature-gated).
//
// This test is expensive (it cross-compiles a multi-MB binary and disassembles
// it), so it lives behind the "evexguard" build tag and does not run as part of
// a plain `go test ./...`. Run it with `go test -tags evexguard ./...`; CI runs
// it in a dedicated step. It skips (rather than fails) when its prerequisites
// are unavailable — llvm-objdump missing, or the build can't be produced — so
// it never reports false failures.
func TestNoEVEXInAVX2ReachableFunctions(t *testing.T) {
	objdump := findObjdump(t)

	repoRoot := repoRoot(t)
	modPath := modulePath(t, repoRoot)

	// Enumerate the amd64-buildable hwy packages. Packages with no buildable
	// files for linux/amd64 are dropped by go list automatically.
	pkgs := goListAMD64(t, repoRoot)
	if len(pkgs) == 0 {
		t.Skip("go list returned no hwy packages for linux/amd64")
	}

	// Generate a main package that blank-imports them all. It must live inside
	// the module tree so the imports resolve, so use a temp dir under repoRoot.
	scanDir, err := os.MkdirTemp(repoRoot, "evexguard-scan-")
	if err != nil {
		t.Skipf("could not create scan dir under repo root (read-only?): %v", err)
	}
	defer os.RemoveAll(scanDir)

	var src strings.Builder
	src.WriteString("package main\n\nimport (\n")
	for _, p := range pkgs {
		src.WriteString("\t_ \"")
		src.WriteString(p)
		src.WriteString("\"\n")
	}
	src.WriteString(")\n\nfunc main() {}\n")
	if err := os.WriteFile(filepath.Join(scanDir, "main.go"), []byte(src.String()), 0o644); err != nil {
		t.Skipf("could not write scan main.go: %v", err)
	}

	binPath := filepath.Join(scanDir, "scanbin")
	build := exec.Command("go", "build", "-o", binPath, ".")
	build.Dir = scanDir
	build.Env = append(os.Environ(),
		"GOOS=linux", "GOARCH=amd64", "GOEXPERIMENT=simd", "CGO_ENABLED=0")
	if out, err := build.CombinedOutput(); err != nil {
		t.Skipf("cross-build of scan binary failed (skipping EVEX scan): %v\n%s", err, out)
	}

	dump := exec.Command(objdump, "-d", binPath)
	out, err := dump.Output()
	if err != nil {
		t.Skipf("%s failed (skipping EVEX scan): %v", objdump, err)
	}

	offenders := scanEVEX(out, modPath)
	if len(offenders) > 0 {
		t.Errorf("found %d EVEX (AVX-512-only) instruction(s) inside non-AVX-512 go-highway "+
			"functions; these SIGILL on AVX2-but-not-AVX-512 CPUs. Redirect the offending op to "+
			"a scalar/AVX2 emulation in hwy/ops_avx2.go and the hwygen transformer (see "+
			"cmd/hwygen/transformer_ops.go). Offenders:", len(offenders))
		for _, o := range offenders {
			t.Errorf("  %s", o)
		}
	}
}

// findObjdump locates an llvm-objdump binary, skipping the test if none exists.
// Go's `go tool objdump` mis-decodes newer SIMD instructions, so llvm-objdump
// is required for a trustworthy scan.
func findObjdump(t *testing.T) string {
	t.Helper()
	if p, err := exec.LookPath("llvm-objdump"); err == nil {
		return p
	}
	for _, cand := range []string{
		"/opt/homebrew/opt/llvm/bin/llvm-objdump",
		"/usr/local/opt/llvm/bin/llvm-objdump",
		"/usr/bin/llvm-objdump",
	} {
		if _, err := os.Stat(cand); err == nil {
			return cand
		}
	}
	t.Skip("llvm-objdump not found; install LLVM to run the EVEX guard")
	return ""
}

func repoRoot(t *testing.T) string {
	t.Helper()
	out, err := exec.Command("go", "env", "GOMOD").Output()
	if err != nil {
		t.Skipf("go env GOMOD failed: %v", err)
	}
	gomod := strings.TrimSpace(string(out))
	if gomod == "" || gomod == os.DevNull {
		t.Skip("not in a module; cannot locate repo root")
	}
	return filepath.Dir(gomod)
}

// modulePath returns the module path (e.g. "github.com/ajroetker/go-highway")
// so the scan can scope to go-highway symbols without hardcoding the path.
func modulePath(t *testing.T, repoRoot string) string {
	t.Helper()
	cmd := exec.Command("go", "list", "-m")
	cmd.Dir = repoRoot
	out, err := cmd.Output()
	if err != nil {
		t.Skipf("go list -m failed: %v", err)
	}
	mod := strings.TrimSpace(string(out))
	if mod == "" {
		t.Skip("empty module path")
	}
	return mod
}

func goListAMD64(t *testing.T, repoRoot string) []string {
	t.Helper()
	cmd := exec.Command("go", "list", "./hwy/...")
	cmd.Dir = repoRoot
	cmd.Env = append(os.Environ(), "GOOS=linux", "GOARCH=amd64", "GOEXPERIMENT=simd")
	out, err := cmd.Output()
	if err != nil {
		t.Skipf("go list ./hwy/... for linux/amd64 failed: %v", err)
	}
	var pkgs []string
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			pkgs = append(pkgs, line)
		}
	}
	return pkgs
}
