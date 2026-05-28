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

// Package evexguard hosts a build-time guard that fails if any go-highway
// function reachable without AVX-512 emits an AVX-512-only (EVEX-encoded)
// instruction. Go's simd/archsimd maps some 256-bit int64 methods to EVEX-only
// opcodes (VPSRAQ, VPMULLQ, VCVTQQ2PD, VPABSQ, ...), which SIGILL on
// AVX2-but-not-AVX-512 CPUs. hwygen redirects the known offenders to scalar
// emulation, but the only way to be sure nothing reintroduces the bug class is
// to disassemble the linked binary and look for the EVEX prefix (0x62).
//
// The expensive integration scan (cross-compile + disassemble) lives in
// evexguard_integration_test.go behind the "evexguard" build tag, so it does
// not slow down every `go test ./...`. CI runs it in a dedicated step with
// `-tags evexguard`; run it locally the same way. The fast parser unit test
// below always runs.
package evexguard

import (
	"bufio"
	"strings"
	"testing"
)

// TestScanEVEX exercises the parser on synthetic llvm-objdump output so the
// detection logic is verified even on hosts where the integration scan skips
// (and without the "evexguard" build tag).
func TestScanEVEX(t *testing.T) {
	const mod = "github.com/ajroetker/go-highway"
	const dump = `
0000000000401000 <github.com/ajroetker/go-highway/hwy.Good_avx2_Int64>:
  401000: c5 fe 6f 04 24            	vmovdqu	(%rsp), %ymm0
  401005: 48 83 c4 62              	addq	$0x62, %rsp
0000000000401100 <github.com/ajroetker/go-highway/hwy.Bad_avx2_Int64>:
  401100: 62 f1 fd 28 6f 04 24      	vmovdqa64	(%rsp), %ymm0
0000000000401180 <github.com/ajroetker/go-highway/hwy.genericHelper>:
  401180: 62 f2 fd 28 40 c1        	vpmullq	%ymm1, %ymm0, %ymm0
0000000000401200 <github.com/ajroetker/go-highway/hwy.Fine_avx512_Int64>:
  401200: 62 f2 fd 28 40 c1        	vpmullq	%ymm1, %ymm0, %ymm0
0000000000401300 <runtime.asyncPreempt.abi0>:
  401300: 62 f1 fd 48 6f 04 24      	vmovdqu64	(%rsp), %zmm0
`
	got := scanEVEX([]byte(dump), mod)
	if len(got) != 2 {
		t.Fatalf("expected exactly 2 offenders (avx2 + generic go-highway), got %d: %v", len(got), got)
	}
	joined := strings.Join(got, "\n")
	for _, want := range []string{"Bad_avx2_Int64", "genericHelper"} {
		if !strings.Contains(joined, want) {
			t.Errorf("offenders should include %s, got: %v", want, got)
		}
	}
	for _, unwanted := range []string{"Fine_avx512_Int64", "asyncPreempt"} {
		if strings.Contains(joined, unwanted) {
			t.Errorf("offenders should NOT include %s (out of scope), got: %v", unwanted, got)
		}
	}
}

// scanEVEX parses llvm-objdump -d output and returns a description of every
// instruction whose first byte is 0x62 (EVEX prefix) inside an in-scope
// function. A function is in scope when its symbol contains modPath (i.e. it is
// go-highway code, not the Go runtime/stdlib) and does NOT contain "avx512"
// (our own AVX-512 functions are legitimately EVEX and runtime-gated).
func scanEVEX(dump []byte, modPath string) []string {
	var offenders []string
	var curFn string
	inScope := false

	sc := bufio.NewScanner(strings.NewReader(string(dump)))
	sc.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	for sc.Scan() {
		line := sc.Text()

		// Function header: "<hexaddr> <symbol>:"
		if name, ok := parseFuncHeader(line); ok {
			curFn = name
			inScope = strings.Contains(name, modPath) &&
				!strings.Contains(strings.ToLower(name), "avx512")
			continue
		}
		if !inScope {
			continue
		}

		// Instruction line: "  <hexaddr>: <byte> <byte> ... \t mnemonic ..."
		if firstByte, ok := parseFirstByte(line); ok && firstByte == "62" {
			offenders = append(offenders, curFn+": "+strings.TrimSpace(line))
		}
	}
	return offenders
}

// parseFuncHeader returns the symbol name from a line like
// "00000000004904e0 <pkg.Func_avx2_Int64>:" and ok=true if it is a header.
func parseFuncHeader(line string) (string, bool) {
	if !strings.HasSuffix(line, ">:") {
		return "", false
	}
	open := strings.IndexByte(line, '<')
	if open < 0 {
		return "", false
	}
	// Everything before '<' should be a hex address followed by a space.
	addr := strings.TrimSpace(line[:open])
	if addr == "" || !isHex(addr) {
		return "", false
	}
	name := line[open+1 : len(line)-2] // strip "<" and ">:"
	return name, true
}

// parseFirstByte returns the first opcode byte (hex pair) of an instruction
// line like "  4904fd: c5 fe 7f ... \t vmovdqu ...". ok is false for any line
// that is not an instruction (blank lines, "...", continuations).
func parseFirstByte(line string) (string, bool) {
	colon := strings.IndexByte(line, ':')
	if colon < 0 {
		return "", false
	}
	addr := strings.TrimSpace(line[:colon])
	if addr == "" || !isHex(addr) {
		return "", false
	}
	rest := strings.TrimSpace(line[colon+1:])
	if rest == "" {
		return "", false
	}
	// First whitespace-separated token is the first byte.
	tok := rest
	if i := strings.IndexAny(rest, " \t"); i >= 0 {
		tok = rest[:i]
	}
	if len(tok) != 2 || !isHex(tok) {
		return "", false
	}
	return strings.ToLower(tok), true
}

func isHex(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}
