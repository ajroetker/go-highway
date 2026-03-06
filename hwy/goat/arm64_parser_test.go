// Copyright 2025 goat Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"testing"
)

func TestTransformStackInstruction_PreIndexedSTP(t *testing.T) {
	// Test pre-indexed STP: stp x28, x27, [sp, #-80]!
	// Original binary: 0xa9bb6ffc
	// Expected: transform to signed-offset stp x28, x27, [sp, #0]
	line := &arm64Line{
		Assembly: "stp\tx28, x27, [sp, #-80]!",
		Binary:   "a9bb6ffc",
	}

	newBinary, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}
	if newBinary == "" {
		t.Fatal("expected non-empty binary, got empty (removed)")
	}

	// Verify the transformation:
	// - Bit 23 should be cleared (pre-indexed -> signed offset)
	// - imm7 should be zeroed
	// Original: 1010 1001 1011 1011 0110 1111 1111 1100
	// After:    1010 1001 0000 0000 0110 1111 1111 1100
	expected := "a9006ffc"
	if newBinary != expected {
		t.Errorf("expected binary %s, got %s", expected, newBinary)
	}
}

func TestTransformStackInstruction_PostIndexedLDP(t *testing.T) {
	// Test post-indexed LDP: ldp x28, x27, [sp], #80
	// Original binary: 0xa8c56ffc (post-indexed load)
	// Expected: transform to signed-offset ldp x28, x27, [sp, #0]
	line := &arm64Line{
		Assembly: "ldp\tx28, x27, [sp], #80",
		Binary:   "a8c56ffc",
	}

	newBinary, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}
	if newBinary == "" {
		t.Fatal("expected non-empty binary, got empty (removed)")
	}

	// Verify transformation to signed-offset mode
	// Post-indexed has bits 25-23 = 001, signed-offset has 010
	// Set bit 24, clear bit 23, zero imm7
	// L bit (bit 22) = 1 for load, so result is a9406ffc not a9006ffc
	expected := "a9406ffc"
	if newBinary != expected {
		t.Errorf("expected binary %s, got %s", expected, newBinary)
	}
}

func TestTransformStackInstruction_SubSP(t *testing.T) {
	// Test explicit stack allocation: sub sp, sp, #80
	// Should be removed (return empty string)
	line := &arm64Line{
		Assembly: "sub\tsp, sp, #80",
		Binary:   "d10140ff",
	}

	newBinary, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}
	if newBinary != "" {
		t.Errorf("expected empty binary (removed), got %s", newBinary)
	}
}

func TestTransformStackInstruction_AddSP(t *testing.T) {
	// Test explicit stack deallocation: add sp, sp, #80
	// Should be removed (return empty string)
	line := &arm64Line{
		Assembly: "add\tsp, sp, #80",
		Binary:   "910140ff",
	}

	newBinary, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}
	if newBinary != "" {
		t.Errorf("expected empty binary (removed), got %s", newBinary)
	}
}

func TestTransformStackInstruction_RegularSTP(t *testing.T) {
	// Test regular signed-offset STP: stp x26, x25, [sp, #16]
	// Should NOT be transformed (no pre/post indexing)
	line := &arm64Line{
		Assembly: "stp\tx26, x25, [sp, #16]",
		Binary:   "a90167fa",
	}

	_, transformed := line.transformStackInstruction()
	if transformed {
		t.Error("regular signed-offset STP should not be transformed")
	}
}

func TestTransformStackInstruction_NonStackInstruction(t *testing.T) {
	// Test non-stack instruction: add x0, x1, x2
	// Should NOT be transformed
	line := &arm64Line{
		Assembly: "add\tx0, x1, x2",
		Binary:   "8b020020",
	}

	_, transformed := line.transformStackInstruction()
	if transformed {
		t.Error("non-stack instruction should not be transformed")
	}
}

func TestString_TransformedInstruction(t *testing.T) {
	// Test that String() properly outputs transformed instructions
	line := &arm64Line{
		Assembly: "stp\tx28, x27, [sp, #-80]!",
		Binary:   "a9bb6ffc",
	}

	output := line.String()

	// Should contain the transformed binary
	if output == "" {
		t.Error("expected non-empty output")
	}

	// Should contain [transformed] marker
	if !contains(output, "[transformed]") {
		t.Errorf("expected output to contain [transformed], got: %s", output)
	}

	// Should contain the new binary (a9006ffc)
	if !contains(output, "0xa9006ffc") {
		t.Errorf("expected output to contain transformed binary 0xa9006ffc, got: %s", output)
	}
}

func TestString_RemovedInstruction(t *testing.T) {
	// Test that String() returns empty for removed instructions
	line := &arm64Line{
		Assembly: "sub\tsp, sp, #80",
		Binary:   "d10140ff",
	}

	output := line.String()
	if output != "" {
		t.Errorf("expected empty output for removed instruction, got: %s", output)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestTransformStackInstruction_PreIndexedSTR(t *testing.T) {
	// Test single-register pre-indexed STR: str x25, [sp, #-80]!
	// Original binary: 0xf81b0ff9
	// STR encoding: imm9 at bits 20:12, mode at bits 11:10
	// Should transform to unscaled offset: imm9=0, mode=00
	line := &arm64Line{
		Assembly: "str\tx25, [sp, #-80]!",
		Binary:   "f81b0ff9",
	}

	newBinary, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}
	if newBinary == "" {
		t.Fatal("expected non-empty binary, got empty (removed)")
	}

	// Expected: zero imm9 (bits 20-12) and mode (bits 11-10)
	// 0xf81b0ff9 = 1111 1000 0001 1011 0000 1111 1111 1001
	// After:       1111 1000 0000 0000 0000 0011 1111 1001 = 0xf80003f9
	expected := "f80003f9"
	if newBinary != expected {
		t.Errorf("expected binary %s, got %s", expected, newBinary)
	}
}

func TestTransformStackInstruction_PostIndexedLDR(t *testing.T) {
	// Test single-register post-indexed LDR: ldr x25, [sp], #80
	// Original binary: 0xf8450419 -> need to compute actual
	// LDR post-indexed: imm9 at bits 20:12, mode 01 at bits 11:10
	// Should transform to unscaled offset: imm9=0, mode=00
	line := &arm64Line{
		Assembly: "ldr\tx25, [sp], #80",
		Binary:   "f84507f9",
	}

	newBinary, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}
	if newBinary == "" {
		t.Fatal("expected non-empty binary, got empty (removed)")
	}

	// Expected: zero imm9 (bits 20-12) and mode (bits 11-10)
	// Mode was 01, becomes 00; imm9 was non-zero, becomes 0
	// 0xf84503f9 with imm9 zeroed and mode zeroed
	expected := "f84003f9"
	if newBinary != expected {
		t.Errorf("expected binary %s, got %s", expected, newBinary)
	}
}

func TestTransformStackInstruction_PreIndexedSTRWithSpOffset(t *testing.T) {
	// When SpOffset is set (callee-save in pre-decrement prologue),
	// the transformed instruction should encode the offset.
	// SpOffset=432 > 255 (imm9 max), so scratch register path is used.
	line := &arm64Line{
		Assembly: "str\tx25, [sp, #-80]!",
		Binary:   "f81b0ff9",
		SpOffset: 432, // sub sp, sp, #432 follows
	}

	result, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}

	// SpOffset=432 overflows imm9 (max 255), so we get multi-line scratch base output
	if result == "" || result[0] != '\t' {
		t.Fatalf("expected multi-line output starting with tab, got: %q", result)
	}
	// ADD x16, sp, #432 → 0x9106c3f0
	if !contains(result, "0x9106c3f0") {
		t.Errorf("expected ADD x16, sp, #432 (0x9106c3f0) in output, got:\n%s", result)
	}
	// STR with x16 base → 0xf8000219
	if !contains(result, "0xf8000219") {
		t.Errorf("expected STR with x16 base (0xf8000219) in output, got:\n%s", result)
	}
}

func TestTransformStackInstruction_PreIndexedSTRWithSmallSpOffset(t *testing.T) {
	// SpOffset=200 fits in imm9 (max 255), so inline encoding is used.
	line := &arm64Line{
		Assembly: "str\tx25, [sp, #-80]!",
		Binary:   "f81b0ff9",
		SpOffset: 200,
	}

	newBinary, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}

	// Should encode imm9 = 200 = 0xc8 at bits 20:12, mode = 00 (unscaled)
	// 0xf80003f9 base + (0xc8 << 12) = 0xf80c83f9
	// Wait: 0xc8 << 12 = 0xc8000
	// 0xf8000000 | 0xc8000 | 0x3f9 = 0xf80c83f9
	expected := "f80c83f9"
	if newBinary != expected {
		t.Errorf("expected binary %s, got %s", expected, newBinary)
	}
}

func TestTransformStackInstruction_PostIndexedLDRWithSpOffset(t *testing.T) {
	// SpOffset=432 > 255 (imm9 max), so scratch register path is used.
	line := &arm64Line{
		Assembly: "ldr\tx25, [sp], #80",
		Binary:   "f84507f9",
		SpOffset: 432,
	}

	result, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}

	// SpOffset=432 overflows imm9 (max 255), so we get multi-line scratch base output
	if result == "" || result[0] != '\t' {
		t.Fatalf("expected multi-line output starting with tab, got: %q", result)
	}
	// ADD x16, sp, #432 → 0x9106c3f0
	if !contains(result, "0x9106c3f0") {
		t.Errorf("expected ADD x16, sp, #432 (0x9106c3f0) in output, got:\n%s", result)
	}
	// LDR with x16 base → 0xf8400219
	if !contains(result, "0xf8400219") {
		t.Errorf("expected LDR with x16 base (0xf8400219) in output, got:\n%s", result)
	}
}

func TestString_CalleeSaveWithSpOffset(t *testing.T) {
	// Regular signed-offset STP with SpOffset should get offset adjusted
	line := &arm64Line{
		Assembly: "stp\tx24, x23, [sp, #16]",
		Binary:   "a9015ff8",
		SpOffset: 432,
	}

	output := line.String()
	if output == "" {
		t.Fatal("expected non-empty output")
	}
	if !contains(output, "[offset adjusted]") {
		t.Errorf("expected [offset adjusted] marker, got: %s", output)
	}
	// Original imm7 = 16/8 = 2, new imm7 = (16+432)/8 = 56 = 0x38
	// Binary: a9015ff8 -> imm7 at bits 21:15
	// Original imm7 = 2 (bits 21:15 = 0000010)
	// New imm7 = 56 (bits 21:15 = 0111000)
	// a9015ff8 = 1010 1001 0000 0001 0101 1111 1111 1000
	// After:     1010 1001 0001 1100 0101 1111 1111 1000 = a91c5ff8
	if !contains(output, "0xa91c5ff8") {
		t.Errorf("expected adjusted binary 0xa91c5ff8, got: %s", output)
	}
}

// Constant pool tests

func TestConstPoolLabelRegex(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
	}{
		{"lCPI0_0:", true},
		{".LCPI0_0:", true},
		{"CPI0_0:", true},
		{"lCPI12_34:", true},
		{".LCPI99_99:", true},
		{"LBB0_1:", false},       // branch label, not const pool
		{".LBB0_1:", false},      // branch label, not const pool
		{"_my_function:", false}, // function name
		{"some_label:", false},   // generic label
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			matched := arm64ConstPoolLabel.MatchString(tt.input)
			if matched != tt.expected {
				t.Errorf("ConstPoolLabel(%q) = %v, want %v", tt.input, matched, tt.expected)
			}
		})
	}
}

func TestLongDirectiveRegex(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
		value    string
	}{
		{"\t.long\t0", true, "0"},
		{"\t.long\t1", true, "1"},
		{"\t.long\t0x3f800000", true, "0x3f800000"},
		{"  .long  12345", true, "12345"},
		{"\t.quad\t0", false, ""},    // quad, not long
		{"\tadd\tx0, x1", false, ""}, // not a directive
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			matches := arm64LongDirective.FindStringSubmatch(tt.input)
			matched := matches != nil
			if matched != tt.expected {
				t.Errorf("LongDirective(%q) matched = %v, want %v", tt.input, matched, tt.expected)
			}
			if matched && tt.value != "" && matches[1] != tt.value {
				t.Errorf("LongDirective(%q) value = %q, want %q", tt.input, matches[1], tt.value)
			}
		})
	}
}

func TestParseIntValue(t *testing.T) {
	tests := []struct {
		input    string
		expected uint64
	}{
		{"0", 0},
		{"1", 1},
		{"12345", 12345},
		{"0x0", 0},
		{"0x1", 1},
		{"0x3f800000", 0x3f800000},
		{"0xFFFFFFFF", 0xFFFFFFFF},
		{"0X10", 16},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := parseIntValue(tt.input)
			if result != tt.expected {
				t.Errorf("parseIntValue(%q) = %d, want %d", tt.input, result, tt.expected)
			}
		})
	}
}

func TestGoRegisterName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		// General purpose registers
		{"x0", "R0"},
		{"x9", "R9"},
		{"x30", "R30"},
		{"w0", "R0"},
		{"w15", "R15"},
		// NEON vector registers
		{"q0", "V0"},
		{"q15", "V15"},
		{"d0", "V0"},
		{"d31", "V31"},
		// Floating point registers
		{"s0", "F0"},
		{"s7", "F7"},
		// Mixed case
		{"X0", "R0"},
		{"Q0", "V0"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := goRegisterName(tt.input)
			if result != tt.expected {
				t.Errorf("goRegisterName(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestAdrpConstPoolRegex(t *testing.T) {
	tests := []struct {
		input      string
		expected   bool
		baseReg    string
		constLabel string
	}{
		{"adrp\tx9, lCPI0_0@PAGE", true, "x9", "0_0"},
		{"adrp\tx10, lCPI12_34@PAGE", true, "x10", "12_34"},
		{"adrp\tx0, .LCPI5_6@PAGE", true, "x0", "5_6"},
		{"adrp\tx9, some_symbol@PAGE", false, "", ""},
		{"ldr\tq0, [x9]", false, "", ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			matches := arm64AdrpConstPool.FindStringSubmatch(tt.input)
			matched := matches != nil
			if matched != tt.expected {
				t.Errorf("AdrpConstPool(%q) matched = %v, want %v", tt.input, matched, tt.expected)
			}
			if matched {
				if matches[1] != tt.baseReg {
					t.Errorf("AdrpConstPool(%q) baseReg = %q, want %q", tt.input, matches[1], tt.baseReg)
				}
				if matches[2] != tt.constLabel {
					t.Errorf("AdrpConstPool(%q) constLabel = %q, want %q", tt.input, matches[2], tt.constLabel)
				}
			}
		})
	}
}

func TestLdrConstPoolPageoffRegex(t *testing.T) {
	tests := []struct {
		input      string
		expected   bool
		destReg    string
		baseReg    string
		constLabel string
	}{
		{"ldr\tq1, [x9, lCPI0_0@PAGEOFF]", true, "q1", "x9", "0_0"},
		{"ldr\tq2, [x10, lCPI12_34@PAGEOFF]", true, "q2", "x10", "12_34"},
		{"ldr\td0, [x0, .LCPI5_6@PAGEOFF]", true, "d0", "x0", "5_6"},
		{"ldr\tx0, [x1, lCPI0_0@PAGEOFF]", true, "x0", "x1", "0_0"},
		{"ldr\tq0, [x0]", false, "", "", ""},
		{"str\tq0, [x9, lCPI0_0@PAGEOFF]", false, "", "", ""}, // store, not load
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			matches := arm64LdrConstPoolPageoff.FindStringSubmatch(tt.input)
			matched := matches != nil
			if matched != tt.expected {
				t.Errorf("LdrConstPoolPageoff(%q) matched = %v, want %v", tt.input, matched, tt.expected)
			}
			if matched {
				if matches[1] != tt.destReg {
					t.Errorf("LdrConstPoolPageoff(%q) destReg = %q, want %q", tt.input, matches[1], tt.destReg)
				}
				if matches[2] != tt.baseReg {
					t.Errorf("LdrConstPoolPageoff(%q) baseReg = %q, want %q", tt.input, matches[2], tt.baseReg)
				}
				if matches[3] != tt.constLabel {
					t.Errorf("LdrConstPoolPageoff(%q) constLabel = %q, want %q", tt.input, matches[3], tt.constLabel)
				}
			}
		})
	}
}

func TestArm64StpLdpScale(t *testing.T) {
	tests := []struct {
		name     string
		binary   uint64
		expected int
	}{
		{"W registers (opc=00, V=0)", 0x29000000, 4},
		{"X registers (opc=10, V=0)", 0xa9000000, 8},
		{"S registers (opc=00, V=1)", 0x2d000000, 4},
		{"D registers (opc=01, V=1)", 0x6d000000, 8},
		{"Q registers (opc=10, V=1)", 0xad000000, 16},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := arm64StpLdpScale(tt.binary)
			if got != tt.expected {
				t.Errorf("arm64StpLdpScale(0x%08x) = %d, want %d", tt.binary, got, tt.expected)
			}
		})
	}
}

func TestEmitSpOffsetAdjusted_LargeSpOffset_STP(t *testing.T) {
	// stp d13, d12, [sp] with SpOffset=10624 (the SDPA case)
	// Binary 6d0033ed: opc=01 V=1 → D registers, scale=8
	// newImm7 = 0 + 10624/8 = 1328, overflows imm7 max of 63
	line := &arm64Line{
		Assembly: "stp\td13, d12, [sp]",
		Binary:   "6d0033ed",
		SpOffset: 10624,
	}

	output := line.emitSpOffsetAdjusted()

	// Should contain scratch base ADD instructions
	if !contains(output, "scratch") {
		t.Errorf("expected scratch base in output, got:\n%s", output)
	}
	// 10624 = 0x2980: hi=2, lo=0x980=2432
	// ADD x16, sp, #2, lsl #12 → 0x91400bf0
	if !contains(output, "0x91400bf0") {
		t.Errorf("expected ADD x16, sp, #2, lsl #12 (0x91400bf0) in output, got:\n%s", output)
	}
	// ADD x16, x16, #2432 → 0x91260210
	if !contains(output, "0x91260210") {
		t.Errorf("expected ADD x16, x16, #2432 (0x91260210) in output, got:\n%s", output)
	}
	// STP with x16 base (Rn changed from 31 to 16): 0x6d00320d
	if !contains(output, "0x6d00320d") {
		t.Errorf("expected STP with x16 base (0x6d00320d) in output, got:\n%s", output)
	}
}

func TestEmitSpOffsetAdjusted_SmallSpOffset_STP(t *testing.T) {
	// Verify that small offsets still work (no regression)
	// stp d13, d12, [sp] with SpOffset=32 → newImm7 = 0+32/8 = 4, fits
	line := &arm64Line{
		Assembly: "stp\td13, d12, [sp]",
		Binary:   "6d0033ed",
		SpOffset: 32,
	}

	output := line.emitSpOffsetAdjusted()

	if contains(output, "scratch") {
		t.Errorf("small offset should not use scratch, got:\n%s", output)
	}
	if !contains(output, "[offset adjusted]") {
		t.Errorf("expected [offset adjusted] marker, got:\n%s", output)
	}
}

func TestTransformStackInstruction_PreIndexedSTP_LargeSpOffset(t *testing.T) {
	// stp x28, x27, [sp, #-80]! with SpOffset=10624
	// Scale=8 (X registers), scaledOffset=10624/8=1328, overflows imm7
	line := &arm64Line{
		Assembly: "stp\tx28, x27, [sp, #-80]!",
		Binary:   "a9bb6ffc",
		SpOffset: 10624,
	}

	result, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}
	// Should be multi-line (starts with \t)
	if result == "" || result[0] != '\t' {
		t.Fatalf("expected multi-line output starting with tab, got: %q", result)
	}
	// Should contain scratch base instructions
	if !contains(result, "scratch") {
		t.Errorf("expected scratch base in output, got:\n%s", result)
	}
	// ADD x16, sp, #2, lsl #12 → 0x91400bf0
	if !contains(result, "0x91400bf0") {
		t.Errorf("expected ADD x16 hi in output, got:\n%s", result)
	}
	// ADD x16, x16, #2432 → 0x91260210
	if !contains(result, "0x91260210") {
		t.Errorf("expected ADD x16 lo in output, got:\n%s", result)
	}
}

func TestTransformStackInstruction_PostIndexedLDP_LargeSpOffset(t *testing.T) {
	// ldp x28, x27, [sp], #80 with SpOffset=10624
	// Scale=8 (X registers), scaledOffset=1328, overflows imm7
	line := &arm64Line{
		Assembly: "ldp\tx28, x27, [sp], #80",
		Binary:   "a8c56ffc",
		SpOffset: 10624,
	}

	result, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}
	if result == "" || result[0] != '\t' {
		t.Fatalf("expected multi-line output starting with tab, got: %q", result)
	}
	if !contains(result, "scratch") {
		t.Errorf("expected scratch base in output, got:\n%s", result)
	}
}

func TestTransformStackInstruction_PreIndexedSTR_LargeSpOffset(t *testing.T) {
	// str x25, [sp, #-80]! with SpOffset=10624
	// imm9 max is 255, SpOffset=10624 overflows
	line := &arm64Line{
		Assembly: "str\tx25, [sp, #-80]!",
		Binary:   "f81b0ff9",
		SpOffset: 10624,
	}

	result, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}
	if result == "" || result[0] != '\t' {
		t.Fatalf("expected multi-line output starting with tab, got: %q", result)
	}
	if !contains(result, "scratch") {
		t.Errorf("expected scratch base in output, got:\n%s", result)
	}
}

func TestString_LargeSpOffset_MultiLine(t *testing.T) {
	// Verify String() correctly passes through multi-line transformed output
	line := &arm64Line{
		Assembly: "stp\td13, d12, [sp]",
		Binary:   "6d0033ed",
		SpOffset: 10624,
	}

	output := line.String()
	if output == "" {
		t.Fatal("expected non-empty output")
	}
	if !contains(output, "scratch") {
		t.Errorf("expected scratch base in String() output, got:\n%s", output)
	}
}

func TestConstPoolStruct(t *testing.T) {
	pool := &ConstPool{
		Label: "CPI0_0",
		Data:  []uint32{0, 1, 2, 3},
	}

	if pool.Label != "CPI0_0" {
		t.Errorf("Label = %q, want %q", pool.Label, "CPI0_0")
	}
	if len(pool.Data) != 4 {
		t.Errorf("len(Data) = %d, want 4", len(pool.Data))
	}
}
