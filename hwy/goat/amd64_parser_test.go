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

func TestShouldSkip_StackAllocation(t *testing.T) {
	// Test stack allocation: subq $496, %rsp
	// Should be skipped (Go handles frame allocation via TEXT directive)
	line := &amd64Line{
		Assembly: "subq\t$496, %rsp",
		Binary:   []string{"48", "81", "ec", "f0", "01", "00", "00"},
	}

	if !line.shouldSkip() {
		t.Error("expected subq $N, %rsp to be skipped")
	}
}

func TestShouldSkip_StackDeallocation(t *testing.T) {
	// Test stack deallocation: addq $496, %rsp
	// Should be skipped
	line := &amd64Line{
		Assembly: "addq\t$496, %rsp",
		Binary:   []string{"48", "81", "c4", "f0", "01", "00", "00"},
	}

	if !line.shouldSkip() {
		t.Error("expected addq $N, %rsp to be skipped")
	}
}

func TestShouldSkip_CalleeSavedPush(t *testing.T) {
	// Test callee-saved register push: pushq %rbx
	// Should be skipped (Go doesn't use C calling convention)
	tests := []struct {
		name     string
		assembly string
	}{
		{"rbx", "pushq\t%rbx"},
		{"rbp", "pushq\t%rbp"},
		{"r12", "pushq\t%r12"},
		{"r13", "pushq\t%r13"},
		{"r14", "pushq\t%r14"},
		{"r15", "pushq\t%r15"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			line := &amd64Line{
				Assembly: tt.assembly,
				Binary:   []string{"53"}, // placeholder
			}
			if !line.shouldSkip() {
				t.Errorf("expected %s to be skipped", tt.assembly)
			}
		})
	}
}

func TestShouldSkip_CalleeSavedPop(t *testing.T) {
	// Test callee-saved register pop: popq %rbx
	// Should be skipped
	tests := []struct {
		name     string
		assembly string
	}{
		{"rbx", "popq\t%rbx"},
		{"rbp", "popq\t%rbp"},
		{"r12", "popq\t%r12"},
		{"r13", "popq\t%r13"},
		{"r14", "popq\t%r14"},
		{"r15", "popq\t%r15"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			line := &amd64Line{
				Assembly: tt.assembly,
				Binary:   []string{"5b"}, // placeholder
			}
			if !line.shouldSkip() {
				t.Errorf("expected %s to be skipped", tt.assembly)
			}
		})
	}
}

func TestShouldSkip_RegularInstruction(t *testing.T) {
	// Test regular instruction: movq %rax, %rbx
	// Should NOT be skipped
	line := &amd64Line{
		Assembly: "movq\t%rax, %rbx",
		Binary:   []string{"48", "89", "c3"},
	}

	if line.shouldSkip() {
		t.Error("regular movq instruction should not be skipped")
	}
}

func TestShouldSkip_VectorInstruction(t *testing.T) {
	// Test vector instruction: vmovups %ymm0, 464(%rsp)
	// Should NOT be skipped (it uses the stack frame for spilling)
	line := &amd64Line{
		Assembly: "vmovups\t%ymm0, 464(%rsp)",
		Binary:   []string{"c5", "fc", "11", "84", "24", "d0", "01", "00", "00"},
	}

	if line.shouldSkip() {
		t.Error("vmovups spill instruction should not be skipped")
	}
}

func TestShouldSkip_NonCalleeSavedPush(t *testing.T) {
	// Test push of non-callee-saved register: pushq %rax
	// Should NOT be skipped (not callee-saved, might be intentional)
	line := &amd64Line{
		Assembly: "pushq\t%rax",
		Binary:   []string{"50"},
	}

	if line.shouldSkip() {
		t.Error("pushq %rax should not be skipped (not callee-saved)")
	}
}

// Constant pool tests

func TestAmd64ConstPoolLabelRegex(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
	}{
		{".LCPI0_0:", true},
		{".LCPI12_34:", true},
		{"LCPI0_0:", true},   // macOS style (no dot)
		{"LCPI99_99:", true}, // macOS style
		{".LBB0_1:", false},  // branch label, not const pool
		{"LBB0_1:", false},   // branch label
		{"_my_function:", false},
		{"some_label:", false},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			matched := amd64ConstPoolLabel.MatchString(tt.input)
			if matched != tt.expected {
				t.Errorf("ConstPoolLabel(%q) = %v, want %v", tt.input, matched, tt.expected)
			}
		})
	}
}

func TestAmd64RIPRelativeConstPoolRegex(t *testing.T) {
	tests := []struct {
		input      string
		expected   bool
		constLabel string
	}{
		{".LCPI0_0(%rip)", true, "0_0"},
		{".LCPI12_34(%rip)", true, "12_34"},
		{"LCPI0_0(%rip)", true, "0_0"}, // macOS style
		{"vmovaps .LCPI0_0(%rip), %ymm0", true, "0_0"},
		{"movq .LCPI5_6(%rip), %xmm0", true, "5_6"},
		{"leaq .LCPI0_0(%rip), %rax", true, "0_0"},
		{"movq %rax, %rbx", false, ""},
		{"(%rip)", false, ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			matches := amd64RIPRelativeConstPool.FindStringSubmatch(tt.input)
			matched := matches != nil
			if matched != tt.expected {
				t.Errorf("RIPRelativeConstPool(%q) matched = %v, want %v", tt.input, matched, tt.expected)
			}
			if matched && tt.constLabel != "" && matches[1] != tt.constLabel {
				t.Errorf("RIPRelativeConstPool(%q) label = %q, want %q", tt.input, matches[1], tt.constLabel)
			}
		})
	}
}

func TestAmd64Line_HasConstPoolRef(t *testing.T) {
	tests := []struct {
		name     string
		assembly string
		expected bool
	}{
		{"vmovaps with const pool", "vmovaps\t.LCPI0_0(%rip), %ymm0", true},
		{"movq with const pool", "movq\t.LCPI5_6(%rip), %xmm0", true},
		{"leaq with const pool", "leaq\t.LCPI0_0(%rip), %rax", true},
		{"regular movq", "movq\t%rax, %rbx", false},
		{"stack access", "movq\t16(%rsp), %rax", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			line := &amd64Line{Assembly: tt.assembly}
			if line.hasConstPoolRef() != tt.expected {
				t.Errorf("hasConstPoolRef(%q) = %v, want %v", tt.assembly, line.hasConstPoolRef(), tt.expected)
			}
		})
	}
}

func TestAmd64Line_GetConstPoolLabel(t *testing.T) {
	tests := []struct {
		name     string
		assembly string
		expected string
	}{
		{"vmovaps", "vmovaps\t.LCPI0_0(%rip), %ymm0", "CPI0_0"},
		{"movq", "movq\t.LCPI12_34(%rip), %xmm0", "CPI12_34"},
		{"macOS style", "vmovaps\tLCPI5_6(%rip), %ymm0", "CPI5_6"},
		{"no const pool", "movq\t%rax, %rbx", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			line := &amd64Line{Assembly: tt.assembly}
			result := line.getConstPoolLabel()
			if result != tt.expected {
				t.Errorf("getConstPoolLabel(%q) = %q, want %q", tt.assembly, result, tt.expected)
			}
		})
	}
}

func TestAmd64RewriteConstPoolRef(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		constLabel string
		expected   string
	}{
		{
			name:       "vmovaps ymm",
			input:      "vmovaps\t.LCPI0_0(%rip), %ymm0",
			constLabel: "CPI0_0",
			expected:   "\tVMOVAPS CPI0_0<>(SB), Y0\t// vmovaps\t.LCPI0_0(%rip), %ymm0\n",
		},
		{
			name:       "vmovdqa xmm",
			input:      "vmovdqa\t.LCPI1_2(%rip), %xmm5",
			constLabel: "CPI1_2",
			expected:   "\tVMOVDQA CPI1_2<>(SB), X5\t// vmovdqa\t.LCPI1_2(%rip), %xmm5\n",
		},
		{
			name:       "vbroadcastss",
			input:      "vbroadcastss\t.LCPI0_0(%rip), %ymm1",
			constLabel: "CPI0_0",
			expected:   "\tVBROADCASTSS CPI0_0<>(SB), Y1\t// vbroadcastss\t.LCPI0_0(%rip), %ymm1\n",
		},
		{
			name:       "movq to xmm",
			input:      "movq\t.LCPI0_0(%rip), %xmm0",
			constLabel: "CPI0_0",
			expected:   "\tMOVQ CPI0_0<>(SB), X0\t// movq\t.LCPI0_0(%rip), %xmm0\n",
		},
		{
			name:       "leaq",
			input:      "leaq\t.LCPI0_0(%rip), %rax",
			constLabel: "CPI0_0",
			expected:   "\tLEAQ CPI0_0<>(SB), AX\t// leaq\t.LCPI0_0(%rip), %rax\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := amd64RewriteConstPoolRef(tt.input, tt.constLabel)
			if result != tt.expected {
				t.Errorf("amd64RewriteConstPoolRef(%q, %q) = %q, want %q",
					tt.input, tt.constLabel, result, tt.expected)
			}
		})
	}
}

func TestAmd64ConstPoolStruct(t *testing.T) {
	pool := &ConstPool{
		Label: "CPI0_0",
		Data:  []uint32{0x3f800000, 0x40000000, 0x40400000, 0x40800000},
	}

	if pool.Label != "CPI0_0" {
		t.Errorf("Label = %q, want %q", pool.Label, "CPI0_0")
	}
	if len(pool.Data) != 4 {
		t.Errorf("len(Data) = %d, want 4", len(pool.Data))
	}
}
