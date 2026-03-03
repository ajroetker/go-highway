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
	"os"
	"path/filepath"
	"testing"
)

// writeTestAssembly writes content to a temp .s file and returns the path.
func writeTestAssembly(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "test.s")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write test assembly: %v", err)
	}
	return path
}

// wantPool describes expected constant pool contents for test assertions.
type wantPool struct {
	data []uint32
	size int
}

// checkConstPools verifies that actual constant pools match expectations.
func checkConstPools(t *testing.T, actual map[string]wantPool, expected map[string]wantPool) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("got %d pools, want %d", len(actual), len(expected))
	}
	for label, want := range expected {
		got, ok := actual[label]
		if !ok {
			t.Fatalf("missing pool %q", label)
		}
		if got.size != want.size {
			t.Errorf("pool %q: size = %d, want %d", label, got.size, want.size)
		}
		if len(got.data) != len(want.data) {
			t.Fatalf("pool %q: len(data) = %d, want %d", label, len(got.data), len(want.data))
		}
		for i, v := range want.data {
			if got.data[i] != v {
				t.Errorf("pool %q: data[%d] = 0x%08x, want 0x%08x", label, i, got.data[i], v)
			}
		}
	}
}

// --- ARM64 byte directive tests ---

func TestARM64ByteDirectiveRegex(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
		value    string
	}{
		{"\t.byte\t0x3f", true, "0x3f"},
		{"\t.byte\t255", true, "255"},
		{"\t.byte\t0xFF", true, "0xFF"},
		{"  .byte 0x00", true, "0x00"},
		{"\t.long\t0x3f800000", false, ""},
		{"\tmov x0, x1", false, ""},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			matches := arm64ByteDirective.FindStringSubmatch(tt.input)
			matched := matches != nil
			if matched != tt.expected {
				t.Errorf("arm64ByteDirective(%q) matched = %v, want %v", tt.input, matched, tt.expected)
			}
			if matched && matches[1] != tt.value {
				t.Errorf("arm64ByteDirective(%q) value = %q, want %q", tt.input, matches[1], tt.value)
			}
		})
	}
}

func TestARM64ParseAssemblyByteDirective(t *testing.T) {
	// ARM64 uses macOS-style lCPI labels (lowercase l) which normalize to CPI
	// after stripping the leading "l" prefix.
	tests := []struct {
		name      string
		assembly  string
		wantPools map[string]wantPool
	}{
		{
			name: "4 bytes into 1 word little-endian",
			assembly: "\t.section\t.rodata\n" +
				"lCPI0_0:\n" +
				"\t.byte\t0x01\n" +
				"\t.byte\t0x02\n" +
				"\t.byte\t0x03\n" +
				"\t.byte\t0x04\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x04030201}, size: 4},
			},
		},
		{
			name: "8 bytes into 2 words",
			assembly: "\t.section\t.rodata\n" +
				"lCPI0_0:\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x80\n" +
				"\t.byte\t0x3f\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x40\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x3f800000, 0x40000000}, size: 8},
			},
		},
		{
			name: "partial bytes flushed at EOF",
			assembly: "\t.section\t.rodata\n" +
				"lCPI0_0:\n" +
				"\t.byte\t0xAA\n" +
				"\t.byte\t0xBB\n" +
				"\t.byte\t0xCC\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x00CCBBAA}, size: 4},
			},
		},
		{
			name: "bytes flushed before long",
			assembly: "\t.section\t.rodata\n" +
				"lCPI0_0:\n" +
				"\t.byte\t0x01\n" +
				"\t.byte\t0x02\n" +
				"\t.long\t0xDEADBEEF\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x00000201, 0xDEADBEEF}, size: 8},
			},
		},
		{
			name: "partial bytes flushed at new label",
			assembly: "\t.section\t.rodata\n" +
				"lCPI0_0:\n" +
				"\t.byte\t0xFF\n" +
				"\t.byte\t0xFE\n" +
				"lCPI0_1:\n" +
				"\t.long\t0x12345678\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x0000FEFF}, size: 4},
				"CPI0_1": {data: []uint32{0x12345678}, size: 4},
			},
		},
		{
			name: "bytes flushed before quad",
			assembly: "\t.section\t.rodata\n" +
				"lCPI0_0:\n" +
				"\t.byte\t0x0A\n" +
				"\t.byte\t0x0B\n" +
				"\t.byte\t0x0C\n" +
				"\t.quad\t0x0000000100000002\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x000C0B0A, 0x00000002, 0x00000001}, size: 12},
			},
		},
		{
			name: "partial bytes flushed at section change",
			assembly: "\t.section\t.rodata\n" +
				"lCPI0_0:\n" +
				"\t.byte\t0x11\n" +
				"\t.byte\t0x22\n" +
				"\t.section\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x00002211}, size: 4},
			},
		},
	}

	p := &ARM64Parser{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeTestAssembly(t, tt.assembly)
			_, _, _, _, _, constPools, err := p.parseAssembly(path, "linux")
			if err != nil {
				t.Fatalf("parseAssembly failed: %v", err)
			}
			actual := make(map[string]wantPool)
			for label, pool := range constPools {
				actual[label] = wantPool{data: pool.Data, size: len(pool.Data) * 4}
			}
			checkConstPools(t, actual, tt.wantPools)
		})
	}
}

// --- AMD64 byte directive tests ---

func TestAMD64ByteDirectiveRegex(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
		value    string
	}{
		{"\t.byte\t0x3f", true, "0x3f"},
		{"\t.byte\t255", true, "255"},
		{"\t.byte\t0xFF", true, "0xFF"},
		{"  .byte 0x00", true, "0x00"},
		{"\t.long\t0x3f800000", false, ""},
		{"\tvmovaps %ymm0, %ymm1", false, ""},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			matches := amd64ByteDirective.FindStringSubmatch(tt.input)
			matched := matches != nil
			if matched != tt.expected {
				t.Errorf("amd64ByteDirective(%q) matched = %v, want %v", tt.input, matched, tt.expected)
			}
			if matched && matches[1] != tt.value {
				t.Errorf("amd64ByteDirective(%q) value = %q, want %q", tt.input, matches[1], tt.value)
			}
		})
	}
}

func TestAMD64ParseAssemblyByteDirective(t *testing.T) {
	tests := []struct {
		name      string
		assembly  string
		wantPools map[string]wantPool
	}{
		{
			name: "4 bytes into 1 word little-endian",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0x01\n" +
				"\t.byte\t0x02\n" +
				"\t.byte\t0x03\n" +
				"\t.byte\t0x04\n" +
				"\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x04030201}, size: 4},
			},
		},
		{
			name: "8 bytes into 2 words",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x80\n" +
				"\t.byte\t0x3f\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x40\n" +
				"\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x3f800000, 0x40000000}, size: 8},
			},
		},
		{
			name: "partial bytes flushed at EOF",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0xAA\n" +
				"\t.byte\t0xBB\n" +
				"\t.byte\t0xCC\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x00CCBBAA}, size: 4},
			},
		},
		{
			name: "bytes flushed before long",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0x01\n" +
				"\t.byte\t0x02\n" +
				"\t.long\t0xDEADBEEF\n" +
				"\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x00000201, 0xDEADBEEF}, size: 8},
			},
		},
		{
			name: "partial bytes flushed at new label",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0xFF\n" +
				"\t.byte\t0xFE\n" +
				".LCPI0_1:\n" +
				"\t.long\t0x12345678\n" +
				"\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x0000FEFF}, size: 4},
				"CPI0_1": {data: []uint32{0x12345678}, size: 4},
			},
		},
		{
			name: "bytes flushed before quad",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0x0A\n" +
				"\t.byte\t0x0B\n" +
				"\t.byte\t0x0C\n" +
				"\t.quad\t0x0000000100000002\n" +
				"\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x000C0B0A, 0x00000002, 0x00000001}, size: 12},
			},
		},
		{
			name: "partial bytes flushed at section change",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0x11\n" +
				"\t.byte\t0x22\n" +
				"\t.section\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x00002211}, size: 4},
			},
		},
	}

	p := &AMD64Parser{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeTestAssembly(t, tt.assembly)
			_, _, _, constPools, err := p.parseAssembly(path, "linux")
			if err != nil {
				t.Fatalf("parseAssembly failed: %v", err)
			}
			actual := make(map[string]wantPool)
			for label, pool := range constPools {
				actual[label] = wantPool{data: pool.Data, size: len(pool.Data) * 4}
			}
			checkConstPools(t, actual, tt.wantPools)
		})
	}
}

// --- Loong64 byte directive tests ---

func TestLoong64ByteDirectiveRegex(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
		value    string
	}{
		{"\t.byte\t0x3f", true, "0x3f"},
		{"\t.byte\t255", true, "255"},
		{"\t.byte\t0xFF", true, "0xFF"},
		{"  .byte 0x00", true, "0x00"},
		{"\t.word\t0x3f800000", false, ""},
		{"\tpcaddu12i $a0, %pc_hi20(.LCPI0_0)", false, ""},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			matches := loong64ByteDirective.FindStringSubmatch(tt.input)
			matched := matches != nil
			if matched != tt.expected {
				t.Errorf("loong64ByteDirective(%q) matched = %v, want %v", tt.input, matched, tt.expected)
			}
			if matched && matches[1] != tt.value {
				t.Errorf("loong64ByteDirective(%q) value = %q, want %q", tt.input, matches[1], tt.value)
			}
		})
	}
}

func TestLoong64ParseAssemblyByteDirective(t *testing.T) {
	// Loong64 doesn't use a rodata section flag; constant pool parsing
	// begins at the label and ends at a function name or section directive.
	tests := []struct {
		name      string
		assembly  string
		wantPools map[string]wantPool
	}{
		{
			name: "4 bytes into 1 word little-endian",
			assembly: ".LCPI0_0:\n" +
				"\t.byte\t0x01\n" +
				"\t.byte\t0x02\n" +
				"\t.byte\t0x03\n" +
				"\t.byte\t0x04\n" +
				"\t.section\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x04030201}, size: 4},
			},
		},
		{
			name: "8 bytes into 2 words",
			assembly: ".LCPI0_0:\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x80\n" +
				"\t.byte\t0x3f\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x40\n" +
				"\t.section\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x3f800000, 0x40000000}, size: 8},
			},
		},
		{
			name: "partial bytes flushed at EOF",
			assembly: ".LCPI0_0:\n" +
				"\t.byte\t0xAA\n" +
				"\t.byte\t0xBB\n" +
				"\t.byte\t0xCC\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x00CCBBAA}, size: 4},
			},
		},
		{
			name: "bytes flushed before word",
			assembly: ".LCPI0_0:\n" +
				"\t.byte\t0x01\n" +
				"\t.byte\t0x02\n" +
				"\t.word\t0xDEADBEEF\n" +
				"\t.section\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x00000201, 0xDEADBEEF}, size: 8},
			},
		},
		{
			name: "partial bytes flushed at new label",
			assembly: ".LCPI0_0:\n" +
				"\t.byte\t0xFF\n" +
				"\t.byte\t0xFE\n" +
				".LCPI0_1:\n" +
				"\t.word\t0x12345678\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x0000FEFF}, size: 4},
				"CPI0_1": {data: []uint32{0x12345678}, size: 4},
			},
		},
		{
			name: "bytes flushed before dword",
			assembly: ".LCPI0_0:\n" +
				"\t.byte\t0x0A\n" +
				"\t.byte\t0x0B\n" +
				"\t.byte\t0x0C\n" +
				"\t.dword\t0x0000000100000002\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x000C0B0A, 0x00000002, 0x00000001}, size: 12},
			},
		},
	}

	p := &Loong64Parser{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeTestAssembly(t, tt.assembly)
			_, _, constPools, err := p.parseAssembly(path)
			if err != nil {
				t.Fatalf("parseAssembly failed: %v", err)
			}
			actual := make(map[string]wantPool)
			for label, pool := range constPools {
				actual[label] = wantPool{data: pool.Data, size: len(pool.Data) * 4}
			}
			checkConstPools(t, actual, tt.wantPools)
		})
	}
}

// --- RISCV64 byte directive tests ---

func TestRISCV64ByteDirectiveRegex(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
		value    string
	}{
		{"\t.byte\t0x3f", true, "0x3f"},
		{"\t.byte\t255", true, "255"},
		{"\t.byte\t0xFF", true, "0xFF"},
		{"  .byte 0x00", true, "0x00"},
		{"\t.word\t0x3f800000", false, ""},
		{"\tauipc a0, %pcrel_hi(.LCPI0_0)", false, ""},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			matches := riscv64ByteDirective.FindStringSubmatch(tt.input)
			matched := matches != nil
			if matched != tt.expected {
				t.Errorf("riscv64ByteDirective(%q) matched = %v, want %v", tt.input, matched, tt.expected)
			}
			if matched && matches[1] != tt.value {
				t.Errorf("riscv64ByteDirective(%q) value = %q, want %q", tt.input, matches[1], tt.value)
			}
		})
	}
}

func TestRISCV64ParseAssemblyByteDirective(t *testing.T) {
	tests := []struct {
		name      string
		assembly  string
		wantPools map[string]wantPool
	}{
		{
			name: "4 bytes into 1 word little-endian",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0x01\n" +
				"\t.byte\t0x02\n" +
				"\t.byte\t0x03\n" +
				"\t.byte\t0x04\n" +
				"\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x04030201}, size: 4},
			},
		},
		{
			name: "8 bytes into 2 words",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x80\n" +
				"\t.byte\t0x3f\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x00\n" +
				"\t.byte\t0x40\n" +
				"\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x3f800000, 0x40000000}, size: 8},
			},
		},
		{
			name: "partial bytes flushed at EOF",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0xAA\n" +
				"\t.byte\t0xBB\n" +
				"\t.byte\t0xCC\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x00CCBBAA}, size: 4},
			},
		},
		{
			name: "bytes flushed before word",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0x01\n" +
				"\t.byte\t0x02\n" +
				"\t.word\t0xDEADBEEF\n" +
				"\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x00000201, 0xDEADBEEF}, size: 8},
			},
		},
		{
			name: "partial bytes flushed at new label",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0xFF\n" +
				"\t.byte\t0xFE\n" +
				".LCPI0_1:\n" +
				"\t.word\t0x12345678\n" +
				"\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x0000FEFF}, size: 4},
				"CPI0_1": {data: []uint32{0x12345678}, size: 4},
			},
		},
		{
			name: "bytes flushed before dword",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0x0A\n" +
				"\t.byte\t0x0B\n" +
				"\t.byte\t0x0C\n" +
				"\t.dword\t0x0000000100000002\n" +
				"\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x000C0B0A, 0x00000002, 0x00000001}, size: 12},
			},
		},
		{
			name: "partial bytes flushed at section change",
			assembly: "\t.section\t.rodata\n" +
				".LCPI0_0:\n" +
				"\t.byte\t0x11\n" +
				"\t.byte\t0x22\n" +
				"\t.section\t.text\n",
			wantPools: map[string]wantPool{
				"CPI0_0": {data: []uint32{0x00002211}, size: 4},
			},
		},
	}

	p := &RISCV64Parser{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeTestAssembly(t, tt.assembly)
			_, _, constPools, err := p.parseAssembly(path)
			if err != nil {
				t.Fatalf("parseAssembly failed: %v", err)
			}
			actual := make(map[string]wantPool)
			for label, pool := range constPools {
				actual[label] = wantPool{data: pool.Data, size: len(pool.Data) * 4}
			}
			checkConstPools(t, actual, tt.wantPools)
		})
	}
}
