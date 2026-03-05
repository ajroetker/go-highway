// Copyright 2022 gorse Project Authors
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
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// Shared regex patterns for parsing assembly and objdump output.
// These are identical across all architecture parsers.
var (
	attributeLine = regexp.MustCompile(`^\s+\..+$`)
	nameLine      = regexp.MustCompile(`^\w+:.+$`)
	symbolLine    = regexp.MustCompile(`^\w+\s+<\w+>:$`)
	dataLine      = regexp.MustCompile(`^\w+:\s+\w+\s+.+$`)
)

// Note: Each architecture parser defines its own Line type internally.
// AMD64 uses Binary []string, others use Binary string.
// The parsers handle this internally and don't expose Line in the interface.

// ArchParser defines the interface for architecture-specific parsing and code generation.
type ArchParser interface {
	// Name returns the architecture name (e.g., "amd64", "arm64")
	Name() string

	// BuildTags returns the Go build constraint for generated files
	BuildTags() string

	// BuildTarget returns the clang target triple for the given OS
	BuildTarget(goos string) string

	// CompilerFlags returns architecture-specific compiler flags (e.g., -ffixed-x18 for arm64)
	CompilerFlags() []string

	// Prologue returns C parser prologue with architecture-specific type definitions
	Prologue() string

	// TranslateAssembly parses assembly, objdump, and generates Go assembly.
	// This encapsulates the full translation pipeline so Line types stay internal.
	TranslateAssembly(t *TranslateUnit, functions []Function) error
}

// parsers holds the registered architecture parsers
var parsers = map[string]ArchParser{}

// RegisterParser registers an architecture parser
func RegisterParser(arch string, p ArchParser) {
	parsers[arch] = p
}

// GetParser returns the parser for the given architecture
func GetParser(arch string) (ArchParser, error) {
	if p, ok := parsers[arch]; ok {
		return p, nil
	}
	return nil, fmt.Errorf("unsupported architecture: %s (available: amd64, arm64, loong64, riscv64)", arch)
}

// ConstPool represents a constant pool entry with its label and data.
// Used by all architecture parsers for .long/.word/.quad/.dword data directives.
type ConstPool struct {
	Label string   // e.g., "CPI0_0" (normalized, without leading dots or L prefix)
	Data  []uint32 // Data as 32-bit words
}

// normalizeLabel strips the leading "." and "L"/"l" prefixes from compiler-generated
// labels. Linux uses ".LBB0_2", macOS uses "LBB0_2" or "lCPI0_0"; Go assembler wants "BB0_2"/"CPI0_0".
func normalizeLabel(label string) string {
	label = strings.TrimPrefix(label, ".")
	label = strings.TrimPrefix(label, "L")
	label = strings.TrimPrefix(label, "l")
	return label
}

// parseIntValue parses a decimal or hex integer value from a string.
func parseIntValue(s string) uint64 {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X") {
		val, _ := strconv.ParseUint(s[2:], 16, 64)
		return val
	}
	val, _ := strconv.ParseUint(s, 10, 64)
	return val
}

// ConstPoolAccumulator manages the state machine for accumulating constant pool
// bytes from .byte/.long/.quad/.word/.dword directives into 32-bit words.
type ConstPoolAccumulator struct {
	pools map[string]*ConstPool
	cur   *ConstPool
	accum uint32 // accumulates .byte values into 32-bit words (little-endian)
	count int    // number of bytes accumulated (0-3)
}

// NewConstPoolAccumulator creates a new accumulator with an empty pool map.
func NewConstPoolAccumulator() *ConstPoolAccumulator {
	return &ConstPoolAccumulator{pools: make(map[string]*ConstPool)}
}

// flushBytes writes any partially accumulated bytes as a uint32 word.
func (a *ConstPoolAccumulator) flushBytes() {
	if a.count > 0 && a.cur != nil {
		a.cur.Data = append(a.cur.Data, a.accum)
		a.accum = 0
		a.count = 0
	}
}

// FinishPool saves the current pool (if any) to the map and resets state.
func (a *ConstPoolAccumulator) FinishPool() {
	if a.cur != nil {
		a.flushBytes()
		if len(a.cur.Data) > 0 {
			a.pools[a.cur.Label] = a.cur
		}
		a.cur = nil
	}
}

// StartPool finishes any in-progress pool and begins a new one with the given label.
func (a *ConstPoolAccumulator) StartPool(label string) {
	a.FinishPool()
	a.cur = &ConstPool{Label: label}
}

// Active returns true if a constant pool is currently being accumulated.
func (a *ConstPoolAccumulator) Active() bool {
	return a.cur != nil
}

// AddLong adds a 32-bit value (.long or .word directive).
func (a *ConstPoolAccumulator) AddLong(val uint64) {
	if a.cur == nil {
		return
	}
	a.flushBytes()
	a.cur.Data = append(a.cur.Data, uint32(val))
}

// AddQuad adds a 64-bit value as two 32-bit words in little-endian order (.quad or .dword directive).
func (a *ConstPoolAccumulator) AddQuad(val uint64) {
	if a.cur == nil {
		return
	}
	a.flushBytes()
	a.cur.Data = append(a.cur.Data, uint32(val), uint32(val>>32))
}

// AccumulateByte adds a single byte, flushing when 4 bytes have been collected.
func (a *ConstPoolAccumulator) AccumulateByte(val uint64) {
	if a.cur == nil {
		return
	}
	a.accum |= uint32(val&0xFF) << (a.count * 8)
	a.count++
	if a.count == 4 {
		a.cur.Data = append(a.cur.Data, a.accum)
		a.accum = 0
		a.count = 0
	}
}

// AccumulateAscii parses a C-style .ascii string literal and accumulates the bytes.
// Handles standard escape sequences: \NNN (octal), \xHH (hex), \t, \n, \r, \b, \f, \\, \", \0.
func (a *ConstPoolAccumulator) AccumulateAscii(s string) {
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' && i+1 < len(s) {
			i++
			switch {
			case s[i] >= '0' && s[i] <= '7':
				// Octal escape: 1-3 digits
				val := uint64(s[i] - '0')
				for j := 1; j < 3 && i+1 < len(s) && s[i+1] >= '0' && s[i+1] <= '7'; j++ {
					i++
					val = val*8 + uint64(s[i]-'0')
				}
				a.AccumulateByte(val)
			case s[i] == 'x' && i+1 < len(s):
				// Hex escape: \xHH
				i++
				val := uint64(0)
				for j := 0; j < 2 && i < len(s); j++ {
					c := s[i]
					switch {
					case c >= '0' && c <= '9':
						val = val*16 + uint64(c-'0')
					case c >= 'a' && c <= 'f':
						val = val*16 + uint64(c-'a'+10)
					case c >= 'A' && c <= 'F':
						val = val*16 + uint64(c-'A'+10)
					default:
						i-- // not a hex digit, back up
						goto done_hex
					}
					if j < 1 {
						i++
					}
				}
			done_hex:
				a.AccumulateByte(val)
			case s[i] == 'n':
				a.AccumulateByte(10)
			case s[i] == 't':
				a.AccumulateByte(9)
			case s[i] == 'r':
				a.AccumulateByte(13)
			case s[i] == 'b':
				a.AccumulateByte(8)
			case s[i] == 'f':
				a.AccumulateByte(12)
			case s[i] == '\\':
				a.AccumulateByte('\\')
			case s[i] == '"':
				a.AccumulateByte('"')
			default:
				// Unknown escape, just emit the character
				a.AccumulateByte(uint64(s[i]))
			}
		} else {
			a.AccumulateByte(uint64(s[i]))
		}
	}
}

// Pools returns the accumulated constant pools.
func (a *ConstPoolAccumulator) Pools() map[string]*ConstPool {
	return a.pools
}

// emitConstPools writes DATA/GLOBL directives for all constant pools to the builder.
func emitConstPools(builder *strings.Builder, constPools map[string]*ConstPool) {
	if len(constPools) > 0 {
		builder.WriteString("\n#include \"textflag.h\"\n")
		builder.WriteString("\n// Constant pool data\n")
		for label, pool := range constPools {
			for i, val := range pool.Data {
				fmt.Fprintf(builder, "DATA %s<>+%d(SB)/4, $0x%08x\n", label, i*4, val)
			}
			fmt.Fprintf(builder, "GLOBL %s<>(SB), (RODATA|NOPTR), $%d\n", label, len(pool.Data)*4)
		}
	}
}

// extractObjDumpFunctionName extracts the symbol name from a "<name>" pattern
// in objdump output and strips the macOS underscore prefix if applicable.
func extractObjDumpFunctionName(line string, targetOS string) string {
	name := strings.Split(line, "<")[1]
	name = strings.Split(name, ">")[0]
	if targetOS == "darwin" && strings.HasPrefix(name, "_") {
		name = name[1:]
	}
	return name
}

// parseObjDumpDataLine splits an objdump data line (after the address colon)
// into binary hex tokens and assembly text.
func parseObjDumpDataLine(line string) ([]string, string) {
	data := strings.TrimSpace(strings.Split(line, ":")[1])
	splits := strings.Split(data, " ")
	var binary []string
	var assembly string
	for i, s := range splits {
		if s == "" || unicode.IsSpace(rune(s[0])) {
			assembly = strings.TrimSpace(strings.Join(splits[i:], " "))
			break
		}
		binary = append(binary, s)
	}
	return binary, assembly
}

// openAssemblyFile opens an assembly file and returns a scanner and a cleanup function.
func openAssemblyFile(path string) (*bufio.Scanner, func(), error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	cleanup := func() {
		if err := file.Close(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	return bufio.NewScanner(file), cleanup, nil
}

// ListArchitectures returns a list of supported architectures
func ListArchitectures() []string {
	archs := make([]string, 0, len(parsers))
	for arch := range parsers {
		archs = append(archs, arch)
	}
	return archs
}
