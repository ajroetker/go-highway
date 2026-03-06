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
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"github.com/klauspost/asmfmt"
	"github.com/samber/lo"
)

// ARM64Parser implements ArchParser for ARM64/NEON architecture
type ARM64Parser struct{}

// arm64 regex patterns
var (
	// Match labels like .LBB0_2: (Linux) or LBB0_2: (macOS)
	arm64LabelLine = regexp.MustCompile(`^\.?\w+_\d+:.*$`)
	arm64CodeLine  = regexp.MustCompile(`^\s+(?:\w+|\.inst\b).*$`)
	// Match jumps to labels with or without leading dot
	arm64JmpLine = regexp.MustCompile(`^(b|b\.\w{2})\t\.?\w+_\d+$`)
	// Match CBZ/CBNZ: "cbz x10, LBB0_40" or "cbnz w8, .LBB0_5"
	arm64CbzLine = regexp.MustCompile(`^(cbz|cbnz)\t(\w+),\s*\.?(\w+_\d+)$`)
	// Match TBZ/TBNZ: "tbz w12, #0, LBB0_21" or "tbnz x5, #31, .LBB0_3"
	arm64TbzLine = regexp.MustCompile(`^(tbz|tbnz)\t(\w+),\s*#(\d+),\s*\.?(\w+_\d+)$`)
	// Match stack frame allocation: "sub sp, sp, #N" with hex or decimal, optional lsl #12
	arm64StackAllocLine = regexp.MustCompile(`^\s*sub\s+sp,\s*sp,\s*#(0x[0-9a-fA-F]+|\d+)(?:,\s*lsl\s*#(\d+))?`)
	// Match pre-decrement stack allocation: "stp ... [sp, #-N]!" or "str ... [sp, #-N]!"
	arm64StackPreDecLine = regexp.MustCompile(`\[sp,\s*#-(\d+)\]!`)
	// Match post-increment stack deallocation: "ldp ... [sp], #N" or "ldr ... [sp], #N"
	arm64StackPostIncLine = regexp.MustCompile(`\[sp\],\s*#(\d+)`)
	// Match single-register str/ldr (vs pair stp/ldp) for different encoding handling
	arm64SingleRegLine = regexp.MustCompile(`^\s*(str|ldr)\s+`)
	// Match explicit stack deallocation: "add sp, sp, #N" with hex or decimal, optional lsl #12
	arm64StackDeallocLine = regexp.MustCompile(`^\s*add\s+sp,\s*sp,\s*#(0x[0-9a-fA-F]+|\d+)(?:,\s*lsl\s*#(\d+))?`)
	// Match frame pointer or register setup from SP: "add xN, sp, #N" (NOT "add sp, sp, #N")
	arm64AddFromSpLine = regexp.MustCompile(`^\s*add\s+x(\d+),\s*sp,\s*#`)
	// Match frame pointer restore to SP: "sub sp, xN, #N" (epilogue counterpart of add xN, sp, #N)
	arm64SubSpFromRegLine = regexp.MustCompile(`^\s*sub\s+sp,\s*x(\d+),\s*#`)

	// Match rdsvl instruction: "rdsvl xN, #M" — reads streaming vector length
	arm64RdsvlLine = regexp.MustCompile(`^\s*rdsvl\s+x(\d+),\s*#(\d+)`)
	// Match msub with same source regs: "msub xA, xB, xB, xC" — computes xC - xB*xB
	arm64MsubSameRegLine = regexp.MustCompile(`^\s*msub\s+x\d+,\s*x(\d+),\s*x(\d+),\s*x\d+`)

	// Match "mov sp, xN" — dynamic SP write from register (VLA allocation or SP restore)
	arm64MovToSpLine = regexp.MustCompile(`^\s*mov\s+sp,\s*x(\d+)`)
	// Match "mov xN, sp" — save SP to register (before VLA allocation)
	arm64MovFromSpLine = regexp.MustCompile(`^\s*mov\s+x(\d+),\s*sp`)

	// Constant pool patterns
	// Match constant pool labels: lCPI0_0:, LCPI0_0:, .LCPI0_0:, .lCPI0_0:, CPI0_0:
	arm64ConstPoolLabel = regexp.MustCompile(`^\.?[lL]?CPI\d+_\d+:`)
	// Match .long directive with hex or decimal value (hex must come first in alternation)
	arm64LongDirective = regexp.MustCompile(`^\s+\.long\s+(0x[0-9a-fA-F]+|\d+)`)
	// Match .quad directive with hex or decimal value (hex must come first in alternation)
	arm64QuadDirective = regexp.MustCompile(`^\s+\.quad\s+(0x[0-9a-fA-F]+|\d+)`)
	// Match .byte directive with hex or decimal value
	arm64ByteDirective = regexp.MustCompile(`^\s+\.byte\s+(0x[0-9a-fA-F]+|\d+)`)
	// Match section directive for literal data (macOS: __TEXT,__literal* or __TEXT,__const; Linux: .rodata)
	arm64LiteralSection = regexp.MustCompile(`^\s+\.section\s+(__TEXT,__(?:literal|const)|\.rodata)`)
	// Match .ascii directive with string content
	arm64AsciiDirective = regexp.MustCompile(`^\s+\.ascii\s+"((?:[^"\\]|\\.)*)"`)
	// Match adrp instruction referencing constant pool.
	// macOS Mach-O: adrp x0, .LCPI0_0@PAGE
	// Linux ELF:    adrp x0, .LCPI0_0
	arm64AdrpConstPool = regexp.MustCompile(`adrp\s+(\w+),\s*\.?[lL]?CPI(\d+_\d+)(?:@PAGE)?`)
	// Match ldr instruction with constant pool page-offset reference.
	// macOS Mach-O: ldr v1, [x0, .LCPI0_0@PAGEOFF]
	// Linux ELF:    ldr v1, [x0, :lo12:.LCPI0_0]
	arm64LdrConstPoolPageoff = regexp.MustCompile(`ldr\s+(\w+),\s*\[(\w+),\s*(?::lo12:)?\.?[lL]?CPI(\d+_\d+)(?:@PAGEOFF)?\]`)
	// Match ldr instruction with just register (for Linux-style PC-relative)
	arm64LdrConstPoolReg = regexp.MustCompile(`ldr\s+(\w+),\s*\[(\w+)\]`)
	// Match adrp referencing static data symbol (macOS: _label@PAGE)
	arm64AdrpStaticData = regexp.MustCompile(`adrp\s+(\w+),\s*_(\w+)@PAGE`)
	// Match add with PAGEOFF static data reference (macOS: _label@PAGEOFF)
	arm64AddStaticDataPageoff = regexp.MustCompile(`add\s+(\w+),\s*(\w+),\s*_(\w+)@PAGEOFF`)
	// Match add with :lo12: static data reference (Linux)
	arm64AddStaticDataLo12 = regexp.MustCompile(`add\s+(\w+),\s*(\w+),\s*:lo12:(\w+)`)
)

// arm64 register sets
var (
	arm64Registers     = []string{"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"}
	arm64FPRegisters   = []string{"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"}
	arm64NeonRegisters = []string{"V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"}
)

// arm64Line represents a single assembly instruction for ARM64
// Binary is string because ARM64 has fixed-width 32-bit instructions
type arm64Line struct {
	Labels   []string
	Assembly string
	Binary   string
	// SpOffset is added to sp-relative offsets during transformation.
	// Used when clang emits a pre-decrement prologue (str [sp, #-N]!)
	// followed by sub sp, sp, #M: callee-save instructions between
	// the pre-decrement and sub-sp need their offsets shifted up by M
	// to account for GOAT's merged frame.
	SpOffset int
	// DynAllocPad controls transformation for dynamic stack allocation (VLAs).
	// 0: no transformation (default)
	// -1: NOP (remove instruction, e.g., mov sp, xN)
	// >0: transform "mov xN, sp" to "add xN, sp, #DynAllocPad"
	DynAllocPad int
}

// arm64ConstPoolRef tracks a constant pool reference that needs to be rewritten
// When we see adrp+ldr pairs that reference constant pools, we need to rewrite them
type arm64ConstPoolRef struct {
	AdrpIndex int    // Index of adrp instruction in function lines
	LdrIndex  int    // Index of ldr instruction in function lines
	Label     string // Constant pool label being referenced
	BaseReg   string // Register used for address calculation
	DestReg   string // Destination register for the load
}

// transformStackInstruction transforms pre/post-indexed stack operations to
// regular offset addressing. This is necessary because Go assembly allocates
// the stack frame in the TEXT directive, but C compilers generate pre/post-indexed
// addressing that also adjusts SP.
//
// Transforms:
//   - stp Rt, Rt2, [sp, #-N]! -> stp Rt, Rt2, [sp, #0]  (pre-indexed store)
//   - ldp Rt, Rt2, [sp], #N   -> ldp Rt, Rt2, [sp, #0]  (post-indexed load)
//   - sub sp, sp, #N          -> (removed, returns empty)
//   - add sp, sp, #N          -> (removed, returns empty)
//
// ARM64 LDP/STP instruction encoding (64-bit registers):
//
//	Bits 31-30: opc (10 for 64-bit)
//	Bits 29-27: 101 (op0 for LDP/STP)
//	Bit 26: V (0 for integer, 1 for SIMD)
//	Bits 25-23: op2 (includes addressing mode)
//	  - 011 = pre-indexed (writeback before)
//	  - 001 = post-indexed (writeback after)
//	  - 010 = signed offset (no writeback)
//	Bit 22: L (0 for store, 1 for load)
//	Bits 21-15: imm7 (signed, scaled by 8 for 64-bit)
//	Bits 14-10: Rt2
//	Bits 9-5: Rn (base register)
//	Bits 4-0: Rt
//
// parseArm64Immediate parses a hex or decimal immediate value with an optional
// left shift. matches[1] is the immediate (e.g. "0x5a0" or "1440"), and
// shiftStr is the shift amount (e.g. "12" from "lsl #12") or empty.
func parseArm64Immediate(immStr, shiftStr string) (int, error) {
	var val int64
	var err error
	if strings.HasPrefix(immStr, "0x") || strings.HasPrefix(immStr, "0X") {
		val, err = strconv.ParseInt(immStr[2:], 16, 64)
	} else {
		val, err = strconv.ParseInt(immStr, 10, 64)
	}
	if err != nil {
		return 0, err
	}
	if shiftStr != "" {
		shift, err := strconv.Atoi(shiftStr)
		if err != nil {
			return 0, err
		}
		val <<= shift
	}
	return int(val), nil
}

func (line *arm64Line) transformStackInstruction() (string, bool) {
	asm := line.Assembly

	// Handle dynamic allocation (VLA) transformations
	if line.DynAllocPad == -1 {
		return "", true // NOP: remove mov sp, xN
	}
	if line.DynAllocPad > 0 {
		// Transform "mov xN, sp" (= add xN, sp, #0) to "add xN, sp, #pad, lsl #12"
		// so VLA pointers computed as (xN - VLA_size) land within the enlarged frame.
		binary, err := strconv.ParseUint(line.Binary, 16, 32)
		if err != nil {
			return "", false
		}
		rd := binary & 0x1f // Extract destination register (bits 4:0)
		// Encode: ADD Xd, SP, #(pad >> 12), LSL #12
		// sf=1 op=0 S=0 10001 sh=1 imm12 Rn=31(sp) Rd
		padImm12 := uint64(line.DynAllocPad >> 12)
		newBinary := uint64(0x91400000) | (padImm12 << 10) | (31 << 5) | rd
		return fmt.Sprintf("%08x", newBinary), true
	}

	// Skip explicit stack allocation/deallocation - Go handles this
	if arm64StackAllocLine.MatchString(asm) || arm64StackDeallocLine.MatchString(asm) {
		return "", true // Remove this instruction
	}

	// Check for pre-indexed store with SP
	if arm64StackPreDecLine.MatchString(asm) {
		binary, err := strconv.ParseUint(line.Binary, 16, 32)
		if err != nil {
			return "", false
		}

		if arm64SingleRegLine.MatchString(asm) {
			// STR (single register) pre-indexed: bits 11:10 = 11, imm9 at bits 20:12
			// Transform to unscaled offset with SpOffset applied
			binary &^= (0x1ff << 12) // Zero imm9 (bits 20-12)
			binary &^= (3 << 10)     // Change mode 11 -> 00 (unscaled offset, no writeback)
			if line.SpOffset > 255 {
				// imm9 overflow: emit scratch base + STR with x16 base
				scratch := arm64EmitScratchBase(line.SpOffset, "transformed")
				binary = arm64ChangeBaseFromSpToX16(binary)
				return scratch + fmt.Sprintf("\tWORD $0x%08x\t// %s [transformed, scratch x16]\n", binary, line.Assembly), true
			}
			if line.SpOffset > 0 {
				// Encode SpOffset as imm9 (signed, unscaled bytes)
				imm9 := uint64(line.SpOffset) & 0x1ff
				binary |= imm9 << 12
			}
		} else {
			// STP (pair) pre-indexed: bits 25:23 = 011, imm7 at bits 21:15
			// Transform to signed-offset with SpOffset applied
			binary &^= (1 << 23)    // Clear bit 23 (pre-indexed -> signed offset)
			binary &^= (0x7f << 15) // Zero imm7
			if line.SpOffset > 0 {
				scale := arm64StpLdpScale(binary)
				scaledOffset := line.SpOffset / scale
				if scaledOffset > 63 {
					// imm7 overflow: emit scratch base + STP with x16 base
					scratch := arm64EmitScratchBase(line.SpOffset, "transformed")
					binary = arm64ChangeBaseFromSpToX16(binary)
					return scratch + fmt.Sprintf("\tWORD $0x%08x\t// %s [transformed, scratch x16]\n", binary, line.Assembly), true
				}
				binary |= uint64(scaledOffset) << 15
			}
		}

		return fmt.Sprintf("%08x", binary), true
	}

	// Check for post-indexed load with SP
	if arm64StackPostIncLine.MatchString(asm) {
		binary, err := strconv.ParseUint(line.Binary, 16, 32)
		if err != nil {
			return "", false
		}

		if arm64SingleRegLine.MatchString(asm) {
			// LDR (single register) post-indexed: bits 11:10 = 01, imm9 at bits 20:12
			// Transform to unscaled offset with SpOffset applied
			binary &^= (0x1ff << 12) // Zero imm9 (bits 20-12)
			binary &^= (3 << 10)     // Change mode 01 -> 00 (unscaled offset, no writeback)
			if line.SpOffset > 255 {
				// imm9 overflow: emit scratch base + LDR with x16 base
				scratch := arm64EmitScratchBase(line.SpOffset, "transformed")
				binary = arm64ChangeBaseFromSpToX16(binary)
				return scratch + fmt.Sprintf("\tWORD $0x%08x\t// %s [transformed, scratch x16]\n", binary, line.Assembly), true
			}
			if line.SpOffset > 0 {
				imm9 := uint64(line.SpOffset) & 0x1ff
				binary |= imm9 << 12
			}
		} else {
			// LDP (pair) post-indexed: bits 25:23 = 001, imm7 at bits 21:15
			// Transform to signed-offset with SpOffset applied
			binary |= (1 << 24)     // Set bit 24
			binary &^= (1 << 23)    // Clear bit 23 (post-indexed -> signed offset)
			binary &^= (0x7f << 15) // Zero imm7
			if line.SpOffset > 0 {
				scale := arm64StpLdpScale(binary)
				scaledOffset := line.SpOffset / scale
				if scaledOffset > 63 {
					// imm7 overflow: emit scratch base + LDP with x16 base
					scratch := arm64EmitScratchBase(line.SpOffset, "transformed")
					binary = arm64ChangeBaseFromSpToX16(binary)
					return scratch + fmt.Sprintf("\tWORD $0x%08x\t// %s [transformed, scratch x16]\n", binary, line.Assembly), true
				}
				binary |= uint64(scaledOffset) << 15
			}
		}

		return fmt.Sprintf("%08x", binary), true
	}

	return "", false
}

func (line *arm64Line) emitTransformed(newBinary string) string {
	return fmt.Sprintf("\tWORD $0x%s\t// %s [transformed]\n", newBinary, line.Assembly)
}

func (line *arm64Line) emitBranch() string {
	splits := strings.Split(line.Assembly, "\t")
	instruction := strings.Map(func(r rune) rune {
		if r == '.' {
			return -1
		}
		return unicode.ToUpper(r)
	}, splits[0])
	label := normalizeLabel(splits[1])
	return fmt.Sprintf("\t%s %s\n", instruction, label)
}

func (line *arm64Line) emitCbz(m []string) string {
	mnemonic := strings.ToUpper(m[1])
	reg := m[2]
	if strings.HasPrefix(reg, "w") {
		mnemonic += "W"
	}
	goReg := goRegisterName(reg)
	label := normalizeLabel(m[3])
	return fmt.Sprintf("\t%s %s, %s\n", mnemonic, goReg, label)
}

func (line *arm64Line) emitTbz(m []string) string {
	mnemonic := strings.ToUpper(m[1])
	goReg := goRegisterName(m[2])
	label := normalizeLabel(m[4])
	return fmt.Sprintf("\t%s $%s, %s, %s\n", mnemonic, m[3], goReg, label)
}

// arm64StpLdpScale returns the byte scale factor for STP/LDP based on
// the opc (bits 31:30) and V (bit 26) fields of the instruction encoding.
func arm64StpLdpScale(binary uint64) int {
	opc := (binary >> 30) & 3
	v := (binary >> 26) & 1
	if v == 0 {
		// Integer registers
		if opc == 0 {
			return 4 // W registers (32-bit)
		}
		return 8 // X registers (64-bit)
	}
	// SIMD/FP registers
	switch opc {
	case 0:
		return 4 // S registers (32-bit FP)
	case 1:
		return 8 // D registers (64-bit FP)
	case 2:
		return 16 // Q registers (128-bit)
	default:
		return 8
	}
}

// arm64EmitScratchBase emits ADD instruction(s) to compute x16 = sp + spOffset.
// x16 (IP0) is the ARM64 intra-procedure-call scratch register, safe to clobber.
func arm64EmitScratchBase(spOffset int, comment string) string {
	var sb strings.Builder
	if spOffset <= 4095 {
		// Single ADD x16, sp, #spOffset
		binary := uint64(0x91000000) | (uint64(spOffset) << 10) | (31 << 5) | 16
		sb.WriteString(fmt.Sprintf("\tWORD $0x%08x\t// add x16, sp, #%d [%s scratch base]\n", binary, spOffset, comment))
	} else {
		// ADD x16, sp, #(spOffset>>12), lsl #12
		hi := uint64(spOffset >> 12)
		binary1 := uint64(0x91400000) | (hi << 10) | (31 << 5) | 16
		sb.WriteString(fmt.Sprintf("\tWORD $0x%08x\t// add x16, sp, #%d, lsl #12 [%s scratch hi]\n", binary1, spOffset>>12, comment))
		// ADD x16, x16, #(spOffset & 0xfff)
		lo := uint64(spOffset & 0xfff)
		if lo > 0 {
			binary2 := uint64(0x91000000) | (lo << 10) | (16 << 5) | 16
			sb.WriteString(fmt.Sprintf("\tWORD $0x%08x\t// add x16, x16, #%d [%s scratch lo]\n", binary2, spOffset&0xfff, comment))
		}
	}
	return sb.String()
}

// arm64ChangeBaseFromSpToX16 changes the Rn field (bits 9:5) from SP (31)
// to x16, so the instruction uses the scratch register as its base.
func arm64ChangeBaseFromSpToX16(binary uint64) uint64 {
	binary &^= 0x1f << 5 // Clear Rn
	binary |= 16 << 5    // Set Rn = x16
	return binary
}

func (line *arm64Line) emitSpOffsetAdjusted() string {
	binary, err := strconv.ParseUint(line.Binary, 16, 32)
	if err == nil {
		if arm64AddFromSpLine.MatchString(line.Assembly) || arm64SubSpFromRegLine.MatchString(line.Assembly) {
			// ADD Xd, SP, #imm or SUB SP, Xn, #imm: imm12 at bits 21:10
			sh := (binary >> 22) & 1
			imm12 := (binary >> 10) & 0xfff
			shift := uint64(0)
			if sh == 1 {
				shift = 12
			}
			newImm12 := imm12 + uint64(line.SpOffset>>shift)
			if newImm12 > 4095 {
				scratch := arm64EmitScratchBase(line.SpOffset, "offset adjusted")
				rebased := arm64ChangeBaseFromSpToX16(binary)
				return scratch + fmt.Sprintf("\tWORD $0x%08x\t// %s [offset adjusted, scratch x16]\n", rebased, line.Assembly)
			}
			binary &^= 0xfff << 10
			binary |= (newImm12 & 0xfff) << 10
		} else if arm64SingleRegLine.MatchString(line.Assembly) {
			// STR/LDR unsigned offset: imm12 at bits 21:10
			if (binary>>24)&3 == 1 {
				imm12 := (binary >> 10) & 0xfff
				scale := 1 << ((binary >> 30) & 3)
				newImm12 := imm12 + uint64(line.SpOffset)/uint64(scale)
				if newImm12 > 4095 {
					scratch := arm64EmitScratchBase(line.SpOffset, "offset adjusted")
					rebased := arm64ChangeBaseFromSpToX16(binary)
					return scratch + fmt.Sprintf("\tWORD $0x%08x\t// %s [offset adjusted, scratch x16]\n", rebased, line.Assembly)
				}
				binary &^= 0xfff << 10
				binary |= (newImm12 & 0xfff) << 10
			}
		} else {
			// STP/LDP signed-offset: imm7 at bits 21:15
			scale := arm64StpLdpScale(binary)
			imm7 := (binary >> 15) & 0x7f
			newImm7 := imm7 + uint64(line.SpOffset/scale)
			if newImm7 > 63 {
				scratch := arm64EmitScratchBase(line.SpOffset, "offset adjusted")
				rebased := arm64ChangeBaseFromSpToX16(binary)
				return scratch + fmt.Sprintf("\tWORD $0x%08x\t// %s [offset adjusted, scratch x16]\n", rebased, line.Assembly)
			}
			binary &^= 0x7f << 15
			binary |= (newImm7 & 0x7f) << 15
		}
	}
	return fmt.Sprintf("\tWORD $0x%08x\t// %s [offset adjusted]\n", binary, line.Assembly)
}

func (line *arm64Line) emitDefault() string {
	return fmt.Sprintf("\tWORD $0x%v\t// %s\n", line.Binary, line.Assembly)
}

func (line *arm64Line) String() string {
	if line.Binary == "" && line.Assembly == "" {
		return ""
	}
	if newBinary, transformed := line.transformStackInstruction(); transformed {
		if newBinary == "" {
			return ""
		}
		// Multi-line output from overflow handling starts with \t
		if strings.HasPrefix(newBinary, "\t") {
			return newBinary
		}
		return line.emitTransformed(newBinary)
	}
	if arm64JmpLine.MatchString(line.Assembly) {
		return line.emitBranch()
	}
	if m := arm64CbzLine.FindStringSubmatch(line.Assembly); m != nil {
		return line.emitCbz(m)
	}
	if m := arm64TbzLine.FindStringSubmatch(line.Assembly); m != nil {
		return line.emitTbz(m)
	}
	if line.SpOffset > 0 {
		return line.emitSpOffsetAdjusted()
	}
	return line.emitDefault()
}

// Name returns the architecture name
func (p *ARM64Parser) Name() string {
	return "arm64"
}

// BuildTags returns the Go build constraint
func (p *ARM64Parser) BuildTags() string {
	return "//go:build !noasm && arm64\n"
}

// BuildTarget returns the clang target triple
func (p *ARM64Parser) BuildTarget(goos string) string {
	if goos == "darwin" {
		return "arm64-apple-darwin"
	}
	return "arm64-linux-gnu"
}

// CompilerFlags returns architecture-specific compiler flags
func (p *ARM64Parser) CompilerFlags() []string {
	return []string{
		"-ffixed-x18", // ARM64 platform register (reserved on some OSes)
		"-ffixed-x26", // Go REGCTXT: closure context register
		"-ffixed-x27", // Go REGTMP: reserved for linker temporaries
		"-ffixed-x28", // Go REGG: goroutine pointer (g)
	}
}

// Prologue returns C parser prologue for ARM64 NEON types
func (p *ARM64Parser) Prologue() string {
	var prologue strings.Builder
	// Define GOAT_PARSER to skip includes during parsing
	prologue.WriteString("#define GOAT_PARSER 1\n")
	// Define include guards so real system headers are skipped during parsing.
	// The modernc.org C parser can't handle GCC/Clang builtins in these headers.
	// All NEON/SVE types are provided as typedefs below instead.
	prologue.WriteString("#define _AARCH64_NEON_H_\n") // GCC arm_neon.h
	prologue.WriteString("#define __ARM_NEON_H 1\n")   // Clang arm_neon.h
	prologue.WriteString("#define _ARM_NEON_H_ 1\n")   // alternative arm_neon.h
	prologue.WriteString("#define _AARCH64_SVE_H_\n")  // GCC arm_sve.h
	prologue.WriteString("#define __ARM_SVE_H 1\n")    // Clang arm_sve.h
	prologue.WriteString("#define _ARM_FP16_H_ 1\n")   // arm_fp16.h
	prologue.WriteString("#define _ARM_BF16_H_ 1\n")   // arm_bf16.h

	// Define __bf16 for arm_bf16.h (compiler built-in type)
	prologue.WriteString("typedef short __bf16;\n")
	// Define __fp16 for arm_fp16.h
	prologue.WriteString("typedef short __fp16;\n")
	// Define standard C type aliases (used in SVE intrinsics)
	prologue.WriteString("typedef __fp16 float16_t;\n")
	prologue.WriteString("typedef __bf16 bfloat16_t;\n")

	// Define NEON 64-bit vector types
	prologue.WriteString("typedef struct { char _[8]; } int8x8_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } int16x4_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } int32x2_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } int64x1_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } uint8x8_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } uint16x4_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } uint32x2_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } uint64x1_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } float32x2_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } float64x1_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } float16x4_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } bfloat16x4_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } poly8x8_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } poly16x4_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } poly64x1_t;\n")

	// Define NEON 128-bit vector types
	prologue.WriteString("typedef struct { char _[16]; } int8x16_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } int16x8_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } int32x4_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } int64x2_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } uint8x16_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } uint16x8_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } uint32x4_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } uint64x2_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } float32x4_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } float64x2_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } float16x8_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } bfloat16x8_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } poly8x16_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } poly16x8_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } poly64x2_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } poly128_t;\n")

	// Define NEON 128-bit array types (x2, x3, x4)
	for _, base := range []string{"int8x16", "int16x8", "int32x4", "int64x2",
		"uint8x16", "uint16x8", "uint32x4", "uint64x2",
		"float32x4", "float64x2", "float16x8", "bfloat16x8",
		"poly8x16", "poly16x8", "poly64x2"} {
		prologue.WriteString(fmt.Sprintf("typedef struct { char _[32]; } %sx2_t;\n", base))
		prologue.WriteString(fmt.Sprintf("typedef struct { char _[48]; } %sx3_t;\n", base))
		prologue.WriteString(fmt.Sprintf("typedef struct { char _[64]; } %sx4_t;\n", base))
	}

	// Define NEON 64-bit array types (x2, x3, x4)
	for _, base := range []string{"int8x8", "int16x4", "int32x2", "int64x1",
		"uint8x8", "uint16x4", "uint32x2", "uint64x1",
		"float32x2", "float64x1", "float16x4", "bfloat16x4",
		"poly8x8", "poly16x4", "poly64x1"} {
		prologue.WriteString(fmt.Sprintf("typedef struct { char _[16]; } %sx2_t;\n", base))
		prologue.WriteString(fmt.Sprintf("typedef struct { char _[24]; } %sx3_t;\n", base))
		prologue.WriteString(fmt.Sprintf("typedef struct { char _[32]; } %sx4_t;\n", base))
	}

	// Add SVE/SME types and intrinsic stubs from arm64_sve.go
	prologue.WriteString(SVEPrologue())

	return prologue.String()
}

// TranslateAssembly implements the full translation pipeline for ARM64
func (p *ARM64Parser) TranslateAssembly(t *TranslateUnit, functions []Function) error {
	// Parse assembly
	assembly, stackSizes, preDecSizes, dynSVLAlloc, dynRegAlloc, constPools, err := p.parseAssembly(t.Assembly, t.TargetOS)
	if err != nil {
		return err
	}

	// Run objdump
	dump, err := runCommand("objdump", "-d", t.Object)
	if err != nil {
		return err
	}

	// Parse object dump
	if err := p.parseObjectDump(dump, assembly, t.TargetOS); err != nil {
		return err
	}

	// Auto-transform SVE/SME functions (inject smstart/smstop, fix forbidden instructions)
	for fnName, lines := range assembly {
		assembly[fnName] = TransformSVEFunction(lines)
	}

	// Copy stack sizes to functions, setting SpillBase to the original C frame size
	// before adding dynamic allocation padding. SpillBase is where overflow args
	// are placed — it must match the offset the C body expects to read them at.
	for i, fn := range functions {
		if sz, ok := stackSizes[fn.Name]; ok {
			functions[i].SpillBase = sz // Original C frame size for overflow arg placement
			// Add dynamic SVL-based stack allocation if detected.
			// rdsvl+msub allocates SVL^2 bytes at runtime. We must declare
			// enough frame space for the worst-case SVL. ARM SME allows
			// SVL up to 256 bytes (2048 bits); Apple M4 uses 64 bytes.
			// Use 256 bytes as worst case: 256^2 = 65536.
			if dynSVLAlloc[fn.Name] {
				const maxSVLBytes = 256 // ARM architectural maximum
				sz += maxSVLBytes * maxSVLBytes
			}
			// Add padding for register-based dynamic stack allocation (VLAs).
			// Functions with "mov sp, xN" allocate runtime-sized arrays on the stack.
			// We enlarge the frame to accommodate worst-case VLA size.
			if dynRegAlloc[fn.Name] {
				const maxDynAllocBytes = 131072 // 128KB for VLAs
				sz += maxDynAllocBytes
			}
			functions[i].StackSize = sz
		}
	}
	_ = preDecSizes // preDecSizes already consumed during callee-save adjustment in parseAssembly

	// Generate Go assembly with constant pools
	return p.generateGoAssembly(t, functions, assembly, constPools)
}

func (p *ARM64Parser) parseAssembly(path string, targetOS string) (map[string][]*arm64Line, map[string]int, map[string]int, map[string]bool, map[string]bool, map[string]*ConstPool, error) {
	scanner, cleanup, err := openAssemblyFile(path)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	defer cleanup()

	var (
		stackSizes     = make(map[string]int)
		preDecSizes    = make(map[string]int)         // pre-decrement prologue sizes per function
		hasDynSVLAlloc = make(map[string]bool)        // functions with rdsvl+msub dynamic stack alloc
		hasDynRegAlloc = make(map[string]bool)        // functions with register-based dynamic SP adjustment (VLAs)
		rdsvlRegs      = make(map[string]map[int]int) // funcName -> {regNum: multiplier} from rdsvl
		functions      = make(map[string][]*arm64Line)
		cpa            = NewConstPoolAccumulator()
		functionName   string
		labelName      string
		// Constant pool parsing state
		inLiteralSection bool
	)
	for scanner.Scan() {
		line := scanner.Text()

		// Check for literal data section
		if arm64LiteralSection.MatchString(line) {
			inLiteralSection = true
			continue
		}

		// Check for constant pool label (lCPI0_0: or .LCPI0_0:)
		if arm64ConstPoolLabel.MatchString(line) {
			cpa.StartPool(normalizeLabel(strings.Split(line, ":")[0]))
			continue
		}

		// Recognize data labels in literal/const sections (non-CPI labels like _streamVByte32DataLen:)
		if inLiteralSection {
			// Strip comments ("; ...") before checking for label colon
			labelLine := line
			if idx := strings.Index(labelLine, ";"); idx >= 0 {
				labelLine = labelLine[:idx]
			}
			labelLine = strings.TrimSpace(labelLine)
			if strings.HasSuffix(labelLine, ":") && !strings.HasPrefix(labelLine, ".") {
				label := strings.TrimSuffix(labelLine, ":")
				// Strip macOS underscore prefix
				label = strings.TrimPrefix(label, "_")
				// Skip CPI labels (already handled above)
				if label != "" && !arm64ConstPoolLabel.MatchString(labelLine) {
					cpa.StartPool(label)
					continue
				}
			}
		}

		// Parse .long/.quad/.byte/.ascii directives for constant pool data
		if inLiteralSection || cpa.Active() {
			if matches := arm64LongDirective.FindStringSubmatch(line); matches != nil {
				cpa.AddLong(parseIntValue(matches[1]))
				continue
			}
			if matches := arm64QuadDirective.FindStringSubmatch(line); matches != nil {
				cpa.AddQuad(parseIntValue(matches[1]))
				continue
			}
			if matches := arm64ByteDirective.FindStringSubmatch(line); matches != nil {
				cpa.AccumulateByte(parseIntValue(matches[1]))
				continue
			}
			if matches := arm64AsciiDirective.FindStringSubmatch(line); matches != nil {
				cpa.AccumulateAscii(matches[1])
				continue
			}
		}

		// Check for section change that exits literal section
		if strings.HasPrefix(strings.TrimSpace(line), ".section") && !arm64LiteralSection.MatchString(line) {
			inLiteralSection = false
			cpa.FinishPool()
		}

		if attributeLine.MatchString(line) && !strings.HasPrefix(strings.TrimSpace(line), ".inst") {
			continue
		} else if arm64LabelLine.MatchString(line) {
			// Check labels BEFORE function names because labels like "LBB0_2: ; comment"
			// can match the function name pattern due to content after the colon
			labelName = normalizeLabel(strings.Split(line, ":")[0])
			lines := functions[functionName]
			if len(lines) == 0 || lines[len(lines)-1].Assembly != "" {
				functions[functionName] = append(functions[functionName], &arm64Line{Labels: []string{labelName}})
			} else {
				lines[len(lines)-1].Labels = append(lines[len(lines)-1].Labels, labelName)
			}
		} else if nameLine.MatchString(line) {
			functionName = strings.Split(line, ":")[0]
			// On macOS, function names are prefixed with underscore - strip it
			if targetOS == "darwin" && strings.HasPrefix(functionName, "_") {
				functionName = functionName[1:]
			}
			functions[functionName] = make([]*arm64Line, 0)
			labelName = ""
		} else if arm64CodeLine.MatchString(line) {
			asm := strings.Split(line, "//")[0]
			asm = strings.TrimSpace(asm)

			// Detect stack frame allocation patterns:
			// 1. "sub sp, sp, #N" - explicit stack allocation (cumulative)
			// 2. "stp ... [sp, #-N]!" - pre-decrement stack allocation
			// Sum all sub sp allocations since clang may split the frame
			// into multiple sub instructions.
			// Track pre-decrement sizes separately for callee-save offset adjustment.
			if matches := arm64StackAllocLine.FindStringSubmatch(asm); matches != nil {
				if size, err := parseArm64Immediate(matches[1], matches[2]); err == nil {
					stackSizes[functionName] += size
				}
			} else if matches := arm64StackPreDecLine.FindStringSubmatch(asm); matches != nil {
				if size, err := strconv.Atoi(matches[1]); err == nil {
					preDecSizes[functionName] = size
					if current, ok := stackSizes[functionName]; !ok || size > current {
						stackSizes[functionName] = size
					}
				}
			}

			// Detect dynamic SVL-based stack allocation:
			// rdsvl xN, #M sets xN = SVL * M
			// msub xA, xN, xN, xB computes xB - xN*xN = sp - (SVL*M)^2
			// This pattern appears in SME functions that use VLAs.
			if matches := arm64RdsvlLine.FindStringSubmatch(asm); matches != nil {
				regNum, _ := strconv.Atoi(matches[1])
				mult, _ := strconv.Atoi(matches[2])
				if rdsvlRegs[functionName] == nil {
					rdsvlRegs[functionName] = make(map[int]int)
				}
				rdsvlRegs[functionName][regNum] = mult
			} else if matches := arm64MsubSameRegLine.FindStringSubmatch(asm); matches != nil {
				reg1, _ := strconv.Atoi(matches[1])
				reg2, _ := strconv.Atoi(matches[2])
				if reg1 == reg2 {
					if regs, ok := rdsvlRegs[functionName]; ok {
						if _, hasReg := regs[reg1]; hasReg {
							hasDynSVLAlloc[functionName] = true
						}
					}
				}
			}

			// Detect dynamic register-based stack allocation (VLAs):
			// "mov sp, xN" indicates runtime SP adjustment from a register.
			// This is distinct from "sub sp, sp, #imm" (immediate, already handled)
			// and rdsvl+msub (SVL VLAs, handled above).
			if arm64MovToSpLine.MatchString(asm) {
				hasDynRegAlloc[functionName] = true
			}

			if labelName == "" {
				functions[functionName] = append(functions[functionName], &arm64Line{Assembly: asm})
			} else {
				lines := functions[functionName]
				if len(lines) > 0 {
					lines[len(lines)-1].Assembly = asm
				}
				labelName = ""
			}
		}
	}

	// Save any remaining constant pool
	cpa.FinishPool()
	constPools := cpa.Pools()

	if err = scanner.Err(); err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	// Adjust callee-save offsets for functions with pre-decrement prologues.
	// When clang emits: str x25, [sp, #-N]! ; stp ..., [sp, #16] ; sub sp, sp, #M
	// The stp offsets are relative to sp1 = sp0-N, but GOAT's merged frame puts
	// sp_go = sp0-(N+M). Callee saves need offsets shifted up by M.
	// This applies to both prologue saves and epilogue restores.
	spRelativeLine := regexp.MustCompile(`\[sp,?\s*#?\d*\]`)
	for fn, preDecSize := range preDecSizes {
		subSpSize := stackSizes[fn] - preDecSize
		if subSpSize <= 0 {
			continue // No sub sp, nothing to adjust
		}

		lines := functions[fn]

		// Forward pass: mark prologue callee saves (pre-dec through first sub sp).
		// Also adjusts "add xN, sp, #imm" instructions (frame pointer setup) which
		// use sp as a register operand rather than in a memory address.
		foundPreDec := false
		for _, line := range lines {
			if arm64StackPreDecLine.MatchString(line.Assembly) {
				foundPreDec = true
				line.SpOffset = subSpSize
				continue
			}
			if foundPreDec {
				if arm64StackAllocLine.MatchString(line.Assembly) {
					break // Reached the sub sp, stop
				}
				if spRelativeLine.MatchString(line.Assembly) || arm64AddFromSpLine.MatchString(line.Assembly) {
					line.SpOffset = subSpSize
				}
			}
		}

		// Second pass: mark ALL epilogue callee restores (post-inc and preceding ldp).
		// There may be multiple exit paths (early returns), each with its own epilogue.
		for i, line := range lines {
			if arm64StackPostIncLine.MatchString(line.Assembly) {
				line.SpOffset = subSpSize
				// Mark preceding ldp instructions (epilogue restores)
				for j := i - 1; j >= 0; j-- {
					prev := lines[j]
					if arm64StackDeallocLine.MatchString(prev.Assembly) {
						continue // Skip add sp (will be removed)
					}
					if arm64SubSpFromRegLine.MatchString(prev.Assembly) {
						// "sub sp, xN, #imm" (frame pointer restore) needs offset adjustment
						prev.SpOffset = subSpSize
						continue
					}
					if !spRelativeLine.MatchString(prev.Assembly) {
						break // Not an sp-relative instruction, stop
					}
					prev.SpOffset = subSpSize
				}
			}
		}
	}

	// Adjust sp-relative offsets for functions with multiple sub-sp prologues.
	// When clang splits the frame into multiple sub sp instructions:
	//   sub sp, sp, #A    ; first allocation (callee-save region)
	//   <callee saves>    ; offsets relative to sp = orig - A
	//   sub sp, sp, #B    ; additional allocation(s)
	//   <body>            ; offsets relative to sp = orig - A - B
	//   add sp, sp, #B    ; epilogue reversal
	//   <callee restores> ; offsets relative to sp = orig - A
	//   add sp, sp, #A    ; final deallocation
	// GOAT merges all sub/add sp into a single TEXT $A+B, making sp = orig-(A+B)
	// throughout. Instructions in callee-save regions (where running delta != total)
	// need their sp-relative offsets shifted by (total - running_delta).
	for fn, totalSize := range stackSizes {
		if _, hasPreDec := preDecSizes[fn]; hasPreDec {
			continue // Already handled by pre-decrement logic above
		}
		lines := functions[fn]
		if len(lines) == 0 {
			continue
		}

		// Count sub sp instructions to detect multi-sub prologues
		subSpCount := 0
		for _, line := range lines {
			if arm64StackAllocLine.MatchString(line.Assembly) {
				subSpCount++
			}
		}
		if subSpCount < 2 {
			continue
		}

		// Track running sp delta and adjust sp-relative instructions
		// that are not at the final sp level.
		runningDelta := 0
		for _, line := range lines {
			if matches := arm64StackAllocLine.FindStringSubmatch(line.Assembly); matches != nil {
				if size, err := parseArm64Immediate(matches[1], matches[2]); err == nil {
					runningDelta += size
				}
			} else if matches := arm64StackDeallocLine.FindStringSubmatch(line.Assembly); matches != nil {
				if size, err := parseArm64Immediate(matches[1], matches[2]); err == nil {
					runningDelta -= size
				}
			} else if runningDelta > 0 && runningDelta < totalSize {
				if spRelativeLine.MatchString(line.Assembly) {
					line.SpOffset = totalSize - runningDelta
				}
			}
		}
	}

	// Mark instructions for dynamic register-based allocation (VLA) transformation.
	// When a function has "mov sp, xN" (VLA alloc), we:
	// 1. NOP "mov sp, xN" — Go manages SP, don't allow dynamic adjustment
	// 2. Transform "mov xN, sp" to "add xN, sp, #pad" — VLA pointers land within enlarged frame
	const maxDynAllocBytes = 131072 // 128KB, accommodates VLAs up to ~128KB
	for fn, hasDyn := range hasDynRegAlloc {
		if !hasDyn {
			continue
		}
		for _, line := range functions[fn] {
			if arm64MovToSpLine.MatchString(line.Assembly) {
				line.DynAllocPad = -1 // NOP
			} else if arm64MovFromSpLine.MatchString(line.Assembly) {
				line.DynAllocPad = maxDynAllocBytes // Transform
			}
		}
	}

	return functions, stackSizes, preDecSizes, hasDynSVLAlloc, hasDynRegAlloc, constPools, nil
}

// goRegisterName converts ARM64 register names to Go assembly register names
// x0-x30 -> R0-R30, w0-w30 -> R0-R30, q0-q31 -> V0-V31, d0-d31 -> V0-V31, s0-s31 -> F0-F31
func goRegisterName(armReg string) string {
	armReg = strings.ToLower(armReg)
	if strings.HasPrefix(armReg, "x") || strings.HasPrefix(armReg, "w") {
		// General purpose register
		return "R" + armReg[1:]
	} else if strings.HasPrefix(armReg, "q") || strings.HasPrefix(armReg, "d") {
		// NEON vector register
		return "V" + armReg[1:]
	} else if strings.HasPrefix(armReg, "s") {
		// Floating point register (single precision)
		return "F" + armReg[1:]
	}
	// Return as-is if not recognized
	return strings.ToUpper(armReg)
}

func (p *ARM64Parser) parseObjectDump(dump string, functions map[string][]*arm64Line, targetOS string) error {
	var (
		functionName string
		lineNumber   int
	)
	for i, line := range strings.Split(dump, "\n") {
		line = strings.TrimSpace(line)
		if symbolLine.MatchString(line) {
			functionName = extractObjDumpFunctionName(line, targetOS)
			lineNumber = 0
		} else if dataLine.MatchString(line) {
			binaryTokens, _ := parseObjDumpDataLine(line)
			if len(binaryTokens) == 0 {
				continue
			}
			if lineNumber >= len(functions[functionName]) {
				return fmt.Errorf("%d: unexpected objectdump line: %s", i, line)
			}
			// ARM64 uses fixed-width 32-bit encoding — last token is the full binary
			functions[functionName][lineNumber].Binary = binaryTokens[len(binaryTokens)-1]
			lineNumber++
		}
	}
	return nil
}

func (p *ARM64Parser) generateGoAssembly(t *TranslateUnit, functions []Function, assembly map[string][]*arm64Line, constPools map[string]*ConstPool) error {
	var builder strings.Builder
	builder.WriteString(p.BuildTags())
	t.writeHeader(&builder)

	emitConstPools(&builder, constPools)

	for _, function := range functions {
		// Calculate return size based on type
		returnSize := 0
		if function.Type != "void" {
			if sz := NeonTypeSize(function.Type); sz > 0 {
				returnSize = sz // Use actual NEON type size
			} else if sz, ok := supportedTypes[function.Type]; ok {
				returnSize = sz // Use actual scalar type size
			} else {
				returnSize = 8 // Default 8-byte slot for pointers/unknown types
			}
		}

		registerCount, fpRegisterCount, neonRegisterCount, offset := 0, 0, 0, 0
		var stack []lo.Tuple2[int, Parameter]
		var argsBuilder strings.Builder

		for _, param := range function.Parameters {
			// Calculate slot size based on type
			sz := 8 // Default 8-byte slot
			if !param.Pointer {
				if neonSz := NeonTypeSize(param.Type); neonSz > 0 {
					sz = neonSz // Use actual NEON type size
				} else if typeSz, ok := supportedTypes[param.Type]; ok {
					sz = typeSz // Use actual scalar type size (4 for int32_t/float, 8 for int64_t/double/long, 1 for _Bool)
				}
			}
			// Go's ABI uses 8-byte alignment for stack parameters on 64-bit systems.
			// Parameters are placed at 8-byte aligned offsets.
			// The natural alignment of SIMD types is a hardware concern handled by registers,
			// not by padding the stack frame.
			alignTo := min(sz,
				// Smaller types can use their natural alignment
				8)
			// Align offset to parameter boundary
			if offset%alignTo != 0 {
				offset += alignTo - offset%alignTo
			}
			// Frame size uses actual type size (go vet validates this)

			if !param.Pointer && IsNeonType(param.Type) {
				// NEON vector type - load into V register(s)
				vecCount := NeonVectorCount(param.Type)
				is64bit := IsNeon64Type(param.Type)

				if neonRegisterCount+vecCount <= len(arm64NeonRegisters) {
					for v := range vecCount {
						vecOffset := offset + v*16
						if is64bit {
							vecOffset = offset + v*8
						}

						if is64bit {
							// 64-bit vector: single MOVD, load into D[0] only
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s+%d(FP), R9\n", param.Name, vecOffset))
							argsBuilder.WriteString(fmt.Sprintf("\tVMOV R9, %s.D[0]\n", arm64NeonRegisters[neonRegisterCount+v]))
						} else {
							// 128-bit vector: two MOVDs, load into D[0] and D[1]
							// Use _N suffixes (N=offset within param) so go vet accepts different offsets
							localOffset := v * 16 // offset within this parameter's storage
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s_%d+%d(FP), R9\n", param.Name, localOffset, vecOffset))
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s_%d+%d(FP), R10\n", param.Name, localOffset+8, vecOffset+8))
							argsBuilder.WriteString(fmt.Sprintf("\tVMOV R9, %s.D[0]\n", arm64NeonRegisters[neonRegisterCount+v]))
							argsBuilder.WriteString(fmt.Sprintf("\tVMOV R10, %s.D[1]\n", arm64NeonRegisters[neonRegisterCount+v]))
						}
					}
					neonRegisterCount += vecCount
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			} else if !param.Pointer && (param.Type == "float" || param.Type == "double" || param.Type == "float16_t") {
				if fpRegisterCount < len(arm64FPRegisters) {
					if param.Type == "float16_t" {
						// Load 16-bit value to GP register, then move to FP register
						argsBuilder.WriteString(fmt.Sprintf("\tMOVHU %s+%d(FP), R9\n", param.Name, offset))
						argsBuilder.WriteString(fmt.Sprintf("\tFMOVS R9, %s\n", arm64FPRegisters[fpRegisterCount]))
					} else if param.Type == "float" {
						argsBuilder.WriteString(fmt.Sprintf("\tFMOVS %s+%d(FP), %s\n", param.Name, offset, arm64FPRegisters[fpRegisterCount]))
					} else {
						argsBuilder.WriteString(fmt.Sprintf("\tFMOVD %s+%d(FP), %s\n", param.Name, offset, arm64FPRegisters[fpRegisterCount]))
					}
					fpRegisterCount++
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			} else {
				if registerCount < len(arm64Registers) {
					// Use appropriate load instruction based on type size
					loadInstr := "MOVD" // Default 8-byte load
					if !param.Pointer {
						switch sz {
						case 4:
							loadInstr = "MOVWU" // 4-byte unsigned load (zero-extends to 64-bit)
						case 2:
							loadInstr = "MOVHU" // 2-byte unsigned load
						case 1:
							loadInstr = "MOVBU" // 1-byte unsigned load
						}
					}
					argsBuilder.WriteString(fmt.Sprintf("\t%s %s+%d(FP), %s\n", loadInstr, param.Name, offset, arm64Registers[registerCount]))
					registerCount++
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			}
			offset += sz
		}

		// Check if 128-bit+ NEON types are used (for stack frame alignment)
		has128BitNeon := false
		for _, param := range function.Parameters {
			if !param.Pointer && IsNeonType(param.Type) && !IsNeon64Type(param.Type) {
				has128BitNeon = true
				break
			}
		}
		if !has128BitNeon && IsNeonType(function.Type) && !IsNeon64Type(function.Type) {
			has128BitNeon = true
		}
		// Note: Don't align offset to 16 bytes here - Go's ABI only requires 8-byte
		// alignment for return values. The 16-byte alignment is only needed for the
		// stack frame (stackOffset), not for the FP-relative parameter offsets.

		// Overflow args must be placed above the C function's local variables.
		// The C compiler emits code that reads overflow args at [sp + localFrameSize],
		// so we store them at RSP + SpillBase (the original C frame size, before
		// dynamic allocation padding like VLAs). When there's no dynamic padding,
		// SpillBase equals StackSize.
		spillBase := function.SpillBase
		if spillBase == 0 {
			spillBase = function.StackSize
		}
		stackOffset := 0
		if len(stack) > 0 {
			for i := 0; i < len(stack); i++ {
				if neonSz := NeonTypeSize(stack[i].B.Type); neonSz > 0 {
					// NEON vector: copy all bytes to stack
					is64bit := IsNeon64Type(stack[i].B.Type)
					vecCount := NeonVectorCount(stack[i].B.Type)
					for v := range vecCount {
						srcOffset := stack[i].A + v*16
						if is64bit {
							srcOffset = stack[i].A + v*8
						}
						if is64bit {
							// 64-bit vector: single 8-byte copy
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s+%d(FP), R8\n", stack[i].B.Name, srcOffset))
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD R8, %d(RSP)\n", spillBase+stackOffset))
							stackOffset += 8
						} else {
							// 128-bit vector: two 8-byte copies
							// Use _N suffixes (N=offset within param) so go vet accepts different offsets
							localOffset := v * 16 // offset within this parameter's storage
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s_%d+%d(FP), R8\n", stack[i].B.Name, localOffset, srcOffset))
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD R8, %d(RSP)\n", spillBase+stackOffset))
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s_%d+%d(FP), R8\n", stack[i].B.Name, localOffset+8, srcOffset+8))
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD R8, %d(RSP)\n", spillBase+stackOffset+8))
							stackOffset += 16
						}
					}
				} else {
					argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s+%d(FP), R8\n", stack[i].B.Name, stack[i].A))
					argsBuilder.WriteString(fmt.Sprintf("\tMOVD R8, %d(RSP)\n", spillBase+stackOffset))
					if stack[i].B.Pointer {
						stackOffset += 8
					} else {
						stackOffset += supportedTypes[stack[i].B.Type]
					}
				}
			}
		}

		// Align to 16 bytes only if 128-bit+ NEON types are used
		if has128BitNeon && stackOffset%16 != 0 {
			stackOffset += 16 - stackOffset%16
		}

		// Return value must be 8-byte aligned in Go's ABI (only if there is a return value)
		if returnSize > 0 && offset%8 != 0 {
			offset += 8 - offset%8
		}

		// The frame must hold both the C function's local variables and the overflow args.
		frameSize := function.StackSize + stackOffset
		// Ensure 16-byte alignment for the frame (required by Go assembler on ARM64)
		if frameSize > 0 && frameSize%16 != 0 {
			frameSize += 16 - frameSize%16
		}

		builder.WriteString(fmt.Sprintf("\nTEXT ·%v(SB), $%d-%d\n",
			function.Name, frameSize, offset+returnSize))
		builder.WriteString(argsBuilder.String())

		// First pass: find adrp instructions that set up constant pool addresses
		// and track which register they use
		constPoolRegs := make(map[string]string) // baseReg -> constLabel

		for _, line := range assembly[function.Name] {
			if matches := arm64AdrpConstPool.FindStringSubmatch(line.Assembly); matches != nil {
				baseReg := strings.ToLower(matches[1])
				constLabel := "CPI" + matches[2]
				if _, hasPool := constPools[constLabel]; hasPool {
					constPoolRegs[baseReg] = constLabel
				}
			}
			// Also track adrp for static data symbols (macOS: _label@PAGE)
			if matches := arm64AdrpStaticData.FindStringSubmatch(line.Assembly); matches != nil {
				baseReg := strings.ToLower(matches[1])
				constLabel := matches[2]
				if _, hasPool := constPools[constLabel]; hasPool {
					constPoolRegs[baseReg] = constLabel
				}
			}
		}

		for _, line := range assembly[function.Name] {
			// Skip adrp instructions that reference constant pools (they're no longer needed)
			if matches := arm64AdrpConstPool.FindStringSubmatch(line.Assembly); matches != nil {
				constLabel := "CPI" + matches[2]
				if _, hasPool := constPools[constLabel]; hasPool {
					// Emit any labels that were on this line
					for _, label := range line.Labels {
						builder.WriteString(label)
						builder.WriteString(":\n")
					}
					continue
				}
			}

			// Skip adrp instructions that reference static data symbols
			if matches := arm64AdrpStaticData.FindStringSubmatch(line.Assembly); matches != nil {
				constLabel := matches[2]
				if _, hasPool := constPools[constLabel]; hasPool {
					for _, label := range line.Labels {
						builder.WriteString(label)
						builder.WriteString(":\n")
					}
					continue
				}
			}

			// Replace add PAGEOFF instructions for static data with MOVD of Go symbol address
			if matches := arm64AddStaticDataPageoff.FindStringSubmatch(line.Assembly); matches != nil {
				destReg := strings.ToLower(matches[1])
				constLabel := matches[3]
				if _, hasPool := constPools[constLabel]; hasPool {
					for _, label := range line.Labels {
						builder.WriteString(label)
						builder.WriteString(":\n")
					}
					fmt.Fprintf(&builder, "\tMOVD $%s<>(SB), %s\n",
						constLabel, goRegisterName(destReg))
					continue
				}
			}

			// Replace add :lo12: instructions for static data (Linux)
			if matches := arm64AddStaticDataLo12.FindStringSubmatch(line.Assembly); matches != nil {
				destReg := strings.ToLower(matches[1])
				constLabel := matches[3]
				if _, hasPool := constPools[constLabel]; hasPool {
					for _, label := range line.Labels {
						builder.WriteString(label)
						builder.WriteString(":\n")
					}
					fmt.Fprintf(&builder, "\tMOVD $%s<>(SB), %s\n",
						constLabel, goRegisterName(destReg))
					continue
				}
			}

			// Replace ldr instructions that load from constant pools
			if matches := arm64LdrConstPoolPageoff.FindStringSubmatch(line.Assembly); matches != nil {
				destReg := matches[1]
				baseReg := strings.ToLower(matches[2])
				constLabel := "CPI" + matches[3]
				if _, hasPool := constPools[constLabel]; hasPool {
					// Emit any labels
					for _, label := range line.Labels {
						builder.WriteString(label)
						builder.WriteString(":\n")
					}
					// Emit load address of constant pool
					builder.WriteString(fmt.Sprintf("\tMOVD $%s<>(SB), %s\n",
						constLabel, goRegisterName(baseReg)))
					// Emit vector/scalar load from the address
					if strings.HasPrefix(destReg, "q") {
						// 128-bit vector load
						vReg := "V" + destReg[1:]
						builder.WriteString(fmt.Sprintf("\tVLD1 (%s), [%s.B16]\n",
							goRegisterName(baseReg), vReg))
					} else if strings.HasPrefix(destReg, "d") {
						// 64-bit vector load
						vReg := "V" + destReg[1:]
						builder.WriteString(fmt.Sprintf("\tVLD1 (%s), [%s.B8]\n",
							goRegisterName(baseReg), vReg))
					} else {
						// Scalar load
						builder.WriteString(fmt.Sprintf("\tMOVD (%s), %s\n",
							goRegisterName(baseReg), goRegisterName(destReg)))
					}
					continue
				}
			}

			for _, label := range line.Labels {
				builder.WriteString(label)
				builder.WriteString(":\n")
			}
			if line.Assembly == "ret" {
				if function.Type != "void" {
					switch function.Type {
					case "int64_t", "uint64_t", "long", "_Bool":
						builder.WriteString(fmt.Sprintf("\tMOVD R0, result+%d(FP)\n", offset))
					case "int32_t", "uint32_t":
						builder.WriteString(fmt.Sprintf("\tMOVW R0, result+%d(FP)\n", offset))
					case "double":
						builder.WriteString(fmt.Sprintf("\tFMOVD F0, result+%d(FP)\n", offset))
					case "float":
						builder.WriteString(fmt.Sprintf("\tFMOVS F0, result+%d(FP)\n", offset))
					case "float16_t":
						// Store 16-bit float from FP register via GP register
						builder.WriteString("\tFMOVS F0, R9\n")
						builder.WriteString(fmt.Sprintf("\tMOVH R9, result+%d(FP)\n", offset))
					default:
						// Check for NEON vector return types
						if IsNeonType(function.Type) {
							is64bit := IsNeon64Type(function.Type)
							vecCount := NeonVectorCount(function.Type)
							resultOffset := offset
							for v := range vecCount {
								vReg := arm64NeonRegisters[v] // V0, V1, V2, V3...
								if is64bit {
									// 64-bit vector: extract D[0] only
									builder.WriteString(fmt.Sprintf("\tVMOV %s.D[0], R9\n", vReg))
									builder.WriteString(fmt.Sprintf("\tMOVD R9, result+%d(FP)\n", resultOffset))
									resultOffset += 8
								} else {
									// 128-bit vector: extract both D[0] and D[1]
									// Use _N suffixes (N=offset within result) so go vet accepts different offsets
									localOffset := v * 16 // offset within result parameter
									builder.WriteString(fmt.Sprintf("\tVMOV %s.D[0], R9\n", vReg))
									builder.WriteString(fmt.Sprintf("\tVMOV %s.D[1], R10\n", vReg))
									builder.WriteString(fmt.Sprintf("\tMOVD R9, result_%d+%d(FP)\n", localOffset, resultOffset))
									builder.WriteString(fmt.Sprintf("\tMOVD R10, result_%d+%d(FP)\n", localOffset+8, resultOffset+8))
									resultOffset += 16
								}
							}
						} else {
							return fmt.Errorf("unsupported return type: %v", function.Type)
						}
					}
				}
				builder.WriteString("\tRET\n")
			} else {
				builder.WriteString(line.String())
			}
		}
	}

	// Write file
	f, err := os.Create(t.GoAssembly)
	if err != nil {
		return err
	}
	defer func(f *os.File) {
		if err = f.Close(); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}(f)
	bytes, err := asmfmt.Format(strings.NewReader(builder.String()))
	if err != nil {
		return err
	}
	_, err = f.Write(bytes)
	return err
}

func init() {
	RegisterParser("arm64", &ARM64Parser{})
}
