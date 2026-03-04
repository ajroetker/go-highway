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
)

// strings package is used for assembly instruction parsing

// SVE/SME/SVE2 instruction patterns for automatic detection and transformation.
// These patterns detect ARM scalable vector instructions in compiler output
// and enable automatic streaming mode injection for macOS compatibility.

var (
	// SVE (Scalable Vector Extension) patterns
	// Z registers: z0-z31 with element type suffix (z0.s, z0.d, etc.)
	sveZReg = regexp.MustCompile(`\bz\d+\.\w+`)
	// Predicate registers: p0-p15 with suffix
	svePReg = regexp.MustCompile(`\bp\d+[/\.]`)
	// PTRUE: set predicate true
	svePtrue = regexp.MustCompile(`ptrue\s+p\d+`)
	// CNT variants: count active elements
	sveCnt = regexp.MustCompile(`cnt[bhwd]\s+`)
	// SVE loads: ld1b, ld1h, ld1w, ld1d
	sveLD1 = regexp.MustCompile(`ld1[bhwd]\s+\{z\d+`)
	// SVE stores: st1b, st1h, st1w, st1d
	sveST1 = regexp.MustCompile(`st1[bhwd]\s+\{z\d+`)
	// SVE contiguous loads: ldr z0, [x0]
	sveLDR = regexp.MustCompile(`ldr\s+z\d+,`)
	// SVE contiguous stores: str z0, [x0]
	sveSTR = regexp.MustCompile(`str\s+z\d+,`)
	// DUP: broadcast scalar to vector
	sveDUP = regexp.MustCompile(`dup\s+z\d+\.\w+,\s*[wx]\d+`)

	// SVE prologue/epilogue instructions (stack management)
	// addvl: add vector length scaled immediate to register
	sveAddvl = regexp.MustCompile(`addvl\s+`)
	// rdsvl: read scalable vector length
	sveRdsvl = regexp.MustCompile(`rdsvl\s+`)

	// SVE2 (extended) patterns
	sve2Match  = regexp.MustCompile(`(match|nmatch|histcnt|histseg)\s+`)
	sve2Crypto = regexp.MustCompile(`(aesd|aese|aesmc|aesimc)\s+`)

	// SME (Scalable Matrix Extension) patterns
	// FMOPA: outer product accumulate (key instruction)
	smeFMOPA = regexp.MustCompile(`fmopa\s+za\d+`)
	// MOVA: move to/from ZA tiles
	smeMOVA = regexp.MustCompile(`mova\s+(z\d+|za\d+)`)
	// ZERO: zero ZA tiles
	smeZero = regexp.MustCompile(`zero\s+\{za`)

	// MOVA ZA→Z (tile-to-vector read) - needs encoding fix (bit 17)
	movaZAtoZ = regexp.MustCompile(`mova\s+z\d+\.\w+,\s*p\d+/m,\s*za\d+[hv]`)

	// Forbidden instructions in streaming mode (macOS)
	// movi d0, #0 causes SIGILL in streaming mode
	forbidMOVI = regexp.MustCompile(`movi\s+d(\d+),\s*#0`)
	// Scalar float loads from memory
	scalarFloatLoad = regexp.MustCompile(`ldr\s+s\d+,`)

	// Branch detection patterns for streaming mode injection
	// Matches various branch instructions: B, B.CC, Bcc, CBZ/CBNZ, TBZ/TBNZ
	sveBranchDetect = regexp.MustCompile(`(?i)^(?:b(?:\.[a-z]+|[a-z]{2})?\s|cbn?z|tbn?z)`)
	// Matches branch instructions and captures the target label
	sveBranchTarget = regexp.MustCompile(`(?i)^(?:B(?:\.[A-Z]+|[A-Z]{2})?|CBN?Z|TBN?Z)\s+.*?(\w+)\s*$`)
)

// SVE scalable vector types - size is runtime-determined (-1)
var sveTypes = map[string]int{
	// Signed integers
	"svint8_t":  -1,
	"svint16_t": -1,
	"svint32_t": -1,
	"svint64_t": -1,
	// Unsigned integers
	"svuint8_t":  -1,
	"svuint16_t": -1,
	"svuint32_t": -1,
	"svuint64_t": -1,
	// Floating point
	"svfloat16_t": -1,
	"svfloat32_t": -1,
	"svfloat64_t": -1,
	// BFloat16
	"svbfloat16_t": -1,
	// Predicate (boolean mask)
	"svbool_t": -1,
}

// IsSVEType returns true if the type is an SVE scalable vector type
func IsSVEType(t string) bool {
	_, ok := sveTypes[t]
	return ok
}

// SVEPrologue returns C parser prologue for SVE/SME types and intrinsic stubs
func SVEPrologue() string {
	var prologue strings.Builder

	// Define SVE scalable vector types (size is runtime-determined)
	// Use 256 bytes as placeholder size (max SVE vector length on current hardware)
	prologue.WriteString("\n// SVE scalable vector types\n")
	prologue.WriteString("typedef struct { char _[256]; } svint8_t;\n")
	prologue.WriteString("typedef struct { char _[256]; } svint16_t;\n")
	prologue.WriteString("typedef struct { char _[256]; } svint32_t;\n")
	prologue.WriteString("typedef struct { char _[256]; } svint64_t;\n")
	prologue.WriteString("typedef struct { char _[256]; } svuint8_t;\n")
	prologue.WriteString("typedef struct { char _[256]; } svuint16_t;\n")
	prologue.WriteString("typedef struct { char _[256]; } svuint32_t;\n")
	prologue.WriteString("typedef struct { char _[256]; } svuint64_t;\n")
	prologue.WriteString("typedef struct { char _[256]; } svfloat16_t;\n")
	prologue.WriteString("typedef struct { char _[256]; } svfloat32_t;\n")
	prologue.WriteString("typedef struct { char _[256]; } svfloat64_t;\n")
	prologue.WriteString("typedef struct { char _[256]; } svbfloat16_t;\n")
	prologue.WriteString("typedef struct { char _[32]; } svbool_t;\n")

	// SME function attributes (macros that expand to nothing for parser)
	prologue.WriteString("\n// SME function attributes (stubs for parser)\n")
	prologue.WriteString("#define __arm_streaming\n")
	prologue.WriteString("#define __arm_streaming_compatible\n")
	prologue.WriteString("#define __arm_locally_streaming\n")
	prologue.WriteString("#define __arm_new_za\n")
	prologue.WriteString("#define __arm_shared_za\n")
	prologue.WriteString("#define __arm_preserves_za\n")
	prologue.WriteString("#define __arm_in(x)\n")
	prologue.WriteString("#define __arm_out(x)\n")
	prologue.WriteString("#define __arm_inout(x)\n")

	// SVE/SME intrinsic function stubs (for parser only, clang provides real ones)
	prologue.WriteString("\n// SVE/SME intrinsic stubs (parser only)\n")
	// Predicate functions
	prologue.WriteString("static inline svbool_t svptrue_b8(void) { svbool_t r; return r; }\n")
	prologue.WriteString("static inline svbool_t svptrue_b16(void) { svbool_t r; return r; }\n")
	prologue.WriteString("static inline svbool_t svptrue_b32(void) { svbool_t r; return r; }\n")
	prologue.WriteString("static inline svbool_t svptrue_b64(void) { svbool_t r; return r; }\n")

	// Undefined value functions
	prologue.WriteString("static inline svfloat32_t svundef_f32(void) { svfloat32_t r; return r; }\n")
	prologue.WriteString("static inline svfloat64_t svundef_f64(void) { svfloat64_t r; return r; }\n")
	prologue.WriteString("static inline svint32_t svundef_s32(void) { svint32_t r; return r; }\n")
	prologue.WriteString("static inline svint64_t svundef_s64(void) { svint64_t r; return r; }\n")

	// Load functions
	prologue.WriteString("static inline svfloat32_t svld1_f32(svbool_t p, const float *ptr) { svfloat32_t r; (void)p; (void)ptr; return r; }\n")
	prologue.WriteString("static inline svfloat64_t svld1_f64(svbool_t p, const double *ptr) { svfloat64_t r; (void)p; (void)ptr; return r; }\n")
	prologue.WriteString("static inline svint32_t svld1_s32(svbool_t p, const int *ptr) { svint32_t r; (void)p; (void)ptr; return r; }\n")
	prologue.WriteString("static inline svint64_t svld1_s64(svbool_t p, const long *ptr) { svint64_t r; (void)p; (void)ptr; return r; }\n")

	// Store functions
	prologue.WriteString("static inline void svst1_f32(svbool_t p, float *ptr, svfloat32_t v) { (void)p; (void)ptr; (void)v; }\n")
	prologue.WriteString("static inline void svst1_f64(svbool_t p, double *ptr, svfloat64_t v) { (void)p; (void)ptr; (void)v; }\n")
	prologue.WriteString("static inline void svst1_s32(svbool_t p, int *ptr, svint32_t v) { (void)p; (void)ptr; (void)v; }\n")
	prologue.WriteString("static inline void svst1_s64(svbool_t p, long *ptr, svint64_t v) { (void)p; (void)ptr; (void)v; }\n")

	// SME zero ZA
	prologue.WriteString("static inline void svzero_za(void) { }\n")

	// SME FMOPA (outer product accumulate) - the key SME instruction
	prologue.WriteString("static inline void svmopa_za32_f32_m(int tile, svbool_t pn, svbool_t pm, svfloat32_t zn, svfloat32_t zm) { (void)tile; (void)pn; (void)pm; (void)zn; (void)zm; }\n")
	prologue.WriteString("static inline void svmopa_za64_f64_m(int tile, svbool_t pn, svbool_t pm, svfloat64_t zn, svfloat64_t zm) { (void)tile; (void)pn; (void)pm; (void)zn; (void)zm; }\n")

	// SME MOVA read from ZA tile
	prologue.WriteString("static inline svfloat32_t svread_hor_za32_f32_m(svfloat32_t zd, svbool_t pg, int tile, int row) { (void)zd; (void)pg; (void)tile; (void)row; svfloat32_t r; return r; }\n")
	prologue.WriteString("static inline svfloat64_t svread_hor_za64_f64_m(svfloat64_t zd, svbool_t pg, int tile, int row) { (void)zd; (void)pg; (void)tile; (void)row; svfloat64_t r; return r; }\n")

	return prologue.String()
}

// usesSVEorSME checks if the function uses any SVE/SME instructions
func usesSVEorSME(lines []*arm64Line) bool {
	for _, line := range lines {
		if line.Assembly == "" {
			continue
		}
		// Check for Z registers (SVE vectors)
		if sveZReg.MatchString(line.Assembly) {
			return true
		}
		// Check for SVE loads/stores (ldr z0, str z0)
		if sveLDR.MatchString(line.Assembly) || sveSTR.MatchString(line.Assembly) {
			return true
		}
		// Check for SME outer product
		if smeFMOPA.MatchString(line.Assembly) {
			return true
		}
		// Check for ZA tile operations
		if smeMOVA.MatchString(line.Assembly) {
			return true
		}
		if smeZero.MatchString(line.Assembly) {
			return true
		}
	}
	return false
}

// transformForbiddenInstructions replaces instructions that cause SIGILL
// in streaming mode on macOS with compatible alternatives.
func transformForbiddenInstructions(lines []*arm64Line) []*arm64Line {
	result := make([]*arm64Line, 0, len(lines))
	for _, line := range lines {
		if line.Assembly == "" {
			result = append(result, line)
			continue
		}

		// Replace movi d0, #0 with fmov s0, wzr
		if matches := forbidMOVI.FindStringSubmatch(line.Assembly); len(matches) > 1 {
			regNum := matches[1]
			regNumInt, _ := strconv.Atoi(regNum)
			// Calculate encoding: fmov sN, wzr = 0x1e2703e0 + (N << 0)
			encoding := uint32(0x1e2703e0) + uint32(regNumInt)
			result = append(result, &arm64Line{
				Labels:   line.Labels,
				Assembly: fmt.Sprintf("fmov\ts%s, wzr", regNum),
				Binary:   fmt.Sprintf("%08x", encoding),
			})
			continue
		}

		result = append(result, line)
	}
	return result
}

// fixMOVAEncoding ensures MOVA ZA→Z (tile-to-vector read) has bit 17 set.
// This is critical for Apple M4 compatibility where 0xc080 fails but 0xc082 works.
func fixMOVAEncoding(lines []*arm64Line) []*arm64Line {
	result := make([]*arm64Line, 0, len(lines))
	for _, line := range lines {
		if line.Assembly == "" || line.Binary == "" {
			result = append(result, line)
			continue
		}

		// Check for MOVA ZA→Z instruction
		if movaZAtoZ.MatchString(line.Assembly) {
			// Parse the binary encoding
			if len(line.Binary) >= 8 {
				encoding, err := strconv.ParseUint(line.Binary, 16, 32)
				if err == nil {
					// Check if this is a ZA→Z read (should have 0xc082 prefix, not 0xc080)
					// The encoding starts with c08 but bit 17 (0x00020000) must be set
					if (encoding & 0xfff00000) == 0xc0800000 {
						// Bit 17 not set - fix it
						if (encoding & 0x00020000) == 0 {
							encoding |= 0x00020000
							result = append(result, &arm64Line{
								Labels:   line.Labels,
								Assembly: line.Assembly,
								Binary:   fmt.Sprintf("%08x", encoding),
							})
							continue
						}
					}
				}
			}
		}

		result = append(result, line)
	}
	return result
}

// SVEContext holds pre-computed analysis results for streaming mode injection.
// A single pass over the instruction list populates all fields, avoiding
// redundant scans in the dispatch paths.
type SVEContext struct {
	FirstSVE           int            // Index of first SVE/SME instruction (-1 if none)
	HasSmstart         bool           // Compiler already generated smstart
	HasBranchBeforeSVE bool           // Branch instructions exist before first SVE
	HasLabelsBeforeSVE bool           // Labels exist before first SVE (skip entry point)
	LabelToLine        map[string]int // Label name → line index mapping
}

// analyzeSVEContext performs a single pass over the instruction list to
// compute all the information needed for streaming mode injection.
func analyzeSVEContext(lines []*arm64Line) SVEContext {
	ctx := SVEContext{
		FirstSVE:    -1,
		LabelToLine: make(map[string]int),
	}

	for i, line := range lines {
		for _, label := range line.Labels {
			ctx.LabelToLine[label] = i
		}

		if line.Assembly == "" {
			if ctx.FirstSVE < 0 && i > 0 && len(line.Labels) > 0 {
				ctx.HasLabelsBeforeSVE = true
			}
			continue
		}

		if strings.Contains(line.Assembly, "smstart") {
			ctx.HasSmstart = true
		}

		if ctx.FirstSVE < 0 {
			if isSVEInstruction(line.Assembly) {
				ctx.FirstSVE = i
			} else {
				if sveBranchDetect.MatchString(line.Assembly) {
					ctx.HasBranchBeforeSVE = true
				}
				if i > 0 && len(line.Labels) > 0 {
					ctx.HasLabelsBeforeSVE = true
				}
			}
		}
	}

	return ctx
}

// injectStreamingMode wraps SVE/SME code sections with smstart/smstop.
// It analyzes the function's control flow in a single pass, then dispatches
// to the appropriate injection strategy.
func injectStreamingMode(lines []*arm64Line) []*arm64Line {
	ctx := analyzeSVEContext(lines)

	if ctx.HasSmstart {
		return ensureSmstopBeforeRet(lines)
	}

	if ctx.FirstSVE < 0 {
		return lines
	}

	if ctx.HasBranchBeforeSVE || ctx.HasLabelsBeforeSVE {
		return injectStreamingModeConservative(lines, ctx)
	}

	return injectStreamingModeSimple(lines, ctx.FirstSVE)
}

// injectStreamingModeSimple handles functions with simple control flow
// (no branches or labels before the first SVE instruction).
// It moves SVE setup instructions (ptrue, cnt*) inside the streaming section.
func injectStreamingModeSimple(lines []*arm64Line, firstSVE int) []*arm64Line {
	// Separate instructions before firstSVE into:
	// - setupLines: ptrue, cnt* (must be inside streaming mode)
	// - preambleLines: everything else (must be before streaming mode)
	var setupLines, preambleLines []*arm64Line
	for i := range firstSVE {
		if lines[i].Assembly == "" {
			// Labels belong with their next instruction
			if len(lines[i].Labels) > 0 {
				// Check what comes next to decide where label goes
				if i+1 < firstSVE {
					preambleLines = append(preambleLines, lines[i])
				} else {
					setupLines = append(setupLines, lines[i])
				}
			} else {
				preambleLines = append(preambleLines, lines[i])
			}
			continue
		}

		if svePtrue.MatchString(lines[i].Assembly) ||
			sveCnt.MatchString(lines[i].Assembly) {
			setupLines = append(setupLines, lines[i])
		} else {
			preambleLines = append(preambleLines, lines[i])
		}
	}

	// Build result: preamble → smstart → setup → body → smstop → ret
	result := make([]*arm64Line, 0, len(lines)+2+len(setupLines))
	result = append(result, preambleLines...)
	result = append(result, &arm64Line{
		Assembly: "smstart\tsm",
		Binary:   "d503477f",
	})
	result = append(result, setupLines...)

	for i := firstSVE; i < len(lines); i++ {
		if lines[i].Assembly == "ret" {
			result = append(result, &arm64Line{
				Assembly: "smstop\tsm",
				Binary:   "d503467f",
			})
		}
		result = append(result, lines[i])
	}

	return result
}

// injectStreamingModeConservative adds smstart at multiple entry points in
// complex control flow functions. This ensures all SVE code paths have
// streaming mode enabled while avoiding redundant smstart calls in loops.
//
// The strategy:
//  1. Find all branch targets from BEFORE firstSVE that have SVE instructions
//     (these are code paths that bypass the main streaming entry)
//  2. Inject smstart before firstSVE and before the first SVE in each bypass path
//  3. Inject smstop before ALL ret instructions
//
// This avoids injecting redundant smstart in loop bodies that are only reached
// from already-streaming code.
//
// Note: smstop when not in streaming mode is harmless (no-op)
func injectStreamingModeConservative(lines []*arm64Line, ctx SVEContext) []*arm64Line {
	// Find branch targets from BEFORE firstSVE
	// These are the blocks that can bypass the main streaming entry
	branchTargetsBeforeFirstSVE := make(map[int]bool)
	for i := range ctx.FirstSVE {
		if matches := sveBranchTarget.FindStringSubmatch(lines[i].Assembly); len(matches) > 1 {
			targetLabel := matches[1]
			lookupLabel := normalizeLabel(targetLabel)
			if targetLine, ok := ctx.LabelToLine[lookupLabel]; ok {
				// Only consider targets that come after firstSVE (bypass paths)
				if targetLine > ctx.FirstSVE {
					branchTargetsBeforeFirstSVE[targetLine] = true
				}
			}
		}
	}

	// Find all positions where smstart should be injected
	smstartPositions := make(map[int]bool)

	// Always inject before firstSVE
	smstartPositions[ctx.FirstSVE] = true

	// For each branch target that bypasses firstSVE, find first SVE in that block.
	// We scan forward through fallthrough blocks (labels don't stop execution flow)
	// until we find an SVE instruction, a ret, or an unconditional branch.
	for targetLine := range branchTargetsBeforeFirstSVE {
		for j := targetLine; j < len(lines); j++ {
			if isSVEInstruction(lines[j].Assembly) {
				smstartPositions[j] = true
				break
			}
			// Stop at ret or unconditional branch (no fallthrough possible)
			asm := lines[j].Assembly
			if asm == "ret" {
				break
			}
			if strings.HasPrefix(asm, "b\t") || strings.HasPrefix(asm, "b ") || asm == "b" {
				break
			}
		}
	}

	result := make([]*arm64Line, 0, len(lines)+10)

	for i, line := range lines {
		// Inject smstart before positions that need it
		if smstartPositions[i] {
			result = append(result, &arm64Line{
				Assembly: "smstart\tsm",
				Binary:   "d503477f",
			})
		}

		// Inject smstop before ALL ret instructions in the function
		// smstop when not in streaming mode is harmless (no-op)
		//
		// IMPORTANT: If the ret line also has labels (branch targets), we need to:
		// 1. Put the labels on the smstop line (so branches land at smstop)
		// 2. Then output ret (without labels)
		// This ensures all paths to ret go through smstop.
		if line.Assembly == "ret" {
			if len(line.Labels) > 0 {
				// Put labels on smstop, then ret without labels
				result = append(result, &arm64Line{
					Labels:   line.Labels,
					Assembly: "smstop\tsm",
					Binary:   "d503467f",
				})
				result = append(result, &arm64Line{
					Assembly: "ret",
				})
				continue // Skip the normal line addition
			}
			// No labels - just add smstop before ret
			result = append(result, &arm64Line{
				Assembly: "smstop\tsm",
				Binary:   "d503467f",
			})
		}

		// Add the current line
		result = append(result, line)
	}

	return result
}

// isSVEInstruction checks if an assembly instruction is an SVE/SME instruction
func isSVEInstruction(asm string) bool {
	if asm == "" {
		return false
	}
	return sveZReg.MatchString(asm) ||
		smeFMOPA.MatchString(asm) ||
		smeMOVA.MatchString(asm) ||
		smeZero.MatchString(asm) ||
		sveLD1.MatchString(asm) ||
		sveST1.MatchString(asm) ||
		sveLDR.MatchString(asm) ||
		sveSTR.MatchString(asm) ||
		sveDUP.MatchString(asm) ||
		svePtrue.MatchString(asm) ||
		sveCnt.MatchString(asm) ||
		sveAddvl.MatchString(asm) ||
		sveRdsvl.MatchString(asm)
}

// ensureSmstopBeforeRet adds smstop before all ret instructions in an SVE function.
// The caller has already verified that smstart is present.
// smstop when not in streaming mode is a no-op, so it's safe to add conservatively.
func ensureSmstopBeforeRet(lines []*arm64Line) []*arm64Line {
	// Add smstop before ALL ret instructions that don't already have one.
	// smstop when not in streaming mode is a no-op, so it's safe to add.
	result := make([]*arm64Line, 0, len(lines)+5)

	for i, line := range lines {
		// Add smstop before ret if not already present
		//
		// IMPORTANT: If the ret line also has labels (branch targets), we need to:
		// 1. Put the labels on the smstop line (so branches land at smstop)
		// 2. Then output ret (without labels)
		// This ensures all paths to ret go through smstop.
		if line.Assembly == "ret" {
			if len(line.Labels) > 0 {
				// Put labels on smstop, then ret without labels
				result = append(result, &arm64Line{
					Labels:   line.Labels,
					Assembly: "smstop\tsm",
					Binary:   "d503467f",
				})
				result = append(result, &arm64Line{
					Assembly: "ret",
				})
				continue // Skip the normal line addition
			}
			// No labels - check if previous line is already smstop
			alreadyHasSmstop := false
			if i > 0 && len(result) > 0 {
				prev := result[len(result)-1]
				if prev.Assembly != "" && strings.Contains(prev.Assembly, "smstop") {
					alreadyHasSmstop = true
				}
			}
			if !alreadyHasSmstop {
				result = append(result, &arm64Line{
					Assembly: "smstop\tsm",
					Binary:   "d503467f",
				})
			}
		}
		result = append(result, line)
	}

	return result
}

// TransformSVEFunction applies all SVE/SME transformations to a function.
// This is the main entry point called from arm64_parser.go.
func TransformSVEFunction(lines []*arm64Line) []*arm64Line {
	if !usesSVEorSME(lines) {
		return lines
	}

	// Apply transformations in order:
	// 1. Fix MOVA encoding (bit 17 for ZA→Z reads)
	lines = fixMOVAEncoding(lines)

	// 2. Replace forbidden instructions (movi d0, #0 → fmov s0, wzr)
	lines = transformForbiddenInstructions(lines)

	// 3. Inject streaming mode (smstart/smstop)
	lines = injectStreamingMode(lines)

	return lines
}

// smeMovaImm matches MOVA/MOV instructions with an immediate #N base offset
// instead of a register (w12-w15) in ZA tile access.
// Clang 22 no longer accepts this shorthand; the base must be a register.
// Example: "mova z29.s, p0/m, za0h.s[#0, 1]" → "mov z29.s, p0/m, za0h.s[w12, 1]"
var smeMovaImm = regexp.MustCompile(
	`(?P<pre>(?:mova|mov)\s+z\d+\.\w+,\s*p\d+/m,\s*za\d+[hv]\.\w+\[)#(?P<base>\d+)(?P<post>,\s*\d+\])`,
)

// smeMovaBaseToReg maps immediate base values to the corresponding w12-w15
// register names.  The SME MOVA encoding uses a 2-bit Rv field where
// 0→w12, 1→w13, 2→w14, 3→w15.
var smeMovaBaseToReg = map[string]string{
	"0": "w12",
	"1": "w13",
	"2": "w14",
	"3": "w15",
}

// fixSMEMovaImmediate rewrites the clang-generated assembly file in place,
// replacing MOVA/MOV instructions that use an immediate #N base offset in
// ZA tile access with the register form (w12-w15).
//
// Clang ≥22's integrated assembler rejects the immediate form that clang's
// own code generator emits (e.g., "mova z29.s, p0/m, za0h.s[#0, 1]").
// The correct syntax is "mov z29.s, p0/m, za0h.s[w12, 1]" with w12 set
// to 0 beforehand.
//
// For each rewritten instruction, this function inserts a preceding
// "mov wR, #N" to ensure the register holds the correct base value.
// This is safe because:
//   - The inserted mov and the tile read are back-to-back (no intervening use of wR)
//   - The SME tile read is the sole consumer of wR at that point
//   - Clang's own register allocator already avoids w12-w15 for general use
//     when SME tile operations are present in the function
func fixSMEMovaImmediate(asmPath string) error {
	data, err := os.ReadFile(asmPath)
	if err != nil {
		return fmt.Errorf("fixSMEMovaImmediate: read %s: %w", asmPath, err)
	}

	lines := strings.Split(string(data), "\n")
	changed := false
	var result []string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		loc := smeMovaImm.FindStringSubmatchIndex(trimmed)
		if loc == nil {
			result = append(result, line)
			continue
		}

		// Extract named groups
		baseStr := trimmed[loc[4]:loc[5]] // "base" group — the immediate value
		reg, ok := smeMovaBaseToReg[baseStr]
		if !ok {
			// Immediate value outside 0-3 — leave unchanged (shouldn't happen
			// for valid SME assembly, but be defensive).
			result = append(result, line)
			continue
		}

		// Determine leading whitespace to preserve indentation
		indent := line[:len(line)-len(strings.TrimLeft(line, " \t"))]

		// Insert "mov wR, #N" before the tile read instruction.
		// When the base is 0 we use wzr for clarity; otherwise #N.
		var movLine string
		if baseStr == "0" {
			movLine = indent + "mov\t" + reg + ", wzr"
		} else {
			movLine = indent + "mov\t" + reg + ", #" + baseStr
		}
		result = append(result, movLine)

		// Rewrite the MOVA/MOV instruction:
		// 1. Replace #N with the register name
		// 2. Normalize mnemonic to "mov" (mova is an alias)
		//
		// Submatch indices: loc[0:2]=full, loc[2:4]=pre, loc[4:6]=base, loc[6:8]=post
		// pre includes up to "[", then "#base" is replaced by reg, then post is ", N]"
		fixed := trimmed[:loc[3]] + reg + trimmed[loc[6]:]
		if strings.HasPrefix(strings.TrimSpace(fixed), "mova") {
			fixed = strings.Replace(fixed, "mova", "mov", 1)
		}
		result = append(result, indent+fixed)
		changed = true
	}

	if !changed {
		return nil
	}

	return os.WriteFile(asmPath, []byte(strings.Join(result, "\n")), 0644)
}

// Note: SVE type definitions are in arm64_parser.go Prologue() function
