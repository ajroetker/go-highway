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
	"strings"
	"unicode"

	"github.com/klauspost/asmfmt"
	"github.com/samber/lo"
)

// Loong64Parser implements ArchParser for LoongArch64 architecture
type Loong64Parser struct{}

// loong64 regex patterns
var (
	loong64LabelLine = regexp.MustCompile(`^\.L\w*_\d+:.*$`)
	loong64CodeLine  = regexp.MustCompile(`^\s+\w+.+$`)

	// Constant pool patterns
	// Match constant pool labels: .LCPI0_0:
	loong64ConstPoolLabel = regexp.MustCompile(`^\.LCPI\d+_\d+:`)
	// Match .word directive with hex or decimal value (32-bit)
	loong64WordDirective = regexp.MustCompile(`^\s+\.word\s+(0x[0-9a-fA-F]+|\d+)`)
	// Match .dword directive with hex or decimal value (64-bit)
	loong64DwordDirective = regexp.MustCompile(`^\s+\.dword\s+(0x[0-9a-fA-F]+|\d+)`)
	// Match .byte directive with hex or decimal value
	loong64ByteDirective = regexp.MustCompile(`^\s+\.byte\s+(0x[0-9a-fA-F]+|\d+)`)
	// Match pcaddu12i instruction referencing constant pool: pcaddu12i $r4, %pc_hi20(.LCPI0_0)
	loong64PcadduConstPool = regexp.MustCompile(`pcaddu12i\s+(\$\w+),\s*%pc_hi20\(\.LCPI(\d+_\d+)\)`)
	// Match ld.w/ld.d instruction with constant pool reference: ld.d $r4, $r4, %pc_lo12(.LCPI0_0)
	loong64LdConstPool = regexp.MustCompile(`ld\.[wd]\s+(\$\w+),\s*(\$\w+),\s*%pc_lo12\(\.LCPI(\d+_\d+)\)`)
)

// loong64 register sets
var (
	loong64Registers   = []string{"R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}
	loong64FPRegisters = []string{"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"}
)

// loong64 register aliases (ABI names to Go names)
var loong64RegistersAlias = map[string]string{
	"$zero": "R0",
	"$ra":   "R1",
	"$tp":   "R2",
	"$sp":   "R3",
	"$a0":   "R4",
	"$a1":   "R5",
	"$a2":   "R6",
	"$a3":   "R7",
	"$a4":   "R8",
	"$a5":   "R9",
	"$a6":   "R10",
	"$a7":   "R11",
	"$t0":   "R12",
	"$t1":   "R13",
	"$t2":   "R14",
	"$t3":   "R15",
	"$t4":   "R16",
	"$t5":   "R17",
	"$t6":   "R18",
	"$t7":   "R19",
	"$t8":   "R20",
	"$fp":   "R22",
	"$s0":   "R23",
	"$s1":   "R24",
	"$s2":   "R25",
	"$s3":   "R26",
	"$s4":   "R27",
	"$s5":   "R28",
	"$s6":   "R29",
	"$s7":   "R30",
	"$s8":   "R31",
	"$s9":   "R22",
}

// loong64 operation aliases
var loong64OpAlias = map[string]string{
	"b":    "JMP",
	"bnez": "BNE",
}

// loong64Line represents a single assembly instruction for LoongArch64
// Binary is string because LoongArch has fixed-width 32-bit instructions
type loong64Line struct {
	Labels   []string
	Assembly string
	Binary   string
}

func (line *loong64Line) String() string {
	var builder strings.Builder
	builder.WriteString("\t")
	if strings.HasPrefix(line.Assembly, "b") && !strings.HasPrefix(line.Assembly, "bstrins") {
		splits := strings.Split(line.Assembly, ".")
		op := strings.TrimSpace(splits[0])
		registers := strings.FieldsFunc(op, func(r rune) bool {
			return unicode.IsSpace(r) || r == ','
		})
		if o, ok := loong64OpAlias[registers[0]]; !ok {
			builder.WriteString(strings.ToUpper(registers[0]))
		} else {
			builder.WriteString(o)
		}
		builder.WriteRune(' ')
		for i := 1; i < len(registers); i++ {
			if r, ok := loong64RegistersAlias[registers[i]]; !ok {
				_, _ = fmt.Fprintln(os.Stderr, "unexpected register alias:", registers[i])
				os.Exit(1)
			} else {
				builder.WriteString(r)
				builder.WriteRune(',')
			}
		}
		builder.WriteString(splits[1])
	} else {
		builder.WriteString("\t")
		builder.WriteString(fmt.Sprintf("WORD $0x%v", line.Binary))
		builder.WriteString("\t// ")
		builder.WriteString(line.Assembly)
	}
	builder.WriteString("\n")
	return builder.String()
}

// Name returns the architecture name
func (p *Loong64Parser) Name() string {
	return "loong64"
}

// BuildTags returns the Go build constraint
func (p *Loong64Parser) BuildTags() string {
	return "//go:build !noasm && loong64\n"
}

// BuildTarget returns the clang target triple
func (p *Loong64Parser) BuildTarget(goos string) string {
	return "loongarch64-linux-gnu"
}

// CompilerFlags returns architecture-specific compiler flags
func (p *Loong64Parser) CompilerFlags() []string {
	return nil // LoongArch64 doesn't need special fixed-register flags
}

// Prologue returns C parser prologue for architecture-specific types
func (p *Loong64Parser) Prologue() string {
	var prologue strings.Builder
	prologue.WriteString("#define GOAT_PARSER 1\n")
	// Define include guards for LoongArch SIMD headers
	prologue.WriteString("#define _LSXINTRIN_H 1\n")  // LSX (128-bit SIMD)
	prologue.WriteString("#define _LASXINTRIN_H 1\n") // LASX (256-bit SIMD)
	return prologue.String()
}

// TranslateAssembly implements the full translation pipeline for LoongArch64
func (p *Loong64Parser) TranslateAssembly(t *TranslateUnit, functions []Function) error {
	// Parse assembly
	assembly, stackSizes, constPools, err := p.parseAssembly(t.Assembly)
	if err != nil {
		return err
	}

	// Run objdump, falling back to llvm-objdump for cross-compilation
	dump, err := runObjdump("-d", t.Object)
	if err != nil {
		return err
	}

	// Parse object dump
	if err := p.parseObjectDump(dump, assembly); err != nil {
		return err
	}

	// Copy stack sizes to functions
	for i, fn := range functions {
		if sz, ok := stackSizes[fn.Name]; ok {
			functions[i].StackSize = sz
		}
	}

	// Generate Go assembly with constant pools
	return p.generateGoAssembly(t, functions, assembly, constPools)
}

func (p *Loong64Parser) parseAssembly(path string) (map[string][]*loong64Line, map[string]int, map[string]*ConstPool, error) {
	scanner, cleanup, err := openAssemblyFile(path)
	if err != nil {
		return nil, nil, nil, err
	}
	defer cleanup()

	var (
		stackSizes   = make(map[string]int)
		functions    = make(map[string][]*loong64Line)
		cpa          = NewConstPoolAccumulator()
		functionName string
		labelName    string
	)
	for scanner.Scan() {
		line := scanner.Text()

		// Check for constant pool label (.LCPI0_0:)
		if loong64ConstPoolLabel.MatchString(line) {
			labelPart := strings.TrimPrefix(strings.Split(line, ":")[0], ".L")
			cpa.StartPool(labelPart)
			continue
		}

		// Parse .word/.dword/.byte directives for constant pool data
		if cpa.Active() {
			if matches := loong64WordDirective.FindStringSubmatch(line); matches != nil {
				cpa.AddLong(parseIntValue(matches[1]))
				continue
			}
			if matches := loong64DwordDirective.FindStringSubmatch(line); matches != nil {
				cpa.AddQuad(parseIntValue(matches[1]))
				continue
			}
			if matches := loong64ByteDirective.FindStringSubmatch(line); matches != nil {
				cpa.AccumulateByte(parseIntValue(matches[1]))
				continue
			}
		}

		// Check for section change or function start that ends constant pool parsing
		if nameLine.MatchString(line) || strings.HasPrefix(strings.TrimSpace(line), ".section") {
			cpa.FinishPool()
		}

		if attributeLine.MatchString(line) {
			continue
		} else if nameLine.MatchString(line) {
			functionName = strings.Split(line, ":")[0]
			functions[functionName] = make([]*loong64Line, 0)
		} else if loong64LabelLine.MatchString(line) {
			labelName = strings.Split(line, ":")[0]
			labelName = labelName[1:]
			lines := functions[functionName]
			if len(lines) == 1 || lines[len(lines)-1].Assembly != "" {
				functions[functionName] = append(functions[functionName], &loong64Line{Labels: []string{labelName}})
			} else {
				lines[len(lines)-1].Labels = append(lines[len(lines)-1].Labels, labelName)
			}
		} else if loong64CodeLine.MatchString(line) {
			asm := strings.Split(line, "//")[0]
			asm = strings.TrimSpace(asm)
			if labelName == "" {
				functions[functionName] = append(functions[functionName], &loong64Line{Assembly: asm})
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
		return nil, nil, nil, err
	}
	return functions, stackSizes, constPools, nil
}

func (p *Loong64Parser) parseObjectDump(dump string, functions map[string][]*loong64Line) error {
	var (
		functionName string
		lineNumber   int
	)
	for i, line := range strings.Split(dump, "\n") {
		line = strings.TrimSpace(line)
		if symbolLine.MatchString(line) {
			functionName = extractObjDumpFunctionName(line, "")
			lineNumber = 0
		} else if dataLine.MatchString(line) {
			binaryTokens, assembly := parseObjDumpDataLine(line)
			if len(binaryTokens) == 0 || assembly == "nop" {
				continue
			}
			if lineNumber >= len(functions[functionName]) {
				return fmt.Errorf("%d: unexpected objectdump line: %s", i, line)
			}
			functions[functionName][lineNumber].Binary = binaryTokens[len(binaryTokens)-1]
			lineNumber++
		}
	}
	return nil
}

// loong64GoRegisterName converts LoongArch register names to Go assembly register names
// $a0-$a7 -> R4-R11, $t0-$t8 -> R12-R20, etc.
func loong64GoRegisterName(loongReg string) string {
	if goReg, ok := loong64RegistersAlias[loongReg]; ok {
		return goReg
	}
	// Handle $rN format directly
	if strings.HasPrefix(loongReg, "$r") {
		return "R" + loongReg[2:]
	}
	// Return as-is if not recognized
	return strings.ToUpper(loongReg)
}

func (p *Loong64Parser) generateGoAssembly(t *TranslateUnit, functions []Function, assembly map[string][]*loong64Line, constPools map[string]*ConstPool) error {
	var builder strings.Builder
	builder.WriteString(p.BuildTags())
	t.writeHeader(&builder)

	emitConstPools(&builder, constPools)

	for _, function := range functions {
		// Calculate return size based on type
		returnSize := 0
		if function.Type != "void" {
			if sz, ok := supportedTypes[function.Type]; ok {
				returnSize = sz // Use actual scalar type size
			} else {
				returnSize = 8 // Default 8-byte slot for pointers/unknown types
			}
		}

		builder.WriteString(fmt.Sprintf("\nTEXT ·%v(SB), $%d-%d\n",
			function.Name, returnSize, len(function.Parameters)*8))

		registerCount, fpRegisterCount, offset := 0, 0, 0
		var stack []lo.Tuple2[int, Parameter]

		for _, param := range function.Parameters {
			sz := 8
			if param.Pointer {
				sz = 8
			} else {
				sz = supportedTypes[param.Type]
			}
			if offset%sz != 0 {
				offset += sz - offset%sz
			}
			if !param.Pointer && (param.Type == "double" || param.Type == "float" || param.Type == "float16_t") {
				if fpRegisterCount < len(loong64FPRegisters) {
					if param.Type == "double" {
						builder.WriteString(fmt.Sprintf("\tMOVD %s+%d(FP), %s\n", param.Name, offset, loong64FPRegisters[fpRegisterCount]))
					} else if param.Type == "float16_t" {
						// Load 16-bit to GP register, then move to FP register
						builder.WriteString(fmt.Sprintf("\tMOVH %s+%d(FP), R12\n", param.Name, offset))
						builder.WriteString(fmt.Sprintf("\tMOVF R12, %s\n", loong64FPRegisters[fpRegisterCount]))
					} else {
						builder.WriteString(fmt.Sprintf("\tMOVF %s+%d(FP), %s\n", param.Name, offset, loong64FPRegisters[fpRegisterCount]))
					}
					fpRegisterCount++
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			} else {
				if registerCount < len(loong64Registers) {
					builder.WriteString(fmt.Sprintf("\tMOVV %s+%d(FP), %s\n", param.Name, offset, loong64Registers[registerCount]))
					registerCount++
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			}
			offset += sz
		}

		if offset%8 != 0 {
			offset += 8 - offset%8
		}

		frameSize := 0
		if len(stack) > 0 {
			for i := 0; i < len(stack); i++ {
				if stack[i].B.Pointer {
					frameSize += 8
				} else {
					frameSize += supportedTypes[stack[i].B.Type]
				}
			}
			builder.WriteString(fmt.Sprintf("\tADDV $-%d, R3\n", frameSize))
			stackoffset := 0
			for i := 0; i < len(stack); i++ {
				builder.WriteString(fmt.Sprintf("\tMOVV %s+%d(FP), R12\n", stack[i].B.Name, frameSize+stack[i].A))
				builder.WriteString(fmt.Sprintf("\tMOVV R12, (%d)(R3)\n", stackoffset))
				if stack[i].B.Pointer {
					stackoffset += 8
				} else {
					stackoffset += supportedTypes[stack[i].B.Type]
				}
			}
		}

		for _, line := range assembly[function.Name] {
			// Skip pcaddu12i instructions that reference constant pools (they're no longer needed)
			if matches := loong64PcadduConstPool.FindStringSubmatch(line.Assembly); matches != nil {
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

			// Replace ld.w/ld.d instructions that load from constant pools
			if matches := loong64LdConstPool.FindStringSubmatch(line.Assembly); matches != nil {
				destReg := matches[1]
				baseReg := matches[2]
				constLabel := "CPI" + matches[3]
				if _, hasPool := constPools[constLabel]; hasPool {
					// Emit any labels
					for _, label := range line.Labels {
						builder.WriteString(label)
						builder.WriteString(":\n")
					}
					// Emit load address of constant pool into the base register
					builder.WriteString(fmt.Sprintf("\tMOVV $%s<>(SB), %s\n",
						constLabel, loong64GoRegisterName(baseReg)))
					// Emit load from the address
					// Determine if it's a 32-bit or 64-bit load based on the original instruction
					if strings.Contains(line.Assembly, "ld.d") {
						builder.WriteString(fmt.Sprintf("\tMOVV (%s), %s\n",
							loong64GoRegisterName(baseReg), loong64GoRegisterName(destReg)))
					} else {
						// ld.w - 32-bit load
						builder.WriteString(fmt.Sprintf("\tMOVW (%s), %s\n",
							loong64GoRegisterName(baseReg), loong64GoRegisterName(destReg)))
					}
					continue
				}
			}

			for _, label := range line.Labels {
				builder.WriteString(label)
				builder.WriteString(":\n")
			}
			if line.Assembly == "ret" {
				if frameSize > 0 {
					builder.WriteString(fmt.Sprintf("\tADDV $%d, R3\n", frameSize))
				}
				if function.Type != "void" {
					switch function.Type {
					case "int64_t", "uint64_t", "long", "_Bool":
						builder.WriteString(fmt.Sprintf("\tMOVV R4, result+%d(FP)\n", offset))
					case "int32_t", "uint32_t":
						builder.WriteString(fmt.Sprintf("\tMOVW R4, result+%d(FP)\n", offset))
					case "double":
						builder.WriteString(fmt.Sprintf("\tMOVD F0, result+%d(FP)\n", offset))
					case "float":
						builder.WriteString(fmt.Sprintf("\tMOVF F0, result+%d(FP)\n", offset))
					case "float16_t":
						// Store 16-bit float from FP register via GP register
						builder.WriteString("\tMOVF F0, R12\n")
						builder.WriteString(fmt.Sprintf("\tMOVH R12, result+%d(FP)\n", offset))
					default:
						return fmt.Errorf("unsupported return type: %v", function.Type)
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
	RegisterParser("loong64", &Loong64Parser{})
}
