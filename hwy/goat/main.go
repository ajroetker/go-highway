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
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"unicode"
	"sort"
	"strings"

	"github.com/spf13/cobra"
	"modernc.org/cc/v4"
)

var supportedTypes = map[string]int{
	"int32_t":    4,
	"int64_t":    8,
	"uint32_t":   4,
	"uint64_t":   8,
	"long":       8,
	"float":      4,
	"double":     8,
	"_Bool":      1,
	"float16_t":  2,
}

type TranslateUnit struct {
	Source       string
	Assembly     string
	Object       string
	GoAssembly   string
	Go           string
	Package      string
	Options      []string
	IncludePaths []string // Additional include paths for C parser
	Sysroot      string   // Explicit sysroot path for cross-compilation
	Offset       int
	Target       string     // Target architecture (amd64, arm64, etc.)
	TargetOS     string     // Target OS (darwin, linux, etc.)
	parser       ArchParser // Architecture-specific parser
}

func NewTranslateUnit(source string, outputDir string, target string, targetOS string, includePaths []string, sysroot string, options ...string) (TranslateUnit, error) {
	sourceExt := filepath.Ext(source)
	noExtSourcePath := source[:len(source)-len(sourceExt)]
	noExtSourceBase := filepath.Base(noExtSourcePath)

	parser, err := GetParser(target)
	if err != nil {
		return TranslateUnit{}, err
	}

	return TranslateUnit{
		Source:       source,
		Assembly:     noExtSourcePath + ".s",
		Object:       noExtSourcePath + ".o",
		GoAssembly:   filepath.Join(outputDir, noExtSourceBase+".s"),
		Go:           filepath.Join(outputDir, noExtSourceBase+".go"),
		Package:      filepath.Base(outputDir),
		Options:      options,
		IncludePaths: includePaths,
		Sysroot:      sysroot,
		Target:       target,
		TargetOS:     targetOS,
		parser:       parser,
	}, nil
}

// parseSource parse C source file and extract functions declarations.
func (t *TranslateUnit) parseSource() ([]Function, error) {
	f, err := os.Open(t.Source)
	if err != nil {
		return nil, err
	}
	cfg, err := cc.NewConfig(t.TargetOS, t.Target)
	if err != nil {
		return nil, err
	}
	// Add custom include paths for cross-compilation
	if len(t.IncludePaths) > 0 {
		cfg.SysIncludePaths = append(t.IncludePaths, cfg.SysIncludePaths...)
	}
	var prologue strings.Builder
	// Add RISC-V vector type definitions for the C parser
	if t.Target == "riscv64" {
		prologue.WriteString("#define __riscv_vector 1\n")
		for _, typeStr := range []string{"int64", "uint64", "int32", "uint32", "int16", "uint16", "int8", "uint8", "float64", "float32", "float16"} {
			for i := 1; i <= 8; i *= 2 {
				prologue.WriteString(fmt.Sprintf("typedef char v%sm%d_t;\n", typeStr, i))
			}
		}
	}
	// Provide standard integer type aliases from <stdint.h>.
	// Architecture prologues block system SIMD headers (arm_neon.h, immintrin.h,
	// etc.) via include guards, which prevents <stdint.h> from being transitively
	// included. The C parser needs these typedefs to recognize int64_t, int32_t,
	// etc. in function signatures and bodies.
	prologue.WriteString("typedef signed char int8_t;\n")
	prologue.WriteString("typedef short int16_t;\n")
	prologue.WriteString("typedef int int32_t;\n")
	prologue.WriteString("typedef long int64_t;\n")
	prologue.WriteString("typedef unsigned char uint8_t;\n")
	prologue.WriteString("typedef unsigned short uint16_t;\n")
	prologue.WriteString("typedef unsigned int uint32_t;\n")
	prologue.WriteString("typedef unsigned long uint64_t;\n")

	// Add architecture-specific prologue from parser
	prologue.WriteString(t.parser.Prologue())
	ast, err := cc.Parse(cfg, []cc.Source{
		{Name: "<predefined>", Value: cfg.Predefined},
		{Name: "<builtin>", Value: cc.Builtin},
		{Name: "<prologue>", Value: prologue.String()},
		{Name: t.Source, Value: f},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to parse source file %v: %w", t.Source, err)
	}
	var functions []Function
	for tu := ast.TranslationUnit; tu != nil; tu = tu.TranslationUnit {
		externalDeclaration := tu.ExternalDeclaration
		if externalDeclaration.Position().Filename == t.Source && externalDeclaration.Case == cc.ExternalDeclarationFuncDef {
			ds := externalDeclaration.FunctionDefinition.DeclarationSpecifiers
			if hasInlineSpecifier(ds) || hasStorageClass(ds) {
				// Skip inline and static functions — they are internal
				// helpers inlined by clang into exported functions.
				continue
			}
			if function, err := t.convertFunction(externalDeclaration.FunctionDefinition); err != nil {
				return nil, err
			} else {
				functions = append(functions, function)
			}
		}
	}
	sort.Slice(functions, func(i, j int) bool {
		return functions[i].Position < functions[j].Position
	})
	return functions, nil
}

func (t *TranslateUnit) generateGoStubs(functions []Function) error {
	// generate code
	var builder strings.Builder
	builder.WriteString(t.parser.BuildTags())
	t.writeHeader(&builder)
	builder.WriteString(fmt.Sprintf("package %v\n", t.Package))
	if hasPointer(functions) {
		builder.WriteString("\nimport \"unsafe\"\n")
	}
	for _, function := range functions {
		builder.WriteString("\n//go:noescape\n")
		builder.WriteString("func ")
		builder.WriteString(function.Name)
		builder.WriteRune('(')
		for i, param := range function.Parameters {
			if i > 0 {
				builder.WriteString(", ")
			}
			builder.WriteString(param.Name)
			if i+1 == len(function.Parameters) || function.Parameters[i+1].String() != param.String() {
				builder.WriteRune(' ')
				builder.WriteString(param.String())
			}
		}
		builder.WriteRune(')')
		if function.Type != "void" {
			switch function.Type {
			case "_Bool":
				builder.WriteString(" (result bool)")
			case "double":
				builder.WriteString(" (result float64)")
			case "float":
				builder.WriteString(" (result float32)")
			case "float16_t":
				builder.WriteString(" (result uint16)")
			case "int32_t":
				builder.WriteString(" (result int32)")
			case "int64_t", "long":
				builder.WriteString(" (result int64)")
			case "uint32_t":
				builder.WriteString(" (result uint32)")
			case "uint64_t":
				builder.WriteString(" (result uint64)")
			default:
				// Check for NEON vector types
				if sz := NeonTypeSize(function.Type); sz > 0 {
					builder.WriteString(fmt.Sprintf(" (result [%d]byte)", sz))
				} else if sz := X86SIMDTypeSize(function.Type); sz > 0 {
					// Check for x86 SIMD vector types
					builder.WriteString(fmt.Sprintf(" (result [%d]byte)", sz))
				} else {
					return fmt.Errorf("unsupported return type: %v", function.Type)
				}
			}
		}
		builder.WriteRune('\n')
	}

	// write file
	f, err := os.Create(t.Go)
	if err != nil {
		return err
	}
	defer func(f *os.File) {
		if err = f.Close(); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}(f)
	_, err = f.WriteString(builder.String())
	return err
}

func (t *TranslateUnit) compile(args ...string) error {
	args = append(args, "-mno-red-zone", "-mllvm", "-inline-threshold=1000",
		"-fno-asynchronous-unwind-tables", "-fno-exceptions", "-fno-rtti", "-fno-builtin",
		"-fomit-frame-pointer", "-fno-stack-protector")
	// Add architecture-specific compiler flags
	args = append(args, t.parser.CompilerFlags()...)

	// Include path strategy: explicit --sysroot flag > auto-detect sysroot > -nostdlibinc.
	// GOAT generates Go assembly that never links against libc, so system headers
	// (glibc/musl) are never needed. When no sysroot is available, -nostdlibinc
	// suppresses system include dirs while keeping clang's built-in headers
	// (arm_neon.h, immintrin.h, stdint.h, etc.) from the resource directory.
	if !argsContainSysroot(args) {
		sysroot := t.Sysroot
		if sysroot == "" && (t.Target != runtime.GOARCH || t.TargetOS != runtime.GOOS) {
			sysroot = detectCrossCompileSysroot(t.Target, t.TargetOS)
		}
		if sysroot != "" {
			args = append(args, "--sysroot="+sysroot)
		} else {
			args = append(args, "-nostdlibinc")
		}
	}

	target := t.parser.BuildTarget(t.TargetOS)
	_, err := runCommand("clang", append([]string{"-S", "-target", target, "-c", t.Source, "-o", t.Assembly}, args...)...)
	if err != nil {
		return err
	}

	// Fix clang ≥22 SME assembly: replace immediate base offsets (#N) in
	// MOVA ZA tile access with .inst directives containing the computed
	// binary encoding.  This bypasses clang's assembler rejecting the #N
	// shorthand without modifying any GP register, preserving clang's
	// register allocation exactly.
	if t.Target == "arm64" {
		if err := fixSMEMovaImmediate(t.Assembly); err != nil {
			return err
		}
	}

	_, err = runCommand("clang", append([]string{"-target", target, "-c", t.Assembly, "-o", t.Object}, args...)...)
	return err
}

// argsContainSysroot checks whether --sysroot is already present in the compiler args
// (e.g. passed via -e/--extra-option).
func argsContainSysroot(args []string) bool {
	for _, arg := range args {
		if arg == "--sysroot" || strings.HasPrefix(arg, "--sysroot=") {
			return true
		}
	}
	return false
}

// detectCrossCompileSysroot auto-detects the sysroot path for cross-compilation.
// Returns empty string if no sysroot is found.
func detectCrossCompileSysroot(targetArch, targetOS string) string {
	switch runtime.GOOS {
	case "linux":
		return detectLinuxSysroot(targetArch, targetOS)
	case "darwin":
		return detectDarwinSysroot(targetArch, targetOS)
	case "windows":
		return detectWindowsSysroot(targetArch, targetOS)
	default:
		return ""
	}
}

// detectLinuxSysroot probes Debian/Ubuntu and Fedora/RHEL cross-compilation sysroot paths.
func detectLinuxSysroot(targetArch, targetOS string) string {
	if targetOS != "linux" {
		return ""
	}

	archToGNUPrefix := map[string]string{
		"arm64":   "aarch64-linux-gnu",
		"amd64":   "x86_64-linux-gnu",
		"riscv64": "riscv64-linux-gnu",
		"loong64": "loongarch64-linux-gnu",
	}

	prefix, ok := archToGNUPrefix[targetArch]
	if !ok {
		return ""
	}

	// Debian/Ubuntu/Arch: /usr/{triplet}/
	sysroot := "/usr/" + prefix
	if _, err := os.Stat(filepath.Join(sysroot, "include")); err == nil {
		return sysroot
	}

	// Fedora/RHEL: /usr/{triplet}/sys-root/
	sysroot = "/usr/" + prefix + "/sys-root"
	if _, err := os.Stat(filepath.Join(sysroot, "usr", "include")); err == nil {
		return sysroot
	}

	return ""
}

// detectDarwinSysroot uses xcrun to locate the macOS SDK for cross-arch compilation.
func detectDarwinSysroot(targetArch, targetOS string) string {
	if targetOS != "darwin" {
		return ""
	}

	out, err := exec.Command("xcrun", "--show-sdk-path").Output()
	if err != nil {
		return ""
	}

	sdkPath := strings.TrimSpace(string(out))
	if _, err := os.Stat(sdkPath); err == nil {
		return sdkPath
	}

	return ""
}

// detectWindowsSysroot probes standard MSYS2/MinGW install locations.
func detectWindowsSysroot(targetArch, targetOS string) string {
	if targetOS != "windows" {
		return ""
	}

	var candidates []string
	switch targetArch {
	case "amd64":
		candidates = []string{
			`C:\msys64\mingw64`,
			`C:\msys64\clang64`,
		}
	case "arm64":
		candidates = []string{
			`C:\msys64\clangarm64`,
		}
	default:
		return ""
	}

	for _, c := range candidates {
		if _, err := os.Stat(filepath.Join(c, "include")); err == nil {
			return c
		}
	}

	return ""
}

func (t *TranslateUnit) Translate() error {
	functions, err := t.parseSource()
	if err != nil {
		return err
	}
	if err = t.generateGoStubs(functions); err != nil {
		return err
	}
	if err = t.compile(t.Options...); err != nil {
		return err
	}
	// Use architecture-specific parser for assembly translation
	return t.parser.TranslateAssembly(t, functions)
}

type ParameterType struct {
	Type    string
	Pointer bool
}

func (p ParameterType) String() string {
	if p.Pointer {
		return "unsafe.Pointer"
	}
	// Check for NEON vector types first
	if sz := NeonTypeSize(p.Type); sz > 0 {
		// NEON vectors are passed as fixed-size byte arrays in Go
		return fmt.Sprintf("[%d]byte", sz)
	}
	// Check for x86 SIMD vector types
	if sz := X86SIMDTypeSize(p.Type); sz > 0 {
		// x86 SIMD vectors are passed as fixed-size byte arrays in Go
		return fmt.Sprintf("[%d]byte", sz)
	}
	// Check for SVE scalable vector types (size determined at runtime)
	if IsSVEType(p.Type) {
		// SVE vectors must be passed as pointers since size is unknown at compile time
		return "unsafe.Pointer"
	}
	switch p.Type {
	case "_Bool":
		return "bool"
	case "int32_t":
		return "int32"
	case "int64_t", "long":
		return "int64"
	case "uint32_t":
		return "uint32"
	case "uint64_t":
		return "uint64"
	case "double":
		return "float64"
	case "float":
		return "float32"
	case "float16_t":
		return "uint16"
	default:
		_, _ = fmt.Fprintln(os.Stderr, "unsupported param type:", p.Type)
		os.Exit(1)
		return ""
	}
}

type Parameter struct {
	Name string
	ParameterType
}

type Function struct {
	Name       string
	Position   int
	Type       string
	Parameters []Parameter
	StackSize  int
	SpillBase  int // Offset for overflow arg storage; 0 means use StackSize
}

// hasInlineSpecifier walks the DeclarationSpecifiers chain and returns true
// if any node contains an inline function specifier.
func hasInlineSpecifier(ds *cc.DeclarationSpecifiers) bool {
	for ; ds != nil; ds = ds.DeclarationSpecifiers {
		if ds.Case == cc.DeclarationSpecifiersFunc {
			if ds.FunctionSpecifier != nil && ds.FunctionSpecifier.Case == cc.FunctionSpecifierInline {
				return true
			}
		}
	}
	return false
}

// hasStorageClass walks the DeclarationSpecifiers chain and returns true
// if any node contains a storage class specifier (static, extern, etc.).
func hasStorageClass(ds *cc.DeclarationSpecifiers) bool {
	for ; ds != nil; ds = ds.DeclarationSpecifiers {
		if ds.Case == cc.DeclarationSpecifiersStorage {
			return true
		}
	}
	return false
}

// findTypeSpecifier walks the DeclarationSpecifiers chain and returns the
// first TypeSpecifier found, skipping storage class specifiers, type
// qualifiers, function specifiers, and alignment specifiers.
func findTypeSpecifier(ds *cc.DeclarationSpecifiers) (*cc.TypeSpecifier, error) {
	for ; ds != nil; ds = ds.DeclarationSpecifiers {
		if ds.Case == cc.DeclarationSpecifiersTypeSpec {
			return ds.TypeSpecifier, nil
		}
	}
	return nil, fmt.Errorf("no type specifier found in declaration")
}

// convertFunction extracts the function definition from cc.DirectDeclarator.
func (t *TranslateUnit) convertFunction(functionDefinition *cc.FunctionDefinition) (Function, error) {
	// parse return type
	declarationSpecifiers := functionDefinition.DeclarationSpecifiers
	typeSpec, err := findTypeSpecifier(declarationSpecifiers)
	if err != nil {
		return Function{}, fmt.Errorf("invalid function return type: %w", err)
	}
	returnType := typeSpec.Token.SrcStr()
	// parse parameters
	directDeclarator := functionDefinition.Declarator.DirectDeclarator
	if directDeclarator.Case != cc.DirectDeclaratorFuncParam {
		return Function{}, fmt.Errorf("invalid function parameter: %v", directDeclarator.Case)
	}
	params, err := t.convertFunctionParameters(directDeclarator.ParameterTypeList.ParameterList)
	if err != nil {
		return Function{}, err
	}
	return Function{
		Name:       directDeclarator.DirectDeclarator.Token.SrcStr(),
		Position:   directDeclarator.Position().Line,
		Type:       returnType,
		Parameters: params,
	}, nil
}

// convertFunctionParameters extracts function parameters from cc.ParameterList.
func (t *TranslateUnit) convertFunctionParameters(params *cc.ParameterList) ([]Parameter, error) {
	declaration := params.ParameterDeclaration
	paramName := declaration.Declarator.DirectDeclarator.Token.SrcStr()
	typeSpec, err := findTypeSpecifier(declaration.DeclarationSpecifiers)
	if err != nil {
		position := declaration.Position()
		return nil, fmt.Errorf("%v:%v:%v: error: %w",
			position.Filename, position.Line+t.Offset, position.Column, err)
	}
	paramType := typeSpec.Token.SrcStr()
	isPointer := declaration.Declarator.Pointer != nil
	// Accept scalar types, NEON vector types, x86 SIMD types, SVE types, or pointers
	if _, ok := supportedTypes[paramType]; !ok && !IsNeonType(paramType) && !IsX86SIMDType(paramType) && !IsSVEType(paramType) && !isPointer {
		position := declaration.Position()
		return nil, fmt.Errorf("%v:%v:%v: error: unsupported type: %v",
			position.Filename, position.Line+t.Offset, position.Column, paramType)
	}
	paramNames := []Parameter{{
		Name: sanitizeAsmParamName(paramName),
		ParameterType: ParameterType{
			Type:    paramType,
			Pointer: isPointer,
		},
	}}
	if params.ParameterList != nil {
		if nextParamNames, err := t.convertFunctionParameters(params.ParameterList); err != nil {
			return nil, err
		} else {
			paramNames = append(paramNames, nextParamNames...)
		}
	}
	return paramNames, nil
}

// reservedAsmParamNames are names that conflict with Go's plan9 assembler
// pseudo-registers. In ARM64 (and other arches), "g" refers to the goroutine
// pointer, so a parameter named "g" in "g+8(FP)" is misinterpreted as
// register+offset addressing rather than a symbolic frame reference.
var reservedAsmParamNames = map[string]string{
	"g":  "gv",  // goroutine register on ARM64
	"FP": "fp_", // frame pointer pseudo-register
	"SP": "sp_", // stack pointer pseudo-register
	"SB": "sb_", // static base pseudo-register
	"PC": "pc_", // program counter pseudo-register
}

// sanitizeAsmParamName renames C parameter names that conflict with Go plan9
// assembler reserved names. For example, a C function parameter named "g"
// would generate "g+8(FP)" in assembly, which the Go assembler interprets
// as goroutine_register+8(frame_pointer) instead of param_g at offset 8.
func sanitizeAsmParamName(name string) string {
	if replacement, ok := reservedAsmParamNames[name]; ok {
		return replacement
	}
	return name
}

func (t *TranslateUnit) writeHeader(builder *strings.Builder) {
	builder.WriteString("// Code generated by GoAT. DO NOT EDIT.\n")
	builder.WriteString("// versions:\n")
	builder.WriteString(fmt.Sprintf("// 	clang   %s\n", fetchVersion("clang")))
	builder.WriteString(fmt.Sprintf("// 	objdump %s\n", fetchVersion("objdump")))
	builder.WriteString("// flags:")
	for _, option := range t.Options {
		builder.WriteString(" ")
		builder.WriteString(option)
	}
	builder.WriteRune('\n')
	builder.WriteString(fmt.Sprintf("// source: %v\n", t.Source))
	builder.WriteRune('\n')
}

// runCommand runs a command and extract its output.
func runCommand(name string, arg ...string) (string, error) {
	if verbose {
		fmt.Fprintf(os.Stderr, "Running %v\n", append([]string{name}, arg...))
	}
	cmd := exec.Command(name, arg...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", errors.New(string(output))
	}
	return string(output), nil
}

func fetchVersion(command string) string {
	version, err := runCommand(command, "--version")
	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	version = strings.Split(version, "\n")[0]
	idx := strings.IndexFunc(version, unicode.IsDigit)
	if idx < 0 {
		_, _ = fmt.Fprintln(os.Stderr, "failed to fetch version")
		os.Exit(1)
	}
	return version[idx:]
}

func hasPointer(functions []Function) bool {
	for _, function := range functions {
		for _, param := range function.Parameters {
			if param.Pointer {
				return true
			}
		}
	}
	return false
}

var verbose bool

var command = &cobra.Command{
	Use:  "goat source [-o output_directory]",
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		output, _ := cmd.PersistentFlags().GetString("output")
		if output == "" {
			var err error
			if output, err = os.Getwd(); err != nil {
				_, _ = fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		}
		target, _ := cmd.PersistentFlags().GetString("target")
		targetOS, _ := cmd.PersistentFlags().GetString("target-os")

		var options []string
		machineOptions, _ := cmd.PersistentFlags().GetStringSlice("machine-option")
		for _, m := range machineOptions {
			options = append(options, "-m"+m)
		}
		extraOptions, _ := cmd.PersistentFlags().GetStringSlice("extra-option")
		options = append(options, extraOptions...)
		optimizeLevel, _ := cmd.PersistentFlags().GetInt("optimize-level")
		options = append(options, fmt.Sprintf("-O%d", optimizeLevel))
		includePaths, _ := cmd.PersistentFlags().GetStringSlice("include-path")
		sysroot, _ := cmd.PersistentFlags().GetString("sysroot")
		file, err := NewTranslateUnit(args[0], output, target, targetOS, includePaths, sysroot, options...)
		if err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if err := file.Translate(); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	},
}

func init() {
	command.PersistentFlags().StringP("output", "o", "", "output directory of generated files")
	command.PersistentFlags().StringSliceP("machine-option", "m", nil, "machine option for clang")
	command.PersistentFlags().StringSliceP("extra-option", "e", nil, "extra option for clang")
	command.PersistentFlags().IntP("optimize-level", "O", 0, "optimization level for clang")
	command.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "if set, increase verbosity level")
	command.PersistentFlags().StringP("target", "t", runtime.GOARCH, "target architecture (amd64, arm64, loong64, riscv64)")
	command.PersistentFlags().String("target-os", runtime.GOOS, "target operating system (darwin, linux, windows)")
	command.PersistentFlags().StringSliceP("include-path", "I", nil, "additional include path for C parser (for cross-compilation)")
	command.PersistentFlags().String("sysroot", "", "sysroot path for cross-compilation (passed as --sysroot to clang)")
}

func main() {
	if err := command.Execute(); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
