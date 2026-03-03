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

package main

import (
	"fmt"
	"maps"
	"slices"
	"strings"
)

// TargetMode specifies how code should be generated for a target.
type TargetMode int

const (
	TargetModeGoSimd TargetMode = iota // Pure Go simd/archsimd
	TargetModeAsm                      // C -> GOAT -> Go assembly
	TargetModeC                        // C only (inspection)
)

// TargetSpec pairs a Target with its generation mode.
type TargetSpec struct {
	Target Target
	Mode   TargetMode
}

// parseTargetSpec parses a target spec string like "neon:asm" into its name and mode.
func parseTargetSpec(spec string) (name string, mode TargetMode) {
	if idx := strings.Index(spec, ":"); idx > 0 {
		switch spec[idx+1:] {
		case "asm":
			return spec[:idx], TargetModeAsm
		case "c":
			return spec[:idx], TargetModeC
		}
	}
	return spec, TargetModeGoSimd
}

// Target represents an architecture-specific code generation target.
type Target struct {
	Name       string            // "AVX2", "AVX512", "NEON", "Fallback"
	BuildTag   string            // "amd64 && goexperiment.simd", "arm64", "", etc.
	VecWidth   int               // 32 for AVX2, 64 for AVX512, 16 for NEON/fallback
	VecPackage string            // "archsimd" for AVX, "asm" for NEON, "" for fallback
	TypeMap    map[string]string // "float32" -> vector type name (without package prefix)
	OpMap      map[string]OpInfo // "Add" -> operation info
	Mode       TargetMode        // GoSimd, Asm, or C — controls dispatch generation
}

// IsFallback returns true if this is the scalar fallback target.
func (t Target) IsFallback() bool { return t.Name == "Fallback" }

// IsNEON returns true if this is the ARM64 NEON target.
func (t Target) IsNEON() bool { return t.Name == "NEON" }

// IsAVX2 returns true if this is the x86-64 AVX2 target.
func (t Target) IsAVX2() bool { return t.Name == "AVX2" }

// IsAVX512 returns true if this is the x86-64 AVX-512 target.
func (t Target) IsAVX512() bool { return t.Name == "AVX512" }

// IsAVX returns true if this is an AVX target (AVX2 or AVX-512).
func (t Target) IsAVX() bool { return t.IsAVX2() || t.IsAVX512() }

// OpInfo describes how to transform a hwy operation for this target.
type OpInfo struct {
	Package    string // "" for archsimd methods, "hwy" for core package, "math"/"vec" for contrib
	SubPackage string // For contrib: "math", "vec", "matvec", "matmul", "algo", "image", "bitpack", "sort"
	Name       string // Target function/method name
	IsMethod   bool   // true if a.Add(b), false if Add(a, b)
	IsInPlace  bool   // true for in-place ops: v.OpAcc(args, &acc) modifies acc, doesn't return
	AccArg     int    // For in-place ops: which arg index is the accumulator (gets &)
	InPlaceOf  string // Name of the non-in-place op this replaces (e.g., "MulAdd" for MulAddAcc)
}

// ---------------------------------------------------------------------------
// Shared op-map builders
// ---------------------------------------------------------------------------

// contribMathOps returns the contrib/math transcendental function ops shared
// by all targets. Every target maps these identically.
func contribMathOps() map[string]OpInfo {
	return map[string]OpInfo{
		"Exp":     {Package: "math", SubPackage: "math", Name: "BaseExpVec", IsMethod: false},
		"Exp2":    {Package: "math", SubPackage: "math", Name: "BaseExp2Vec", IsMethod: false},
		"Exp10":   {Package: "math", SubPackage: "math", Name: "BaseExp10Vec", IsMethod: false},
		"Log":     {Package: "math", SubPackage: "math", Name: "BaseLogVec", IsMethod: false},
		"Log2":    {Package: "math", SubPackage: "math", Name: "BaseLog2Vec", IsMethod: false},
		"Log10":   {Package: "math", SubPackage: "math", Name: "BaseLog10Vec", IsMethod: false},
		"Sin":     {Package: "math", SubPackage: "math", Name: "BaseSinVec", IsMethod: false},
		"Cos":     {Package: "math", SubPackage: "math", Name: "BaseCosVec", IsMethod: false},
		"SinCos":  {Package: "math", SubPackage: "math", Name: "BaseSinCosVec", IsMethod: false},
		"Tanh":    {Package: "math", SubPackage: "math", Name: "BaseTanhVec", IsMethod: false},
		"Sinh":    {Package: "math", SubPackage: "math", Name: "BaseSinhVec", IsMethod: false},
		"Cosh":    {Package: "math", SubPackage: "math", Name: "BaseCoshVec", IsMethod: false},
		"Asinh":   {Package: "math", SubPackage: "math", Name: "BaseAsinhVec", IsMethod: false},
		"Acosh":   {Package: "math", SubPackage: "math", Name: "BaseAcoshVec", IsMethod: false},
		"Atanh":   {Package: "math", SubPackage: "math", Name: "BaseAtanhVec", IsMethod: false},
		"Sigmoid": {Package: "math", SubPackage: "math", Name: "BaseSigmoidVec", IsMethod: false},
		"Erf":     {Package: "math", SubPackage: "math", Name: "BaseErfVec", IsMethod: false},
		"Pow":     {Package: "math", SubPackage: "math", Name: "BasePowVec", IsMethod: false},
		"Dot":     {Package: "vec", SubPackage: "vec", Name: "Dot", IsMethod: false},
	}
}

// specialOps returns ops that are identical across all SIMD targets (not
// fallback). These are handled by special-case code in the transformer.
func specialOps() map[string]OpInfo {
	return map[string]OpInfo{
		"MaxLanes": {Package: "special", Name: "MaxLanes", IsMethod: false},
		"NumLanes": {Package: "special", Name: "NumLanes", IsMethod: false},
		"Lanes":    {Package: "special", Name: "Lanes", IsMethod: false},
		"Vec":      {Package: "special", Name: "Vec", IsMethod: false},
		"Mask":     {Package: "special", Name: "Mask", IsMethod: false},
		"MaskNot":  {Package: "special", Name: "MaskNot", IsMethod: false},
		"IsInf":    {Package: "special", Name: "IsInf", IsMethod: true},
		"IsNaN":    {Package: "special", Name: "IsNaN", IsMethod: true},
	}
}

// inPlaceOps returns the in-place (allocation-free) ops shared by all ARM
// targets (NEON, SVE_DARWIN, SVE_LINUX).
func inPlaceOps() map[string]OpInfo {
	return map[string]OpInfo{
		"MulAddAcc":  {Name: "MulAddAcc", IsMethod: true, IsInPlace: true, AccArg: 1, InPlaceOf: "MulAdd"},
		"MulAddInto": {Name: "MulAddInto", IsMethod: true, IsInPlace: true, AccArg: 2, InPlaceOf: "MulAdd"},
		"AddInto":    {Name: "AddInto", IsMethod: true, IsInPlace: true, AccArg: 1, InPlaceOf: "Add"},
		"SubInto":    {Name: "SubInto", IsMethod: true, IsInPlace: true, AccArg: 1, InPlaceOf: "Sub"},
		"MulInto":    {Name: "MulInto", IsMethod: true, IsInPlace: true, AccArg: 1, InPlaceOf: "Mul"},
		"DivInto":    {Name: "DivInto", IsMethod: true, IsInPlace: true, AccArg: 1, InPlaceOf: "Div"},
		"MinInto":    {Name: "MinInto", IsMethod: true, IsInPlace: true, AccArg: 1, InPlaceOf: "Min"},
		"MaxInto":    {Name: "MaxInto", IsMethod: true, IsInPlace: true, AccArg: 1, InPlaceOf: "Max"},
	}
}

// armBaseOps returns the base op-map for ARM targets (NEON, SVE_DARWIN,
// SVE_LINUX). The targetName is used for RSqrtNewtonRaphson/RSqrtPrecise
// suffixes, and f32Lanes/f64Lanes set the lane counts for reinterpretation ops.
func armBaseOps(targetName string, f32Lanes, f64Lanes string) map[string]OpInfo {
	m := map[string]OpInfo{
		// Load/Store
		"Load":       {Name: "Load", IsMethod: false},
		"LoadSlice":  {Name: "LoadSlice", IsMethod: false},
		"Load4":      {Name: "Load4", IsMethod: false},
		"Store":      {Name: "Store", IsMethod: true},
		"StoreSlice": {Name: "StoreSlice", IsMethod: true},
		"Set":        {Name: "Broadcast", IsMethod: false},
		"Const":      {Name: "Broadcast", IsMethod: false},
		"Zero":       {Name: "Zero", IsMethod: false},
		"MaskLoad":   {Name: "MaskLoad", IsMethod: false},
		"MaskStore":  {Name: "MaskStore", IsMethod: true},

		// Arithmetic
		"Add": {Name: "Add", IsMethod: true},
		"Sub": {Name: "Sub", IsMethod: true},
		"Mul": {Name: "Mul", IsMethod: true},
		"Div": {Name: "Div", IsMethod: true},
		"Neg": {Name: "Neg", IsMethod: true},
		"Abs": {Name: "Abs", IsMethod: true},
		"Min": {Name: "Min", IsMethod: true},
		"Max": {Name: "Max", IsMethod: true},

		// Logical
		"And":    {Name: "And", IsMethod: true},
		"Or":     {Name: "Or", IsMethod: true},
		"Xor":    {Name: "Xor", IsMethod: true},
		"AndNot": {Name: "AndNot", IsMethod: true},
		"Not":    {Name: "Not", IsMethod: true},

		// Shuffle
		"TableLookupBytes": {Name: "TableLookupBytes", IsMethod: true},

		// Core math
		"Sqrt":               {Name: "Sqrt", IsMethod: true},
		"RSqrt":              {Name: "ReciprocalSqrt", IsMethod: true},
		"RSqrtNewtonRaphson": {Package: "hwy", Name: "RSqrtNewtonRaphson_" + targetName, IsMethod: false},
		"RSqrtPrecise":       {Package: "hwy", Name: "RSqrtPrecise_" + targetName, IsMethod: false},
		"FMA":                {Name: "MulAdd", IsMethod: true},
		"MulAdd":             {Name: "MulAdd", IsMethod: true},
		"Pow":                {Name: "Pow", IsMethod: true},

		// Rounding
		"RoundToEven": {Name: "RoundToEven", IsMethod: true},

		// Type reinterpretation (lane suffixes vary by target width)
		"AsInt32":   {Name: "AsInt32x" + f32Lanes, IsMethod: true},
		"AsFloat32": {Name: "AsFloat32x" + f32Lanes, IsMethod: true},
		"AsInt64":   {Name: "AsInt64x" + f64Lanes, IsMethod: true},
		"AsFloat64": {Name: "AsFloat64x" + f64Lanes, IsMethod: true},

		// Comparisons (return masks)
		"Greater": {Name: "Greater", IsMethod: true},
		"Less":    {Name: "Less", IsMethod: true},

		// Mask ops
		"MaskAnd":    {Name: "And", IsMethod: true},
		"MaskOr":     {Name: "Or", IsMethod: true},
		"MaskAndNot": {Name: "MaskAndNot", IsMethod: false},

		// Conditional/Blend
		"Merge": {Name: "Merge", IsMethod: true},

		// Integer shifts
		"ShiftAllLeft":  {Name: "ShiftAllLeft", IsMethod: true},
		"ShiftAllRight": {Name: "ShiftAllRight", IsMethod: true},
		"ShiftLeft":     {Name: "ShiftAllLeft", IsMethod: true},
		"ShiftRight":    {Name: "ShiftAllRight", IsMethod: true},

		// Reductions
		"ReduceSum": {Name: "ReduceSum", IsMethod: true},
		"ReduceMin": {Name: "ReduceMin", IsMethod: true},
		"ReduceMax": {Name: "ReduceMax", IsMethod: true},

		// Bit manipulation
		"PopCount": {Package: "hwy", Name: "PopCount", IsMethod: false},

		// Comparisons
		"Equal":        {Name: "Equal", IsMethod: true},
		"NotEqual":     {Name: "NotEqual", IsMethod: true},
		"LessThan":     {Name: "LessThan", IsMethod: true},
		"GreaterThan":  {Name: "GreaterThan", IsMethod: true},
		"LessEqual":    {Name: "LessEqual", IsMethod: true},
		"GreaterEqual": {Name: "GreaterEqual", IsMethod: true},

		// Conditional
		"IfThenElse": {Name: "IfThenElse", IsMethod: false},

		// Initialization
		"Iota":    {Name: "Iota", IsMethod: false},
		"SignBit": {Name: "SignBit", IsMethod: false},

		// Permutation/Shuffle
		"Reverse":            {Name: "Reverse", IsMethod: true},
		"Reverse2":           {Name: "Reverse2", IsMethod: false},
		"Reverse4":           {Name: "Reverse4", IsMethod: false},
		"Broadcast":          {Name: "Broadcast", IsMethod: true},
		"GetLane":            {Name: "Get", IsMethod: true},
		"InsertLane":         {Name: "InsertLane", IsMethod: false},
		"InterleaveLower":    {Package: "hwy", Name: "InterleaveLower", IsMethod: false},
		"InterleaveUpper":    {Package: "hwy", Name: "InterleaveUpper", IsMethod: false},
		"ConcatLowerLower":   {Name: "ConcatLowerLower", IsMethod: false},
		"ConcatUpperUpper":   {Name: "ConcatUpperUpper", IsMethod: false},
		"ConcatLowerUpper":   {Name: "ConcatLowerUpper", IsMethod: false},
		"ConcatUpperLower":   {Name: "ConcatUpperLower", IsMethod: false},
		"OddEven":            {Name: "OddEven", IsMethod: false},
		"DupEven":            {Name: "DupEven", IsMethod: false},
		"DupOdd":             {Name: "DupOdd", IsMethod: false},
		"SwapAdjacentBlocks": {Name: "SwapAdjacentBlocks", IsMethod: false},
		"SlideUpLanes":       {Package: "asm", Name: "SlideUpLanes", IsMethod: false},
		"SlideDownLanes":     {Package: "asm", Name: "SlideDownLanes", IsMethod: false},

		// Type Conversions
		"ConvertToInt32":   {Name: "ConvertToInt32", IsMethod: true},
		"ConvertToFloat32": {Name: "ConvertToFloat32", IsMethod: true},
		"Round":            {Name: "Round", IsMethod: false},
		"Trunc":            {Name: "Trunc", IsMethod: false},
		"Ceil":             {Name: "Ceil", IsMethod: false},
		"Floor":            {Name: "Floor", IsMethod: false},
		"NearestInt":       {Name: "NearestInt", IsMethod: false},
		"Clamp":            {Name: "Clamp", IsMethod: false},

		// Compress/Expand
		"Compress":      {Name: "Compress", IsMethod: false},
		"Expand":        {Name: "Expand", IsMethod: false},
		"CompressStore": {Name: "CompressStore", IsMethod: false},
		"CountTrue":     {Name: "CountTrue", IsMethod: false},
		"AllTrue":       {Name: "AllTrue", IsMethod: false},
		"AllFalse":      {Name: "AllFalse", IsMethod: false},
		"FindFirstTrue": {Name: "FindFirstTrue", IsMethod: false},
		"FindLastTrue":  {Name: "FindLastTrue", IsMethod: false},
		"FirstN":        {Name: "FirstN", IsMethod: false},
		"LastN":         {Name: "LastN", IsMethod: false},
		"BitsFromMask":  {Package: "hwy", Name: "BitsFromMask", IsMethod: false},

		// IEEE 754 Exponent/Mantissa
		"GetExponent": {Name: "GetExponent", IsMethod: true},
		"GetMantissa": {Name: "GetMantissa", IsMethod: true},

		// IEEE 754 Operations
		"Pow2": {Name: "Pow2", IsMethod: true},
	}

	maps.Copy(m, contribMathOps())
	maps.Copy(m, specialOps())
	maps.Copy(m, inPlaceOps())
	return m
}

// avxBaseOps returns the base op-map for AVX targets (AVX2, AVX512). The
// targetName is used for RSqrtNewtonRaphson/RSqrtPrecise suffixes, and
// f32Lanes/f64Lanes set the lane counts for reinterpretation ops.
func avxBaseOps(targetName string, f32Lanes, f64Lanes string) map[string]OpInfo {
	m := map[string]OpInfo{
		// Load/Store
		"Load":       {Name: "Load", IsMethod: false},
		"LoadSlice":  {Name: "LoadSlice", IsMethod: false},
		"Load4":      {Package: "hwy", Name: "Load4", IsMethod: false},
		"Store":      {Name: "Store", IsMethod: true},
		"StoreSlice": {Name: "StoreSlice", IsMethod: true},
		"Set":        {Name: "Broadcast", IsMethod: false},
		"Const":      {Name: "Broadcast", IsMethod: false},
		"Zero":       {Package: "special", Name: "Zero", IsMethod: false},
		"MaskLoad":   {Name: "MaskLoad", IsMethod: false},
		"MaskStore":  {Name: "MaskStore", IsMethod: true},

		// Arithmetic
		"Add": {Name: "Add", IsMethod: true},
		"Sub": {Name: "Sub", IsMethod: true},
		"Mul": {Name: "Mul", IsMethod: true},
		"Div": {Name: "Div", IsMethod: true},
		"Neg": {Name: "Neg", IsMethod: true},
		"Abs": {Package: "special", Name: "Abs", IsMethod: true},
		"Min": {Name: "Min", IsMethod: true},
		"Max": {Name: "Max", IsMethod: true},

		// Logical (float types handled specially by transformer using hwy wrappers)
		"And":    {Name: "And", IsMethod: true},
		"Or":     {Name: "Or", IsMethod: true},
		"Xor":    {Name: "Xor", IsMethod: true},
		"AndNot": {Name: "AndNot", IsMethod: true},
		"Not":    {Name: "Not", IsMethod: true},

		// Shuffle
		"TableLookupBytes": {Package: "hwy", Name: "TableLookupBytes", IsMethod: false},

		// Core math
		"Sqrt":               {Name: "Sqrt", IsMethod: true},
		"RSqrt":              {Name: "ReciprocalSqrt", IsMethod: true},
		"RSqrtNewtonRaphson": {Package: "hwy", Name: "RSqrtNewtonRaphson_" + targetName, IsMethod: false},
		"RSqrtPrecise":       {Package: "hwy", Name: "RSqrtPrecise_" + targetName, IsMethod: false},
		"FMA":                {Name: "MulAdd", IsMethod: true},
		"MulAdd":             {Name: "MulAdd", IsMethod: true},

		// Float decomposition (handled by special case code in transformer.go)
		"GetExponent": {Package: "special", Name: "GetExponent", IsMethod: true},
		"GetMantissa": {Package: "special", Name: "GetMantissa", IsMethod: true},

		// Type reinterpretation (lane suffixes vary by target width)
		"AsInt32":   {Name: "AsInt32x" + f32Lanes, IsMethod: true},
		"AsFloat32": {Name: "AsFloat32x" + f32Lanes, IsMethod: true},
		"AsInt64":   {Name: "AsInt64x" + f64Lanes, IsMethod: true},
		"AsFloat64": {Name: "AsFloat64x" + f64Lanes, IsMethod: true},

		// Comparisons (return masks)
		"Greater": {Name: "Greater", IsMethod: true},
		"Less":    {Name: "Less", IsMethod: true},

		// Mask ops
		"MaskAnd":    {Name: "And", IsMethod: true},
		"MaskOr":     {Name: "Or", IsMethod: true},
		"MaskAndNot": {Package: "hwy", Name: "MaskAndNot", IsMethod: false},

		// Conditional/Blend
		"Merge": {Name: "Merge", IsMethod: true},

		// Integer shifts
		"ShiftAllLeft":  {Name: "ShiftAllLeft", IsMethod: true},
		"ShiftAllRight": {Name: "ShiftAllRight", IsMethod: true},
		"ShiftLeft":     {Name: "ShiftAllLeft", IsMethod: true},
		"ShiftRight":    {Name: "ShiftAllRight", IsMethod: true},

		// Reductions (archsimd lacks ReduceSum/Min/Max — using hwy wrappers)
		"ReduceSum": {Package: "hwy", Name: "ReduceSum", IsMethod: false},
		"ReduceMin": {Package: "hwy", Name: "ReduceMin", IsMethod: false},
		"ReduceMax": {Package: "hwy", Name: "ReduceMax", IsMethod: false},

		// Bit manipulation (archsimd lacks PopCount — using hwy wrapper)
		"PopCount": {Package: "hwy", Name: "PopCount", IsMethod: false},

		// Comparisons (archsimd uses Less/Greater, not LessThan/GreaterThan)
		"Equal":        {Name: "Equal", IsMethod: true},
		"NotEqual":     {Name: "NotEqual", IsMethod: true},
		"LessThan":     {Name: "Less", IsMethod: true},
		"GreaterThan":  {Name: "Greater", IsMethod: true},
		"LessEqual":    {Name: "LessEqual", IsMethod: true},
		"GreaterEqual": {Name: "GreaterEqual", IsMethod: true},

		// Initialization
		"Iota":    {Name: "Iota", IsMethod: false},
		"SignBit": {Package: "hwy", Name: "SignBit", IsMethod: false},

		// Permutation/Shuffle
		"Reverse":            {Name: "Reverse", IsMethod: true},
		"Reverse2":           {Name: "Reverse2", IsMethod: false},
		"Reverse4":           {Name: "Reverse4", IsMethod: false},
		"Reverse8":           {Name: "Reverse8", IsMethod: false},
		"Broadcast":          {Name: "Broadcast", IsMethod: true},
		"GetLane":            {Package: "hwy", Name: "GetLane", IsMethod: false},
		"InsertLane":         {Name: "InsertLane", IsMethod: false},
		"InterleaveLower":    {Package: "hwy", Name: "InterleaveLower", IsMethod: false},
		"InterleaveUpper":    {Package: "hwy", Name: "InterleaveUpper", IsMethod: false},
		"ConcatLowerLower":   {Name: "ConcatLowerLower", IsMethod: false},
		"ConcatUpperUpper":   {Name: "ConcatUpperUpper", IsMethod: false},
		"ConcatLowerUpper":   {Name: "ConcatLowerUpper", IsMethod: false},
		"ConcatUpperLower":   {Name: "ConcatUpperLower", IsMethod: false},
		"OddEven":            {Name: "OddEven", IsMethod: false},
		"DupEven":            {Name: "DupEven", IsMethod: false},
		"DupOdd":             {Name: "DupOdd", IsMethod: false},
		"SwapAdjacentBlocks": {Name: "SwapAdjacentBlocks", IsMethod: false},
		"SlideUpLanes":       {Package: "hwy", Name: "SlideUpLanes", IsMethod: false},
		"SlideDownLanes":     {Package: "hwy", Name: "SlideDownLanes", IsMethod: false},

		// Type Conversions
		"ConvertToInt32":   {Name: "ConvertToInt32", IsMethod: true},
		"ConvertToFloat32": {Name: "ConvertToFloat32", IsMethod: true},
		"Round":            {Package: "hwy", Name: "Round", IsMethod: false},
		"Trunc":            {Package: "hwy", Name: "Trunc", IsMethod: false},
		"Ceil":             {Package: "hwy", Name: "Ceil", IsMethod: false},
		"Floor":            {Package: "hwy", Name: "Floor", IsMethod: false},
		"NearestInt":       {Name: "NearestInt", IsMethod: false},
		"Clamp":            {Name: "Clamp", IsMethod: false},

		// Compress/Expand (archsimd lacks these — using hwy wrappers)
		"Compress":      {Package: "hwy", Name: "Compress", IsMethod: false},
		"CompressStore": {Package: "hwy", Name: "CompressStore", IsMethod: false},
		"CountTrue":     {Package: "hwy", Name: "CountTrue", IsMethod: false},
		"FirstN":        {Package: "hwy", Name: "FirstN", IsMethod: false},

		// Conditional (archsimd lacks IfThenElse — using hwy wrapper)
		"IfThenElse": {Package: "hwy", Name: "IfThenElse", IsMethod: false},

		// Mask operations (archsimd Mask types lack these — using hwy wrappers)
		"AllTrue":       {Package: "hwy", Name: "AllTrue", IsMethod: false},
		"AllFalse":      {Package: "hwy", Name: "AllFalse", IsMethod: false},
		"FindFirstTrue": {Package: "hwy", Name: "FindFirstTrue", IsMethod: false},
		"FindLastTrue":  {Package: "hwy", Name: "FindLastTrue", IsMethod: false},
		"LastN":         {Package: "hwy", Name: "LastN", IsMethod: false},
		"Expand":        {Package: "hwy", Name: "Expand", IsMethod: false},
		"BitsFromMask":  {Package: "hwy", Name: "BitsFromMask", IsMethod: false},

		// IEEE 754 Operations
		"Pow2": {Package: "hwy", Name: "Pow2", IsMethod: false},
	}

	maps.Copy(m, contribMathOps())
	maps.Copy(m, specialOps())
	return m
}

// ---------------------------------------------------------------------------
// Target constructors
// ---------------------------------------------------------------------------

// AVX2Target returns the target configuration for AVX2 (256-bit SIMD).
func AVX2Target() Target {
	ops := avxBaseOps("AVX2", "8", "4")

	// AVX2: RoundToEven is a method on archsimd types
	ops["RoundToEven"] = OpInfo{Name: "RoundToEven", IsMethod: true}

	return Target{
		Name:       "AVX2",
		BuildTag:   "amd64 && goexperiment.simd",
		VecWidth:   32,
		VecPackage: "archsimd",
		TypeMap: map[string]string{
			"float32":      "Float32x8",
			"float64":      "Float64x4",
			"int32":        "Int32x8",
			"int64":        "Int64x4",
			"uint32":       "Uint32x8",
			"uint64":       "Uint64x4",
			"hwy.Float16":  "Float16x8AVX2",
			"hwy.BFloat16": "BFloat16x8AVX2",
		},
		OpMap: ops,
	}
}

// AVX512Target returns the target configuration for AVX-512 (512-bit SIMD).
func AVX512Target() Target {
	ops := avxBaseOps("AVX512", "16", "8")

	// AVX512: archsimd lacks plain RoundToEven — use hwy package function
	ops["RoundToEven"] = OpInfo{Package: "hwy", Name: "RoundToEven", IsMethod: false}

	// AVX512 has additional 64-bit conversion ops
	ops["ConvertToInt64"] = OpInfo{Name: "ConvertToInt64", IsMethod: true}
	ops["ConvertToFloat64"] = OpInfo{Name: "ConvertToFloat64", IsMethod: true}

	return Target{
		Name:       "AVX512",
		BuildTag:   "amd64 && goexperiment.simd",
		VecWidth:   64,
		VecPackage: "archsimd",
		TypeMap: map[string]string{
			"float32":      "Float32x16",
			"float64":      "Float64x8",
			"int32":        "Int32x16",
			"int64":        "Int64x8",
			"uint32":       "Uint32x16",
			"uint64":       "Uint64x8",
			"hwy.Float16":  "Float16x16AVX512",
			"hwy.BFloat16": "BFloat16x16AVX512",
		},
		OpMap: ops,
	}
}

// NEONTarget returns the target configuration for ARM NEON (128-bit SIMD).
// Uses the asm package since simd/archsimd doesn't support NEON yet.
func NEONTarget() Target {
	ops := armBaseOps("NEON", "4", "2")

	return Target{
		Name:       "NEON",
		BuildTag:   "arm64",
		VecWidth:   16,
		VecPackage: "asm",
		TypeMap: map[string]string{
			"float32":      "Float32x4",
			"float64":      "Float64x2",
			"int32":        "Int32x4",
			"int64":        "Int64x2",
			"uint32":       "Uint32x4",
			"uint64":       "Uint64x2",
			"hwy.Float16":  "Float16x8",
			"hwy.BFloat16": "BFloat16x8",
		},
		OpMap: ops,
	}
}

// SVEDarwinTarget returns the target configuration for SVE on macOS (Apple M4+).
// Uses SME streaming mode with fixed SVL=512.
func SVEDarwinTarget() Target {
	ops := armBaseOps("SVE_DARWIN", "16", "8")

	return Target{
		Name:       "SVE_DARWIN",
		BuildTag:   "darwin && arm64",
		VecWidth:   64,
		VecPackage: "asm",
		TypeMap: map[string]string{
			"float32":      "Float32x16",
			"float64":      "Float64x8",
			"int32":        "Int32x16",
			"int64":        "Int64x8",
			"uint32":       "Uint32x16",
			"uint64":       "Uint64x8",
			"hwy.Float16":  "Float16x32",
			"hwy.BFloat16": "BFloat16x32",
		},
		OpMap: ops,
	}
}

// SVELinuxTarget returns the target configuration for SVE on Linux (Graviton 3/4, Neoverse).
// Uses native SVE with dynamic vector length.
func SVELinuxTarget() Target {
	ops := armBaseOps("SVE_LINUX", "16", "8")

	return Target{
		Name:       "SVE_LINUX",
		BuildTag:   "linux && arm64",
		VecWidth:   64,
		VecPackage: "asm",
		TypeMap: map[string]string{
			"float32":      "Float32x16",
			"float64":      "Float64x8",
			"int32":        "Int32x16",
			"int64":        "Int64x8",
			"uint32":       "Uint32x16",
			"uint64":       "Uint64x8",
			"hwy.Float16":  "Float16x32",
			"hwy.BFloat16": "BFloat16x32",
		},
		OpMap: ops,
	}
}

// FallbackTarget returns the target configuration for scalar fallback.
func FallbackTarget() Target {
	ops := map[string]OpInfo{
		// Load/Store — use hwy package
		"Load":       {Package: "hwy", Name: "Load", IsMethod: false},
		"LoadSlice":  {Package: "hwy", Name: "LoadSlice", IsMethod: false},
		"Load4":      {Package: "hwy", Name: "Load4", IsMethod: false},
		"Store":      {Package: "hwy", Name: "Store", IsMethod: false},
		"StoreSlice": {Package: "hwy", Name: "StoreSlice", IsMethod: false},
		"Set":        {Package: "hwy", Name: "Set", IsMethod: false},
		"Zero":       {Package: "hwy", Name: "Zero", IsMethod: false},
		"MaskLoad":   {Package: "hwy", Name: "MaskLoad", IsMethod: false},
		"MaskStore":  {Package: "hwy", Name: "MaskStore", IsMethod: false},

		// Arithmetic
		"Add": {Package: "hwy", Name: "Add", IsMethod: false},
		"Sub": {Package: "hwy", Name: "Sub", IsMethod: false},
		"Mul": {Package: "hwy", Name: "Mul", IsMethod: false},
		"Div": {Package: "hwy", Name: "Div", IsMethod: false},
		"Neg": {Package: "hwy", Name: "Neg", IsMethod: false},
		"Abs": {Package: "hwy", Name: "Abs", IsMethod: false},
		"Min": {Package: "hwy", Name: "Min", IsMethod: false},
		"Max": {Package: "hwy", Name: "Max", IsMethod: false},

		// Logical
		"And":    {Package: "hwy", Name: "And", IsMethod: false},
		"Or":     {Package: "hwy", Name: "Or", IsMethod: false},
		"Xor":    {Package: "hwy", Name: "Xor", IsMethod: false},
		"AndNot": {Package: "hwy", Name: "AndNot", IsMethod: false},
		"Not":    {Package: "hwy", Name: "Not", IsMethod: false},

		// Shuffle
		"TableLookupBytes": {Package: "hwy", Name: "TableLookupBytes", IsMethod: false},

		// Core math
		"Sqrt":               {Package: "hwy", Name: "Sqrt", IsMethod: false},
		"RSqrt":              {Package: "hwy", Name: "RSqrt", IsMethod: false},
		"RSqrtNewtonRaphson": {Package: "hwy", Name: "RSqrtNewtonRaphson", IsMethod: false},
		"RSqrtPrecise":       {Package: "hwy", Name: "RSqrtPrecise", IsMethod: false},
		"FMA":                {Package: "hwy", Name: "FMA", IsMethod: false},
		"MulAdd":             {Package: "hwy", Name: "MulAdd", IsMethod: false},
		"Pow":                {Package: "hwy", Name: "Pow", IsMethod: false},

		// Rounding
		"RoundToEven": {Package: "hwy", Name: "RoundToEven", IsMethod: false},

		// Type reinterpretation
		"AsInt32":   {Package: "hwy", Name: "AsInt32", IsMethod: false},
		"AsFloat32": {Package: "hwy", Name: "AsFloat32", IsMethod: false},
		"AsInt64":   {Package: "hwy", Name: "AsInt64", IsMethod: false},
		"AsFloat64": {Package: "hwy", Name: "AsFloat64", IsMethod: false},

		// Comparisons (return masks)
		"Greater": {Package: "hwy", Name: "Greater", IsMethod: false},
		"Less":    {Package: "hwy", Name: "Less", IsMethod: false},

		// Mask ops
		"MaskAnd":    {Package: "hwy", Name: "MaskAnd", IsMethod: false},
		"MaskOr":     {Package: "hwy", Name: "MaskOr", IsMethod: false},
		"MaskNot":    {Package: "hwy", Name: "MaskNot", IsMethod: false},
		"MaskAndNot": {Package: "hwy", Name: "MaskAndNot", IsMethod: false},

		// Conditional/Blend
		"Merge": {Package: "hwy", Name: "Merge", IsMethod: false},

		// Integer shifts (asm types have ShiftAllLeft/ShiftAllRight methods)
		"ShiftAllLeft":  {Name: "ShiftAllLeft", IsMethod: true},
		"ShiftAllRight": {Name: "ShiftAllRight", IsMethod: true},
		"ShiftLeft":     {Name: "ShiftAllLeft", IsMethod: true},
		"ShiftRight":    {Name: "ShiftAllRight", IsMethod: true},

		// Reductions
		"ReduceSum": {Package: "hwy", Name: "ReduceSum", IsMethod: false},
		"ReduceMin": {Package: "hwy", Name: "ReduceMin", IsMethod: false},
		"ReduceMax": {Package: "hwy", Name: "ReduceMax", IsMethod: false},

		// Bit manipulation
		"PopCount": {Package: "hwy", Name: "PopCount", IsMethod: false},

		// Comparisons
		"Equal":        {Package: "hwy", Name: "Equal", IsMethod: false},
		"NotEqual":     {Package: "hwy", Name: "NotEqual", IsMethod: false},
		"LessThan":     {Package: "hwy", Name: "LessThan", IsMethod: false},
		"GreaterThan":  {Package: "hwy", Name: "GreaterThan", IsMethod: false},
		"LessEqual":    {Package: "hwy", Name: "LessEqual", IsMethod: false},
		"GreaterEqual": {Package: "hwy", Name: "GreaterEqual", IsMethod: false},

		// Conditional
		"IfThenElse": {Package: "hwy", Name: "IfThenElse", IsMethod: false},

		// Initialization
		"Iota":    {Package: "hwy", Name: "Iota", IsMethod: false},
		"SignBit": {Package: "hwy", Name: "SignBit", IsMethod: false},

		// Special / Type references
		"MaxLanes": {Package: "special", Name: "MaxLanes", IsMethod: false},
		"NumLanes": {Package: "special", Name: "NumLanes", IsMethod: false},
		"Lanes":    {Package: "special", Name: "Lanes", IsMethod: false},
		"Vec":      {Package: "special", Name: "Vec", IsMethod: false},
		"Mask":     {Package: "special", Name: "Mask", IsMethod: false},

		// Permutation/Shuffle
		"Reverse":            {Package: "hwy", Name: "Reverse", IsMethod: false},
		"Reverse2":           {Package: "hwy", Name: "Reverse2", IsMethod: false},
		"Reverse4":           {Package: "hwy", Name: "Reverse4", IsMethod: false},
		"Reverse8":           {Package: "hwy", Name: "Reverse8", IsMethod: false},
		"Broadcast":          {Package: "hwy", Name: "Broadcast", IsMethod: false},
		"GetLane":            {Package: "hwy", Name: "GetLane", IsMethod: false},
		"InsertLane":         {Package: "hwy", Name: "InsertLane", IsMethod: false},
		"InterleaveLower":    {Package: "hwy", Name: "InterleaveLower", IsMethod: false},
		"InterleaveUpper":    {Package: "hwy", Name: "InterleaveUpper", IsMethod: false},
		"ConcatLowerLower":   {Package: "hwy", Name: "ConcatLowerLower", IsMethod: false},
		"ConcatUpperUpper":   {Package: "hwy", Name: "ConcatUpperUpper", IsMethod: false},
		"ConcatLowerUpper":   {Package: "hwy", Name: "ConcatLowerUpper", IsMethod: false},
		"ConcatUpperLower":   {Package: "hwy", Name: "ConcatUpperLower", IsMethod: false},
		"OddEven":            {Package: "hwy", Name: "OddEven", IsMethod: false},
		"DupEven":            {Package: "hwy", Name: "DupEven", IsMethod: false},
		"DupOdd":             {Package: "hwy", Name: "DupOdd", IsMethod: false},
		"SwapAdjacentBlocks": {Package: "hwy", Name: "SwapAdjacentBlocks", IsMethod: false},
		"SlideUpLanes":       {Package: "hwy", Name: "SlideUpLanes", IsMethod: false},
		"SlideDownLanes":     {Package: "hwy", Name: "SlideDownLanes", IsMethod: false},

		// Type Conversions
		"ConvertToInt32":   {Package: "hwy", Name: "ConvertToInt32", IsMethod: false},
		"ConvertToInt64":   {Package: "hwy", Name: "ConvertToInt64", IsMethod: false},
		"ConvertToFloat32": {Package: "hwy", Name: "ConvertToFloat32", IsMethod: false},
		"ConvertToFloat64": {Package: "hwy", Name: "ConvertToFloat64", IsMethod: false},
		"Round":            {Package: "hwy", Name: "Round", IsMethod: false},
		"Trunc":            {Package: "hwy", Name: "Trunc", IsMethod: false},
		"Ceil":             {Package: "hwy", Name: "Ceil", IsMethod: false},
		"Floor":            {Package: "hwy", Name: "Floor", IsMethod: false},
		"NearestInt":       {Package: "hwy", Name: "NearestInt", IsMethod: false},
		"Clamp":            {Package: "hwy", Name: "Clamp", IsMethod: false},

		// IEEE 754 Exponent/Mantissa
		"GetExponent": {Package: "hwy", Name: "GetExponent", IsMethod: false},
		"GetMantissa": {Package: "hwy", Name: "GetMantissa", IsMethod: false},

		// Compress/Expand
		"Compress":      {Package: "hwy", Name: "Compress", IsMethod: false},
		"Expand":        {Package: "hwy", Name: "Expand", IsMethod: false},
		"CompressStore": {Package: "hwy", Name: "CompressStore", IsMethod: false},
		"CountTrue":     {Package: "hwy", Name: "CountTrue", IsMethod: false},
		"AllTrue":       {Package: "hwy", Name: "AllTrue", IsMethod: false},
		"AllFalse":      {Package: "hwy", Name: "AllFalse", IsMethod: false},
		"FindFirstTrue": {Package: "hwy", Name: "FindFirstTrue", IsMethod: false},
		"FindLastTrue":  {Package: "hwy", Name: "FindLastTrue", IsMethod: false},
		"FirstN":        {Package: "hwy", Name: "FirstN", IsMethod: false},
		"LastN":         {Package: "hwy", Name: "LastN", IsMethod: false},
		"BitsFromMask":  {Package: "hwy", Name: "BitsFromMask", IsMethod: false},

		// IEEE 754 Operations
		"Pow2": {Package: "hwy", Name: "Pow2", IsMethod: false},

		// Special float checks
		"IsInf": {Package: "hwy", Name: "IsInf", IsMethod: false},
		"IsNaN": {Package: "hwy", Name: "IsNaN", IsMethod: false},
	}

	maps.Copy(ops, contribMathOps())
	// Fallback Pow uses hwy package, not contrib/math — override.
	ops["Pow"] = OpInfo{Package: "hwy", Name: "Pow", IsMethod: false}

	return Target{
		Name:       "Fallback",
		BuildTag:   "",
		VecWidth:   16,
		VecPackage: "",
		TypeMap: map[string]string{
			"hwy.Float16":  "hwy.Vec[hwy.Float16]",
			"hwy.BFloat16": "hwy.Vec[hwy.BFloat16]",
			"float32":      "hwy.Vec[float32]",
			"float64":      "hwy.Vec[float64]",
			"int32":        "hwy.Vec[int32]",
			"int64":        "hwy.Vec[int64]",
			"uint32":       "hwy.Vec[uint32]",
			"uint64":       "hwy.Vec[uint64]",
		},
		OpMap: ops,
	}
}

// ---------------------------------------------------------------------------
// Target registry and lookup
// ---------------------------------------------------------------------------

// targetRegistry maps target names to their constructor functions.
var targetRegistry = map[string]func() Target{
	"avx2":       AVX2Target,
	"avx512":     AVX512Target,
	"neon":       NEONTarget,
	"sve_darwin": SVEDarwinTarget,
	"sve_linux":  SVELinuxTarget,
	"fallback":   FallbackTarget,
}

// AvailableTargets returns a sorted list of valid target names.
func AvailableTargets() []string {
	names := make([]string, 0, len(targetRegistry))
	for name := range targetRegistry {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

// GetTarget returns the target configuration for the given name.
func GetTarget(name string) (Target, error) {
	factory, ok := targetRegistry[name]
	if !ok {
		return Target{}, fmt.Errorf("unknown target: %s (valid: %s)", name, strings.Join(AvailableTargets(), ", "))
	}
	return factory(), nil
}

// Suffix returns the filename suffix for this target (e.g., "_avx2").
func (t Target) Suffix() string {
	switch t.Name {
	case "AVX2":
		return "_avx2"
	case "AVX512":
		return "_avx512"
	case "NEON":
		return "_neon"
	case "SVE_DARWIN":
		return "_sve_darwin"
	case "SVE_LINUX":
		return "_sve_linux"
	case "Fallback":
		return "_fallback"
	default:
		return ""
	}
}

// Arch returns the architecture for this target.
func (t Target) Arch() string {
	switch t.Name {
	case "AVX2", "AVX512":
		return "amd64"
	case "NEON", "SVE_DARWIN", "SVE_LINUX":
		return "arm64"
	default:
		return ""
	}
}

// LanesFor returns the number of lanes for the given element type.
func (t Target) LanesFor(elemType string) int {
	var elemSize int
	switch elemType {
	case "float32", "int32", "uint32":
		elemSize = 4
	case "float64", "int64", "uint64":
		elemSize = 8
	case "int16", "uint16", "hwy.Float16", "hwy.BFloat16", "Float16", "BFloat16":
		// On AVX2/AVX512, half-precision uses promoted float32 storage
		if t.Name == "AVX2" || t.Name == "AVX512" {
			elemSize = 4
		} else {
			elemSize = 2
		}
	case "int8", "uint8":
		elemSize = 1
	default:
		return 1
	}
	return t.VecWidth / elemSize
}
