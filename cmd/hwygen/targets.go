package main

import "fmt"

// Target represents an architecture-specific code generation target.
type Target struct {
	Name     string            // "AVX2", "AVX512", "Fallback"
	BuildTag string            // "amd64 && simd", "", etc.
	VecWidth int               // 32 for AVX2, 64 for AVX512, 16 for fallback
	TypeMap  map[string]string // "float32" -> vector type name
	OpMap    map[string]OpInfo // "Add" -> operation info
}

// OpInfo describes how to transform a hwy operation for this target.
type OpInfo struct {
	Package    string // "" for archsimd methods, "hwy" for core package, "math"/"dot" for contrib
	SubPackage string // For contrib: "math", "dot", "matvec", "algo"
	Name       string // Target function/method name
	IsMethod   bool   // true if a.Add(b), false if Add(a, b)
}

// AVX2Target returns the target configuration for AVX2 (256-bit SIMD).
func AVX2Target() Target {
	return Target{
		Name:     "AVX2",
		BuildTag: "amd64 && goexperiment.simd",
		VecWidth: 32,
		TypeMap: map[string]string{
			"float32": "archsimd.Float32x8",
			"float64": "archsimd.Float64x4",
			"int32":   "archsimd.Int32x8",
			"int64":   "archsimd.Int64x4",
		},
		OpMap: map[string]OpInfo{
			// ===== Load/Store operations =====
			"Load":      {Name: "Load", IsMethod: false},      // archsimd.LoadFloat32x8Slice
			"Store":     {Name: "Store", IsMethod: true},      // v.StoreSlice
			"Set":       {Name: "Broadcast", IsMethod: false}, // archsimd.BroadcastFloat32x8
			"Zero":      {Name: "Zero", IsMethod: false},      // archsimd.ZeroFloat32x8
			"MaskLoad":  {Name: "MaskLoad", IsMethod: false},
			"MaskStore": {Name: "MaskStore", IsMethod: true},

			// ===== Arithmetic operations (methods on vector types) =====
			"Add": {Name: "Add", IsMethod: true},
			"Sub": {Name: "Sub", IsMethod: true},
			"Mul": {Name: "Mul", IsMethod: true},
			"Div": {Name: "Div", IsMethod: true},
			"Neg": {Name: "Neg", IsMethod: true}, // Implemented as 0 - x
			"Abs": {Name: "Abs", IsMethod: true},
			"Min": {Name: "Min", IsMethod: true},
			"Max": {Name: "Max", IsMethod: true},

			// ===== Logical operations =====
			"And":    {Name: "And", IsMethod: true},
			"Or":     {Name: "Or", IsMethod: true},
			"Xor":    {Name: "Xor", IsMethod: true},
			"AndNot": {Name: "AndNot", IsMethod: true},

			// ===== Core math operations (hardware instructions) =====
			"Sqrt": {Name: "Sqrt", IsMethod: true}, // VSQRTPS/VSQRTPD
			"FMA":  {Name: "FMA", IsMethod: true},  // VFMADD*

			// ===== Reductions =====
			"ReduceSum": {Name: "ReduceSum", IsMethod: true},
			"ReduceMin": {Name: "ReduceMin", IsMethod: true},
			"ReduceMax": {Name: "ReduceMax", IsMethod: true},

			// ===== Comparisons =====
			"Equal":       {Name: "Equal", IsMethod: true},
			"LessThan":    {Name: "LessThan", IsMethod: true},
			"GreaterThan": {Name: "GreaterThan", IsMethod: true},
			"LessEqual":   {Name: "LessEqual", IsMethod: true},
			"GreaterEqual": {Name: "GreaterEqual", IsMethod: true},

			// ===== Conditional =====
			"IfThenElse": {Name: "IfThenElse", IsMethod: false},

			// ===== Initialization =====
			"Iota":    {Name: "Iota", IsMethod: false},
			"SignBit": {Name: "SignBit", IsMethod: false},

			// ===== Permutation =====
			"Reverse":   {Name: "Reverse", IsMethod: true},
			"Broadcast": {Name: "Broadcast", IsMethod: true},

			// ===== contrib/math: Transcendental functions =====
			// The transformer adds target and type suffix (e.g., Exp -> Exp_AVX2_F32x8)
			"Exp":     {Package: "math", SubPackage: "math", Name: "Exp", IsMethod: false},
			"Exp2":    {Package: "math", SubPackage: "math", Name: "Exp2", IsMethod: false},
			"Exp10":   {Package: "math", SubPackage: "math", Name: "Exp10", IsMethod: false},
			"Log":     {Package: "math", SubPackage: "math", Name: "Log", IsMethod: false},
			"Log2":    {Package: "math", SubPackage: "math", Name: "Log2", IsMethod: false},
			"Log10":   {Package: "math", SubPackage: "math", Name: "Log10", IsMethod: false},
			"Sin":     {Package: "math", SubPackage: "math", Name: "Sin", IsMethod: false},
			"Cos":     {Package: "math", SubPackage: "math", Name: "Cos", IsMethod: false},
			"SinCos":  {Package: "math", SubPackage: "math", Name: "SinCos", IsMethod: false},
			"Tanh":    {Package: "math", SubPackage: "math", Name: "Tanh", IsMethod: false},
			"Sinh":    {Package: "math", SubPackage: "math", Name: "Sinh", IsMethod: false},
			"Cosh":    {Package: "math", SubPackage: "math", Name: "Cosh", IsMethod: false},
			"Asinh":   {Package: "math", SubPackage: "math", Name: "Asinh", IsMethod: false},
			"Acosh":   {Package: "math", SubPackage: "math", Name: "Acosh", IsMethod: false},
			"Atanh":   {Package: "math", SubPackage: "math", Name: "Atanh", IsMethod: false},
			"Sigmoid": {Package: "math", SubPackage: "math", Name: "Sigmoid", IsMethod: false},
			"Erf":     {Package: "math", SubPackage: "math", Name: "Erf", IsMethod: false},

			// ===== contrib/dot: Dot product operations =====
			"Dot": {Package: "dot", SubPackage: "dot", Name: "Dot", IsMethod: false},
		},
	}
}

// AVX512Target returns the target configuration for AVX-512 (512-bit SIMD).
func AVX512Target() Target {
	return Target{
		Name:     "AVX512",
		BuildTag: "amd64 && goexperiment.simd",
		VecWidth: 64,
		TypeMap: map[string]string{
			"float32": "archsimd.Float32x16",
			"float64": "archsimd.Float64x8",
			"int32":   "archsimd.Int32x16",
			"int64":   "archsimd.Int64x8",
		},
		OpMap: map[string]OpInfo{
			// ===== Load/Store operations =====
			"Load":      {Name: "Load", IsMethod: false},
			"Store":     {Name: "Store", IsMethod: true},
			"Set":       {Name: "Broadcast", IsMethod: false},
			"Zero":      {Name: "Zero", IsMethod: false},
			"MaskLoad":  {Name: "MaskLoad", IsMethod: false},
			"MaskStore": {Name: "MaskStore", IsMethod: true},

			// ===== Arithmetic operations =====
			"Add": {Name: "Add", IsMethod: true},
			"Sub": {Name: "Sub", IsMethod: true},
			"Mul": {Name: "Mul", IsMethod: true},
			"Div": {Name: "Div", IsMethod: true},
			"Neg": {Name: "Neg", IsMethod: true},
			"Abs": {Name: "Abs", IsMethod: true},
			"Min": {Name: "Min", IsMethod: true},
			"Max": {Name: "Max", IsMethod: true},

			// ===== Logical operations =====
			"And":    {Name: "And", IsMethod: true},
			"Or":     {Name: "Or", IsMethod: true},
			"Xor":    {Name: "Xor", IsMethod: true},
			"AndNot": {Name: "AndNot", IsMethod: true},

			// ===== Core math operations =====
			"Sqrt": {Name: "Sqrt", IsMethod: true},
			"FMA":  {Name: "FMA", IsMethod: true},

			// ===== Reductions =====
			"ReduceSum": {Name: "ReduceSum", IsMethod: true},
			"ReduceMin": {Name: "ReduceMin", IsMethod: true},
			"ReduceMax": {Name: "ReduceMax", IsMethod: true},

			// ===== Comparisons =====
			"Equal":        {Name: "Equal", IsMethod: true},
			"LessThan":     {Name: "LessThan", IsMethod: true},
			"GreaterThan":  {Name: "GreaterThan", IsMethod: true},
			"LessEqual":    {Name: "LessEqual", IsMethod: true},
			"GreaterEqual": {Name: "GreaterEqual", IsMethod: true},

			// ===== Conditional =====
			"IfThenElse": {Name: "IfThenElse", IsMethod: false},

			// ===== Initialization =====
			"Iota":    {Name: "Iota", IsMethod: false},
			"SignBit": {Name: "SignBit", IsMethod: false},

			// ===== Permutation =====
			"Reverse":   {Name: "Reverse", IsMethod: true},
			"Broadcast": {Name: "Broadcast", IsMethod: true},

			// ===== contrib/math: Transcendental functions =====
			"Exp":     {Package: "math", SubPackage: "math", Name: "Exp", IsMethod: false},
			"Exp2":    {Package: "math", SubPackage: "math", Name: "Exp2", IsMethod: false},
			"Exp10":   {Package: "math", SubPackage: "math", Name: "Exp10", IsMethod: false},
			"Log":     {Package: "math", SubPackage: "math", Name: "Log", IsMethod: false},
			"Log2":    {Package: "math", SubPackage: "math", Name: "Log2", IsMethod: false},
			"Log10":   {Package: "math", SubPackage: "math", Name: "Log10", IsMethod: false},
			"Sin":     {Package: "math", SubPackage: "math", Name: "Sin", IsMethod: false},
			"Cos":     {Package: "math", SubPackage: "math", Name: "Cos", IsMethod: false},
			"SinCos":  {Package: "math", SubPackage: "math", Name: "SinCos", IsMethod: false},
			"Tanh":    {Package: "math", SubPackage: "math", Name: "Tanh", IsMethod: false},
			"Sinh":    {Package: "math", SubPackage: "math", Name: "Sinh", IsMethod: false},
			"Cosh":    {Package: "math", SubPackage: "math", Name: "Cosh", IsMethod: false},
			"Asinh":   {Package: "math", SubPackage: "math", Name: "Asinh", IsMethod: false},
			"Acosh":   {Package: "math", SubPackage: "math", Name: "Acosh", IsMethod: false},
			"Atanh":   {Package: "math", SubPackage: "math", Name: "Atanh", IsMethod: false},
			"Sigmoid": {Package: "math", SubPackage: "math", Name: "Sigmoid", IsMethod: false},
			"Erf":     {Package: "math", SubPackage: "math", Name: "Erf", IsMethod: false},

			// ===== contrib/dot: Dot product operations =====
			"Dot": {Package: "dot", SubPackage: "dot", Name: "Dot", IsMethod: false},
		},
	}
}

// FallbackTarget returns the target configuration for scalar fallback.
func FallbackTarget() Target {
	return Target{
		Name:     "Fallback",
		BuildTag: "", // No build tag - always available
		VecWidth: 16, // Minimal width for fallback
		TypeMap: map[string]string{
			"float32": "hwy.Vec[float32]",
			"float64": "hwy.Vec[float64]",
			"int32":   "hwy.Vec[int32]",
			"int64":   "hwy.Vec[int64]",
		},
		OpMap: map[string]OpInfo{
			// ===== Load/Store operations - use hwy package =====
			"Load":      {Package: "hwy", Name: "Load", IsMethod: false},
			"Store":     {Package: "hwy", Name: "Store", IsMethod: false},
			"Set":       {Package: "hwy", Name: "Set", IsMethod: false},
			"Zero":      {Package: "hwy", Name: "Zero", IsMethod: false},
			"MaskLoad":  {Package: "hwy", Name: "MaskLoad", IsMethod: false},
			"MaskStore": {Package: "hwy", Name: "MaskStore", IsMethod: false},

			// ===== Arithmetic operations =====
			"Add": {Package: "hwy", Name: "Add", IsMethod: false},
			"Sub": {Package: "hwy", Name: "Sub", IsMethod: false},
			"Mul": {Package: "hwy", Name: "Mul", IsMethod: false},
			"Div": {Package: "hwy", Name: "Div", IsMethod: false},
			"Neg": {Package: "hwy", Name: "Neg", IsMethod: false},
			"Abs": {Package: "hwy", Name: "Abs", IsMethod: false},
			"Min": {Package: "hwy", Name: "Min", IsMethod: false},
			"Max": {Package: "hwy", Name: "Max", IsMethod: false},

			// ===== Logical operations =====
			"And":    {Package: "hwy", Name: "And", IsMethod: false},
			"Or":     {Package: "hwy", Name: "Or", IsMethod: false},
			"Xor":    {Package: "hwy", Name: "Xor", IsMethod: false},
			"AndNot": {Package: "hwy", Name: "AndNot", IsMethod: false},

			// ===== Core math operations =====
			"Sqrt": {Package: "hwy", Name: "Sqrt", IsMethod: false},
			"FMA":  {Package: "hwy", Name: "FMA", IsMethod: false},

			// ===== Reductions =====
			"ReduceSum": {Package: "hwy", Name: "ReduceSum", IsMethod: false},
			"ReduceMin": {Package: "hwy", Name: "ReduceMin", IsMethod: false},
			"ReduceMax": {Package: "hwy", Name: "ReduceMax", IsMethod: false},

			// ===== Comparisons =====
			"Equal":        {Package: "hwy", Name: "Equal", IsMethod: false},
			"LessThan":     {Package: "hwy", Name: "LessThan", IsMethod: false},
			"GreaterThan":  {Package: "hwy", Name: "GreaterThan", IsMethod: false},
			"LessEqual":    {Package: "hwy", Name: "LessEqual", IsMethod: false},
			"GreaterEqual": {Package: "hwy", Name: "GreaterEqual", IsMethod: false},

			// ===== Conditional =====
			"IfThenElse": {Package: "hwy", Name: "IfThenElse", IsMethod: false},

			// ===== Initialization =====
			"Iota":    {Package: "hwy", Name: "Iota", IsMethod: false},
			"SignBit": {Package: "hwy", Name: "SignBit", IsMethod: false},

			// ===== Permutation =====
			"Reverse":   {Package: "hwy", Name: "Reverse", IsMethod: false},
			"Broadcast": {Package: "hwy", Name: "Broadcast", IsMethod: false},

			// ===== contrib/math: Use scalar math package for fallback =====
			// For fallback, these use the standard library math functions
			"Exp":     {Package: "hwy", SubPackage: "math", Name: "Exp", IsMethod: false},
			"Exp2":    {Package: "hwy", SubPackage: "math", Name: "Exp2", IsMethod: false},
			"Exp10":   {Package: "hwy", SubPackage: "math", Name: "Exp10", IsMethod: false},
			"Log":     {Package: "hwy", SubPackage: "math", Name: "Log", IsMethod: false},
			"Log2":    {Package: "hwy", SubPackage: "math", Name: "Log2", IsMethod: false},
			"Log10":   {Package: "hwy", SubPackage: "math", Name: "Log10", IsMethod: false},
			"Sin":     {Package: "hwy", SubPackage: "math", Name: "Sin", IsMethod: false},
			"Cos":     {Package: "hwy", SubPackage: "math", Name: "Cos", IsMethod: false},
			"SinCos":  {Package: "hwy", SubPackage: "math", Name: "SinCos", IsMethod: false},
			"Tanh":    {Package: "hwy", SubPackage: "math", Name: "Tanh", IsMethod: false},
			"Sinh":    {Package: "hwy", SubPackage: "math", Name: "Sinh", IsMethod: false},
			"Cosh":    {Package: "hwy", SubPackage: "math", Name: "Cosh", IsMethod: false},
			"Asinh":   {Package: "hwy", SubPackage: "math", Name: "Asinh", IsMethod: false},
			"Acosh":   {Package: "hwy", SubPackage: "math", Name: "Acosh", IsMethod: false},
			"Atanh":   {Package: "hwy", SubPackage: "math", Name: "Atanh", IsMethod: false},
			"Sigmoid": {Package: "hwy", SubPackage: "math", Name: "Sigmoid", IsMethod: false},
			"Erf":     {Package: "hwy", SubPackage: "math", Name: "Erf", IsMethod: false},

			// ===== contrib/dot: Dot product =====
			"Dot": {Package: "hwy", SubPackage: "dot", Name: "Dot", IsMethod: false},
		},
	}
}

// GetTarget returns the target configuration for the given name.
func GetTarget(name string) (Target, error) {
	switch name {
	case "avx2":
		return AVX2Target(), nil
	case "avx512":
		return AVX512Target(), nil
	case "fallback":
		return FallbackTarget(), nil
	default:
		return Target{}, fmt.Errorf("unknown target: %s (valid: avx2, avx512, fallback)", name)
	}
}

// Suffix returns the filename suffix for this target (e.g., "_avx2").
func (t Target) Suffix() string {
	switch t.Name {
	case "AVX2":
		return "_avx2"
	case "AVX512":
		return "_avx512"
	case "Fallback":
		return "_fallback"
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
	case "int16", "uint16":
		elemSize = 2
	case "int8", "uint8":
		elemSize = 1
	default:
		return 1
	}
	return t.VecWidth / elemSize
}
