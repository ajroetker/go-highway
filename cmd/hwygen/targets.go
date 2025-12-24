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
	Package  string // "" for method calls, "hwy" for fallback package functions
	Name     string // Target function/method name
	IsMethod bool   // true if a.Add(b), false if Add(a, b)
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
			// Load/Store operations
			"Load":      {Package: "", Name: "Load", IsMethod: false},     // archsimd.LoadFloat32x8Slice
			"Store":     {Package: "", Name: "Store", IsMethod: true},     // v.StoreSlice
			"Set":       {Package: "", Name: "Broadcast", IsMethod: false}, // archsimd.BroadcastFloat32x8
			"Zero":      {Package: "", Name: "Zero", IsMethod: false},     // archsimd.ZeroFloat32x8
			"MaskLoad":  {Package: "", Name: "MaskLoad", IsMethod: false},
			"MaskStore": {Package: "", Name: "MaskStore", IsMethod: true},

			// Arithmetic operations (methods on vector types)
			"Add": {Package: "", Name: "Add", IsMethod: true},
			"Sub": {Package: "", Name: "Sub", IsMethod: true},
			"Mul": {Package: "", Name: "Mul", IsMethod: true},
			"Div": {Package: "", Name: "Div", IsMethod: true},
			"Neg": {Package: "", Name: "Neg", IsMethod: true},
			"Abs": {Package: "", Name: "Abs", IsMethod: true},
			"Min": {Package: "", Name: "Min", IsMethod: true},
			"Max": {Package: "", Name: "Max", IsMethod: true},

			// Math operations
			"Sqrt": {Package: "", Name: "Sqrt", IsMethod: true},
			"FMA":  {Package: "", Name: "FMA", IsMethod: true},

			// Reductions
			"ReduceSum": {Package: "", Name: "ReduceSum", IsMethod: true},

			// Comparisons
			"Equal":       {Package: "", Name: "Equal", IsMethod: true},
			"LessThan":    {Package: "", Name: "LessThan", IsMethod: true},
			"GreaterThan": {Package: "", Name: "GreaterThan", IsMethod: true},

			// Conditional
			"IfThenElse": {Package: "", Name: "IfThenElse", IsMethod: false},

			// Contrib math functions (package-level functions that take archsimd types)
			// The transformer will add the target and type suffix (e.g., Exp -> Exp_AVX2_F32x8)
			"Exp":     {Package: "contrib", Name: "Exp", IsMethod: false},
			"Log":     {Package: "contrib", Name: "Log", IsMethod: false},
			"Sin":     {Package: "contrib", Name: "Sin", IsMethod: false},
			"Cos":     {Package: "contrib", Name: "Cos", IsMethod: false},
			"Tanh":    {Package: "contrib", Name: "Tanh", IsMethod: false},
			"Sigmoid": {Package: "contrib", Name: "Sigmoid", IsMethod: false},
			"Erf":     {Package: "contrib", Name: "Erf", IsMethod: false},
		},
	}
}

// AVX512Target returns the target configuration for AVX-512 (512-bit SIMD).
func AVX512Target() Target {
	return Target{
		Name:     "AVX512",
		BuildTag: "amd64 && simd && avx512",
		VecWidth: 64,
		TypeMap: map[string]string{
			"float32": "archsimd.Float32x16",
			"float64": "archsimd.Float64x8",
			"int32":   "archsimd.Int32x16",
			"int64":   "archsimd.Int64x8",
		},
		OpMap: map[string]OpInfo{
			// Load/Store operations
			"Load":      {Package: "", Name: "Load", IsMethod: false},
			"Store":     {Package: "", Name: "Store", IsMethod: true},
			"Set":       {Package: "", Name: "Broadcast", IsMethod: false},
			"Zero":      {Package: "", Name: "Zero", IsMethod: false},
			"MaskLoad":  {Package: "", Name: "MaskLoad", IsMethod: false},
			"MaskStore": {Package: "", Name: "MaskStore", IsMethod: true},

			// Arithmetic operations
			"Add": {Package: "", Name: "Add", IsMethod: true},
			"Sub": {Package: "", Name: "Sub", IsMethod: true},
			"Mul": {Package: "", Name: "Mul", IsMethod: true},
			"Div": {Package: "", Name: "Div", IsMethod: true},
			"Neg": {Package: "", Name: "Neg", IsMethod: true},
			"Abs": {Package: "", Name: "Abs", IsMethod: true},
			"Min": {Package: "", Name: "Min", IsMethod: true},
			"Max": {Package: "", Name: "Max", IsMethod: true},

			// Math operations
			"Sqrt": {Package: "", Name: "Sqrt", IsMethod: true},
			"FMA":  {Package: "", Name: "FMA", IsMethod: true},

			// Reductions
			"ReduceSum": {Package: "", Name: "ReduceSum", IsMethod: true},

			// Comparisons
			"Equal":       {Package: "", Name: "Equal", IsMethod: true},
			"LessThan":    {Package: "", Name: "LessThan", IsMethod: true},
			"GreaterThan": {Package: "", Name: "GreaterThan", IsMethod: true},

			// Conditional
			"IfThenElse": {Package: "", Name: "IfThenElse", IsMethod: false},
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
			// All operations use hwy package functions
			"Load":      {Package: "hwy", Name: "Load", IsMethod: false},
			"Store":     {Package: "hwy", Name: "Store", IsMethod: false},
			"Set":       {Package: "hwy", Name: "Set", IsMethod: false},
			"Zero":      {Package: "hwy", Name: "Zero", IsMethod: false},
			"MaskLoad":  {Package: "hwy", Name: "MaskLoad", IsMethod: false},
			"MaskStore": {Package: "hwy", Name: "MaskStore", IsMethod: false},

			"Add": {Package: "hwy", Name: "Add", IsMethod: false},
			"Sub": {Package: "hwy", Name: "Sub", IsMethod: false},
			"Mul": {Package: "hwy", Name: "Mul", IsMethod: false},
			"Div": {Package: "hwy", Name: "Div", IsMethod: false},
			"Neg": {Package: "hwy", Name: "Neg", IsMethod: false},
			"Abs": {Package: "hwy", Name: "Abs", IsMethod: false},
			"Min": {Package: "hwy", Name: "Min", IsMethod: false},
			"Max": {Package: "hwy", Name: "Max", IsMethod: false},

			"Sqrt": {Package: "hwy", Name: "Sqrt", IsMethod: false},
			"FMA":  {Package: "hwy", Name: "FMA", IsMethod: false},

			"ReduceSum": {Package: "hwy", Name: "ReduceSum", IsMethod: false},

			"Equal":       {Package: "hwy", Name: "Equal", IsMethod: false},
			"LessThan":    {Package: "hwy", Name: "LessThan", IsMethod: false},
			"GreaterThan": {Package: "hwy", Name: "GreaterThan", IsMethod: false},

			"IfThenElse": {Package: "hwy", Name: "IfThenElse", IsMethod: false},

			// Contrib math functions (use hwy.Vec wrappers for fallback)
			"Exp":     {Package: "contrib", Name: "Exp", IsMethod: false},
			"Log":     {Package: "contrib", Name: "Log", IsMethod: false},
			"Sin":     {Package: "contrib", Name: "Sin", IsMethod: false},
			"Cos":     {Package: "contrib", Name: "Cos", IsMethod: false},
			"Tanh":    {Package: "contrib", Name: "Tanh", IsMethod: false},
			"Sigmoid": {Package: "contrib", Name: "Sigmoid", IsMethod: false},
			"Erf":     {Package: "contrib", Name: "Erf", IsMethod: false},
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
