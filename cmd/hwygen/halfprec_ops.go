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
	"go/ast"
	"strconv"
)

// transformHalfPrecMerge handles Merge and IfThenElse for half-precision types.
// It handles three tiers: NEON asm types, AVX promoted archsimd types, and
// fallback hwy.IfThenElseF16/BF16 functions.
//
// setHwyFallback is called to set the call target to hwy.<name> when falling
// back to the generic hwy package function (neither NEON nor AVX promoted).
//
// Returns true if the operation was handled.
func transformHalfPrecMerge(call *ast.CallExpr, funcName string, ctx *transformContext, setHwyFallback func(string)) bool {
	if funcName != "Merge" && funcName != "IfThenElse" {
		return false
	}
	if len(call.Args) < 3 {
		return false
	}

	// For NEON target, convert Merge/IfThenElse to asm.IfThenElseFloat16/BFloat16
	if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON {
		if funcName == "Merge" {
			asmFunc := "IfThenElseFloat16"
			if isBFloat16Type(ctx.elemType) {
				asmFunc = "IfThenElseBFloat16"
			}
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("asm"),
				Sel: ast.NewIdent(asmFunc),
			}
			// Reorder: (yes, no, mask) -> (mask, yes, no)
			call.Args = []ast.Expr{call.Args[2], call.Args[0], call.Args[1]}
			return true
		}
		if funcName == "IfThenElse" {
			asmFunc := "IfThenElseFloat16"
			if isBFloat16Type(ctx.elemType) {
				asmFunc = "IfThenElseBFloat16"
			}
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("asm"),
				Sel: ast.NewIdent(asmFunc),
			}
			return true
		}
	}

	// For AVX promoted types, convert Merge/IfThenElse to method calls
	if ctx.isAVXPromoted {
		if funcName == "Merge" {
			// hwy.Merge(yes, no, mask) -> yes.Merge(no, mask)
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent("Merge"),
			}
			call.Args = []ast.Expr{call.Args[1], call.Args[2]}
			return true
		}
		if funcName == "IfThenElse" {
			// hwy.IfThenElse(mask, yes, no) -> yes.Merge(no, mask)
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[1],
				Sel: ast.NewIdent("Merge"),
			}
			call.Args = []ast.Expr{call.Args[2], call.Args[0]}
			return true
		}
	}

	// Fallback: use hwy.IfThenElseF16/BF16
	suffix := "F16"
	if isBFloat16Type(ctx.elemType) {
		suffix = "BF16"
	}
	if funcName == "Merge" {
		setHwyFallback("IfThenElse" + suffix)
		// Reorder: (yes, no, mask) -> (mask, yes, no)
		call.Args = []ast.Expr{call.Args[2], call.Args[0], call.Args[1]}
		return true
	}
	if funcName == "IfThenElse" {
		setHwyFallback("IfThenElse" + suffix)
		// Args stay in same order
		return true
	}

	return false
}

// transformHalfPrecF16FuncOps handles operations that have F16/BF16 specific
// function variants (Add, Sub, Mul, Div, etc.) via getHalfPrecisionFuncName.
//
// The checkInt32ForAll parameter controls whether the int32 operand check
// applies to all operations (true, as in transformToMethod) or only comparison
// operations (false, as in transformToFunction).
//
// setHwyFallback sets the call target to hwy.<name>.
//
// Returns true if the operation was handled.
func transformHalfPrecF16FuncOps(call *ast.CallExpr, funcName string, ctx *transformContext, setHwyFallback func(string), checkInt32ForAll bool) bool {
	f16FuncName := getHalfPrecisionFuncName(funcName, ctx.elemType)
	if f16FuncName == "" {
		return false
	}

	// For operations with 2 operands, check if both are int32 - if so, keep generic hwy function.
	// In transformToMethod context, this applies to all ops (arithmetic, comparisons, bitwise).
	// In transformToFunction context, this only applies to comparison ops.
	if len(call.Args) >= 2 {
		shouldCheckInt32 := checkInt32ForAll || isComparisonOp(funcName)
		if shouldCheckInt32 && isInt32Expr(call.Args[0], ctx) && isInt32Expr(call.Args[1], ctx) {
			if ctx.isAVXPromoted {
				// For AVX promoted, int32 variables are archsimd.Int32x8 - use method calls
				convertToMethodCall(call, funcName)
				return true
			}
			// Keep as generic hwy.Add, hwy.Equal, hwy.And, etc. for int32 operands
			setHwyFallback(funcName)
			return true
		}
	}

	// For NEON and AVX promoted targets, use method calls on asm types
	if (ctx.target.IsNEON() && !ctx.skipHalfPrecNEON) || ctx.isAVXPromoted {
		switch funcName {
		case "Add", "Sub", "Mul", "Div", "Min", "Max":
			if len(call.Args) >= 2 {
				convertToMethodCall(call, funcName)
				return true
			}
		case "FMA", "MulAdd":
			if len(call.Args) >= 3 {
				convertToMethodCall(call, "MulAdd")
				return true
			}
		case "Neg", "Abs", "Sqrt":
			if len(call.Args) >= 1 {
				convertToUnaryMethodCall(call, funcName)
				return true
			}
		case "ReduceSum":
			if len(call.Args) >= 1 {
				convertToUnaryMethodCall(call, "ReduceSum")
				return true
			}
		case "ReduceMax", "ReduceMin":
			if len(call.Args) >= 1 {
				convertToUnaryMethodCall(call, funcName)
				return true
			}
		case "GreaterThan", "Greater", "LessThan", "Less",
			"GreaterEqual", "GreaterThanOrEqual", "LessEqual", "LessThanOrEqual",
			"Equal", "NotEqual":
			if len(call.Args) >= 2 {
				methodName := funcName
				if ctx.isAVXPromoted {
					// AVX promoted types use archsimd-style short names
					switch funcName {
					case "GreaterThan":
						methodName = "Greater"
					case "LessThan":
						methodName = "Less"
					case "GreaterThanOrEqual":
						methodName = "GreaterEqual"
					case "LessThanOrEqual":
						methodName = "LessEqual"
					}
				} else {
					// NEON uses long names
					switch funcName {
					case "Greater":
						methodName = "GreaterThan"
					case "Less":
						methodName = "LessThan"
					case "GreaterEqual":
						methodName = "GreaterThanOrEqual"
					case "LessEqual":
						methodName = "LessThanOrEqual"
					}
				}
				convertToMethodCall(call, methodName)
				return true
			}
		}
	}

	// Fallback: transform to hwy.AddF16(a, b), hwy.MulF16(a, b), etc.
	setHwyFallback(f16FuncName)
	// Args stay as-is (already in the correct order for function calls)
	return true
}

// transformHalfPrecRoundConvert handles RoundToEven, ConvertToInt32, and
// ConvertToFloat32 for half-precision types. AVX promoted types use method
// calls; other targets keep as hwy function calls.
//
// Returns true if the operation was handled.
func transformHalfPrecRoundConvert(call *ast.CallExpr, funcName string, ctx *transformContext, setHwyFallback func(string)) bool {
	switch funcName {
	case "RoundToEven", "ConvertToInt32", "ConvertToFloat32":
		// For AVX promoted types, use method calls
		if ctx.isAVXPromoted {
			if len(call.Args) >= 1 {
				convertToUnaryMethodCall(call, funcName)
				return true
			}
		}
		// Keep as hwy function call for Fallback - do NOT convert to method
		setHwyFallback(funcName)
		return true
	}
	return false
}

// transformHalfPrecNot handles hwy.Not for half-precision types.
// NEON/AVX promoted use method calls; others keep as hwy function.
//
// Returns true if the operation was handled.
func transformHalfPrecNot(call *ast.CallExpr, funcName string, ctx *transformContext, setHwyFallback func(string)) bool {
	if funcName != "Not" {
		return false
	}
	if (ctx.target.IsNEON() && !ctx.skipHalfPrecNEON && ctx.isHalfPrec) || ctx.isAVXPromoted {
		if len(call.Args) >= 1 {
			convertToUnaryMethodCall(call, "Not")
			return true
		}
	}
	setHwyFallback(funcName)
	return true
}

// transformHalfPrecBinaryBitwise handles Xor and And for half-precision types.
// NEON/AVX promoted use method calls; others keep as hwy function.
//
// Returns true if the operation was handled.
func transformHalfPrecBinaryBitwise(call *ast.CallExpr, funcName string, ctx *transformContext, setHwyFallback func(string)) bool {
	if funcName != "Xor" && funcName != "And" {
		return false
	}
	if (ctx.target.IsNEON() && !ctx.skipHalfPrecNEON && ctx.isHalfPrec) || ctx.isAVXPromoted {
		if len(call.Args) >= 2 {
			convertToMethodCall(call, funcName)
			return true
		}
	}
	setHwyFallback(funcName)
	return true
}

// transformHalfPrecOrAndNot handles Or and AndNot for half-precision types.
// AVX promoted uses method calls; others keep as hwy function.
//
// Returns true if the operation was handled.
func transformHalfPrecOrAndNot(call *ast.CallExpr, funcName string, ctx *transformContext, setHwyFallback func(string)) bool {
	if funcName != "Or" && funcName != "AndNot" {
		return false
	}
	if ctx.isAVXPromoted {
		if len(call.Args) >= 2 {
			convertToMethodCall(call, funcName)
			return true
		}
	}
	setHwyFallback(funcName)
	return true
}

// transformHalfPrecNotEqual handles NotEqual for half-precision types.
// AVX promoted uses method calls; others keep as hwy function.
//
// Returns true if the operation was handled.
func transformHalfPrecNotEqual(call *ast.CallExpr, funcName string, ctx *transformContext, setHwyFallback func(string)) bool {
	if funcName != "NotEqual" {
		return false
	}
	if ctx.isAVXPromoted {
		if len(call.Args) >= 2 {
			convertToMethodCall(call, "NotEqual")
			return true
		}
	}
	setHwyFallback("NotEqual")
	return true
}

// transformHalfPrecPow handles Pow for half-precision types.
// AVX promoted inlines scalar Pow via float32 buffers; others keep as hwy.Pow.
//
// Returns true if the operation was handled.
func transformHalfPrecPow(call *ast.CallExpr, funcName string, ctx *transformContext, setHwyFallback func(string)) bool {
	if funcName != "Pow" {
		return false
	}
	if ctx.isAVXPromoted && len(call.Args) >= 2 {
		// For AVX promoted: inline scalar Pow via float32 buffers
		lanes := ctx.target.LanesFor("float32")
		asmType := ctx.target.TypeMap[ctx.elemType]
		wrapFunc := fmt.Sprintf("%sFromFloat32x%d", asmType, lanes)
		loadFunc := fmt.Sprintf("LoadFloat32x%dSlice", lanes)
		asF32Method := fmt.Sprintf("AsFloat32x%d", lanes)
		vecPkg := getVecPackageName(ctx.target)
		lanesStr := strconv.Itoa(lanes)
		baseArg := call.Args[0]
		expArg := call.Args[1]

		call.Fun = genPowIIFE(asmType, wrapFunc, loadFunc, asF32Method, vecPkg, lanesStr, baseArg, expArg)
		call.Args = nil
		return true
	}
	// Keep as hwy.Pow(base, exp)
	setHwyFallback("Pow")
	return true
}

// transformHalfPrecMaskOps handles MaskAnd, MaskOr, MaskXor, MaskAndNot for
// half-precision types in the transformToMethod context.
// AVX promoted uses method calls with Mask prefix stripped; others keep as hwy function.
//
// Returns true if the operation was handled.
func transformHalfPrecMaskOps(call *ast.CallExpr, funcName string, ctx *transformContext, setHwyFallback func(string)) bool {
	switch funcName {
	case "MaskAnd", "MaskOr", "MaskXor", "MaskAndNot":
		// handled below
	default:
		return false
	}
	if ctx.isAVXPromoted && len(call.Args) >= 2 {
		methodName := funcName[4:] // Strip "Mask" prefix: MaskAnd -> And
		convertToMethodCall(call, methodName)
		return true
	}
	setHwyFallback(funcName)
	return true
}

// transformHalfPrecMaskOpsFunc handles MaskAnd and MaskOr for half-precision
// types in the transformToFunction context. (MaskXor/MaskAndNot are only
// handled in transformToMethod.)
//
// Returns true if the operation was handled.
func transformHalfPrecMaskOpsFunc(call *ast.CallExpr, funcName string, ctx *transformContext, setHwyFallback func(string)) bool {
	if funcName != "MaskAnd" && funcName != "MaskOr" {
		return false
	}
	if ctx.isAVXPromoted && len(call.Args) >= 2 {
		methodName := "And"
		if funcName == "MaskOr" {
			methodName = "Or"
		}
		convertToMethodCall(call, methodName)
		return true
	}
	setHwyFallback(funcName)
	return true
}

// transformHalfPrecStoreMethod handles Store for half-precision types in
// the method context (opInfo.IsMethod == true).
// NEON/AVX promoted use StorePtr; fallback keeps as-is.
//
// Returns true if the operation was handled.
func transformHalfPrecStoreMethod(call *ast.CallExpr, funcName string, ctx *transformContext) bool {
	if funcName != "Store" {
		return false
	}
	// For NEON target, convert to method call with unsafe.Pointer
	if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON && len(call.Args) >= 2 {
		vecArg := call.Args[0]
		sliceArg := call.Args[1]
		call.Fun = &ast.SelectorExpr{
			X:   vecArg,
			Sel: ast.NewIdent("StorePtr"),
		}
		call.Args = []ast.Expr{unsafeSlicePointer(sliceArg)}
		return true
	}
	// For AVX promoted types, convert to pointer-based method call: v.StorePtr(ptr)
	if ctx.isAVXPromoted && len(call.Args) >= 2 {
		vecArg := call.Args[0]
		sliceArg := call.Args[1]
		call.Fun = &ast.SelectorExpr{
			X:   vecArg,
			Sel: ast.NewIdent("StorePtr"),
		}
		call.Args = []ast.Expr{unsafeSlicePointer(sliceArg)}
		return true
	}
	// For Fallback, keep hwy.StoreSlice(v, dst) as-is
	return true
}

// transformHalfPrecStoreFunc handles Store for half-precision types in the
// function context (opInfo.IsMethod == false).
// NEON/AVX promoted use StorePtr; fallback keeps as hwy.Store.
//
// Returns true if the operation was handled.
func transformHalfPrecStoreFunc(call *ast.CallExpr, funcName string, ctx *transformContext, setHwyFallback func(string)) bool {
	if funcName != "Store" {
		return false
	}
	// For NEON target, convert to method call with pointer access
	if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON && len(call.Args) >= 2 {
		vecArg := call.Args[0]
		sliceArg := call.Args[1]
		call.Fun = &ast.SelectorExpr{
			X:   vecArg,
			Sel: ast.NewIdent("StorePtr"),
		}
		call.Args = []ast.Expr{unsafeSlicePointer(sliceArg)}
		return true
	}
	// For AVX promoted types, convert to pointer-based method call: v.StorePtr(ptr)
	if ctx.isAVXPromoted && len(call.Args) >= 2 {
		vecArg := call.Args[0]
		sliceArg := call.Args[1]
		call.Fun = &ast.SelectorExpr{
			X:   vecArg,
			Sel: ast.NewIdent("StorePtr"),
		}
		call.Args = []ast.Expr{unsafeSlicePointer(sliceArg)}
		return true
	}
	setHwyFallback("Store")
	return true
}

// transformHalfPrecStoreSliceFunc handles StoreSlice for half-precision types
// in the function context.
//
// Returns true if the operation was handled.
func transformHalfPrecStoreSliceFunc(call *ast.CallExpr, funcName string, ctx *transformContext, setHwyFallback func(string)) bool {
	if funcName != "StoreSlice" {
		return false
	}
	// For NEON target, convert to slice-based method call
	if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON && len(call.Args) >= 2 {
		call.Fun = &ast.SelectorExpr{
			X:   call.Args[0],
			Sel: ast.NewIdent("StoreSlice"),
		}
		call.Args = []ast.Expr{halfPrecSliceToUint16(call.Args[1])}
		return true
	}
	// For AVX promoted types, convert to slice-based method call: v.StoreSlice(dst)
	if ctx.isAVXPromoted && len(call.Args) >= 2 {
		call.Fun = &ast.SelectorExpr{
			X:   call.Args[0],
			Sel: ast.NewIdent("StoreSlice"),
		}
		call.Args = []ast.Expr{halfPrecSliceToUint16(call.Args[1])}
		return true
	}
	setHwyFallback("StoreSlice")
	return true
}

// transformHalfPrecPow2 handles Pow2 for half-precision types.
// AVX promoted wraps hwy.Pow2_AVX2_F32x8 with asm type conversion;
// others use hwy.Pow2[T] with type parameter.
//
// Returns true if the operation was handled.
func transformHalfPrecPow2(call *ast.CallExpr, funcName string, ctx *transformContext) bool {
	if funcName != "Pow2" {
		return false
	}
	// For AVX promoted types: Pow2 operates on float32 internally
	if ctx.isAVXPromoted && len(call.Args) >= 1 {
		pow2Func := fmt.Sprintf("Pow2_%s_F32x%d", ctx.target.Name, ctx.target.LanesFor("float32"))
		wrapFunc := ctx.target.TypeMap[ctx.elemType] + "FromFloat32x" + fmt.Sprintf("%d", ctx.target.LanesFor("float32"))
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent(wrapFunc),
		}
		call.Args = []ast.Expr{
			&ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent(pow2Func),
				},
				Args: call.Args,
			},
		}
		return true
	}
	// Fallback/NEON: Pow2 needs a type parameter: hwy.Pow2[hwy.Float16](kInt)
	call.Fun = &ast.IndexExpr{
		X: &ast.SelectorExpr{
			X:   ast.NewIdent("hwy"),
			Sel: ast.NewIdent("Pow2"),
		},
		Index: ast.NewIdent(ctx.elemType),
	}
	return true
}

// transformHalfPrecSignBit handles SignBit for half-precision types.
// NEON uses asm.SignBitFloat16x8, AVX promoted uses asm.SignBit<Type>,
// fallback uses hwy.SignBit[T].
//
// Returns true if the operation was handled.
func transformHalfPrecSignBit(call *ast.CallExpr, funcName string, ctx *transformContext) bool {
	if funcName != "SignBit" {
		return false
	}
	// For NEON target with asm types
	if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON {
		signBitFuncName := "SignBitFloat16x8"
		if isBFloat16Type(ctx.elemType) {
			signBitFuncName = "SignBitBFloat16x8"
		}
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent(signBitFuncName),
		}
		call.Args = nil
		return true
	}
	// For AVX promoted types
	if ctx.isAVXPromoted {
		signBitFuncName := "SignBit" + ctx.target.TypeMap[ctx.elemType]
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent(signBitFuncName),
		}
		call.Args = nil
		return true
	}
	// For Fallback, SignBit needs a type parameter: hwy.SignBit[hwy.Float16]()
	call.Fun = &ast.IndexExpr{
		X: &ast.SelectorExpr{
			X:   ast.NewIdent("hwy"),
			Sel: ast.NewIdent("SignBit"),
		},
		Index: ast.NewIdent(ctx.elemType),
	}
	return true
}

// transformHalfPrecZero handles Zero for half-precision types.
// NEON uses asm.ZeroFloat16x8, AVX promoted uses asm.Zero<Type>,
// fallback keeps as-is.
//
// Returns true if the operation was handled.
func transformHalfPrecZero(call *ast.CallExpr, funcName string, ctx *transformContext) bool {
	if funcName != "Zero" {
		return false
	}
	if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON {
		zeroFuncName := "ZeroFloat16x8"
		if isBFloat16Type(ctx.elemType) {
			zeroFuncName = "ZeroBFloat16x8"
		}
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent(zeroFuncName),
		}
		// Remove type parameter args if present (transformToFunction has call.Args = nil)
		call.Args = nil
		return true
	}
	if ctx.isAVXPromoted {
		typeName := ctx.target.TypeMap[ctx.elemType]
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent("Zero" + typeName),
		}
		call.Args = nil
		return true
	}
	// For Fallback targets, keep as-is (caller handles)
	return false
}

// transformHalfPrecSet handles Set for half-precision types with uint16 cast.
// NEON uses asm.BroadcastFloat16x8(uint16(val)),
// AVX promoted uses asm.Broadcast<Type>(uint16(val)),
// fallback keeps as-is.
//
// Returns true if the operation was handled.
func transformHalfPrecSet(call *ast.CallExpr, funcName string, ctx *transformContext) bool {
	if funcName != "Set" {
		return false
	}
	if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON {
		broadcastFuncName := "BroadcastFloat16x8"
		if isBFloat16Type(ctx.elemType) {
			broadcastFuncName = "BroadcastBFloat16x8"
		}
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent(broadcastFuncName),
		}
		// Convert arg to uint16 - hwy.Float16/BFloat16 are uint16 aliases
		if len(call.Args) > 0 {
			call.Args[0] = &ast.CallExpr{
				Fun:  ast.NewIdent("uint16"),
				Args: []ast.Expr{call.Args[0]},
			}
		}
		return true
	}
	if ctx.isAVXPromoted {
		typeName := ctx.target.TypeMap[ctx.elemType]
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent("Broadcast" + typeName),
		}
		if len(call.Args) > 0 {
			call.Args[0] = &ast.CallExpr{
				Fun:  ast.NewIdent("uint16"),
				Args: []ast.Expr{call.Args[0]},
			}
		}
		return true
	}
	// For Fallback targets, keep as-is
	return false
}

// transformHalfPrecConst handles Const for half-precision types with
// float32-to-half conversion wrapping.
// NEON uses asm.BroadcastFloat16x8(wrapConstForHalfPrecBroadcast(...)),
// AVX promoted uses asm.Broadcast<Type>(wrapConstForHalfPrecBroadcast(...)),
// fallback keeps as-is.
//
// Returns true if the operation was handled.
func transformHalfPrecConst(call *ast.CallExpr, funcName string, ctx *transformContext) bool {
	if funcName != "Const" {
		return false
	}
	if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON {
		broadcastFuncName := "BroadcastFloat16x8"
		if isBFloat16Type(ctx.elemType) {
			broadcastFuncName = "BroadcastBFloat16x8"
		}
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent(broadcastFuncName),
		}
		if len(call.Args) > 0 {
			call.Args[0] = wrapConstForHalfPrecBroadcast(call.Args[0], ctx.elemType)
		}
		return true
	}
	if ctx.isAVXPromoted {
		typeName := ctx.target.TypeMap[ctx.elemType]
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent("Broadcast" + typeName),
		}
		if len(call.Args) > 0 {
			call.Args[0] = wrapConstForHalfPrecBroadcast(call.Args[0], ctx.elemType)
		}
		return true
	}
	// For Fallback / skip targets, keep as-is
	return false
}

// transformHalfPrecLoadSlice handles slice-based Load for half-precision types
// (called "Load" in method context, "LoadSlice" in function context).
// NEON uses asm.LoadFloat16x8Slice, AVX promoted uses asm.Load<Type>Slice,
// fallback keeps as-is.
//
// Returns true if the operation was handled.
func transformHalfPrecLoadSlice(call *ast.CallExpr, ctx *transformContext) bool {
	if len(call.Args) < 1 {
		return false
	}
	if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON {
		loadFuncName := "LoadFloat16x8Slice"
		if isBFloat16Type(ctx.elemType) {
			loadFuncName = "LoadBFloat16x8Slice"
		}
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent(loadFuncName),
		}
		call.Args[0] = halfPrecSliceToUint16(call.Args[0])
		return true
	}
	if ctx.isAVXPromoted {
		loadFuncName := "Load" + ctx.target.TypeMap[ctx.elemType] + "Slice"
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent(loadFuncName),
		}
		call.Args[0] = halfPrecSliceToUint16(call.Args[0])
		return true
	}
	// For Fallback, keep as-is
	return false
}

// transformHalfPrecLoadPtr handles pointer-based Load for half-precision types
// (called "Load" in function context where Load uses unsafe pointer).
// NEON uses asm.LoadFloat16x8Ptr, AVX promoted uses asm.Load<Type>Ptr,
// fallback keeps as-is.
//
// Returns true if the operation was handled.
func transformHalfPrecLoadPtr(call *ast.CallExpr, ctx *transformContext, setHwyFallback func(string)) bool {
	if len(call.Args) < 1 {
		return false
	}
	if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON {
		loadFuncName := "LoadFloat16x8Ptr"
		if isBFloat16Type(ctx.elemType) {
			loadFuncName = "LoadBFloat16x8Ptr"
		}
		sliceArg := call.Args[0]
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent(loadFuncName),
		}
		call.Args = []ast.Expr{unsafeSlicePointer(sliceArg)}
		return true
	}
	if ctx.isAVXPromoted {
		loadFuncName := "Load" + ctx.target.TypeMap[ctx.elemType] + "Ptr"
		sliceArg := call.Args[0]
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent(loadFuncName),
		}
		call.Args = []ast.Expr{unsafeSlicePointer(sliceArg)}
		return true
	}
	setHwyFallback("Load")
	return true
}

// transformHalfPrecInterleave handles InterleaveLower and InterleaveUpper
// for half-precision types in the function context.
// NEON/AVX promoted use method calls; fallback keeps as hwy function.
//
// Returns true if the operation was handled.
func transformHalfPrecInterleave(call *ast.CallExpr, funcName string, ctx *transformContext, setHwyFallback func(string)) bool {
	if funcName != "InterleaveLower" && funcName != "InterleaveUpper" {
		return false
	}
	if len(call.Args) >= 2 {
		if (ctx.target.IsNEON() && !ctx.skipHalfPrecNEON && ctx.isHalfPrec) || ctx.isAVXPromoted {
			convertToMethodCall(call, funcName)
			return true
		}
	}
	setHwyFallback(funcName)
	return true
}

// transformHalfPrecLoad4 handles Load4 for half-precision types.
// NEON uses asm.Load4Float16x8, AVX promoted uses asm.Load4<Type>Slice,
// fallback uses hwy.Load4[T].
//
// Returns true if the operation was handled.
func transformHalfPrecLoad4(call *ast.CallExpr, funcName string, ctx *transformContext) bool {
	if funcName != "Load4" {
		return false
	}
	// For NEON target
	if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON && len(call.Args) >= 1 {
		load4Func := "Load4Float16x8"
		if isBFloat16Type(ctx.elemType) {
			load4Func = "Load4BFloat16x8"
		}
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent(load4Func),
		}
		call.Args = []ast.Expr{unsafeSlicePointer(call.Args[0])}
		return true
	}
	// For AVX promoted types
	if ctx.isAVXPromoted && len(call.Args) >= 1 {
		asmType := ctx.target.TypeMap[ctx.elemType]
		load4Func := fmt.Sprintf("Load4%sSlice", asmType)
		call.Fun = &ast.SelectorExpr{
			X:   ast.NewIdent("asm"),
			Sel: ast.NewIdent(load4Func),
		}
		call.Args = []ast.Expr{halfPrecSliceToUint16(call.Args[0])}
		return true
	}
	// Fallback: use generic hwy.Load4 with type param
	call.Fun = &ast.IndexExpr{
		X: &ast.SelectorExpr{
			X:   ast.NewIdent("hwy"),
			Sel: ast.NewIdent("Load4"),
		},
		Index: ast.NewIdent(ctx.elemType),
	}
	return true
}
