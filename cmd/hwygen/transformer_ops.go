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
	"go/token"
	"strconv"
	"strings"
)

// isUnsignedIntType returns true if the element type is an unsigned integer type.
func isUnsignedIntType(elemType string) bool {
	return elemType == "uint8" || elemType == "uint16" || elemType == "uint32" || elemType == "uint64"
}

// is64BitIntType returns true if the element type is a 64-bit integer (signed or unsigned).
func is64BitIntType(elemType string) bool {
	return elemType == "int64" || elemType == "uint64"
}

// returnsVecType checks if any return type contains "Vec" or "Mask".
// Vec-returning functions use hwy.Vec operations which already work for Float16/BFloat16.
func returnsVecType(returns []Param) bool {
	for _, ret := range returns {
		if strings.Contains(ret.Type, "Vec") || strings.Contains(ret.Type, "Mask") {
			return true
		}
	}
	return false
}

// optimizeSliceToPointer converts a slice expression to an optimized address expression.
// For slice expressions like src[i:], it generates &src[i] instead of &src[i:][0].
// This avoids the performance overhead where Go doesn't optimize &slice[i:][0] to &slice[i].
//
// Examples:
//   - src[i:]    -> &src[i]      (optimized)
//   - src[i+n:]  -> &src[i+n]    (optimized)
//   - src        -> &src[0]      (no slice, use index 0)
//   - src[i:j]   -> &src[i:j][0] (has high bound, can't optimize)
func optimizeSliceToPointer(expr ast.Expr) *ast.UnaryExpr {
	// Check if expr is a slice expression like src[i:]
	if sliceExpr, ok := expr.(*ast.SliceExpr); ok {
		// Only optimize src[low:] patterns (no high bound, no max)
		// src[low:high] needs to keep bounds for safety
		if sliceExpr.Low != nil && sliceExpr.High == nil && !sliceExpr.Slice3 {
			// Transform src[low:] to &src[low]
			return &ast.UnaryExpr{
				Op: token.AND,
				X: &ast.IndexExpr{
					X:     sliceExpr.X,
					Index: sliceExpr.Low,
				},
			}
		}
	}

	// Default: use &expr[0]
	return &ast.UnaryExpr{
		Op: token.AND,
		X: &ast.IndexExpr{
			X:     expr,
			Index: &ast.BasicLit{Kind: token.INT, Value: "0"},
		},
	}
}

// unsafeSlicePointer builds the AST for unsafe.Pointer(&slice[idx]),
// using optimizeSliceToPointer to handle sub-slice expressions efficiently.
func unsafeSlicePointer(sliceExpr ast.Expr) *ast.CallExpr {
	return &ast.CallExpr{
		Fun: &ast.SelectorExpr{
			X:   ast.NewIdent("unsafe"),
			Sel: ast.NewIdent("Pointer"),
		},
		Args: []ast.Expr{optimizeSliceToPointer(sliceExpr)},
	}
}

// convertToMethodCall transforms a function call hwy.Op(receiver, args...) into
// receiver.methodName(args...) by making the first argument the receiver.
func convertToMethodCall(call *ast.CallExpr, methodName string) {
	call.Fun = &ast.SelectorExpr{
		X:   call.Args[0],
		Sel: ast.NewIdent(methodName),
	}
	call.Args = call.Args[1:]
}

// convertToUnaryMethodCall transforms hwy.Op(receiver) into receiver.methodName()
// with no remaining arguments.
func convertToUnaryMethodCall(call *ast.CallExpr, methodName string) {
	call.Fun = &ast.SelectorExpr{
		X:   call.Args[0],
		Sel: ast.NewIdent(methodName),
	}
	call.Args = nil
}

// transformMaskNot transforms hwy.MaskNot(mask) into mask.Xor(allTrue),
// where allTrue = Broadcast(1).Equal(Broadcast(1)).
// This pattern is used in both transformToMethod and transformToFunction.
func transformMaskNot(call *ast.CallExpr, pkgName, vecTypeName, elemType string) {
	mask := call.Args[0]

	// Create pkg.Broadcast*(1.0) for float types or 1 for int types
	var oneLit ast.Expr
	if elemType == "float32" || elemType == "float64" {
		oneLit = &ast.BasicLit{Kind: token.FLOAT, Value: "1.0"}
	} else {
		oneLit = &ast.BasicLit{Kind: token.INT, Value: "1"}
	}
	oneCall := &ast.CallExpr{
		Fun: &ast.SelectorExpr{
			X:   ast.NewIdent(pkgName),
			Sel: ast.NewIdent("Broadcast" + vecTypeName),
		},
		Args: []ast.Expr{oneLit},
	}
	// Create one.Equal(one) to get all-true mask
	allTrue := &ast.CallExpr{
		Fun: &ast.SelectorExpr{
			X:   oneCall,
			Sel: ast.NewIdent("Equal"),
		},
		Args: []ast.Expr{cloneExpr(oneCall)},
	}
	// Create mask.Xor(allTrue) to invert
	call.Fun = &ast.SelectorExpr{
		X:   mask,
		Sel: ast.NewIdent("Xor"),
	}
	call.Args = []ast.Expr{allTrue}
}

// hwyWrapperName returns the hwy wrapper function name for a given op, target, and element type.
// This is the naming convention: OpName_TargetName_ShortTypeName (e.g., Compress_AVX2_F32x8).
func hwyWrapperName(opName string, ctx *transformContext) string {
	return fmt.Sprintf("%s_%s_%s", opName, ctx.target.Name, getShortTypeName(ctx.elemType, ctx.target))
}

// arrayPointerCast builds a (*[lanes]elemType)(ptr) type conversion expression.
// This casts an unsafe.Pointer to a fixed-size array pointer for SIMD load/store operations.
func arrayPointerCast(lanes int, elemType string, ptr ast.Expr) *ast.CallExpr {
	return &ast.CallExpr{
		Fun: &ast.ParenExpr{
			X: &ast.StarExpr{
				X: &ast.ArrayType{
					Len: &ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(lanes)},
					Elt: ast.NewIdent(elemType),
				},
			},
		},
		Args: []ast.Expr{ptr},
	}
}

// redirectToHwyWrapper rewrites a call expression to use a hwy wrapper function.
// The wrapper name follows the pattern: FuncName_TargetName_VecTypeName (e.g., GetLane_AVX2_Float32x8).
// Arguments are preserved as-is.
func redirectToHwyWrapper(call *ast.CallExpr, funcName string, ctx *transformContext) {
	vecTypeName := ctx.vecTypeName
	wrapperName := fmt.Sprintf("%s_%s_%s", funcName, ctx.target.Name, vecTypeName)
	call.Fun = &ast.SelectorExpr{
		X:   ast.NewIdent("hwy"),
		Sel: ast.NewIdent(wrapperName),
	}
}

// transformToMethod converts hwy.Add(a, b) to a.Add(b) for SIMD targets.
// For Fallback, keeps hwy.Add(a, b) as-is.
func transformToMethod(call *ast.CallExpr, funcName string, opInfo OpInfo, ctx *transformContext) {
	if len(call.Args) < 1 {
		return
	}

	// For fallback, keep hwy calls as-is (don't convert to method calls)
	if ctx.target.IsFallback() {
		// Just update the package name if needed
		switch fun := call.Fun.(type) {
		case *ast.SelectorExpr:
			fun.X = ast.NewIdent("hwy")
		case *ast.IndexExpr:
			if sel, ok := fun.X.(*ast.SelectorExpr); ok {
				sel.X = ast.NewIdent("hwy")
			}
		}
		return
	}

	// For Float16/BFloat16 on SIMD targets, use hwy package functions instead of methods.
	// archsimd doesn't have native support for half-precision types.
	if ctx.isHalfPrec {
		setFallback := func(name string) {
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent(name),
			}
		}

		if transformHalfPrecMerge(call, funcName, ctx, setFallback) {
			return
		}

		if transformHalfPrecF16FuncOps(call, funcName, ctx, setFallback, true) {
			return
		}

		// For operations that don't have F16/BF16 variants, keep as hwy function calls
		// instead of converting to method calls (which don't exist on hwy.Vec[Float16])
		switch funcName {
		case "RoundToEven", "ConvertToInt32", "ConvertToFloat32":
			if transformHalfPrecRoundConvert(call, funcName, ctx, setFallback) {
				return
			}
		case "Not":
			if transformHalfPrecNot(call, funcName, ctx, setFallback) {
				return
			}
		case "Xor", "And":
			if transformHalfPrecBinaryBitwise(call, funcName, ctx, setFallback) {
				return
			}
		case "Or", "AndNot":
			if transformHalfPrecOrAndNot(call, funcName, ctx, setFallback) {
				return
			}
		case "NotEqual":
			if transformHalfPrecNotEqual(call, funcName, ctx, setFallback) {
				return
			}
		case "Pow":
			if transformHalfPrecPow(call, funcName, ctx, setFallback) {
				return
			}
		case "MaskAnd", "MaskOr", "MaskXor", "MaskAndNot":
			if transformHalfPrecMaskOps(call, funcName, ctx, setFallback) {
				return
			}
		case "Store":
			if transformHalfPrecStoreMethod(call, funcName, ctx) {
				return
			}
		case "Pow2":
			if transformHalfPrecPow2(call, funcName, ctx) {
				return
			}
		case "SignBit":
			if transformHalfPrecSignBit(call, funcName, ctx) {
				return
			}
		case "Set", "Zero", "Const":
			// For Set and Const, the method context uses uint16 cast (not wrapConstForHalfPrecBroadcast)
			if funcName == "Zero" {
				if transformHalfPrecZero(call, funcName, ctx) {
					return
				}
			}
			if funcName == "Set" || funcName == "Const" {
				if transformHalfPrecSet(call, funcName, ctx) {
					return
				}
			}
			// For Fallback targets, keep hwy.Set[T](val), hwy.Zero[T](), and hwy.Const[T](val) as-is
			return
		case "Load":
			if transformHalfPrecLoadSlice(call, ctx) {
				return
			}
			// For Fallback, keep hwy.Load as-is
			return
		}

		// For operations without F16/BF16 variants, fall through to regular handling
		// but this may cause issues if they try to use method calls
	}

	// For AVX2/AVX512, use wrapper functions for ReduceMax (unsigned only) and GetLane (all types).
	// archsimd doesn't have ReduceMax for unsigned types or a direct GetLane method.
	if ctx.target.IsAVX() {
		switch funcName {
		case "ReduceMax":
			if isUnsignedIntType(ctx.elemType) && len(call.Args) >= 1 {
				redirectToHwyWrapper(call, "ReduceMax", ctx)
				return
			}
		case "GetLane":
			if len(call.Args) >= 2 {
				redirectToHwyWrapper(call, "GetLane", ctx)
				return
			}
		}
	}

	// For 64-bit integer types on AVX2, use wrapper functions for Max and Min.
	// AVX2 doesn't have VPMAXSQ/VPMINUQ/VPMAXUQ/VPMINSQ instructions (only AVX-512 has them).
	if is64BitIntType(ctx.elemType) && ctx.target.IsAVX2() {
		switch funcName {
		case "Max":
			if len(call.Args) >= 2 {
				redirectToHwyWrapper(call, "Max", ctx)
				return
			}
		case "Min":
			if len(call.Args) >= 2 {
				redirectToHwyWrapper(call, "Min", ctx)
				return
			}
		}
	}

	// For SIMD targets, convert to method calls on archsimd types
	switch funcName {
	case "StoreSlice":
		// For half-precision types with skipHalfPrecNEON (generic hwy.Vec path),
		// keep as hwy.StoreSlice(v, dst) - don't convert to method call
		if ctx.isHalfPrec && ctx.skipHalfPrecNEON {
			return
		}
		// hwy.StoreSlice(v, dst) -> v.StoreSlice(dst)
		if len(call.Args) >= 2 {
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent("StoreSlice"),
			}
			sliceArg := call.Args[1]
			// For NEON and AVX promoted half-precision: cast []hwy.Float16/[]hwy.BFloat16 -> []uint16
			if ctx.isNEONHalfPrec() || ctx.isAVXPromoted {
				sliceArg = halfPrecSliceToUint16(sliceArg)
			}
			call.Args = []ast.Expr{sliceArg}
		}

	case "Store":
		// For NEON half-precision or AVX promoted half-precision:
		// hwy.StoreSlice(v, dst) -> v.StorePtr(unsafe.Pointer(&dst[0]))
		if ctx.isNEONHalfPrec() || ctx.isAVXPromoted {
			if len(call.Args) >= 2 {
				vecArg := call.Args[0]
				sliceArg := call.Args[1]
				call.Fun = &ast.SelectorExpr{
					X:   vecArg,
					Sel: ast.NewIdent("StorePtr"),
				}
				call.Args = []ast.Expr{unsafeSlicePointer(sliceArg)}
			}
			return
		}
		// Keep hwy.StoreSlice(v, dst) as-is for Fallback half-precision types
		if ctx.isHalfPrec {
			return
		}

		// hwy.StoreSlice(v, dst) -> v.Store((*[8]float32)(unsafe.Pointer(&dst[0])))
		if len(call.Args) >= 2 {
			lanes := ctx.target.LanesFor(ctx.elemType)
			dst := call.Args[1]
			cast := arrayPointerCast(lanes, ctx.elemType, unsafeSlicePointer(dst))

			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent("Store"),
			}
			call.Args = []ast.Expr{cast}
		}

	case "MaskStore":
		// hwy.MaskStore(mask, v, dst) -> v.MaskStoreSlice(mask, dst)
		if len(call.Args) >= 3 {
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[1],
				Sel: ast.NewIdent("MaskStoreSlice"),
			}
			call.Args = []ast.Expr{call.Args[0], call.Args[2]}
		}

	case "Neg":
		// hwy.Neg(x) -> pkg.BroadcastFloat32x8(0).Sub(x) for SIMD
		// (archsimd/asm types don't have a Neg method, so we use 0 - x)
		if len(call.Args) >= 1 {
			vecTypeName := ctx.vecTypeName
			pkgName := ctx.vecPkgName
			// Create pkg.BroadcastFloat32x8(0)
			zeroLit := &ast.BasicLit{Kind: token.INT, Value: "0"}
			zeroCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{zeroLit},
			}
			call.Fun = &ast.SelectorExpr{
				X:   zeroCall,
				Sel: ast.NewIdent("Sub"),
			}
			// Args stays as [x]
		}

	case "Pow2":
		// hwy.Pow2[T](kInt) -> kInt.Pow2Float32() or kInt.Pow2Float64()
		// based on context element type
		if len(call.Args) >= 1 {
			var methodName string
			switch ctx.elemType {
			case "float32":
				methodName = "Pow2Float32"
			case "float64":
				methodName = "Pow2Float64"
			default:
				methodName = "Pow2Float32" // fallback
			}
			convertToUnaryMethodCall(call, methodName)
		}

	case "GetExponent":
		// For Float16/BFloat16 on Fallback, use hwy.GetExponent which has proper handling
		if ctx.isHalfPrec && !ctx.isAVXPromoted {
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent("GetExponent"),
			}
			return
		}
		if len(call.Args) >= 1 {
			x := call.Args[0]
			// For AVX promoted half-precision, the underlying data is float32
			effectiveElem := ctx.elemType
			if ctx.isAVXPromoted {
				effectiveElem = "float32"
			}
			intVecTypeName := getVectorTypeNameForInt("int32", effectiveElem, ctx.target)
			if effectiveElem == "float64" {
				intVecTypeName = getVectorTypeNameForInt("int64", effectiveElem, ctx.target)
			}
			pkgName := ctx.vecPkgName

			// 1. x.AsInt32() / x.AsInt64()
			var asIntMethod string
			var shift int
			var mask string
			var bias string

			if effectiveElem == "float32" {
				asIntMethod = "AsInt32x8"
				// Check targets.go OpMap["AsInt32"].Name
				if op, ok := ctx.target.OpMap["AsInt32"]; ok {
					asIntMethod = op.Name
				}
				shift = 23
				mask = "255" // 0xFF
				bias = "127"
			} else {
				asIntMethod = "AsInt64x4"
				if op, ok := ctx.target.OpMap["AsInt64"]; ok {
					asIntMethod = op.Name
				}
				shift = 52
				mask = "2047" // 0x7FF
				bias = "1023"
			}

			// x.AsInt32()
			expr := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   cloneExpr(x),
					Sel: ast.NewIdent(asIntMethod),
				},
			}

			// .ShiftAllRight(shift)
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("ShiftAllRight"),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(shift)}},
			}

			// .And(Broadcast(mask))
			broadcastMask := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: mask}},
			}
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("And"),
				},
				Args: []ast.Expr{broadcastMask},
			}

			// .Sub(Broadcast(bias))
			broadcastBias := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: bias}},
			}
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("Sub"),
				},
				Args: []ast.Expr{broadcastBias},
			}

			// NOTE: Don't add .ConvertToFloat() here - let ConvertExponentToFloat handle the
			// int-to-float conversion. This keeps GetExponent returning integers as expected.

			*call = *expr
		}

	case "GetMantissa":
		// For non-AVX-promoted half-precision, use hwy.GetMantissa which has proper handling
		if ctx.isHalfPrec && !ctx.isAVXPromoted {
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent("GetMantissa"),
			}
			return
		}
		if len(call.Args) >= 1 {
			x := call.Args[0]
			// For AVX promoted half-precision, the underlying data is float32
			effectiveElem := ctx.elemType
			if ctx.isAVXPromoted {
				effectiveElem = "float32"
			}
			intVecTypeName := getVectorTypeNameForInt("int32", effectiveElem, ctx.target)
			if effectiveElem == "float64" {
				intVecTypeName = getVectorTypeNameForInt("int64", effectiveElem, ctx.target)
			}
			pkgName := ctx.vecPkgName

			var asIntMethod string
			var mask string
			var one string
			var asFloatMethod string

			if effectiveElem == "float32" {
				asIntMethod = "AsInt32x8"
				if op, ok := ctx.target.OpMap["AsInt32"]; ok {
					asIntMethod = op.Name
				}
				mask = "8388607"   // 0x7FFFFF
				one = "1065353216" // 0x3F800000
				asFloatMethod = "AsFloat32x8"
				if op, ok := ctx.target.OpMap["AsFloat32"]; ok {
					asFloatMethod = op.Name
				}
			} else {
				asIntMethod = "AsInt64x4"
				if op, ok := ctx.target.OpMap["AsInt64"]; ok {
					asIntMethod = op.Name
				}
				mask = "4503599627370495"   // 0x000FFFFFFFFFFFFF
				one = "4607182418800017408" // 0x3FF0000000000000
				asFloatMethod = "AsFloat64x4"
				if op, ok := ctx.target.OpMap["AsFloat64"]; ok {
					asFloatMethod = op.Name
				}
			}

			// x.AsInt32()
			expr := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   cloneExpr(x),
					Sel: ast.NewIdent(asIntMethod),
				},
			}

			// .And(Broadcast(mask))
			broadcastMask := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: mask}},
			}
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("And"),
				},
				Args: []ast.Expr{broadcastMask},
			}

			// .Or(Broadcast(one))
			broadcastOne := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + intVecTypeName),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: one}},
			}
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent("Or"),
				},
				Args: []ast.Expr{broadcastOne},
			}

			// .AsFloat32()
			expr = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   expr,
					Sel: ast.NewIdent(asFloatMethod),
				},
			}

			// For AVX promoted half-precision, wrap the Float32x8 result back in the asm type
			if ctx.isAVXPromoted {
				wrapFunc := avxPromotedWrapFromFloat32Func(ctx.target, ctx.elemType)
				expr = &ast.CallExpr{
					Fun: &ast.SelectorExpr{
						X:   ast.NewIdent("asm"),
						Sel: ast.NewIdent(wrapFunc),
					},
					Args: []ast.Expr{expr},
				}
			}

			*call = *expr
		}

	case "Abs":
		// hwy.Abs(x) -> x.Max(negX) where negX = pkg.Broadcast*(0).Sub(x)
		// archsimd doesn't have Abs method, so we implement |x| = max(x, -x)
		if opInfo.Package == "special" && len(call.Args) >= 1 {
			vecTypeName := ctx.vecTypeName
			pkgName := ctx.vecPkgName
			x := call.Args[0]
			// Create pkg.Broadcast*(0)
			zeroLit := &ast.BasicLit{Kind: token.INT, Value: "0"}
			zeroCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{zeroLit},
			}
			// Create pkg.Broadcast*(0).Sub(x) = -x
			negX := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   zeroCall,
					Sel: ast.NewIdent("Sub"),
				},
				Args: []ast.Expr{cloneExpr(x)},
			}
			// Create x.Max(-x)
			call.Fun = &ast.SelectorExpr{
				X:   x,
				Sel: ast.NewIdent("Max"),
			}
			call.Args = []ast.Expr{negX}
		} else {
			// Normal Abs method call
			if len(call.Args) >= 1 {
				convertToUnaryMethodCall(call, opInfo.Name)
			}
		}

	case "IsNaN":
		// hwy.IsNaN(x) -> x.Equal(x).Xor(one.Equal(one))
		// NaN != NaN, so x.Equal(x) is false (all 0s) for NaN elements
		// We XOR with all-true mask to invert, giving true for NaN
		if opInfo.Package == "special" && len(call.Args) >= 1 {
			vecTypeName := ctx.vecTypeName
			pkgName := ctx.vecPkgName
			x := call.Args[0]
			// Create pkg.Broadcast*(1.0)
			oneLit := &ast.BasicLit{Kind: token.FLOAT, Value: "1.0"}
			oneCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{oneLit},
			}
			// Create one.Equal(one) to get all-true mask
			allTrue := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   oneCall,
					Sel: ast.NewIdent("Equal"),
				},
				Args: []ast.Expr{cloneExpr(oneCall)},
			}
			// Create x.Equal(x)
			xEqX := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   x,
					Sel: ast.NewIdent("Equal"),
				},
				Args: []ast.Expr{cloneExpr(x)},
			}
			// Create x.Equal(x).Xor(allTrue) to invert
			call.Fun = &ast.SelectorExpr{
				X:   xEqX,
				Sel: ast.NewIdent("Xor"),
			}
			call.Args = []ast.Expr{allTrue}
		}

	case "MaskNot":
		if opInfo.Package == "special" && len(call.Args) >= 1 {
			transformMaskNot(call, ctx.vecPkgName, ctx.vecTypeName, ctx.elemType)
		}

	case "IsInf":
		// hwy.IsInf(x, sign) -> compare with +Inf and/or -Inf
		// sign=0: either +Inf or -Inf, sign=1: +Inf only, sign=-1: -Inf only
		if opInfo.Package == "special" && len(call.Args) >= 2 {
			vecTypeName := ctx.vecTypeName
			pkgName := ctx.vecPkgName
			x := call.Args[0]
			signArg := call.Args[1]

			// Determine sign value (0, 1, or -1)
			signVal := 0
			if lit, ok := signArg.(*ast.BasicLit); ok && lit.Kind == token.INT {
				if lit.Value == "1" {
					signVal = 1
				} else if lit.Value == "-1" {
					signVal = -1
				}
			} else if unary, ok := signArg.(*ast.UnaryExpr); ok && unary.Op == token.SUB {
				if lit, ok := unary.X.(*ast.BasicLit); ok && lit.Kind == token.INT && lit.Value == "1" {
					signVal = -1
				}
			}

			// Create math.Inf(1) with type conversion for float32
			posInfExpr := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent("stdmath"),
					Sel: ast.NewIdent("Inf"),
				},
				Args: []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "1"}},
			}
			// For float32, wrap in type conversion
			var posInfArg ast.Expr = posInfExpr
			if ctx.elemType == "float32" {
				posInfArg = &ast.CallExpr{
					Fun:  ast.NewIdent("float32"),
					Args: []ast.Expr{posInfExpr},
				}
			}

			// Create pkg.Broadcast*(posInf) for +Inf
			posInfCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{posInfArg},
			}

			// Create math.Inf(-1) with type conversion for float32
			negInfExpr := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent("stdmath"),
					Sel: ast.NewIdent("Inf"),
				},
				Args: []ast.Expr{
					&ast.UnaryExpr{
						Op: token.SUB,
						X:  &ast.BasicLit{Kind: token.INT, Value: "1"},
					},
				},
			}
			// For float32, wrap in type conversion
			var negInfArg ast.Expr = negInfExpr
			if ctx.elemType == "float32" {
				negInfArg = &ast.CallExpr{
					Fun:  ast.NewIdent("float32"),
					Args: []ast.Expr{negInfExpr},
				}
			}

			// Create pkg.Broadcast*(negInf) for -Inf
			negInfCall := &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent(pkgName),
					Sel: ast.NewIdent("Broadcast" + vecTypeName),
				},
				Args: []ast.Expr{negInfArg},
			}

			switch signVal {
			case 1:
				// Check +Inf only: x.Equal(posInf)
				call.Fun = &ast.SelectorExpr{
					X:   x,
					Sel: ast.NewIdent("Equal"),
				}
				call.Args = []ast.Expr{posInfCall}
			case -1:
				// Check -Inf only: x.Equal(negInf)
				call.Fun = &ast.SelectorExpr{
					X:   x,
					Sel: ast.NewIdent("Equal"),
				}
				call.Args = []ast.Expr{negInfCall}
			default:
				// Check either: x.Equal(posInf).Or(x.Equal(negInf))
				posInfMask := &ast.CallExpr{
					Fun: &ast.SelectorExpr{
						X:   cloneExpr(x),
						Sel: ast.NewIdent("Equal"),
					},
					Args: []ast.Expr{posInfCall},
				}
				negInfMask := &ast.CallExpr{
					Fun: &ast.SelectorExpr{
						X:   x,
						Sel: ast.NewIdent("Equal"),
					},
					Args: []ast.Expr{negInfCall},
				}
				call.Fun = &ast.SelectorExpr{
					X:   posInfMask,
					Sel: ast.NewIdent("Or"),
				}
				call.Args = []ast.Expr{negInfMask}
			}
		}

	case "ShiftRight", "ShiftLeft", "ShiftAllRight", "ShiftAllLeft":
		// hwy.ShiftRight(v, shift) -> v.ShiftAllRight(uint64(shift))
		// archsimd's ShiftAllRight/ShiftAllLeft expect uint64, but hwy uses int
		if len(call.Args) >= 2 {
			call.Fun = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent(opInfo.Name),
			}
			shiftArg := call.Args[1]
			// Wrap shift in uint64() cast for archsimd targets
			if ctx.target.VecPackage == "archsimd" {
				shiftArg = &ast.CallExpr{
					Fun:  ast.NewIdent("uint64"),
					Args: []ast.Expr{shiftArg},
				}
			}
			call.Args = []ast.Expr{shiftArg}
		}

	case "And", "Xor":
		// archsimd float types don't have And/Xor methods, only int types do.
		// For float types on archsimd, use hwy wrappers.
		// BUT: if both operands are int32 vectors, use method call since Int32x8 has And/Xor.
		if ctx.target.VecPackage == "archsimd" && (ctx.elemType == "float32" || ctx.elemType == "float64") {
			// Check if both operands are int32 - if so, use method call
			if len(call.Args) >= 2 && isInt32Expr(call.Args[0], ctx) && isInt32Expr(call.Args[1], ctx) {
				// Int32x8 has .And() method - use method call a.And(b)
				convertToMethodCall(call, opInfo.Name)
			} else {
				// hwy.And(a, b) -> hwy.And_AVX2_F32x8(a, b)
				fullName := hwyWrapperName(opInfo.Name, ctx)
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent(fullName),
				}
				// Keep args as [a, b]
			}
		} else if ctx.isHalfPrec {
			// For half-precision contexts, integer operations (like octant masking in sin/cos)
			// may use hwy.Vec[int32] (Fallback/NEON) or archsimd.Int32x8 (AVX promoted).
			if ctx.isAVXPromoted && len(call.Args) >= 2 &&
				isInt32Expr(call.Args[0], ctx) && isInt32Expr(call.Args[1], ctx) {
				// AVX promoted: int32 variables are archsimd.Int32x8 which has And/Xor methods
				convertToMethodCall(call, opInfo.Name)
			} else {
				// Non-AVX promoted: keep as generic hwy function
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent(opInfo.Name),
				}
			}
			// Keep args as [a, b] (for non-AVX case)
		} else {
			// Integer types or non-archsimd: use method call a.And(b)
			if len(call.Args) >= 2 {
				convertToMethodCall(call, opInfo.Name)
			}
		}

	case "Not":
		// archsimd float types don't have Not method, only int types do.
		// For float types on archsimd, use hwy wrappers.
		if ctx.target.VecPackage == "archsimd" && (ctx.elemType == "float32" || ctx.elemType == "float64") {
			// hwy.Not(a) -> hwy.Not_AVX2_F32x8(a)
			fullName := hwyWrapperName(opInfo.Name, ctx)
			call.Fun = &ast.SelectorExpr{
				X:   ast.NewIdent("hwy"),
				Sel: ast.NewIdent(fullName),
			}
			// Keep args as [a]
		} else {
			// Integer types or non-archsimd: use method call a.Not()
			if len(call.Args) >= 1 {
				convertToUnaryMethodCall(call, opInfo.Name)
			}
		}

	default:
		// Binary operations: hwy.Add(a, b) -> a.Add(b)
		if len(call.Args) >= 2 {
			convertToMethodCall(call, opInfo.Name)
		} else if len(call.Args) == 1 {
			// Other unary operations
			convertToUnaryMethodCall(call, opInfo.Name)
		}
	}
}

// transformToFunction converts hwy.LoadSlice(src) to archsimd.LoadFloat32x8Slice(src).
func transformToFunction(call *ast.CallExpr, funcName string, opInfo OpInfo, ctx *transformContext) {
	// Handle both SelectorExpr (hwy.Load) and IndexExpr (hwy.Zero[float32])
	var selExpr *ast.SelectorExpr
	var explicitTypeParam string // Explicit type parameter from hwy.Load[uint8] style calls
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		selExpr = fun
	case *ast.IndexExpr:
		// For generic functions like hwy.Load[uint8]() or hwy.Zero[float32]()
		selExpr = fun.X.(*ast.SelectorExpr)
		// Extract explicit type parameter if it's a concrete type (not a generic type param like T)
		if typeIdent, ok := fun.Index.(*ast.Ident); ok {
			typeName := typeIdent.Name
			// Check if this is a concrete type, not a generic type parameter
			isTypeParam := false
			for _, tp := range ctx.typeParams {
				if typeName == tp.Name {
					isTypeParam = true
					break
				}
			}
			if !isTypeParam {
				// This is an explicit concrete type like uint8, float32, etc.
				explicitTypeParam = typeName
			}
		}
	default:
		return
	}

	if ctx.target.IsFallback() {
		// For fallback, use the appropriate package
		if opInfo.SubPackage != "" {
			// Contrib functions use their subpackage with target suffix
			// e.g., contrib.Sigmoid -> math.BaseSigmoidVec_fallback
			selExpr.X = ast.NewIdent(opInfo.SubPackage)
			fullName := fmt.Sprintf("%s_%s%s", opInfo.Name, strings.ToLower(ctx.target.Name), getHwygenTypeSuffix(ctx.elemType))
			selExpr.Sel.Name = fullName
		} else {
			// Core ops use hwy package
			selExpr.X = ast.NewIdent("hwy")
			// Use opInfo.Name if it differs from the source funcName (e.g., ShiftAllRight -> ShiftRight)
			if opInfo.Name != "" {
				selExpr.Sel.Name = opInfo.Name
			} else {
				selExpr.Sel.Name = funcName
			}
		}
		return
	}

	// For Float16/BFloat16 on SIMD targets, use hwy package functions instead of archsimd calls.
	// archsimd doesn't have native support for half-precision types.
	if ctx.isHalfPrec {
		setFallback := func(name string) {
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = name
		}

		if transformHalfPrecMerge(call, funcName, ctx, setFallback) {
			return
		}

		if transformHalfPrecF16FuncOps(call, funcName, ctx, setFallback, false) {
			return
		}

		// For Load/Store/Set/Zero on F16/BF16, use asm types for NEON or generic hwy functions
		switch funcName {
		case "Load":
			// hwy.Load = fast pointer-based load
			if transformHalfPrecLoadPtr(call, ctx, setFallback) {
				return
			}
		case "LoadSlice":
			// hwy.LoadSlice = safe slice-based load
			if transformHalfPrecLoadSlice(call, ctx) {
				return
			}
			setFallback("LoadSlice")
			return
		case "Store":
			if transformHalfPrecStoreFunc(call, funcName, ctx, setFallback) {
				return
			}
		case "StoreSlice":
			if transformHalfPrecStoreSliceFunc(call, funcName, ctx, setFallback) {
				return
			}
		case "Set":
			if transformHalfPrecSet(call, funcName, ctx) {
				return
			}
			setFallback("Set")
			// Note: For half-precision, argument wrapping is handled in
			// transformHalfPrecisionFallback after scalar variables are tracked.
			return
		case "Zero":
			if transformHalfPrecZero(call, funcName, ctx) {
				return
			}
			setFallback("Zero")
			return
		case "RoundToEven", "ConvertToInt32":
			if transformHalfPrecRoundConvert(call, funcName, ctx, setFallback) {
				return
			}
		case "ConvertExponentToFloat":
			// For Float16/BFloat16, use dedicated conversion functions
			if ctx.elemType == "hwy.Float16" {
				selExpr.X = ast.NewIdent("hwy")
				selExpr.Sel.Name = "ConvertToF16"
			} else {
				selExpr.X = ast.NewIdent("hwy")
				selExpr.Sel.Name = "ConvertToBF16"
			}
			// Strip the type parameter if present
			if indexExpr, ok := call.Fun.(*ast.IndexExpr); ok {
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent(selExpr.Sel.Name),
				}
				_ = indexExpr // used to strip type param
			}
			return
		case "Pow2":
			if transformHalfPrecPow2(call, funcName, ctx) {
				return
			}
		case "Const":
			if transformHalfPrecConst(call, funcName, ctx) {
				return
			}
			// Keep hwy.Const[T] with type parameter for Fallback / skip targets
			return
		case "Not":
			if transformHalfPrecNot(call, funcName, ctx, setFallback) {
				return
			}
		case "Xor", "And":
			if transformHalfPrecBinaryBitwise(call, funcName, ctx, setFallback) {
				return
			}
		case "Or", "AndNot":
			if transformHalfPrecOrAndNot(call, funcName, ctx, setFallback) {
				return
			}
		case "NotEqual":
			if transformHalfPrecNotEqual(call, funcName, ctx, setFallback) {
				return
			}
		case "Pow":
			if transformHalfPrecPow(call, funcName, ctx, setFallback) {
				return
			}
		case "SignBit":
			if transformHalfPrecSignBit(call, funcName, ctx) {
				return
			}
		case "InterleaveLower", "InterleaveUpper":
			if transformHalfPrecInterleave(call, funcName, ctx, setFallback) {
				return
			}
		case "Load4":
			if transformHalfPrecLoad4(call, funcName, ctx) {
				return
			}
		case "MaskAnd", "MaskOr":
			if transformHalfPrecMaskOpsFunc(call, funcName, ctx, setFallback) {
				return
			}
		}
		// For other operations without F16/BF16 variants, fall through
	}

	// For SIMD targets, transform to package calls (archsimd for AVX, asm for NEON)
	var fullName string
	// Determine element type: explicit type param > function's default
	effectiveElemType := ctx.elemType
	if explicitTypeParam != "" {
		effectiveElemType = explicitTypeParam
	}
	vecTypeName := getVectorTypeName(effectiveElemType, ctx.target)
	pkgName := ctx.vecPkgName

	// Check if this op should be redirected to hwy wrappers (archsimd doesn't have it)
	if opInfo.Package == "hwy" && opInfo.SubPackage == "" {
		// Use hwy wrapper instead of archsimd
		// Try to infer lanes and element type from any argument (for operations like TableLookupBytes)
		shortTypeName := getShortTypeName(effectiveElemType, ctx.target)
		inferredLanes := 0
		inferredElemType := effectiveElemType
		for _, arg := range call.Args {
			if argIdent, ok := arg.(*ast.Ident); ok {
				if lanes, found := ctx.varVecLanes[argIdent.Name]; found {
					inferredLanes = lanes
					// Also check if we have an element type for this variable
					if elemType, hasType := ctx.varVecElemType[argIdent.Name]; hasType {
						inferredElemType = elemType
					}
					break // Use the first known lanes
				}
			}
		}
		if inferredLanes > 0 {
			shortTypeName = getShortTypeNameForLanes(inferredElemType, inferredLanes)
		} else if ctx.inferredFuncLanes > 0 {
			// Fall back to function-level inferred lanes (from Load calls)
			// Cap at target's max lanes to avoid generating invalid types (e.g., Uint8x32 on NEON)
			useLanes := ctx.inferredFuncLanes
			targetLanes := ctx.target.LanesFor(effectiveElemType)
			if useLanes > targetLanes {
				useLanes = targetLanes
			}
			shortTypeName = getShortTypeNameForLanes(effectiveElemType, useLanes)
		}
		fullName = fmt.Sprintf("%s_%s_%s", opInfo.Name, ctx.target.Name, shortTypeName)
		selExpr.X = ast.NewIdent("hwy")
		selExpr.Sel.Name = fullName
		// Strip the IndexExpr if call.Fun was hwy.Func[T]() - the wrapper doesn't use type params
		call.Fun = selExpr
		return
	}

	switch funcName {
	case "LoadSlice":
		// Check if we can determine the slice size from the argument
		// For example, hwy.LoadSlice(data[:16]) with uint8 should use Uint8x16, not Uint8x32
		loadVecTypeName := vecTypeName
		if len(call.Args) > 0 {
			sliceBytes := getSliceSize(call.Args[0])
			elemSize := elemTypeSize(effectiveElemType)
			targetLanes := ctx.target.LanesFor(effectiveElemType)
			if sliceBytes > 0 && elemSize > 0 {
				detectedLanes := sliceBytes / elemSize
				// Only use smaller type if detected lanes is less than target default
				// and is a valid vector size (power of 2, typically 2, 4, 8, 16, 32, 64)
				if detectedLanes < targetLanes && detectedLanes > 0 {
					loadVecTypeName = getVectorTypeNameForLanes(effectiveElemType, detectedLanes)
				}
			} else if ctx.inferredFuncLanes > 0 && ctx.inferredFuncLanes < targetLanes {
				// No explicit size, but we have inferred lanes from earlier in the function
				loadVecTypeName = getVectorTypeNameForLanes(effectiveElemType, ctx.inferredFuncLanes)
			}
		}
		fullName = fmt.Sprintf("Load%sSlice", loadVecTypeName)
		selExpr.X = ast.NewIdent(pkgName)
	case "Load":
		// hwy.LoadSlice(src) -> pointer-based load for performance (no bounds check)
		// For half-precision types on SIMD targets
		if isHalfPrecisionType(effectiveElemType) {
			if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON && len(call.Args) >= 1 {
				// NEON: use asm.LoadFloat16x8Ptr(unsafe.Pointer(&slice[0]))
				loadFuncName := "LoadFloat16x8Ptr"
				if isBFloat16Type(effectiveElemType) {
					loadFuncName = "LoadBFloat16x8Ptr"
				}
				sliceArg := call.Args[0]
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("asm"),
					Sel: ast.NewIdent(loadFuncName),
				}
				call.Args = []ast.Expr{unsafeSlicePointer(sliceArg)}
				return
			}
			// AVX2/AVX512: use asm load functions with pointer-based access (no bounds check)
			if isAVXPromotedHalfPrec(ctx.target, effectiveElemType) && len(call.Args) >= 1 {
				loadFuncName := avxPromotedLoadPtrFunc(ctx.target, effectiveElemType)
				sliceArg := call.Args[0]
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("asm"),
					Sel: ast.NewIdent(loadFuncName),
				}
				call.Args = []ast.Expr{unsafeSlicePointer(sliceArg)}
				return
			}
			// Fallback: keep hwy.LoadSlice() for half-precision
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = "Load"
			return
		}

		if ctx.target.IsAVX() || ctx.target.IsNEON() {
			// For SIMD targets, use unsafe pointer cast to avoid bounds checks
			// pkg.LoadFloat32x8((*[8]float32)(unsafe.Pointer(&src[idx])))
			lanes := ctx.target.LanesFor(effectiveElemType)
			fullName = fmt.Sprintf("Load%s", vecTypeName)
			selExpr.X = ast.NewIdent(pkgName)

			// Transform argument to pointer cast
			if len(call.Args) > 0 {
				call.Args[0] = arrayPointerCast(lanes, effectiveElemType, unsafeSlicePointer(call.Args[0]))
			}
		} else {
			// Fallback: keep generic hwy.Load
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = "Load"
			return
		}
	case "Store":
		// For Fallback (IsMethod: false), use generic hwy.Store
		// NEON/AVX use IsMethod: true, handled in transformToMethod
		selExpr.X = ast.NewIdent("hwy")
		selExpr.Sel.Name = "Store"
		return
	case "Load4":
		// For Vec types (Float16/BFloat16), use hwy wrapper or asm function
		if strings.HasPrefix(vecTypeName, "Vec") || strings.HasPrefix(vecTypeName, "hwy.Vec") {
			if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON && isHalfPrecisionType(effectiveElemType) {
				// NEON half-precision: use asm.Load4Float16x8/Load4BFloat16x8
				load4Func := "Load4Float16x8"
				if isBFloat16Type(effectiveElemType) {
					load4Func = "Load4BFloat16x8"
				}
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("asm"),
					Sel: ast.NewIdent(load4Func),
				}
				if len(call.Args) > 0 {
					call.Args = []ast.Expr{unsafeSlicePointer(call.Args[0])}
				}
				return
			}
			if isAVXPromotedHalfPrec(ctx.target, effectiveElemType) {
				// AVX promoted half-precision: use asm.Load4Float16x8AVX2Slice etc.
				load4Func := avxPromotedLoad4SliceFunc(ctx.target, effectiveElemType)
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("asm"),
					Sel: ast.NewIdent(load4Func),
				}
				// Transform arg: []hwy.Float16 -> []uint16
				if len(call.Args) > 0 {
					call.Args = []ast.Expr{halfPrecSliceToUint16(call.Args[0])}
				}
				return
			}
			fullName = fmt.Sprintf("Load4_%s_Vec", ctx.target.Name)
			selExpr.X = ast.NewIdent("hwy")
		} else {
			// For NEON: asm.Load4Float32x4Slice (single ld1 instruction)
			// For AVX2/512/Fallback: handled by hwy wrapper at line 2094-2100
			fullName = fmt.Sprintf("Load4%sSlice", vecTypeName)
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "Set", "Const":
		// Both Set and Const broadcast a scalar value to all lanes
		fullName = fmt.Sprintf("Broadcast%s", vecTypeName)
		selExpr.X = ast.NewIdent(pkgName)
	case "Zero":
		if opInfo.Package == "special" {
			// archsimd doesn't have Zero*, use Broadcast with 0
			fullName = fmt.Sprintf("Broadcast%s", vecTypeName)
			selExpr.X = ast.NewIdent(pkgName)
			// Add 0 as argument
			call.Args = []ast.Expr{&ast.BasicLit{Kind: token.INT, Value: "0"}}
		} else {
			fullName = fmt.Sprintf("Zero%s", vecTypeName)
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "SlideUpLanes", "SlideDownLanes":
		// For NEON: hwy.Slide*Lanes(v, offset) -> asm.Slide*LanesFloat32x4(v, offset)
		// For AVX2/AVX512: hwy.Slide*Lanes(v, offset) -> hwy.Slide*Lanes_AVX2_F32x8(v, offset)
		if ctx.target.IsAVX() {
			shortTypeName := getShortTypeName(ctx.elemType, ctx.target)
			fullName = fmt.Sprintf("%s_%s_%s", funcName, ctx.target.Name, shortTypeName)
			selExpr.X = ast.NewIdent("hwy")
		} else {
			fullName = fmt.Sprintf("%s%s", funcName, vecTypeName)
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "InsertLane":
		// hwy.InsertLane(v, idx, val) -> asm.InsertLaneFloat32x4(v, idx, val)
		fullName = fmt.Sprintf("InsertLane%s", vecTypeName)
		selExpr.X = ast.NewIdent(pkgName)
	case "MaskLoad":
		fullName = fmt.Sprintf("MaskLoad%sSlice", vecTypeName)
		selExpr.X = ast.NewIdent(pkgName)
	case "Compress":
		// Use hwy wrapper if configured
		if opInfo.Package == "hwy" {
			fullName = hwyWrapperName(opInfo.Name, ctx)
			selExpr.X = ast.NewIdent("hwy")
		} else {
			// Compress returns (Vec, int). Maps to CompressKeysF32x4, etc.
			switch ctx.elemType {
			case "float32":
				fullName = "CompressKeysF32x4"
			case "float64":
				fullName = "CompressKeysF64x2"
			case "int32":
				fullName = "CompressKeysI32x4"
			case "int64":
				fullName = "CompressKeysI64x2"
			case "uint32":
				fullName = "CompressKeysU32x4"
			case "uint64":
				fullName = "CompressKeysU64x2"
			default:
				fullName = "CompressKeysF32x4"
			}
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "CompressStore":
		// Use hwy wrapper if configured
		if opInfo.Package == "hwy" {
			fullName = hwyWrapperName(opInfo.Name, ctx)
			selExpr.X = ast.NewIdent("hwy")
		} else {
			// CompressStore has type-specific versions: CompressStore (float32), CompressStoreFloat64, etc.
			switch ctx.elemType {
			case "float32":
				fullName = "CompressStore"
			case "float64":
				fullName = "CompressStoreFloat64"
			case "int32":
				fullName = "CompressStoreInt32"
			case "int64":
				fullName = "CompressStoreInt64"
			case "uint32":
				fullName = "CompressStoreUint32"
			case "uint64":
				fullName = "CompressStoreUint64"
			default:
				fullName = "CompressStore"
			}
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "FirstN":
		// Use hwy wrapper if configured
		if opInfo.Package == "hwy" {
			fullName = hwyWrapperName(opInfo.Name, ctx)
			selExpr.X = ast.NewIdent("hwy")
		} else {
			// FirstN returns a mask type: Int32x4 for 4-lane, Int64x2 for 2-lane
			switch ctx.elemType {
			case "float32":
				fullName = "FirstN"
			case "float64":
				fullName = "FirstNFloat64"
			case "int32", "uint32":
				fullName = "FirstN" // Int32x4 mask for 32-bit types
			case "int64", "uint64":
				fullName = "FirstNInt64" // Int64x2 mask for 64-bit types
			default:
				fullName = "FirstN"
			}
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "IfThenElse":
		// Use hwy wrapper if configured
		if opInfo.Package == "hwy" {
			fullName = hwyWrapperName(opInfo.Name, ctx)
			selExpr.X = ast.NewIdent("hwy")
		} else {
			// IfThenElse has type-specific versions for NEON
			switch ctx.elemType {
			case "float32":
				fullName = "IfThenElse"
			case "float64":
				fullName = "IfThenElseFloat64"
			case "int32":
				fullName = "IfThenElseInt32"
			case "int64":
				fullName = "IfThenElseInt64"
			default:
				fullName = "IfThenElse"
			}
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "AllTrue", "AllFalse":
		// AllTrue/AllFalse have type-specific versions for inlining:
		// e.g. AllTrueVal for Int32x4 masks, AllTrueValFloat64 for Int64x2 masks
		baseName := funcName + "Val" // AllTrueVal or AllFalseVal
		switch ctx.elemType {
		case "float32", "int32":
			fullName = baseName
		case "float64", "int64":
			fullName = baseName + "Float64"
		case "uint32":
			fullName = baseName + "Uint32"
		case "uint64":
			fullName = baseName + "Uint64"
		default:
			fullName = baseName
		}
		selExpr.X = ast.NewIdent(pkgName)
	case "SignBit":
		// SignBit has type-specific versions for NEON: SignBitFloat32x4, SignBitFloat64x2
		// For AVX2/AVX512, archsimd.SignBit() is generic
		if ctx.target.IsNEON() {
			switch ctx.elemType {
			case "float32":
				fullName = "SignBitFloat32x4"
			case "float64":
				fullName = "SignBitFloat64x2"
			default:
				fullName = "SignBitFloat32x4"
			}
		} else {
			fullName = "SignBit"
		}
		selExpr.X = ast.NewIdent(pkgName)
	case "Iota":
		// Iota needs target-specific handling since archsimd doesn't have a generic Iota.
		// NEON: type-specific asm functions (IotaFloat32x4, IotaFloat64x2, etc.)
		// AVX2/AVX512: hwy wrapper functions (Iota_AVX2_F32x8, Iota_AVX512_F32x16, etc.)
		// Float16/BFloat16 on any target: hwy.Iota[T]() generic function
		if isHalfPrecisionType(effectiveElemType) {
			if ctx.target.IsNEON() && !ctx.skipHalfPrecNEON {
				// NEON: use asm.IotaFloat16x8() / asm.IotaBFloat16x8()
				iotaFunc := "IotaFloat16x8"
				if isBFloat16Type(effectiveElemType) {
					iotaFunc = "IotaBFloat16x8"
				}
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("asm"),
					Sel: ast.NewIdent(iotaFunc),
				}
				call.Args = nil
				return
			}
			// AVX2/AVX512: use asm.IotaFloat16x8AVX2() etc.
			if isAVXPromotedHalfPrec(ctx.target, effectiveElemType) {
				iotaFunc := avxPromotedIotaFunc(ctx.target, effectiveElemType)
				call.Fun = &ast.SelectorExpr{
					X:   ast.NewIdent("asm"),
					Sel: ast.NewIdent(iotaFunc),
				}
				call.Args = nil
				return
			}
			// Fallback: use hwy.Iota[T]() generic function
			call.Fun = &ast.IndexExpr{
				X: &ast.SelectorExpr{
					X:   ast.NewIdent("hwy"),
					Sel: ast.NewIdent("Iota"),
				},
				Index: ast.NewIdent(ctx.elemType),
			}
			return
		}
		if ctx.target.IsNEON() {
			switch ctx.elemType {
			case "float32":
				fullName = "IotaFloat32x4"
			case "float64":
				fullName = "IotaFloat64x2"
			case "uint32":
				fullName = "IotaUint32x4"
			case "uint64":
				fullName = "IotaUint64x2"
			default:
				fullName = "Iota"
			}
			selExpr.X = ast.NewIdent(pkgName)
		} else if ctx.target.VecPackage == "archsimd" {
			// AVX2/AVX512: use hwy.Iota_{target}_{shortType}()
			shortTypeName := getShortTypeName(effectiveElemType, ctx.target)
			fullName = fmt.Sprintf("Iota_%s_%s", ctx.target.Name, shortTypeName)
			selExpr.X = ast.NewIdent("hwy")
		} else {
			fullName = opInfo.Name
			selExpr.X = ast.NewIdent(pkgName)
		}
	case "MaskNot":
		if opInfo.Package == "special" && len(call.Args) >= 1 {
			transformMaskNot(call, pkgName, ctx.vecTypeName, ctx.elemType)
		}
		return // Don't set fullName, we've already transformed the call
	case "ShiftRight", "ShiftLeft", "ShiftAllRight", "ShiftAllLeft":
		// archsimd's ShiftAllRight/ShiftAllLeft expect uint64, but hwy uses int.
		// After function-to-method transformation, the shift is the last arg.
		// Wrap it in a uint64() cast for archsimd targets.
		if ctx.target.VecPackage == "archsimd" && len(call.Args) >= 1 {
			lastIdx := len(call.Args) - 1
			call.Args[lastIdx] = &ast.CallExpr{
				Fun:  ast.NewIdent("uint64"),
				Args: []ast.Expr{call.Args[lastIdx]},
			}
		}
		fullName = opInfo.Name
		selExpr.X = ast.NewIdent(pkgName)
	case "Clamp":
		// hwy.Clamp(v, lo, hi) -> v.Max(lo).Min(hi)
		// archsimd and asm vector types have Max/Min methods.
		// For Fallback, keep as hwy.Clamp(v, lo, hi) since Vec[T] uses the generic function.
		if ctx.target.IsFallback() {
			selExpr.X = ast.NewIdent("hwy")
			selExpr.Sel.Name = "Clamp"
			return
		}
		if len(call.Args) >= 3 {
			v := call.Args[0]
			lo := call.Args[1]
			hi := call.Args[2]
			// v.Max(lo).Min(hi)
			call.Fun = &ast.SelectorExpr{
				X: &ast.CallExpr{
					Fun: &ast.SelectorExpr{
						X:   v,
						Sel: ast.NewIdent("Max"),
					},
					Args: []ast.Expr{lo},
				},
				Sel: ast.NewIdent("Min"),
			}
			call.Args = []ast.Expr{hi}
		}
		return
	case "ConvertExponentToFloat":
		// Convert Vec[int32] to Vec[T] for the target float type
		// For native float types, transform to e.ConvertToFloat32() method call
		if len(call.Args) >= 1 {
			var methodName string
			switch ctx.elemType {
			case "float32":
				methodName = "ConvertToFloat32"
			case "float64":
				methodName = "ConvertToFloat64"
			default:
				// Half-precision handled earlier in the isHalfPrecisionType block
				methodName = "ConvertToFloat32"
			}
			// Transform hwy.ConvertExponentToFloat[T](e) to e.ConvertToFloat32()
			convertToUnaryMethodCall(call, methodName)
		}
		return
	default:
		// For contrib functions (SubPackage), use hwygen's naming convention:
		// lowercase target, type suffix only for non-default types
		// e.g., math.BaseExpVec_avx2, math.BaseExpVec_avx2_Float64
		if opInfo.SubPackage != "" {
			fullName = fmt.Sprintf("%s_%s%s", opInfo.Name, strings.ToLower(ctx.target.Name), getHwygenTypeSuffix(ctx.elemType))
			selExpr.X = ast.NewIdent(opInfo.SubPackage) // math, vec, matvec, algo
		} else if opInfo.Package == "hwy" {
			// Core ops from hwy package (e.g., hwy.Sqrt_AVX2_F32x8)
			fullName = hwyWrapperName(opInfo.Name, ctx)
			selExpr.X = ast.NewIdent("hwy")
		} else {
			fullName = opInfo.Name
			selExpr.X = ast.NewIdent(pkgName)
		}
	}

	selExpr.Sel.Name = fullName

	// If call.Fun is an IndexExpr (from explicit type param like hwy.Load[uint8]),
	// strip the IndexExpr since asm/archsimd package functions don't use type params
	if _, ok := call.Fun.(*ast.IndexExpr); ok {
		call.Fun = selExpr
	}
}
