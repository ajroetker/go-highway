package main

import "slices"

// CIntrinsicProfile defines the complete set of C intrinsics and metadata
// for a specific target architecture + element type combination.
// The CEmitter uses these profiles to generate correct GOAT-compatible C code
// for each supported SIMD target.
type CIntrinsicProfile struct {
	ElemType   string // "float32", "float64", "float16", "bfloat16"
	TargetName string // "NEON", "AVX2", "AVX512"
	Include          string   // "#include <arm_neon.h>" or "#include <immintrin.h>"
	PolyfillDefines  []string // #define macros for missing intrinsics (emitted after #include)
	CType            string   // "float", "double", "unsigned short"

	// VecTypes maps tier name to C vector type.
	// For simple profiles (f32/f64), there is typically one main vector type.
	// For half-precision, different tiers may use different types (e.g., q vs d).
	VecTypes map[string]string

	// Tiers defines the loop hierarchy from widest to narrowest.
	Tiers []CLoopTier

	// Intrinsics per tier (key = tier name).
	// The CEmitter selects the correct intrinsic based on the current tier.
	LoadFn    map[string]string
	StoreFn   map[string]string
	AddFn     map[string]string
	SubFn     map[string]string
	MulFn     map[string]string
	DivFn     map[string]string
	FmaFn     map[string]string
	NegFn     map[string]string
	AbsFn     map[string]string
	SqrtFn    map[string]string
	RSqrtFn   map[string]string
	MinFn     map[string]string
	MaxFn     map[string]string
	DupFn     map[string]string // Broadcast scalar to all lanes
	GetLaneFn map[string]string // Extract single lane
	Load4Fn   map[string]string // Multi-load: vld1q_u64_x4; nil for AVX (falls back to 4× LoadFn)
	VecX4Type map[string]string // Multi-load struct type: "uint64x2x4_t"; nil for AVX

	// SlideUp: maps tier to the vext intrinsic for lane shifting (NEON only).
	// For AVX targets, this is nil and the translator emits a fallback comment.
	SlideUpExtFn map[string]string // "q": "vextq_f32"

	// Reduction
	ReduceSumFn map[string]string // vaddvq_f32, _mm256_reduce_add_ps

	// Shuffle/Permute
	InterleaveLowerFn  map[string]string // vzip1q_f32, _mm256_unpacklo_ps
	InterleaveUpperFn  map[string]string // vzip2q_f32, _mm256_unpackhi_ps
	TableLookupBytesFn map[string]string // vqtbl1q_u8, _mm256_shuffle_epi8

	// Bitwise
	AndFn map[string]string // vandq_u64, _mm256_and_si256
	OrFn  map[string]string // vorrq_u64, _mm256_or_si256
	XorFn map[string]string // veorq_u64, _mm256_xor_si256

	// PopCount - complex on NEON (vcntq_u8 + pairwise adds)
	PopCountFn map[string]string

	// Deferred popcount accumulation: replaces per-iteration horizontal
	// reduction (ReduceSum(PopCount(And(...)))) with vector accumulators
	// that are reduced once after the loop. Only used for NEON uint64.
	// nil/empty disables the optimization.
	PopCountPartialFn map[string]string // e.g. "neon_popcnt_u64_to_u32" — returns uint32x4_t
	AccVecType        map[string]string // e.g. "uint32x4_t"
	AccAddFn          map[string]string // e.g. "vaddq_u32"
	AccReduceFn       map[string]string // e.g. "vaddvq_u32"

	// Comparison (returns mask type)
	LessThanFn    map[string]string // vcltq_f32, _mm256_cmp_ps
	EqualFn       map[string]string // vceqq_f32, _mm256_cmp_ps
	GreaterThanFn    map[string]string // vcgtq_f32, _mm256_cmp_ps
	GreaterEqualFn   map[string]string // vcgeq_f32, _mm256_cmp_ps

	// Conditional select
	IfThenElseFn map[string]string // vbslq_f32, _mm256_blendv_ps

	// Mask extraction
	BitsFromMaskFn map[string]string // manual (NEON), _mm256_movemask_epi8

	// Mask vector type (comparison results)
	MaskType map[string]string // "uint32x4_t" (NEON), "__m256" (AVX)

	// Reduction min/max
	ReduceMinFn map[string]string // vminvq_f32
	ReduceMaxFn map[string]string // vmaxvq_f32

	// Mask logical operations
	MaskAndFn    map[string]string // vandq_u32
	MaskOrFn     map[string]string // vorrq_u32
	MaskAndNotFn map[string]string // vbicq_u32 (a AND NOT b)

	// Mask query operations (return scalar long)
	AllTrueFn       map[string]string // inline helper
	AllFalseFn      map[string]string // inline helper
	FindFirstTrueFn map[string]string // inline helper
	CountTrueFn     map[string]string // inline helper

	// Mask creation
	FirstNFn map[string]string // inline helper: mask with first N lanes true

	// Vector creation
	IotaFn map[string]string // inline helper: vector {0, 1, 2, ...}

	// Compress store (stream compaction)
	CompressStoreFn map[string]string // inline helper

	// Dot-product accumulation: BFDOT (NEON BF16), VDPBF16PS (AVX-512 BF16).
	// hwy.DotAccumulate(a, b, acc) → vbfdotq_f32(acc, a, b)
	// Returns a wider accumulator type (float32), not the narrow element type.
	DotAccFn     map[string]string // "q": "vbfdotq_f32", "zmm": "_mm512_dpbf16_ps"
	DotAccType   map[string]string // return type: "q": "float32x4_t", "zmm": "__m512"

	// Accumulator widening: keep accumulators in a wider type (e.g., f32 for BF16)
	// to avoid promote/demote round-trips on every FMA iteration.
	// When enabled, hwy.Zero[T]() in := assignments emits two widened halves
	// (e.g., float32x4_t acc_lo, acc_hi), hwy.MulAdd promotes inputs and uses
	// native f32 FMA, and hwy.Store demotes+combines once at store time.
	WidenAccumulators bool   // Enable widened accumulator optimization
	WidenedAccZero    string // Zero init expr: "vdupq_n_f32(0.0f)"
	WidenedAccType    string // Widened type: "float32x4_t"
	WidenedFmaFn      string // Native FMA on widened type: "vfmaq_f32"
	WidenedAddFn      string // Native Add on widened type: "vaddq_f32"

	// InlineHelpers contains C helper function source code that should be
	// emitted at the top of generated C files (before main functions).
	// Used for complex intrinsic sequences like NEON popcount chains.
	InlineHelpers []string

	// Math strategy: "native" means arithmetic is done directly in the element
	// type. "promoted" means elements must be promoted to a wider type (typically
	// float32) for arithmetic, then demoted back.
	MathStrategy string // "native" or "promoted"

	// NativeArithmetic is true when the profile has native SIMD arithmetic
	// (add, mul, fma, etc.) even though MathStrategy may be "promoted" for
	// complex math functions (exp, log). For example, NEON f16 has native
	// vaddq_f16/vfmaq_f16 but uses f32 promotion for exp/log.
	NativeArithmetic bool

	// ScalarArithType is the C scalar type for native arithmetic when it differs
	// from CType. For example, NEON f16 uses "float16_t" for scalar arithmetic
	// but "unsigned short" as CType (for pointer signatures).
	ScalarArithType string

	// PointerElemType overrides the C type used for slice/array pointer parameters
	// in generated function signatures. When empty, defaults to ScalarArithType
	// (if set) or CType. This is needed when ScalarArithType has a different size
	// than the actual element. For example, BF16 uses ScalarArithType="float"
	// (4 bytes) for scalar arithmetic, but elements are 2 bytes, so
	// PointerElemType="unsigned short" ensures correct array stride.
	PointerElemType string
	PromoteFn    string // e.g., "vcvt_f32_f16" -- called as PromoteFn(narrowVec)
	DemoteFn     string // fmt.Sprintf template, e.g., "vcvt_f16_f32(%s)" or "_mm256_cvtps_ph(%s, 0)"

	// Split-promote fields for NEON f16 where one float16x8_t is split into
	// two float32x4_t halves for computation, then recombined.
	SplitPromoteLo string // e.g., "vcvt_f32_f16(vget_low_f16(x))"  -- promote low half
	SplitPromoteHi string // e.g., "vcvt_f32_f16(vget_high_f16(x))" -- promote high half
	CombineFn      string // e.g., "vcombine_f16" -- recombine two narrow halves

	// Scalar promote/demote for the scalar tail loop.
	ScalarPromote string // e.g., a C expression to convert scalar elem to float
	ScalarDemote  string // e.g., a C expression to convert float back to elem type

	// CastExpr is a pointer cast expression applied to load/store pointer
	// arguments when the C type differs from the intrinsic's expected pointer
	// type.  For example, NEON f16 intrinsics expect (float16_t*) but the
	// function signature uses (unsigned short*), so CastExpr = "(float16_t*)".
	// Empty string means no cast is needed.
	CastExpr string

	// FmaArgOrder describes how the FMA intrinsic orders its arguments.
	// "acc_first" means FMA(acc, a, b) = acc + a*b  (NEON convention)
	// "acc_last"  means FMA(a, b, acc) = a*b + acc  (AVX convention)
	// Go's hwy.MulAdd(a, b, acc) follows the AVX convention (acc last).
	FmaArgOrder string // "acc_first" or "acc_last"

	// GOAT compilation settings
	GoatTarget     string   // "arm64" or "amd64"
	GoatExtraFlags []string // e.g., ["-march=armv8-a+simd+fp"]

	// SVE predicate support: every SVE intrinsic requires an svbool_t predicate.
	NeedsPredicate bool   // true for SVE — every intrinsic needs svbool_t pg
	PredicateDecl  string // e.g. "svptrue_b32()" or "svptrue_b64()"

	// FuncAttrs is appended after the parameter list in C function signatures.
	// Used for SVE streaming mode on darwin: "__arm_streaming".
	FuncAttrs string
}

// CLoopTier represents one level of the tiered loop structure used in
// generated C code. Each tier processes a different number of elements
// per iteration, from widest SIMD down to scalar.
type CLoopTier struct {
	Name         string // "zmm", "ymm", "xmm", "q", "d", "sve", "scalar"
	Lanes        int    // Number of elements per vector at this tier (0 if dynamic)
	DynamicLanes string // Runtime expression for lane count, e.g. "svcntw()" (empty = use Lanes)
	Unroll       int    // Number of vectors processed per iteration (4 for main, 1 for single)
	IsScalar     bool   // True if this tier processes one element at a time
}

// cProfileRegistry holds all known target+type profile combinations.
// Keyed as "TargetName:elemType", e.g., "NEON:float32".
var cProfileRegistry map[string]*CIntrinsicProfile

func init() {
	cProfileRegistry = make(map[string]*CIntrinsicProfile)

	// Register all profiles
	for _, p := range []*CIntrinsicProfile{
		neonF32Profile(),
		neonF64Profile(),
		neonF16Profile(),
		neonBF16Profile(),
		avx2F16Profile(),
		avx512F16Profile(),
		avx512BF16Profile(),
		neonUint64Profile(),
		neonUint8Profile(),
		neonUint32Profile(),
		neonInt32Profile(),
		neonInt64Profile(),
		sveDarwinF32Profile(),
		sveDarwinF64Profile(),
		sveLinuxF32Profile(),
		sveLinuxF64Profile(),
	} {
		// Primary key: "TargetName:ElemType"
		key := p.TargetName + ":" + p.ElemType
		cProfileRegistry[key] = p
	}

	// Register aliases for bare type names (without hwy. prefix)
	aliases := map[string]string{
		"float16":  "hwy.Float16",
		"bfloat16": "hwy.BFloat16",
	}
	for bare, qualified := range aliases {
		for _, target := range []string{"NEON", "AVX2", "AVX512", "SVE_DARWIN", "SVE_LINUX"} {
			qualifiedKey := target + ":" + qualified
			if p, ok := cProfileRegistry[qualifiedKey]; ok {
				cProfileRegistry[target+":"+bare] = p
			}
		}
	}
}

// GetCProfile returns the CIntrinsicProfile for the given target and element
// type, or nil if no profile exists for that combination.
func GetCProfile(targetName, elemType string) *CIntrinsicProfile {
	key := targetName + ":" + elemType
	return cProfileRegistry[key]
}

// ---------------------------------------------------------------------------
// NEON float32
// ---------------------------------------------------------------------------

func neonF32Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "float32",
		TargetName: "NEON",
		Include:    "#include <arm_neon.h>",
		CType:      "float",
		VecTypes: map[string]string{
			"q":      "float32x4_t",
			"scalar": "float32x4_t",
		},
		Tiers: []CLoopTier{
			{Name: "q", Lanes: 4, Unroll: 4, IsScalar: false},
			{Name: "q", Lanes: 4, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn:    map[string]string{"q": "vld1q_f32", "scalar": "vld1q_dup_f32"},
		StoreFn:   map[string]string{"q": "vst1q_f32", "scalar": "vst1q_lane_f32"},
		AddFn:     map[string]string{"q": "vaddq_f32"},
		SubFn:     map[string]string{"q": "vsubq_f32"},
		MulFn:     map[string]string{"q": "vmulq_f32"},
		DivFn:     map[string]string{"q": "vdivq_f32"},
		FmaFn:     map[string]string{"q": "vfmaq_f32"},
		NegFn:     map[string]string{"q": "vnegq_f32"},
		AbsFn:     map[string]string{"q": "vabsq_f32"},
		SqrtFn:    map[string]string{"q": "vsqrtq_f32"},
		RSqrtFn:   map[string]string{"q": "_v_rsqrt_f32"},
		MinFn:     map[string]string{"q": "vminq_f32"},
		MaxFn:     map[string]string{"q": "vmaxq_f32"},
		DupFn:     map[string]string{"q": "vdupq_n_f32"},
		GetLaneFn: map[string]string{"q": "vgetq_lane_f32"},
		Load4Fn:   map[string]string{"q": "vld1q_f32_x4"},
		VecX4Type: map[string]string{"q": "float32x4x4_t"},

		SlideUpExtFn:      map[string]string{"q": "vextq_f32"},
		ReduceSumFn:       map[string]string{"q": "vaddvq_f32"},
		InterleaveLowerFn: map[string]string{"q": "vzip1q_f32"},
		InterleaveUpperFn: map[string]string{"q": "vzip2q_f32"},
		LessThanFn:        map[string]string{"q": "vcltq_f32"},
		EqualFn:           map[string]string{"q": "vceqq_f32"},
		GreaterThanFn:     map[string]string{"q": "vcgtq_f32"},
		GreaterEqualFn:    map[string]string{"q": "vcgeq_f32"},
		IfThenElseFn:      map[string]string{"q": "vbslq_f32"},
		MaskType:          map[string]string{"q": "uint32x4_t"},
		ReduceMinFn:       map[string]string{"q": "vminvq_f32"},
		ReduceMaxFn:       map[string]string{"q": "vmaxvq_f32"},

		MaskAndFn:    map[string]string{"q": "vandq_u32"},
		MaskOrFn:     map[string]string{"q": "vorrq_u32"},
		MaskAndNotFn: map[string]string{"q": "vbicq_u32"},

		AllTrueFn:       map[string]string{"q": "hwy_all_true_u32"},
		AllFalseFn:      map[string]string{"q": "hwy_all_false_u32"},
		FindFirstTrueFn: map[string]string{"q": "hwy_find_first_true_u32"},
		CountTrueFn:     map[string]string{"q": "hwy_count_true_u32"},
		FirstNFn:        map[string]string{"q": "hwy_first_n_u32"},
		IotaFn:          map[string]string{"q": "hwy_iota_f32"},
		CompressStoreFn: map[string]string{"q": "hwy_compress_store_f32"},

		InlineHelpers: slices.Concat([]string{
			`static inline unsigned int float_to_bits(float f) {
    unsigned int bits;
    __builtin_memcpy(&bits, &f, 4);
    return bits;
}`,
			`static inline float bits_to_float(unsigned int bits) {
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}`,
		}, neonF32MathHelpers, scalarF64MathHelpers, neonF32MaskHelpers),

		MathStrategy:   "native",
		FmaArgOrder:    "acc_first",
		GoatTarget:     "arm64",
		GoatExtraFlags: []string{"-march=armv8-a+simd+fp"},
	}
}

// ---------------------------------------------------------------------------
// NEON float64
// ---------------------------------------------------------------------------

func neonF64Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "float64",
		TargetName: "NEON",
		Include:    "#include <arm_neon.h>",
		CType:      "double",
		VecTypes: map[string]string{
			"q":      "float64x2_t",
			"scalar": "float64x2_t",
		},
		Tiers: []CLoopTier{
			{Name: "q", Lanes: 2, Unroll: 4, IsScalar: false},
			{Name: "q", Lanes: 2, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn:    map[string]string{"q": "vld1q_f64", "scalar": "vld1q_dup_f64"},
		StoreFn:   map[string]string{"q": "vst1q_f64", "scalar": "vst1q_lane_f64"},
		AddFn:     map[string]string{"q": "vaddq_f64"},
		SubFn:     map[string]string{"q": "vsubq_f64"},
		MulFn:     map[string]string{"q": "vmulq_f64"},
		DivFn:     map[string]string{"q": "vdivq_f64"},
		FmaFn:     map[string]string{"q": "vfmaq_f64"},
		NegFn:     map[string]string{"q": "vnegq_f64"},
		AbsFn:     map[string]string{"q": "vabsq_f64"},
		SqrtFn:    map[string]string{"q": "vsqrtq_f64"},
		RSqrtFn:   map[string]string{"q": "_v_rsqrt_f64"},
		MinFn:     map[string]string{"q": "vminq_f64"},
		MaxFn:     map[string]string{"q": "vmaxq_f64"},
		DupFn:     map[string]string{"q": "vdupq_n_f64"},
		GetLaneFn: map[string]string{"q": "vgetq_lane_f64"},
		Load4Fn:   map[string]string{"q": "vld1q_f64_x4"},
		VecX4Type: map[string]string{"q": "float64x2x4_t"},

		SlideUpExtFn:      map[string]string{"q": "vextq_f64"},
		ReduceSumFn:       map[string]string{"q": "vaddvq_f64"},
		InterleaveLowerFn: map[string]string{"q": "vzip1q_f64"},
		InterleaveUpperFn: map[string]string{"q": "vzip2q_f64"},
		LessThanFn:        map[string]string{"q": "vcltq_f64"},
		EqualFn:           map[string]string{"q": "vceqq_f64"},
		GreaterThanFn:     map[string]string{"q": "vcgtq_f64"},
		GreaterEqualFn:    map[string]string{"q": "vcgeq_f64"},
		IfThenElseFn:      map[string]string{"q": "vbslq_f64"},
		MaskType:          map[string]string{"q": "uint64x2_t"},
		ReduceMinFn:       map[string]string{"q": "vminvq_f64"},
		ReduceMaxFn:       map[string]string{"q": "vmaxvq_f64"},

		MaskAndFn:    map[string]string{"q": "vandq_u64"},
		MaskOrFn:     map[string]string{"q": "vorrq_u64"},
		MaskAndNotFn: map[string]string{"q": "vbicq_u64"},

		AllTrueFn:       map[string]string{"q": "hwy_all_true_u64"},
		AllFalseFn:      map[string]string{"q": "hwy_all_false_u64"},
		FindFirstTrueFn: map[string]string{"q": "hwy_find_first_true_u64"},
		CountTrueFn:     map[string]string{"q": "hwy_count_true_u64"},
		FirstNFn:        map[string]string{"q": "hwy_first_n_u64"},
		IotaFn:          map[string]string{"q": "hwy_iota_f64"},
		CompressStoreFn: map[string]string{"q": "hwy_compress_store_f64"},

		InlineHelpers: slices.Concat(neonF64MathHelpers, neonF64MaskHelpers),

		MathStrategy:   "native",
		FmaArgOrder:    "acc_first",
		GoatTarget:     "arm64",
		GoatExtraFlags: []string{"-march=armv8-a+simd+fp"},
	}
}

// ---------------------------------------------------------------------------
// NEON float16 (ARMv8.2-A native FP16)
// ---------------------------------------------------------------------------
// NEON FP16 has native arithmetic at both 128-bit (float16x8_t, 8 lanes)
// and 64-bit (float16x4_t, 4 lanes) widths. Math functions (exp, log, etc.)
// use the "promoted" strategy: promote to float32, compute, demote back.

func neonF16Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "hwy.Float16",
		TargetName: "NEON",
		Include:    "#include <arm_neon.h>",
		PolyfillDefines: []string{
			// vaddvq_f16 is not available on all targets (requires Armv8.4-A+).
			// Polyfill: promote to f32, horizontal sum, demote back to f16.
			"#define vaddvq_f16(v) ((__fp16)vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(v)), vcvt_f32_f16(vget_high_f16(v)))))",
		},
		CType: "unsigned short",
		VecTypes: map[string]string{
			"q":      "float16x8_t",
			"d":      "float16x4_t",
			"wide":   "float32x4_t", // promoted f32 vector for math computations
			"half":   "float16x4_t", // demoted half after f32 computation
			"scalar": "float16x4_t",
		},
		Tiers: []CLoopTier{
			{Name: "q", Lanes: 8, Unroll: 4, IsScalar: false},
			{Name: "q", Lanes: 8, Unroll: 1, IsScalar: false},
			{Name: "d", Lanes: 4, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn: map[string]string{
			"q":      "vld1q_f16",
			"d":      "vld1_f16",
			"scalar": "vld1_dup_f16",
		},
		StoreFn: map[string]string{
			"q":      "vst1q_f16",
			"d":      "vst1_f16",
			"scalar": "vst1_lane_f16",
		},
		AddFn: map[string]string{
			"q": "vaddq_f16",
			"d": "vadd_f16",
		},
		SubFn: map[string]string{
			"q": "vsubq_f16",
			"d": "vsub_f16",
		},
		MulFn: map[string]string{
			"q": "vmulq_f16",
			"d": "vmul_f16",
		},
		DivFn: map[string]string{
			"q": "vdivq_f16",
			"d": "vdiv_f16",
		},
		FmaFn: map[string]string{
			"q": "vfmaq_f16",
			"d": "vfma_f16",
		},
		NegFn: map[string]string{
			"q": "vnegq_f16",
			"d": "vneg_f16",
		},
		AbsFn: map[string]string{
			"q": "vabsq_f16",
			"d": "vabs_f16",
		},
		SqrtFn: map[string]string{
			"q": "vsqrtq_f16",
			// d-register sqrt not available; scalar falls back to promote-compute-demote
		},
		MinFn: map[string]string{
			"q": "vminq_f16",
			"d": "vmin_f16",
		},
		MaxFn: map[string]string{
			"q": "vmaxq_f16",
			"d": "vmax_f16",
		},
		DupFn: map[string]string{
			"q":      "vdupq_n_f16",
			"d":      "vdup_n_f16",
			"scalar": "vdup_n_f16",
		},
		GetLaneFn: map[string]string{
			"q":      "vgetq_lane_f16",
			"d":      "vget_lane_f16",
			"scalar": "vst1_lane_f16",
		},

		SlideUpExtFn:      map[string]string{"q": "vextq_f16"},
		ReduceSumFn:       map[string]string{"q": "vaddvq_f16", "d": "vaddv_f16"},
		InterleaveLowerFn: map[string]string{"q": "vzip1q_f16", "d": "vzip1_f16"},
		InterleaveUpperFn: map[string]string{"q": "vzip2q_f16", "d": "vzip2_f16"},
		LessThanFn:        map[string]string{"q": "vcltq_f16", "d": "vclt_f16"},
		EqualFn:           map[string]string{"q": "vceqq_f16", "d": "vceq_f16"},
		GreaterThanFn:     map[string]string{"q": "vcgtq_f16", "d": "vcgt_f16"},
		GreaterEqualFn:    map[string]string{"q": "vcgeq_f16", "d": "vcge_f16"},
		IfThenElseFn:      map[string]string{"q": "vbslq_f16", "d": "vbsl_f16"},
		MaskType:          map[string]string{"q": "uint16x8_t", "d": "uint16x4_t"},
		ReduceMinFn:       map[string]string{"q": "vminvq_f16"},
		ReduceMaxFn:       map[string]string{"q": "vmaxvq_f16"},

		MaskAndFn:    map[string]string{"q": "vandq_u16"},
		MaskOrFn:     map[string]string{"q": "vorrq_u16"},
		MaskAndNotFn: map[string]string{"q": "vbicq_u16"},

		AllTrueFn:       map[string]string{"q": "hwy_all_true_u16"},
		AllFalseFn:      map[string]string{"q": "hwy_all_false_u16"},
		FindFirstTrueFn: map[string]string{"q": "hwy_find_first_true_u16"},
		CountTrueFn:     map[string]string{"q": "hwy_count_true_u16"},
		FirstNFn:        map[string]string{"q": "hwy_first_n_u16"},
		IotaFn:          map[string]string{"q": "hwy_iota_f16"},
		CompressStoreFn: map[string]string{"q": "hwy_compress_store_f16"},

		NativeArithmetic: true,
		ScalarArithType:  "float16_t",
		InlineHelpers:    slices.Concat(neonF32MathHelpers, scalarF64MathHelpers, neonF16MaskHelpers),
		MathStrategy:     "promoted",
		PromoteFn:      "vcvt_f32_f16",
		DemoteFn:       "vcvt_f16_f32(%s)",
		SplitPromoteLo: "vcvt_f32_f16(vget_low_f16(%s))",   // %s = narrow vector variable
		SplitPromoteHi: "vcvt_f32_f16(vget_high_f16(%s))",  // %s = narrow vector variable
		CombineFn:      "vcombine_f16(%s, %s)", // %s = lo half, %s = hi half
		CastExpr:       "(float16_t*)",
		FmaArgOrder:    "acc_first",
		GoatTarget:     "arm64",
		GoatExtraFlags: []string{"-march=armv8.2-a+fp16+simd"},
	}
}

// ---------------------------------------------------------------------------
// NEON bfloat16 (ARMv8.6-A BF16 extension)
// ---------------------------------------------------------------------------
// BFloat16 has NO native SIMD arithmetic. All operations use the
// promote-to-F32 -> compute -> demote-to-BF16 pattern.
// Promotion: vshll_n_u16(val, 16) + vreinterpretq_f32_u32
// Demotion: round-to-nearest-even bias + vmovn_u32
// Special instructions: vbfdotq_f32, vbfmmlaq_f32 for dot product / matmul.

func neonBF16Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "hwy.BFloat16",
		TargetName: "NEON",
		Include:    "#include <arm_neon.h>",
		CType:      "unsigned short",
		VecTypes: map[string]string{
			"q":      "bfloat16x8_t",
			"wide":   "float32x4_t", // promoted f32 vector for math computations
			"half":   "uint16x4_t",  // demoted half for bf16 recombine
			"scalar": "bfloat16x8_t",
		},
		Tiers: []CLoopTier{
			{Name: "q", Lanes: 8, Unroll: 4, IsScalar: false},
			{Name: "q", Lanes: 8, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn: map[string]string{
			"q":      "vld1q_bf16",
			"scalar": "vld1q_bf16", // load full vector even for scalar tail (masked by loop bound)
		},
		StoreFn: map[string]string{
			"q":      "vst1q_bf16",
			"scalar": "vst1q_bf16",
		},
		// BF16 arithmetic uses inline helpers that handle promote-compute-demote.
		// Each helper takes/returns bfloat16x8_t, so they compose correctly.
		AddFn:  map[string]string{"q": "bf16_add_q"},
		SubFn:  map[string]string{"q": "bf16_sub_q"},
		MulFn:  map[string]string{"q": "bf16_mul_q"},
		DivFn:  map[string]string{"q": "bf16_div_q"},
		FmaFn:  map[string]string{"q": "bf16_fma_q"},
		NegFn:  map[string]string{"q": "bf16_neg_q"},
		AbsFn:  map[string]string{"q": "bf16_abs_q"},
		SqrtFn: map[string]string{"q": "bf16_sqrt_q"},
		MinFn:  map[string]string{"q": "bf16_min_q"},
		MaxFn:  map[string]string{"q": "bf16_max_q"},
		DupFn: map[string]string{
			"q":      "bf16_dup_q",
			"scalar": "bf16_dup_q",
		},
		GetLaneFn: map[string]string{
			"q": "vgetq_lane_bf16",
		},

		ReduceSumFn: map[string]string{"q": "bf16_reducesum_q"},
		ReduceMinFn: map[string]string{"q": "bf16_reducemin_q"},
		ReduceMaxFn: map[string]string{"q": "bf16_reducemax_q"},

		LessThanFn:     map[string]string{"q": "bf16_lt_q"},
		EqualFn:        map[string]string{"q": "bf16_eq_q"},
		GreaterThanFn:  map[string]string{"q": "bf16_gt_q"},
		GreaterEqualFn: map[string]string{"q": "bf16_ge_q"},
		IfThenElseFn:   map[string]string{"q": "bf16_ifelse_q"},
		MaskType:       map[string]string{"q": "uint16x8_t"},

		DotAccFn:   map[string]string{"q": "vbfdotq_f32"},
		DotAccType: map[string]string{"q": "float32x4_t"},

		MaskAndFn:    map[string]string{"q": "vandq_u16"},
		MaskOrFn:     map[string]string{"q": "vorrq_u16"},
		MaskAndNotFn: map[string]string{"q": "vbicq_u16"},

		AllTrueFn:       map[string]string{"q": "hwy_all_true_u16"},
		AllFalseFn:      map[string]string{"q": "hwy_all_false_u16"},
		FindFirstTrueFn: map[string]string{"q": "hwy_find_first_true_u16"},
		CountTrueFn:     map[string]string{"q": "hwy_count_true_u16"},
		FirstNFn:        map[string]string{"q": "hwy_first_n_u16"},
		IotaFn:          map[string]string{"q": "hwy_iota_bf16"},
		CompressStoreFn: map[string]string{"q": "hwy_compress_store_f16"}, // reuse f16 compress_store (same lane layout)

		ScalarArithType: "float",
		PointerElemType: "unsigned short", // BF16 elements are 2 bytes, not 4 (float)
		ScalarPromote:   "bf16_scalar_to_f32",
		ScalarDemote:    "f32_scalar_to_bf16",
		InlineHelpers:   slices.Concat(neonBF16ArithHelpers, neonF32MathHelpers, scalarF64MathHelpers, neonF16MaskHelpers, neonBF16MaskHelpers),
		MathStrategy:    "promoted",
		PromoteFn:       "bf16_promote_lo",
		DemoteFn:        "bf16_demote_half(%s)",
		SplitPromoteLo:  "bf16_promote_lo(%s)",
		SplitPromoteHi:  "bf16_promote_hi(%s)",
		CombineFn:       "bf16_combine(%s, %s)",
		CastExpr:        "(bfloat16_t*)",
		NativeArithmetic: true, // Inline helpers provide full SIMD arithmetic via promote-compute-demote
		FmaArgOrder:      "acc_first",

		WidenAccumulators: true,
		WidenedAccZero:    "vdupq_n_f32(0.0f)",
		WidenedAccType:    "float32x4_t",
		WidenedFmaFn:      "vfmaq_f32",
		WidenedAddFn:      "vaddq_f32",

		GoatTarget:       "arm64",
		GoatExtraFlags:   []string{"-march=armv8.6-a+bf16+simd"},
	}
}

// ---------------------------------------------------------------------------
// AVX2 float16 (F16C conversion-only, compute in float32)
// ---------------------------------------------------------------------------
// AVX2 + F16C provides conversion instructions only (VCVTPH2PS, VCVTPS2PH).
// There is NO native float16 arithmetic. All computation is done in float32
// after promoting from float16, then demoted back to float16 for storage.
// Storage uses __m128i (8 x uint16), compute uses __m256 (8 x float32).

func avx2F16Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "hwy.Float16",
		TargetName: "AVX2",
		Include:    "#include <immintrin.h>",
		CType:      "unsigned short",
		VecTypes: map[string]string{
			"ymm":     "__m256",    // Compute type: 8 x float32 in YMM
			"xmm_f16": "__m128i",   // Storage type: 8 x float16 in XMM
			"xmm":     "__m128",    // Compute type: 4 x float32 in XMM
			"wide":    "__m256",    // Promoted f32 vector for math computations
			"scalar":  "unsigned int",
		},
		Tiers: []CLoopTier{
			{Name: "ymm", Lanes: 8, Unroll: 4, IsScalar: false},
			{Name: "ymm", Lanes: 8, Unroll: 1, IsScalar: false},
			{Name: "xmm", Lanes: 4, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn: map[string]string{
			"ymm":    "_mm_loadu_si128",   // Load 8 f16 as __m128i, then promote
			"xmm":    "_mm_loadl_epi64",   // Load 4 f16 as __m128i low half
			"scalar": "*(unsigned int*)",   // Scalar bit load
		},
		StoreFn: map[string]string{
			"ymm":    "_mm_storeu_si128",  // Store 8 f16 from __m128i after demote
			"xmm":    "_mm_storel_epi64",  // Store 4 f16 from __m128i low half
			"scalar": "*(unsigned short*)", // Scalar bit store
		},
		// All arithmetic is done after promotion to float32 in YMM registers.
		AddFn:  map[string]string{"ymm": "_mm256_add_ps", "xmm": "_mm_add_ps"},
		SubFn:  map[string]string{"ymm": "_mm256_sub_ps", "xmm": "_mm_sub_ps"},
		MulFn:  map[string]string{"ymm": "_mm256_mul_ps", "xmm": "_mm_mul_ps"},
		DivFn:  map[string]string{"ymm": "_mm256_div_ps", "xmm": "_mm_div_ps"},
		FmaFn:  map[string]string{"ymm": "_mm256_fmadd_ps", "xmm": "_mm_fmadd_ps"},
		NegFn:  map[string]string{"ymm": "_mm256_sub_ps(zero, x)", "xmm": "_mm_sub_ps(zero, x)"},
		AbsFn:  map[string]string{"ymm": "_mm256_andnot_ps(signmask, x)", "xmm": "_mm_andnot_ps(signmask, x)"},
		SqrtFn: map[string]string{"ymm": "_mm256_sqrt_ps", "xmm": "_mm_sqrt_ps"},
		MinFn:  map[string]string{"ymm": "_mm256_min_ps", "xmm": "_mm_min_ps"},
		MaxFn:  map[string]string{"ymm": "_mm256_max_ps", "xmm": "_mm_max_ps"},
		DupFn:  map[string]string{"ymm": "_mm256_set1_ps", "xmm": "_mm_set1_ps"},
		GetLaneFn: map[string]string{
			"ymm": "_mm_cvtss_f32(_mm256_castps256_ps128(x))",
			"xmm": "_mm_cvtss_f32",
		},

		MathStrategy:   "promoted",
		PromoteFn:      "_mm256_cvtph_ps",  // VCVTPH2PS: __m128i (8 f16) -> __m256 (8 f32)
		DemoteFn:       "_mm256_cvtps_ph(%s, 0)",  // VCVTPS2PH: __m256 (8 f32) -> __m128i (8 f16), round to nearest
		CastExpr:       "(__m128i*)",
		FmaArgOrder:    "acc_last",
		GoatTarget:     "amd64",
		GoatExtraFlags: []string{"-mf16c", "-mavx2", "-mfma"},
	}
}

// ---------------------------------------------------------------------------
// AVX-512 float16 (native FP16 arithmetic, Sapphire Rapids+)
// ---------------------------------------------------------------------------
// AVX-512 FP16 provides full native float16 arithmetic with up to 32 lanes
// per 512-bit ZMM register. This includes VADDPH, VSUBPH, VMULPH, VDIVPH,
// VFMADD132PH, VSQRTPH, etc. No promotion needed for basic arithmetic.

func avx512F16Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "hwy.Float16",
		TargetName: "AVX512",
		Include:    "#include <immintrin.h>",
		CType:      "unsigned short",
		VecTypes: map[string]string{
			"zmm":    "__m512h",
			"ymm":    "__m256h",
			"xmm":    "__m128h",
			"scalar": "__m128h",
		},
		Tiers: []CLoopTier{
			{Name: "zmm", Lanes: 32, Unroll: 4, IsScalar: false},
			{Name: "zmm", Lanes: 32, Unroll: 1, IsScalar: false},
			{Name: "ymm", Lanes: 16, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn: map[string]string{
			"zmm":    "_mm512_loadu_ph",
			"ymm":    "_mm256_loadu_ph",
			"xmm":    "_mm_loadu_ph",
			"scalar": "_mm_load_sh",
		},
		StoreFn: map[string]string{
			"zmm":    "_mm512_storeu_ph",
			"ymm":    "_mm256_storeu_ph",
			"xmm":    "_mm_storeu_ph",
			"scalar": "_mm_store_sh",
		},
		AddFn: map[string]string{
			"zmm":    "_mm512_add_ph",
			"ymm":    "_mm256_add_ph",
			"scalar": "_mm_add_sh",
		},
		SubFn: map[string]string{
			"zmm":    "_mm512_sub_ph",
			"ymm":    "_mm256_sub_ph",
			"scalar": "_mm_sub_sh",
		},
		MulFn: map[string]string{
			"zmm":    "_mm512_mul_ph",
			"ymm":    "_mm256_mul_ph",
			"scalar": "_mm_mul_sh",
		},
		DivFn: map[string]string{
			"zmm":    "_mm512_div_ph",
			"ymm":    "_mm256_div_ph",
			"scalar": "_mm_div_sh",
		},
		FmaFn: map[string]string{
			"zmm":    "_mm512_fmadd_ph",
			"ymm":    "_mm256_fmadd_ph",
			"scalar": "_mm_fmadd_sh",
		},
		NegFn: map[string]string{
			"zmm": "_mm512_sub_ph(_mm512_setzero_ph(), x)",
			"ymm": "_mm256_sub_ph(_mm256_setzero_ph(), x)",
		},
		AbsFn: map[string]string{
			"zmm": "_mm512_abs_ph",
			"ymm": "_mm256_abs_ph",
		},
		SqrtFn: map[string]string{
			"zmm":    "_mm512_sqrt_ph",
			"ymm":    "_mm256_sqrt_ph",
			"scalar": "_mm_sqrt_sh",
		},
		MinFn: map[string]string{
			"zmm":    "_mm512_min_ph",
			"ymm":    "_mm256_min_ph",
			"scalar": "_mm_min_sh",
		},
		MaxFn: map[string]string{
			"zmm":    "_mm512_max_ph",
			"ymm":    "_mm256_max_ph",
			"scalar": "_mm_max_sh",
		},
		DupFn: map[string]string{
			"zmm":    "_mm512_set1_ph",
			"ymm":    "_mm256_set1_ph",
			"scalar": "_mm_set_sh",
		},
		GetLaneFn: map[string]string{
			"zmm":    "_mm_cvtsh_ss + _mm_cvtss_f32",
			"ymm":    "_mm_cvtsh_ss + _mm_cvtss_f32",
			"scalar": "_mm_cvtsh_ss + _mm_cvtss_f32",
		},

		MathStrategy:   "native",
		FmaArgOrder:    "acc_last",
		GoatTarget:     "amd64",
		GoatExtraFlags: []string{"-mavx512fp16", "-mavx512f", "-mavx512vl"},
	}
}

// ---------------------------------------------------------------------------
// AVX-512 bfloat16 (AVX-512 BF16 extension, Cooper Lake+ / Zen4+)
// ---------------------------------------------------------------------------
// AVX-512 BF16 does NOT provide native BF16 arithmetic (add, sub, mul, div).
// It provides:
//   - VCVTNEPS2BF16: Convert F32 to BF16 (round to nearest even)
//   - VCVTNE2PS2BF16: Convert two F32 vectors to one BF16 vector
//   - VDPBF16PS: BF16 dot product with F32 accumulator (key for ML)
//
// For general arithmetic, the promote -> F32 compute -> demote pattern is used.
// Storage type is __m512bh / __m256bh, compute type is __m512 / __m256.

func avx512BF16Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "hwy.BFloat16",
		TargetName: "AVX512",
		Include:    "#include <immintrin.h>",
		CType:      "unsigned short",
		VecTypes: map[string]string{
			"zmm":    "__m256i",  // Storage: 16 x bf16 in YMM-width (256 bits)
			"wide":   "__m512",   // Promoted f32 vector for math computations
			"scalar": "__m256i",
		},
		Tiers: []CLoopTier{
			{Name: "zmm", Lanes: 16, Unroll: 4, IsScalar: false},
			{Name: "zmm", Lanes: 16, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn: map[string]string{
			"zmm":    "_mm256_loadu_si256",  // Load 16 bf16 as __m256i
			"scalar": "*(unsigned short*)",
		},
		StoreFn: map[string]string{
			"zmm":    "_mm256_storeu_si256",  // Store 16 bf16 as __m256i
			"scalar": "*(unsigned short*)",
		},
		// BF16 arithmetic uses inline helpers that handle promote-compute-demote.
		// Each helper takes/returns __m256i (16 bf16), so they compose correctly.
		AddFn:  map[string]string{"zmm": "avx512_bf16_add"},
		SubFn:  map[string]string{"zmm": "avx512_bf16_sub"},
		MulFn:  map[string]string{"zmm": "avx512_bf16_mul"},
		DivFn:  map[string]string{"zmm": "avx512_bf16_div"},
		FmaFn:  map[string]string{"zmm": "avx512_bf16_fma"},
		NegFn:  map[string]string{"zmm": "avx512_bf16_neg"},
		AbsFn:  map[string]string{"zmm": "avx512_bf16_abs"},
		SqrtFn: map[string]string{"zmm": "avx512_bf16_sqrt"},
		MinFn:  map[string]string{"zmm": "avx512_bf16_min"},
		MaxFn:  map[string]string{"zmm": "avx512_bf16_max"},
		DupFn:  map[string]string{"zmm": "avx512_bf16_dup"},
		GetLaneFn: map[string]string{
			"zmm": "_mm_cvtss_f32(_mm256_castps256_ps128(x))",
		},

		ReduceSumFn: map[string]string{"zmm": "avx512_bf16_reducesum"},
		ReduceMinFn: map[string]string{"zmm": "avx512_bf16_reducemin"},
		ReduceMaxFn: map[string]string{"zmm": "avx512_bf16_reducemax"},

		LessThanFn:     map[string]string{"zmm": "avx512_bf16_lt"},
		EqualFn:        map[string]string{"zmm": "avx512_bf16_eq"},
		GreaterThanFn:  map[string]string{"zmm": "avx512_bf16_gt"},
		GreaterEqualFn: map[string]string{"zmm": "avx512_bf16_ge"},
		IfThenElseFn:   map[string]string{"zmm": "avx512_bf16_ifelse"},
		MaskType:       map[string]string{"zmm": "__mmask16"},

		DotAccFn:   map[string]string{"zmm": "_mm512_dpbf16_ps"},
		DotAccType: map[string]string{"zmm": "__m512"},

		ScalarArithType:  "float",
		PointerElemType:  "unsigned short", // BF16 elements are 2 bytes, not 4 (float)
		ScalarPromote:    "bf16_scalar_to_f32",
		ScalarDemote:     "f32_scalar_to_bf16",
		InlineHelpers:    avx512BF16ArithHelpers,
		MathStrategy:     "promoted",
		NativeArithmetic: true, // Inline helpers provide full SIMD arithmetic via promote-compute-demote
		PromoteFn:        "avx512_bf16_promote",
		DemoteFn:         "avx512_bf16_demote(%s)",
		CastExpr:         "(__m256i*)",
		FmaArgOrder:      "acc_last",
		GoatTarget:       "amd64",
		GoatExtraFlags:   []string{"-mavx512bf16", "-mavx512f", "-mavx512vl"},
	}
}

// ---------------------------------------------------------------------------
// NEON uint64 (for RaBitQ bit product)
// ---------------------------------------------------------------------------

func neonUint64Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "uint64",
		TargetName: "NEON",
		Include:    "#include <arm_neon.h>",
		CType:      "unsigned long",
		VecTypes: map[string]string{
			"q": "uint64x2_t",
		},
		Tiers: []CLoopTier{
			{Name: "q", Lanes: 2, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn:    map[string]string{"q": "vld1q_u64"},
		StoreFn:   map[string]string{"q": "vst1q_u64"},
		AddFn:     map[string]string{"q": "vaddq_u64"},
		DupFn:     map[string]string{"q": "vdupq_n_u64"},
		Load4Fn:   map[string]string{"q": "vld1q_u64_x4"},
		VecX4Type: map[string]string{"q": "uint64x2x4_t"},

		SlideUpExtFn: map[string]string{"q": "vextq_u64"},
		SubFn:        map[string]string{"q": "vsubq_u64"},
		MinFn:        map[string]string{"q": "hwy_min_u64"},
		MaxFn:        map[string]string{"q": "hwy_max_u64"},
		AndFn:        map[string]string{"q": "vandq_u64"},
		OrFn:         map[string]string{"q": "vorrq_u64"},
		XorFn:        map[string]string{"q": "veorq_u64"},
		PopCountFn:   map[string]string{"q": "neon_popcnt_u64"},
		GetLaneFn:    map[string]string{"q": "vgetq_lane_u64"},

		ReduceSumFn: map[string]string{"q": "vaddvq_u64"},
		ReduceMinFn: map[string]string{"q": "hwy_reducemin_u64"},
		ReduceMaxFn: map[string]string{"q": "hwy_reducemax_u64"},

		InterleaveLowerFn: map[string]string{"q": "vzip1q_u64"},
		InterleaveUpperFn: map[string]string{"q": "vzip2q_u64"},

		// Deferred popcount accumulation: accumulate at uint32x4_t width
		// inside the loop, reduce once after the loop with vaddvq_u32.
		PopCountPartialFn: map[string]string{"q": "neon_popcnt_u64_to_u32"},
		AccVecType:        map[string]string{"q": "uint32x4_t"},
		AccAddFn:          map[string]string{"q": "vaddq_u32"},
		AccReduceFn:       map[string]string{"q": "vaddvq_u32"},

		EqualFn:       map[string]string{"q": "vceqq_u64"},
		LessThanFn:    map[string]string{"q": "vcltq_u64"},
		GreaterThanFn: map[string]string{"q": "vcgtq_u64"},
		GreaterEqualFn: map[string]string{"q": "vcgeq_u64"},
		IfThenElseFn:  map[string]string{"q": "vbslq_u64"},
		MaskType:      map[string]string{"q": "uint64x2_t"},

		MaskAndFn:    map[string]string{"q": "vandq_u64"},
		MaskOrFn:     map[string]string{"q": "vorrq_u64"},
		MaskAndNotFn: map[string]string{"q": "vbicq_u64"},

		AllTrueFn:       map[string]string{"q": "hwy_all_true_u64"},
		AllFalseFn:      map[string]string{"q": "hwy_all_false_u64"},
		FindFirstTrueFn: map[string]string{"q": "hwy_find_first_true_u64"},
		CountTrueFn:     map[string]string{"q": "hwy_count_true_u64"},
		FirstNFn:        map[string]string{"q": "hwy_first_n_u64"},
		IotaFn:          map[string]string{"q": "hwy_iota_u64"},
		CompressStoreFn: map[string]string{"q": "hwy_compress_store_u64"},

		MathStrategy:   "native",
		GoatTarget:     "arm64",
		GoatExtraFlags: []string{"-march=armv8-a+simd+fp"},

		InlineHelpers: slices.Concat(neonU64MaskHelpers, neonU64MaxMinHelpers, []string{
			`static inline uint64x2_t neon_popcnt_u64(uint64x2_t v) {
    uint8x16_t bytes = vreinterpretq_u8_u64(v);
    uint8x16_t counts = vcntq_u8(bytes);
    uint16x8_t pairs = vpaddlq_u8(counts);
    uint32x4_t quads = vpaddlq_u16(pairs);
    uint64x2_t result = vpaddlq_u32(quads);
    return result;
}`,
			`static inline uint32x4_t neon_popcnt_u64_to_u32(uint64x2_t v) {
    uint8x16_t bytes = vreinterpretq_u8_u64(v);
    uint8x16_t counts = vcntq_u8(bytes);
    uint16x8_t pairs = vpaddlq_u8(counts);
    uint32x4_t quads = vpaddlq_u16(pairs);
    return quads;
}`,
		}),
	}
}

// ---------------------------------------------------------------------------
// NEON uint8 (for varint boundary detection)
// ---------------------------------------------------------------------------

func neonUint8Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "uint8",
		TargetName: "NEON",
		Include:    "#include <arm_neon.h>",
		CType:      "unsigned char",
		VecTypes: map[string]string{
			"q": "uint8x16_t",
		},
		Tiers: []CLoopTier{
			{Name: "q", Lanes: 16, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn:    map[string]string{"q": "vld1q_u8"},
		StoreFn:   map[string]string{"q": "vst1q_u8"},
		DupFn:     map[string]string{"q": "vdupq_n_u8"},
		Load4Fn:   map[string]string{"q": "vld1q_u8_x4"},
		VecX4Type: map[string]string{"q": "uint8x16x4_t"},

		SlideUpExtFn:       map[string]string{"q": "vextq_u8"},
		LessThanFn:         map[string]string{"q": "vcltq_u8"},
		BitsFromMaskFn:     map[string]string{"q": "neon_bits_from_mask_u8"},
		TableLookupBytesFn: map[string]string{"q": "vqtbl1q_u8"},
		MaskType:           map[string]string{"q": "uint8x16_t"},

		MathStrategy:   "native",
		GoatTarget:     "arm64",
		GoatExtraFlags: []string{"-march=armv8-a+simd+fp"},

		InlineHelpers: []string{
			`static inline unsigned int neon_bits_from_mask_u8(uint8x16_t v) {
    // Extract one bit per byte from a NEON mask vector using volatile stack spill.
    // v has 0xFF (true) or 0x00 (false) per byte lane.
    // This avoids static const data that GOAT may not relocate properly.
    volatile unsigned char tmp[16];
    vst1q_u8((unsigned char *)tmp, v);
    unsigned int mask = 0;
    int i;
    for (i = 0; i < 16; i++) {
        if (tmp[i]) mask |= (1u << i);
    }
    return mask;
}`,
		},
	}
}

// ---------------------------------------------------------------------------
// NEON uint32 (for RaBitQ code counts)
// ---------------------------------------------------------------------------

func neonUint32Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "uint32",
		TargetName: "NEON",
		Include:    "#include <arm_neon.h>",
		CType:      "unsigned int",
		VecTypes: map[string]string{
			"q": "uint32x4_t",
		},
		Tiers: []CLoopTier{
			{Name: "q", Lanes: 4, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn:    map[string]string{"q": "vld1q_u32"},
		StoreFn:   map[string]string{"q": "vst1q_u32"},
		AddFn:     map[string]string{"q": "vaddq_u32"},
		DupFn:     map[string]string{"q": "vdupq_n_u32"},
		Load4Fn:   map[string]string{"q": "vld1q_u32_x4"},
		VecX4Type: map[string]string{"q": "uint32x4x4_t"},

		SlideUpExtFn: map[string]string{"q": "vextq_u32"},
		SubFn:        map[string]string{"q": "vsubq_u32"},
		AndFn:        map[string]string{"q": "vandq_u32"},
		OrFn:         map[string]string{"q": "vorrq_u32"},
		XorFn:        map[string]string{"q": "veorq_u32"},

		MinFn:   map[string]string{"q": "vminq_u32"},
		MaxFn:   map[string]string{"q": "vmaxq_u32"},

		ReduceSumFn: map[string]string{"q": "vaddvq_u32"},
		ReduceMinFn: map[string]string{"q": "vminvq_u32"},
		ReduceMaxFn: map[string]string{"q": "vmaxvq_u32"},

		LessThanFn:  map[string]string{"q": "vcltq_u32"},
		EqualFn:     map[string]string{"q": "vceqq_u32"},
		GreaterThanFn: map[string]string{"q": "vcgtq_u32"},
		GreaterEqualFn: map[string]string{"q": "vcgeq_u32"},
		IfThenElseFn: map[string]string{"q": "vbslq_u32"},
		MaskType:    map[string]string{"q": "uint32x4_t"},
		GetLaneFn:   map[string]string{"q": "vgetq_lane_u32"},

		InterleaveLowerFn: map[string]string{"q": "vzip1q_u32"},
		InterleaveUpperFn: map[string]string{"q": "vzip2q_u32"},

		TableLookupBytesFn: map[string]string{"q": "vqtbl1q_u8"},

		MaskAndFn:    map[string]string{"q": "vandq_u32"},
		MaskOrFn:     map[string]string{"q": "vorrq_u32"},
		MaskAndNotFn: map[string]string{"q": "vbicq_u32"},

		AllTrueFn:       map[string]string{"q": "hwy_all_true_u32"},
		AllFalseFn:      map[string]string{"q": "hwy_all_false_u32"},
		FindFirstTrueFn: map[string]string{"q": "hwy_find_first_true_u32"},
		CountTrueFn:     map[string]string{"q": "hwy_count_true_u32"},
		FirstNFn:        map[string]string{"q": "hwy_first_n_u32"},
		IotaFn:          map[string]string{"q": "hwy_iota_u32"},
		CompressStoreFn: map[string]string{"q": "hwy_compress_store_u32"},

		MathStrategy:   "native",
		GoatTarget:     "arm64",
		GoatExtraFlags: []string{"-march=armv8-a+simd+fp"},

		InlineHelpers: neonU32MaskHelpers,
	}
}

// ---------------------------------------------------------------------------
// NEON int32 (for RCT color transforms and other signed integer SIMD)
// ---------------------------------------------------------------------------

func neonInt32Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "int32",
		TargetName: "NEON",
		Include:    "#include <arm_neon.h>",
		CType:      "int",
		VecTypes: map[string]string{
			"q": "int32x4_t",
		},
		Tiers: []CLoopTier{
			{Name: "q", Lanes: 4, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn:  map[string]string{"q": "vld1q_s32"},
		StoreFn: map[string]string{"q": "vst1q_s32"},
		AddFn:   map[string]string{"q": "vaddq_s32"},
		SubFn:   map[string]string{"q": "vsubq_s32"},
		MulFn:   map[string]string{"q": "vmulq_s32"},
		NegFn:   map[string]string{"q": "vnegq_s32"},
		AbsFn:   map[string]string{"q": "vabsq_s32"},
		MinFn:   map[string]string{"q": "vminq_s32"},
		MaxFn:   map[string]string{"q": "vmaxq_s32"},
		DupFn:   map[string]string{"q": "vdupq_n_s32"},

		AndFn: map[string]string{"q": "vandq_s32"},
		OrFn:  map[string]string{"q": "vorrq_s32"},
		XorFn: map[string]string{"q": "veorq_s32"},

		ReduceSumFn: map[string]string{"q": "vaddvq_s32"},
		ReduceMinFn: map[string]string{"q": "vminvq_s32"},
		ReduceMaxFn: map[string]string{"q": "vmaxvq_s32"},

		InterleaveLowerFn: map[string]string{"q": "vzip1q_s32"},
		InterleaveUpperFn: map[string]string{"q": "vzip2q_s32"},

		EqualFn:       map[string]string{"q": "vceqq_s32"},
		LessThanFn:    map[string]string{"q": "vcltq_s32"},
		GreaterThanFn: map[string]string{"q": "vcgtq_s32"},
		GreaterEqualFn: map[string]string{"q": "vcgeq_s32"},
		IfThenElseFn:  map[string]string{"q": "vbslq_s32"},
		MaskType:      map[string]string{"q": "uint32x4_t"},

		MaskAndFn:    map[string]string{"q": "vandq_u32"},
		MaskOrFn:     map[string]string{"q": "vorrq_u32"},
		MaskAndNotFn: map[string]string{"q": "vbicq_u32"},

		AllTrueFn:       map[string]string{"q": "hwy_all_true_u32"},
		AllFalseFn:      map[string]string{"q": "hwy_all_false_u32"},
		FindFirstTrueFn: map[string]string{"q": "hwy_find_first_true_u32"},
		CountTrueFn:     map[string]string{"q": "hwy_count_true_u32"},
		FirstNFn:        map[string]string{"q": "hwy_first_n_u32"},
		IotaFn:          map[string]string{"q": "hwy_iota_s32"},
		CompressStoreFn: map[string]string{"q": "hwy_compress_store_s32"},

		MathStrategy:     "native",
		NativeArithmetic: true,
		GoatTarget:       "arm64",
		GoatExtraFlags:   []string{"-march=armv8-a+simd+fp"},

		InlineHelpers: neonS32MaskHelpers,
	}
}

// ---------------------------------------------------------------------------
// NEON int64 (for RCT color transforms with 64-bit signed integers)
// ---------------------------------------------------------------------------

func neonInt64Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "int64",
		TargetName: "NEON",
		Include:    "#include <arm_neon.h>",
		CType:      "long",
		VecTypes: map[string]string{
			"q": "int64x2_t",
		},
		Tiers: []CLoopTier{
			{Name: "q", Lanes: 2, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn:  map[string]string{"q": "vld1q_s64"},
		StoreFn: map[string]string{"q": "vst1q_s64"},
		AddFn:   map[string]string{"q": "vaddq_s64"},
		SubFn:   map[string]string{"q": "vsubq_s64"},
		NegFn:   map[string]string{"q": "vnegq_s64"},
		AbsFn:   map[string]string{"q": "vabsq_s64"},
		MinFn:   map[string]string{"q": "hwy_min_s64"},
		MaxFn:   map[string]string{"q": "hwy_max_s64"},
		DupFn:   map[string]string{"q": "vdupq_n_s64"},

		AndFn: map[string]string{"q": "vandq_s64"},
		OrFn:  map[string]string{"q": "vorrq_s64"},
		XorFn: map[string]string{"q": "veorq_s64"},

		ReduceSumFn: map[string]string{"q": "vaddvq_s64"},
		ReduceMinFn: map[string]string{"q": "hwy_reducemin_s64"},
		ReduceMaxFn: map[string]string{"q": "hwy_reducemax_s64"},

		InterleaveLowerFn: map[string]string{"q": "vzip1q_s64"},
		InterleaveUpperFn: map[string]string{"q": "vzip2q_s64"},

		EqualFn:       map[string]string{"q": "vceqq_s64"},
		LessThanFn:    map[string]string{"q": "vcltq_s64"},
		GreaterThanFn: map[string]string{"q": "vcgtq_s64"},
		GreaterEqualFn: map[string]string{"q": "vcgeq_s64"},
		IfThenElseFn:  map[string]string{"q": "vbslq_s64"},
		MaskType:      map[string]string{"q": "uint64x2_t"},

		MaskAndFn:    map[string]string{"q": "vandq_u64"},
		MaskOrFn:     map[string]string{"q": "vorrq_u64"},
		MaskAndNotFn: map[string]string{"q": "vbicq_u64"},

		AllTrueFn:       map[string]string{"q": "hwy_all_true_u64"},
		AllFalseFn:      map[string]string{"q": "hwy_all_false_u64"},
		FindFirstTrueFn: map[string]string{"q": "hwy_find_first_true_u64"},
		CountTrueFn:     map[string]string{"q": "hwy_count_true_u64"},
		FirstNFn:        map[string]string{"q": "hwy_first_n_u64"},
		IotaFn:          map[string]string{"q": "hwy_iota_s64"},
		CompressStoreFn: map[string]string{"q": "hwy_compress_store_s64"},

		MathStrategy:     "native",
		NativeArithmetic: true,
		GoatTarget:       "arm64",
		GoatExtraFlags:   []string{"-march=armv8-a+simd+fp"},

		InlineHelpers: slices.Concat(neonS64MaskHelpers, neonS64MaxMinHelpers),
	}
}

// ---------------------------------------------------------------------------
// SVE darwin float32 (Apple M4 via SME streaming mode, hardcoded SVL=512)
// ---------------------------------------------------------------------------
// On macOS M4+, SVE instructions execute in __arm_streaming mode (via SME).
// The streaming vector length (SVL) is fixed at 512 bits = 16 f32 lanes.
// Every SVE intrinsic except dup and reinterpret requires an svbool_t predicate.

func sveDarwinF32Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "float32",
		TargetName: "SVE_DARWIN",
		Include:    "#include <arm_sve.h>",
		CType:      "float",
		VecTypes: map[string]string{
			"sve":    "svfloat32_t",
			"scalar": "svfloat32_t",
		},
		Tiers: []CLoopTier{
			{Name: "sve", Lanes: 16, Unroll: 4, IsScalar: false},
			{Name: "sve", Lanes: 16, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn:    map[string]string{"sve": "svld1_f32"},
		StoreFn:   map[string]string{"sve": "svst1_f32"},
		AddFn:     map[string]string{"sve": "svadd_f32_x"},
		SubFn:     map[string]string{"sve": "svsub_f32_x"},
		MulFn:     map[string]string{"sve": "svmul_f32_x"},
		DivFn:     map[string]string{"sve": "svdiv_f32_x"},
		FmaFn:     map[string]string{"sve": "svmla_f32_x"},
		NegFn:     map[string]string{"sve": "svneg_f32_x"},
		AbsFn:     map[string]string{"sve": "svabs_f32_x"},
		SqrtFn:    map[string]string{"sve": "svsqrt_f32_x"},
		MinFn:     map[string]string{"sve": "svmin_f32_x"},
		MaxFn:     map[string]string{"sve": "svmax_f32_x"},
		DupFn:     map[string]string{"sve": "svdup_f32"},
		GetLaneFn: map[string]string{"sve": "svlasta_f32"},

		ReduceSumFn:       map[string]string{"sve": "svaddv_f32"},
		ReduceMinFn:       map[string]string{"sve": "svminv_f32"},
		ReduceMaxFn:       map[string]string{"sve": "svmaxv_f32"},
		InterleaveLowerFn: map[string]string{"sve": "svzip1_f32"},
		InterleaveUpperFn: map[string]string{"sve": "svzip2_f32"},
		LessThanFn:        map[string]string{"sve": "svcmplt_f32"},
		EqualFn:           map[string]string{"sve": "svcmpeq_f32"},
		GreaterThanFn:     map[string]string{"sve": "svcmpgt_f32"},
		GreaterEqualFn:    map[string]string{"sve": "svcmpge_f32"},
		IfThenElseFn:      map[string]string{"sve": "svsel_f32"},
		MaskType:          map[string]string{"sve": "svbool_t"},

		MathStrategy:     "native",
		NativeArithmetic: true,
		FmaArgOrder:      "acc_first",
		GoatTarget:       "arm64",
		GoatExtraFlags:   []string{"-march=armv9-a+sme"},
		NeedsPredicate:   true,
		PredicateDecl:    "svptrue_b32()",
		// NOTE: Do NOT use __arm_streaming here. Clang places smstart after
		// early-exit branches, creating paths where SVE instructions execute
		// outside streaming mode. Instead, GOAT injects smstart/smstop
		// conservatively before the first SVE instruction on each code path.
	}
}

// ---------------------------------------------------------------------------
// SVE darwin float64 (Apple M4 via SME streaming mode, hardcoded SVL=512)
// ---------------------------------------------------------------------------

func sveDarwinF64Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "float64",
		TargetName: "SVE_DARWIN",
		Include:    "#include <arm_sve.h>",
		CType:      "double",
		VecTypes: map[string]string{
			"sve":    "svfloat64_t",
			"scalar": "svfloat64_t",
		},
		Tiers: []CLoopTier{
			{Name: "sve", Lanes: 8, Unroll: 4, IsScalar: false},
			{Name: "sve", Lanes: 8, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn:    map[string]string{"sve": "svld1_f64"},
		StoreFn:   map[string]string{"sve": "svst1_f64"},
		AddFn:     map[string]string{"sve": "svadd_f64_x"},
		SubFn:     map[string]string{"sve": "svsub_f64_x"},
		MulFn:     map[string]string{"sve": "svmul_f64_x"},
		DivFn:     map[string]string{"sve": "svdiv_f64_x"},
		FmaFn:     map[string]string{"sve": "svmla_f64_x"},
		NegFn:     map[string]string{"sve": "svneg_f64_x"},
		AbsFn:     map[string]string{"sve": "svabs_f64_x"},
		SqrtFn:    map[string]string{"sve": "svsqrt_f64_x"},
		MinFn:     map[string]string{"sve": "svmin_f64_x"},
		MaxFn:     map[string]string{"sve": "svmax_f64_x"},
		DupFn:     map[string]string{"sve": "svdup_f64"},
		GetLaneFn: map[string]string{"sve": "svlasta_f64"},

		ReduceSumFn:       map[string]string{"sve": "svaddv_f64"},
		ReduceMinFn:       map[string]string{"sve": "svminv_f64"},
		ReduceMaxFn:       map[string]string{"sve": "svmaxv_f64"},
		InterleaveLowerFn: map[string]string{"sve": "svzip1_f64"},
		InterleaveUpperFn: map[string]string{"sve": "svzip2_f64"},
		LessThanFn:        map[string]string{"sve": "svcmplt_f64"},
		EqualFn:           map[string]string{"sve": "svcmpeq_f64"},
		GreaterThanFn:     map[string]string{"sve": "svcmpgt_f64"},
		GreaterEqualFn:    map[string]string{"sve": "svcmpge_f64"},
		IfThenElseFn:      map[string]string{"sve": "svsel_f64"},
		MaskType:          map[string]string{"sve": "svbool_t"},

		MathStrategy:     "native",
		NativeArithmetic: true,
		FmaArgOrder:      "acc_first",
		GoatTarget:       "arm64",
		GoatExtraFlags:   []string{"-march=armv9-a+sme"},
		NeedsPredicate:   true,
		PredicateDecl:    "svptrue_b64()",
		// NOTE: Do NOT use __arm_streaming here — see f32 profile comment.
	}
}

// ---------------------------------------------------------------------------
// SVE linux float32 (Graviton 3/4, Neoverse — native SVE, dynamic VL)
// ---------------------------------------------------------------------------
// On Linux ARM64 with SVE, the vector length is determined at runtime via
// svcntw() (f32) or svcntd() (f64). No streaming mode needed.

func sveLinuxF32Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "float32",
		TargetName: "SVE_LINUX",
		Include:    "#include <arm_sve.h>",
		CType:      "float",
		VecTypes: map[string]string{
			"sve":    "svfloat32_t",
			"scalar": "svfloat32_t",
		},
		Tiers: []CLoopTier{
			{Name: "sve", DynamicLanes: "svcntw()", Unroll: 4, IsScalar: false},
			{Name: "sve", DynamicLanes: "svcntw()", Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn:    map[string]string{"sve": "svld1_f32"},
		StoreFn:   map[string]string{"sve": "svst1_f32"},
		AddFn:     map[string]string{"sve": "svadd_f32_x"},
		SubFn:     map[string]string{"sve": "svsub_f32_x"},
		MulFn:     map[string]string{"sve": "svmul_f32_x"},
		DivFn:     map[string]string{"sve": "svdiv_f32_x"},
		FmaFn:     map[string]string{"sve": "svmla_f32_x"},
		NegFn:     map[string]string{"sve": "svneg_f32_x"},
		AbsFn:     map[string]string{"sve": "svabs_f32_x"},
		SqrtFn:    map[string]string{"sve": "svsqrt_f32_x"},
		MinFn:     map[string]string{"sve": "svmin_f32_x"},
		MaxFn:     map[string]string{"sve": "svmax_f32_x"},
		DupFn:     map[string]string{"sve": "svdup_f32"},
		GetLaneFn: map[string]string{"sve": "svlasta_f32"},

		ReduceSumFn:       map[string]string{"sve": "svaddv_f32"},
		ReduceMinFn:       map[string]string{"sve": "svminv_f32"},
		ReduceMaxFn:       map[string]string{"sve": "svmaxv_f32"},
		InterleaveLowerFn: map[string]string{"sve": "svzip1_f32"},
		InterleaveUpperFn: map[string]string{"sve": "svzip2_f32"},
		LessThanFn:        map[string]string{"sve": "svcmplt_f32"},
		EqualFn:           map[string]string{"sve": "svcmpeq_f32"},
		GreaterThanFn:     map[string]string{"sve": "svcmpgt_f32"},
		GreaterEqualFn:    map[string]string{"sve": "svcmpge_f32"},
		IfThenElseFn:      map[string]string{"sve": "svsel_f32"},
		MaskType:          map[string]string{"sve": "svbool_t"},

		MathStrategy:     "native",
		NativeArithmetic: true,
		FmaArgOrder:      "acc_first",
		GoatTarget:       "arm64",
		GoatExtraFlags:   []string{"-march=armv9-a+sve2"},
		NeedsPredicate:   true,
		PredicateDecl:    "svptrue_b32()",
	}
}

// ---------------------------------------------------------------------------
// SVE linux float64 (Graviton 3/4, Neoverse — native SVE, dynamic VL)
// ---------------------------------------------------------------------------

func sveLinuxF64Profile() *CIntrinsicProfile {
	return &CIntrinsicProfile{
		ElemType:   "float64",
		TargetName: "SVE_LINUX",
		Include:    "#include <arm_sve.h>",
		CType:      "double",
		VecTypes: map[string]string{
			"sve":    "svfloat64_t",
			"scalar": "svfloat64_t",
		},
		Tiers: []CLoopTier{
			{Name: "sve", DynamicLanes: "svcntd()", Unroll: 4, IsScalar: false},
			{Name: "sve", DynamicLanes: "svcntd()", Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn:    map[string]string{"sve": "svld1_f64"},
		StoreFn:   map[string]string{"sve": "svst1_f64"},
		AddFn:     map[string]string{"sve": "svadd_f64_x"},
		SubFn:     map[string]string{"sve": "svsub_f64_x"},
		MulFn:     map[string]string{"sve": "svmul_f64_x"},
		DivFn:     map[string]string{"sve": "svdiv_f64_x"},
		FmaFn:     map[string]string{"sve": "svmla_f64_x"},
		NegFn:     map[string]string{"sve": "svneg_f64_x"},
		AbsFn:     map[string]string{"sve": "svabs_f64_x"},
		SqrtFn:    map[string]string{"sve": "svsqrt_f64_x"},
		MinFn:     map[string]string{"sve": "svmin_f64_x"},
		MaxFn:     map[string]string{"sve": "svmax_f64_x"},
		DupFn:     map[string]string{"sve": "svdup_f64"},
		GetLaneFn: map[string]string{"sve": "svlasta_f64"},

		ReduceSumFn:       map[string]string{"sve": "svaddv_f64"},
		ReduceMinFn:       map[string]string{"sve": "svminv_f64"},
		ReduceMaxFn:       map[string]string{"sve": "svmaxv_f64"},
		InterleaveLowerFn: map[string]string{"sve": "svzip1_f64"},
		InterleaveUpperFn: map[string]string{"sve": "svzip2_f64"},
		LessThanFn:        map[string]string{"sve": "svcmplt_f64"},
		EqualFn:           map[string]string{"sve": "svcmpeq_f64"},
		GreaterThanFn:     map[string]string{"sve": "svcmpgt_f64"},
		GreaterEqualFn:    map[string]string{"sve": "svcmpge_f64"},
		IfThenElseFn:      map[string]string{"sve": "svsel_f64"},
		MaskType:          map[string]string{"sve": "svbool_t"},

		MathStrategy:     "native",
		NativeArithmetic: true,
		FmaArgOrder:      "acc_first",
		GoatTarget:       "arm64",
		GoatExtraFlags:   []string{"-march=armv9-a+sve2"},
		NeedsPredicate:   true,
		PredicateDecl:    "svptrue_b64()",
	}
}

// ---------------------------------------------------------------------------
// NEON BF16 arithmetic helpers (static inline, inlined by clang at -O3)
// ---------------------------------------------------------------------------
// BFloat16 has NO native SIMD arithmetic on NEON. All ops use:
//   promote bfloat16x8_t → 2×float32x4_t → compute → demote → bfloat16x8_t
//
// Promotion: shift left 16 (u16→u32 zero extend, then reinterpret as f32).
// Demotion: round-to-nearest-even bias, then narrow to u16.

var neonBF16ArithHelpers = []string{
	// --- Promote/Demote primitives ---
	`static inline float32x4_t bf16_promote_lo(bfloat16x8_t v) {
    uint16x4_t lo = vget_low_u16(vreinterpretq_u16_bf16(v));
    uint32x4_t wide = vshll_n_u16(lo, 16);
    return vreinterpretq_f32_u32(wide);
}`,
	`static inline float32x4_t bf16_promote_hi(bfloat16x8_t v) {
    uint16x4_t hi = vget_high_u16(vreinterpretq_u16_bf16(v));
    uint32x4_t wide = vshll_n_u16(hi, 16);
    return vreinterpretq_f32_u32(wide);
}`,
	// Demote f32→bf16 (4 lanes) with round-to-nearest-even.
	`static inline uint16x4_t bf16_demote_half(float32x4_t v) {
    uint32x4_t bits = vreinterpretq_u32_f32(v);
    uint32x4_t lsb = vshrq_n_u32(bits, 16);
    lsb = vandq_u32(lsb, vdupq_n_u32(1));
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7FFF), lsb);
    bits = vaddq_u32(bits, bias);
    return vshrn_n_u32(bits, 16);
}`,
	`static inline bfloat16x8_t bf16_combine(uint16x4_t lo, uint16x4_t hi) {
    return vreinterpretq_bf16_u16(vcombine_u16(lo, hi));
}`,

	// --- Binary ops (take/return bfloat16x8_t) ---
	`static inline bfloat16x8_t bf16_add_q(bfloat16x8_t a, bfloat16x8_t b) {
    float32x4_t a_lo = bf16_promote_lo(a), a_hi = bf16_promote_hi(a);
    float32x4_t b_lo = bf16_promote_lo(b), b_hi = bf16_promote_hi(b);
    return bf16_combine(bf16_demote_half(vaddq_f32(a_lo, b_lo)),
                        bf16_demote_half(vaddq_f32(a_hi, b_hi)));
}`,
	`static inline bfloat16x8_t bf16_sub_q(bfloat16x8_t a, bfloat16x8_t b) {
    float32x4_t a_lo = bf16_promote_lo(a), a_hi = bf16_promote_hi(a);
    float32x4_t b_lo = bf16_promote_lo(b), b_hi = bf16_promote_hi(b);
    return bf16_combine(bf16_demote_half(vsubq_f32(a_lo, b_lo)),
                        bf16_demote_half(vsubq_f32(a_hi, b_hi)));
}`,
	`static inline bfloat16x8_t bf16_mul_q(bfloat16x8_t a, bfloat16x8_t b) {
    float32x4_t a_lo = bf16_promote_lo(a), a_hi = bf16_promote_hi(a);
    float32x4_t b_lo = bf16_promote_lo(b), b_hi = bf16_promote_hi(b);
    return bf16_combine(bf16_demote_half(vmulq_f32(a_lo, b_lo)),
                        bf16_demote_half(vmulq_f32(a_hi, b_hi)));
}`,
	`static inline bfloat16x8_t bf16_div_q(bfloat16x8_t a, bfloat16x8_t b) {
    float32x4_t a_lo = bf16_promote_lo(a), a_hi = bf16_promote_hi(a);
    float32x4_t b_lo = bf16_promote_lo(b), b_hi = bf16_promote_hi(b);
    return bf16_combine(bf16_demote_half(vdivq_f32(a_lo, b_lo)),
                        bf16_demote_half(vdivq_f32(a_hi, b_hi)));
}`,
	`static inline bfloat16x8_t bf16_min_q(bfloat16x8_t a, bfloat16x8_t b) {
    float32x4_t a_lo = bf16_promote_lo(a), a_hi = bf16_promote_hi(a);
    float32x4_t b_lo = bf16_promote_lo(b), b_hi = bf16_promote_hi(b);
    return bf16_combine(bf16_demote_half(vminq_f32(a_lo, b_lo)),
                        bf16_demote_half(vminq_f32(a_hi, b_hi)));
}`,
	`static inline bfloat16x8_t bf16_max_q(bfloat16x8_t a, bfloat16x8_t b) {
    float32x4_t a_lo = bf16_promote_lo(a), a_hi = bf16_promote_hi(a);
    float32x4_t b_lo = bf16_promote_lo(b), b_hi = bf16_promote_hi(b);
    return bf16_combine(bf16_demote_half(vmaxq_f32(a_lo, b_lo)),
                        bf16_demote_half(vmaxq_f32(a_hi, b_hi)));
}`,

	// --- FMA: acc + a*b (acc_first convention) ---
	`static inline bfloat16x8_t bf16_fma_q(bfloat16x8_t acc, bfloat16x8_t a, bfloat16x8_t b) {
    float32x4_t acc_lo = bf16_promote_lo(acc), acc_hi = bf16_promote_hi(acc);
    float32x4_t a_lo = bf16_promote_lo(a), a_hi = bf16_promote_hi(a);
    float32x4_t b_lo = bf16_promote_lo(b), b_hi = bf16_promote_hi(b);
    return bf16_combine(bf16_demote_half(vfmaq_f32(acc_lo, a_lo, b_lo)),
                        bf16_demote_half(vfmaq_f32(acc_hi, a_hi, b_hi)));
}`,

	// --- Unary ops ---
	`static inline bfloat16x8_t bf16_neg_q(bfloat16x8_t v) {
    uint16x8_t bits = vreinterpretq_u16_bf16(v);
    bits = veorq_u16(bits, vdupq_n_u16(0x8000));
    return vreinterpretq_bf16_u16(bits);
}`,
	`static inline bfloat16x8_t bf16_abs_q(bfloat16x8_t v) {
    uint16x8_t bits = vreinterpretq_u16_bf16(v);
    bits = vandq_u16(bits, vdupq_n_u16(0x7FFF));
    return vreinterpretq_bf16_u16(bits);
}`,
	`static inline bfloat16x8_t bf16_sqrt_q(bfloat16x8_t v) {
    float32x4_t lo = bf16_promote_lo(v), hi = bf16_promote_hi(v);
    return bf16_combine(bf16_demote_half(vsqrtq_f32(lo)),
                        bf16_demote_half(vsqrtq_f32(hi)));
}`,

	// --- Broadcast / Zero ---
	`static inline bfloat16x8_t bf16_zero_q(void) {
    return vreinterpretq_bf16_u16(vdupq_n_u16(0));
}`,
	`static inline bfloat16x8_t bf16_dup_q(unsigned short val) {
    return vreinterpretq_bf16_u16(vdupq_n_u16(val));
}`,

	// --- Reductions (return float) ---
	`static inline float bf16_reducesum_q(bfloat16x8_t v) {
    float32x4_t lo = bf16_promote_lo(v);
    float32x4_t hi = bf16_promote_hi(v);
    return vaddvq_f32(vaddq_f32(lo, hi));
}`,
	`static inline float bf16_reducemin_q(bfloat16x8_t v) {
    float32x4_t lo = bf16_promote_lo(v);
    float32x4_t hi = bf16_promote_hi(v);
    return vminvq_f32(vminq_f32(lo, hi));
}`,
	`static inline float bf16_reducemax_q(bfloat16x8_t v) {
    float32x4_t lo = bf16_promote_lo(v);
    float32x4_t hi = bf16_promote_hi(v);
    return vmaxvq_f32(vmaxq_f32(lo, hi));
}`,

	// --- Comparisons (return uint16x8_t mask) ---
	`static inline uint16x8_t bf16_lt_q(bfloat16x8_t a, bfloat16x8_t b) {
    uint32x4_t m_lo = vcltq_f32(bf16_promote_lo(a), bf16_promote_lo(b));
    uint32x4_t m_hi = vcltq_f32(bf16_promote_hi(a), bf16_promote_hi(b));
    return vcombine_u16(vmovn_u32(m_lo), vmovn_u32(m_hi));
}`,
	`static inline uint16x8_t bf16_eq_q(bfloat16x8_t a, bfloat16x8_t b) {
    uint32x4_t m_lo = vceqq_f32(bf16_promote_lo(a), bf16_promote_lo(b));
    uint32x4_t m_hi = vceqq_f32(bf16_promote_hi(a), bf16_promote_hi(b));
    return vcombine_u16(vmovn_u32(m_lo), vmovn_u32(m_hi));
}`,
	`static inline uint16x8_t bf16_gt_q(bfloat16x8_t a, bfloat16x8_t b) {
    uint32x4_t m_lo = vcgtq_f32(bf16_promote_lo(a), bf16_promote_lo(b));
    uint32x4_t m_hi = vcgtq_f32(bf16_promote_hi(a), bf16_promote_hi(b));
    return vcombine_u16(vmovn_u32(m_lo), vmovn_u32(m_hi));
}`,
	`static inline uint16x8_t bf16_ge_q(bfloat16x8_t a, bfloat16x8_t b) {
    uint32x4_t m_lo = vcgeq_f32(bf16_promote_lo(a), bf16_promote_lo(b));
    uint32x4_t m_hi = vcgeq_f32(bf16_promote_hi(a), bf16_promote_hi(b));
    return vcombine_u16(vmovn_u32(m_lo), vmovn_u32(m_hi));
}`,
	// IfThenElse operates on bit patterns — no promotion needed.
	`static inline bfloat16x8_t bf16_ifelse_q(uint16x8_t mask, bfloat16x8_t yes, bfloat16x8_t no) {
    return vreinterpretq_bf16_u16(vbslq_u16(mask,
        vreinterpretq_u16_bf16(yes), vreinterpretq_u16_bf16(no)));
}`,

	// --- Scalar promote/demote (for scalar tail arithmetic) ---
	`static inline float bf16_scalar_to_f32(unsigned short v) {
    unsigned int bits = (unsigned int)v << 16;
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}`,
	`static inline unsigned short f32_scalar_to_bf16(float f) {
    unsigned int bits;
    __builtin_memcpy(&bits, &f, 4);
    unsigned int lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return (unsigned short)(bits >> 16);
}`,
}

// ---------------------------------------------------------------------------
// AVX-512 BF16 arithmetic helpers (static inline, inlined by clang at -O3)
// ---------------------------------------------------------------------------
// AVX-512 BF16 stores 16 bf16 values in a __m256i (256 bits).
// Compute uses __m512 (16 x float32). Hardware VCVTNEPS2BF16 for demotion.

var avx512BF16ArithHelpers = []string{
	// --- Promote/Demote primitives ---
	`static inline __m512 avx512_bf16_promote(__m256i v) {
    __m512i wide = _mm512_cvtepu16_epi32(v);
    wide = _mm512_slli_epi32(wide, 16);
    return _mm512_castsi512_ps(wide);
}`,
	// VCVTNEPS2BF16: hardware round-to-nearest-even demotion.
	`static inline __m256i avx512_bf16_demote(__m512 v) {
    return (__m256i)_mm512_cvtneps_pbh(v);
}`,

	// --- Binary ops (take/return __m256i) ---
	`static inline __m256i avx512_bf16_add(__m256i a, __m256i b) {
    return avx512_bf16_demote(_mm512_add_ps(avx512_bf16_promote(a), avx512_bf16_promote(b)));
}`,
	`static inline __m256i avx512_bf16_sub(__m256i a, __m256i b) {
    return avx512_bf16_demote(_mm512_sub_ps(avx512_bf16_promote(a), avx512_bf16_promote(b)));
}`,
	`static inline __m256i avx512_bf16_mul(__m256i a, __m256i b) {
    return avx512_bf16_demote(_mm512_mul_ps(avx512_bf16_promote(a), avx512_bf16_promote(b)));
}`,
	`static inline __m256i avx512_bf16_div(__m256i a, __m256i b) {
    return avx512_bf16_demote(_mm512_div_ps(avx512_bf16_promote(a), avx512_bf16_promote(b)));
}`,
	`static inline __m256i avx512_bf16_min(__m256i a, __m256i b) {
    return avx512_bf16_demote(_mm512_min_ps(avx512_bf16_promote(a), avx512_bf16_promote(b)));
}`,
	`static inline __m256i avx512_bf16_max(__m256i a, __m256i b) {
    return avx512_bf16_demote(_mm512_max_ps(avx512_bf16_promote(a), avx512_bf16_promote(b)));
}`,

	// --- FMA: a*b + acc (acc_last convention for AVX) ---
	`static inline __m256i avx512_bf16_fma(__m256i a, __m256i b, __m256i acc) {
    __m512 af = avx512_bf16_promote(a);
    __m512 bf = avx512_bf16_promote(b);
    __m512 cf = avx512_bf16_promote(acc);
    return avx512_bf16_demote(_mm512_fmadd_ps(af, bf, cf));
}`,

	// --- Unary ops ---
	`static inline __m256i avx512_bf16_neg(__m256i v) {
    return _mm256_xor_si256(v, _mm256_set1_epi16((short)0x8000));
}`,
	`static inline __m256i avx512_bf16_abs(__m256i v) {
    return _mm256_and_si256(v, _mm256_set1_epi16(0x7FFF));
}`,
	`static inline __m256i avx512_bf16_sqrt(__m256i v) {
    return avx512_bf16_demote(_mm512_sqrt_ps(avx512_bf16_promote(v)));
}`,

	// --- Broadcast / Zero ---
	`static inline __m256i avx512_bf16_zero(void) {
    return _mm256_setzero_si256();
}`,
	`static inline __m256i avx512_bf16_dup(unsigned short val) {
    return _mm256_set1_epi16((short)val);
}`,

	// --- Reductions (return float) ---
	`static inline float avx512_bf16_reducesum(__m256i v) {
    return _mm512_reduce_add_ps(avx512_bf16_promote(v));
}`,
	`static inline float avx512_bf16_reducemin(__m256i v) {
    return _mm512_reduce_min_ps(avx512_bf16_promote(v));
}`,
	`static inline float avx512_bf16_reducemax(__m256i v) {
    return _mm512_reduce_max_ps(avx512_bf16_promote(v));
}`,

	// --- Comparisons (return __mmask16) ---
	`static inline __mmask16 avx512_bf16_lt(__m256i a, __m256i b) {
    return _mm512_cmp_ps_mask(avx512_bf16_promote(a), avx512_bf16_promote(b), _CMP_LT_OQ);
}`,
	`static inline __mmask16 avx512_bf16_eq(__m256i a, __m256i b) {
    return _mm512_cmp_ps_mask(avx512_bf16_promote(a), avx512_bf16_promote(b), _CMP_EQ_OQ);
}`,
	`static inline __mmask16 avx512_bf16_gt(__m256i a, __m256i b) {
    return _mm512_cmp_ps_mask(avx512_bf16_promote(a), avx512_bf16_promote(b), _CMP_GT_OQ);
}`,
	// IfThenElse: mask-based blend on the u16 bit patterns.
	`static inline __m256i avx512_bf16_ifelse(__mmask16 mask, __m256i yes, __m256i no) {
    return _mm256_mask_blend_epi16(mask, no, yes);
}`,

	// --- Scalar promote/demote (for scalar tail arithmetic) ---
	// Same implementation as NEON — scalar bf16↔f32 is architecture-independent.
	`static inline float bf16_scalar_to_f32(unsigned short v) {
    unsigned int bits = (unsigned int)v << 16;
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}`,
	`static inline unsigned short f32_scalar_to_bf16(float f) {
    unsigned int bits;
    __builtin_memcpy(&bits, &f, 4);
    unsigned int lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return (unsigned short)(bits >> 16);
}`,
}
