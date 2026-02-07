package main

// CIntrinsicProfile defines the complete set of C intrinsics and metadata
// for a specific target architecture + element type combination.
// The CEmitter uses these profiles to generate correct GOAT-compatible C code
// for each supported SIMD target.
type CIntrinsicProfile struct {
	ElemType   string // "float32", "float64", "float16", "bfloat16"
	TargetName string // "NEON", "AVX2", "AVX512"
	Include    string // "#include <arm_neon.h>" or "#include <immintrin.h>"
	CType      string // "float", "double", "unsigned short"

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
	GreaterThanFn map[string]string // vcgtq_f32, _mm256_cmp_ps

	// Conditional select
	IfThenElseFn map[string]string // vbslq_f32, _mm256_blendv_ps

	// Mask extraction
	BitsFromMaskFn map[string]string // manual (NEON), _mm256_movemask_epi8

	// Mask vector type (comparison results)
	MaskType map[string]string // "uint32x4_t" (NEON), "__m256" (AVX)

	// Reduction min/max
	ReduceMinFn map[string]string // vminvq_f32
	ReduceMaxFn map[string]string // vmaxvq_f32

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
		IfThenElseFn:      map[string]string{"q": "vbslq_f32"},
		MaskType:          map[string]string{"q": "uint32x4_t"},
		ReduceMinFn:       map[string]string{"q": "vminvq_f32"},
		ReduceMaxFn:       map[string]string{"q": "vmaxvq_f32"},

		InlineHelpers: []string{
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
		},

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
		IfThenElseFn:      map[string]string{"q": "vbslq_f64"},
		MaskType:          map[string]string{"q": "uint64x2_t"},
		ReduceMinFn:       map[string]string{"q": "vminvq_f64"},
		ReduceMaxFn:       map[string]string{"q": "vmaxvq_f64"},

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
		CType:      "unsigned short",
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
		IfThenElseFn:      map[string]string{"q": "vbslq_u16", "d": "vbsl_u16"},
		MaskType:          map[string]string{"q": "uint16x8_t", "d": "uint16x4_t"},
		ReduceMinFn:       map[string]string{"q": "vminvq_f16"},
		ReduceMaxFn:       map[string]string{"q": "vmaxvq_f16"},

		NativeArithmetic: true,
		ScalarArithType:  "float16_t",
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
			"d":      "uint16x4_t",
			"wide":   "float32x4_t", // promoted f32 vector for math computations
			"half":   "uint16x4_t",  // demoted half for bf16 recombine
			"scalar": "uint16x4_t",
		},
		Tiers: []CLoopTier{
			{Name: "q", Lanes: 8, Unroll: 4, IsScalar: false},
			{Name: "q", Lanes: 8, Unroll: 1, IsScalar: false},
			{Name: "d", Lanes: 4, Unroll: 1, IsScalar: false},
			{Name: "scalar", Lanes: 1, Unroll: 1, IsScalar: true},
		},
		LoadFn: map[string]string{
			"q":      "vld1q_bf16",
			"d":      "vld1_u16",
			"scalar": "vld1_dup_u16",
		},
		StoreFn: map[string]string{
			"q":      "vst1q_bf16",
			"d":      "vst1_u16",
			"scalar": "vst1_lane_u16",
		},
		// BF16 has no native arithmetic intrinsics; the CEmitter must generate
		// inline promote-compute-demote sequences. These entries represent the
		// *compute* step which operates on promoted float32 values.
		AddFn:  map[string]string{"q": "vaddq_f32"},
		SubFn:  map[string]string{"q": "vsubq_f32"},
		MulFn:  map[string]string{"q": "vmulq_f32"},
		DivFn:  map[string]string{"q": "vdivq_f32"},
		FmaFn:  map[string]string{"q": "vfmaq_f32"},
		NegFn:  map[string]string{"q": "vnegq_f32"},
		AbsFn:  map[string]string{"q": "vabsq_f32"},
		SqrtFn: map[string]string{"q": "vsqrtq_f32"},
		MinFn:  map[string]string{"q": "vminq_f32"},
		MaxFn:  map[string]string{"q": "vmaxq_f32"},
		DupFn: map[string]string{
			"q":      "vld1q_dup_bf16",
			"d":      "vld1_dup_u16",
			"scalar": "vld1_dup_u16",
		},
		GetLaneFn: map[string]string{
			"q":      "vgetq_lane_bf16",
			"d":      "vget_lane_u16",
			"scalar": "vst1_lane_u16",
		},

		// Promote: vshll_n_u16(vget_low_u16(vreinterpretq_u16_bf16(x)), 16)
		//          then vreinterpretq_f32_u32
		// Demote:  round-to-nearest-even + vmovn_u32 + vcombine_u16
		//          then vreinterpretq_bf16_u16
		MathStrategy:   "promoted",
		PromoteFn:      "vshll_n_u16(..., 16) + vreinterpretq_f32_u32",
		DemoteFn:       "round_bias_vmovn_bf16(%s)", // placeholder: inline bf16 demote sequence
		SplitPromoteLo: "vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vreinterpretq_u16_bf16(%s)), 16))",  // %s = narrow vector variable
		SplitPromoteHi: "vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vreinterpretq_u16_bf16(%s)), 16))", // %s = narrow vector variable
		CombineFn:      "vreinterpretq_bf16_u16(vcombine_u16(%s, %s))",                                       // %s = lo half, %s = hi half
		CastExpr:       "(bfloat16_t*)",
		FmaArgOrder:    "acc_first",
		GoatTarget:     "arm64",
		GoatExtraFlags: []string{"-march=armv8.6-a+bf16+simd"},
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
			"zmm":         "__m512bh",  // Storage: 32 x bf16 in ZMM
			"zmm_compute": "__m512",    // Compute: 16 x f32 in ZMM
			"ymm":         "__m256bh",  // Storage: 16 x bf16 in YMM
			"ymm_compute": "__m256",    // Compute: 8 x f32 in YMM
			"wide":        "__m512",    // Promoted f32 vector for math computations
			"scalar":      "unsigned int",
		},
		Tiers: []CLoopTier{
			// Main loop processes 32 bf16 = 2x16 f32 after promotion
			{Name: "zmm", Lanes: 32, Unroll: 4, IsScalar: false},
			// Single ZMM processes 16 bf16 via one zmm of f32
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
		// All arithmetic is done after promotion to float32.
		AddFn:  map[string]string{"zmm": "_mm256_add_ps", "zmm_wide": "_mm512_add_ps"},
		SubFn:  map[string]string{"zmm": "_mm256_sub_ps", "zmm_wide": "_mm512_sub_ps"},
		MulFn:  map[string]string{"zmm": "_mm256_mul_ps", "zmm_wide": "_mm512_mul_ps"},
		DivFn:  map[string]string{"zmm": "_mm256_div_ps", "zmm_wide": "_mm512_div_ps"},
		FmaFn:  map[string]string{"zmm": "_mm256_fmadd_ps", "zmm_wide": "_mm512_fmadd_ps"},
		NegFn:  map[string]string{"zmm": "_mm256_sub_ps(zero, x)"},
		AbsFn:  map[string]string{"zmm": "_mm256_andnot_ps(signmask, x)"},
		SqrtFn: map[string]string{"zmm": "_mm256_sqrt_ps"},
		MinFn:  map[string]string{"zmm": "_mm256_min_ps"},
		MaxFn:  map[string]string{"zmm": "_mm256_max_ps"},
		DupFn:  map[string]string{"zmm": "_mm256_set1_ps"},
		GetLaneFn: map[string]string{
			"zmm": "_mm_cvtss_f32(_mm256_castps256_ps128(x))",
		},

		// Promote: unpack u16->u32 with zero extend, shift left 16, reinterpret as f32
		// Demote: _mm512_cvtneps_pbh (VCVTNEPS2BF16)
		MathStrategy:   "promoted",
		PromoteFn:      "unpacklo/hi_epi16 + slli_epi32(16)",
		DemoteFn:       "_mm512_cvtneps_pbh(%s)",
		FmaArgOrder:    "acc_last",
		GoatTarget:     "amd64",
		GoatExtraFlags: []string{"-mavx512bf16", "-mavx512f", "-mavx512vl"},
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
		AndFn:        map[string]string{"q": "vandq_u64"},
		OrFn:         map[string]string{"q": "vorrq_u64"},
		XorFn:        map[string]string{"q": "veorq_u64"},
		PopCountFn:   map[string]string{"q": "neon_popcnt_u64"},

		ReduceSumFn: map[string]string{"q": "vaddvq_u64"},

		InterleaveLowerFn: map[string]string{"q": "vzip1q_u64"},
		InterleaveUpperFn: map[string]string{"q": "vzip2q_u64"},

		// Deferred popcount accumulation: accumulate at uint32x4_t width
		// inside the loop, reduce once after the loop with vaddvq_u32.
		PopCountPartialFn: map[string]string{"q": "neon_popcnt_u64_to_u32"},
		AccVecType:        map[string]string{"q": "uint32x4_t"},
		AccAddFn:          map[string]string{"q": "vaddq_u32"},
		AccReduceFn:       map[string]string{"q": "vaddvq_u32"},

		MathStrategy:   "native",
		GoatTarget:     "arm64",
		GoatExtraFlags: []string{"-march=armv8-a+simd+fp"},

		InlineHelpers: []string{
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
		},
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
		AndFn:        map[string]string{"q": "vandq_u32"},
		OrFn:         map[string]string{"q": "vorrq_u32"},
		XorFn:        map[string]string{"q": "veorq_u32"},

		ReduceSumFn: map[string]string{"q": "vaddvq_u32"},
		LessThanFn:  map[string]string{"q": "vcltq_u32"},
		MaskType:    map[string]string{"q": "uint32x4_t"},

		InterleaveLowerFn: map[string]string{"q": "vzip1q_u32"},
		InterleaveUpperFn: map[string]string{"q": "vzip2q_u32"},

		MathStrategy:   "native",
		GoatTarget:     "arm64",
		GoatExtraFlags: []string{"-march=armv8-a+simd+fp"},
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

		MathStrategy:     "native",
		NativeArithmetic: true,
		GoatTarget:       "arm64",
		GoatExtraFlags:   []string{"-march=armv8-a+simd+fp"},
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
		DupFn:   map[string]string{"q": "vdupq_n_s64"},

		AndFn: map[string]string{"q": "vandq_s64"},
		OrFn:  map[string]string{"q": "vorrq_s64"},
		XorFn: map[string]string{"q": "veorq_s64"},

		ReduceSumFn: map[string]string{"q": "vaddvq_s64"},

		InterleaveLowerFn: map[string]string{"q": "vzip1q_s64"},
		InterleaveUpperFn: map[string]string{"q": "vzip2q_s64"},

		MathStrategy:     "native",
		NativeArithmetic: true,
		GoatTarget:       "arm64",
		GoatExtraFlags:   []string{"-march=armv8-a+simd+fp"},
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
