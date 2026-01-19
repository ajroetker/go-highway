# Full-Text Search SIMD Primitives

This document specifies SIMD primitives needed to accelerate full-text search engines, based on analysis of the [zapx](https://github.com/blevesearch/zapx) indexing library (used by Bleve).

## Motivation

Full-text search engines like Bleve/zapx perform millions of integer operations during query execution and index maintenance:

- **Varint decoding**: Every posting list entry, term frequency, and location is varint-encoded
- **Norm calculations**: BM25/TF-IDF scoring requires `1/sqrt(x)` per document
- **Document remapping**: Segment merges require array lookups for docID translation
- **Bitmap operations**: Roaring bitmaps for posting lists and result filtering

These operations are currently scalar and represent significant optimization opportunities.

---

## Target Use Cases from Zapx

### 1. Varint Decoding (Critical Hot Path)

**Source**: `zapx/memuvarint.go:41-77`, `zapx/intDecoder.go:140-210`

Current scalar implementation:
```go
func (r *memUvarintReader) ReadUvarint() (uint64, error) {
    var x uint64
    var s uint
    for {
        b := S[C]
        C++
        if b < 0x80 {
            return x | uint64(b)<<s, nil
        }
        x |= uint64(b&0x7f) << s
        s += 7
    }
}
```

**Call frequency**: Millions per query (every posting, freq, norm, location)

### 2. Norm/Score Calculations

**Source**: `zapx/posting.go:891-893`

```go
func (p *Posting) Norm() float64 {
    return float64(float32(1.0 / math.Sqrt(float64(math.Float32bits(p.norm)))))
}
```

**Call frequency**: Once per matching document during scoring

### 3. Document Number Remapping

**Source**: `zapx/merge.go:214, 250`

```go
hitNewDocNum := newDocNums[nextDocNum]  // uint64 array lookup
```

**Call frequency**: Every posting during segment merge

### 4. Chunk Offset Decoding

**Source**: `zapx/docvalues.go:140-149`, `zapx/intDecoder.go:56-57`

Sequential varint reads into arrays:
```go
for i := 0; i < int(numChunks); i++ {
    loc, read := binary.Uvarint(mem[offset:])
    chunkOffsets[i] = loc
    offset += uint64(read)
}
```

---

## Proposed Primitives

### Priority 1: RSqrt (Reciprocal Square Root)

**Rationale**: Direct hardware support via `VRSQRTPS` (AVX) and `FRSQRTE` (NEON). Fundamental operation for scoring.

#### API

```go
// hwy/ops_avx2.go

// RSqrt_AVX2_F32x8 computes approximate 1/sqrt(x) for 8 float32 values.
// Uses VRSQRTPS instruction (~12-bit precision).
// For values where x <= 0, result is undefined.
func RSqrt_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8

// RSqrtPrecise_AVX2_F32x8 computes precise 1/sqrt(x) via sqrt + div.
// Uses VSQRTPS + VDIVPS for full precision.
func RSqrtPrecise_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8

// RSqrtNewtonRaphson_AVX2_F32x8 computes 1/sqrt(x) with one Newton-Raphson refinement.
// Provides ~23-bit precision (sufficient for float32).
func RSqrtNewtonRaphson_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8
```

#### Implementation

```go
// Fast approximate (VRSQRTPS)
func RSqrt_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
    return x.ApproxRecipSqrt()  // Maps to VRSQRTPS
}

// Newton-Raphson refinement: y = y * (1.5 - 0.5 * x * y * y)
func RSqrtNewtonRaphson_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
    half := archsimd.BroadcastFloat32x8(0.5)
    threeHalf := archsimd.BroadcastFloat32x8(1.5)

    y := x.ApproxRecipSqrt()

    // One iteration: y = y * (1.5 - 0.5 * x * y * y)
    xHalf := x.Mul(half)
    yy := y.Mul(y)
    xyy := xHalf.Mul(yy)
    correction := threeHalf.Sub(xyy)
    return y.Mul(correction)
}

// Precise via sqrt + reciprocal
func RSqrtPrecise_AVX2_F32x8(x archsimd.Float32x8) archsimd.Float32x8 {
    one := archsimd.BroadcastFloat32x8(1.0)
    sqrtX := x.Sqrt()
    return one.Div(sqrtX)
}
```

#### Variants Needed

| Function | Target | Precision | Latency |
|----------|--------|-----------|---------|
| `RSqrt_AVX2_F32x8` | AVX2 | ~12-bit | Low |
| `RSqrtNewtonRaphson_AVX2_F32x8` | AVX2 | ~23-bit | Medium |
| `RSqrtPrecise_AVX2_F32x8` | AVX2 | Full | High |
| `RSqrt_AVX512_F32x16` | AVX-512 | ~12-bit | Low |
| `RSqrt_NEON_F32x4` | NEON | ~8-bit | Low |
| `RSqrt_F64x*` variants | All | Various | Various |

---

### Priority 2: Hardware Gather

**Rationale**: Current gather implementations are scalar loops. AVX2 has `VPGATHERDPS`/`VPGATHERDPD` instructions.

#### Current State

```go
// Current implementation (scalar fallback)
func GatherIndex_AVX2_F32x8(src []float32, indices archsimd.Int32x8) archsimd.Float32x8 {
    var idxData [8]int32
    indices.StoreSlice(idxData[:])
    var result [8]float32
    for i := 0; i < 8; i++ {
        idx := int(idxData[i])
        if idx >= 0 && idx < len(src) {
            result[i] = src[idx]
        }
    }
    return archsimd.LoadFloat32x8Slice(result[:])
}
```

#### Proposed Upgrade

```go
// Use actual VPGATHERDPS instruction
func GatherIndex_AVX2_F32x8(src []float32, indices archsimd.Int32x8) archsimd.Float32x8 {
    // TODO: Use archsimd.GatherFloat32x8 when available
    // Maps to: VPGATHERDPS ymm, [base + ymm*4], ymm_mask
    return archsimd.GatherFloat32x8(unsafe.Pointer(&src[0]), indices)
}

// For uint64 gather (docNum remapping)
func GatherIndex_AVX2_U64x4(src []uint64, indices archsimd.Int32x4) archsimd.Uint64x4
func GatherIndex_AVX512_U64x8(src []uint64, indices archsimd.Int32x8) archsimd.Uint64x8
```

#### Gather Performance Notes

| Instruction | Target | Throughput | Use Case |
|-------------|--------|------------|----------|
| VPGATHERDPS | AVX2 | ~5 cycles | Float32 gather |
| VPGATHERDD | AVX2 | ~5 cycles | Int32 gather |
| VPGATHERDQ | AVX2 | ~5 cycles | Int64 with Int32 indices |
| VPGATHERQQ | AVX2 | ~5 cycles | Int64 with Int64 indices |
| VPGATHER* | AVX-512 | ~3 cycles | Better throughput |

**Note**: Gather performance depends heavily on cache locality. Random access patterns may not benefit.

---

### Priority 3: Prefix Sum (Parallel Scan)

**Rationale**: Required for efficient delta decoding. Current `DeltaDecode` is scalar due to sequential dependency.

#### API

```go
// hwy/contrib/algo/prefix_sum.go

// PrefixSum computes inclusive prefix sum: dst[i] = sum(src[0:i+1])
func PrefixSum[T hwy.Integers | hwy.Floats](src []T) []T

// PrefixSumExclusive computes exclusive prefix sum: dst[i] = sum(src[0:i])
func PrefixSumExclusive[T hwy.Integers | hwy.Floats](src []T) []T
```

#### Low-Level Implementation

```go
// Within a single vector (Hillis-Steele algorithm)
func PrefixSum_AVX2_I64x4(v archsimd.Int64x4) archsimd.Int64x4 {
    // Step 1: shift by 1 and add
    // [a, b, c, d] -> [a, a+b, b+c, c+d]
    shifted1 := ShiftLeftLanes_1(v)  // [0, a, b, c]
    sum1 := v.Add(shifted1)          // [a, a+b, b+c, c+d]

    // Step 2: shift by 2 and add
    // [a, a+b, b+c, c+d] -> [a, a+b, a+b+c, a+b+c+d]
    shifted2 := ShiftLeftLanes_2(sum1)  // [0, 0, a, a+b]
    sum2 := sum1.Add(shifted2)           // [a, a+b, a+b+c, a+b+c+d]

    return sum2
}
```

#### Use Case: Accelerated Delta Decode

```go
// Current (scalar, sequential)
func DeltaDecode[T hwy.UnsignedInts](src []T, base T, dst []T) {
    current := base
    for i := range src {
        current += src[i]
        dst[i] = current
    }
}

// Proposed (SIMD-accelerated)
func DeltaDecodeSIMD[T hwy.UnsignedInts](src []T, base T, dst []T) {
    lanes := hwy.Zero[T]().NumLanes()

    // Process in blocks, using prefix sum within each block
    carry := base
    for i := 0; i+lanes <= len(src); i += lanes {
        block := hwy.Load(src[i:])
        prefixed := PrefixSum(block)
        prefixed = hwy.Add(prefixed, hwy.Set(carry))
        hwy.Store(prefixed, dst[i:])
        carry = hwy.GetLane(prefixed, lanes-1)
    }
    // Handle tail...
}
```

---

### Priority 4: Varint Codec (contrib/varint)

**Rationale**: Highest impact for zapx but most complex. Variable-length encoding is inherently serial, but batch approaches exist.

#### Approach 1: Standard Varint with Batch Boundary Detection

Detect continuation bytes in parallel, then decode serially with known lengths.

```go
// hwy/contrib/varint/varint.go

// DecodeUvarint64 decodes up to n varint-encoded uint64 values.
// Returns (values decoded, bytes consumed).
func DecodeUvarint64(src []byte, dst []uint64, n int) (int, int)

// DecodeUvarint64Batch decodes exactly 4 varints in parallel when possible.
// Requires src to contain at least 4 complete varints.
// Returns bytes consumed.
func DecodeUvarint64Batch(src []byte, dst *[4]uint64) int
```

#### Approach 2: Group Varint (SIMD-Friendly)

Store 4 values with a single control byte indicating byte lengths.

```go
// GroupVarint format:
// [control byte][value0 bytes][value1 bytes][value2 bytes][value3 bytes]
// Control byte: 2 bits per value indicating length (1-4 bytes each)

// DecodeGroupVarint32 decodes 4 uint32 values from group-varint format.
func DecodeGroupVarint32(src []byte, dst *[4]uint32) int

// EncodeGroupVarint32 encodes 4 uint32 values to group-varint format.
func EncodeGroupVarint32(src [4]uint32, dst []byte) int
```

#### Approach 3: Stream-VByte (Optimized for Sequential Access)

Separate control stream from data stream for better SIMD utilization.

```go
// StreamVByte format:
// [control bytes...][data bytes...]
// Control: 2 bits per value (4 values per control byte)

type StreamVByteDecoder struct {
    control []byte
    data    []byte
    pos     int
}

func (d *StreamVByteDecoder) Decode4(dst *[4]uint32) int
```

#### Implementation Sketch: Batch Boundary Detection

```go
func findVarintBoundaries(src []byte) (lengths [8]int) {
    // Load 64 bytes
    v := archsimd.LoadUint8x32(src[:32])

    // Check continuation bits: byte & 0x80 == 0 means end of varint
    mask := archsimd.BroadcastUint8x32(0x80)
    continuations := v.And(mask)

    // Find positions where continuation == 0 (varint ends)
    endMask := continuations.CompareEqual(archsimd.ZeroUint8x32())

    // Extract positions and compute lengths
    // ... (complex bit manipulation)
}
```

#### Varint Performance Comparison

| Method | Throughput | Complexity | Compatibility |
|--------|------------|------------|---------------|
| Standard (scalar) | ~200 MB/s | Low | Universal |
| Batch boundary | ~400 MB/s | Medium | Standard varint |
| Group varint | ~800 MB/s | Medium | Requires re-encoding |
| Stream-VByte | ~1.2 GB/s | High | Requires re-encoding |

**Recommendation**: Start with batch boundary detection for standard varint compatibility, then add group-varint as an option.

---

## Implementation Roadmap

### Phase 1: Core Operations (Low Effort, High Value)

1. **RSqrt functions** - 1-2 days
   - `RSqrt_AVX2_F32x8`, `RSqrt_AVX512_F32x16`, `RSqrt_NEON_F32x4`
   - Newton-Raphson refinement variants
   - Fallback implementation

2. **Hardware gather intrinsics** - 2-3 days
   - Upgrade existing gather functions to use `VPGATHERDPS` etc.
   - Requires archsimd support or inline assembly

### Phase 2: Algorithms (Medium Effort)

3. **Prefix sum** - 3-5 days
   - In-register Hillis-Steele for small vectors
   - Block-based algorithm for slices
   - Integrate with `DeltaDecode`

### Phase 3: Varint (Higher Effort)

4. **Varint batch decoder** - 1-2 weeks
   - Boundary detection with SIMD comparison
   - Standard varint compatibility
   - Benchmarks vs scalar baseline

5. **Group varint (optional)** - 1 week
   - Encoder and decoder
   - Format documentation

---

## Benchmarking Plan

### Microbenchmarks

```go
func BenchmarkRSqrt_AVX2_F32x8(b *testing.B)
func BenchmarkGather_AVX2_U64x4(b *testing.B)
func BenchmarkPrefixSum_AVX2_I64x4(b *testing.B)
func BenchmarkVarintDecode_Scalar(b *testing.B)
func BenchmarkVarintDecode_SIMD(b *testing.B)
```

### Integration Benchmarks (with zapx)

1. Query latency on standard benchmark corpus
2. Segment merge throughput
3. Memory bandwidth utilization

---

## References

- [Stream VByte paper](https://arxiv.org/abs/1709.08990)
- [Group Varint (Google)](https://static.googleusercontent.com/media/research.google.com/en//people/jeff/WSDM09-keynote.pdf)
- [SIMD-friendly integer compression](https://lemire.me/blog/2012/09/12/fast-integer-compression-decoding-billions-of-integers-per-second/)
- [Intel Intrinsics Guide - VPGATHERDPS](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=vpgatherdps)
- [Bleve/zapx source](https://github.com/blevesearch/zapx)

---

## Appendix: Zapx Hot Path Analysis

### File Locations

| Operation | File | Lines | Frequency |
|-----------|------|-------|-----------|
| Varint decode | `memuvarint.go` | 41-77 | Critical |
| Chunk offsets | `intDecoder.go` | 56-57, 140-210 | Critical |
| Posting iteration | `posting.go` | 468-596 | Hot |
| Freq/norm decode | `posting.go` | 411-466 | Hot |
| Norm calculation | `posting.go` | 891-893 | Hot |
| Doc remapping | `merge.go` | 207-318 | Very Hot |
| Bitmap operations | `posting.go` | 128-249 | Hot |
| Doc values | `docvalues.go` | 140-206 | Hot |

### Data Types

- `uint64`: Document numbers, frequencies, positions
- `uint32`: Roaring bitmap entries, term counts
- `float32`: Norm values (packed as bits)
- `[]byte`: Varint-encoded data streams
