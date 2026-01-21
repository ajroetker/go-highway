# SIMD StreamVByte Integration for Bleve/Zapx

## Summary

This document captures benchmark findings and recommendations for integrating SIMD-accelerated StreamVByte encoding into bleve's zapx segment format, replacing standard varint encoding for location data.

## Recommendation

**Use Columnar + Delta + StreamVByte as the new location encoding format.**

This combination provides:
- **1.28x faster** merge cycles for large documents
- **17% smaller** encoded size for large positions
- Negligible overhead (~2ns) for small documents
- SIMD acceleration on ARM64 (NEON) and AMD64 (AVX2/AVX-512)

## Benchmark Results

### Encoded Sizes

| Scenario | Varint | SVB Row | Columnar | Columnar+Delta |
|----------|--------|---------|----------|----------------|
| Small positions (0-190) | 115 B | 130 B | 125 B | 125 B |
| Large positions (0-47500) | 197 B | 187 B | 182 B | **163 B** |

### Merge Cycle Performance (CPU only)

| Encoding | Small Values | Large Values |
|----------|--------------|--------------|
| Varint | 191 ns | 315 ns |
| StreamVByte (row) | 129 ns | 252 ns |
| Columnar | 133 ns | - |
| Columnar+Delta | 134 ns | - |

### Total Time (CPU + Disk I/O)

#### Small Positions
| Storage | Varint | SVB Row | Columnar+Delta |
|---------|--------|---------|----------------|
| NVMe (3 GB/s) | 305 ns | **214 ns** | 216 ns |
| SATA (500 MB/s) | 645 ns | 649 ns | **632 ns** |

#### Large Positions (positions 0-47500, ~2500 apart)
| Storage | Varint | SVB Row | Columnar+Delta |
|---------|--------|---------|----------------|
| NVMe (3 GB/s) | 384 ns | 380 ns | **300 ns** |
| SATA (500 MB/s) | 1039 ns | - | **842 ns** |

## Architecture

### Data Layout

```
Raw location data (row-oriented):
  [fieldID, pos, start, end, numAP] × N locations

Columnar+Delta layout:
  fieldIDs:   [f0, f1, f2, ...]        (not delta - small values 0-2)
  posDelta:   [p0, p1-p0, p2-p1, ...]  (delta encoded)
  startDelta: [s0, s1-s0, s2-s1, ...]  (delta encoded)
  lengths:    [e0-s0, e1-s1, ...]      (end - start, not delta)
  numAPs:     [0, 0, 0, ...]           (usually all zeros)
```

### Why This Works

1. **Columnar** groups similar values together for better compression patterns
2. **Delta** converts large monotonic values (positions, offsets) to small constants
3. **StreamVByte** encodes small values efficiently with SIMD fast path

When positions are evenly spaced (common case), delta encoding produces constant values that all fit in 1 byte, hitting StreamVByte's optimized path.

## go-highway APIs

### StreamVByte Encoding/Decoding

```go
import "github.com/ajroetker/go-highway/hwy/contrib/varint"

// Encode
ctrl, data := varint.EncodeStreamVByte32(values)

// Encode into existing buffers (zero allocation)
ctrl, data = varint.EncodeStreamVByte32Into(values, ctrlBuf, dataBuf)

// Decode
varint.DecodeStreamVByte32Into(ctrl, data, dst)
```

### Delta Decoding (Prefix Sum)

```go
import "github.com/ajroetker/go-highway/hwy/contrib/algo"

// Delta decode in place (reconstructs absolute values from deltas)
// data[i] becomes base + data[0] + data[1] + ... + data[i]
algo.DeltaDecode(data, base)

// Or use PrefixSum directly (base=0)
algo.PrefixSum(data)
```

## Implementation Outline

### Segment Format V2

```go
// LocationBlockV2 encodes location data using columnar+delta+StreamVByte
type LocationBlockV2 struct {
    NumLocations uint32  // number of locations

    // Each column is StreamVByte-encoded (control bytes + data bytes)
    FieldIDs    EncodedColumn  // raw values (small)
    PosDelta    EncodedColumn  // delta-encoded positions
    StartDelta  EncodedColumn  // delta-encoded byte offsets
    Lengths     EncodedColumn  // end - start (not delta)
    NumAPs      EncodedColumn  // array positions
}

type EncodedColumn struct {
    Control []byte
    Data    []byte
}
```

### Encoding

```go
func EncodeLocations(locs []Location) *LocationBlockV2 {
    n := len(locs)

    // Extract columns
    fieldIDs := make([]uint32, n)
    positions := make([]uint32, n)
    starts := make([]uint32, n)
    lengths := make([]uint32, n)
    numAPs := make([]uint32, n)

    for i, loc := range locs {
        fieldIDs[i] = loc.FieldID
        positions[i] = loc.Position
        starts[i] = loc.Start
        lengths[i] = loc.End - loc.Start
        numAPs[i] = loc.NumArrayPositions
    }

    // Delta encode monotonic columns
    posDelta := deltaEncode(positions)
    startDelta := deltaEncode(starts)

    // StreamVByte encode each column
    return &LocationBlockV2{
        NumLocations: uint32(n),
        FieldIDs:     encodeColumn(fieldIDs),
        PosDelta:     encodeColumn(posDelta),
        StartDelta:   encodeColumn(startDelta),
        Lengths:      encodeColumn(lengths),
        NumAPs:       encodeColumn(numAPs),
    }
}

func deltaEncode(values []uint32) []uint32 {
    if len(values) == 0 {
        return nil
    }
    result := make([]uint32, len(values))
    result[0] = values[0]
    for i := 1; i < len(values); i++ {
        result[i] = values[i] - values[i-1]
    }
    return result
}

func encodeColumn(values []uint32) EncodedColumn {
    ctrl, data := varint.EncodeStreamVByte32(values)
    return EncodedColumn{Control: ctrl, Data: data}
}
```

### Decoding

```go
func DecodeLocations(block *LocationBlockV2, dst []Location) {
    n := int(block.NumLocations)

    // Decode columns
    fieldIDs := make([]uint32, n)
    positions := make([]uint32, n)
    starts := make([]uint32, n)
    lengths := make([]uint32, n)
    numAPs := make([]uint32, n)

    varint.DecodeStreamVByte32Into(block.FieldIDs.Control, block.FieldIDs.Data, fieldIDs)
    varint.DecodeStreamVByte32Into(block.PosDelta.Control, block.PosDelta.Data, positions)
    varint.DecodeStreamVByte32Into(block.StartDelta.Control, block.StartDelta.Data, starts)
    varint.DecodeStreamVByte32Into(block.Lengths.Control, block.Lengths.Data, lengths)
    varint.DecodeStreamVByte32Into(block.NumAPs.Control, block.NumAPs.Data, numAPs)

    // Delta decode (prefix sum) to reconstruct absolute values
    algo.DeltaDecode(positions, 0)
    algo.DeltaDecode(starts, 0)

    // Reconstruct locations
    for i := 0; i < n; i++ {
        dst[i] = Location{
            FieldID:           fieldIDs[i],
            Position:          positions[i],
            Start:             starts[i],
            End:               starts[i] + lengths[i],
            NumArrayPositions: numAPs[i],
        }
    }
}
```

## Migration Path

1. **Add format version flag** to segment header
2. **Write new segments** in V2 format (columnar+delta+SVB)
3. **Read both formats** - V1 (varint) and V2
4. **Merge produces V2** - old segments get upgraded during compaction
5. **Eventually all segments are V2**

## Key Optimizations in StreamVByte

1. **Fast path for small values**: When all 4 values in a group fit in 1 byte, skip SIMD shuffle entirely
2. **Allocation-free encoding**: `EncodeStreamVByte32Into` reuses buffers
3. **SIMD table lookup**: 256-entry shuffle mask table for decode
4. **Unsafe pointer cast**: Direct memory reinterpretation instead of byte-by-byte extraction

## Test Data Characteristics

### Small positions (typical short documents)
- 20 locations × 5 fields = 100 values
- Positions: 0, 5, 10, 15, ... (small increments)
- All values fit in 1-2 bytes

### Large positions (large documents)
- 20 locations spread across ~50KB document
- Positions: 0, 2500, 5000, ... , 47500
- Values require 2 bytes, but deltas are constant (2500) = 2 bytes
- With delta: lengths are constant (50) = 1 byte each

## Conclusion

Columnar+Delta+StreamVByte provides the best combination of:
- **Speed**: SIMD-accelerated encode/decode
- **Compression**: Delta encoding reduces large monotonic values
- **Simplicity**: Clean separation of concerns (layout → transform → encoding)

The format handles both small and large documents efficiently, with only ~2ns overhead for small documents compared to the simpler row-oriented approach.
