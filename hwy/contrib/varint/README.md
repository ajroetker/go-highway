# Varint - SIMD-Accelerated Variable-Length Integer Encoding

This package provides SIMD-accelerated encoding and decoding of variable-length integers, commonly used in search engines, databases, and protocol buffers.

## Encoding Formats

### Stream-VByte

Stream-VByte separates control bytes from data bytes for better SIMD efficiency. Each control byte describes 4 values, with 2 bits per value indicating byte length (1-4 bytes).

```go
import "github.com/ajroetker/go-highway/hwy/contrib/varint"

// Streaming encoder
enc := varint.NewStreamVByteEncoder()
enc.Add(300)
enc.Add(5)
enc.Add(1000)
enc.Add(2)
control, data := enc.Finish()

// Batch decode
values := make([]uint32, 4)
decoded, bytesRead := varint.DecodeStreamVByte32(control, data, values)

// Calculate data length from control bytes
dataLen := varint.StreamVByte32DataLen(control)
```

### Group Varint

Group Varint packs 4 values with a leading control byte. More compact than Stream-VByte but slightly slower to decode.

```go
// Encode 4 uint32 values
values := [4]uint32{300, 5, 1000, 2}
dst := make([]byte, 17) // max 1 control + 4×4 data
n := varint.EncodeGroupVarint32(values, dst)

// Decode
decoded, bytesRead := varint.DecodeGroupVarint32(dst[:n])

// Pre-calculate encoded length
length := varint.GroupVarint32Len(values)
```

### Masked VByte

Standard LEB128/VByte encoding where each byte uses 7 bits for data and 1 bit as a continuation flag.

```go
// Batch decode varints
values := make([]uint64, 100)
decoded, consumed := varint.DecodeUvarint64Batch(data, values, 100)

// Fixed-count decoders for common patterns
freq, norm, n := varint.Decode2Uvarint64(data)      // freq/norm pairs
loc, n := varint.Decode5Uvarint64(data)              // location fields

// Find varint boundaries in parallel
mask := varint.FindVarintEnds(data[:32])
// bit i set means data[i] is the last byte of a varint
```

## Delta Encoding

For sorted sequences (like posting lists), delta encoding dramatically improves compression:

```go
// Encode deltas
docIDs := []uint32{100, 105, 110, 200, 250}
deltas := make([]uint32, len(docIDs))
varint.DeltaEncode32(docIDs, deltas)  // [100, 5, 5, 90, 50]

// Decode with SIMD prefix sum
varint.DeltaDecode32(deltas, docIDs)  // [100, 105, 110, 200, 250]
```

## SIMD Acceleration

| Operation | AVX2 | AVX-512 | NEON | Fallback |
|-----------|------|---------|------|----------|
| Stream-VByte decode | VPSHUFB | VPSHUFB | TBL | Scalar |
| Group Varint decode | VPSHUFB | VPSHUFB | TBL | Scalar |
| Varint boundary detection | VPCMPGTB | VPCMPGTB | CMGT | Scalar |
| Delta decode (prefix sum) | VPADDD | VPADDD | ADD | Scalar |

## Performance

Typical speedups over scalar code:

- **Stream-VByte decode**: 4-8x faster
- **Group Varint decode**: 3-6x faster
- **Boundary detection**: 8-16x faster
- **Delta decode**: 2-4x faster

## Use Cases

- **Search engines**: Posting list compression (docIDs, term frequencies)
- **Databases**: Integer column compression
- **Protocol Buffers**: Varint field decoding
- **Time series**: Delta-encoded timestamps

## References

- [Stream VByte: Faster Byte-Oriented Integer Compression](https://arxiv.org/abs/1709.08990)
- [Decoding billions of integers per second through vectorization](https://arxiv.org/abs/1209.2137)
