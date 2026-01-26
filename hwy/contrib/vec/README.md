# Vec - SIMD-Accelerated Vector Operations

This package provides SIMD-accelerated vector operations commonly used in machine learning, search, and scientific computing.

## Operations

### Dot Product

```go
import "github.com/ajroetker/go-highway/hwy/contrib/vec"

a := []float32{1, 2, 3, 4}
b := []float32{5, 6, 7, 8}
result := vec.Dot(a, b)  // 1*5 + 2*6 + 3*7 + 4*8 = 70
```

### Distance Functions

```go
a := []float32{1, 2, 3}
b := []float32{4, 5, 6}

// Squared Euclidean distance (faster, no sqrt)
sqDist := vec.L2SquaredDistance(a, b)  // (1-4)² + (2-5)² + (3-6)² = 27

// Euclidean distance
dist := vec.L2Distance(a, b)  // √27 ≈ 5.196
```

### Norms

```go
v := []float32{3, 4}

// Squared L2 norm (sum of squares)
sqNorm := vec.SquaredNorm(v)  // 3² + 4² = 25

// L2 norm (Euclidean magnitude)
norm := vec.Norm(v)  // √25 = 5
```

### Normalization

```go
v := []float32{3, 4}
normalized := make([]float32, 2)

// Normalize to unit vector
vec.Normalize(v, normalized)  // [0.6, 0.8]
```

### Arithmetic Operations

```go
a := []float32{1, 2, 3, 4}
b := []float32{5, 6, 7, 8}
result := make([]float32, 4)

// Element-wise operations
vec.Add(a, b, result)       // [6, 8, 10, 12]
vec.Sub(a, b, result)       // [-4, -4, -4, -4]
vec.Mul(a, b, result)       // [5, 12, 21, 32]

// Scalar operations
vec.Scale(a, 2.0, result)   // [2, 4, 6, 8]
vec.AddScalar(a, 10, result) // [11, 12, 13, 14]
```

### Reductions

```go
v := []float32{1, 5, 2, 8, 3}

sum := vec.Sum(v)     // 19
mean := vec.Mean(v)   // 3.8
max := vec.Max(v)     // 8
min := vec.Min(v)     // 1

// ArgMax/ArgMin return the index
maxIdx := vec.ArgMax(v)  // 3 (index of 8)
minIdx := vec.ArgMin(v)  // 0 (index of 1)
```

### Batch Operations

Process multiple vectors efficiently:

```go
// Batch dot products
vectors := [][]float32{v1, v2, v3, v4}
query := []float32{...}
results := make([]float32, 4)
vec.BatchDot(vectors, query, results)

// Batch distances
vec.BatchL2SquaredDistance(vectors, query, results)
```

## Type Support

All operations support both `float32` and `float64`:

```go
// float32
vec.Dot([]float32{1, 2}, []float32{3, 4})

// float64
vec.Dot([]float64{1, 2}, []float64{3, 4})
```

## SIMD Acceleration

| Operation | AVX2 | AVX-512 | NEON | SME |
|-----------|------|---------|------|-----|
| Dot product | FMA | FMA | FMLA | FMOPA |
| L2 distance | FMA | FMA | FMLA | - |
| Norm | FMA | FMA | FMLA | - |
| ArgMax/ArgMin | VPCMPGTPS | VPCMPPS | FCMGT | - |
| Add/Sub/Mul | VADDPS | VADDPS | FADD | - |

Uses 4x loop unrolling with multiple accumulators for maximum instruction-level parallelism.

## Performance

Typical speedups over scalar code (1024-element vectors):

- **Dot product**: 8-16x faster
- **L2 distance**: 6-12x faster
- **Normalize**: 4-8x faster
- **ArgMax**: 4-8x faster

## Use Cases

- **Vector search**: Similarity computation (cosine, L2)
- **Machine learning**: Forward/backward passes, loss computation
- **Embeddings**: Nearest neighbor search
- **Scientific computing**: Linear algebra primitives
