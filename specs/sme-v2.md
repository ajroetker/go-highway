Last login: Wed Jan 14 17:36:50 on ttys010

~ on ☁️  ajroetker@antfly.io
❯ z go-hig

go-highway on  require-workerpool-pool [$] via 🐹 v1.26rc2 on ☁️  ajroetker@antfly.io
❯ claude

 ▐▛███▜▌   Claude Code v2.1.23
▝▜█████▛▘  Opus 4.5 · Claude Max
  ▘▘ ▝▝    ~/go/src/github.com/ajroetker/go-highway

❯ Another claude told me ```  1. Use Apple Accelerate for matmul — Estimated 3-5x matmul speedup

  This is the single biggest easy win. cblas_sgemm uses Apple's AMX coprocessor (dedicated matrix hardware on M-series chips), which is significantly faster than NEON SIMD for large matrices.

  // #cgo LDFLAGS: -framework Accelerate
  // #include <Accelerate/Accelerate.h>
  import "C"

  func matmulAccelerate(A, B, C []float32, M, N, K int) {
  C.cblas_sgemm(C.CblasRowMajor, C.CblasNoTrans, C.CblasNoTrans,
  C.int(M), C.int(N), C.int(K),
  1.0, (*C.float)(&A[0]), C.int(K),
  (*C.float)(&B[0]), C.int(N),
  0.0, (*C.float)(&C[0]), C.int(N))
  }

  The AMX is a separate execution unit — it runs matrix math without using the CPU's NEON pipeline at all. This is how CoreML hits 33ms.
  ``` I'm wondering though if we could speed up our blockedmatmul using sme somehow though? Could you deep dive into this?

⏺ Explore(Explore matmul and SME code)
  ⎿  Done (37 tool uses · 151.4k tokens · 2m 9s)

⏺ Searched for 2 patterns, read 5 files (ctrl+o to expand)

⏺ Here's the deep dive on SME vs Accelerate for your matmul.

  You already have SME, and it's close to peak

  The other Claude's suggestion about Apple Accelerate/AMX is reasonable in general, but your codebase already uses SME FMOPA extensively, and on Apple M4 the SME unit is the AMX coprocessor (or rather, they share the same matrix execution
  hardware). So you're already hitting the same hardware that cblas_sgemm would use.

  What you're currently achieving

  From DARWIN_SME.md:186-188:
  ┌───────┬────────────┬───────────────────┐
  │ Size  │ GoAT FMOPA │ Handwritten FMOPA │
  ├───────┼────────────┼───────────────────┤
  │ 32×32 │ 313 GFLOPS │ 430 GFLOPS        │
  ├───────┼────────────┼───────────────────┤
  │ 48×48 │ 389 GFLOPS │ 479 GFLOPS        │
  ├───────┼────────────┼───────────────────┤
  │ 64×64 │ 447 GFLOPS │ 500 GFLOPS        │
  └───────┴────────────┴───────────────────┘
  Peak reported is ~2008 GFLOPS. So you're at 22-25% of theoretical peak with GoAT-generated code.

  Why there's a gap (and it's not AMX vs SME)

  The gap isn't because you're using the wrong hardware unit. SME's FMOPA is Apple's matrix accelerator, exposed through the ARM ISA rather than Apple's proprietary AMX interface. The gap comes from:

  1. Memory latency hiding (documented in your DARWIN_SME.md lines 194-247): GoAT generates tight 7-instruction K-loops where back-to-back loads stall the pipeline. Handwritten assembly uses 13+ instructions with address calculations interleaved
   between loads, hiding memory latency. Clang optimizes away the "useless" instructions that would fill stall cycles.
  2. Single-tile utilization: Your current blocked_fmopa_arm64.c:81-92 processes one 16×16 ZA tile at a time (svzero_za → K-loop → store). Apple M4 has 4 ZA tiles (ZA0-ZA3). Using multiple tiles simultaneously would allow you to amortize load
  latency across more FMOPAs per K iteration.
  3. Transpose overhead: Every SME matmul path in z_matmul_arm64.go pre-transposes A (and B for KLast) using sync.Pool buffers. For matmulKLastFMOPA (lines 635-678), you transpose both A and B. This is O(MK + NK) extra work.

  Where the real speedup opportunities are

  Multi-tile FMOPA (biggest win, estimated 2-3x)

  Your current C kernel processes one ZA tile per iteration:
  // blocked_fmopa_arm64.c:81-92 - current: 1 tile
  svzero_za();
  for (long kk = 0; kk < k; kk++) {
      svfloat32_t a_col = svld1_f32(pg, at + kk * m + ti);
      svfloat32_t b_row = svld1_f32(pg, b + kk * n + tj);
      svmopa_za32_f32_m(0, pg, pg, a_col, b_row);
  }

  With 4 tiles, you could process a 32×16 or 16×32 output region per K iteration, issuing 2 or 4 FMOPAs per pair of loads:

  // Hypothetical 2-tile: 32×16 output, 2 FMOPAs per K iteration
  svzero_za();  // zeros all 4 tiles
  for (long kk = 0; kk < k; kk++) {
      svfloat32_t a_col0 = svld1_f32(pg, at + kk * m + ti);       // rows 0-15
      svfloat32_t a_col1 = svld1_f32(pg, at + kk * m + ti + 16);  // rows 16-31
      svfloat32_t b_row  = svld1_f32(pg, b + kk * n + tj);
      svmopa_za32_f32_m(0, pg, pg, a_col0, b_row);  // ZA0: C[0:16, 0:16]
      svmopa_za32_f32_m(1, pg, pg, a_col1, b_row);  // ZA1: C[16:32, 0:16]
  }

  This doubles compute per B-load. The 3-load + 2-FMOPA pattern also naturally creates more instruction-level parallelism, partially addressing the latency hiding problem without needing handwritten assembly.

  However, GOAT has the same clang optimization problem -- it'll likely collapse the address math. So for maximum performance here, you'd want handwritten assembly for the multi-tile kernel.

  Avoid double transpose for KLast

  For the matmulKLastFMOPA path (the one nn.DenseAuto uses via MatMulKLastAuto), you're transposing both A and B. Since FMOPA computes ZA += col(A) ⊗ row(B), and in KLast layout B is already [N, K], you could write a kernel that loads B rows
  directly (along K) and A rows directly, accumulating with a different tile access pattern. This would eliminate the B transpose entirely.

  Wire up SME nn dispatch

  The exploration found that hwy/contrib/nn/c/ has SME-specific C files for qkvdense_sme_arm64.c and sdpa_sme_arm64.c, with Go wrappers in asm/, but z_nn_arm64.go only dispatches to NEON paths. Wiring these up would avoid the overhead of going
  through the generic DenseAuto → MatMulKLastAuto → transpose → FMOPA path, and instead fuse the bias-add and other operations directly with the SME matmul.

  Would Accelerate actually be faster?

  Apple's cblas_sgemm achieves near-peak GFLOPS because:
  - It uses Apple's proprietary AMX interface (not the ARM SME ISA), which may have lower overhead
  - Apple has hand-tuned the packing, tiling, and prefetch patterns for their specific microarchitecture
  - It uses all available execution resources optimally

  The AMX coprocessor and SME likely share the same physical hardware, but the AMX interface may have lower dispatch overhead. That said, calling through cgo adds ~100-200ns per call for the C→Go boundary, which matters for small matrices.

  For large matrices (1024×1024+), Accelerate would likely be faster than your current GoAT FMOPA by 2-4x, purely because of Apple's hand-tuned tiling strategy. For small matrices (32-256), the cgo overhead may negate the gains.

  Summary: what to prioritize

  1. Multi-tile FMOPA kernel (handwritten assembly): Use 2-4 ZA tiles to process larger output regions per K iteration. This is the single biggest improvement available within pure Go/SME.
  2. Eliminate KLast double-transpose: Write an FMOPA kernel that works directly with [M,K] × [N,K] layout, avoiding the B transpose.
  3. Wire up SME nn paths: Connect the existing qkvdense_sme and sdpa_sme wrappers in z_nn_arm64.go.
  4. Accelerate via cgo as optional backend: For large GEMMs where cgo overhead is negligible, cblas_sgemm would give you Apple's fully optimized path. This could be a build-tag-gated option rather than replacing SME.

✻ Churned for 3m 22s

❯ So it's actually the NEON Blocked MatMul showing up in the profile, does that mean the matrix has a specific shape?

⏺ Yes. Looking at the dispatch logic in z_matmul_arm64.go:415-451, the SME blocked path falls back to NEON in several cases:

  Alignment requirements (line 418):
  if m%16 != 0 || n%16 != 0 || k%16 != 0 {
      asm.MatMulNEONF32(a, b, c, m, n, k)
      return
  }

  Small dimension threshold (line 424):
  if m < minDimForBlockedSME || n < minDimForBlockedSME || k < minDimForBlockedSME {
      asm.MatMulNEONF32(a, b, c, m, n, k)
      return
  }

  Where minDimForBlockedSME = 32.

  So you're hitting NEON blocked if any of M, N, or K is:
  - Not a multiple of 16, or
  - Less than 32

  The most common case in transformer workloads would be M not being a multiple of 16 -- M is typically the batch size or sequence length, which can be any value (1 for single-token inference, 7 for a prompt, etc.). N and K are usually model
  dimensions (multiples of 64 or 128), so those are rarely the culprit.

  Can you check what M, N, K values are being passed? If M is small (like 1-8 for token-at-a-time inference), that's expected behavior -- SME's 16×16 tile minimum makes it unsuitable for M < 16. But if M is something like 17 or 33 (just barely
  not aligned), you could pad to the next multiple of 16 and get a massive speedup by hitting the SME path instead.

❯ Should we add padding to our version?

  Could you look at the model we're using ```⏺ Bash(cat /Users/ajroetker/go/src/github.com/antflydb/antfly/termite/models/embedders/Snowflake/snowflake-arctic-embed-l-v2.0/config.json 2>/dev/null | python3 -m jso…)
  ⎿  {
  "word_embedding_dimension": 1024,
  "pooling_mode_cls_token": true,
  … +7 lines (ctrl+o to expand)
  ⎿  Interrupted · What should Claude do instead?

  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ───────
  ❯
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ───────
  [Opus 4.5] 📁 /Users/ajroetker/go/src/github.com/antflydb/antfly/termite | 🌿 chore/audit-backend-support | ☸️  gke_antfly-prod-01_us-central1_antfly-prod | 🏗️  no-stack | 💰 $42.5358```?

⏺ Explore(Find model config and dimensions)
  ⎿  Done (24 tool uses · 91.1k tokens · 3m 10s)

⏺ This clarifies the situation. The model dimensions are:
  ┌─────────────────────────┬─────┬──────┬──────┐
  │ Operation (×24 layers)  │  M  │  K   │  N   │
  ├─────────────────────────┼─────┼──────┼──────┤
  │ QKV projections (×3)    │ S   │ 1024 │ 1024 │
  ├─────────────────────────┼─────┼──────┼──────┤
  │ Attn scores (×16 heads) │ S   │ 64   │ S    │
  ├─────────────────────────┼─────┼──────┼──────┤
  │ Attn × V (×16 heads)    │ S   │ S    │ 64   │
  ├─────────────────────────┼─────┼──────┼──────┤
  │ Output projection       │ S   │ 1024 │ 1024 │
  ├─────────────────────────┼─────┼──────┼──────┤
  │ FFN up                  │ S   │ 1024 │ 4096 │
  ├─────────────────────────┼─────┼──────┼──────┤
  │ FFN down                │ S   │ 4096 │ 1024 │
  └─────────────────────────┴─────┴──────┴──────┘
  N and K are always 64, 1024, or 4096 -- all multiples of 16. The problem is M = S (sequence length), which is whatever the input text tokenizes to. A 50-word sentence might tokenize to 73 tokens, which fails the m%16 != 0 check and drops to
  NEON.

  Padding M makes sense here because:

  - You only need to pad the input rows of A (and corresponding output rows of C) up to the next multiple of 16
  - The extra output rows are garbage that the caller ignores -- the actual result in rows [0, M) is correct
  - N and K are already aligned, so no padding needed there
  - The cost of padding is small: at worst you compute 15 extra rows, but you gain SME's ~300-500 GFLOPS vs NEON's ~25-75 GFLOPS

  For the attention score matmuls where N = S too, you'd need to pad both M and N, which means padding the B matrix columns as well (zero-padded, so the extra attention scores are zero and don't affect softmax meaningfully -- though you'd want
  to mask them properly).

  The cleanest place to add padding would be in blockedMatMulFMOPA and matmulKLastFMOPA in z_matmul_arm64.go, so it's transparent to callers. Want me to implement that?

✻ Churned for 2m 4s

❯ Yes please!
