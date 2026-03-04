package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFixReservedAsmNames_Assembly(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    string
		changed bool
	}{
		{
			name:    "standalone g is renamed",
			input:   "\tMOVD g+0(FP), R0\n\tMOVD out+8(FP), R1\n",
			want:    "\tMOVD gv+0(FP), R0\n\tMOVD out+8(FP), R1\n",
			changed: true,
		},
		{
			name:    "img is not corrupted to imgv",
			input:   "\tMOVD img+0(FP), R0\n\tMOVD out+8(FP), R1\n",
			want:    "\tMOVD img+0(FP), R0\n\tMOVD out+8(FP), R1\n",
			changed: false,
		},
		{
			name:    "g renamed but img preserved in same file",
			input:   "\tMOVD img+0(FP), R0\n\tMOVD g+8(FP), R1\n\tMOVD out+16(FP), R2\n",
			want:    "\tMOVD img+0(FP), R0\n\tMOVD gv+8(FP), R1\n\tMOVD out+16(FP), R2\n",
			changed: true,
		},
		{
			name:    "flag not corrupted to flav",
			input:   "\tMOVD flag+0(FP), R0\n",
			want:    "\tMOVD flag+0(FP), R0\n",
			changed: false,
		},
		{
			name:    "no reserved names leaves file unchanged",
			input:   "\tMOVD src+0(FP), R0\n\tMOVD dst+8(FP), R1\n",
			want:    "\tMOVD src+0(FP), R0\n\tMOVD dst+8(FP), R1\n",
			changed: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "test.s")
			if err := os.WriteFile(path, []byte(tt.input), 0o644); err != nil {
				t.Fatal(err)
			}

			if err := fixReservedAsmNames(path); err != nil {
				t.Fatal(err)
			}

			got, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tt.want {
				t.Errorf("got:\n%s\nwant:\n%s", got, tt.want)
			}
		})
	}
}

func TestFixReservedAsmNames_GoWrapper(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    string
		changed bool
	}{
		{
			name:    "g as middle parameter is renamed",
			input:   "func foo(r, g, b unsafe.Pointer)\n",
			want:    "func foo(r, gv, b unsafe.Pointer)\n",
			changed: true,
		},
		{
			name:    "g as first parameter is renamed",
			input:   "func foo(g, out unsafe.Pointer)\n",
			want:    "func foo(gv, out unsafe.Pointer)\n",
			changed: true,
		},
		{
			name:    "g as last parameter is renamed",
			input:   "func foo(r, g unsafe.Pointer)\n",
			want:    "func foo(r, gv unsafe.Pointer)\n",
			changed: true,
		},
		{
			name:    "img is not corrupted to imgv",
			input:   "func foo(img, out unsafe.Pointer)\n",
			want:    "func foo(img, out unsafe.Pointer)\n",
			changed: false,
		},
		{
			name:    "flag is not corrupted",
			input:   "func foo(flag, out unsafe.Pointer)\n",
			want:    "func foo(flag, out unsafe.Pointer)\n",
			changed: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "test.gen.go")
			if err := os.WriteFile(path, []byte(tt.input), 0o644); err != nil {
				t.Fatal(err)
			}

			if err := fixReservedAsmNames(path); err != nil {
				t.Fatal(err)
			}

			got, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tt.want {
				t.Errorf("got:\n%s\nwant:\n%s", got, tt.want)
			}
		})
	}
}

func TestFixReservedAsmNames_NonexistentFile(t *testing.T) {
	err := fixReservedAsmNames(filepath.Join(t.TempDir(), "nonexistent.s"))
	if err != nil {
		t.Errorf("expected nil for nonexistent file, got: %v", err)
	}
}

// TestProfileOpFn_Deterministic verifies that profileOpFn returns the widest
// tier's intrinsic deterministically, not an arbitrary map entry.
func TestProfileOpFn_Deterministic(t *testing.T) {
	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("no NEON float32 profile")
	}
	emitter := &CEmitter{profile: profile, elemType: "float32"}

	tests := []struct {
		op   string
		want string
	}{
		{"Add", profile.AddFn["q"]},
		{"Sub", profile.SubFn["q"]},
		{"Mul", profile.MulFn["q"]},
		{"Neg", profile.NegFn["q"]},
	}
	for _, tt := range tests {
		// Run multiple times to verify determinism (mapFirstValue was random).
		for range 10 {
			got := emitter.profileOpFn(tt.op)
			if got != tt.want {
				t.Errorf("profileOpFn(%q) = %q, want %q", tt.op, got, tt.want)
			}
		}
	}
}

// TestProfileTierOpFn verifies that profileTierOpFn returns the correct
// intrinsic for a specific tier.
func TestProfileTierOpFn(t *testing.T) {
	profile := GetCProfile("NEON", "float32")
	if profile == nil {
		t.Fatal("no NEON float32 profile")
	}
	emitter := &CEmitter{profile: profile, elemType: "float32"}

	got := emitter.profileTierOpFn("Add", "q")
	if got != profile.AddFn["q"] {
		t.Errorf("profileTierOpFn(Add, q) = %q, want %q", got, profile.AddFn["q"])
	}

	got = emitter.profileTierOpFn("Add", "nonexistent")
	if got != "" {
		t.Errorf("profileTierOpFn(Add, nonexistent) = %q, want empty", got)
	}

	got = emitter.profileTierOpFn("UnknownOp", "q")
	if got != "" {
		t.Errorf("profileTierOpFn(UnknownOp, q) = %q, want empty", got)
	}
}

// TestCEmitter_NilProfile verifies that CEmitter intrinsic methods return
// correct NEON fallback values when profile is nil.
func TestCEmitter_NilProfile(t *testing.T) {
	e32 := NewCEmitter("pkg", "float32", Target{})
	e64 := NewCEmitter("pkg", "float64", Target{})

	if got := e32.vecType(); got != "float32x4_t" {
		t.Errorf("vecType(f32, nil profile) = %q, want float32x4_t", got)
	}
	if got := e64.vecType(); got != "float64x2_t" {
		t.Errorf("vecType(f64, nil profile) = %q, want float64x2_t", got)
	}
	if got := e32.loadIntrinsic(); got != "vld1q_f32" {
		t.Errorf("loadIntrinsic(f32, nil profile) = %q, want vld1q_f32", got)
	}
	if got := e32.storeIntrinsic(); got != "vst1q_f32" {
		t.Errorf("storeIntrinsic(f32, nil profile) = %q, want vst1q_f32", got)
	}
	if got := e32.dupIntrinsic(); got != "vdupq_n_f32" {
		t.Errorf("dupIntrinsic(f32, nil profile) = %q, want vdupq_n_f32", got)
	}
	if got := e32.getLaneIntrinsic(); got != "vgetq_lane_f32" {
		t.Errorf("getLaneIntrinsic(f32, nil profile) = %q, want vgetq_lane_f32", got)
	}
	if got := e32.profileOpFn("Add"); got != "" {
		t.Errorf("profileOpFn(Add, nil profile) = %q, want empty", got)
	}
}

// TestNeonNoSimdGuard verifies that neonNoSimdGuard returns the NoSimdEnv
// guard for NEON targets and empty for all others.
func TestNeonNoSimdGuard(t *testing.T) {
	tests := []struct {
		target Target
		want   string
	}{
		{Target{Name: "NEON"}, "hwy.NoSimdEnv()"},
		{Target{Name: "AVX2"}, ""},
		{Target{Name: "AVX512"}, ""},
		{Target{Name: "SVE_DARWIN"}, ""},
		{Target{Name: "SVE_LINUX"}, ""},
	}
	for _, tt := range tests {
		got := neonNoSimdGuard(tt.target)
		if got != tt.want {
			t.Errorf("neonNoSimdGuard(%s) = %q, want %q", tt.target.Name, got, tt.want)
		}
	}
}
