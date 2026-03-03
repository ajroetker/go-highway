package main

import (
	"fmt"
	"strings"
	"testing"
)

func TestIsSVEInstruction(t *testing.T) {
	tests := []struct {
		asm  string
		want bool
	}{
		// Positive cases: all 14 regex patterns
		{"ptrue\tp0.s", true},
		{"ld1w\t{z0.s}, p0/z, [x0]", true},
		{"st1w\t{z0.s}, p0, [x0]", true},
		{"ldr\tz0, [x0]", true},
		{"str\tz0, [x0]", true},
		{"dup\tz0.s, w0", true},
		{"cntw\tx0", true},
		{"addvl\tsp, sp, #-2", true},
		{"rdsvl\tx8, #1", true},
		{"fmopa\tza0.s, p0/m, p0/m, z0.s, z1.s", true},
		{"mova\tz0.s, p0/m, za0h.s[w12, 0]", true},
		{"zero\t{za}", true},
		// Z-register usage (via sveZReg)
		{"fadd\tz0.s, z1.s, z2.s", true},
		// SVE store
		{"st1d\t{z3.d}, p0, [x0]", true},

		// Negative cases
		{"mov\tx0, x1", false},
		{"add\tx0, x1, x2", false},
		{"ret", false},
		{"", false},
		{"ldr\tx0, [sp, #16]", false},
		{"str\tx0, [sp, #16]", false},
	}

	for _, tt := range tests {
		t.Run(tt.asm, func(t *testing.T) {
			if got := isSVEInstruction(tt.asm); got != tt.want {
				t.Errorf("isSVEInstruction(%q) = %v, want %v", tt.asm, got, tt.want)
			}
		})
	}
}

func TestUsesSVEorSME(t *testing.T) {
	tests := []struct {
		name  string
		lines []*arm64Line
		want  bool
	}{
		{
			name:  "empty",
			lines: nil,
			want:  false,
		},
		{
			name: "regular instructions only",
			lines: []*arm64Line{
				{Assembly: "mov\tx0, x1"},
				{Assembly: "add\tx0, x1, x2"},
				{Assembly: "ret"},
			},
			want: false,
		},
		{
			name: "z-register usage",
			lines: []*arm64Line{
				{Assembly: "mov\tx0, x1"},
				{Assembly: "fadd\tz0.s, z1.s, z2.s"},
			},
			want: true,
		},
		{
			name: "SVE load",
			lines: []*arm64Line{
				{Assembly: "ldr\tz0, [x0]"},
			},
			want: true,
		},
		{
			name: "SVE store",
			lines: []*arm64Line{
				{Assembly: "str\tz0, [x0]"},
			},
			want: true,
		},
		{
			name: "SME fmopa",
			lines: []*arm64Line{
				{Assembly: "fmopa\tza0.s, p0/m, p0/m, z0.s, z1.s"},
			},
			want: true,
		},
		{
			name: "ZA mova",
			lines: []*arm64Line{
				{Assembly: "mova\tz0.s, p0/m, za0h.s[w12, 0]"},
			},
			want: true,
		},
		{
			name: "zero ZA",
			lines: []*arm64Line{
				{Assembly: "zero\t{za}"},
			},
			want: true,
		},
		{
			name: "empty assembly lines skipped",
			lines: []*arm64Line{
				{Assembly: ""},
				{Assembly: ""},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := usesSVEorSME(tt.lines); got != tt.want {
				t.Errorf("usesSVEorSME() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTransformForbiddenInstructions(t *testing.T) {
	tests := []struct {
		name    string
		input   []*arm64Line
		wantAsm []string
		wantBin []string
	}{
		{
			name: "movi d0 #0 -> fmov s0 wzr",
			input: []*arm64Line{
				{Assembly: "movi\td0, #0", Binary: "2f00e400"},
			},
			wantAsm: []string{"fmov\ts0, wzr"},
			wantBin: []string{"1e2703e0"},
		},
		{
			name: "movi d5 #0 -> fmov s5 wzr",
			input: []*arm64Line{
				{Assembly: "movi\td5, #0", Binary: "2f00e405"},
			},
			wantAsm: []string{"fmov\ts5, wzr"},
			wantBin: []string{fmt.Sprintf("%08x", uint32(0x1e2703e0)+5)},
		},
		{
			name: "passthrough regular instructions",
			input: []*arm64Line{
				{Assembly: "mov\tx0, x1", Binary: "aa0103e0"},
				{Assembly: "add\tx0, x1, x2", Binary: "8b020020"},
			},
			wantAsm: []string{"mov\tx0, x1", "add\tx0, x1, x2"},
			wantBin: []string{"aa0103e0", "8b020020"},
		},
		{
			name: "labels preserved on transformed line",
			input: []*arm64Line{
				{Labels: []string{"BB0_1"}, Assembly: "movi\td0, #0", Binary: "2f00e400"},
			},
			wantAsm: []string{"fmov\ts0, wzr"},
			wantBin: []string{"1e2703e0"},
		},
		{
			name: "empty assembly passthrough",
			input: []*arm64Line{
				{Assembly: "", Binary: ""},
			},
			wantAsm: []string{""},
			wantBin: []string{""},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := transformForbiddenInstructions(tt.input)
			if len(result) != len(tt.wantAsm) {
				t.Fatalf("got %d lines, want %d", len(result), len(tt.wantAsm))
			}
			for i := range result {
				if result[i].Assembly != tt.wantAsm[i] {
					t.Errorf("line %d: asm = %q, want %q", i, result[i].Assembly, tt.wantAsm[i])
				}
				if result[i].Binary != tt.wantBin[i] {
					t.Errorf("line %d: binary = %q, want %q", i, result[i].Binary, tt.wantBin[i])
				}
			}
			// Verify label preservation
			if tt.name == "labels preserved on transformed line" {
				if len(result[0].Labels) != 1 || result[0].Labels[0] != "BB0_1" {
					t.Errorf("labels not preserved: got %v", result[0].Labels)
				}
			}
		})
	}
}

func TestFixMOVAEncoding(t *testing.T) {
	tests := []struct {
		name    string
		input   []*arm64Line
		wantBin []string
	}{
		{
			name: "bit 17 not set - fixed",
			input: []*arm64Line{
				{Assembly: "mova\tz0.s, p0/m, za0h.s[w12, 0]", Binary: "c0800000"},
			},
			wantBin: []string{"c0820000"},
		},
		{
			name: "bit 17 already set - unchanged",
			input: []*arm64Line{
				{Assembly: "mova\tz0.s, p0/m, za0h.s[w12, 0]", Binary: "c0820000"},
			},
			wantBin: []string{"c0820000"},
		},
		{
			name: "non-MOVA unchanged",
			input: []*arm64Line{
				{Assembly: "mov\tx0, x1", Binary: "aa0103e0"},
			},
			wantBin: []string{"aa0103e0"},
		},
		{
			name: "empty binary passthrough",
			input: []*arm64Line{
				{Assembly: "mova\tz0.s, p0/m, za0h.s[w12, 0]", Binary: ""},
			},
			wantBin: []string{""},
		},
		{
			name: "empty assembly passthrough",
			input: []*arm64Line{
				{Assembly: "", Binary: ""},
			},
			wantBin: []string{""},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := fixMOVAEncoding(tt.input)
			if len(result) != len(tt.wantBin) {
				t.Fatalf("got %d lines, want %d", len(result), len(tt.wantBin))
			}
			for i := range result {
				if result[i].Binary != tt.wantBin[i] {
					t.Errorf("line %d: binary = %q, want %q", i, result[i].Binary, tt.wantBin[i])
				}
			}
		})
	}
}

func TestEnsureSmstopBeforeRet(t *testing.T) {
	tests := []struct {
		name    string
		input   []*arm64Line
		wantAsm []string
	}{
		{
			name: "smstart present - smstop inserted before ret",
			input: []*arm64Line{
				{Assembly: "smstart\tsm"},
				{Assembly: "fadd\tz0.s, z1.s, z2.s"},
				{Assembly: "ret"},
			},
			wantAsm: []string{"smstart\tsm", "fadd\tz0.s, z1.s, z2.s", "smstop\tsm", "ret"},
		},
		{
			name: "no duplicate smstop",
			input: []*arm64Line{
				{Assembly: "smstart\tsm"},
				{Assembly: "fadd\tz0.s, z1.s, z2.s"},
				{Assembly: "smstop\tsm"},
				{Assembly: "ret"},
			},
			wantAsm: []string{"smstart\tsm", "fadd\tz0.s, z1.s, z2.s", "smstop\tsm", "ret"},
		},
		{
			name: "labels on ret moved to smstop",
			input: []*arm64Line{
				{Assembly: "smstart\tsm"},
				{Assembly: "fadd\tz0.s, z1.s, z2.s"},
				{Labels: []string{"BB0_exit"}, Assembly: "ret"},
			},
			wantAsm: []string{"smstart\tsm", "fadd\tz0.s, z1.s, z2.s", "smstop\tsm", "ret"},
		},
		{
			name: "multiple rets",
			input: []*arm64Line{
				{Assembly: "smstart\tsm"},
				{Assembly: "ret"},
				{Assembly: "fadd\tz0.s, z1.s, z2.s"},
				{Assembly: "ret"},
			},
			wantAsm: []string{"smstart\tsm", "smstop\tsm", "ret", "fadd\tz0.s, z1.s, z2.s", "smstop\tsm", "ret"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ensureSmstopBeforeRet(tt.input)
			gotAsm := make([]string, len(result))
			for i, line := range result {
				gotAsm[i] = line.Assembly
			}
			if len(gotAsm) != len(tt.wantAsm) {
				t.Fatalf("got %d lines %v, want %d lines %v", len(gotAsm), gotAsm, len(tt.wantAsm), tt.wantAsm)
			}
			for i := range gotAsm {
				if gotAsm[i] != tt.wantAsm[i] {
					t.Errorf("line %d: asm = %q, want %q", i, gotAsm[i], tt.wantAsm[i])
				}
			}
			// For "labels on ret" case, verify labels were moved to smstop line
			if tt.name == "labels on ret moved to smstop" {
				// smstop should have the label
				smstopLine := result[2]
				if len(smstopLine.Labels) != 1 || smstopLine.Labels[0] != "BB0_exit" {
					t.Errorf("smstop labels = %v, want [BB0_exit]", smstopLine.Labels)
				}
				// ret should have no labels
				retLine := result[3]
				if len(retLine.Labels) != 0 {
					t.Errorf("ret labels = %v, want []", retLine.Labels)
				}
			}
		})
	}
}

func TestInjectStreamingMode(t *testing.T) {
	tests := []struct {
		name    string
		input   []*arm64Line
		wantAsm []string
	}{
		{
			name: "no SVE - unchanged",
			input: []*arm64Line{
				{Assembly: "mov\tx0, x1"},
				{Assembly: "ret"},
			},
			wantAsm: []string{"mov\tx0, x1", "ret"},
		},
		{
			name: "simple flow - smstart/smstop injected",
			input: []*arm64Line{
				{Assembly: "ld1w\t{z0.s}, p0/z, [x0]"},
				{Assembly: "fadd\tz0.s, z1.s, z2.s"},
				{Assembly: "ret"},
			},
			wantAsm: []string{"smstart\tsm", "ld1w\t{z0.s}, p0/z, [x0]", "fadd\tz0.s, z1.s, z2.s", "smstop\tsm", "ret"},
		},
		{
			name: "existing smstart - delegates to ensureSmstopBeforeRet",
			input: []*arm64Line{
				{Assembly: "smstart\tsm"},
				{Assembly: "fadd\tz0.s, z1.s, z2.s"},
				{Assembly: "ret"},
			},
			wantAsm: []string{"smstart\tsm", "fadd\tz0.s, z1.s, z2.s", "smstop\tsm", "ret"},
		},
		{
			name: "ptrue/cnt moved inside streaming section",
			input: []*arm64Line{
				{Assembly: "ptrue\tp0.s"},
				{Assembly: "cntw\tx9"},
				{Assembly: "ld1w\t{z0.s}, p0/z, [x0]"},
				{Assembly: "ret"},
			},
			wantAsm: []string{"smstart\tsm", "ptrue\tp0.s", "cntw\tx9", "ld1w\t{z0.s}, p0/z, [x0]", "smstop\tsm", "ret"},
		},
		{
			name: "preamble kept outside streaming mode",
			input: []*arm64Line{
				{Assembly: "mov\tx9, x0"},
				{Assembly: "ptrue\tp0.s"},
				{Assembly: "ld1w\t{z0.s}, p0/z, [x0]"},
				{Assembly: "ret"},
			},
			wantAsm: []string{"mov\tx9, x0", "smstart\tsm", "ptrue\tp0.s", "ld1w\t{z0.s}, p0/z, [x0]", "smstop\tsm", "ret"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := injectStreamingMode(tt.input)
			gotAsm := make([]string, len(result))
			for i, line := range result {
				gotAsm[i] = line.Assembly
			}
			if len(gotAsm) != len(tt.wantAsm) {
				t.Fatalf("got %d lines %v, want %d lines %v", len(gotAsm), gotAsm, len(tt.wantAsm), tt.wantAsm)
			}
			for i := range gotAsm {
				if gotAsm[i] != tt.wantAsm[i] {
					t.Errorf("line %d: asm = %q, want %q\ngot:  %v\nwant: %v", i, gotAsm[i], tt.wantAsm[i], gotAsm, tt.wantAsm)
				}
			}
		})
	}
}

func TestTransformSVEFunction(t *testing.T) {
	tests := []struct {
		name    string
		input   []*arm64Line
		wantAsm []string
	}{
		{
			name: "no SVE - passthrough",
			input: []*arm64Line{
				{Assembly: "mov\tx0, x1"},
				{Assembly: "ret"},
			},
			wantAsm: []string{"mov\tx0, x1", "ret"},
		},
		{
			name: "full pipeline: fix MOVA + transform forbidden + inject streaming",
			input: []*arm64Line{
				{Assembly: "movi\td0, #0", Binary: "2f00e400"},
				{Assembly: "ld1w\t{z0.s}, p0/z, [x0]", Binary: "a5404000"},
				{Assembly: "ret", Binary: "d65f03c0"},
			},
			// After transformForbiddenInstructions: movi -> fmov (non-SVE, stays in preamble)
			// After injectStreamingMode: fmov stays outside, smstart before ld1w, smstop before ret
			wantAsm: []string{"fmov\ts0, wzr", "smstart\tsm", "ld1w\t{z0.s}, p0/z, [x0]", "smstop\tsm", "ret"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := TransformSVEFunction(tt.input)
			gotAsm := make([]string, len(result))
			for i, line := range result {
				gotAsm[i] = line.Assembly
			}
			if len(gotAsm) != len(tt.wantAsm) {
				t.Fatalf("got %d lines %v, want %d lines %v", len(gotAsm), gotAsm, len(tt.wantAsm), tt.wantAsm)
			}
			for i := range gotAsm {
				if gotAsm[i] != tt.wantAsm[i] {
					t.Errorf("line %d: asm = %q, want %q", i, gotAsm[i], tt.wantAsm[i])
				}
			}
		})
	}
}

func TestIsSVEType(t *testing.T) {
	tests := []struct {
		typ  string
		want bool
	}{
		{"svint32_t", true},
		{"svfloat32_t", true},
		{"svbool_t", true},
		{"svuint8_t", true},
		{"svbfloat16_t", true},
		{"int32x4_t", false},
		{"float", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.typ, func(t *testing.T) {
			if got := IsSVEType(tt.typ); got != tt.want {
				t.Errorf("IsSVEType(%q) = %v, want %v", tt.typ, got, tt.want)
			}
		})
	}
}

func TestInjectStreamingModeConservative(t *testing.T) {
	tests := []struct {
		name     string
		input    []*arm64Line
		firstSVE int
		wantAsm  []string
	}{
		{
			name: "basic - smstart before firstSVE and smstop before ret",
			input: []*arm64Line{
				{Assembly: "mov\tx0, x1"},
				{Assembly: "ld1w\t{z0.s}, p0/z, [x0]"},
				{Assembly: "ret"},
			},
			firstSVE: 1,
			wantAsm:  []string{"mov\tx0, x1", "smstart\tsm", "ld1w\t{z0.s}, p0/z, [x0]", "smstop\tsm", "ret"},
		},
		{
			name: "branch bypass - smstart at bypass SVE too",
			input: []*arm64Line{
				// Line 0: branch that bypasses firstSVE (targets BB0_2 at line 4)
				{Assembly: "b.eq\tLBB0_2"},
				// Line 1: firstSVE on the non-bypass path
				{Assembly: "ld1w\t{z0.s}, p0/z, [x0]"},
				{Assembly: "ret"},
				// Line 3: label target (non-SVE preamble in bypass block)
				{Labels: []string{"BB0_2"}, Assembly: "mov\tx9, x1"},
				// Line 4: SVE in bypass block
				{Assembly: "st1w\t{z1.s}, p0, [x1]"},
				{Assembly: "ret"},
			},
			firstSVE: 1,
			wantAsm: []string{
				"b.eq\tLBB0_2",
				"smstart\tsm", "ld1w\t{z0.s}, p0/z, [x0]",
				"smstop\tsm", "ret",
				"mov\tx9, x1",
				"smstart\tsm", "st1w\t{z1.s}, p0, [x1]",
				"smstop\tsm", "ret",
			},
		},
		{
			name: "fallthrough past label finds SVE instruction",
			input: []*arm64Line{
				// Line 0: branch targeting BB0_2
				{Assembly: "b.ne\tLBB0_2"},
				// Line 1: firstSVE
				{Assembly: "fadd\tz0.s, z1.s, z2.s"},
				{Assembly: "ret"},
				// Line 3: branch target label, no SVE here
				{Labels: []string{"BB0_2"}, Assembly: "mov\tx0, #0"},
				// Line 4: another label (fallthrough), still no SVE
				{Labels: []string{"BB0_3"}, Assembly: "add\tx1, x1, #1"},
				// Line 5: SVE reached via fallthrough
				{Assembly: "ld1w\t{z0.s}, p0/z, [x0]"},
				{Assembly: "ret"},
			},
			firstSVE: 1,
			wantAsm: []string{
				"b.ne\tLBB0_2",
				"smstart\tsm", "fadd\tz0.s, z1.s, z2.s",
				"smstop\tsm", "ret",
				"mov\tx0, #0",
				"add\tx1, x1, #1",
				"smstart\tsm", "ld1w\t{z0.s}, p0/z, [x0]",
				"smstop\tsm", "ret",
			},
		},
		{
			name: "unconditional branch stops scan",
			input: []*arm64Line{
				{Assembly: "b.eq\tLBB0_2"},
				// firstSVE
				{Assembly: "ld1w\t{z0.s}, p0/z, [x0]"},
				{Assembly: "ret"},
				// Branch target: non-SVE then unconditional branch
				{Labels: []string{"BB0_2"}, Assembly: "mov\tx0, #0"},
				{Assembly: "b\tLBB0_3"},
				// SVE after unconditional branch - should NOT get smstart
				{Labels: []string{"BB0_3"}, Assembly: "fadd\tz0.s, z1.s, z2.s"},
				{Assembly: "ret"},
			},
			firstSVE: 1,
			wantAsm: []string{
				"b.eq\tLBB0_2",
				"smstart\tsm", "ld1w\t{z0.s}, p0/z, [x0]",
				"smstop\tsm", "ret",
				"mov\tx0, #0",
				"b\tLBB0_3",
				// No smstart here - unreachable via fallthrough from bypass
				"fadd\tz0.s, z1.s, z2.s",
				"smstop\tsm", "ret",
			},
		},
		{
			name: "ret stops scan without SVE",
			input: []*arm64Line{
				{Assembly: "cbz\tx0, LBB0_2"},
				// firstSVE
				{Assembly: "ld1w\t{z0.s}, p0/z, [x0]"},
				{Assembly: "ret"},
				// Branch target: no SVE before ret
				{Labels: []string{"BB0_2"}, Assembly: "mov\tx0, #0"},
				{Assembly: "ret"},
			},
			firstSVE: 1,
			wantAsm: []string{
				"cbz\tx0, LBB0_2",
				"smstart\tsm", "ld1w\t{z0.s}, p0/z, [x0]",
				"smstop\tsm", "ret",
				// No smstart - bypass path has no SVE
				"mov\tx0, #0",
				"smstop\tsm", "ret",
			},
		},
		{
			name: "labels on ret moved to smstop in conservative mode",
			input: []*arm64Line{
				{Assembly: "ld1w\t{z0.s}, p0/z, [x0]"},
				{Labels: []string{"BB0_exit"}, Assembly: "ret"},
			},
			firstSVE: 0,
			wantAsm: []string{
				"smstart\tsm", "ld1w\t{z0.s}, p0/z, [x0]",
				"smstop\tsm", "ret",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := analyzeSVEContext(tt.input)
			ctx.FirstSVE = tt.firstSVE
			result := injectStreamingModeConservative(tt.input, ctx)
			gotAsm := make([]string, len(result))
			for i, line := range result {
				gotAsm[i] = line.Assembly
			}
			if len(gotAsm) != len(tt.wantAsm) {
				t.Fatalf("got %d lines %v, want %d lines %v", len(gotAsm), gotAsm, len(tt.wantAsm), tt.wantAsm)
			}
			for i := range gotAsm {
				if gotAsm[i] != tt.wantAsm[i] {
					t.Errorf("line %d: asm = %q, want %q\ngot:  %v\nwant: %v", i, gotAsm[i], tt.wantAsm[i], gotAsm, tt.wantAsm)
				}
			}
		})
	}
}

func TestSVEPrologue(t *testing.T) {
	prologue := SVEPrologue()
	// Check that essential type definitions are present
	for _, typ := range []string{"svint32_t", "svfloat32_t", "svbool_t", "svuint8_t"} {
		if !strings.Contains(prologue, typ) {
			t.Errorf("SVEPrologue() missing type definition for %s", typ)
		}
	}
	// Check that SME attributes are defined
	for _, attr := range []string{"__arm_streaming", "__arm_new_za"} {
		if !strings.Contains(prologue, attr) {
			t.Errorf("SVEPrologue() missing SME attribute %s", attr)
		}
	}
}
