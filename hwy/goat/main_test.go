package main

import (
	"testing"
)

func TestSupportedTypes_Float16(t *testing.T) {
	sz, ok := supportedTypes["float16_t"]
	if !ok {
		t.Fatal("float16_t not in supportedTypes")
	}
	if sz != 2 {
		t.Errorf("supportedTypes[float16_t] = %d, want 2", sz)
	}
}

func TestParameterType_String_Float16(t *testing.T) {
	pt := ParameterType{Type: "float16_t", Pointer: false}
	got := pt.String()
	if got != "uint16" {
		t.Errorf("ParameterType{float16_t}.String() = %q, want %q", got, "uint16")
	}
}

func TestParameterType_String_Float16Pointer(t *testing.T) {
	pt := ParameterType{Type: "float16_t", Pointer: true}
	got := pt.String()
	if got != "unsafe.Pointer" {
		t.Errorf("ParameterType{float16_t, Pointer}.String() = %q, want %q", got, "unsafe.Pointer")
	}
}

func TestIsNeonType(t *testing.T) {
	tests := []struct {
		typ  string
		want bool
	}{
		{"int8x16_t", true},
		{"float32x4_t", true},
		{"int8x8_t", true},
		{"float64x1_t", true},
		{"int8x16x2_t", true},
		{"int8x8x2_t", true},
		{"__m128", false},
		{"int64_t", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.typ, func(t *testing.T) {
			if got := IsNeonType(tt.typ); got != tt.want {
				t.Errorf("IsNeonType(%q) = %v, want %v", tt.typ, got, tt.want)
			}
		})
	}
}

func TestNeonTypeSize(t *testing.T) {
	tests := []struct {
		typ  string
		want int
	}{
		{"int8x16_t", 16},
		{"float32x4_t", 16},
		{"int8x8_t", 8},
		{"float64x1_t", 8},
		{"int8x16x2_t", 32},
		{"int8x16x3_t", 48},
		{"int8x16x4_t", 64},
		{"int8x8x2_t", 16},
		{"int8x8x3_t", 24},
		{"int8x8x4_t", 32},
		{"__m128", 0},
		{"int64_t", 0},
	}
	for _, tt := range tests {
		t.Run(tt.typ, func(t *testing.T) {
			if got := NeonTypeSize(tt.typ); got != tt.want {
				t.Errorf("NeonTypeSize(%q) = %d, want %d", tt.typ, got, tt.want)
			}
		})
	}
}

func TestNeonVectorCount(t *testing.T) {
	tests := []struct {
		typ  string
		want int
	}{
		{"int8x16_t", 1},
		{"float32x4_t", 1},
		{"int8x8_t", 1},
		{"float64x1_t", 1},
		{"int8x16x2_t", 2},
		{"int8x16x3_t", 3},
		{"int8x16x4_t", 4},
		{"int8x8x2_t", 2},
		{"int8x8x3_t", 3},
		{"int8x8x4_t", 4},
		{"__m128", 0},
		{"int64_t", 0},
		{"", 0},
	}
	for _, tt := range tests {
		t.Run(tt.typ, func(t *testing.T) {
			if got := NeonVectorCount(tt.typ); got != tt.want {
				t.Errorf("NeonVectorCount(%q) = %d, want %d", tt.typ, got, tt.want)
			}
		})
	}
}

func TestIsNeon64Type(t *testing.T) {
	tests := []struct {
		typ  string
		want bool
	}{
		{"int8x8_t", true},
		{"float64x1_t", true},
		{"int8x8x2_t", true},
		{"int8x16_t", false},
		{"float32x4_t", false},
		{"int8x16x2_t", false},
		{"__m128", false},
	}
	for _, tt := range tests {
		t.Run(tt.typ, func(t *testing.T) {
			if got := IsNeon64Type(tt.typ); got != tt.want {
				t.Errorf("IsNeon64Type(%q) = %v, want %v", tt.typ, got, tt.want)
			}
		})
	}
}

func TestIsX86SIMDType(t *testing.T) {
	tests := []struct {
		typ  string
		want bool
	}{
		{"__m128", true},
		{"__m128d", true},
		{"__m128i", true},
		{"__m256", true},
		{"__m256d", true},
		{"__m256i", true},
		{"__m512", true},
		{"__m512d", true},
		{"__m512i", true},
		{"int8x16_t", false},
		{"float", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.typ, func(t *testing.T) {
			if got := IsX86SIMDType(tt.typ); got != tt.want {
				t.Errorf("IsX86SIMDType(%q) = %v, want %v", tt.typ, got, tt.want)
			}
		})
	}
}

func TestX86SIMDTypeSize(t *testing.T) {
	tests := []struct {
		typ  string
		want int
	}{
		{"__m128", 16},
		{"__m128d", 16},
		{"__m256", 32},
		{"__m256i", 32},
		{"__m512", 64},
		{"__m512d", 64},
		{"int8x16_t", 0},
		{"float", 0},
	}
	for _, tt := range tests {
		t.Run(tt.typ, func(t *testing.T) {
			if got := X86SIMDTypeSize(tt.typ); got != tt.want {
				t.Errorf("X86SIMDTypeSize(%q) = %d, want %d", tt.typ, got, tt.want)
			}
		})
	}
}

func TestX86SIMDAlignment(t *testing.T) {
	tests := []struct {
		typ  string
		want int
	}{
		{"__m128", 16},
		{"__m256", 32},
		{"__m512", 64},
		{"int8x16_t", 0},
		{"float", 0},
	}
	for _, tt := range tests {
		t.Run(tt.typ, func(t *testing.T) {
			if got := X86SIMDAlignment(tt.typ); got != tt.want {
				t.Errorf("X86SIMDAlignment(%q) = %d, want %d", tt.typ, got, tt.want)
			}
		})
	}
}

func TestArgsContainSysroot(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want bool
	}{
		{"empty", nil, false},
		{"no sysroot", []string{"-O2", "-mno-red-zone"}, false},
		{"--sysroot=path", []string{"-O2", "--sysroot=/usr/aarch64-linux-gnu"}, true},
		{"--sysroot separate", []string{"-O2", "--sysroot", "/usr/aarch64-linux-gnu"}, true},
		{"prefix match only", []string{"-O2", "--sysrootfoo"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := argsContainSysroot(tt.args); got != tt.want {
				t.Errorf("argsContainSysroot(%v) = %v, want %v", tt.args, got, tt.want)
			}
		})
	}
}

func TestParameterType_String_ExistingTypes(t *testing.T) {
	tests := []struct {
		ptype   string
		pointer bool
		want    string
	}{
		{"float", false, "float32"},
		{"double", false, "float64"},
		{"int32_t", false, "int32"},
		{"int64_t", false, "int64"},
		{"long", false, "int64"},
		{"_Bool", false, "bool"},
		{"float16_t", false, "uint16"},
		{"float", true, "unsafe.Pointer"},
	}

	for _, tt := range tests {
		t.Run(tt.ptype, func(t *testing.T) {
			pt := ParameterType{Type: tt.ptype, Pointer: tt.pointer}
			got := pt.String()
			if got != tt.want {
				t.Errorf("ParameterType{%q, ptr=%v}.String() = %q, want %q",
					tt.ptype, tt.pointer, got, tt.want)
			}
		})
	}
}
