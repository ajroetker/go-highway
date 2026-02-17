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
