//go:build linux && arm64

package tests

import (
	"encoding/binary"
	"os"
)

func hasSMESupport() bool {
	const (
		atHWCAP2  = 26
		hwcap2SME = 1 << 23
	)
	f, err := os.Open("/proc/self/auxv")
	if err != nil {
		return false
	}
	defer f.Close()
	buf := make([]byte, 16)
	for {
		n, err := f.Read(buf)
		if n < 16 || err != nil {
			return false
		}
		key := binary.LittleEndian.Uint64(buf[0:8])
		val := binary.LittleEndian.Uint64(buf[8:16])
		if key == atHWCAP2 {
			return val&hwcap2SME != 0
		}
		if key == 0 {
			return false
		}
	}
}
