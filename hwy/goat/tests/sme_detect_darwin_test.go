//go:build darwin && arm64

package tests

import "syscall"

func hasSMESupport() bool {
	val, err := syscall.Sysctl("hw.optional.arm.FEAT_SME")
	if err != nil {
		return false
	}
	return len(val) > 0 && val[0] == 1
}
