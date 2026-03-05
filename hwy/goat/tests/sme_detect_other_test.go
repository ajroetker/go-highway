//go:build arm64 && !darwin && !linux

package tests

func hasSMESupport() bool { return false }
