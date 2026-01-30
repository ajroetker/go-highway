// Copyright 2025 go-highway Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !noasm && darwin && arm64

package asm

import (
	"runtime"
	"syscall"
	"unsafe"
)

// Signal masking constants for macOS.
const (
	sigBlock   = 1  // SIG_BLOCK
	sigSetmask = 3  // SIG_SETMASK
	sigURG     = 16 // SIGURG on macOS (0x10)
)

// SMEGuard prepares the current goroutine for SME streaming mode execution.
// See matmul/asm.SMEGuard for full documentation.
func SMEGuard() func() {
	runtime.LockOSThread()
	var oldmask, newmask uint32
	newmask = 1 << (sigURG - 1)
	syscall.RawSyscall(syscall.SYS_SIGPROCMASK, sigBlock,
		uintptr(unsafe.Pointer(&newmask)),
		uintptr(unsafe.Pointer(&oldmask)))
	return func() {
		syscall.RawSyscall(syscall.SYS_SIGPROCMASK, sigSetmask,
			uintptr(unsafe.Pointer(&oldmask)),
			0)
		runtime.UnlockOSThread()
	}
}
