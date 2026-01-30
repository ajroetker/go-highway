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

//go:build !noasm && linux && arm64

package asm

import (
	"runtime"
	"syscall"
	"unsafe"
)

// Signal masking constants for Linux.
const (
	sigBlock   = 0  // SIG_BLOCK
	sigSetmask = 2  // SIG_SETMASK
	sigURG     = 23 // SIGURG on Linux
)

// SMEGuard prepares the current goroutine for SME streaming mode execution.
// See matmul/asm.SMEGuard for full documentation.
func SMEGuard() func() {
	runtime.LockOSThread()
	var oldmask, newmask uint64
	newmask = 1 << (sigURG - 1)
	syscall.RawSyscall6(syscall.SYS_RT_SIGPROCMASK, sigBlock,
		uintptr(unsafe.Pointer(&newmask)),
		uintptr(unsafe.Pointer(&oldmask)),
		8, 0, 0)
	return func() {
		syscall.RawSyscall6(syscall.SYS_RT_SIGPROCMASK, sigSetmask,
			uintptr(unsafe.Pointer(&oldmask)),
			0, 8, 0, 0)
		runtime.UnlockOSThread()
	}
}
