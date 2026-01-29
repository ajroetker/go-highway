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
// It pins the goroutine to its OS thread (preventing migration) and blocks
// SIGURG to prevent Go's async preemption from corrupting the ZA accumulator
// registers during SME streaming mode.
//
// On Linux/ARM64, the kernel may not properly save and restore ZA register
// state when SIGURG interrupts a thread in streaming mode.
// This causes silent data corruption in FMOPA tile computations.
//
// No mutex is needed because SME state (ZA registers) is per-thread.
// Multiple goroutines can safely execute SME on different OS threads in parallel.
//
// Returns a cleanup function that must be deferred:
//
//	defer asm.SMEGuard()()
func SMEGuard() func() {
	runtime.LockOSThread()
	// Block SIGURG to prevent Go's async preemption signal from corrupting
	// ZA register state while the thread is in SME streaming mode.
	var oldmask, newmask uint64
	newmask = 1 << (sigURG - 1)
	// Linux uses rt_sigprocmask with 8-byte signal sets
	syscall.RawSyscall6(syscall.SYS_RT_SIGPROCMASK, sigBlock,
		uintptr(unsafe.Pointer(&newmask)),
		uintptr(unsafe.Pointer(&oldmask)),
		8, 0, 0)
	return func() {
		// Restore original signal mask (unblocks SIGURG)
		syscall.RawSyscall6(syscall.SYS_RT_SIGPROCMASK, sigSetmask,
			uintptr(unsafe.Pointer(&oldmask)),
			0, 8, 0, 0)
		runtime.UnlockOSThread()
	}
}
