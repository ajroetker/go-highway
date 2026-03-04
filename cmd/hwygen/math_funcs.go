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

package main

import "strings"

// baseVecMathFuncs maps contrib/math Base*Vec function names to their C stdlib equivalents.
// This is the single source of truth — other maps derive from this one.
var baseVecMathFuncs = map[string]string{
	"BaseExpVec":   "exp",
	"BaseExp2Vec":  "exp2",
	"BaseExp10Vec": "exp10",
	"BaseLogVec":   "log",
	"BaseLog2Vec":  "log2",
	"BaseLog10Vec": "log10",
	"BaseSinVec":   "sin",
	"BaseCosVec":   "cos",
	"BaseTanhVec":  "tanh",
	"BaseSinhVec":  "sinh",
	"BaseCoshVec":  "cosh",
	"BaseAsinhVec": "asinh",
	"BaseAcoshVec": "acosh",
	"BaseAtanhVec": "atanh",
	"BaseErfVec":   "erf",
}

// baseVecMathGoNames returns a map from Base*Vec function names to Go stdlib names (capitalized).
func baseVecMathGoNames() map[string]string {
	m := make(map[string]string, len(baseVecMathFuncs))
	for k, v := range baseVecMathFuncs {
		m[k] = strings.ToUpper(v[:1]) + v[1:]
	}
	return m
}
