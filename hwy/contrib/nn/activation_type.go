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

package nn

// ActivationType specifies which activation function to apply after a linear layer.
type ActivationType int

const (
	// ActivationNone applies no activation (identity).
	ActivationNone ActivationType = iota
	// ActivationGelu applies the Gaussian Error Linear Unit activation.
	ActivationGelu
	// ActivationRelu applies the Rectified Linear Unit activation.
	ActivationRelu
	// ActivationSilu applies the Sigmoid Linear Unit (Swish) activation.
	ActivationSilu
	// ActivationTanh applies the hyperbolic tangent activation.
	ActivationTanh
)
