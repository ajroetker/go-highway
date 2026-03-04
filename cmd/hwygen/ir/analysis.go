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

package ir

import "slices"

// Analyze performs data flow analysis on an IRFunction.
// It computes:
// - Producer-consumer relationships between nodes
// - Def-use chains for variables
// - Loop-carried dependencies
//
// This information is used by the fusion pass to identify fusible operations.
func Analyze(fn *IRFunction) {
	// Build def-use chains
	defs := buildDefMap(fn)

	// Link producers and consumers
	linkProducersConsumers(fn, defs)

	// Propagate loop ranges
	propagateLoopRanges(fn)
}

// defInfo tracks where a variable is defined.
type defInfo struct {
	def *IRNode // The node that defines (produces) this variable
}

// buildDefMap builds a map from variable name to definition info.
func buildDefMap(fn *IRFunction) map[string]*defInfo {
	defs := make(map[string]*defInfo)
	WalkNodes(fn.Operations, func(node *IRNode) {
		for _, out := range node.Outputs {
			defs[out] = &defInfo{def: node}
		}
	})
	return defs
}

// linkProducersConsumers establishes producer-consumer relationships.
func linkProducersConsumers(fn *IRFunction, defs map[string]*defInfo) {
	WalkNodes(fn.Operations, func(node *IRNode) {
		// Link inputs (IRNode references) to their producers
		for _, input := range node.Inputs {
			if input != nil {
				input.Consumers = appendUnique(input.Consumers, node)
				node.Producers = appendUnique(node.Producers, input)
			}
		}

		// Link InputNames (variable references) to their producers
		for _, name := range node.InputNames {
			if info, ok := defs[name]; ok && info.def != nil {
				info.def.Consumers = appendUnique(info.def.Consumers, node)
				node.Producers = appendUnique(node.Producers, info.def)
			}
		}
	})
}

// propagateLoopRanges ensures all nodes inside loops have their LoopRange set.
func propagateLoopRanges(fn *IRFunction) {
	var walkNodes func([]*IRNode, *LoopRange)
	walkNodes = func(nodes []*IRNode, enclosingRange *LoopRange) {
		for _, node := range nodes {
			// Propagate enclosing loop range if not already set
			if node.LoopRange == nil && enclosingRange != nil {
				node.LoopRange = enclosingRange.Clone()
			}

			// For loop nodes, propagate their range to children
			if node.Kind == OpKindLoop && len(node.Children) > 0 {
				walkNodes(node.Children, node.LoopRange)
			}
		}
	}
	walkNodes(fn.Operations, nil)
}

// appendUnique appends a node to a slice if not already present.
func appendUnique(slice []*IRNode, node *IRNode) []*IRNode {
	for _, n := range slice {
		if n.ID == node.ID {
			return slice
		}
	}
	return append(slice, node)
}

// FusionCandidate represents a potential fusion opportunity.
type FusionCandidate struct {
	// Producer is the first operation in the chain.
	Producer *IRNode

	// Consumer is the operation that uses Producer's output.
	Consumer *IRNode

	// Pattern describes the type of fusion possible.
	Pattern string

	// Benefit estimates the value of this fusion (higher = better).
	Benefit int
}

// FindFusionCandidates identifies operations that could be fused.
func FindFusionCandidates(fn *IRFunction) []FusionCandidate {
	var candidates []FusionCandidate
	WalkNodes(fn.Operations, func(node *IRNode) {
		for _, consumer := range node.Consumers {
			if candidate := checkFusionPair(node, consumer); candidate != nil {
				candidates = append(candidates, *candidate)
			}
		}
	})
	return candidates
}

// checkFusionPair checks if two nodes can be fused.
func checkFusionPair(producer, consumer *IRNode) *FusionCandidate {
	// Same loop range is required for fusion
	if !loopRangesCompatible(producer.LoopRange, consumer.LoopRange) {
		return nil
	}

	// Check fusion patterns
	switch {
	case producer.Kind == OpKindElementwise && consumer.Kind == OpKindElementwise:
		// Vertical fusion: chain elementwise ops
		return &FusionCandidate{
			Producer: producer,
			Consumer: consumer,
			Pattern:  PatternElemElem,
			Benefit:  10, // Eliminates intermediate store/load
		}

	case producer.Kind == OpKindElementwise && consumer.Kind == OpKindReduction:
		// MapReduce fusion: compute and reduce in one pass
		return &FusionCandidate{
			Producer: producer,
			Consumer: consumer,
			Pattern:  PatternElemReduce,
			Benefit:  20, // Eliminates temp array + extra pass
		}

	case producer.Kind == OpKindAlloc && consumer.Kind == OpKindStore:
		// Allocation followed by store - check if temp can be eliminated
		if producer.HasSingleConsumer() {
			return &FusionCandidate{
				Producer: producer,
				Consumer: consumer,
				Pattern:  PatternAllocElim,
				Benefit:  30, // Eliminates heap allocation
			}
		}

	case producer.Kind == OpKindLoad && consumer.Kind == OpKindElementwise:
		// Load + compute can be fused if load is single-use
		if producer.HasSingleConsumer() {
			return &FusionCandidate{
				Producer: producer,
				Consumer: consumer,
				Pattern:  PatternLoadElem,
				Benefit:  5, // Minor benefit
			}
		}

	case producer.Kind == OpKindElementwise && consumer.Kind == OpKindStore:
		// Compute + store can be fused
		return &FusionCandidate{
			Producer: producer,
			Consumer: consumer,
			Pattern:  PatternElemStore,
			Benefit:  5,
		}
	}

	return nil
}

// loopRangesCompatible checks if two nodes can be fused based on their iteration spaces.
func loopRangesCompatible(a, b *LoopRange) bool {
	// If either is nil, they're not in loops - can't fuse
	if a == nil || b == nil {
		return false
	}

	// Same iteration space
	return a.Same(b)
}

// IdentifyAllocationsToEliminate finds temporary allocations that can be removed.
func IdentifyAllocationsToEliminate(fn *IRFunction) []*IRNode {
	var toEliminate []*IRNode
	WalkNodes(fn.Operations, func(node *IRNode) {
		if node.Kind == OpKindAlloc && canEliminateAlloc(node) {
			toEliminate = append(toEliminate, node)
		}
	})
	return toEliminate
}

// canEliminateAlloc checks if an allocation can be replaced with registers.
// This returns true for same-loop elimination only. For cross-loop elimination,
// use IdentifyCrossLoopAllocations.
func canEliminateAlloc(alloc *IRNode) bool {
	// Allocation must have exactly one consumer (the loop that uses it)
	if len(alloc.Consumers) != 1 {
		return false
	}

	consumer := alloc.Consumers[0]

	// Consumer must be a loop that stores into the allocated array
	// and another operation reads from it in the same loop
	if consumer.Kind != OpKindLoop {
		return false
	}

	// Check if the array is both written and read within the same loop,
	// and not used outside the loop
	arrayName := alloc.Outputs[0]
	var hasWrite, hasRead bool
	for _, child := range consumer.Children {
		if hasWrite && hasRead {
			break
		}
		if child.Kind == OpKindStore && slices.Contains(child.InputNames, arrayName) {
			hasWrite = true
		}
		if child.Kind == OpKindLoad && slices.Contains(child.InputNames, arrayName) {
			hasRead = true
		}
	}

	return hasWrite && hasRead
}

// CrossLoopTempArray represents a temporary array that is written in one loop
// and read in another loop, making it a candidate for cross-loop fusion.
type CrossLoopTempArray struct {
	Alloc        *IRNode // The allocation node
	WriteLoop    *IRNode // Loop that writes to the array
	ReadLoop     *IRNode // Loop that reads from the array
	WriteStore   *IRNode // The store operation in the write loop
	ReadLoad     *IRNode // The load operation in the read loop
	ArrayName    string  // The variable name of the array
}

// IdentifyCrossLoopAllocations finds temporary arrays that are written in one loop
// and read in another, which are candidates for cross-loop fusion.
func IdentifyCrossLoopAllocations(fn *IRFunction) []CrossLoopTempArray {
	var candidates []CrossLoopTempArray

	// Build a map of array name -> allocation
	allocMap := make(map[string]*IRNode)
	for _, node := range fn.AllNodes {
		if node.Kind == OpKindAlloc && len(node.Outputs) > 0 {
			allocMap[node.Outputs[0]] = node
		}
	}

	// For each allocation, find write and read loops
	for arrayName, alloc := range allocMap {
		var writeLoop, readLoop *IRNode
		var writeStore, readLoad *IRNode

		// Scan all loops in the function (including nested)
		WalkNodes(fn.Operations, func(op *IRNode) {
			if op.Kind != OpKindLoop {
				return
			}

			// Check children for both writes and reads in a single pass
			for _, child := range op.Children {
				if child.Kind == OpKindStore && slices.Contains(child.InputNames, arrayName) {
					writeLoop = op
					writeStore = child
				}
				if child.Kind == OpKindLoad && slices.Contains(child.InputNames, arrayName) {
					// Only consider as read loop if different from write loop
					if writeLoop != nil && op.ID != writeLoop.ID {
						readLoop = op
						readLoad = child
					}
				}
			}
		})

		// If we found both write and read loops with same iteration space
		if writeLoop != nil && readLoop != nil {
			if writeLoop.LoopRange != nil && readLoop.LoopRange != nil &&
				writeLoop.LoopRange.Same(readLoop.LoopRange) {
				candidates = append(candidates, CrossLoopTempArray{
					Alloc:      alloc,
					WriteLoop:  writeLoop,
					ReadLoop:   readLoop,
					WriteStore: writeStore,
					ReadLoad:   readLoad,
					ArrayName:  arrayName,
				})
			}
		}
	}

	return candidates
}

// FindLoopChain finds a chain of loops connected by temporary arrays.
// Returns loops in execution order that can be fused together.
func FindLoopChain(fn *IRFunction, startLoop *IRNode, tempArrays []CrossLoopTempArray) []*IRNode {
	chain := []*IRNode{startLoop}
	visited := map[int]bool{startLoop.ID: true}

	// Follow the chain: if this loop writes to a temp array, add the reader
	for {
		currentLoop := chain[len(chain)-1]
		foundNext := false

		for _, temp := range tempArrays {
			if temp.WriteLoop.ID == currentLoop.ID && !visited[temp.ReadLoop.ID] {
				chain = append(chain, temp.ReadLoop)
				visited[temp.ReadLoop.ID] = true
				foundNext = true
				break
			}
		}

		if !foundNext {
			break
		}
	}

	return chain
}

// CountMemoryPasses estimates the number of memory passes for the current IR.
// Used to measure fusion effectiveness.
func CountMemoryPasses(fn *IRFunction) int {
	passes := 0
	WalkNodes(fn.Operations, func(node *IRNode) {
		if node.Kind == OpKindLoop || node.Kind == OpKindAlloc {
			passes++
		}
	})
	return passes
}

// CountAllocations returns the number of heap allocations in the IR.
func CountAllocations(fn *IRFunction) int {
	count := 0
	WalkNodes(fn.Operations, func(node *IRNode) {
		if node.Kind == OpKindAlloc {
			count++
		}
	})
	return count
}
