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

import (
	"slices"
	"sort"
)

// FusionRule defines a pattern for fusing operations.
type FusionRule struct {
	// Name identifies this rule for debugging.
	Name string

	// Priority determines application order (higher = applied first).
	Priority int

	// Match checks if this rule applies to a sequence of nodes.
	Match func(nodes []*IRNode) bool

	// CanFuse performs deeper validation after Match succeeds.
	CanFuse func(producer, consumer *IRNode) bool

	// Apply performs the fusion transformation.
	Apply func(fn *IRFunction, nodes []*IRNode) *FusionGroup
}

// builtinRules are the default fusion rules.
var builtinRules = []FusionRule{
	{
		Name:     "Elem+Elem",
		Priority: 10,
		Match:    matchElemElem,
		CanFuse:  canFuseElemElem,
		Apply:    applyElemElem,
	},
	{
		Name:     "Elem+Reduce",
		Priority: 20,
		Match:    matchElemReduce,
		CanFuse:  canFuseElemReduce,
		Apply:    applyElemReduce,
	},
	{
		Name:     "AllocElim",
		Priority: 30,
		Match:    matchAllocElim,
		CanFuse:  canFuseAllocElim,
		Apply:    applyAllocElim,
	},
	{
		Name:     "Load+Elem",
		Priority: 5,
		Match:    matchLoadElem,
		CanFuse:  canFuseLoadElem,
		Apply:    applyLoadElem,
	},
	{
		Name:     "Elem+Store",
		Priority: 5,
		Match:    matchElemStore,
		CanFuse:  canFuseElemStore,
		Apply:    applyElemStore,
	},
}

// ApplyFusionRules runs the fusion pass on an IRFunction.
// It applies rules in priority order until no more fusions are possible.
func ApplyFusionRules(fn *IRFunction) {
	// First, run analysis to populate producer-consumer relationships
	Analyze(fn)

	// Sort rules by priority (descending)
	rules := make([]FusionRule, len(builtinRules))
	copy(rules, builtinRules)
	sort.Slice(rules, func(i, j int) bool {
		return rules[i].Priority > rules[j].Priority
	})

	// Worklist algorithm: keep applying rules until fixpoint
	changed := true
	nextGroupID := 0

	for changed {
		changed = false

		// Find fusion candidates
		candidates := FindFusionCandidates(fn)

		// Try to apply rules
		for _, candidate := range candidates {
			// Skip already fused nodes
			if candidate.Producer.FusionGroup >= 0 || candidate.Consumer.FusionGroup >= 0 {
				continue
			}

			// Find applicable rule
			for _, rule := range rules {
				nodes := []*IRNode{candidate.Producer, candidate.Consumer}
				if rule.Match(nodes) && rule.CanFuse(candidate.Producer, candidate.Consumer) {
					// Apply the rule
					group := rule.Apply(fn, nodes)
					if group != nil {
						group.ID = nextGroupID
						group.Pattern = rule.Name
						nextGroupID++

						// Mark nodes as fused
						for _, id := range group.Members {
							if node := fn.GetNode(id); node != nil {
								node.FusionGroup = group.ID
							}
						}

						// Mark root
						if root := fn.GetNode(group.Root); root != nil {
							root.IsFusionRoot = true
						}

						fn.FusionGroups = append(fn.FusionGroups, *group)
						changed = true
						break
					}
				}
			}
		}
	}

	// Extend fusion groups: try to add more nodes to existing groups
	extendFusionGroups(fn)
}

// extendFusionGroups tries to add more nodes to existing fusion groups.
func extendFusionGroups(fn *IRFunction) {
	for i := range fn.FusionGroups {
		group := &fn.FusionGroups[i]
		extended := true

		for extended {
			extended = false

			// Look at producers of current members
			for _, memberID := range group.Members {
				member := fn.GetNode(memberID)
				if member == nil {
					continue
				}

				for _, producer := range member.Producers {
					if producer.FusionGroup >= 0 {
						continue // Already fused
					}

					// Check if producer can join this group
					if canExtendGroup(group, producer) {
						group.Members = append(group.Members, producer.ID)
						producer.FusionGroup = group.ID
						extended = true
					}
				}
			}

			// Look at consumers of current members
			for _, memberID := range group.Members {
				member := fn.GetNode(memberID)
				if member == nil {
					continue
				}

				for _, consumer := range member.Consumers {
					if consumer.FusionGroup >= 0 {
						continue
					}

					if canExtendGroup(group, consumer) {
						group.Members = append(group.Members, consumer.ID)
						consumer.FusionGroup = group.ID

						// Update root if this is a later operation
						if consumer.ID > group.Root {
							if oldRoot := fn.GetNode(group.Root); oldRoot != nil {
								oldRoot.IsFusionRoot = false
							}
							group.Root = consumer.ID
							consumer.IsFusionRoot = true
						}

						extended = true
					}
				}
			}
		}
	}
}

// canExtendGroup checks if a node can be added to an existing fusion group.
func canExtendGroup(group *FusionGroup, node *IRNode) bool {
	// Must be elementwise or reduction
	if node.Kind != OpKindElementwise && node.Kind != OpKindReduction {
		return false
	}

	// Must have compatible loop range
	if group.LoopRange != nil && node.LoopRange != nil {
		if !group.LoopRange.Same(node.LoopRange) {
			return false
		}
	}

	return true
}

// Rule matchers

func matchElemElem(nodes []*IRNode) bool {
	if len(nodes) != 2 {
		return false
	}
	return nodes[0].Kind == OpKindElementwise && nodes[1].Kind == OpKindElementwise
}

func matchElemReduce(nodes []*IRNode) bool {
	if len(nodes) != 2 {
		return false
	}
	return nodes[0].Kind == OpKindElementwise && nodes[1].Kind == OpKindReduction
}

func matchAllocElim(nodes []*IRNode) bool {
	if len(nodes) != 2 {
		return false
	}
	return nodes[0].Kind == OpKindAlloc
}

func matchLoadElem(nodes []*IRNode) bool {
	if len(nodes) != 2 {
		return false
	}
	return nodes[0].Kind == OpKindLoad && nodes[1].Kind == OpKindElementwise
}

func matchElemStore(nodes []*IRNode) bool {
	if len(nodes) != 2 {
		return false
	}
	return nodes[0].Kind == OpKindElementwise && nodes[1].Kind == OpKindStore
}

// Rule validators

func canFuseElemElem(producer, consumer *IRNode) bool {
	// Must share same loop range
	if !loopRangesCompatible(producer.LoopRange, consumer.LoopRange) {
		return false
	}

	// Producer must be consumed only by this consumer (or be multi-use safe)
	// For now, allow all
	return true
}

func canFuseElemReduce(producer, consumer *IRNode) bool {
	// Must share same loop range
	if !loopRangesCompatible(producer.LoopRange, consumer.LoopRange) {
		return false
	}

	// Producer should be single-use (consumed only by reduction)
	if !producer.HasSingleConsumer() {
		return false
	}

	return true
}

func canFuseAllocElim(producer, consumer *IRNode) bool {
	// Allocation must be single-use
	return producer.HasSingleConsumer()
}

func canFuseLoadElem(producer, consumer *IRNode) bool {
	// Load must be single-use
	return producer.HasSingleConsumer()
}

func canFuseElemStore(producer, consumer *IRNode) bool {
	// Always fusible if loop ranges match
	return loopRangesCompatible(producer.LoopRange, consumer.LoopRange)
}

// Rule applications

func applyElemElem(fn *IRFunction, nodes []*IRNode) *FusionGroup {
	producer, consumer := nodes[0], nodes[1]

	group := &FusionGroup{
		Root:      consumer.ID,
		Members:   []int{producer.ID, consumer.ID},
		LoopRange: producer.LoopRange.Clone(),
	}

	return group
}

func applyElemReduce(fn *IRFunction, nodes []*IRNode) *FusionGroup {
	producer, consumer := nodes[0], nodes[1]

	group := &FusionGroup{
		Root:      consumer.ID,
		Members:   []int{producer.ID, consumer.ID},
		LoopRange: producer.LoopRange.Clone(),
	}

	return group
}

func applyAllocElim(fn *IRFunction, nodes []*IRNode) *FusionGroup {
	alloc, user := nodes[0], nodes[1]

	group := &FusionGroup{
		Root:             user.ID,
		Members:          []int{alloc.ID, user.ID},
		EliminatedAllocs: []int{alloc.ID},
	}

	// Copy loop range from user
	if user.LoopRange != nil {
		group.LoopRange = user.LoopRange.Clone()
	}

	return group
}

func applyLoadElem(fn *IRFunction, nodes []*IRNode) *FusionGroup {
	load, elem := nodes[0], nodes[1]

	group := &FusionGroup{
		Root:      elem.ID,
		Members:   []int{load.ID, elem.ID},
		LoopRange: elem.LoopRange.Clone(),
	}

	return group
}

func applyElemStore(fn *IRFunction, nodes []*IRNode) *FusionGroup {
	elem, store := nodes[0], nodes[1]

	group := &FusionGroup{
		Root:      store.ID,
		Members:   []int{elem.ID, store.ID},
		LoopRange: elem.LoopRange.Clone(),
	}

	return group
}

// FusionStats returns statistics about fusion results.
type FusionStats struct {
	// OriginalPasses is the estimated memory passes before fusion.
	OriginalPasses int

	// FusedPasses is the estimated memory passes after fusion.
	FusedPasses int

	// EliminatedAllocs is the number of allocations eliminated.
	EliminatedAllocs int

	// FusionGroups is the number of fusion groups created.
	FusionGroups int
}

// ComputeFusionStats computes statistics about fusion effectiveness.
func ComputeFusionStats(fn *IRFunction) FusionStats {
	stats := FusionStats{}

	// Count original passes (loops + allocations)
	var countOps func([]*IRNode)
	countOps = func(nodes []*IRNode) {
		for _, node := range nodes {
			if node.Kind == OpKindLoop {
				stats.OriginalPasses++
			}
			if node.Kind == OpKindAlloc {
				stats.OriginalPasses++
			}
			countOps(node.Children)
		}
	}
	countOps(fn.Operations)

	// Count fused passes (fusion groups + unfused loops)
	stats.FusionGroups = len(fn.FusionGroups)

	// Each fusion group replaces multiple passes with one
	fusedLoops := 0
	for _, group := range fn.FusionGroups {
		// Count how many loops are in this group
		loopsInGroup := 0
		for _, id := range group.Members {
			if node := fn.GetNode(id); node != nil && node.Kind == OpKindLoop {
				loopsInGroup++
			}
		}
		if loopsInGroup > 0 {
			fusedLoops += loopsInGroup - 1 // Saved loops
		}

		stats.EliminatedAllocs += len(group.EliminatedAllocs)
	}

	stats.FusedPasses = stats.OriginalPasses - fusedLoops - stats.EliminatedAllocs

	return stats
}

// OptimizeSoftmax applies softmax-specific fusion optimizations.
// This implements the 5→3 pass reduction from the plan:
//   Before: max, alloc shifted, loop shift, loop exp+sum, loop normalize
//   After:  max, loop exp+sum (fused), loop normalize
func OptimizeSoftmax(fn *IRFunction) {
	// Find the softmax pattern:
	// 1. ReduceMax → max
	// 2. Alloc shifted
	// 3. Loop: shifted[i] = input[i] - max
	// 4. Loop (BaseApply): output[i] = exp(shifted[i]); sum += output[i]
	// 5. Loop: output[i] *= 1/sum

	// Step 1: Identify allocations that are consumed by a single loop
	// and produced by another loop
	allocsToEliminate := IdentifyAllocationsToEliminate(fn)

	// Step 2: For each such allocation, try to fuse the producer and consumer loops
	for _, alloc := range allocsToEliminate {
		// Find the producer loop (writes to alloc)
		var producerLoop *IRNode
		for _, consumer := range alloc.Consumers {
			if consumer.Kind == OpKindLoop {
				producerLoop = consumer
				break
			}
		}

		if producerLoop == nil {
			continue
		}

		// Find consumer loop (reads from alloc)
		var consumerLoop *IRNode
		for _, op := range fn.Operations {
			if op.Kind == OpKindLoop && op.ID != producerLoop.ID {
				// Check if this loop reads from the allocation
				for _, child := range op.Children {
					if child.Kind == OpKindLoad {
						if slices.Contains(child.InputNames, alloc.Outputs[0]) {
							consumerLoop = op
							break
						}
					}
				}
			}
		}

		if consumerLoop == nil {
			continue
		}

		// Create a fusion group for these loops
		if producerLoop.LoopRange.Same(consumerLoop.LoopRange) {
			nextID := len(fn.FusionGroups)
			group := FusionGroup{
				ID:               nextID,
				Root:             consumerLoop.ID,
				Members:          []int{alloc.ID, producerLoop.ID, consumerLoop.ID},
				Pattern:          "SoftmaxFusion",
				LoopRange:        producerLoop.LoopRange.Clone(),
				EliminatedAllocs: []int{alloc.ID},
			}

			// Mark nodes as fused
			alloc.FusionGroup = nextID
			producerLoop.FusionGroup = nextID
			consumerLoop.FusionGroup = nextID
			consumerLoop.IsFusionRoot = true

			fn.FusionGroups = append(fn.FusionGroups, group)
		}
	}
}
