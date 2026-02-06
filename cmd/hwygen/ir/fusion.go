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
	"fmt"
	"os"
	"slices"
	"sort"
)

// debugFusion enables debug output for fusion passes.
// Set to true for debugging, false for production.
var debugFusion = os.Getenv("DEBUG_FUSION") != ""

func debugPrint(format string, args ...any) {
	if debugFusion {
		fmt.Printf("[fusion] "+format+"\n", args...)
	}
}

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
//
//	Before: max, alloc shifted, loop shift, loop exp, loop sum, loop normalize
//	After:  max, loop (shift+exp+sum fused), loop normalize
//
// The key optimization is fusing loops connected by temporary arrays like shifted[].
func OptimizeSoftmax(fn *IRFunction) {
	// Step 1: Find cross-loop temporary arrays
	crossLoopTemps := IdentifyCrossLoopAllocations(fn)

	if len(crossLoopTemps) == 0 {
		return
	}

	// Step 2: For each temp array, fuse the write and read loops
	for _, temp := range crossLoopTemps {
		// Skip if already fused
		if temp.WriteLoop.FusionGroup >= 0 || temp.ReadLoop.FusionGroup >= 0 {
			continue
		}

		// Find the full chain of loops connected by temp arrays
		chain := FindLoopChain(fn, temp.WriteLoop, crossLoopTemps)

		if len(chain) < 2 {
			continue
		}

		// Create a fusion group for the entire chain
		nextID := len(fn.FusionGroups)
		members := []int{temp.Alloc.ID}
		eliminatedAllocs := []int{temp.Alloc.ID}

		for _, loop := range chain {
			members = append(members, loop.ID)
			// Also add children of each loop
			for _, child := range loop.Children {
				members = append(members, child.ID)
			}
		}

		// The root is the last loop in the chain
		rootLoop := chain[len(chain)-1]

		group := FusionGroup{
			ID:               nextID,
			Root:             rootLoop.ID,
			Members:          members,
			Pattern:          "CrossLoopFusion",
			LoopRange:        temp.WriteLoop.LoopRange.Clone(),
			EliminatedAllocs: eliminatedAllocs,
		}

		// Mark all nodes as fused
		temp.Alloc.FusionGroup = nextID
		temp.Alloc.IsFusionEliminated = true

		for _, loop := range chain {
			loop.FusionGroup = nextID
			for _, child := range loop.Children {
				child.FusionGroup = nextID
			}
		}
		rootLoop.IsFusionRoot = true

		// Mark the load/store pair for elimination
		if temp.WriteStore != nil {
			temp.WriteStore.IsFusionEliminated = true
		}
		if temp.ReadLoad != nil {
			temp.ReadLoad.IsFusionEliminated = true
		}

		fn.FusionGroups = append(fn.FusionGroups, group)
	}

	// Step 3: Also try to fuse the sum accumulation loop if connected
	// Look for loop that reads from output[] after the exp loop
	fuseSumLoop(fn, crossLoopTemps)
}

// fuseSumLoop tries to fuse the sum accumulation loop into the exp loop.
// Pattern: exp loop writes to output[], sum loop reads from output[] immediately after.
func fuseSumLoop(fn *IRFunction, crossLoopTemps []CrossLoopTempArray) {
	// Find the exp loop (writes to output)
	var expLoop *IRNode
	for _, temp := range crossLoopTemps {
		if temp.ReadLoop != nil {
			// The read loop for shifted[] is the exp loop
			expLoop = temp.ReadLoop
			break
		}
	}

	if expLoop == nil {
		debugPrint("fuseSumLoop: expLoop is nil, returning")
		return
	}
	debugPrint("fuseSumLoop: found expLoop ID=%d", expLoop.ID)

	// Find what array the exp loop writes to (usually "output")
	var outputArrayName string
	for _, child := range expLoop.Children {
		if child.Kind == OpKindStore {
			for _, name := range child.InputNames {
				// The first InputName that's not the loop variable is the array
				if name != expLoop.LoopRange.LoopVar {
					outputArrayName = name
					break
				}
			}
		}
	}

	if outputArrayName == "" {
		debugPrint("fuseSumLoop: outputArrayName is empty, returning")
		return
	}
	debugPrint("fuseSumLoop: outputArrayName=%s", outputArrayName)

	// Find the sum loop (reads from output AND has a reduction)
	var sumLoop *IRNode
	var sumLoad *IRNode
	for _, op := range fn.Operations {
		if op.Kind != OpKindLoop || op.ID == expLoop.ID {
			continue
		}
		if op.FusionGroup >= 0 {
			debugPrint("fuseSumLoop: loop %d already fused (group=%d), skipping", op.ID, op.FusionGroup)
			continue // Already fused
		}

		// Check if this loop reads from output AND has a reduction
		var foundLoad *IRNode
		hasReduction := false
		for _, child := range op.Children {
			if child.Kind == OpKindLoad && slices.Contains(child.InputNames, outputArrayName) {
				foundLoad = child
			}
			if child.Kind == OpKindReduction || child.Op == "Add" || child.Op == "ReduceSum" {
				hasReduction = true
			}
		}

		if foundLoad != nil && hasReduction {
			sumLoop = op
			sumLoad = foundLoad
			debugPrint("fuseSumLoop: found sumLoop=%d (reads from %s, has reduction)", op.ID, outputArrayName)
			break // Found the right loop
		}
	}

	if sumLoop == nil {
		debugPrint("fuseSumLoop: no sumLoop with reduction found, returning")
		return
	}
	if !sumLoop.LoopRange.Same(expLoop.LoopRange) {
		debugPrint("fuseSumLoop: loop ranges don't match, returning")
		return
	}
	debugPrint("fuseSumLoop: sumLoop=%d, loop ranges match, continuing", sumLoop.ID)

	// Find or create the fusion group for expLoop
	var group *FusionGroup
	for i := range fn.FusionGroups {
		if slices.Contains(fn.FusionGroups[i].Members, expLoop.ID) {
			group = &fn.FusionGroups[i]
			break
		}
	}

	if group == nil {
		// Create new group if exp loop wasn't already fused
		nextID := len(fn.FusionGroups)
		group = &FusionGroup{
			ID:        nextID,
			Root:      expLoop.ID,
			Members:   []int{expLoop.ID},
			Pattern:   "ExpSumFusion",
			LoopRange: expLoop.LoopRange.Clone(),
		}
		expLoop.FusionGroup = nextID
		fn.FusionGroups = append(fn.FusionGroups, *group)
		group = &fn.FusionGroups[len(fn.FusionGroups)-1]
	}

	// Add sum loop and its children to the group
	group.Members = append(group.Members, sumLoop.ID)
	sumLoop.FusionGroup = group.ID
	for _, child := range sumLoop.Children {
		if !slices.Contains(group.Members, child.ID) {
			group.Members = append(group.Members, child.ID)
			child.FusionGroup = group.ID
		}
	}

	// Update pattern name
	group.Pattern = "CrossLoopFusion+Sum"

	// Mark the load as eliminated (we read directly from the exp result)
	if sumLoad != nil {
		sumLoad.IsFusionEliminated = true
	}

	// Update root to be the sum loop (later in execution)
	if sumLoop.ID > group.Root {
		if oldRoot := fn.GetNode(group.Root); oldRoot != nil {
			oldRoot.IsFusionRoot = false
		}
		group.Root = sumLoop.ID
		sumLoop.IsFusionRoot = true
	}
}
