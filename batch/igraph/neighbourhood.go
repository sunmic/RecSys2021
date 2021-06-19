package main

// #cgo LDFLAGS: -lm -ligraph
// #include <igraph/igraph.h>
// #include "igraph_ext.h"
import "C"
import (
	"fmt"
	"math"
	"math/rand"
)

const NumRelation = 5

func weightedRandomKSample(population int, weights []int32, result *[]int) {
	if population <= 0 {
		panic("Invalid population")
	}

	var relBuckets [5][]int
	var bucketPos [5]int

	// put indices into buckets based on weights
	for index, weight := range weights {
		for i := NumRelation; i >= 0; i-- {
			if weight&(1<<i) != 0 {
				relBuckets[i] = append(relBuckets[i], index)
				break
			}
		}
	}

	// shuffle each bucket
	for _, bucket := range relBuckets {
		if len(bucket) <= 1 {
			continue
		}

		rand.Shuffle(len(bucket), func(i, j int) {
			bucket[i], bucket[j] = bucket[j], bucket[i]
		})
	}

	// pick random elements
	var bucketIndex int
	var realIndex int
	for len(*result) < population {
		bucketIndex = int(math.Round((NumRelation - 1) * math.Pow(rand.Float64(), 0.5)))
		for realIndex = bucketIndex; realIndex >= 0; realIndex-- {
			pos := bucketPos[realIndex]
			if len(relBuckets[realIndex][pos:]) != 0 {
				*result = append(*result, relBuckets[realIndex][pos])
				bucketPos[realIndex]++
				break
			}
		}
	}
}

func randomKSample(max int, population int, result *[]int) {
	if population <= 0 || max < population {
		panic("Invalid population")
	}

	var pickMap = make(map[int]bool)
	var randomValue int
	for len(*result) < population {
		randomValue = rand.Intn(max)
		if !pickMap[randomValue] {
			pickMap[randomValue] = true
			*result = append(*result, randomValue)
		}
	}
}

type NNSampling struct {
	visited     map[int32]bool
	sources     []int32
	targets     []int32
	edgeTypes   []int32
	startVertex int32
	// Temporary storage
	neighbourhoodBuffer []int32
	edgeTypeBuffer      []int32
	randomMask          []int
}

func (sampling *NNSampling) Print() {
	fmt.Printf("Visited: %v\n", sampling.visited)
	fmt.Printf("Sources: %v\n", sampling.sources)
	fmt.Printf("Targets: %v\n", sampling.targets)
	if len(sampling.edgeTypes) != 0 {
		fmt.Printf("Edge types: %v\n", sampling.edgeTypes)
	}
	fmt.Printf("Buffer:  %v\n", sampling.neighbourhoodBuffer)
	fmt.Printf("Mask:    %v\n", sampling.randomMask)
}

func (sampling *NNSampling) Reset() {
	sampling.visited = make(map[int32]bool)
	sampling.sources = sampling.sources[:0]
	sampling.targets = sampling.targets[:0]
	sampling.edgeTypes = sampling.edgeTypes[:0]
	sampling.neighbourhoodBuffer = sampling.neighbourhoodBuffer[:0]
	sampling.edgeTypeBuffer = sampling.edgeTypeBuffer[:0]
	sampling.randomMask = sampling.randomMask[:0]
}

func getEdgeTypeAttribute(graph *C.igraph_t, from int32, to int32) (int32, error) {
	var eid C.int
	var eval C.double
	C.igraph_get_eid(graph, &eid, C.int(from), C.int(to), booltoint(true), booltoint(false))
	if eid != -1 {
		eval = C.igraph_cattribute_EAN(graph, WeightAttributeName, eid)
		return int32(eval), nil
	} else {
		return 0, fmt.Errorf("edge does not exist")
	}
}

func (state *NNSampling) SampleImmediateNeighbourhood(graph *C.igraph_t, vertex int32, samples int, duplicates bool, clearBuffers bool, allEdges bool, useWeights bool, visitInitial bool, weightedRand bool) {
	var vs C.igraph_vs_t
	var vit C.igraph_vit_t

	// if allEdges IGRAPH_ALL else IGRAPH_OUT
	var mode C.igraph_neimode_t
	if allEdges {
		mode = C.IGRAPH_ALL
	} else {
		mode = C.IGRAPH_OUT
	}

	// set self as visited
	if visitInitial {
		state.visited[vertex] = true
	}

	// find non-visited neighbours using igraph
	C.igraph_vs_adj(&vs, C.int(vertex), mode)
	C.igraph_vit_create(graph, vs, &vit)
	for C.igraph_vit_end(&vit) == 0 {
		adjVertex := int32(C.igraph_vit_get(&vit))

		if duplicates || !state.visited[adjVertex] {
			state.neighbourhoodBuffer = append(state.neighbourhoodBuffer, adjVertex)
			if weightedRand && useWeights {
				eval, _ := getEdgeTypeAttribute(graph, vertex, adjVertex)
				state.edgeTypeBuffer = append(state.edgeTypeBuffer, eval)
			}
		}

		C.igraph_vit_next(&vit)
	}

	if samples >= len(state.neighbourhoodBuffer) {
		// no need to sample
		for index := 0; index < len(state.neighbourhoodBuffer); index++ {
			state.randomMask = append(state.randomMask, index)
		}
	} else {
		// create random index sequence and get those neighbours
		if weightedRand && useWeights {
			weightedRandomKSample(samples, state.edgeTypeBuffer, &state.randomMask)
		} else {
			randomKSample(len(state.neighbourhoodBuffer), samples, &state.randomMask)
		}
	}

	// update state with new edges
	for _, index := range state.randomMask {
		adjVertex := state.neighbourhoodBuffer[index]
		state.visited[adjVertex] = true
		state.sources = append(state.sources, vertex)
		state.targets = append(state.targets, adjVertex)

		if useWeights {
			eval, err := getEdgeTypeAttribute(graph, vertex, adjVertex)
			if err == nil {
				state.edgeTypes = append(state.edgeTypes, eval)
			}
		}
	}

	if clearBuffers {
		// clear keeping capacity
		state.neighbourhoodBuffer = state.neighbourhoodBuffer[:0]
		state.randomMask = state.randomMask[:0]
		state.edgeTypeBuffer = state.edgeTypeBuffer[:0]
	}

	// clean
	C.igraph_vs_destroy(&vs)
	C.igraph_vit_destroy(&vit)
}

func (state *NNSampling) SampleNeighbourhood(graph *C.igraph_t, vertex int32, levelSamples []int, level int, useNames bool) {
	if level >= len(levelSamples) {
		return
	}

	// clear keeping capacity
	state.neighbourhoodBuffer = state.neighbourhoodBuffer[:0]
	state.randomMask = state.randomMask[:0]
	state.edgeTypeBuffer = state.edgeTypeBuffer[:0]

	// sample immediate neighbours
	samples := levelSamples[level]
	state.SampleImmediateNeighbourhood(graph, vertex, samples, false, false, true, false, true, false)

	// sample neighbourhoods of neighbours if needed
	if level+1 < len(levelSamples) {
		sampledNeighbourhood := make([]int32, len(state.randomMask))
		for i, index := range state.randomMask {
			adjVertex := state.neighbourhoodBuffer[index]
			sampledNeighbourhood[i] = adjVertex
		}

		for _, adjVertex := range sampledNeighbourhood {
			state.SampleNeighbourhood(graph, adjVertex, levelSamples, level+1, useNames)
		}
	}
}
