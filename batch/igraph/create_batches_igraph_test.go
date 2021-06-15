package main

import (
	"math/rand"
	"testing"
)

func contains(arr []int32, val int32) bool {
	for _, x := range arr {
		if x == val {
			return true
		}
	}

	return false
}

func TestImmediateNeighbourhood(t *testing.T) {
	rand.Seed(1234)

	graph := TreeGraph(100, 5)

	var expected = map[int32][5]int32{
		0:  {1, 2, 3, 4, 5},
		4:  {21, 22, 23, 24, 25},
		9:  {46, 47, 48, 49, 50},
		16: {81, 82, 83, 84, 85},
	}

	sampling := NNSampling{
		visited: make(map[int32]bool),
	}
	samples := 4
	for key, elements := range expected {
		sampling.startVertex = key
		sampling.SampleImmediateNeighbourhood(graph, key, samples, false, false, false, false)
		//sampling.Print()

		if len(sampling.sources) != 4 {
			t.Fatalf("Number of sources does not match: expected %v, got %v", 4, len(sampling.sources))
		}
		if len(sampling.targets) != 4 {
			t.Fatalf("Number of targets does not match: expected %v, got %v", 4, len(sampling.targets))
		}
		for i, v := range sampling.sources {
			if v != key {
				t.Errorf("Sources for neighbourhood of %v at %v: expected %v, got %v", key, i, key, v)
			}
		}
		for i, v := range sampling.targets {
			if !contains(elements[:], v) {
				t.Errorf("Targets for neighbourhood of %v at %v: expected one of %v, got %v", key, i, elements, v)
			}
		}
		for i, v := range sampling.neighbourhoodBuffer {
			if !contains(elements[:], v) {
				t.Errorf("Neighbourhood buffer of %v at %v: expected %v got %v", key, i, elements, sampling.neighbourhoodBuffer)
			}
		}
		for v, visited := range sampling.visited {
			if visited && v != key && !contains(elements[:], v) {
				t.Errorf("%v is not neighbour of %v should not be visited", v, key)
			}
		}

		if len(sampling.visited) != 5 {
			t.Errorf("Visited map length is %v but should be %v (%v)", len(sampling.visited), samples+1, sampling.visited)
		}

		if !sampling.visited[key] {
			t.Errorf("Initial vertex not visited: %v", key)
		}

		sampling.Reset()

		if len(sampling.visited) != 0 {
			t.Errorf("Visited map should be empty after reset, but has %v elements (%v)", len(sampling.visited), sampling.visited)
		}
	}

	DestroyGraph(graph)
}

func TestNeighbourhood(t *testing.T) {
	rand.Seed(1234)

	graph := TreeGraph(100, 5)

	var expected = map[int32][5]int32{
		0: {1, 2, 3, 4, 5},
		1: {6, 7, 8, 9, 10},
		2: {11, 12, 13, 14, 15},
		3: {16, 17, 18, 19, 20},
		4: {21, 22, 23, 24, 25},
		5: {26, 27, 28, 29, 30},
	}

	levelSamples := []int{5, 3}
	sampling := NNSampling{
		startVertex: 0,
		visited:     make(map[int32]bool),
	}

	sampling.SampleNeighbourhood(graph, 0, levelSamples, 0)
	// 5 + 3 * 5
	if len(sampling.sources) != 20 {
		t.Fatalf("Number of sources does not match: expected %v, got %v (s%v, t%v) ", 20, len(sampling.sources), sampling.sources, sampling.targets)
	}
	if len(sampling.targets) != 20 {
		t.Fatalf("Number of targets does not match: expected %v, got %v (s%v, t%v)", 20, len(sampling.targets), sampling.sources, sampling.targets)
	}

	for i := 0; i < len(sampling.sources); i++ {
		source, target := sampling.sources[i], sampling.targets[i]
		expectedNeighbours, ok := expected[source]
		if !ok {
			t.Errorf("Vertex %v is not expected to be source", source)
		} else if !contains(expectedNeighbours[:], target) {
			t.Errorf("Invalid target for source %v: expected one of %v, got %v", source, expectedNeighbours, target)
		}
	}

	if len(sampling.visited) != 21 {
		t.Fatalf("Visited map has invalid size: expected %v, got %v (%v)", 20, len(sampling.visited), sampling.visited)
	}

	if !sampling.visited[0] {
		t.Errorf("Initial vertex not visited: %v", 0)
	}
}
