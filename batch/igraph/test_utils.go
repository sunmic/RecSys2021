package main

// #cgo pkg-config: igraph
// #include <igraph.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include "igraph_ext.h"
import "C"
import (
	"fmt"
)

func TreeGraph(n int32, nchild int32) *C.igraph_t {
	var graph *C.igraph_t = C.igraph_alloc()
	var cint = C.int(n)
	fmt.Printf("%v", cint)
	C.igraph_tree(graph, C.int(n), C.int(nchild), C.IGRAPH_TREE_OUT)

	return graph
}

func DestroyGraph(graph *C.igraph_t) {
	C.igraph_destroy(graph)
}
