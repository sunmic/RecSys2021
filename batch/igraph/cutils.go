package main

import "C"

var WeightAttributeName = C.CString("edge_type")

func booltoint(in bool) C.int {
	if in {
		return C.int(1)
	}
	return C.int(0)
}
