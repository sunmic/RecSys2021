#include "igraph_ext.h"
#include <stdio.h>
#include <stdint.h>

igraph_t* igraph_alloc() {
    igraph_t* pointer = malloc(sizeof(igraph_t));
    return pointer;
}

int igraph_vit_end(igraph_vit_t* vit) {
    return IGRAPH_VIT_END(*vit);
}

int igraph_vit_get(igraph_vit_t* vit) {
    return IGRAPH_VIT_GET(*vit);
}

void igraph_vit_next(igraph_vit_t* vit) {
    IGRAPH_VIT_NEXT(*vit);
}

igraph_real_t igraph_edge_weight_attribute(igraph_t* graph, igraph_integer_t eid) {
    return igraph_cattribute_EAN(graph, "weight", eid);
}

int igraph_vertex_integer_name_attribute(igraph_t* graph, igraph_integer_t vid) {
    const char* name = igraph_cattribute_VAS(graph, "name", vid);
    return atoi(name);
}