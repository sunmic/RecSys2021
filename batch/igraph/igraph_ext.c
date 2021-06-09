#include "igraph_ext.h"

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