#include <igraph.h>

igraph_t* igraph_alloc();
int igraph_vit_end(igraph_vit_t* vit);
int igraph_vit_get(igraph_vit_t* vit);
void igraph_vit_next(igraph_vit_t* vit);