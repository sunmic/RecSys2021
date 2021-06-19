#include <igraph/igraph.h>
#include <stdio.h>

igraph_t* igraph_alloc();
int igraph_vit_end(igraph_vit_t* vit);
int igraph_vit_get(igraph_vit_t* vit);
void igraph_vit_next(igraph_vit_t* vit);
int igraph_read_graph_weighted_edgelist(igraph_t* graph, FILE *instream, igraph_integer_t n, igraph_bool_t directed, const char* attr_name);