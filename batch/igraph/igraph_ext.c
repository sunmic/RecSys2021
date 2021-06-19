#include "igraph_ext.h"
#include <stdio.h>
#include <stdint.h>
#include <ctype.h>

/**
 * Alloc graph struct on heap
 */ 
igraph_t* igraph_alloc() {
    igraph_t* pointer = malloc(sizeof(igraph_t));
    return pointer;
}

/**
 * golang with cgo does not like macros :(
 */
int igraph_vit_end(igraph_vit_t* vit) {
    return IGRAPH_VIT_END(*vit);
}

int igraph_vit_get(igraph_vit_t* vit) {
    return IGRAPH_VIT_GET(*vit);
}

void igraph_vit_next(igraph_vit_t* vit) {
    IGRAPH_VIT_NEXT(*vit);
}

/**
 * Modifed version of igraph_read_edgelist with support for edge weights
 */ 
int igraph_read_graph_weighted_edgelist(igraph_t* graph, FILE *instream, igraph_integer_t n, igraph_bool_t directed, const char* attr_name) {
    igraph_vector_t edges = IGRAPH_VECTOR_NULL;
    igraph_vector_t weights = IGRAPH_VECTOR_NULL;
    long int from, to, weight;
    int c;

    IGRAPH_VECTOR_INIT_FINALLY(&edges, 0);
    IGRAPH_CHECK(igraph_vector_reserve(&edges, 100));

    IGRAPH_VECTOR_INIT_FINALLY(&weights, 0);
    IGRAPH_CHECK(igraph_vector_reserve(&weights, 100));


    /* skip all whitespace */
    do {
        c = getc (instream);
    } while (isspace (c));
    ungetc (c, instream);

    while (!feof(instream)) {
        int read;

        read = fscanf(instream, "%li", &from);
        if (read != 1) {
            IGRAPH_ERROR("parsing edgelist file failed", IGRAPH_PARSEERROR);
        }

        read = fscanf(instream, "%li", &to);
        if (read != 1) {
            IGRAPH_ERROR("parsing edgelist file failed", IGRAPH_PARSEERROR);
        }

        // read also weight
        read = fscanf(instream, "%li", &weight);
        if (read != 1) {
            IGRAPH_ERROR("parsing edgelist file failed", IGRAPH_PARSEERROR);
        }

        IGRAPH_CHECK(igraph_vector_push_back(&edges, from));
        IGRAPH_CHECK(igraph_vector_push_back(&edges, to));
        IGRAPH_CHECK(igraph_vector_push_back(&weights, weight));

        /* skip all whitespace */
        do {
            c = getc (instream);
        } while (isspace (c));
        ungetc (c, instream);
    }

    IGRAPH_CHECK(igraph_create(graph, &edges, n, directed));
    igraph_cattribute_EAN_setv(graph, attr_name, &weights);

    igraph_vector_destroy(&edges);
    igraph_vector_destroy(&weights);
    IGRAPH_FINALLY_CLEAN(1);
    return 0;
}