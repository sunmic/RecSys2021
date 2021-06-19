#include <igraph/igraph.h>
#include <igraph/igraph_interrupt.h>
#include <stdio.h>
#include <stdbool.h>
#include <ctype.h>
#include "igraph_ext.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s filepath\n", argv[0]);
        return 1;
    }

    const char* filepath = argv[1];
    const char* wname = "weight";

    igraph_t graph;
    igraph_set_attribute_table(&igraph_cattribute_table);

    FILE* file = fopen(filepath, "r");
    int errCode = igraph_read_graph_weighted_edgelist(&graph, file, 0, true, wname);
    if (errCode == IGRAPH_PARSEERROR) {
		printf("Error while reading weighted graph\n");
        return 1;
	}

    fclose(file);

    file = fopen(filepath, "r");
    long int from, to, weight;
    int eid, c;
    igraph_real_t actual;

    /* skip all whitespace */
    do {
        c = getc (file);
    } while (isspace (c));
    ungetc (c, file);

     while (!feof(file)) {
        int read;

        read = fscanf(file, "%li", &from);
        if (read != 1) {
            printf("Parsing edgelist file failed\n");
            return 2;
        }

        read = fscanf(file, "%li", &to);
        if (read != 1) {
            printf("Parsing edgelist file failed\n");
            return 2;
        }

        // read also weight
        read = fscanf(file, "%li", &weight);
        if (read != 1) {
            printf("Parsing edgelist file failed\n");
            return 2;
        }

        igraph_get_eid(&graph, &eid, from, to, true, false);
        if (eid == -1) {
            printf("Error: cannot find edge between %d and %d\n", from, to);
        } else {
            actual = igraph_cattribute_EAN(&graph, wname, eid);
            if (actual != (igraph_real_t) weight) {
                printf("Weights do not match: expected %d, actual %f\n", weight, actual);
            }
        }

        /* skip all whitespace */
        do {
            c = getc (file);
        } while (isspace (c));
        ungetc (c, file);
     }
     fclose(file);

     return 0;
}