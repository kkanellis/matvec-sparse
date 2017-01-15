#include <stdio.h>
#include <stdlib.h>

#define DEBUG
#define MAX_RANDOM_NUM (1<<20)

#include "mmio-wrapper.h"
#include "util.h"

void mat_vec_mult(const double *values,
                    const int *i_idx,
                    const int *j_idx,
                    const double *x,
                    double *y,
                    const int NZ)
{
    for (int k = 0 ; k < NZ; k++) {
        y[ i_idx[k] ] += values[k] * x[ j_idx[k] ];
    }
}


int main(int argc, char * argv[])
{
    char *filename;

    double *values; /* a_values array */
    int *i_idx,     /* i_index array */
        *j_idx;     /* j_index array */

    double *x, *y;  /* Ax = y */
    int N, NZ;

    /* read arguments */
    if (argc != 2) {
        printf("Usage: %s filename\n", argv[0]);
        return 0;
    }
    else filename = argv[1];
    
    /* read matrix */
    if ( read_matrix(filename, &i_idx, &j_idx, &values, &N, &NZ) != 0) {
        exit(EXIT_FAILURE);
    }

    debug("Matrix properties: N = %d, NZ = %d\n", N, NZ);

    /* allocate x, y vector */
    y = (double *)calloc( N, sizeof(double) );
    x = (double *)malloc( N * sizeof(double) );
    if (y == NULL || x == NULL) {
        fprintf(stderr, "malloc: failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }

    /* generate random vector */
    random_vec(x, N, MAX_RANDOM_NUM);
    /*
    for (int i = 0; i < N; i++) {
        x[i] = 1;
    }
    */
    
    /* perform the multiplication */
    mat_vec_mult(values, i_idx, j_idx, x, y, NZ);

    #ifdef DEBUG
    for (int i = 0; i < N; i++) {
        debug("%lf\n", y[i]);
    }
    #endif

    /* free the memory */
    free(values);
    free(i_idx);
    free(j_idx);
    free(x);
    free(y);
    
    return 0;
}
