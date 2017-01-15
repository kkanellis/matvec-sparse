#include <stdlib.h>
#include <time.h>

#include "util.h"

/*
 * Generate a random real vector of size N with
 * elements beloning to [0, limit)
 */
void random_vec (double *v, int N, int limit)
{
    /* fill v with random doubles */
    limit--;
    srand( 410 );
    for (int i = 0; i < N; i++) {
        v[i] = ((double)rand()) / (((double)RAND_MAX) / limit) + ((double)rand()) / ((double)RAND_MAX);
    }
}

