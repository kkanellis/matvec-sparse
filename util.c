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
        v[i] = ((double)rand()) / (((double)RAND_MAX) / limit) + \
               ((double)rand()) / ((double)RAND_MAX);
    }
}

/*
 * Tries to malloc. Terminates on failure.
 */
void * malloc_or_exit(size_t size) {
    void * ptr = malloc( size );
    if ( !ptr ) {
        fprintf(stderr, "malloc: failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

/*
 * Tries to calloc. Terminates on failure.
 */
void * calloc_or_exit(size_t nmemb, size_t size) {
    void * ptr = calloc(nmemb, size);
    if ( !ptr ) {
        fprintf(stderr, "calloc: failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

/*
 * Check if integer num is in range [start, start + count)
 */
char in_range(int num, int start, int count) {
    return (start <= num && num < start + count);
}

