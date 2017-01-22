#ifndef UTIL_H
#define UTIL_H

#include <stdlib.h>

/* define useful debugging print macro */
#ifdef DEBUG
    #define debug(...) fprintf(stderr, __VA_ARGS__);
#else
    #define debug(...) do ; while(0)
#endif

typedef struct {
    int N,              /* dim of matrix */
        NZ,             /* number of non-zero elements */
        nz_count,       /* number of matrix elements for each process */
        nz_start_idx,   /* first matrix element for each process */
        row_count,      /* number of rows to process for each process */
        row_start_idx;  /* first row processed for each process */ 
} proc_info_t;

void random_vec(double *v, int N, int limit);

void * malloc_or_exit(size_t size);
void * calloc_or_exit(size_t nmemb, size_t size);

#endif
