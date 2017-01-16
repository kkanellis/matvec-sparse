#ifndef UTIL_H
#define UTIL_H

/* define useful debugging print macro */
#ifdef DEBUG
    #define debug(...) fprintf(stderr, __VA_ARGS__);
#else
    #define debug(...) do ; while(0)
#endif

void random_vec(double *v, int N, int limit);

void * malloc_or_exit(size_t size);
void * calloc_or_exit(size_t nmemb, size_t size);

#endif
