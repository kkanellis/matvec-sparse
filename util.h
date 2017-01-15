#ifndef UTIL_H
#define UTIL_H

/* define useful debugging print macro */
#ifdef DEBUG
    #define debug(fmt, ...) fprintf(stderr, fmt, __VA_ARGS__);
#else
    #define debug(...) do ; while(0)
#endif

void random_vec(double *v, int N, int limit);

#endif
