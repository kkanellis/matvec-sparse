#include "stopwatch.h"

struct timespec sw[N_STOPWATCHES + 1];

void __sw_start(const int id) {
    clock_gettime(CLOCK_MONOTONIC_RAW, sw + id);
}

void __sw_stop(const int id, double * value) {
    clock_gettime(CLOCK_MONOTONIC_RAW, sw + N_STOPWATCHES);
    *value += (
        (double) ((sw[N_STOPWATCHES].tv_nsec - sw[id].tv_nsec) / 1000000.0 )+
        (double) ((sw[N_STOPWATCHES].tv_sec - sw[id].tv_sec) * 1000.0)
    );
}

