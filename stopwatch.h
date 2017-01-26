#ifndef __STOPWATCH_H
#define __STOPWATCH_H
#include <time.h>

#define N_STOPWATCHES 5

void __sw_start(const int id);
void __sw_stop(const int id, double * value);

#endif
