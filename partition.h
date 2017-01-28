#ifndef POLICY_H
#define POLICY_H

#include "util.h"

enum policies { EQUAL_ROWS, EQUAL_NZ };

void partition_equal_rows(proc_info_t *proc_info, int nprocs, const int *row_idx);
void partition_equal_nz_elements(proc_info_t *proc_info, int nprocs, const int *row_idx);

#endif
