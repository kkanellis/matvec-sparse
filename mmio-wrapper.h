/* 
 * High-level wrapper for Matrix Market I/O library
 *
 * It is used to read sparse, real & square matrices from files
 */

#ifndef MM_IO_WRAPPER_H
#define MM_IO_WRAPPER_H

#include "mmio-wrapper.c"

int read_matrix (const char * filename, int **i_idx, int **j_idx, double **values, int *N, int *NZ);

#endif
