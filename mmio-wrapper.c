/* 
 * High-level wrapper for Matrix Market I/O library
 *
 * It is used to read sparse, real & square matrices from files
 */

#include <stdio.h>
#include <stdlib.h>

#include "mmio-wrapper.h"
#include "mmio.h"

/* Reads a matrix from a Matrix Market file, stored in COO format */
int read_matrix (const char * filename, int **i_idx, int **j_idx, double **values, int *N, int *NZ)
{
    FILE *f;
    MM_typecode matcode;
    int errorcode, nrows, ncols, nz_elements;
     
    /* open the file */
    if ( (f = fopen(filename, "r")) == NULL ) {
        fprintf(stderr, "Cannot open '%s'\n", filename);
        return 1;
    }
    
    /* process first line */
    if ( (errorcode = mm_read_banner(f, &matcode)) != 0 ) {
        fprintf(stderr, "Error while processing banner (file:'%s') (code=%d)\n",
                filename, errorcode);
        return 1;
    }

    /* matrix should be sparse and real */
    if ( !mm_is_matrix(matcode) || 
         !mm_is_real(matcode)   ||
         !mm_is_sparse(matcode) ) {
        fprintf(stderr, "Not supported matrix type: %s\n", mm_typecode_to_str(matcode));
        return 1;
    }

    /* read info */
    if ( (errorcode = mm_read_mtx_crd_size(f, &nrows, &ncols, &nz_elements)) != 0) {
        fprintf(stderr, "Error while processing array (file:'%s') (code:%d)\n",
                filename, errorcode);
        return 1;
    }

    /* matrix should be square */
    if (nrows != ncols) {
        fprintf(stderr, "Matrix is NOT square (rows=%d, cols=%d)\n", nrows, ncols);
        return 1;
    }

    *N = nrows;
    *NZ = nz_elements;

    /* reserve memory for vector */
    *i_idx = (int *)malloc( nz_elements * sizeof(int) );
    *j_idx = (int *)malloc( nz_elements * sizeof(int) );
    *values = (double *)malloc( nz_elements * sizeof(double));
    
    /* read actual matrix */
    for (int i = 0; i < *NZ; i++) {
        fscanf(f, "%d %d %lf", &(*i_idx)[i], &(*j_idx)[i], &(*values)[i]);
        (*i_idx)[i]--; (*j_idx)[i]--;
    }

    /* close the file */
    if ( fclose(f) != 0 ) {
        fprintf(stderr, "Cannot close file (fil:'%s')\n", filename);
    }

    return 0;
}

