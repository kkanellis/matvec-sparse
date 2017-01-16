#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"

#define DEBUG
#define MAX_RANDOM_NUM (1<<20)
#define MASTER 0
#define EPSILON 1e-9

#include "mmio-wrapper.h"
#include "util.h"

// TODO: replace broadcast with point-to-point for x vector

void mat_vec_mult(const double *values,
                    const int *i_idx,
                    const int *j_idx,
                    const double *x,
                    double *y,
                    const int NZ,
                    const int row_offset)
{
    for (int k = 0 ; k < NZ; k++) {
        y[ i_idx[k] - row_offset ] += values[k] * x[ j_idx[k] ];
    }
}

/* 
 * Partition non-zero matrix entries to (almost)
 * equal number of rows for each process
 */
void partition_equal_rows(int N, int NZ, int nprocs, int *row_idx, int *nz_count, 
                            int *nz_start_idx, int *row_count, int *row_start_idx)
{
    int last_row, i = 0;
    double row_chunk = ((double)N) / nprocs; /* rows for each process */

    for (int k = 0; k < nprocs; k++) {
        nz_count[k] = 0;
        nz_start_idx[k] = i;

        last_row = (int)((k + 1) * row_chunk - 1);

        row_start_idx[k] = (int)(k * row_chunk);
        row_count[k] = last_row - row_start_idx[k] + 1;

        while (i < NZ && row_idx[i] <= last_row) {
            nz_count[k]++; i++;
        }

        debug("[%d] row\t %d %d\n", k, row_count[k], row_start_idx[k]);
        debug("[%d] nz\t %d %d\n", k, nz_count[k], nz_start_idx[k]);
    }
    /* add remaining elements (if any) to last task */
    nz_count[nprocs - 1] += NZ - i;  
}

/* 
 * Partition non-zero matrix entries to almost equal
 * number of non-zero elements for each process
 */
void partition_equal_nz_elements() {
    /* Not implemented */
}


int main(int argc, char * argv[])
{
    char *filename;

    double *values; /* a_values array */
    int *i_idx,     /* i_index array */
        *j_idx;     /* j_index array */

    double *x, *y;  /* Ax = y */
    int N,          /* dim of matrix */
        NZ;         /* number of non-zero elements */

    int nprocs,     /* number of tasks/processes */
        rank,       /* id of task/process */
        nelements,  /* elements stored in this process */
        nrows,      /* rows stored in this process */
        first_row;  /* starting row for this process */

    /***** MPI MASTER (root) process only ******/
    int *nz_start_idx,  /* first matrix element for each process */
        *nz_count,      /* number of matrix elements for each process */
        *row_start_idx, /* first row processed for each process */ 
        *row_count;     /* number of rows to process for each process */

    int *buf_i_idx,     /* row index for all matrix elements */
        *buf_j_idx;     /* column index for all matrix elements */
    double *buf_values, /* value for all matrix elements */
           *buf_x,      /* value for all x vector elements */
           *res;        /* final result -> Ax */

    /*******************************************/
    
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* master thread reads matrix */
    if (rank == MASTER) {
        /* read arguments */
        if (argc != 2) {
            printf("Usage: %s filename\n", argv[0]);
            return 0;
        }
        else filename = argv[1];
        
        /* read matrix */
        if ( read_matrix(filename, &buf_i_idx, &buf_j_idx, 
                                    &buf_values, &N, &NZ) != 0) {
            fprintf(stderr, "read_matrix: failed\n");
            exit(EXIT_FAILURE);
        }

        debug("Read matrix from '%s'!\n", filename);
        debug("[%d]: Matrix properties: N = %d, NZ = %d\n",rank, N, NZ);

        /* allocate x, res vector */
        buf_x = (double *)malloc_or_exit( N * sizeof(double) );
        res = (double *)malloc_or_exit( N * sizeof(double) );

        /* generate random vector */
        //random_vec(buf_x, N, MAX_RANDOM_NUM);
        for (int i = 0; i < N; i++) {
            buf_x[i] = 1;
        }

        /* allocate memory */
        nz_count = (int *)malloc_or_exit( nprocs * sizeof(int) );
        nz_start_idx = (int *)malloc_or_exit( nprocs * sizeof(int) );
        row_count = (int *)malloc_or_exit( nprocs * sizeof(int) );
        row_start_idx = (int *)malloc_or_exit( nprocs * sizeof(int) );

        /* divide work across processes */
        partition_equal_rows(N, NZ, nprocs, buf_i_idx, nz_count, 
                                nz_start_idx, row_count, row_start_idx);
    }

    /* broadcast matrix properties N, NZ */
    MPI_Bcast(&N, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&NZ,1, MPI_INT, MASTER, MPI_COMM_WORLD);

    /* scatter number of rows each process is responsible for */
    MPI_Scatter(row_count, 1, MPI_INT, &nrows, 
                1, MPI_INT, MASTER, MPI_COMM_WORLD);
    /* scatter starting row for each process */
    MPI_Scatter(row_start_idx, 1, MPI_INT, &first_row, 
                1, MPI_INT, MASTER, MPI_COMM_WORLD);

    /* scatter number of elements each process will receive */
    MPI_Scatter(nz_count, 1, MPI_INT, &nelements, 
                1, MPI_INT, MASTER, MPI_COMM_WORLD);

    /* allocate y vector */
    y = (double *)calloc_or_exit( nrows, sizeof(double) );
    if (rank != MASTER)
        buf_x = (double *)malloc_or_exit( N * sizeof(double) );

    /* allocate values, i_idx & j_idx arrays */
    i_idx = (int *)malloc_or_exit( nelements * sizeof(int) );
    j_idx = (int *)malloc_or_exit( nelements * sizeof(int) );
    values = (double *)malloc_or_exit( nelements * sizeof(double));
    
    debug("[%d] %d %d %d\n", rank, N, NZ, nelements);
    
    /* broadcast x vector */
    /* TODO: change late to Point-2-Point communication */
    MPI_Bcast(buf_x, N, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* scatter matrix elements to processes */
    MPI_Scatterv(buf_values, nz_count, nz_start_idx, 
                MPI_DOUBLE, values, nelements, MPI_DOUBLE, 
                MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(buf_i_idx, nz_count, nz_start_idx, 
                MPI_INT, i_idx, nelements, MPI_INT, 
                MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(buf_j_idx, nz_count, nz_start_idx, 
                MPI_INT, j_idx, nelements, MPI_INT, 
                MASTER, MPI_COMM_WORLD);

    /* validation check */
    //for (int i = 0; i < nelements; i++) {
    //    debug("[%d] %d %d %lf\n", rank, i_idx[i], j_idx[i], values[i]);
    //}

    /* perform the multiplication */
    mat_vec_mult(values, i_idx, j_idx, buf_x, y, nelements, first_row);

    /* gather y elements from processes */
    MPI_Gatherv(y, nrows, MPI_DOUBLE, res, row_count, 
                row_start_idx, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* MPI: end */
    MPI_Finalize();

    #ifdef DEBUG
    if (rank == MASTER) {
        debug("-------------------------------------\n");
        debug("RESULT\n");
        debug("-------------------------------------\n");
        for (int i = 0; i < N; i++) {
            debug("[%d] %lf\n",rank, res[i]);
        }
    }
    #endif

    /* free the memory */
    free(values);
    free(i_idx);
    free(j_idx);
    free(buf_x);
    free(y);
    
    return 0;
}
