#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"

#define MAX_RANDOM_NUM (1<<20)
#define MASTER 0
#define EPSILON 1e-9

#include "mmio-wrapper.h"
#include "util.h"
#include "policy.h"

// TODO: replace broadcast with point-to-point for x vector

enum policies policy = EQUAL_NZ;
MPI_Datatype proc_info_type;


double* mat_vec_mult_parallel(int rank, int nprocs, proc_info_t *all_proc_info,
                            int *i_idx, int *j_idx, double *values, double *x)
{
    proc_info_t proc_info;  /* info about submatrix; different per process */
    double *res;            /* result of multiplication res = A*x */

    /***** MPI MASTER (root) process only ******/
    int *count, *offset;    /* auxilliary arrays used for Scatterv/Gatherv */

    /* scatter to processors all info that will be needed */
    MPI_Scatter(all_proc_info, 1, proc_info_type, &proc_info,
                    1, proc_info_type, MASTER, MPI_COMM_WORLD);

    /* allocate memory for vectors and submatrixes */
    double *y = (double *)calloc_or_exit( proc_info.row_count, sizeof(double) );
    if (rank != MASTER) {
        x = (double *)malloc_or_exit( proc_info.N * sizeof(double) );
        i_idx = (int *)malloc_or_exit( proc_info.nz_count * sizeof(int) );
        j_idx = (int *)malloc_or_exit( proc_info.nz_count * sizeof(int) );
        values = (double *)malloc_or_exit( proc_info.nz_count * sizeof(double) );
    }
    else {
        res = (double *)malloc_or_exit( proc_info.N * sizeof(double) );
    }

    //debug("[%d] %d %d %d\n", rank, proc_info.N, proc_info.NZ, proc_info.nz_count);
    
    /* broadcast x vector */
    /* TODO: change late to Point-2-Point communication */
    MPI_Bcast(x, proc_info.N, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* scatter matrix elements to processes */
    if (rank == MASTER) {
        count = (int *)malloc_or_exit( nprocs * sizeof(int) );
        offset = (int *)malloc_or_exit( nprocs * sizeof(int) );
        for (int p = 0; p < nprocs; p++) {
            count[p] = all_proc_info[p].nz_count;
            offset[p] = all_proc_info[p].nz_start_idx;
        }
    }
    MPI_Scatterv(i_idx, count, offset, MPI_INT, i_idx, 
                    proc_info.nz_count, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(j_idx, count, offset, MPI_INT, j_idx, 
                    proc_info.nz_count, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(values, count, offset, MPI_DOUBLE, values, 
                    proc_info.nz_count, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* validation check */
    /*for (int i = 0; i < proc_info.nz_count; i++) {
        debug("[%d] %d %d %lf\n", rank, i_idx[i], j_idx[i], values[i]);
    }*/

    /* multiplication kernel */
    for (int k = 0 ; k < proc_info.nz_count; k++) {
        y[ i_idx[k] - proc_info.row_start_idx ] += values[k] * x[ j_idx[k] ];
    }

    /* gather y elements from processes and save it to res */
    if (rank == MASTER) {
        res = (double *)malloc_or_exit( proc_info.N * sizeof(double) );
        for (int p = 0; p < nprocs; p++) {
            count[p] = all_proc_info[p].row_count;
            offset[p] = all_proc_info[p].row_start_idx;
        }
    }
    MPI_Gatherv(y, proc_info.row_count, MPI_DOUBLE, res, count, 
                offset, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* return final result */
    return res;
}

/*
 * Creates two MPI derived datatypes for the respective structs
 */
void create_mpi_datatypes(MPI_Datatype *proc_info_type) {
    MPI_Datatype oldtypes[2];
    MPI_Aint offsets[2], extent;
    int blockcounts[2];

    /* create `proc_info_t` datatype */
    offsets[0] = 0;
    oldtypes[0] = MPI_INT;
    blockcounts[0] = 6;

    MPI_Type_create_struct(1, blockcounts, offsets, oldtypes, proc_info_type);
    MPI_Type_commit(proc_info_type);
}

int main(int argc, char * argv[])
{
    char *in_file,
         *out_file = NULL;

    double *x; /* vector to be multiplied */

    int nprocs,     /* number of tasks/processes */
        rank;       /* id of task/process */
    
    /***** MPI MASTER (root) process only ******/
    proc_info_t *proc_info;

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

    create_mpi_datatypes(&proc_info_type);

    /* master thread reads matrix */
    if (rank == MASTER) {
        /* read arguments */
        if (argc < 2 || argc > 3) {
            printf("Usage: %s input_file [output_file]\n", argv[0]);
            return 0;
        }
        else {
            in_file = argv[1];
            if (argc == 3) 
                out_file = argv[2];
        }

        /* initialize proc_info array */
        proc_info = (proc_info_t *)malloc_or_exit( nprocs * sizeof(proc_info_t) );
        
        /* read matrix */
        if ( read_matrix(in_file, &buf_i_idx, &buf_j_idx, &buf_values,
                            &proc_info[MASTER].N, &proc_info[MASTER].NZ) != 0) {
            fprintf(stderr, "read_matrix: failed\n");
            exit(EXIT_FAILURE);
        }

        debug("[%d] Read matrix from '%s'!\n", rank, in_file);
        debug("[%d] Matrix properties: N = %d, NZ = %d\n\n",rank, 
                                    proc_info[MASTER].N, proc_info[MASTER].NZ);

        /* initialize process info */
        for (int p = 0; p < nprocs; p++) {
            if (p != MASTER) {
                proc_info[p] = proc_info[MASTER];
            }
        }

        /* allocate x, res vector */
        buf_x = (double *)malloc_or_exit( proc_info[MASTER].N * sizeof(double) );
        res = (double *)malloc_or_exit( proc_info[MASTER].N * sizeof(double) );

        /* generate random vector */
        //random_vec(buf_x, N, MAX_RANDOM_NUM);
        for (int i = 0; i < proc_info[MASTER].N; i++) {
            buf_x[i] = 1;
        }

        /* divide work across processes */
        if (policy == EQUAL_ROWS) {
            debug("[%d] Policy: Equal number of ROWS\n", rank);
            partition_equal_rows(proc_info, nprocs, buf_i_idx);
        }
        else if (policy == EQUAL_NZ) {
            debug("[%d] Policy: Equal number of NZ ENTRIES\n", rank);
            partition_equal_nz_elements(proc_info, nprocs, buf_i_idx);
        }
        else {
            fprintf(stderr, "Wrong policy defined...");
            exit(EXIT_FAILURE);
        }

        debug("\n[%d] Starting algorithm...\n", rank);
    }

    /* Matrix-vector multiplication for each processes */
    res = mat_vec_mult_parallel(rank, nprocs, proc_info, buf_i_idx, 
                                buf_j_idx, buf_values, buf_x);
    
    /* MPI: end */
    MPI_Finalize();

    /* write to output file */
    if (rank == MASTER) {
        debug("Finished!\n");
        if (out_file != NULL) {
            printf("Writing result to '%s'\n", out_file);

            /* open file */
            FILE *f;
            if ( !(f = fopen(out_file, "w")) ) {
                fprintf(stderr, "fopen: failed to open file '%s'", out_file);
                exit(EXIT_FAILURE);
            }

            /* write result */
            for (int i = 0; i < proc_info[MASTER].N; i++) {
                fprintf(f, "%.8lf\n", res[i]);
            }
            
            /* close file */
            if ( fclose(f) != 0) {
                fprintf(stderr, "fopen: failed to open file '%s'", out_file);
                exit(EXIT_FAILURE);
            }

            printf("Done!\n");
        }

        /* free the memory */
        free(buf_values);
        free(buf_i_idx);
        free(buf_j_idx);
        free(buf_x);
        free(res);
    }
        
    return 0;
}

