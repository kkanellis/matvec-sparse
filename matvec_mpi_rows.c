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

typedef struct {
    int N,              /* dim of matrix */
        NZ,             /* number of non-zero elements */
        nz_count,       /* number of matrix elements for each process */
        nz_start_idx,   /* first matrix element for each process */
        row_count,      /* number of rows to process for each process */
        row_start_idx;  /* first row processed for each process */ 
} proc_info_t;
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
 * Partition non-zero matrix entries to (almost)
 * equal number of rows for each process
 *
 * Time complexity: O( NZ )
 */
void partition_equal_rows(proc_info_t *proc_info, int nprocs, const int *row_idx)
{
    /* rows for each process */
    double row_chunk = ((double)proc_info[0].N) / nprocs; 

    int last_row, i = 0;
    for (int k = 0; k < nprocs; k++) {
        proc_info[k].nz_count = 0;
        proc_info[k].nz_start_idx = i;

        last_row = (int)((k + 1) * row_chunk - 1);

        proc_info[k].row_start_idx = (int)(k * row_chunk);
        proc_info[k].row_count = last_row - proc_info[k].row_start_idx + 1;

        while (i < proc_info[0].NZ && row_idx[i] <= last_row) {
            proc_info[k].nz_count++; 
            i++;
        }

        debug("[%d] Processor %d takes rows %3d till %3d [%3d] (entries %6d till %6d) [%6d]\n", 
                        MASTER, k, 
                        proc_info[k].row_start_idx, 
                        proc_info[k].row_start_idx + proc_info[k].row_count - 1,
                        proc_info[k].row_count,
                        proc_info[k].nz_start_idx, 
                        proc_info[k].nz_start_idx + proc_info[k].nz_count - 1,
                        proc_info[k].nz_count
        );
    }
    /* add remaining elements (if any) to last task */
    proc_info[nprocs - 1].nz_count += proc_info[0].NZ - i;  
}

/* 
 * Partition non-zero matrix entries to almost equal
 * number of non-zero elements for each process
 *
 * Time complexity: O( N log NZ )
 */
void partition_equal_nz_elements(proc_info_t *proc_info, int nprocs, const int *row_idx) 
{
    /* calculate the number of non-zero elements for each row */
    int *row_elements = (int *)calloc_or_exit( proc_info[0].N, sizeof(int) );
    for (int i = 0; i < proc_info[0].NZ; i++) {
        row_elements[ row_idx[i] ]++;
    }

    /* Binary search for minimuzing the sum of workload (nz-elements)
     * per process. Each process receives consequtive rows */
    int l = 0, h = proc_info[0].NZ;
    while (l < h) {
        int nz = (l + h) / 2;
        
        /* calculate how many process are needed so that each
         * process receives up to nz elements */
        int sum = 0, procs_need = 0;
        for (int i = 0; i < proc_info[0].N; i++) {
            if (sum + row_elements[i] > nz ) {
                sum = row_elements[i];
                procs_need++;
            }
            else sum += row_elements[i];
        }

        /* update searching range */
        if (procs_need <= (nprocs - 1))
            h = nz;
        else
            l = nz + 1;
    }
    
    /* initialize process 0 info */
    proc_info[0] = (proc_info_t) {proc_info[0].N, proc_info[0].NZ, 0, 0, 0, 0};

    /* split rows to process - maximum workload is `l` non-zero elements */
    int sum = 0, total_sum = 0, k = 0;
    for (int i = 0; i < proc_info[0].N; i++) {
        if (sum + row_elements[i] > l) {
            sum = row_elements[i];
            k++;

            /* update process info */
            proc_info[k].row_count = 1;
            proc_info[k].row_start_idx = i;
            proc_info[k].nz_count = row_elements[i];
            proc_info[k].nz_start_idx = total_sum;
        }
        else {
            sum += row_elements[i];
            
            /* update process info */
            proc_info[k].row_count++;
            proc_info[k].nz_count += row_elements[i];
        }
        total_sum += row_elements[i];
    }
    
    for (int k = 0; k < nprocs; k++) {
        debug("[%d] Processor %d takes rows %3d till %3d [%3d] (entries %6d till %6d) [%6d]\n", 
                        MASTER, k, 
                        proc_info[k].row_start_idx, 
                        proc_info[k].row_start_idx + proc_info[k].row_count - 1,
                        proc_info[k].row_count,
                        proc_info[k].nz_start_idx, 
                        proc_info[k].nz_start_idx + proc_info[k].nz_count - 1,
                        proc_info[k].nz_count
        );
    }
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

        debug("Read matrix from '%s'!\n", in_file);
        debug("[%d]: Matrix properties: N = %d, NZ = %d\n",rank, 
                                    proc_info[MASTER].N, proc_info[MASTER].NZ);

        /* initialize process info */
        for (int p = 0; p < nprocs; p++) {
            if (p != MASTER) {
                proc_info[p] = proc_info[MASTER];

                debug("[%d] %d\n", p, proc_info[p].N);
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
        //partition_equal_rows(proc_info, nprocs, buf_i_idx);
        partition_equal_nz_elements(proc_info, nprocs, buf_i_idx);

        debug("Starting algorithm...\n");
    }

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
