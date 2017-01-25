#include <assert.h>
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


enum tag { REQUEST_TAG, REPLY_TAG };

double* mat_vec_mult_parallel(int rank, int nprocs, proc_info_t *all_proc_info,
                            int *i_idx, int *j_idx, double *values, double *x)
{
    proc_info_t *proc_info; /* info about submatrix; different per process */
    double *res;            /* result of multiplication res = A*x */

    /***** MPI MASTER (root) process only ******/
    int *count, *offset;    /* auxilliary arrays used for Scatterv/Gatherv */

    /* scatter to processors all info that will be needed */
    if (rank == MASTER)
        proc_info = all_proc_info;
    else
        proc_info = (proc_info_t *)malloc( nprocs * sizeof(proc_info_t) );

    MPI_Bcast(proc_info, nprocs, proc_info_type, MASTER, MPI_COMM_WORLD);

    /* allocate memory for vectors and submatrixes */
    double *y = (double *)calloc_or_exit( proc_info[rank].row_count, sizeof(double) );
    if (rank != MASTER) {
        x = (double *)malloc_or_exit( proc_info[rank].N * sizeof(double) );
        i_idx = (int *)malloc_or_exit( proc_info[rank].nz_count * sizeof(int) );
        j_idx = (int *)malloc_or_exit( proc_info[rank].nz_count * sizeof(int) );
        values = (double *)malloc_or_exit( proc_info[rank].nz_count * sizeof(double) );
    }
    else {
        res = (double *)malloc_or_exit( proc_info[rank].N * sizeof(double) );
    }

    //debug("[%d] %d %d %d\n", rank, proc_info[rank].N, proc_info[rank].NZ, proc_info[rank].nz_count);
    
    /* broadcast x vector */
    /* TODO: change late to Point-2-Point communication */
    MPI_Bcast(x, proc_info[rank].N, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* scatter matrix elements to processes */
    if (rank == MASTER) {
        count = (int *)malloc_or_exit( nprocs * sizeof(int) );
        offset = (int *)malloc_or_exit( nprocs * sizeof(int) );
        for (int p = 0; p < nprocs; p++) {
            count[p] = proc_info[p].nz_count;
            offset[p] = proc_info[p].nz_start_idx;
        }
    }
    MPI_Scatterv(i_idx, count, offset, MPI_INT, i_idx, 
                    proc_info[rank].nz_count, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(j_idx, count, offset, MPI_INT, j_idx, 
                    proc_info[rank].nz_count, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(values, count, offset, MPI_DOUBLE, values, 
                    proc_info[rank].nz_count, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* validation check */
    /*for (int i = 0; i < proc_info[rank].nz_count; i++) {
        debug("[%d] %d %d %lf\n", rank, i_idx[i], j_idx[i], values[i]);
    }*/

    /* send requests for x elements that are needed and belong to other processes */
    char *sent = (char *)calloc_or_exit( proc_info[rank].N, sizeof(char) );
    MPI_Request *send_reqs = (MPI_Request*)malloc_or_exit( proc_info[rank].NZ * sizeof(MPI_Request) );
    MPI_Request *recv_reqs = (MPI_Request*)malloc_or_exit( proc_info[rank].NZ * sizeof(MPI_Request) );

    int req_count = 0;

    for (int i = 0; i < proc_info[rank].nz_count; i++) {
        int col = j_idx[i];

        /* check whether I have the element */
        if ( in_range(col, proc_info[rank].row_start_idx, proc_info[rank].row_count) )
            continue;

        /* check if I already sent a request for the same element */
        if ( sent[col] )
            continue;

        sent[col] = 1;   /* mark the element */

        /* search which process has the element */
        int dest = -1;
        for (int p = 0; p < nprocs; p++) {
            if ( in_range(col, proc_info[p].row_start_idx, proc_info[p].row_count) ) {
                dest = p;
                break;
            }
        }
        assert( dest >= 0 );
        
        /* send the request */
        debug("[%d] Sending request to process %2d \t[%5d]\n", rank, dest, col);
        MPI_Isend(&col, 1, MPI_INT, dest, REQUEST_TAG , MPI_COMM_WORLD, &send_reqs[req_count]);
        
        /* recv the message (when it comes) */
        MPI_Irecv(&x[col], 1, MPI_DOUBLE, dest, REPLY_TAG, MPI_COMM_WORLD, &recv_reqs[req_count]);

        req_count++; 
    }

    /* wait for all send requests to finish */
    MPI_Waitall(req_count, send_reqs, MPI_STATUSES_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    /* serve requests */
    int unread_msg, col;
    MPI_Status status;

    while( 1 ) {
        /* check for available message */
        MPI_Iprobe(MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &unread_msg, MPI_STATUS_IGNORE);
        if ( !unread_msg )
            break;

        MPI_Recv(&col, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);
        debug("[%d] Replying request from process %2d \t[%5d]\n", rank, status.MPI_SOURCE, col);

        /* send x[col] element */
        MPI_Isend(&x[col], 1, MPI_DOUBLE, status.MPI_SOURCE, REPLY_TAG, MPI_COMM_WORLD, &send_reqs[0]);
    }


    /* perform mutliplication */
    for (int k = 0 ; k < proc_info[rank].nz_count; k++) {
        y[ i_idx[k] - proc_info[rank].row_start_idx ] += values[k] * x[ j_idx[k] ];
    }

    /* gather y elements from processes and save it to res */
    if (rank == MASTER) {
        res = (double *)malloc_or_exit( proc_info[rank].N * sizeof(double) );
        for (int p = 0; p < nprocs; p++) {
            count[p] = proc_info[p].row_count;
            offset[p] = proc_info[p].row_start_idx;
        }
    }
    MPI_Gatherv(y, proc_info[rank].row_count, MPI_DOUBLE, res, count, 
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

    double t, comp_time, partition_time;

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

        t = MPI_Wtime();
        
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
        
        partition_time = (MPI_Wtime() - t) * 1000.0;

        debug("[%d] Partition time: %10.3lf ms\n\n", rank, partition_time);
        debug("[%d] Starting algorithm...\n", rank);
        t = MPI_Wtime();
    }

    /* Matrix-vector multiplication for each processes */
    res = mat_vec_mult_parallel(rank, nprocs, proc_info, buf_i_idx, 
                                buf_j_idx, buf_values, buf_x);

    /* write to output file */
    if (rank == MASTER) {
        comp_time = (MPI_Wtime() - t) * 1000.0; 
        printf("[%d] Computation time: %10.3lf ms\n\n", rank, comp_time);

        printf("[%d] Total execution time: %10.3lf ms\n", rank, comp_time + partition_time);
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

    /* MPI: end */
    MPI_Finalize();

        
    return 0;
}

