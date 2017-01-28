#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"

#undef DEBUG

#define MAX_RANDOM_NUM (1<<20)
#define MASTER 0
#define EPSILON 1e-9

#include "mmio-wrapper.h"
#include "util.h"
#include "policy.h"

/* Partition policy selection {EQUAL_NZ, EQUAL_ROWS} */
enum policies policy = EQUAL_NZ;
MPI_Datatype proc_info_type;

enum tag { REQUEST_TAG, REPLY_TAG };

double* mat_vec_mult_parallel(int rank, int nprocs, proc_info_t *all_proc_info,
                            int *buf_i_idx, int *buf_j_idx, double *buf_values, double *buf_x)
{
    proc_info_t *proc_info; /* info about submatrix; different per process */
    double *res;            /* result of multiplication res = A*x */

    /***** MPI MASTER (root) process only ******/
    int *nz_count, *nz_offset,    /* auxilliary arrays used for Scatterv/Gatherv */
        *row_count, *row_offset;

    /* scatter to processors all info that will be needed */
    if (rank == MASTER)
        proc_info = all_proc_info;
    else
        proc_info = (proc_info_t *)malloc( nprocs * sizeof(proc_info_t) );
    MPI_Bcast(proc_info, nprocs, proc_info_type, MASTER, MPI_COMM_WORLD);

    /* allocate memory for vectors and submatrixes */
    double *y = (double *)calloc_or_exit( proc_info[rank].row_count, sizeof(double) );
    double *x = (double *)malloc_or_exit( proc_info[rank].N * sizeof(double) );
    int *i_idx = (int *)malloc_or_exit( proc_info[rank].nz_count * sizeof(int) );
    int *j_idx = (int *)malloc_or_exit( proc_info[rank].nz_count * sizeof(int) );
    double *values = (double *)malloc_or_exit( proc_info[rank].nz_count * sizeof(double) );
    if (rank == MASTER) {
        res = (double *)malloc_or_exit( proc_info[rank].N * sizeof(double) );
    }
    
    /* process auxilliary arrays for scatterv/gatherv ops */
    if (rank == MASTER) { 
        row_count = (int *)malloc_or_exit( nprocs * sizeof(int) );
        row_offset = (int *)malloc_or_exit( nprocs * sizeof(int) );
        nz_count = (int *)malloc_or_exit( nprocs * sizeof(int) );
        nz_offset = (int *)malloc_or_exit( nprocs * sizeof(int) );

        for (int p = 0; p < nprocs; p++) {
            row_count[p] = proc_info[p].row_count;
            row_offset[p] = proc_info[p].row_start_idx;
            nz_count[p] = proc_info[p].nz_count;
            nz_offset[p] = proc_info[p].nz_start_idx;
        }
    }
        
    /* allocate buffers for requests sending */
    int **send_buf = (int **)malloc_or_exit( nprocs * sizeof(int*) );
    for (int i =0; i < nprocs; i++) {
        if (i != rank && proc_info[i].row_count > 0)
            send_buf[i] = (int *)malloc_or_exit( proc_info[i].row_count * sizeof(int) );
    }

    /* scatter x vector to processes */
    MPI_Scatterv(buf_x, row_count, row_offset, MPI_DOUBLE, &x[ proc_info[rank].row_start_idx ], 
                proc_info[rank].row_count, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* scatter j_idx elements to processes */
    MPI_Scatterv(buf_j_idx, nz_count, nz_offset, MPI_INT, j_idx, 
                    proc_info[rank].nz_count, MPI_INT, MASTER, MPI_COMM_WORLD);

    int *to_send = (int *)calloc_or_exit( nprocs, sizeof(int) );    /* # of req to each proc */
    char* map = (char *)malloc_or_exit( proc_info[rank].N * sizeof(char) );

    /* build sending blocks to processors */
    int dest, col;
    for (int i = 0; i < proc_info[rank].nz_count; i++) {
        col = j_idx[i];

        /* check whether I need to send a request */
        if ( in_range(col, proc_info[rank].row_start_idx, proc_info[rank].row_count) ||
                map[col] > 0) 
            continue;

        /* search which process has the element
         * NOTE: Due to small number or processes, serial search is faster
         */
        dest = -1;
        for (int p = 0; p < nprocs; p++) {
            if ( in_range(col, proc_info[p].row_start_idx, proc_info[p].row_count) ) {
                dest = p;
                break;
            }
        }
        assert( dest >= 0 );

        /* insert new request */
        send_buf[dest][ to_send[dest]++ ] = col;
        map[col] = 1;
    }

    /* MPI request storage */
    MPI_Request *send_reqs = (MPI_Request*)malloc_or_exit( nprocs * sizeof(MPI_Request) );
    MPI_Request *recv_reqs = (MPI_Request*)malloc_or_exit( nprocs * sizeof(MPI_Request) );

    /* receiving blocks storage */
    double **recv_buf = (double **)malloc_or_exit( nprocs * sizeof(double*) );
    for (int p =0; p < nprocs; p++) {
        if (to_send[p] > 0)
            recv_buf[p] = (double *)malloc_or_exit( to_send[p] * sizeof(double) );
    }

    /* sending requests to processes in blocks */
    int req_made = 0;
    int *expect = (int *)calloc_or_exit( nprocs, sizeof(int) );
    for (int p = 0; p < nprocs; p++) {
        /* need to send to this proc? */
        if (p == rank || to_send[p] == 0) {
            send_reqs[p] = recv_reqs[p] = MPI_REQUEST_NULL;
            continue;
        }
        debug("[%d] Sending requests to process %2d \t[%5d]\n", rank, p, to_send[p]);

        /* logistics */
        expect[p] = 1;
        req_made++;

        /* send the request */
        MPI_Isend(send_buf[p], to_send[p], MPI_INT, p, REQUEST_TAG, 
                    MPI_COMM_WORLD, &send_reqs[p]);
        /* recv the block (when it comes) */
        MPI_Irecv(recv_buf[p], to_send[p], MPI_DOUBLE, p, REPLY_TAG, 
                    MPI_COMM_WORLD, &recv_reqs[p]);
    }
    debug("[%d] Sent all requests! [%4d]\n", rank, req_made);

    /* notify the processes about the number of requests they should expect */
    if (rank == MASTER)
        MPI_Reduce(MPI_IN_PLACE, expect, nprocs, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
    else
        MPI_Reduce(expect, expect, nprocs, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
    MPI_Scatter(expect, 1, MPI_INT, &expect[rank], 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    /**** reply to requests ****/
    int *reqs = (int *) malloc_or_exit( proc_info[rank].row_count * sizeof(int) ); 
    double **rep_buf = (double **)malloc_or_exit( nprocs * sizeof(double*) ); /* reply blocks storage */

    MPI_Status status;
    int req_count;
    for (int p = 0; p < expect[rank]; p++) {
        /* Wait until a request comes */
        MPI_Probe(MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &req_count);
        rep_buf[p] = (double *)malloc_or_exit( req_count * sizeof(double) );

        /* fill rep_buf[p] with requested x elements */
        MPI_Recv(reqs, req_count, MPI_INT, status.MPI_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < req_count; i++) {
            rep_buf[p][i] = x[ reqs[i] ];
        }
        
        /* send the requested block */
        MPI_Isend(rep_buf[p], req_count, MPI_DOUBLE, status.MPI_SOURCE, REPLY_TAG, MPI_COMM_WORLD, &send_reqs[0]);
        debug("[%d] Replying requests from process %2d \t[%5d]\n", rank, status.MPI_SOURCE, req_count);
    }
    debug("[%d] Replied to all requests! [%4d]\n", rank, to_send[rank]);
    
    /* scatter j_idx & values to processes */
    MPI_Scatterv(buf_i_idx, nz_count, nz_offset, MPI_INT, i_idx, 
                    proc_info[rank].nz_count, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(buf_values, nz_count, nz_offset, MPI_DOUBLE, values, 
                    proc_info[rank].nz_count, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* Local elements multiplication */
    for (int k = 0 ; k < proc_info[rank].nz_count; k++) {
        if ( in_range( j_idx[k], proc_info[rank].row_start_idx, proc_info[rank].row_count) ) {
            y[ i_idx[k] - proc_info[rank].row_start_idx ] += values[k] * x[ j_idx[k] ];
        }
    }

    /* wait for all blocks to arrive */
    int p;
    debug("[%d] Waiting for %d requests\n", rank, req_made);
    for (int q = 0; q < req_made; q++) {
        MPI_Waitany(nprocs, recv_reqs, &p, MPI_STATUS_IGNORE);
        assert(p != MPI_UNDEFINED);

        /* fill x array with new elements */
        for (int i = 0; i < to_send[p]; i++) 
            x[ send_buf[p][i] ] = recv_buf[p][i];
    }

    /* Global elements multiplication */ 
    for (int k = 0 ; k < proc_info[rank].nz_count; k++) {
        if ( !in_range( j_idx[k], proc_info[rank].row_start_idx, proc_info[rank].row_count) ) {
            y[ i_idx[k] - proc_info[rank].row_start_idx ] += values[k] * x[ j_idx[k] ];
        }
    }

    /* gather y elements from processes and save it to res */
    debug("[%d] Gathering results...\n", rank);
    MPI_Gatherv(y, proc_info[rank].row_count, MPI_DOUBLE, res, row_count, 
                row_offset, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* return final result */
    return res;
}

/*
 * Creates two MPI derived datatypes for the respective structs
 */
void create_mpi_datatypes(MPI_Datatype *proc_info_type) {
    MPI_Datatype oldtypes[2];
    MPI_Aint offsets[2];
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

