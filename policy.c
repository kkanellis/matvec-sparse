#include "policy.h"

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

