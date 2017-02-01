# matvec-sparse
Sparse matrix-vector multiplication using MPI library

### Usage:
> mpirun -np `N_PROCCESSES` `matvec_*` `INPUT` `[OUTPUT]`

You can check out the different versions below:

Version             | Description
--------------------|------------
`matvec_seq`        | Sequential (non-parallel) version
`matvec_mpi_bcast`  | Parallel version: vector `x` is broadcasted to every process
`matvec_mpi_p2p`    | Parallel version: vector `x` is split among the processes

Note that `INPUT` must be a file in *MatrixMarket* coordinate format (which is explained [here](http://math.nist.gov/MatrixMarket/formats.html#coord) in details).
