# C compiler 
CC = gcc
CFLAGS = -O2 -std=gnu99 -DDEBUG
LDFLAGS = -lm -lrt

# MPI compiler wrapper
MPI_C = mpicc
MPI_CFLAGS = -O2 -std=gnu99 -DDEBUG

# Object files from libraries
OBJ = mmio.o mmio-wrapper.o policy.o util.o

all: matvec_seq matvec_mpi_rows

matvec_seq: matvec_seq.c $(OBJ) stopwatch.o
	$(CC) $(CFLAGS) $(OBJ) stopwatch.o $< -o $@ $(LDFLAGS)

matvec_mpi_rows: matvec_mpi_rows.c $(OBJ)
	$(MPI_C) $(MPI_CFLAGS) $(OBJ) $< -o $@ $(LDFLAGS)

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $<

.PHONY: clean
clean:
	rm -f matvec_seq matvec_mpi_rows *.o
