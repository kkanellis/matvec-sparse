# C compiler 
CC = gcc
CFLAGS = -O2 -g
LDFLAGS = -lm

# MPI compiler wrapper
MPI_C = mpicc
MPI_CFLAGS = #--showme:compiler

# Object files from libraries
OBJ = mmio.o mmio-wrapper.o policy.o util.o

all: matvec_seq matvec_mpi_rows

matvec_seq: matvec_seq.c $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $< -o $@ $(LDFLAGS)

matvec_mpi_rows: matvec_mpi_rows.c $(OBJ)
	$(MPI_C) $(MPI_CFLAGS) $(OBJ) $< -o $@ $(LDFLAGS)

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $<

.PHONY: clean
clean:
	rm -f matvec_seq matvec_mpi_rows *.o
