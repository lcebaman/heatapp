CC ?= mpicc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra
LDFLAGS ?=
LDLIBS ?= -lm

.PHONY: all clean

all: mpi_probe

mpi_probe: src/mpi_probe.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	$(RM) mpi_probe
