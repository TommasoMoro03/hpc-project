CC        ?= cc
MPICC     ?= mpicc
STD       ?= -std=c11
CPPFLAGS  ?=
OPTFLAGS  ?= -O3
ARCHFLAGS ?= -march=native
WARNFLAGS ?= -Wall -Wextra -Wpedantic
CFLAGS    ?= $(OPTFLAGS) $(ARCHFLAGS) $(WARNFLAGS)
OMPFLAGS  ?= -fopenmp
LDLIBS    ?= -lm
PRECISION ?= double

PROGRAMS := nbody_direct_serial generate_ic
HEADERS  := nbody_common.h nbody_core.h utils/timing.h
CORE     := nbody_core.c

ifeq ($(PRECISION),float)
PRECISION_CPPFLAGS := -DNBODY_USE_FLOAT
else ifeq ($(PRECISION),double)
PRECISION_CPPFLAGS := -DNBODY_USE_DOUBLE
else
$(error PRECISION must be either double or float)
endif

.PHONY: all clean run-smoke

all: $(PROGRAMS)

nbody_direct_serial: nbody_direct_serial.c $(CORE) $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ nbody_direct_serial.c $(CORE) $(LDLIBS)

nbody_omp: nbody_direct_serial.c $(CORE) $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ nbody_direct_serial.c $(CORE) $(LDLIBS)

nbody_mpi: nbody_mpi.c $(CORE) $(HEADERS)
	$(MPICC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ nbody_mpi.c $(CORE) $(LDLIBS)

nbody_hybrid: nbody_mpi.c $(CORE) $(HEADERS)
	$(MPICC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o $@ nbody_mpi.c $(CORE) $(LDLIBS)

generate_ic: generate_ic.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)


run-smoke: all
	./generate_ic --model 0 --n 128 --seed 42 --output plummer_128.bin
	./nbody_direct_serial --input plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --output plummer_128_final.bin --quiet
	./generate_ic --model 1 --n 128 --seed 43 --output ball_128.bin
	./nbody_direct_serial --input ball_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --output ball_128_final.bin --quiet

clean:
	rm -f $(PROGRAMS) nbody_omp nbody_mpi nbody_hybrid *.o *.bin
