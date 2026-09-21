# Multi-stage build for the direct N-body solver.
#
# Stage 1 (builder): full toolchain + OpenMPI dev, compiles the MPI/hybrid
# binaries. Stage 2 (runtime): ubuntu:22.04 with only the MPI runtime, so the
# shipped image has no compilers.
#
# Design choices:
#  - ubuntu:22.04, not a vendor HPC image: a plain, widely available base keeps
#    the image portable and the build reproducible. The cluster's own MPI is
#    injected at runtime by Singularity, so we do not need a vendor MPI baked in.
#  - -march=x86-64-v3, not -march=native: native would tie the binary to the
#    build host's exact CPU. x86-64-v3 (AVX2, FMA, roughly Haswell and later)
#    runs on any modern x86_64 node, at the cost of not using AVX-512 that the
#    cluster may have. That gap is measured in the report.
#  - OpenMPI is installed for the BUILD (to link nbody_mpi). At runtime on the
#    cluster, Singularity binds the host MPI over it (see nbody.def).

# ----------------------------------------------------------------------------
FROM ubuntu:22.04 AS builder

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
        gcc \
        make \
        libopenmpi-dev \
        openmpi-bin \
        libc6-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY nbody_common.h nbody_core.h nbody_rsqrt.h nbody_core.c \
     nbody_direct_serial.c nbody_mpi.c generate_ic.c Makefile ./
COPY utils/ ./utils/

# Portable target for the container binary. SPLIT_ACC helped on x86 (see notes).
RUN make nbody_mpi nbody_hybrid generate_ic \
        ARCHFLAGS="-march=x86-64-v3" \
        CPPFLAGS="-DNBODY_SPLIT_ACC"

# ----------------------------------------------------------------------------
FROM ubuntu:22.04 AS runtime

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
        openmpi-bin \
        libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# bind points for the host directories on the cluster
RUN mkdir -p /work /scratch

COPY --from=builder /build/nbody_mpi /build/nbody_hybrid /build/generate_ic /usr/local/bin/

WORKDIR /work
CMD ["nbody_mpi", "--help"]
