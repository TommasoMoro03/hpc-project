#!/bin/bash
#SBATCH --job-name=nbody_sysinfo
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=%x.%j.out

# Collect the hardware and software identification
# The compiler/MPI modules must match what the run scripts use.
module load openMPI/4.1.6
module load singularity/4.3.1 2>/dev/null

line () { printf '\n===== %s =====\n' "$1"; }

line "hostname / date"
hostname
date

line "CPU (lscpu)"
lscpu

line "NUMA layout (numactl -H)"
numactl -H 2>/dev/null || echo "numactl not available"

line "memory (/proc/meminfo, first lines)"
head -5 /proc/meminfo

line "gcc version"
gcc --version | head -1

line "OpenMPI (mpirun) version"
mpirun --version 2>&1 | head -2

line "OpenMP runtime (from gcc)"
echo | gcc -fopenmp -dM -E - 2>/dev/null | grep -i _OPENMP || echo "n/a"

line "singularity version"
singularity --version 2>/dev/null || echo "singularity not loaded"

line "compiler flags used by the project"
echo "native builds : -std=c11 -O3 -march=native -Wall -Wextra -Wpedantic (+ -fopenmp for OMP/hybrid)"
echo "container build: -std=c11 -O3 -march=x86-64-v3 ... -DNBODY_SPLIT_ACC"
echo "optional flags : -DNBODY_USE_RSQRT, -DNBODY_SPLIT_ACC, -DNBODY_NEWTON3, PRECISION=float"

line "run configuration (pinning) used by the run scripts"
echo "OMP_PROC_BIND=close  OMP_PLACES=cores  srun --cpu-bind=cores"
