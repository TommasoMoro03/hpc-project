#!/bin/bash
#SBATCH --job-name=nbody_hybrid
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out

# Hybrid MPI+OpenMP layout comparison at a fixed total of 64 cores.
# The 64 cores are split between MPI ranks and OpenMP threads:
# from one rank per core (pure MPI) to few ranks with many threads each.

module load openMPI/4.1.6

make nbody_hybrid generate_ic

./generate_ic --model 0 --n 50000 --seed 7 --output ic_50000.bin

export OMP_PROC_BIND=close
export OMP_PLACES=cores

# each row is "ranks threads" with ranks*threads = 64
for split in "64 1" "32 2" "16 4" "8 8" "4 16" "2 32" "1 64"; do
    set -- $split
    ranks=$1
    threads=$2
    export OMP_NUM_THREADS=$threads
    echo "### ranks=$ranks threads=$threads (total=$((ranks*threads)))"
    srun --ntasks=$ranks --cpus-per-task=$threads \
         ./nbody_hybrid --input ic_50000.bin --nsteps 10 --dt 1e-4 --eps 0.05 --quiet
done
