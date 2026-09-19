#!/bin/bash
#SBATCH --job-name=nbody_omp
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:20:00
#SBATCH --output=%x.%j.out

# gcc is already on PATH on Orfeo, no module needed for this build.

# OMP_PLACES=cores puts one place per physical core; OMP_PROC_BIND=close
# packs threads onto neighbouring cores, filling one socket before the next.
export OMP_PROC_BIND=close
export OMP_PLACES=cores

make nbody_omp generate_ic

./generate_ic --model 0 --n 50000 --seed 7 --output ic_50000.bin

# Thread-count sweep that is capped at the cores slurm actually allocated
for t in 1 2 4 8 16 32 64 128; do
    if [ "$t" -gt "$SLURM_CPUS_PER_TASK" ]; then
        break
    fi
    export OMP_NUM_THREADS=$t
    echo "### OMP_NUM_THREADS=$t"
    ./nbody_omp --input ic_50000.bin --nsteps 10 --dt 1e-4 --eps 0.05 --quiet
done
