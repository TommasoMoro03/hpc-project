#!/bin/bash
#SBATCH --job-name=nbody_omp
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=nbody_omp_%j.out

# gcc is already on PATH on Orfeo, no module needed for this build.

make nbody_omp generate_ic

./generate_ic --model 0 --n 20000 --seed 7 --output ic_20000.bin

for t in 1 2 4 8; do
    export OMP_NUM_THREADS=$t
    echo "### OMP_NUM_THREADS=$t"
    ./nbody_omp --input ic_20000.bin --nsteps 10 --dt 1e-4 --eps 0.05 --quiet
done
