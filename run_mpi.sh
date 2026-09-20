#!/bin/bash
#SBATCH --job-name=nbody_mpi
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=%x.%j.out

# MPI launch check on a single node. Not distributed yet: every rank runs the
# full integration and rank 0 prints. The idea is to confirm MPI builds and
# launches under SLURM before the ring shift is added.

module load openMPI/4.1.6

make nbody_mpi generate_ic

./generate_ic --model 0 --n 3000 --seed 7 --output ic_3000.bin

echo "### ranks=$SLURM_NTASKS"
srun ./nbody_mpi --input ic_3000.bin --nsteps 10 --dt 1e-4 --eps 0.05 --quiet
