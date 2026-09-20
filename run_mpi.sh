#!/bin/bash
#SBATCH --job-name=nbody_mpi
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out

# MPI strong scaling on a single node. Fixed N, vary the number of ranks.
# Pure MPI here (one core per rank, no OpenMP yet), so this isolates the
# ring-shift communication and compute split.

module load openMPI/4.1.6

make nbody_mpi generate_ic

./generate_ic --model 0 --n 50000 --seed 7 --output ic_50000.bin

# Rank sweep, capped at the tasks SLURM allocated. Drift must stay ~constant
# across rank counts (correctness); force-kernel time should drop (scaling).
for p in 1 2 4 8 16 32 64; do
    if [ "$p" -gt "$SLURM_NTASKS" ]; then
        break
    fi
    echo "### ranks=$p"
    srun -n $p ./nbody_mpi --input ic_50000.bin --nsteps 10 --dt 1e-4 --eps 0.05 --quiet
done
