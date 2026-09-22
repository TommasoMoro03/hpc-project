#!/bin/bash
#SBATCH --job-name=nbody_weak
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out

# MPI weak scaling: fix the per-rank particle count, grow total N with the rank
# count P (N = PER_RANK * P). Pure MPI, one core per rank, single node.
#
# Note on N-body direct summation: at fixed N/P the per-rank work is
# O((N/P) * N) = O((N/P)^2 * P), so it GROWS linearly with P even though the
# particle count per rank is constant. Ideal weak scaling (flat time) does NOT
# apply here; the point is to see how time-per-step grows and where comm starts
# to matter. This is the behaviour the assignment asks us to discuss.

module load openMPI/4.1.6

PER_RANK=4000
STEPS=10

make nbody_mpi generate_ic ARCHFLAGS="-march=native"

for P in 1 2 4 8 16 32 64; do
    if [ "$P" -gt "$SLURM_NTASKS" ]; then
        break
    fi
    N=$(( PER_RANK * P ))
    ./generate_ic --model 0 --n $N --seed 7 --output ic_weak.bin
    echo "### ranks=$P N=$N (per_rank=$PER_RANK)"
    srun -n $P --cpu-bind=cores ./nbody_mpi --input ic_weak.bin \
        --nsteps $STEPS --dt 1e-4 --eps 0.05 --quiet
done
