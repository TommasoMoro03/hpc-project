#!/bin/bash
#SBATCH --job-name=nbody_sched
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:15:00
#SBATCH --output=%x.%j.out

# Compare OpenMP loop schedules on the force kernel. The pragma uses
# schedule(runtime), so the policy is picked here via OMP_SCHEDULE without
# recompiling. Thread count is fixed so only the schedule changes.
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

make nbody_omp generate_ic

./generate_ic --model 0 --n 50000 --seed 7 --output ic_50000.bin

for sched in static dynamic guided; do
    export OMP_SCHEDULE=$sched
    echo "### OMP_SCHEDULE=$sched OMP_NUM_THREADS=$OMP_NUM_THREADS"
    ./nbody_omp --input ic_50000.bin --nsteps 10 --dt 1e-4 --eps 0.05 --quiet
done
