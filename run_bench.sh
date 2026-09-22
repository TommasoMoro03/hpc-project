#!/bin/bash
#SBATCH --job-name=nbody_bench
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out

# Report-grade MPI strong-scaling benchmark.
# Fixed N=100000, sweep ranks, one discarded
# warm-up then REPS timed repetitions, reported as median +- sample stddev.
# Uses the shared bench.sh helpers so the statistics match the other scripts.
#
# Both the whole-run wall time and the solver's own force-kernel time are
# reported: wall includes MPI launch, kernel is the pure compute.

module load openMPI/4.1.6
source bench.sh

REPS=5
N=100000
STEPS=10
ARGS="--nsteps $STEPS --dt 1e-4 --eps 0.05 --quiet"

make nbody_mpi generate_ic ARCHFLAGS="-march=native" CPPFLAGS="-DNBODY_SPLIT_ACC"
./generate_ic --model 0 --n $N --seed 7 --output ic_bench.bin

kernel_time () { grep -oE "force_kernel_total=[0-9.]+" | head -1 | cut -d= -f2; }

echo "# N=$N steps=$STEPS reps=$REPS (median +- sample stddev)"
echo "# ranks  wall(s)              kernel(s)"

for P in 1 2 4 8 16 32 64; do
    if [ "$P" -gt "$SLURM_NTASKS" ]; then
        break
    fi

    # warm-up (discarded)
    bench_time srun -n $P --cpu-bind=cores ./nbody_mpi --input ic_bench.bin $ARGS > /dev/null

    wall=""; kern=""
    for r in $(seq 1 $REPS); do
        out=$(srun -n $P --cpu-bind=cores ./nbody_mpi --input ic_bench.bin $ARGS 2>/dev/null)
        kern="$kern $(echo "$out" | kernel_time)"
        wall="$wall $(bench_time srun -n $P --cpu-bind=cores ./nbody_mpi --input ic_bench.bin $ARGS)"
    done

    w=$(echo "$wall" | tr ' ' '\n' | bench_med_sd)
    k=$(echo "$kern" | tr ' ' '\n' | bench_med_sd)
    printf "%-7s %-20s %-20s\n" "$P" "$w" "$k"
done
