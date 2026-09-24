#!/bin/bash
#SBATCH --job-name=nbody_bench
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out

# Report-grade MPI strong-scaling benchmark.
# Fixed N, sweep ranks, one discarded warm-up then REPS timed repetitions,
# reported as median +- sample stddev. Uses the shared bench.sh helpers.
#
# Each rep is a SINGLE solver run: we time the whole srun (wall, includes MPI
# launch) and parse the solver's own force-kernel time (compute) from the same
# run. N=50000, 5 steps keep the O(N^2) 1-rank baseline within the wall limit
# (the earlier N=100000/10-step version timed out at 2h).

module load openMPI/4.1.6
source bench.sh

REPS=5
N=50000
STEPS=5
ARGS="--nsteps $STEPS --dt 1e-4 --eps 0.05 --quiet"

make nbody_mpi generate_ic ARCHFLAGS="-march=native" CPPFLAGS="-DNBODY_SPLIT_ACC"
./generate_ic --model 0 --n $N --seed 7 --output ic_bench.bin

kernel_time () { grep -oE "force_kernel_total=[0-9.]+" | head -1 | cut -d= -f2; }

echo "# N=$N steps=$STEPS reps=$REPS (median +- sample stddev)"
echo "# ranks  wall(s)              kernel(s)"

OUT=$(mktemp)
for P in 1 2 4 8 16 32 64; do
    if [ "$P" -gt "$SLURM_NTASKS" ]; then
        break
    fi

    # warm-up (discarded)
    srun -n $P --cpu-bind=cores ./nbody_mpi --input ic_bench.bin $ARGS > /dev/null 2>&1

    wall=""; kern=""
    for r in $(seq 1 $REPS); do
        start=$(date +%s.%N)
        srun -n $P --cpu-bind=cores ./nbody_mpi --input ic_bench.bin $ARGS > "$OUT" 2>/dev/null
        end=$(date +%s.%N)
        wall="$wall $(echo "$end - $start" | bc -l)"
        kern="$kern $(kernel_time < "$OUT")"
    done

    w=$(echo "$wall" | tr ' ' '\n' | bench_med_sd)
    k=$(echo "$kern" | tr ' ' '\n' | bench_med_sd)
    printf "%-7s %-20s %-20s\n" "$P" "$w" "$k"
done
rm -f "$OUT"
