#!/bin/bash
#SBATCH --job-name=nbody_container
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=%x.%j.out

# Native vs container overhead comparison (mandatory deliverable).
# Same problem and process counts, native binary vs the Singularity-wrapped
# binary, at three process counts, with explicit identical CPU binding.
#
# Controls (the first run gave nonsensical negative overhead that grew with P,
# a sign the comparison was not controlled):
#  - --exclusive so no other job shares the node.
#  - identical srun --cpu-bind=cores for native and container.
#  - one discarded warm-up per configuration (turbo/cache warm-up; the
#    assignment asks warm-up runs to be reported).
#  - all native reps then all container reps, so each set sees a steady state.
#
# Prereq: nbody.sif present (built from the docker archive on the login node:
#   singularity build nbody.sif docker-archive://nbody.tar).

module load openMPI/4.1.6
module load singularity/4.3.1

SIF=${SIF:-nbody.sif}
REPS=5
N=20000
STEPS=10
ARGS="--input ic.bin --nsteps $STEPS --dt 1e-4 --eps 0.05 --quiet"

# native build with the same portable flags as the container
make nbody_mpi generate_ic ARCHFLAGS="-march=x86-64-v3" CPPFLAGS="-DNBODY_SPLIT_ACC"
./generate_ic --model 0 --n $N --seed 7 --output ic.bin

stats () {
    python3 -c '
import sys, statistics as st
xs = [float(x) for x in sys.stdin.read().split()]
m = st.median(xs)
s = st.pstdev(xs) if len(xs) < 2 else st.stdev(xs)
print(f"{m:.4f} {s:.4f}")'
}

timed_run () {
    local start end
    start=$(date +%s.%N)
    "$@" > /dev/null 2>&1
    end=$(date +%s.%N)
    echo "$end - $start" | bc -l
}

# show how ranks are bound, once, to confirm native and container match
echo "### binding check (P=4)"
echo "-- native --"
srun -n 4 --cpu-bind=cores --cpu-bind=verbose true 2>&1 | grep -i bind | head -4
echo "-- container --"
srun -n 4 --cpu-bind=cores --cpu-bind=verbose singularity exec "$SIF" true 2>&1 | grep -i bind | head -4

for P in 2 4 8; do
    echo "### processes=$P"

    # warm-up (discarded)
    timed_run srun -n $P --cpu-bind=cores ./nbody_mpi $ARGS > /dev/null
    timed_run srun -n $P --cpu-bind=cores singularity exec "$SIF" nbody_mpi $ARGS > /dev/null

    nat=""
    for r in $(seq 1 $REPS); do
        nat="$nat $(timed_run srun -n $P --cpu-bind=cores ./nbody_mpi $ARGS)"
    done

    con=""
    for r in $(seq 1 $REPS); do
        con="$con $(timed_run srun -n $P --cpu-bind=cores singularity exec "$SIF" nbody_mpi $ARGS)"
    done

    read nat_med nat_sd < <(echo "$nat" | tr ' ' '\n' | stats)
    read con_med con_sd < <(echo "$con" | tr ' ' '\n' | stats)
    overhead=$(python3 -c "print(f'{(($con_med-$nat_med)/$nat_med*100):.2f}')")

    echo "native    : ${nat_med} +- ${nat_sd} s"
    echo "container : ${con_med} +- ${con_sd} s"
    echo "overhead  : ${overhead} %"
done

echo "### launch overhead (singularity exec ... true, 10 reps)"
lo=""
for r in $(seq 1 10); do
    lo="$lo $(timed_run singularity exec "$SIF" true)"
done
echo "$lo" | tr ' ' '\n' | stats | awk '{print "singularity exec startup: "$1" +- "$2" s"}'
