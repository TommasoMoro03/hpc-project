#!/bin/bash
#SBATCH --job-name=nbody_container
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH --output=%x.%j.out

# Native vs container overhead comparison (mandatory deliverable).
# Same problem and process counts, native binary vs the Singularity-wrapped
# binary, 5 repetitions each, at three process counts. Reports median +- sigma
# of the whole-run wall time and the percentage overhead.
#
# Prereq: build the image on the login node first:
#   module load singularity/4.3.1
#   singularity build nbody.sif nbody.def
# and put nbody.sif in this directory (or set SIF below).

module load openMPI/4.1.6
module load singularity/4.3.1

SIF=${SIF:-nbody.sif}
REPS=5
N=20000
STEPS=10

# native build with the same portable flags as the container, for a fair test
make nbody_mpi generate_ic ARCHFLAGS="-march=x86-64-v3" CPPFLAGS="-DNBODY_SPLIT_ACC"
./generate_ic --model 0 --n $N --seed 7 --output ic.bin

# median and sample stddev of the numbers on stdin
stats () {
    python3 -c '
import sys, statistics as st
xs = [float(x) for x in sys.stdin.read().split()]
m = st.median(xs)
s = st.pstdev(xs) if len(xs) < 2 else st.stdev(xs)
print(f"{m:.4f} {s:.4f}")'
}

# time one full srun run of a command, return elapsed seconds
timed_run () {
    local start end
    start=$(date +%s.%N)
    "$@" > /dev/null 2>&1
    end=$(date +%s.%N)
    echo "$end - $start" | bc -l
}

for P in 2 4 8; do
    echo "### processes=$P"

    nat=""
    con=""
    for r in $(seq 1 $REPS); do
        t=$(timed_run srun -n $P ./nbody_mpi --input ic.bin --nsteps $STEPS --dt 1e-4 --eps 0.05 --quiet)
        nat="$nat $t"
        t=$(timed_run srun -n $P singularity exec "$SIF" nbody_mpi --input ic.bin --nsteps $STEPS --dt 1e-4 --eps 0.05 --quiet)
        con="$con $t"
    done

    read nat_med nat_sd < <(echo "$nat" | tr ' ' '\n' | stats)
    read con_med con_sd < <(echo "$con" | tr ' ' '\n' | stats)
    overhead=$(python3 -c "print(f'{(($con_med-$nat_med)/$nat_med*100):.2f}')")

    echo "native    : ${nat_med} +- ${nat_sd} s"
    echo "container : ${con_med} +- ${con_sd} s"
    echo "overhead  : ${overhead} %"
done

# launch overhead: cost of starting the container itself (1 process)
echo "### launch overhead (singularity exec ... true, 10 reps)"
lo=""
for r in $(seq 1 10); do
    t=$(timed_run singularity exec "$SIF" true)
    lo="$lo $t"
done
echo "$lo" | tr ' ' '\n' | stats | awk '{print "singularity exec startup: "$1" +- "$2" s"}'
