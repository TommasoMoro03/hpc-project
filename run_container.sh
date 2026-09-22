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

# diagnostic: what does each binary report internally? If the compute times
# differ, the two binaries are not doing the same work (e.g. different MPI/OMP).
echo "### internal-timing diagnostic (P=4)"
echo "-- native says --"
srun -n 4 --cpu-bind=cores ./nbody_mpi $ARGS 2>/dev/null | grep -E "final|timing"
echo "-- container says --"
srun -n 4 --cpu-bind=cores singularity exec "$SIF" nbody_mpi $ARGS 2>/dev/null | grep -E "final|timing"
echo "-- which mpirun/mpi does each see --"
echo "native ldd:"; ldd ./nbody_mpi 2>/dev/null | grep -i mpi | head -3
echo "container ldd:"; srun -n 1 singularity exec "$SIF" ldd /usr/local/bin/nbody_mpi 2>/dev/null | grep -i mpi | head -3

# pull the solver's own force_kernel_total (excludes srun launch/teardown) from
# a run's stdout
kernel_time () { grep -oE "force_kernel_total=[0-9.]+" | head -1 | cut -d= -f2; }

for P in 2 4 8; do
    echo "### processes=$P"

    # warm-up (discarded)
    timed_run srun -n $P --cpu-bind=cores ./nbody_mpi $ARGS > /dev/null
    timed_run srun -n $P --cpu-bind=cores singularity exec "$SIF" nbody_mpi $ARGS > /dev/null

    # wall = whole srun (includes MPI launch overhead); kern = solver's own
    # force-kernel time (the honest compute-overhead metric)
    nat_wall=""; nat_kern=""
    for r in $(seq 1 $REPS); do
        out=$(srun -n $P --cpu-bind=cores ./nbody_mpi $ARGS 2>/dev/null)
        nat_kern="$nat_kern $(echo "$out" | kernel_time)"
        nat_wall="$nat_wall $(timed_run srun -n $P --cpu-bind=cores ./nbody_mpi $ARGS)"
    done

    con_wall=""; con_kern=""
    for r in $(seq 1 $REPS); do
        out=$(srun -n $P --cpu-bind=cores singularity exec "$SIF" nbody_mpi $ARGS 2>/dev/null)
        con_kern="$con_kern $(echo "$out" | kernel_time)"
        con_wall="$con_wall $(timed_run srun -n $P --cpu-bind=cores singularity exec "$SIF" nbody_mpi $ARGS)"
    done

    read nw_med nw_sd < <(echo "$nat_wall" | tr ' ' '\n' | stats)
    read cw_med cw_sd < <(echo "$con_wall" | tr ' ' '\n' | stats)
    read nk_med nk_sd < <(echo "$nat_kern" | tr ' ' '\n' | stats)
    read ck_med ck_sd < <(echo "$con_kern" | tr ' ' '\n' | stats)
    ov_wall=$(python3 -c "print(f'{(($cw_med-$nw_med)/$nw_med*100):.2f}')")
    ov_kern=$(python3 -c "print(f'{(($ck_med-$nk_med)/$nk_med*100):.2f}')")

    echo "wall   native ${nw_med} +- ${nw_sd} s | container ${cw_med} +- ${cw_sd} s | overhead ${ov_wall} %"
    echo "kernel native ${nk_med} +- ${nk_sd} s | container ${ck_med} +- ${ck_sd} s | overhead ${ov_kern} %"
done

echo "### launch overhead (singularity exec ... true, 10 reps)"
lo=""
for r in $(seq 1 10); do
    lo="$lo $(timed_run singularity exec "$SIF" true)"
done
echo "$lo" | tr ' ' '\n' | stats | awk '{print "singularity exec startup: "$1" +- "$2" s"}'
