#!/bin/bash
#SBATCH --job-name=nbody_optcompare
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=%x.%j.out

# Single-core comparison of the kernel optimization flags on Orfeo (x86).
# Same problem, one core, no OpenMP/MPI. Isolates the effect of each flag on
# the force kernel. The local ARM result was that rsqrt and split accumulators
# both lose; this checks whether x86 + gcc behaves differently.

module load openMPI/4.1.6

make generate_ic
./generate_ic --model 0 --n 20000 --seed 7 --output ic_20000.bin

for flags in "" "-DNBODY_USE_RSQRT" "-DNBODY_SPLIT_ACC" "-DNBODY_USE_RSQRT -DNBODY_SPLIT_ACC"; do
    gcc -std=c11 -DNBODY_USE_DOUBLE $flags -O3 -march=native -Wall -Wextra \
        nbody_direct_serial.c nbody_core.c -lm -o nb_opt
    echo "### flags: ${flags:-baseline}"
    ./nb_opt --input ic_20000.bin --nsteps 10 --dt 1e-4 --eps 0.05 --quiet | grep -E "final|timing"
done

rm -f nb_opt
