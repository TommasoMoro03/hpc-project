# Shared benchmarking helpers for the run_*.sh scripts.
# Source this file:  source bench.sh
#
# Provides consistent statistics (median, trimmed mean, sample stddev, min, max)
# and a wall-clock timing helper, so every scaling script reports numbers the
# same way. The assignment asks for the median (or trimmed mean) and standard
# deviation of at least five repetitions, with warm-up/outliers discussed.

# Read whitespace-separated numbers on stdin, print:
#   median  mean  trimmed_mean  stddev  min  max
# trimmed_mean drops the single lowest and highest value when there are >=4
# samples (a light outlier guard). stddev is the sample stddev.
bench_stats () {
    python3 -c '
import sys, statistics as st
xs = sorted(float(x) for x in sys.stdin.read().split())
n = len(xs)
if n == 0:
    print("nan nan nan nan nan nan"); sys.exit()
median = st.median(xs)
mean   = st.fmean(xs)
trimmed = st.fmean(xs[1:-1]) if n >= 4 else mean
sd     = st.stdev(xs) if n >= 2 else 0.0
print(f"{median:.5f} {mean:.5f} {trimmed:.5f} {sd:.5f} {xs[0]:.5f} {xs[-1]:.5f}")'
}

# Print median +- stddev only (compact), from numbers on stdin.
bench_med_sd () {
    bench_stats | awk '{print $1" +- "$4}'
}

# Time one command (whole invocation) in seconds, printed to stdout.
bench_time () {
    local start end
    start=$(date +%s.%N)
    "$@" > /dev/null 2>&1
    end=$(date +%s.%N)
    echo "$end - $start" | bc -l
}

# Relative overhead of b over a, as a percentage: (b-a)/a*100.
bench_overhead () {
    python3 -c "print(f'{(($2-$1)/$1*100):.2f}')" "$@"
}
