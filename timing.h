#ifndef TIMING_H
#define TIMING_H

#include <time.h>

/*
 * for the timing things I can directly take inspiration from the course example!
 */

#define CPU_TIME_W ({ struct timespec ts; clock_gettime (CLOCK_REALTIME, &ts), \
      (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9; })

#define CPU_TIME_P ({ struct timespec ts; clock_gettime (CLOCK_PROCESS_CPUTIME_ID, &ts), \
      (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9; })

#define CPU_TIME_T ({ struct timespec ts; clock_gettime (CLOCK_THREAD_CPUTIME_ID, &ts), \
      (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9; })

#endif
