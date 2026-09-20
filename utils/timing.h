#ifndef TIMING_H
#define TIMING_H

/*
 * Ask for the POSIX clocks. Under -std=c11 glibc hides clock_gettime and the
 * CLOCK_* ids unless a feature-test macro is set before <time.h>. Same
 * approach as the course examples (_XOPEN_SOURCE 700 implies POSIX.1-2008).
 */
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include <time.h>

/*
 * for the timing things I can directly take inspiration from the course example!
 */

#define CPU_TIME_W __extension__ ({ struct timespec ts; clock_gettime (CLOCK_REALTIME, &ts), \
      (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9; })

#define CPU_TIME_P __extension__ ({ struct timespec ts; clock_gettime (CLOCK_PROCESS_CPUTIME_ID, &ts), \
      (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9; })

#define CPU_TIME_T __extension__ ({ struct timespec ts; clock_gettime (CLOCK_THREAD_CPUTIME_ID, &ts), \
      (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9; })

#endif
