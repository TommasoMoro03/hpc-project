#ifndef NBODY_RSQRT_H
#define NBODY_RSQRT_H

#include "nbody_common.h"

#include <stdint.h>
#include <string.h>

/*
 * Reciprocal square root, 1 / sqrt(x), used by the force and potential kernels.
 *
 * Default: the accurate libm path, 1 / dtype_sqrt(x).
 *
 * With -DNBODY_USE_RSQRT: a fast approximation seeded with the classic
 * bit-trick magic constant, then refined with Newton-Raphson iterations. This
 * trades accuracy for speed on the hottest operation in the inner loop. The
 * accuracy cost must be checked against the energy-drift diagnostic (a single
 * Newton step in double is already near machine precision for our range).
 */

#if defined (NBODY_USE_RSQRT)

#if defined (NBODY_USE_FLOAT)

/* float: one Newton step after the 32-bit magic-constant seed. */
static inline dtype dtype_rsqrt (dtype x)
{
  const float  half = 0.5f * x;
  uint32_t     bits;
  float        y = x;

  memcpy (&bits, &y, sizeof bits);
  bits = 0x5f3759dfu - (bits >> 1);
  memcpy (&y, &bits, sizeof y);

  y = y * (1.5f - half * y * y);   // Newton step 1
  y = y * (1.5f - half * y * y);   // Newton step 2
  return y;
}

#else

/* double: two Newton steps after the 64-bit magic-constant seed. */
static inline dtype dtype_rsqrt (dtype x)
{
  const double  half = 0.5 * x;
  uint64_t      bits;
  double        y = x;

  memcpy (&bits, &y, sizeof bits);
  bits = UINT64_C (0x5fe6eb50c7b537a9) - (bits >> 1);
  memcpy (&y, &bits, sizeof y);

  y = y * (1.5 - half * y * y);    // Newton step 1
  y = y * (1.5 - half * y * y);    // Newton step 2
  y = y * (1.5 - half * y * y);    // Newton step 3
  return y;
}

#endif

#else

/* accurate default */
static inline dtype dtype_rsqrt (dtype x)
{
  return (dtype) 1.0 / dtype_sqrt (x);
}

#endif

#endif
