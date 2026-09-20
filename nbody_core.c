#include "nbody_core.h"
#include "utils/timing.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void die (const char *format, ...)
{
  va_list  args;

  va_start (args, format);
  vfprintf (stderr, format, args);
  va_end (args);
  fputc ('\n', stderr);
  exit (EXIT_FAILURE);
}

/*
 * Parse a size_t command-line value.  All user-facing quantities that count
 * particles or steps pass through this function so that overflow and malformed
 * input fail early, before any allocation or simulation state is modified.
 */
size_t parse_size (const char *text,     // decimal text to parse
                   const char *name      // option name used in errors
		   )
{
  char               *endptr;
  unsigned long long  value;

  errno = 0;
  value = strtoull (text, &endptr, 10);
  if ((errno != 0) || (endptr == text) || (*endptr != '\0'))
    die ("invalid integer for %s: %s", name, text);
  if (value > (unsigned long long) SIZE_MAX)
    die ("integer for %s is too large: %s", name, text);

  return (size_t) value;
}

/*
 * Parse a finite floating-point command-line value and cast it to dtype.  The
 * parser reads through double because strtof and strtod differ only in final
 * rounding for the ranges used here; the explicit range check keeps float-mode
 * builds from silently accepting values that cannot be represented by dtype.
 */
dtype parse_dtype (const char *text,     // decimal text to parse
                   const char *name      // option name used in errors
		   )
{
  char    *endptr;
  double   value;

  errno = 0;
  value = strtod (text, &endptr);
  if ((errno != 0) || (endptr == text) || (*endptr != '\0') || !isfinite (value))
    die ("invalid floating-point value for %s: %s", name, text);
  if (fabs (value) > (double) DTYPE_MAX_VALUE)
    die ("floating-point value for %s is outside the selected dtype range: %s", name, text);

  return (dtype) value;
}

/*
 * Return the value associated with either "--key value" or "--key=value".
 * The caller passes the loop index by address so that the separated-value
 * form consumes the following argv entry exactly once.
 */
const char *option_value (int        *i,       // current argv index, updated on success
                          int         argc,    // argc from main
                          char      **argv,    // argv from main
                          const char *key      // long option name, including "--"
			  )
{
  const size_t  key_len = strlen (key);
  const char   *arg     = argv[*i];

  if ((strncmp (arg, key, key_len) == 0) && (arg[key_len] == '='))
    return arg + key_len + 1;

  if (strcmp (arg, key) == 0)
    {
      if (*i + 1 >= argc)
        die ("missing value after %s", key);
      *i += 1;
      return argv[*i];
    }

  return NULL;
}

/*
 * Allocate a cache-line aligned block.  Alignment is not required for scalar
 * correctness, but it makes the serial skeleton a better starting point for
 * vectorisation and OpenMP first-touch experiments.
 */
void *checked_aligned_alloc (size_t  nbytes,      // requested useful bytes
                                    size_t  alignment    // power-of-two alignment
				    )
{
  void   *ptr;
  size_t  padded;

  if (nbytes == 0u)
    die ("attempted zero-byte allocation");
  if (alignment == 0u)
    die ("invalid zero alignment");
  if (nbytes > SIZE_MAX - alignment)
    die ("allocation size overflow");

  padded = ((nbytes + alignment - 1u) / alignment) * alignment;
  ptr = aligned_alloc (alignment, padded);
  if (ptr == NULL)
    die ("aligned_alloc failed for %zu bytes", padded);

  return ptr;
}

/*
 * Read exactly nmemb items from a binary stream.  Centralising the check avoids
 * partial binary records being mistaken for valid particles, which is otherwise
 * easy to do when replacing a line-oriented ASCII reader with fread.
 */
static void checked_fread (void       *ptr,       // destination buffer
                           size_t      size,      // item size in bytes
                           size_t      nmemb,     // number of items expected
                           FILE       *fp,        // open input stream
                           const char *path,      // file name for diagnostics
                           const char *what       // logical record name
			   )
{
  const size_t  got = fread (ptr, size, nmemb, fp);

  if (got != nmemb)
    {
      if (ferror (fp))
        die ("read error while reading %s from '%s'", what, path);
      die ("short file while reading %s from '%s'", what, path);
    }
}

/*
 * Write exactly nmemb items to a binary stream.  All output paths go through
 * this helper so that disk-full and permission errors are reported at the point
 * where the data loss happens, not later in a benchmark script.
 */
static void checked_fwrite (const void *ptr,       // source buffer
                            size_t      size,      // item size in bytes
                            size_t      nmemb,     // number of items to write
                            FILE       *fp,        // open output stream
                            const char *path,      // file name for diagnostics
                            const char *what       // logical record name
			    )
{
  const size_t  written = fwrite (ptr, size, nmemb, fp);

  if (written != nmemb)
    die ("write error while writing %s to '%s'", what, path);
}


/*
 * Initialise an empty particle container.  This function does not allocate; it
 * simply gives every pointer a known value so that particles_free can safely be
 * called after a partial failure path.
 */
void particles_init_empty (particles_t *p    // particle container to initialise
				  )
{
  p->n    = 0u;
  p->mass = (dtype) 1.0;
  p->x    = NULL;
  p->y    = NULL;
  p->z    = NULL;
  p->vx   = NULL;
  p->vy   = NULL;
  p->vz   = NULL;
  p->ax   = NULL;
  p->ay   = NULL;
  p->az   = NULL;
}


/*
 * Allocate the SoA storage used by the solver.  Positions, velocities, and
 * accelerations are separate arrays, not an array of structs, because the
 * direct kernel only needs streams of x/y/z coordinates and accumulators.  This
 * layout is also the natural one for later SIMD and MPI ring-buffer work.
 */
void particles_allocate (particles_t  *p,       // output container
                                size_t        n,       // number of particles
                                dtype         mass     // mass of each particle
				)
{
  const size_t  bytes = n * sizeof (dtype);

  if (n == 0u)
    die ("the number of particles must be positive");
  if (n > SIZE_MAX / sizeof (dtype))
    die ("particle count is too large");
  if (!(mass > (dtype) 0.0) || !dtype_isfinite (mass))
    die ("particle mass must be positive and finite");

  particles_init_empty (p);
  p->n    = n;
  p->mass = mass;
  p->x    = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->y    = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->z    = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vx   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vy   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vz   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->ax   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->ay   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->az   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
}

/*
 * Release all particle arrays and return the container to the empty state.  No
 * simulation data survive this call.
 */
void particles_free (particles_t *p    // container to release
			    )
{
  free (p->x);
  free (p->y);
  free (p->z);
  free (p->vx);
  free (p->vy);
  free (p->vz);
  free (p->ax);
  free (p->ay);
  free (p->az);
  particles_init_empty (p);
}

/*
 * Cast one physical quantity to the on-disk type.
 * Nofinite values and overflows should still fail loudly.
 * How do we deal with that?
 * Check the initial condition generator for a more performant
 * implementation.
 *
 * NOTE: may this be a bottleneck in the I/O ? what your profiling says?
 *       if so, you may consider to make this optional only when a "debugging mode"
 *       is set, and to expand to a simple cast otherwose,
 *       Optionally this sanity checks should be made as a loop over particles that
 *       accumulates failures counter, instead on per-particle funciton call
 */
static float dtype_to_storage_float (dtype       value,       // value to store
                                     const char *component,   // component name for diagnostics
                                     size_t      i            // particle index for diagnostics
				     )
{
  const double  as_double = (double) value;

  if (!isfinite (as_double) || (fabs (as_double) > (double) FLT_MAX))
    die ("particle %zu component %s cannot be stored as a finite float", i, component);

  return (float) value;
}

/*
 * Load particle coordinates and velocities from the binary file.
 * The on-disk values are single precision, then converted to dtype so the same
 * initial-condition file can be used for both float and double solver builds.
 * Acceleration arrays are left uninitialised because every force evaluation
 * overwrites them.
 */
void particles_read_binary (const char  *path,       // input file path
                                   dtype        mass,       // mass assigned to each particle
                                   particles_t *p           // output particle container
				   )
{
  FILE           *fp;
  unsigned char   magic[NBODY_BINARY_MAGIC_SIZE];
  uint64_t        n64;
  size_t          n;
  size_t          i;

  fp = fopen (path, "rb");
  if (fp == NULL)
    die ("cannot open input file '%s'", path);

  checked_fread (magic, sizeof magic[0], NBODY_BINARY_MAGIC_SIZE,
                 fp, path, "binary magic");
  if (memcmp (magic, nbody_binary_magic, NBODY_BINARY_MAGIC_SIZE) != 0)
    die ("input file '%s' is not an %s file", path, NBODY_BINARY_VERSION_TEXT);

  checked_fread (&n64, sizeof n64, 1u, fp, path, "particle count");
  if ((n64 == 0u) || (n64 > (uint64_t) SIZE_MAX))
    die ("invalid particle count in '%s'", path);
  n = (size_t) n64;

  particles_allocate (p, n, mass);

  for (i = 0u; i < n; ++i)
    {
      float  record[NBODY_BINARY_COMPONENTS];

      checked_fread (record, sizeof record[0], NBODY_BINARY_COMPONENTS,
                     fp, path, "particle record");

      /*
       * this check is also very costly made like that.
       * either optimize or render it optional for some debugging mode
       */
      if (!isfinite ((double) record[0]) || !isfinite ((double) record[1]) ||
          !isfinite ((double) record[2]) || !isfinite ((double) record[3]) ||
          !isfinite ((double) record[4]) || !isfinite ((double) record[5]))
        die ("non-finite particle value in '%s' at index %zu", path, i);

      p->x[i]  = (dtype) record[0];
      p->y[i]  = (dtype) record[1];
      p->z[i]  = (dtype) record[2];
      p->vx[i] = (dtype) record[3];
      p->vy[i] = (dtype) record[4];
      p->vz[i] = (dtype) record[5];
    }

  if (fclose (fp) != 0)
    die ("error while closing input file '%s'", path);
}

/*
 * Write the current particle state in the same binary format accepted by the
 * reader.  Conversion to single precision is done explicitly record by record;
 * this is simple rather than maximally fast;
 * check in the code for initial condition generator
 */
void particles_write_binary (const char        *path,       // output file path
                                    const particles_t *p           // particle state to write
				    )
{
  FILE          *fp;
  const size_t   n   = p->n;
  uint64_t       n64 = (uint64_t) n;
  size_t         i;

  if ((size_t) n64 != n)
    die ("particle count cannot be represented in the binary header");

  fp = fopen (path, "wb");
  if (fp == NULL)
    die ("cannot open output file '%s'", path);

  checked_fwrite (nbody_binary_magic, sizeof nbody_binary_magic[0],
                  NBODY_BINARY_MAGIC_SIZE, fp, path, "binary magic");
  checked_fwrite (&n64, sizeof n64, 1u, fp, path, "particle count");

  for (i = 0u; i < n; ++i)
    {
      float  record[NBODY_BINARY_COMPONENTS];

      record[0] = dtype_to_storage_float (p->x[i],  "x",  i);
      record[1] = dtype_to_storage_float (p->y[i],  "y",  i);
      record[2] = dtype_to_storage_float (p->z[i],  "z",  i);
      record[3] = dtype_to_storage_float (p->vx[i], "vx", i);
      record[4] = dtype_to_storage_float (p->vy[i], "vy", i);
      record[5] = dtype_to_storage_float (p->vz[i], "vz", i);
      checked_fwrite (record, sizeof record[0], NBODY_BINARY_COMPONENTS,
                      fp, path, "particle record");
    }

  if (fclose (fp) != 0)
    die ("error while closing output file '%s'", path);
}




/* ======================================================================================== 

   : ------------------------------------------------------ :
   :  INTEGRATIION                                          :
   : ------------------------------------------------------ :
 */ 



/*
 * Naive direct O(N^2) softened gravitational acceleration.
 *
 * This is the most interesting kernel.  
 * A very transparent form: one i particle, one j loop, no Newton-third-law
 * reuse, one accumulator per component, and a scalar sqrt from libm.  That is
 * correct, but it leaves the optimisation space visible:
 *
 *   - which data qualifiers must be introduced for the input/output pointers?
 *   - exploit or deliberately avoid Newton's third law;
 *   - split the accumulators to shorten dependency chains;
 *   - use rsqrt plus Newton refinement, then quantify energy error;
 *   - block or transpose data to improve cache/TLB behaviour;
 *   - add OpenMP without atomics in the inner loop;
 *   - later replace the all-pairs loop with an MPI ring shift.
 *
 * ... reason about the needed qualifiers to unleash compiler's optimization
 *
 */
void compute_accelerations_naive (size_t  n,          // number of particles
                                         dtype   g,          // gravitational constant
                                         dtype   mass,       // mass of every source particle
                                         dtype   eps,        // Plummer softening length
                                         dtype * x,          // x positions, read-only
                                         dtype * y,          // y positions, read-only
                                         dtype * z,          // z positions, read-only
                                         dtype * ax,         // x acceleration, overwritten
                                         dtype * ay,         // y acceleration, overwritten
                                         dtype * az          // z acceleration, overwritten
					 )
{
  const dtype  eps2 = eps * eps;
  size_t       i;

// parallelizing outer loop as no dependencies exist between different i particles
// (threads write in different locations)
#pragma omp parallel for schedule(runtime)
  for (i = 0u; i < n; ++i)
    {
      const dtype  xi  = x[i];
      const dtype  yi  = y[i];
      const dtype  zi  = z[i];
      dtype        axi = (dtype) 0.0;
      dtype        ayi = (dtype) 0.0;
      dtype        azi = (dtype) 0.0;

      // making j private to each thread
      for (size_t j = 0u; j < n; ++j)
        {
          if (j != i)
            {
              const dtype  dx   = x[j] - xi;
              const dtype  dy   = y[j] - yi;
              const dtype  dz   = z[j] - zi;
              const dtype  r2   = dx * dx + dy * dy + dz * dz + eps2;
              const dtype  invr = (dtype) 1.0 / dtype_sqrt (r2);
              const dtype  s    = g * mass * invr * invr * invr;

              axi += dx * s;
              ayi += dy * s;
              azi += dz * s;
            }
        }

      ax[i] = axi;
      ay[i] = ayi;
      az[i] = azi;
    }
}

/*
 * Drift all particles by a time interval using the current velocities.  
 * The DKD leapfrog workflow calls it twice per step: a half-drift before the
 * force evaluation and a half-drift after the kick.
 *
 * Again: are data qualifiers missed for optimization?
 */
void drift (particles_t *p,       // particle positions are modified in place
                   dtype        dt       // drift interval, often 0.5 * full step
		   )
{
  size_t  n  = p->n;
  dtype  *x  = p->x;
  dtype  *y  = p->y;
  dtype  *z  = p->z;
  dtype  *vx = p->vx;
  dtype  *vy = p->vy;
  dtype  *vz = p->vz;
  size_t  i;

  for (i = 0u; i < n; ++i)
    {
      x[i] += dt * vx[i];
      y[i] += dt * vy[i];
      z[i] += dt * vz[i];
    }
}

/*
 * Kick all velocities using the current accelerations.  his is the K in DKD
 */
void kick (particles_t *p,       // particle velocities are modified in place
                  dtype        dt       // full kick interval
		  )
{
  size_t   n  = p->n;
  dtype  * vx = p->vx;
  dtype  * vy = p->vy;
  dtype  * vz = p->vz;
  dtype  * ax = p->ax;
  dtype  * ay = p->ay;
  dtype  * az = p->az;
  size_t   i;

  for (i = 0u; i < n; ++i)
    {
      vx[i] += dt * ax[i];
      vy[i] += dt * ay[i];
      vz[i] += dt * az[i];
    }
}

/*
 * Compute one DKD leapfrog step:
 *
 *   1. drift positions by dt/2;
 *   2. compute accelerations at the half-step positions;
 *   3. kick velocities by dt;
 *   4. drift positions by dt/2 with the updated velocities.
 *
 * This keeps positions and velocities synchronised at integer time levels 
 */
double leapfrog_dkd_step (particles_t *p,        // complete particle state, modified in place
                                 dtype        g,        // gravitational constant
                                 dtype        eps,      // softening length
                                 dtype        dt        // full time step
			       )
{
  double  force_start;
  double  force_time;

  drift (p, (dtype) 0.5 * dt);

  force_start = CPU_TIME_W;
  compute_accelerations_naive (p->n, g, p->mass, eps,
                               p->x, p->y, p->z,
                               p->ax, p->ay, p->az);
  force_time = CPU_TIME_W - force_start;

  kick (p, dt);
  drift (p, (dtype) 0.5 * dt);

  return force_time;
}

/*
 * Kinetic energy of the equal-mass system.
 * A long-double accumulator is used so that summation roundoff in the check is less likely to hide
 * errors caused by the integration or the force kernel.
 */
dtype kinetic_energy (const particles_t *p    // particle velocities are read-only
			     )
{
  size_t        n    = p->n;
  dtype         mass = p->mass;
  long double   sum  = 0.0L;
  size_t        i;

  for (i = 0u; i < n; ++i)
    {
      const long double  vx = (long double) p->vx[i];
      const long double  vy = (long double) p->vy[i];
      const long double  vz = (long double) p->vz[i];

      sum += vx * vx + vy * vy + vz * vz;
    }

  return (dtype) (0.5L * (long double) mass * sum);
}

/*
 * Simple O(N^2) potential-energy diagnostic for the same softened potential used
 * by the force kernel.  Not performance critical if called only every
 * K steps, and keeping it independent of compute_accelerations_naive makes it a
 * useful correctness check during optimisation.
 */
dtype potential_energy_naive (particles_t *p,        // particle positions are read-only
                                     dtype        g,        // gravitational constant
                                     dtype        eps       // softening length
				     )
{
  size_t        n    = p->n;
  dtype         eps2 = eps * eps;
  dtype         m2   = p->mass * p->mass;
  long double   sum  = 0.0L;
  size_t        i;
  size_t        j;

  for (i = 0u; i < n; ++i)
    {
      dtype  xi = p->x[i];
      dtype  yi = p->y[i];
      dtype  zi = p->z[i];

      for (j = i + 1u; j < n; ++j)
        {
          dtype  dx   = p->x[j] - xi;
          dtype  dy   = p->y[j] - yi;
          dtype  dz   = p->z[j] - zi;
          dtype  r2   = dx * dx + dy * dy + dz * dz + eps2;
          dtype  invr = (dtype) 1.0 / dtype_sqrt (r2);

          sum -= (long double) g * (long double) m2 * (long double) invr;
        }
    }

  return (dtype) sum;
}

/*
 * Total mechanical energy, returned together with kinetic and potential parts
 * for reporting.
 * The relative drift of this quantity is the main verification
 * metric 
 */
dtype total_energy (particles_t *p,           // complete particle state, read-only
                           dtype        g,           // gravitational constant
                           dtype        eps,         // softening length
                           dtype       *kinetic,     // output kinetic energy
                           dtype       *potential    // output potential energy
			   )
{
  *kinetic   = kinetic_energy (p);
  *potential = potential_energy_naive (p, g, eps);

  return *kinetic + *potential;
}
