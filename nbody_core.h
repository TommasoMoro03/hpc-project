#ifndef NBODY_CORE_H
#define NBODY_CORE_H

#include "nbody_common.h"

#include <stdarg.h>
#include <stddef.h>

/*
 * Shared N-body core: particle storage, binary I/O, the direct force kernel,
 * the DKD leapfrog integrator and the energy diagnostics. Kept separate so the
 * serial/OpenMP driver and the MPI driver can reuse it without duplication.
 */

typedef struct particles_s
{
  size_t  n;
  dtype   mass;
  dtype  *x;
  dtype  *y;
  dtype  *z;
  dtype  *vx;
  dtype  *vy;
  dtype  *vz;
  dtype  *ax;
  dtype  *ay;
  dtype  *az;
} particles_t;

/* Print a message to stderr and exit with failure. */
void die (const char *format, ...);

/* Cache-line aligned allocation, aborts on failure. */
void *checked_aligned_alloc (size_t nbytes, size_t alignment);

/* Particle container lifecycle. */
void particles_init_empty (particles_t *p);
void particles_allocate (particles_t *p, size_t n, dtype mass);
void particles_free (particles_t *p);

/* Binary I/O in the native NBODYF1 format. */
void particles_read_binary (const char *path, dtype mass, particles_t *p);
void particles_write_binary (const char *path, const particles_t *p);

/* Direct O(N^2) softened gravitational acceleration. */
void compute_accelerations_naive (size_t n, dtype g, dtype mass, dtype eps,
                                   dtype *x, dtype *y, dtype *z,
                                   dtype *ax, dtype *ay, dtype *az);

/* DKD leapfrog pieces. leapfrog_dkd_step returns the seconds spent in the
 * force kernel. */
void drift (particles_t *p, dtype dt);
void kick (particles_t *p, dtype dt);
double leapfrog_dkd_step (particles_t *p, dtype g, dtype eps, dtype dt);

/* Energy diagnostics for the verification metric. */
dtype kinetic_energy (const particles_t *p);
dtype potential_energy_naive (particles_t *p, dtype g, dtype eps);
dtype total_energy (particles_t *p, dtype g, dtype eps,
                    dtype *kinetic, dtype *potential);

#endif
