/*
 * nbody_mpi.c
 *
 * MPI driver for the direct N-body solver. This first version is simple: 
 * it sets up MPI, but does not yet distribute particles. Every rank
 * reads the full input and runs the same integration; only rank 0 prints.
 * I will add the ring-shift decomposition in later commits.
 *
 * All the physics, I/O and integrator come from nbody_core, which is the shared module
 */

#include "nbody_core.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

/* MPI datatype matching dtype, so the acceleration exchange follows the
 * precision selected at compile time. */
#if defined (NBODY_USE_FLOAT)
#define NBODY_MPI_DTYPE MPI_FLOAT
#else
#define NBODY_MPI_DTYPE MPI_DOUBLE
#endif

/*
 * One distributed DKD leapfrog step. Positions/velocities are replicated on
 * every rank. Each rank computes the accelerations of its own home particles
 * [i0, i1) against all sources, then the accelerations are gathered so every
 * rank holds the full set before the kick. Returns the seconds spent in the
 * local force kernel.
 *
 * counts/displs describe the per-rank home ranges for the MPI_Allgatherv.
 */
static double distributed_dkd_step (particles_t *p, dtype g, dtype eps, dtype dt,
                                    size_t i0, size_t i1,
                                    const int *counts, const int *displs,
                                    MPI_Comm comm)
{
  double  force_start;
  double  force_time;

  drift (p, (dtype) 0.5 * dt);

  force_start = MPI_Wtime ();
  compute_accelerations_range (i0, i1, p->n, g, p->mass, eps,
                               p->x, p->y, p->z,
                               p->ax, p->ay, p->az);
  force_time = MPI_Wtime () - force_start;

  MPI_Allgatherv (MPI_IN_PLACE, 0, NBODY_MPI_DTYPE,
                  p->ax, counts, displs, NBODY_MPI_DTYPE, comm);
  MPI_Allgatherv (MPI_IN_PLACE, 0, NBODY_MPI_DTYPE,
                  p->ay, counts, displs, NBODY_MPI_DTYPE, comm);
  MPI_Allgatherv (MPI_IN_PLACE, 0, NBODY_MPI_DTYPE,
                  p->az, counts, displs, NBODY_MPI_DTYPE, comm);

  kick (p, dt);
  drift (p, (dtype) 0.5 * dt);

  return force_time;
}

/*
 * Print a compact command-line reference.
 */
static void print_usage (const char *program    // argv[0]
			 )
{
  fprintf (stderr,
           "usage: %s --input FILE [options]\n"
           "\n"
           "options:\n"
           "  --input FILE              input binary particle file (%s)\n"
           "  --output FILE             optional final-state binary file\n"
           "  --nsteps N                number of DKD steps (default: 10)\n"
           "  --dt X                    time step (default: 0.001)\n"
           "  --eps X                   softening length (default: 0.01)\n"
           "  --G X                     gravitational constant (default: 1)\n"
           "  --mass X                  particle mass (default: 1)\n"
           "  --energy-every N          diagnostic period in steps (default: 1)\n"
           "  --energy-tol X            warning tolerance for max relative drift (default: 1e-3)\n"
           "  --quiet                   only print final summary\n"
           "  --help                    show this help message\n",
           program, NBODY_BINARY_VERSION_TEXT);
}


int main (int argc, char **argv)
{
  const char  *input_path    = NULL;
  const char  *output_path   = NULL;
  size_t       nsteps        = 10u;
  size_t       energy_every  = 1u;
  dtype        dt            = (dtype) 1.0e-3;
  dtype        eps           = (dtype) 1.0e-2;
  dtype        g             = (dtype) 1.0;
  dtype        mass          = (dtype) 1.0;
  dtype        energy_tol    = (dtype) 1.0e-3;
  bool         quiet         = false;
  particles_t  particles;
  dtype        kinetic0;
  dtype        potential0;
  dtype        energy0;

  int          rank;
  int          ntasks;
  MPI_Comm     comm;


  // initialize MPI. THREAD_FUNNELED: only the main thread calls MPI, which
  // matches the hybrid design (OpenMP inside the kernel, MPI outside it).
  {
    int provided;

    MPI_Init_thread (&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
      MPI_Abort (MPI_COMM_WORLD, 1);
  }

  // best practice that I take from the course template: work on a private communicator
  MPI_Comm_dup (MPI_COMM_WORLD, &comm);
  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &ntasks);

  particles_init_empty (&particles);


  // parse CLI args
  for (int argi = 1; argi < argc; ++argi)
    {
      const char *value;

      if ((value = option_value (&argi, argc, argv, "--input")) != NULL)
        input_path = value;
      else if ((value = option_value (&argi, argc, argv, "--output")) != NULL)
        output_path = value;
      else if ((value = option_value (&argi, argc, argv, "--nsteps")) != NULL)
        nsteps = parse_size (value, "--nsteps");
      else if ((value = option_value (&argi, argc, argv, "--energy-every")) != NULL)
        energy_every = parse_size (value, "--energy-every");
      else if ((value = option_value (&argi, argc, argv, "--dt")) != NULL)
        dt = parse_dtype (value, "--dt");
      else if ((value = option_value (&argi, argc, argv, "--eps")) != NULL)
        eps = parse_dtype (value, "--eps");
      else if ((value = option_value (&argi, argc, argv, "--G")) != NULL)
        g = parse_dtype (value, "--G");
      else if ((value = option_value (&argi, argc, argv, "--mass")) != NULL)
        mass = parse_dtype (value, "--mass");
      else if ((value = option_value (&argi, argc, argv, "--energy-tol")) != NULL)
        energy_tol = parse_dtype (value, "--energy-tol");
      else if (strcmp (argv[argi], "--quiet") == 0)
        quiet = true;
      else if (strcmp (argv[argi], "--help") == 0)
        {
          if (rank == 0)
            print_usage (argv[0]);
          MPI_Finalize ();
          return EXIT_SUCCESS;
        }
      else
        {
          if (rank == 0)
            print_usage (argv[0]);
          die ("unknown option: %s", argv[argi]);
        }
    }

  if (input_path == NULL)
    {
      if (rank == 0)
        print_usage (argv[0]);
      die ("missing required --input FILE");
    }
  if (!(dt > (dtype) 0.0))
    die ("--dt must be positive");
  if (!(eps >= (dtype) 0.0))
    die ("--eps must be non-negative");
  if (!(g > (dtype) 0.0))
    die ("--G must be positive");
  if (!(mass > (dtype) 0.0))
    die ("--mass must be positive");
  if (energy_every == 0u)
    die ("--energy-every must be positive");
  if (!(energy_tol > (dtype) 0.0))
    die ("--energy-tol must be positive");


  // read particles. Every rank holds the full set; the O(N^2) force work is
  // split by home range and the accelerations are gathered each step.
  particles_read_binary (input_path, mass, &particles);

  // domain decomposition: give each rank a contiguous block of home particles.
  // The remainder is spread over the first (n % ntasks) ranks so block sizes
  // differ by at most one.
  const size_t  n         = particles.n;
  const size_t  base      = n / (size_t) ntasks;
  const size_t  remainder = n % (size_t) ntasks;

  int  *counts = checked_aligned_alloc ((size_t) ntasks * sizeof (int), NBODY_ALIGNMENT);
  int  *displs = checked_aligned_alloc ((size_t) ntasks * sizeof (int), NBODY_ALIGNMENT);

  {
    size_t  offset = 0u;
    for (int r = 0; r < ntasks; ++r)
      {
        const size_t  count = base + ((size_t) r < remainder ? 1u : 0u);
        counts[r] = (int) count;
        displs[r] = (int) offset;
        offset += count;
      }
  }

  const size_t  i0 = (size_t) displs[rank];
  const size_t  i1 = i0 + (size_t) counts[rank];

  energy0 = total_energy (&particles, g, eps, &kinetic0, &potential0);

  if ((rank == 0) && !quiet)
    {
      printf ("# MPI direct N-body DKD (work split by home range)\n");
      printf ("# ranks=%d arithmetic_dtype=%s binary_storage=float32 format=%s\n",
              ntasks, DTYPE_NAME, NBODY_BINARY_VERSION_TEXT);
      printf ("# N=%zu nsteps=%zu dt=%.17g eps=%.17g G=%.17g mass=%.17g\n",
              particles.n, nsteps, (double) dt, (double) eps,
              (double) g, (double) mass);
      printf ("# step time kinetic potential total rel_energy_drift\n");
      printf ("%zu %.17g %.17g %.17g %.17g %.17g\n",
              (size_t) 0u, 0.0, (double) kinetic0, (double) potential0,
              (double) energy0, 0.0);
    }


  // integration
  double max_rel_drift = 0.0;
  double force_time     = 0.0;

  for (size_t step = 1u; step <= nsteps; ++step)
    {
      force_time += distributed_dkd_step (&particles, g, eps, dt,
                                          i0, i1, counts, displs, comm);

      if (((step % energy_every) == 0u) || (step == nsteps))
        {
          dtype         kinetic;
          dtype         potential;
          const dtype   energy = total_energy (&particles, g, eps, &kinetic, &potential);
          const double  denom  = fmax (fabs ((double) energy0), (double) DTYPE_MIN_NORMAL);
          const double  rel    = fabs ((double) (energy - energy0)) / denom;

          if (rel > max_rel_drift)
            max_rel_drift = rel;
          if ((rank == 0) && !quiet)
            printf ("%zu %.17g %.17g %.17g %.17g %.17g\n",
                    step, (double) step * (double) dt, (double) kinetic,
                    (double) potential, (double) energy, rel);
        }
    }


  // write final file (rank 0 only for now)
  if ((output_path != NULL) && (rank == 0))
    particles_write_binary (output_path, &particles);


  // summary
  if (rank == 0)
    {
      printf ("# final: N=%zu steps=%zu ranks=%d arithmetic_dtype=%s max_relative_energy_drift=%.17g tolerance=%.17g status=%s\n",
              particles.n, nsteps, ntasks, DTYPE_NAME, max_rel_drift, (double) energy_tol,
              (max_rel_drift <= (double) energy_tol) ? "OK" : "WARNING");

      printf ("# timing: force_kernel_total=%.6g s force_kernel_per_step=%.6g s\n",
              force_time, force_time / (double) nsteps);
    }

  particles_free (&particles);
  free (counts);
  free (displs);

  MPI_Comm_free (&comm);
  MPI_Finalize ();

  return EXIT_SUCCESS;
}
