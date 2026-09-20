/*
 * nbody_mpi.c
 *
 * MPI driver for the direct N-body solver, using the ring-shift structure
 * required by the assignment. Each rank permanently owns N/P home particles.
 * A buffer chunk of positions rotates around the ring of ranks; at each ring
 * step forces between the home chunk and the buffer chunk are accumulated.
 * After P steps every home particle has seen every source.
 *
 * The physics, I/O and integrator kernels come from nbody_core.
 */

#include "nbody_core.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

/* MPI datatype matching dtype. */
#if defined (NBODY_USE_FLOAT)
#define NBODY_MPI_DTYPE MPI_FLOAT
#else
#define NBODY_MPI_DTYPE MPI_DOUBLE
#endif

/* Tags for the ring shift. */
#define RING_TAG 100


/*
 * Position buffers used by the ring shift. Two are needed (send and receive)
 * plus the accumulators. All are sized to the largest home chunk so any rank's
 * chunk fits while it rotates.
 */
typedef struct ring_s
{
  size_t  capacity;   // max chunk size across ranks
  dtype  *bx;         // current buffer x
  dtype  *by;
  dtype  *bz;
  dtype  *rx;         // receive buffer x
  dtype  *ry;
  dtype  *rz;
} ring_t;


static void ring_allocate (ring_t *r, size_t capacity)
{
  const size_t  bytes = capacity * sizeof (dtype);

  r->capacity = capacity;
  r->bx = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  r->by = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  r->bz = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  r->rx = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  r->ry = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  r->rz = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
}

static void ring_free (ring_t *r)
{
  free (r->bx); free (r->by); free (r->bz);
  free (r->rx); free (r->ry); free (r->rz);
  r->bx = r->by = r->bz = r->rx = r->ry = r->rz = NULL;
  r->capacity = 0u;
}


/* Wall time split for one ring pass: compute in the force kernel versus
 * communication in the MPI_Sendrecv shifts. */
typedef struct ring_times_s
{
  double  kernel;
  double  comm;
} ring_times_t;


/*
 * Accumulate the acceleration on the home particles from every particle in the
 * system, by rotating the home positions around the ring. The buffer starts as
 * a copy of the home chunk and is shifted ntasks-1 times.
 *
 * Communication is overlapped with computation: before working on the current
 * chunk we post non-blocking Isend/Irecv for the next chunk, so the transfer
 * proceeds while the force kernel runs. Chunk sizes are known locally (every
 * rank has all of them in chunk_sizes), so no count message is needed. The
 * chunk held at ring step s originated on rank (rank - s + P) % P.
 *
 * Returns the wall time split between the force kernel and the (non-overlapped)
 * ring comm, i.e. the time actually spent waiting in MPI.
 */
static ring_times_t ring_accumulate (particles_t *home, ring_t *ring,
                                     dtype g, dtype eps,
                                     int rank, int ntasks,
                                     const int *chunk_sizes, MPI_Comm comm)
{
  const int  next = (rank + 1) % ntasks;
  const int  prev = (rank - 1 + ntasks) % ntasks;
  const size_t  nhome = home->n;

  ring_times_t  times = { 0.0, 0.0 };
  double  t0;

  // zero the accumulators
  memset (home->ax, 0, nhome * sizeof (dtype));
  memset (home->ay, 0, nhome * sizeof (dtype));
  memset (home->az, 0, nhome * sizeof (dtype));

  // buffer starts as the home chunk itself
  memcpy (ring->bx, home->x, nhome * sizeof (dtype));
  memcpy (ring->by, home->y, nhome * sizeof (dtype));
  memcpy (ring->bz, home->z, nhome * sizeof (dtype));
  int  buf_count = (int) nhome;

  for (int step = 0; step < ntasks; ++step)
    {
      const bool  same_chunk = (step == 0);
      MPI_Request requests[6];
      int         recv_count = 0;

      // start the transfer of the next chunk before computing this one
      if (step < ntasks - 1)
        {
          // the chunk that will arrive originated (step+1) hops upstream
          const int  src_rank = (rank - (step + 1) + ntasks) % ntasks;
          recv_count = chunk_sizes[src_rank];

          MPI_Irecv (ring->rx, recv_count, NBODY_MPI_DTYPE, prev, RING_TAG, comm, &requests[0]);
          MPI_Irecv (ring->ry, recv_count, NBODY_MPI_DTYPE, prev, RING_TAG, comm, &requests[1]);
          MPI_Irecv (ring->rz, recv_count, NBODY_MPI_DTYPE, prev, RING_TAG, comm, &requests[2]);
          MPI_Isend (ring->bx, buf_count,  NBODY_MPI_DTYPE, next, RING_TAG, comm, &requests[3]);
          MPI_Isend (ring->by, buf_count,  NBODY_MPI_DTYPE, next, RING_TAG, comm, &requests[4]);
          MPI_Isend (ring->bz, buf_count,  NBODY_MPI_DTYPE, next, RING_TAG, comm, &requests[5]);
        }

      t0 = MPI_Wtime ();
      accelerate_from_sources (nhome, (size_t) buf_count, same_chunk,
                               g, home->mass, eps,
                               home->x, home->y, home->z,
                               ring->bx, ring->by, ring->bz,
                               home->ax, home->ay, home->az);
      times.kernel += MPI_Wtime () - t0;

      // collect the transfer that ran alongside the compute, then adopt the
      // received chunk as the current buffer
      if (step < ntasks - 1)
        {
          t0 = MPI_Wtime ();
          MPI_Waitall (6, requests, MPI_STATUSES_IGNORE);
          times.comm += MPI_Wtime () - t0;

          dtype *tx = ring->bx; ring->bx = ring->rx; ring->rx = tx;
          dtype *ty = ring->by; ring->by = ring->ry; ring->ry = ty;
          dtype *tz = ring->bz; ring->bz = ring->rz; ring->rz = tz;
          buf_count = recv_count;
        }
    }

  return times;
}


/*
 * One distributed DKD leapfrog step using the ring shift for the force. Only
 * the home particles are integrated. Returns the kernel/comm time split.
 */
static ring_times_t ring_dkd_step (particles_t *home, ring_t *ring,
                                   dtype g, dtype eps, dtype dt,
                                   int rank, int ntasks,
                                   const int *chunk_sizes, MPI_Comm comm)
{
  drift (home, (dtype) 0.5 * dt);
  ring_times_t  times = ring_accumulate (home, ring, g, eps, rank, ntasks,
                                         chunk_sizes, comm);
  kick (home, dt);
  drift (home, (dtype) 0.5 * dt);

  return times;
}


/*
 * Total energy of the distributed system. Kinetic is a local sum reduced over
 * ranks. Potential is the softened pairwise sum: each rank accumulates the
 * potential of its home particles against every source via one ring pass, then
 * the partial sums are reduced. The 1/2 double-counting is corrected at the end.
 */
static dtype ring_total_energy (particles_t *home, ring_t *ring,
                                dtype g, dtype eps,
                                int rank, int ntasks, MPI_Comm comm,
                                dtype *kinetic_out, dtype *potential_out)
{
  const size_t  nhome = home->n;
  const int     next  = (rank + 1) % ntasks;
  const int     prev  = (rank - 1 + ntasks) % ntasks;
  const dtype   eps2  = eps * eps;
  const dtype   m2    = home->mass * home->mass;

  // kinetic: local sum then reduce
  long double  kin_local = 0.0L;
  for (size_t i = 0u; i < nhome; ++i)
    {
      const long double  vx = (long double) home->vx[i];
      const long double  vy = (long double) home->vy[i];
      const long double  vz = (long double) home->vz[i];
      kin_local += vx * vx + vy * vy + vz * vz;
    }
  kin_local *= 0.5L * (long double) home->mass;

  // potential: ring pass over source chunks. Each ordered pair (home i, source
  // j) is counted once here and once when the roles are reversed on another
  // rank, so the total is halved at the end.
  memcpy (ring->bx, home->x, nhome * sizeof (dtype));
  memcpy (ring->by, home->y, nhome * sizeof (dtype));
  memcpy (ring->bz, home->z, nhome * sizeof (dtype));
  int  buf_count = (int) nhome;

  long double  pot_local = 0.0L;

  for (int step = 0; step < ntasks; ++step)
    {
      const bool  same_chunk = (step == 0);

      for (size_t i = 0u; i < nhome; ++i)
        {
          const dtype  xi = home->x[i];
          const dtype  yi = home->y[i];
          const dtype  zi = home->z[i];

          for (int j = 0; j < buf_count; ++j)
            {
              if (!same_chunk || ((size_t) j != i))
                {
                  const dtype  dx   = ring->bx[j] - xi;
                  const dtype  dy   = ring->by[j] - yi;
                  const dtype  dz   = ring->bz[j] - zi;
                  const dtype  r2   = dx * dx + dy * dy + dz * dz + eps2;
                  const dtype  invr = (dtype) 1.0 / dtype_sqrt (r2);

                  pot_local -= (long double) g * (long double) m2 * (long double) invr;
                }
            }
        }

      if (step < ntasks - 1)
        {
          int  recv_count = 0;

          MPI_Sendrecv (&buf_count, 1, MPI_INT, next, RING_TAG,
                        &recv_count, 1, MPI_INT, prev, RING_TAG,
                        comm, MPI_STATUS_IGNORE);
          MPI_Sendrecv (ring->bx, buf_count, NBODY_MPI_DTYPE, next, RING_TAG,
                        ring->rx, recv_count, NBODY_MPI_DTYPE, prev, RING_TAG,
                        comm, MPI_STATUS_IGNORE);
          MPI_Sendrecv (ring->by, buf_count, NBODY_MPI_DTYPE, next, RING_TAG,
                        ring->ry, recv_count, NBODY_MPI_DTYPE, prev, RING_TAG,
                        comm, MPI_STATUS_IGNORE);
          MPI_Sendrecv (ring->bz, buf_count, NBODY_MPI_DTYPE, next, RING_TAG,
                        ring->rz, recv_count, NBODY_MPI_DTYPE, prev, RING_TAG,
                        comm, MPI_STATUS_IGNORE);

          dtype *tx = ring->bx; ring->bx = ring->rx; ring->rx = tx;
          dtype *ty = ring->by; ring->by = ring->ry; ring->ry = ty;
          dtype *tz = ring->bz; ring->bz = ring->rz; ring->rz = tz;
          buf_count = recv_count;
        }
    }

  // the pairwise sum above double counts every pair, so halve it
  pot_local *= 0.5L;

  double  kin_global = 0.0;
  double  pot_global = 0.0;
  double  kin_send   = (double) kin_local;
  double  pot_send   = (double) pot_local;

  MPI_Allreduce (&kin_send, &kin_global, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce (&pot_send, &pot_global, 1, MPI_DOUBLE, MPI_SUM, comm);

  *kinetic_out   = (dtype) kin_global;
  *potential_out = (dtype) pot_global;

  return (dtype) (kin_global + pot_global);
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

  MPI_Comm_dup (MPI_COMM_WORLD, &comm);
  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &ntasks);


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


  // ························································
  // read the full file, then keep only this rank's home slice. The remainder
  // (N % ntasks) is spread over the first ranks so chunks differ by at most one.
  particles_t  full;
  particles_init_empty (&full);
  particles_read_binary (input_path, mass, &full);

  const size_t  ntot      = full.n;
  const size_t  base      = ntot / (size_t) ntasks;
  const size_t  remainder = ntot % (size_t) ntasks;

  size_t  home_start = 0u;
  for (int r = 0; r < rank; ++r)
    home_start += base + ((size_t) r < remainder ? 1u : 0u);
  const size_t  nhome    = base + ((size_t) rank < remainder ? 1u : 0u);
  const size_t  max_home = base + (remainder > 0u ? 1u : 0u);

  particles_t  home;
  particles_allocate (&home, nhome, mass);
  memcpy (home.x,  full.x  + home_start, nhome * sizeof (dtype));
  memcpy (home.y,  full.y  + home_start, nhome * sizeof (dtype));
  memcpy (home.z,  full.z  + home_start, nhome * sizeof (dtype));
  memcpy (home.vx, full.vx + home_start, nhome * sizeof (dtype));
  memcpy (home.vy, full.vy + home_start, nhome * sizeof (dtype));
  memcpy (home.vz, full.vz + home_start, nhome * sizeof (dtype));
  particles_free (&full);

  ring_t  ring;
  ring_allocate (&ring, max_home);

  // every rank's home size, so the ring can pre-post receives of the right
  // length without exchanging counts
  int  *chunk_sizes = checked_aligned_alloc ((size_t) ntasks * sizeof (int), NBODY_ALIGNMENT);
  for (int r = 0; r < ntasks; ++r)
    chunk_sizes[r] = (int) (base + ((size_t) r < remainder ? 1u : 0u));


  // ························································
  // energy baseline
  dtype  kinetic0;
  dtype  potential0;
  const dtype  energy0 = ring_total_energy (&home, &ring, g, eps,
                                            rank, ntasks, comm,
                                            &kinetic0, &potential0);

  if ((rank == 0) && !quiet)
    {
      printf ("# MPI direct N-body DKD (ring shift)\n");
      printf ("# ranks=%d arithmetic_dtype=%s binary_storage=float32 format=%s\n",
              ntasks, DTYPE_NAME, NBODY_BINARY_VERSION_TEXT);
      printf ("# N=%zu nsteps=%zu dt=%.17g eps=%.17g G=%.17g mass=%.17g\n",
              ntot, nsteps, (double) dt, (double) eps, (double) g, (double) mass);
      printf ("# step time kinetic potential total rel_energy_drift\n");
      printf ("%zu %.17g %.17g %.17g %.17g %.17g\n",
              (size_t) 0u, 0.0, (double) kinetic0, (double) potential0,
              (double) energy0, 0.0);
    }


  // ························································
  // integration
  double max_rel_drift = 0.0;
  double force_time     = 0.0;
  double comm_time      = 0.0;

  for (size_t step = 1u; step <= nsteps; ++step)
    {
      const ring_times_t  st = ring_dkd_step (&home, &ring, g, eps, dt,
                                              rank, ntasks, chunk_sizes, comm);
      force_time += st.kernel;
      comm_time  += st.comm;

      if (((step % energy_every) == 0u) || (step == nsteps))
        {
          dtype         kinetic;
          dtype         potential;
          const dtype   energy = ring_total_energy (&home, &ring, g, eps,
                                                    rank, ntasks, comm,
                                                    &kinetic, &potential);
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


  // ························································
  // gather the home slices back to rank 0 and write the final state
  if (output_path != NULL)
    {
      int  *counts = NULL;
      int  *displs = NULL;
      particles_t  gathered;
      particles_init_empty (&gathered);

      if (rank == 0)
        {
          counts = checked_aligned_alloc ((size_t) ntasks * sizeof (int), NBODY_ALIGNMENT);
          displs = checked_aligned_alloc ((size_t) ntasks * sizeof (int), NBODY_ALIGNMENT);
          size_t  offset = 0u;
          for (int r = 0; r < ntasks; ++r)
            {
              const size_t  c = base + ((size_t) r < remainder ? 1u : 0u);
              counts[r] = (int) c;
              displs[r] = (int) offset;
              offset += c;
            }
          particles_allocate (&gathered, ntot, mass);
        }

      const int  send = (int) nhome;
      MPI_Gatherv (home.x,  send, NBODY_MPI_DTYPE, gathered.x,  counts, displs, NBODY_MPI_DTYPE, 0, comm);
      MPI_Gatherv (home.y,  send, NBODY_MPI_DTYPE, gathered.y,  counts, displs, NBODY_MPI_DTYPE, 0, comm);
      MPI_Gatherv (home.z,  send, NBODY_MPI_DTYPE, gathered.z,  counts, displs, NBODY_MPI_DTYPE, 0, comm);
      MPI_Gatherv (home.vx, send, NBODY_MPI_DTYPE, gathered.vx, counts, displs, NBODY_MPI_DTYPE, 0, comm);
      MPI_Gatherv (home.vy, send, NBODY_MPI_DTYPE, gathered.vy, counts, displs, NBODY_MPI_DTYPE, 0, comm);
      MPI_Gatherv (home.vz, send, NBODY_MPI_DTYPE, gathered.vz, counts, displs, NBODY_MPI_DTYPE, 0, comm);

      if (rank == 0)
        {
          particles_write_binary (output_path, &gathered);
          particles_free (&gathered);
          free (counts);
          free (displs);
        }
    }


  // ························································
  // summary. Timings vary across ranks, so report the slowest rank (the
  // critical path) for each part.
  double  force_max = 0.0;
  double  comm_max  = 0.0;
  MPI_Reduce (&force_time, &force_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce (&comm_time,  &comm_max,  1, MPI_DOUBLE, MPI_MAX, 0, comm);

  if (rank == 0)
    {
      printf ("# final: N=%zu steps=%zu ranks=%d arithmetic_dtype=%s max_relative_energy_drift=%.17g tolerance=%.17g status=%s\n",
              ntot, nsteps, ntasks, DTYPE_NAME, max_rel_drift, (double) energy_tol,
              (max_rel_drift <= (double) energy_tol) ? "OK" : "WARNING");

      printf ("# timing: force_kernel_total=%.6g s comm_total=%.6g s force_per_step=%.6g s comm_per_step=%.6g s\n",
              force_max, comm_max,
              force_max / (double) nsteps, comm_max / (double) nsteps);
    }

  particles_free (&home);
  ring_free (&ring);
  free (chunk_sizes);

  MPI_Comm_free (&comm);
  MPI_Finalize ();

  return EXIT_SUCCESS;
}
