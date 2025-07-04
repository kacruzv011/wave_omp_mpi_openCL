/*
 *    Copyright (C) 2012 Jon Haggblad
 *
 *    This file is part of ParYee.
 *
 *    ParYee is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    ParYee is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with ParYee.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * OpenMP shared memory parallelization
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <omp.h>

#include "yee_common.h"

/* 
 * Main program 
 */
int main(int argc, char *argv[])
{
    /* Parameters */
    double x[] = { 0, 1 };
    double y[] = { 0, 1 };
    double cfl = 0.99 / sqrt(2);        /* CFL condition: c*dt/dx = cfl <= 1/sqrt(2) */
    double T = 1;
    double c = 1;
    long nx = 32;
    long ny = 32;
    struct py_field f;
    char outfile[STR_SIZE] = "yee_naive_omp.tsv";
    int write = 1;
    long threads = 4;
    struct py_cell_partition *part;

    /* Parse parameters from commandline */
    py_parse_cmdline(&nx, &threads, outfile, &write, argc, argv);
    ny = nx;                    /* square domain */
    omp_set_num_threads(threads);
    printf("Domain: %li x %li\n", nx, ny);
    printf("OpenMP threads: %li\n", threads);

    /* Initialize */
    f = py_init_acoustic_field(nx, ny, x, y);
    py_apply_func(&f.p, py_gauss2d);    /* initial data */
    py_set_boundary(&f);

    /* Depends on the numerical variables initialized above */
    f.dt = cfl * f.p.dx / c;
    f.Nt = T / f.dt;

    /* Partition the grid */
    part = py_partition_grid(threads, nx);

    /* Maybe we can make optimization of the inner loop a bit easier for the
     * compiler? 
     * This is probably not needed. */
    double *p = f.p.value;
    double *u = f.u.value;
    double *v = f.v.value;

    /* timestep */
    long n, i, j;
    double tic, toc;
    tic = py_gettime();

    {
        /* private variables used in the time stepping */
        double dt = f.dt;
        double Nt = f.Nt;

        for (n = 0; n < Nt; ++n) {

            /* update the pressure (p) */
#pragma omp parallel for
            for (i = 0; i < nx; ++i) {
                for (j = 0; j < ny; ++j) {
                    P(i, j) +=
                        dt / f.u.dx * (U(i + 1, j) - U(i, j)) +
                        dt / f.v.dy * (V(i, j + 1) - V(i, j));
                }
            }

            /* update the velocity (u,v) */
#pragma omp parallel for
            for (i = 1; i < nx; ++i) {
                for (j = 0; j < ny; ++j) {
                    U(i, j) += dt / f.p.dx * (P(i, j) - P(i - 1, j));
                }
            }

#pragma omp parallel for
            for (i = 0; i < nx; ++i)
                /*for (j = 1; j < ny - 1; ++j) */
                for (j = 1; j < ny; ++j)
                    V(i, j) += dt / f.p.dy * (P(i, j) - P(i, j - 1));
        }
    }

    toc = py_gettime();
    printf("Elapsed: %f seconds\n", toc - tic);

    /* write to disk and free data */
    if (write)
        py_write_to_disk(f.p, outfile);
    free(part);
    py_free_acoustic_field(f);

    return EXIT_SUCCESS;
}
