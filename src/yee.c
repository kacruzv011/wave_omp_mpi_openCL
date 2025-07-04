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
 * Basic single threaded implementation to compare parallelization overhead.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "yee_common.h"

/* 
 * Main program 
 */
int main(int argc, char *argv[])
{
    /* Parameters */
    double x[] = { 0, 1 };
    double y[] = { 0, 1 };
    double cfl = 0.99 / sqrt(2);        /* CFL condition: 
                                           c*dt/dx = cfl <= 1/sqrt(2) */
    double T = 1;
    double c = 1;
    long nx = 32;
    long ny = 32;
    struct py_field f;
    char outfile[STR_SIZE] = "yee.tsv";

    /* initial py_field */
    /*char outfile0[STR_SIZE] = "yee0.tsv";       */
    int write = 1;

    /* Parse parameters from commandline */
    py_parse_cmdline(&nx, NULL, outfile, &write, argc, argv);
    ny = nx;                    /* square domain */
    printf("Domain: %li x %li\n", nx, ny);

    /* Initialize */
    f = py_init_acoustic_field(nx, ny, x, y);
    py_apply_func(&f.p, py_gauss2d);    /* initial data */
    py_set_boundary(&f);
    /*if (write)                        */
    /*write_to_disk(f.p, outfile0); */

    /* Depends on the numerical variables initialized above */
    f.dt = cfl * f.p.dx / c;
    f.Nt = T / f.dt;

    /* timestep */
    double tic, toc;
    tic = py_gettime();
    py_timestep_leapfrog(&f, f.Nt);
    toc = py_gettime();
    printf("Elapsed: %f seconds\n", toc - tic);

    /* write to disk and free data */
    if (write)
        py_write_to_disk(f.p, outfile);

    py_free_acoustic_field(f);
    return EXIT_SUCCESS;
}
