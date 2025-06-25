/*
 *  yee_blocking_mpi.c - main() - VERSIÓN 4.3 CORREGIDA
 *  Se corrige la inconsistencia de parámetros de simulación (Nt) entre
 *  Máster y Workers, que causaba un desajuste de comunicación.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "mpi.h"

#include "yee_common.h"
#include "yee_common_mpi.h"

int main(int argc, char *argv[])
{
    /* --- 1. Inicialización de MPI --- */
    int numtasks, taskid;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    
    /* --- 2. Parámetros y parseo --- */
    long nx = 32;
    long ny = 32;
    char outfile[STR_SIZE] = "yee_mpi.tsv";
    int write = 1;
    long Nt; // Variable para Nt que será común a todos
    double dt; // Variable para dt que será común a todos

    if (taskid == MASTER) {
        py_parse_cmdline(&nx, NULL, outfile, &write, argc, argv);
        ny = nx;
    }

    // El MASTER envía los parámetros de la malla a todos los workers.
    MPI_Bcast(&nx, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
    
    // El MASTER calcula los parámetros de tiempo y los envía.
    if (taskid == MASTER) {
        double T = 1;
        double c = 1;
        double cfl = 0.99 / sqrt(2);
        double dx = 1.0 / (nx - 1); // Asumiendo dominio de 1.0
        dt = cfl * dx / c;
        Nt = T / dt;
    }
    
    // Sincronización de los parámetros de tiempo para todos los procesos.
    MPI_Bcast(&Nt, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);


    /* --- 3. Lógica Master / Worker --- */
    if (taskid == MASTER) {
        /********************* Master code (ESTRUCTURA ORIGINAL) *********************/
        struct py_field f;
        struct py_cell_partition *part;
        double tic, toc;
        double x[] = { 0, 1 };
        double y[] = { 0, 1 };

        printf("Domain: %li x %li\n", nx, ny);
        printf("MPI processes: %d\n", numtasks);
        
        f = py_init_acoustic_field(nx, ny, x, y);
        py_apply_func(&f.p, py_gauss2d);
        py_set_boundary(&f);

        // Asigna los valores de tiempo ya calculados y sincronizados
        f.dt = dt;
        f.Nt = Nt;

        part = py_partition_grid(numtasks, ny);
        
        tic = py_gettime();

        // Envía los datos iniciales a todos los workers
        for (long worker_id = 1; worker_id < numtasks; ++worker_id) {
            long left = worker_id - 1;
            long right = (worker_id == numtasks - 1) ? NONE : worker_id + 1;
            double y_part[2];
            py_get_partition_coords(part[worker_id], &f, y_part);
            send_grid_data(worker_id, left, right, part[worker_id].size, y_part);
            send_field_data(worker_id, &f, part[worker_id]);
        }

        // El Máster y todos los workers ahora ejecutan timestep_mpi en paralelo
        long master_left = NONE;
        long master_right = (numtasks > 1) ? 1 : NONE;
        timestep_mpi(&f, master_left, master_right, part[0].size);

        // El Máster recoge los resultados de todos los workers
        for (long worker_id = 1; worker_id < numtasks; ++worker_id)
            collect_data(worker_id, part[worker_id], &f, &status);
        
        toc = py_gettime();
        printf("Elapsed: %f seconds\n", toc - tic);

        if (write)
            py_write_to_disk(f.p, outfile);
            
        free(part);
        py_free_acoustic_field(f);

    } else {
        /********************* Worker code (CORREGIDO) *********************/
        struct py_field f;
        long left, right, size;
        double y_part[2];
        double x[] = { 0, 1 };

        // El worker recibe su configuración
        receive_grid_data(&left, &right, &size, y_part, &status);
        f = py_init_local_acoustic_field(nx, size, x, y_part);
        receive_field_data(&f, size, &status);
        
        // Asigna los valores de tiempo ya recibidos del Máster
        f.dt = dt;
        f.Nt = Nt;

        if (right == NONE) {
            long j = size;
            for (long i = 0; i < nx; ++i)
                py_assign_to(f.v, i, j, 0);
        }

        // Ejecuta el cálculo en paralelo con el Máster y otros workers
        timestep_mpi(&f, left, right, size);
        // Devuelve los datos al Máster
        return_data(&f, size);
        
        py_free_acoustic_field(f);
    }
    
    MPI_Finalize();
    return 0;
}