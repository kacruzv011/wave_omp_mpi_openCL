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

#include "yee_common_mpi.h"

void send_grid_data(int taskid, long left, long right, long size,
                    double y[2])
{
    MPI_Send(&left, 1, MPI_LONG, taskid, BEGIN, MPI_COMM_WORLD);
    MPI_Send(&right, 1, MPI_LONG, taskid, BEGIN, MPI_COMM_WORLD);
    MPI_Send(&size, 1, MPI_LONG, taskid, BEGIN, MPI_COMM_WORLD);
    MPI_Send(y, 2, MPI_DOUBLE, taskid, BEGIN, MPI_COMM_WORLD);
}

void receive_grid_data(long *left, long *right, long *size, double *y,
                       MPI_Status * status)
{
    MPI_Recv(left, 1, MPI_LONG, MASTER, BEGIN, MPI_COMM_WORLD, status);
    MPI_Recv(right, 1, MPI_LONG, MASTER, BEGIN, MPI_COMM_WORLD, status);
    MPI_Recv(size, 1, MPI_LONG, MASTER, BEGIN, MPI_COMM_WORLD, status);
    MPI_Recv(y, 2, MPI_DOUBLE, MASTER, BEGIN, MPI_COMM_WORLD, status);
}

void send_field_data(long taskid, struct py_field *f,
                     struct py_cell_partition part)
{
    long begin = part.begin;
    long size = part.size;
    long nx = f->p.size_x;
    long begin_p = 0 + begin * nx;
    long begin_u = 0 + begin * (nx + 1);
    long begin_v = 0 + begin * nx;
    long size_p = size * nx;
    long size_u = size * (nx + 1);
    long size_v = size * nx;

    MPI_Send(&f->p.value[begin_p], size_p, MPI_DOUBLE, taskid, BEGIN, MPI_COMM_WORLD);
    MPI_Send(&f->u.value[begin_u], size_u, MPI_DOUBLE, taskid, BEGIN, MPI_COMM_WORLD);
    MPI_Send(&f->v.value[begin_v], size_v, MPI_DOUBLE, taskid, BEGIN, MPI_COMM_WORLD);
}

void receive_field_data(struct py_field *f, long size, MPI_Status * status)
{
    long nx = f->p.size_x;
    long begin_p = 0 + 1 * nx;
    long begin_u = 0 + 0 * (nx + 1);
    long begin_v = 0 + 0 * nx;
    long size_p = size * nx;
    long size_u = size * (nx + 1);
    long size_v = size * nx;

    MPI_Recv(&f->p.value[begin_p], size_p, MPI_DOUBLE, MASTER, BEGIN, MPI_COMM_WORLD, status);
    MPI_Recv(&f->u.value[begin_u], size_u, MPI_DOUBLE, MASTER, BEGIN, MPI_COMM_WORLD, status);
    MPI_Recv(&f->v.value[begin_v], size_v, MPI_DOUBLE, MASTER, BEGIN, MPI_COMM_WORLD, status);
}

void return_data(struct py_field *f, long size)
{
    long nx = f->p.size_x;
    long begin_p = 0 + 1 * nx;
    long begin_u = 0 + 0 * (nx + 1);
    long begin_v = 0 + 0 * nx;
    long size_p = size * nx;
    long size_u = size * (nx + 1);
    long size_v = size * nx;

    MPI_Send(&f->p.value[begin_p], size_p, MPI_DOUBLE, MASTER, COLLECT, MPI_COMM_WORLD);
    MPI_Send(&f->u.value[begin_u], size_u, MPI_DOUBLE, MASTER, COLLECT, MPI_COMM_WORLD);
    MPI_Send(&f->v.value[begin_v], size_v, MPI_DOUBLE, MASTER, COLLECT, MPI_COMM_WORLD);
}

void collect_data(long taskid, struct py_cell_partition part,
                  struct py_field *f, MPI_Status * status)
{
    long begin = part.begin;
    long size = part.size;
    long nx = f->p.size_x;
    long begin_p = 0 + begin * nx;
    long begin_u = 0 + begin * (nx + 1);
    long begin_v = 0 + begin * nx;
    long size_p = size * nx;
    long size_u = size * (nx + 1);
    long size_v = size * nx;

    MPI_Recv(&f->p.value[begin_p], size_p, MPI_DOUBLE, taskid, COLLECT, MPI_COMM_WORLD, status);
    MPI_Recv(&f->u.value[begin_u], size_u, MPI_DOUBLE, taskid, COLLECT, MPI_COMM_WORLD, status);
    MPI_Recv(&f->v.value[begin_v], size_v, MPI_DOUBLE, taskid, COLLECT, MPI_COMM_WORLD, status);
}

void leapfrog_master_p(double *restrict p, double *restrict u,
                       double *restrict v, long nx, double dt, double dx,
                       double dy, long size)
{
    for (long i = 0; i < nx; ++i)
        for (long j = 0; j < size; ++j)
            P(i, j) +=
                dt / dx * (U(i + 1, j) - U(i, j)) +
                dt / dy * (V(i, j + 1) - V(i, j));
}

void leapfrog_worker_p(double *restrict p, double *restrict u,
                       double *restrict v, long nx, double dt, double dx,
                       double dy, long size)
{
    for (long i = 0; i < nx; ++i)
        for (long j = 0; j < size; ++j)
            P(i, j + 1) +=
                dt / dx * (U(i + 1, j) - U(i, j)) +
                dt / dy * (V(i, j + 1) - V(i, j));
}

void leapfrog_master_uv(double *restrict p, double *restrict u,
                        double *restrict v, long nx, double dt, double dx,
                        double dy, long size)
{
    for (long i = 1; i < nx; ++i)
        for (long j = 0; j < size; ++j)
            U(i, j) += dt / dx * (P(i, j) - P(i - 1, j));

    for (long i = 0; i < nx; ++i)
        for (long j = 1; j < size; ++j)
            V(i, j) += dt / dy * (P(i, j) - P(i, j - 1));
}

void leapfrog_worker_uv(double *restrict p, double *restrict u,
                        double *restrict v, long nx, double dt, double dx,
                        double dy, long size)
{
    for (long i = 1; i < nx; ++i)
        for (long j = 0; j < size; ++j)
            U(i, j) += dt / dx * (P(i, j + 1) - P(i - 1, j + 1));

    for (long i = 0; i < nx; ++i)
        for (long j = 0; j < size; ++j)
            V(i, j) += dt / dy * (P(i, j + 1) - P(i, j));
}

void communicate_v(double *v, long left, long right, long nx, long size,
                   MPI_Status * status)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank % 2 == 0) { 
        if (left != NONE) {
            MPI_Send(&v[0 * nx], nx, MPI_DOUBLE, left, VTAG, MPI_COMM_WORLD);
        }
        if (right != NONE) {
            MPI_Recv(&v[size * nx], nx, MPI_DOUBLE, right, VTAG, MPI_COMM_WORLD, status);
        }
    } else { 
        if (right != NONE) {
            MPI_Recv(&v[size * nx], nx, MPI_DOUBLE, right, VTAG, MPI_COMM_WORLD, status);
        }
        if (left != NONE) {
            MPI_Send(&v[0 * nx], nx, MPI_DOUBLE, left, VTAG, MPI_COMM_WORLD);
        }
    }
}

void communicate_p(double *p, long left, long right, long nx, long size,
                   MPI_Status * status)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank % 2 == 0) { 
        if (right != NONE) {
            long begin_p = (left == NONE) ? (size - 1) * nx : size * nx;
            MPI_Send(&p[begin_p], nx, MPI_DOUBLE, right, PTAG, MPI_COMM_WORLD);
        }
        if (left != NONE) {
            MPI_Recv(&p[0], nx, MPI_DOUBLE, left, PTAG, MPI_COMM_WORLD, status);
        }
    } else { 
        if (right != NONE) {
            long begin_p = (left == NONE) ? (size - 1) * nx : size * nx;
            MPI_Send(&p[begin_p], nx, MPI_DOUBLE, right, PTAG, MPI_COMM_WORLD);
        }
        if (left != NONE) {
            MPI_Recv(&p[0], nx, MPI_DOUBLE, left, PTAG, MPI_COMM_WORLD, status);
        }
    }
}

void leapfrog_mpi(struct py_field *f, long left, long right, long size)
{
    MPI_Status status;
    double *p = f->p.value;
    double *u = f->u.value;
    double *v = f->v.value;
    long nx = f->p.size_x;

    communicate_v(v, left, right, nx, size, &status);

    if (left == NONE)
        leapfrog_master_p(p, u, v, nx, f->dt, f->u.dx, f->v.dy, size);
    else
        leapfrog_worker_p(p, u, v, nx, f->dt, f->u.dx, f->v.dy, size);

    communicate_p(p, left, right, nx, size, &status);

    if (left == NONE)
        leapfrog_master_uv(p, u, v, nx, f->dt, f->p.dx, f->p.dy, size);
    else
        leapfrog_worker_uv(p, u, v, nx, f->dt, f->p.dx, f->p.dy, size);
}

void leapfrog_master_p2(double *restrict p, double *restrict u,
                        double *restrict v, long nx, double dt, double dx,
                        double dy, long size, long right,
                        MPI_Request * received)
{
    for (long i = 0; i < nx; ++i)
        for (long j = 0; j < size - 1; ++j)
            P(i, j) +=
                dt / dx * (U(i + 1, j) - U(i, j)) +
                dt / dy * (V(i, j + 1) - V(i, j));

    if (right != NONE)
        MPI_Wait(received, MPI_STATUS_IGNORE);

    long j = size - 1;
    for (long i = 0; i < nx; ++i)
        P(i, j) +=
            dt / dx * (U(i + 1, j) - U(i, j)) +
            dt / dy * (V(i, j + 1) - V(i, j));
}

void leapfrog_worker_p2(double *restrict p, double *restrict u,
                        double *restrict v, long nx, double dt, double dx,
                        double dy, long size, long right,
                        MPI_Request * received)
{
    for (long i = 0; i < nx; ++i)
        for (long j = 0; j < size - 1; ++j)
            P(i, j + 1) +=
                dt / dx * (U(i + 1, j) - U(i, j)) +
                dt / dy * (V(i, j + 1) - V(i, j));

    if (right != NONE)
        MPI_Wait(received, MPI_STATUS_IGNORE);

    long j = size - 1;
    for (long i = 0; i < nx; ++i)
        P(i, j + 1) +=
            dt / dx * (U(i + 1, j) - U(i, j)) +
            dt / dy * (V(i, j + 1) - V(i, j));
}

void leapfrog_worker_uv2(double *restrict p, double *restrict u,
                         double *restrict v, long nx, double dt, double dx,
                         double dy, long size, long left,
                         MPI_Request * received)
{
    for (long i = 1; i < nx; ++i)
        for (long j = 0; j < size; ++j)
            U(i, j) += dt / dx * (P(i, j + 1) - P(i - 1, j + 1));
    for (long i = 0; i < nx; ++i)
        for (long j = 1; j < size; ++j)
            V(i, j) += dt / dy * (P(i, j + 1) - P(i, j));

    if (left != NONE)
        MPI_Wait(received, MPI_STATUS_IGNORE);

    long j = 0;
    for (long i = 0; i < nx; ++i)
        V(i, j) += dt / dy * (P(i, j + 1) - P(i, j));
}

void communicate_v2(double *v, long left, long right, long nx, long size,
                    MPI_Request * sent, MPI_Request * received)
{
    if (left != NONE) {
        MPI_Isend(&v[0], nx, MPI_DOUBLE, left, VTAG, MPI_COMM_WORLD, sent);
    }
    if (right != NONE) {
        MPI_Irecv(&v[size * nx], nx, MPI_DOUBLE, right, VTAG, MPI_COMM_WORLD, received);
    }
}

void communicate_p2(double *p, long left, long right, long nx, long size,
                    MPI_Request * sent, MPI_Request * received)
{
    if (left != NONE) {
        MPI_Irecv(&p[0], nx, MPI_DOUBLE, left, PTAG, MPI_COMM_WORLD, received);
    }
    if (right != NONE) {
        long begin_p = (left == NONE) ? (size - 1) * nx : size * nx;
        MPI_Isend(&p[begin_p], nx, MPI_DOUBLE, right, PTAG, MPI_COMM_WORLD, sent);
    }
}

void leapfrog_mpi2(struct py_field *f, long left, long right, long size)
{
    MPI_Request sent;
    MPI_Request received;
    double *p = f->p.value;
    double *u = f->u.value;
    double *v = f->v.value;
    long nx = f->p.size_x;

    communicate_v2(v, left, right, nx, size, &sent, &received);

    if (left == NONE)
        leapfrog_master_p2(p, u, v, nx, f->dt, f->u.dx, f->v.dy, size, right, &received);
    else
        leapfrog_worker_p2(p, u, v, nx, f->dt, f->u.dx, f->v.dy, size, right, &received);
    if (left != NONE)
        MPI_Wait(&sent, MPI_STATUS_IGNORE);

    communicate_p2(p, left, right, nx, size, &sent, &received);

    if (left == NONE)
        leapfrog_master_uv(p, u, v, nx, f->dt, f->p.dx, f->p.dy, size);
    else
        leapfrog_worker_uv2(p, u, v, nx, f->dt, f->p.dx, f->p.dy, size, left, &received);
    if (right != NONE)
        MPI_Wait(&sent, MPI_STATUS_IGNORE);
}

void timestep_mpi(struct py_field *f, long left, long right, long size)
{
    for (long n = 0; n < f->Nt; ++n)
        leapfrog_mpi(f, left, right, size);
}

void timestep_mpi2(struct py_field *f, long left, long right, long size)
{
    for (long n = 0; n < f->Nt; ++n)
        leapfrog_mpi2(f, left, right, size);
}