/*
 * yee_kernels.cl: Kernels de OpenCL para la simulación acústica (Yee FDTD).
 * Versión final y verificada.
 */

/*
 * Kernel 1: Actualiza la presión 'p'
 * Se ejecuta en un grid 2D de tamaño (nx, ny).
 */
__kernel void update_pressure(
    __global double* p,
    __global const double* u,
    __global const double* v,
    const double dt,
    const double u_dx,
    const double v_dy,
    const long nx,
    const long ny)
{
    long i = get_global_id(0);
    long j = get_global_id(1);

    // Evitar escribir fuera de los límites de la malla 'p'
    if (i >= nx || j >= ny) {
        return;
    }

    // El ancho de la malla 'u' es (nx + 1)
    long u_width = nx + 1;

    // Actualización de la presión usando índices explícitos
    p[j * nx + i] +=
        dt / u_dx * (u[j * u_width + (i + 1)] - u[j * u_width + i]) +
        dt / v_dy * (v[(j + 1) * nx + i] - v[j * nx + i]);
}


/*
 * Kernel 2: Actualiza la velocidad 'u' (componente x)
 * Se ejecuta en un grid 2D de tamaño (nx + 1, ny).
 */
__kernel void update_velocity_u(
    __global const double* p,
    __global double* u,
    const double dt,
    const double p_dx,
    const long nx,
    const long ny)
{
    long i = get_global_id(0);
    long j = get_global_id(1);
    
    long u_width = nx + 1;

    // El bucle original era de i=1 a nx-1. Replicamos esa condición.
    if (i > 0 && i < nx && j < ny) {
        u[j * u_width + i] += dt / p_dx * (p[j * nx + i] - p[j * nx + (i - 1)]);
    }
}


/*
 * Kernel 3: Actualiza la velocidad 'v' (componente y)
 * Se ejecuta en un grid 2D de tamaño (nx, ny + 1).
 */
__kernel void update_velocity_v(
    __global const double* p,
    __global double* v,
    const double dt,
    const double p_dy,
    const long nx,
    const long ny)
{
    long i = get_global_id(0);
    long j = get_global_id(1);

    // El bucle original era de j=1 a ny-1. Replicamos esa condición.
    if (i < nx && j > 0 && j < ny) {
        v[j * nx + i] += dt / p_dy * (p[j * nx + i] - p[(j - 1) * nx + i]);
    }
}