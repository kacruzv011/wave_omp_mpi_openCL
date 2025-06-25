/*
 *    Copyright (C) 2012 Jon Haggblad
 *    Adaptado para OpenCL.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <unistd.h> // <-- CAMBIO 1: Añadido para la función getopt()

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "yee_common.h"

#define MAX_SOURCE_SIZE (0x100000)

char* load_kernel_source(const char* filename) {
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: No se pudo cargar el kernel '%s'\n", filename);
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    source_str[source_size] = '\0';
    return source_str;
}

int main(int argc, char *argv[])
{
    /* 1. INICIALIZACIÓN ORIGINAL DEL PROBLEMA */
    double x[] = { 0, 1 };
    double y[] = { 0, 1 };
    double cfl = 0.99 / sqrt(2);
    double T = 1;
    double c = 1;
    long nx = 32;
    long ny = 32;
    struct py_field f;
    char outfile[STR_SIZE] = "yee_opencl.tsv";
    int write = 1;
    long lws = 16; // <-- CAMBIO 2: Variable para Local Work Size, con valor por defecto 16

    // <-- CAMBIO 3: Reemplazamos py_parse_cmdline por un bucle getopt estándar
    // Esto nos da la flexibilidad de añadir nuestro propio argumento -l.
    int opt;
    while ((opt = getopt(argc, argv, "n:o:ql:")) != -1) {
        switch (opt) {
            case 'n':
                nx = atol(optarg);
                break;
            case 'o':
                strncpy(outfile, optarg, STR_SIZE);
                outfile[STR_SIZE - 1] = '\0';
                break;
            case 'q':
                write = 0;
                break;
            case 'l': // Nuevo argumento para el Local Work Size
                lws = atol(optarg);
                break;
            default:
                fprintf(stderr, "Uso: %s [-n grid_size] [-o outfile] [-q] [-l local_work_size]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    ny = nx;
    printf("Domain: %li x %li\n", nx, ny);
    printf("Usando OpenCL en GPU\n");
    printf("Local Work Size: %li x %li\n", lws, lws);

    f = py_init_acoustic_field(nx, ny, x, y);
    py_apply_func(&f.p, py_gauss2d);
    py_set_boundary(&f);

    f.dt = cfl * f.p.dx / c;
    f.Nt = T / f.dt;
    
    /* 2. CONFIGURACIÓN DE OPENCL */
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

    /* 3. CREAR BUFFERS DE MEMORIA EN LA GPU */
    size_t p_size = nx * ny * sizeof(double);
    size_t u_size = (nx + 1) * ny * sizeof(double);
    size_t v_size = nx * (ny + 1) * sizeof(double);

    cl_mem p_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, p_size, NULL, &ret);
    cl_mem u_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, u_size, NULL, &ret);
    cl_mem v_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, v_size, NULL, &ret);

    /* 4. COPIAR DATOS INICIALES DE CPU A GPU */
    clEnqueueWriteBuffer(command_queue, p_mem_obj, CL_TRUE, 0, p_size, f.p.value, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, u_mem_obj, CL_TRUE, 0, u_size, f.u.value, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, v_mem_obj, CL_TRUE, 0, v_size, f.v.value, 0, NULL, NULL);

    /* 5. CARGAR, COMPILAR Y PREPARAR LOS KERNELS */
    char* source_str = load_kernel_source("src/yee_kernels.cl");
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &ret);
    free(source_str);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "--- Error de compilacion del Kernel ---\n%s\n", log);
        free(log);
        exit(1);
    }
    cl_kernel kernel_p = clCreateKernel(program, "update_pressure", &ret);
    cl_kernel kernel_u = clCreateKernel(program, "update_velocity_u", &ret);
    cl_kernel kernel_v = clCreateKernel(program, "update_velocity_v", &ret);

    clSetKernelArg(kernel_p, 0, sizeof(cl_mem), &p_mem_obj);
    clSetKernelArg(kernel_p, 1, sizeof(cl_mem), &u_mem_obj);
    clSetKernelArg(kernel_p, 2, sizeof(cl_mem), &v_mem_obj);
    clSetKernelArg(kernel_p, 3, sizeof(double), &f.dt);
    clSetKernelArg(kernel_p, 4, sizeof(double), &f.u.dx);
    clSetKernelArg(kernel_p, 5, sizeof(double), &f.v.dy);
    clSetKernelArg(kernel_p, 6, sizeof(long), &nx);
    clSetKernelArg(kernel_p, 7, sizeof(long), &ny);
    
    clSetKernelArg(kernel_u, 0, sizeof(cl_mem), &p_mem_obj);
    clSetKernelArg(kernel_u, 1, sizeof(cl_mem), &u_mem_obj);
    clSetKernelArg(kernel_u, 2, sizeof(double), &f.dt);
    clSetKernelArg(kernel_u, 3, sizeof(double), &f.p.dx);
    clSetKernelArg(kernel_u, 4, sizeof(long), &nx);
    clSetKernelArg(kernel_u, 5, sizeof(long), &ny);
    
    clSetKernelArg(kernel_v, 0, sizeof(cl_mem), &p_mem_obj);
    clSetKernelArg(kernel_v, 1, sizeof(cl_mem), &v_mem_obj);
    clSetKernelArg(kernel_v, 2, sizeof(double), &f.dt);
    clSetKernelArg(kernel_v, 3, sizeof(double), &f.p.dy);
    clSetKernelArg(kernel_v, 4, sizeof(long), &nx);
    clSetKernelArg(kernel_v, 5, sizeof(long), &ny);

    /* 6. BUCLE DE SIMULACIÓN (EJECUCIÓN DE KERNELS) */
    double tic, toc;
    tic = py_gettime();
    long n;
    
    // <-- CAMBIO 4: Usamos la variable 'lws' para configurar el tamaño del grupo
    size_t local_work_size[2] = { (size_t)lws, (size_t)lws };
    
    size_t global_work_size_p[2] = {(nx + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0], (ny + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1]};
    size_t global_work_size_u[2] = {(nx + 1 + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0], (ny + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1]};
    size_t global_work_size_v[2] = {(nx + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0], (ny + 1 + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1]};

    for (n = 0; n < f.Nt; ++n) {
        clEnqueueNDRangeKernel(command_queue, kernel_p, 2, NULL, global_work_size_p, local_work_size, 0, NULL, NULL);
        clEnqueueNDRangeKernel(command_queue, kernel_u, 2, NULL, global_work_size_u, local_work_size, 0, NULL, NULL);
        clEnqueueNDRangeKernel(command_queue, kernel_v, 2, NULL, global_work_size_v, local_work_size, 0, NULL, NULL);
    }
    clFinish(command_queue);
    toc = py_gettime();
    printf("Elapsed: %f seconds\n", toc - tic);

    /* 7. COPIAR RESULTADOS Y LIMPIEZA */
    clEnqueueReadBuffer(command_queue, p_mem_obj, CL_TRUE, 0, p_size, f.p.value, 0, NULL, NULL);
    if (write) py_write_to_disk(f.p, outfile);

    clFlush(command_queue);
    clFinish(command_queue);
    clReleaseKernel(kernel_p);
    clReleaseKernel(kernel_u);
    clReleaseKernel(kernel_v);
    clReleaseProgram(program);
    clReleaseMemObject(p_mem_obj);
    clReleaseMemObject(u_mem_obj);
    clReleaseMemObject(v_mem_obj);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    py_free_acoustic_field(f);

    return EXIT_SUCCESS;
}