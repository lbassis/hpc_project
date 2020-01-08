/**
 *
 * @file codelet_dlacpy.c
 *
 * @copyright 2019-2019 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief dlacpy StarPU codelet
 *
 * @version 0.1.0
 * @author Mathieu Faverge
 * @date 2019-12-01
 *
 */
#include "codelets.h"

/**
 * @brief Structure to gather static parameters of the kernel
 */
typedef struct cl_dlacpy_arg_s {
    int             m;
    int             n;
    int             lda;
    int             ldb;
} cl_dlacpy_arg_t;

/**
 * @brief Codelet CPU function
 */
static void
cl_dlacpy_cpu_func( void *descr[], void *cl_arg )
{
    cl_dlacpy_arg_t args;
    double *A;
    double *B;

    A = tile_interface_get(descr[0]);
    B = tile_interface_get(descr[1]);

    starpu_codelet_unpack_args( cl_arg, &args );

    LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A',
                         args.m, args.n, A, args.lda, B, args.ldb );
}

/**
 * @brief Define the StarPU codelet structure
 */
struct starpu_codelet cl_dlacpy = {
    .where      = STARPU_CPU,
    .cpu_func   = cl_dlacpy_cpu_func,
    .cuda_flags = { 0 },
    .cuda_func  = NULL,
    .nbuffers   = 2,
    .name       = "lacpy"
};

/**
 * @brief Insert task funtion
 */
void
insert_dlacpy( int                  m,
               int                  n,
               starpu_data_handle_t A,
               int                  lda,
               starpu_data_handle_t B,
               int                  ldb )
{
    cl_dlacpy_arg_t args = {
        .m      = m,
        .n      = n,
        .lda    = lda,
        .ldb    = ldb,
    };

    starpu_insert_task(
        starpu_mpi_codelet(&cl_dlacpy),
        STARPU_VALUE, &args, sizeof(cl_dlacpy_arg_t),
        STARPU_R,      A,
        STARPU_W,      B,
        0);
}
