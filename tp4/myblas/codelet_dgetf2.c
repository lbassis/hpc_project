/**
 *
 * @file codelet_dgetf2.c
 *
 * @copyright 2019-2019 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief dgetf2 StarPU codelet
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
typedef struct cl_dgetf2_arg_s {
    int             m;
    int             n;
    int             lda;
} cl_dgetf2_arg_t;

/**
 * @brief Codelet CPU function
 */
static void
cl_dgetf2_cpu_func( void *descr[], void *cl_arg )
{
    cl_dgetf2_arg_t args;
    double *A;

    A = tile_interface_get(descr[0]);

    starpu_codelet_unpack_args( cl_arg, &args );

    LAPACKE_dgetf2(CblasColMajor, args.m, args.n, A, args.lda, NULL);
}


/**
 * @brief Define the StarPU codelet structure
 */
struct starpu_codelet cl_dgetf2 = {
    .where      = STARPU_CPU,
    .cpu_func   = cl_dgetf2_cpu_func,
    .nbuffers   = 2,
    .name       = "getf2"
};

/**
 * @brief Insert task funtion
 */
void
insert_dgetf2(int                  m,
              int                  n,
              starpu_data_handle_t A,
              int                  lda)
{
    cl_dgetf2_arg_t args = {
        .m      = m,
        .n      = n,
        .lda    = lda,
    };

    starpu_insert_task(
        starpu_mpi_codelet(&cl_dgetf2),
        STARPU_VALUE, &args, sizeof(cl_dgetf2_arg_t),
        STARPU_RW,     A,
        0);
}
