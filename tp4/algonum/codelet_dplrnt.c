/**
 *
 * @file codelet_dplrnt.c
 *
 * @copyright 2019-2019 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief dplrnt StarPU codelet
 *
 * @version 0.1.0
 * @author Mathieu Faverge
 * @date 2019-12-01
 *
 */
#include "algonum.h"
#include "codelets.h"

/**
 * @brief Structure to gather static parameters of the kernel
 */
typedef struct cl_dplrnt_arg_s {
    double bump;
    int m;
    int n;
    int lda;
    int bigM;
    int m0;
    int n0;
    unsigned long long int seed;
} cl_dplrnt_arg_t;

/**
 * @brief Codelet CPU function
 */
static void
cl_dplrnt_cpu_func( void *descr[], void *cl_arg )
{
    cl_dplrnt_arg_t args;
    double *A;

    A = tile_interface_get(descr[0]);

    starpu_codelet_unpack_args( cl_arg, &args );

    CORE_dplrnt( args.bump, args.m, args.n, A, args.lda,
                 args.bigM, args.m0, args.n0, args.seed );
}

/**
 * @brief Define the StarPU codelet structure
 */
struct starpu_codelet cl_dplrnt = {
    .where      = STARPU_CPU,
    .cpu_func   = cl_dplrnt_cpu_func,
    .cuda_flags = { 0 },
    .cuda_func  = NULL,
    .nbuffers   = 1,
    .name       = "plrnt"
};

/**
 * @brief Insert task funtion
 */
void
insert_dplrnt( double               bump,
               int                  m,
               int                  n,
               starpu_data_handle_t A,
               int                  lda,
               int                  bigM,
               int                  m0,
               int                  n0,
               unsigned long long int seed )
{
    cl_dplrnt_arg_t args = {
        .bump = bump,
        .m    = m,
        .n    = n,
        .lda  = lda,
        .bigM = bigM,
        .m0   = m0,
        .n0   = n0,
        .seed = seed,
    };

    starpu_insert_task(
        starpu_mpi_codelet(&cl_dplrnt),
        STARPU_VALUE, &args, sizeof(cl_dplrnt_arg_t),
        STARPU_W,      A,
        0);
}
