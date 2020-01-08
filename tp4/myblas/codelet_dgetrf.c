/**
 *
 * @file codelet_dgetrf.c
 *
 * @copyright 2019-2019 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief dgetrf StarPU codelet
 *
 * @version 0.1.0
 * @author Mathieu Faverge
 * @date 2019-12-01
 *
 */
#include "codelets.h"
#include <assert.h>

/**
 * @brief Structure to gather static parameters of the kernel
 */
typedef struct cl_dgetrf_arg_s {
    int m;
    int n;
    int lda;
} cl_dgetrf_arg_t;

/**
 * @brief Codelet CPU function
 */
static void
cl_dgetrf_cpu_func( void *descr[], void *cl_arg )
{
    cl_dgetrf_arg_t args;
    double *A;
    int minmn;
    A = tile_interface_get(descr[0]);

    starpu_codelet_unpack_args( cl_arg, &args );
    minmn = (args.m < args.n ) ? args.m : args.n;

    {
        int ipiv[ minmn ];
        int rc = 0;

        rc = LAPACKE_dgetrf_work( LAPACK_COL_MAJOR, args.m, args.n,
                                  A, args.lda, ipiv );
        assert( rc == 0 );
#if !defined(NDEBUG)
        for( rc = 0; rc < minmn; rc ++ ) {
            assert( ipiv[rc] == rc + 1 );
        }
#endif
    }
}

/**
 * @brief Define the StarPU codelet structure
 */
struct starpu_codelet cl_dgetrf = {
    .where      = STARPU_CPU,
    .cpu_func   = cl_dgetrf_cpu_func,
    .cuda_flags = { 0 },
    .cuda_func  = NULL,
    .nbuffers   = 2,
    .name       = "getrf"
};

/**
 * @brief Insert task funtion
 */
void
insert_dgetrf( int                  m,
               int                  n,
               starpu_data_handle_t A,
               int                  lda )
{
    cl_dgetrf_arg_t args = {
        .m       = m,
        .n       = n,
        .lda     = lda,
    };

    starpu_insert_task(
        starpu_mpi_codelet(&cl_dgetrf),
        STARPU_VALUE, &args, sizeof(cl_dgetrf_arg_t),
        STARPU_RW,    A,
        0);
}
