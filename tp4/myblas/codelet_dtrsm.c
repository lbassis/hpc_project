/**
 *
 * @file codelet_dtrsm.c
 *
 * @copyright 2019-2019 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief dtrsm StarPU codelet
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
typedef struct cl_dtrsm_arg_s {
    CBLAS_SIDE      side;
    CBLAS_UPLO      uplo;
    CBLAS_TRANSPOSE transA;
    CBLAS_DIAG      diag;
    int             m;
    int             n;
    double          alpha;
    int             lda;
    int             ldb;
} cl_dtrsm_arg_t;

/**
 * @brief Codelet CPU function
 */
static void
cl_dtrsm_cpu_func( void *descr[], void *cl_arg )
{
    cl_dtrsm_arg_t args;
    double *A;
    double *B;

    A = tile_interface_get(descr[0]);
    B = tile_interface_get(descr[1]);

    starpu_codelet_unpack_args( cl_arg, &args );

    cblas_dtrsm( CblasColMajor, args.side, args.uplo, args.transA, args.diag,
                 args.m, args.n, args.alpha, A, args.lda, B, args.ldb );
}

/**
 * @brief Codelet CUDA function
 */
#if defined(ENABLE_CUDA)
static void cl_dtrsm_cuda_func(void *descr[], void *cl_arg)
{
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cl_dtrsm_arg_t args;
    double *A;
    double *B;

    A = tile_interface_get(descr[0]);
    B = tile_interface_get(descr[1]);

    starpu_codelet_unpack_args( cl_arg, &args );

    cublasDtrsm( handle,
                 get_cublas_side( args.side ),
                 get_cublas_uplo( args.uplo ),
                 get_cublas_trans( args.transA ),
                 get_cublas_diag( args.diag ),
                 args.m, args.n, &(args.alpha), A, args.lda, B, args.ldb );

#if !defined(STARPU_CUDA_ASYNC)
    cudaStreamSynchronize( stream );
#endif

    return;
}
#else
static void cl_dtrsm_cuda_func(void *descr[], void *cl_arg)
{
    (void)descr;
    (void)cl_arg;
    return;
}
#endif /* ENABLE_CUDA */

/**
 * @brief Define the StarPU codelet structure
 */
struct starpu_codelet cl_dtrsm = {
#if defined(ENABLE_CUDA)
    .where      = STARPU_CUDA | STARPU_CPU,
#else
    .where      = STARPU_CPU,
#endif
    .cpu_func   = cl_dtrsm_cpu_func,
    .cuda_flags = { STARPU_CUDA_ASYNC },
    .cuda_func  = cl_dtrsm_cuda_func,
    .nbuffers   = 2,
    .name       = "trsm"
};

/**
 * @brief Insert task funtion
 */
void
insert_dtrsm( CBLAS_SIDE           side,
              CBLAS_UPLO           uplo,
              CBLAS_TRANSPOSE      transA,
              CBLAS_DIAG           diag,
              int                  m,
              int                  n,
              double               alpha,
              starpu_data_handle_t A,
              int                  lda,
              starpu_data_handle_t B,
              int                  ldb )
{
    cl_dtrsm_arg_t args = {
        .side   = side,
        .uplo   = uplo,
        .transA = transA,
        .diag   = diag,
        .m      = m,
        .n      = n,
        .alpha  = alpha,
        .lda    = lda,
        .ldb    = ldb,
    };

    starpu_insert_task(
        starpu_mpi_codelet(&cl_dtrsm),
        STARPU_VALUE, &args, sizeof(cl_dtrsm_arg_t),
        STARPU_R,     A,
        STARPU_RW,     B,
        0);
}
