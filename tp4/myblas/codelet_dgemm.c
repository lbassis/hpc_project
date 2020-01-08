/**
 *
 * @file codelet_dgemm.c
 *
 * @copyright 2019-2019 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief dgemm StarPU codelet
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
typedef struct cl_dgemm_arg_s {
    CBLAS_TRANSPOSE transA;
    CBLAS_TRANSPOSE transB;
    int             m;
    int             n;
    int             k;
    double          alpha;
    int             lda;
    int             ldb;
    double          beta;
    int             ldc;
} cl_dgemm_arg_t;

/**
 * @brief Codelet CPU function
 */
static void
cl_dgemm_cpu_func( void *descr[], void *cl_arg )
{
    cl_dgemm_arg_t args;
    double *A;
    double *B;
    double *C;

    A = tile_interface_get(descr[0]);
    B = tile_interface_get(descr[1]);
    C = tile_interface_get(descr[2]);

    starpu_codelet_unpack_args( cl_arg, &args );

    cblas_dgemm( CblasColMajor, args.transA, args.transB,
                 args.m, args.n, args.k, args.alpha, A, args.lda, B, args.ldb,
                 args.beta, C, args.ldc );
}

/**
 * @brief Codelet CUDA function
 */
#if defined(ENABLE_CUDA)
static void cl_dgemm_cuda_func(void *descr[], void *cl_arg)
{
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cl_dgemm_arg_t args;
    double *A;
    double *B;
    double *C;

    A = tile_interface_get(descr[0]);
    B = tile_interface_get(descr[1]);
    C = tile_interface_get(descr[2]);

    starpu_codelet_unpack_args( cl_arg, &args );

    cublasDgemm( handle,
                 get_cublas_trans( args.transA ), get_cublas_trans( args.transB ),
                 args.m, args.n, args.k, args.alpha, A, args.lda, B, args.ldb,
                 args.beta, C, args.ldc );
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
static void cl_dgemm_cuda_func(void *descr[], void *cl_arg)
{
    (void)descr;
    (void)cl_arg;
    return;
}
#endif /* ENABLE_CUDA */

/**
 * @brief Define the StarPU codelet structure
 */
struct starpu_codelet cl_dgemm = {
#if defined(ENABLE_CUDA)
    .where      = STARPU_CUDA | STARPU_CPU,
#else
    .where      = STARPU_CPU,
#endif
    .cpu_func   = cl_dgemm_cpu_func,
    .cuda_flags = { STARPU_CUDA_ASYNC },
    .cuda_func  = cl_dgemm_cuda_func,
    .nbuffers   = 3,
    .name       = "gemm",
};

/**
 * @brief Insert task funtion
 */
void
insert_dgemm( CBLAS_TRANSPOSE      transA,
              CBLAS_TRANSPOSE      transB,
              int                  m,
              int                  n,
              int                  k,
              double               alpha,
              starpu_data_handle_t A,
              int                  lda,
              starpu_data_handle_t B,
              int                  ldb,
              double               beta,
              starpu_data_handle_t C,
              int                  ldc )
{
    cl_dgemm_arg_t args = {
        .transA  = transA,
        .transB  = transB,
        .m       = m,
        .n       = n,
        .k       = k,
        .alpha   = alpha,
        .lda     = lda,
        .ldb     = ldb,
        .beta   = beta,
        .ldc     = ldc,
    };

    starpu_insert_task(
        starpu_mpi_codelet(&cl_dgemm),
        STARPU_VALUE, &args, sizeof(cl_dgemm_arg_t),
        STARPU_R,    A,
        STARPU_R,    B,
        STARPU_RW,    C,
        0);
}
