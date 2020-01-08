#include "algonum_int.h"

int
check_dgemm( CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
             int M, int N, int K,
             double alpha, const double *A, int lda, const double *B, int ldb,
             double beta, double *Cref, const double *C, int ldc )
{
    int info_solution;
    double Anorm, Bnorm, Crefnorm, Rnorm, result;
    double eps = LAPACKE_dlamch_work('e');
    double *work = malloc( max(M, max(N, K) ) * sizeof(double) );

    /* Calculates the dimensions according to the transposition */
    if ( transA == CblasNoTrans ) {
        Anorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'I', M, K, A, lda, work );
    } else {
        Anorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'O', K, M, A, lda, work );
    }
    if ( transB == CblasNoTrans ) {
        Bnorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'O', K, N, B, ldb, work );
    } else {
        Bnorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'I', N, K, B, ldb, work );
    }

    /* Computes the norms for comparing */
    Crefnorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'M', M, N, Cref, ldc, NULL );

    /* Makes the multiplication with the core function */
    cblas_dgemm( CblasColMajor, transA, transB, M, N, K,
                 alpha, A, lda, B, ldb, beta, Cref, ldc );
    cblas_daxpy( ldc * N, -1., C, 1, Cref, 1 );

    /* Calculates the norm with the core function's result */
    Rnorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'M', M, N, Cref, ldc, NULL );

    result = ((abs(alpha) * max(Anorm, Bnorm) + abs(beta) * Crefnorm) * K * eps);
    if ( result > 0. ) {
        result = Rnorm / result;
    }

    /* Verifies if the result is inside a threshold */
    if (  isnan(Rnorm) || isinf(Rnorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
        info_solution = 1;
    }
    else {
        info_solution = 0;
    }

    free(work);

    return info_solution;
}

int
testone_dgemm( dgemm_fct_t dgemm,
               CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB,
               int M, int N, int K,
               int check )
{
    int     Am, An, Bm, Bn;
    int     hres = 0;
    double *A, *B, *C;
    int     lda, ldb, ldc;
    double  alpha, beta;
    int     seedA = random();
    int     seedB = random();
    int     seedC = random();
    perf_t  start, stop;

    double gflops;
    double flops = flops_dgemm( M, N, K );

    if ( transA == CblasNoTrans ) {
        Am = M;
        An = K;
    }
    else {
        Am = K;
        An = M;
    }
    lda = max( Am, 1 );

    if ( transB == CblasNoTrans ) {
        Bm = K;
        Bn = N;
    }
    else {
        Bm = N;
        Bn = K;
    }
    ldb = max( Bm, 1 );
    ldc = max( M, 1 );

    /* Initialize alpha and beta */
    CORE_dplrnt( 0, 1, 1, &alpha, lda, 1, 0, 0, random() );
    CORE_dplrnt( 0, 1, 1, &beta,  ldb, 1, 0, 0, random() );

    /* Create the matrices */
    A = malloc( lda * An * sizeof(double) );
    B = malloc( ldb * Bn * sizeof(double) );
    C = malloc( ldc * N  * sizeof(double) );

    /* Fill the matrices with random values */
    CORE_dplrnt( 0, Am, An, A, lda, Am, 0, 0, seedA );
    CORE_dplrnt( 0, Bm, Bn, B, ldb, An, 0, 0, seedB );
    CORE_dplrnt( 0, M,  N,  C, ldc, M,  0, 0, seedC );

    /* Calculate the product */
    perf( &start );
    dgemm( CblasColMajor, transA, transB, M, N, K,
           alpha, A, lda, B, ldb,
           beta, C, ldc );
    perf( &stop );

    perf_diff( &start, &stop );
    if ( flops > 0. ) {
        gflops = perf_gflops( &stop, flops );
    }
    else {
        gflops = 0.;
    }

    /* Check the solution */
    if ( check ) {
        double *Cinit = malloc( ldc * N  * sizeof(double) );
        CORE_dplrnt( 0, M, N, Cinit, ldc, M, 0, 0, seedC );

        hres += check_dgemm( transA, transB, M, N, K,
                             alpha, A, lda, B, ldb,
                             beta, Cinit, C, ldc );

        if ( hres ) {
            fprintf( stderr,
                     "tA=%s tB=%s M= %4d N= %4d K= %4d: FAILED\n",
                     (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                     (transB == CblasNoTrans) ? "NoTrans" : "Trans",
                     M, N, K );
        }
        free( Cinit );
    }
    else {
        printf( "tA=%s tB=%s M= %4d N= %4d K= %4d: %le GFlop/s\n",
                (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                (transB == CblasNoTrans) ? "NoTrans" : "Trans",
                M, N, K, gflops );
    }

    free( A );
    free( B );
    free( C );

    return hres;
}

int
testall_dgemm( dgemm_fct_t tested_dgemm )
{
    int all_M[] = { 0, 5, 8, 17, 57 };
    int all_N[] = { 0, 3, 5, 17, 64 };
    int all_K[] = { 0, 1, 8, 24, 47 };

    int nb_M = sizeof( all_M ) / sizeof( int );
    int nb_N = sizeof( all_N ) / sizeof( int );
    int nb_K = sizeof( all_K ) / sizeof( int );

    int im, in, ik, m, n, k;
    int nbfailed = 0;
    int nbpassed = 0;
    int nbtests = 4 * nb_M * nb_N * nb_K;
    CBLAS_TRANSPOSE tA, tB;

    for ( tA = CblasNoTrans; tA <= CblasTrans; tA ++ ) {
        for ( tB = CblasNoTrans; tB <= CblasTrans; tB ++ ) {
            for( im = 0; im < nb_M; im ++ ) {
                m = all_M[im];
                for( in = 0; in < nb_N; in ++ ) {
                    n = all_N[in];
                    for( ik = 0; ik < nb_K; ik ++ ) {
                        k = all_K[ik];

                        nbfailed += testone_dgemm( tested_dgemm, tA, tB, m, n, k, 1 );
                        nbpassed++;
                        fprintf( stdout, "\r %4d / %4d", nbpassed, nbtests );
                    }
                }
            }
        }
    }

    if ( nbfailed > 0 ) {
        fprintf( stdout, "\n %4d tests failed out of %d\n",
                 nbfailed, nbtests );
    }
    else {
        fprintf( stdout, "\n Congratulations all %4d tests succeeded\n",
                 nbtests );
    }
    return nbfailed;
}

int
testone_dgemm_tiled( dplrnt_tiled_fct_t dplrnt,
                     dgemm_tiled_fct_t  dgemm,
                     CBLAS_TRANSPOSE transA,
                     CBLAS_TRANSPOSE transB,
                     int M, int N, int K, int b,
                     int check )
{
    int     Am, An, Bm, Bn;
    int     hres = 0;
    double **Atile, **Btile, **Ctile;
    int     lda, ldb, ldc;
    double  alpha, beta;
    int     seedA = random();
    int     seedB = random();
    int     seedC = random();
    perf_t  start, stop;

    double gflops;
    double flops = flops_dgemm( M, N, K );

    if ( transA == CblasNoTrans ) {
        Am = M;
        An = K;
    }
    else {
        Am = K;
        An = M;
    }
    lda = max( Am, 1 );

    if ( transB == CblasNoTrans ) {
        Bm = K;
        Bn = N;
    }
    else {
        Bm = N;
        Bn = K;
    }
    ldb = max( Bm, 1 );
    ldc = max( M, 1 );

    /* Initialize alpha and beta */
    CORE_dplrnt( 0, 1, 1, &alpha, lda, 1, 0, 0, random() );
    CORE_dplrnt( 0, 1, 1, &beta,  ldb, 1, 0, 0, random() );

    Atile = lapack2tile( Am, An, b, NULL, lda );
    Btile = lapack2tile( Bm, Bn, b, NULL, ldb );
    Ctile = lapack2tile( M,  N,  b, NULL, ldc );

    dplrnt( 0., Am, An, b, Atile, seedA );
    dplrnt( 0., Bm, Bn, b, Btile, seedB );
    dplrnt( 0., M,  N,  b, Ctile, seedC );

    /* Calculate the product */
    perf( &start );
    dgemm( CblasColMajor, transA, transB, M, N, K, b,
           alpha, (const double **)Atile, (const double **)Btile, beta, Ctile );
    perf( &stop );

    perf_diff( &start, &stop );
    if ( flops > 0. ) {
        gflops = perf_gflops( &stop, flops );
    }
    else {
        gflops = 0.;
    }

    /* Check the solution */
    if ( check ) {
        double *A, *B, *C, *Cinit;

        /* Create the matrices for the test */
        A = malloc( lda * An * sizeof(double) );
        B = malloc( ldb * Bn * sizeof(double) );
        C = malloc( ldc * N  * sizeof(double) );

        /* Fill the matrices with the same random values */
        tile2lapack( Am, An, b, (const double **)Atile, A, lda );
        tile2lapack( Bm, Bn, b, (const double **)Btile, B, ldb );
        tile2lapack( M,  N,  b, (const double **)Ctile, C, ldc );

        /* Create the original C to compare with */
        Cinit = malloc( ldc * N  * sizeof(double) );
        CORE_dplrnt( 0, M, N, Cinit, ldc, M, 0, 0, seedC );

        hres += check_dgemm( transA, transB, M, N, K,
                             alpha, A, lda, B, ldb,
                             beta, Cinit, C, ldc );

        if ( hres ) {
            fprintf( stderr,
                     "tA=%s tB=%s M= %4d N= %4d K= %4d b=%3d: FAILED\n",
                     (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                     (transB == CblasNoTrans) ? "NoTrans" : "Trans",
                     M, N, K, b );
        }
        free( A );
        free( B );
        free( C );
        free( Cinit );
    }
    else {
        printf( "tA=%s tB=%s M= %4d N= %4d K= %4d: %le GFlop/s\n",
                (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                (transB == CblasNoTrans) ? "NoTrans" : "Trans",
                M, N, K, gflops );
    }

    tileFree( Am, An, b, Atile );
    tileFree( Bm, Bn, b, Btile );
    tileFree( M,  N,  b, Ctile );

    return hres;
}

int
testall_dgemm_tiled( dplrnt_tiled_fct_t dplrnt,
                     dgemm_tiled_fct_t tested_dgemm )
{
    int all_M[] = { 0, 5, 8, 17, 57 };
    int all_N[] = { 0, 3, 5, 17, 64 };
    int all_K[] = { 0, 1, 8, 24, 47 };
    int all_b[] = { 1, 3, 16 };

    int nb_M = sizeof( all_M ) / sizeof( int );
    int nb_N = sizeof( all_N ) / sizeof( int );
    int nb_K = sizeof( all_K ) / sizeof( int );
    int nb_b = sizeof( all_b ) / sizeof( int );

    int im, in, ik, ib, m, n, k, b;
    int nbfailed = 0;
    int nbpassed = 0;
    int nbtests = 4 * nb_M * nb_N * nb_K * nb_b;
    CBLAS_TRANSPOSE tA, tB;

    for ( tA = CblasNoTrans; tA <= CblasTrans; tA ++ ) {
        for ( tB = CblasNoTrans; tB <= CblasTrans; tB ++ ) {
            for( im = 0; im < nb_M; im ++ ) {
                m = all_M[im];
                for( in = 0; in < nb_N; in ++ ) {
                    n = all_N[in];
                    for( ik = 0; ik < nb_K; ik ++ ) {
                        k = all_K[ik];
                        for( ib = 0; ib < nb_b; ib ++ ) {
                            b = all_b[ib];

                            nbfailed += testone_dgemm_tiled( dplrnt, tested_dgemm, tA, tB, m, n, k, b, 1 );
                            nbpassed++;
                            fprintf( stdout, "\r %4d / %4d", nbpassed, nbtests );
                        }
                    }
                }
            }
        }
    }

    if ( nbfailed > 0 ) {
        fprintf( stdout, "\n %4d tests failed out of %d\n",
                 nbfailed, nbtests );
    }
    else {
        fprintf( stdout, "\n Congratulations all %4d tests succeeded\n",
                 nbtests );
    }
    return nbfailed;
}
