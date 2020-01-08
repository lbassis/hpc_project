#include "algonum_int.h"

int
check_dgetrf( int M, int N,
              double *LU, double *A, int lda )
{
    int info_solution;
    double Anorm, Rnorm, result;
    double eps = LAPACKE_dlamch_work('e');
    double *L, *U;
    int ldl, ldu;
    int minMN = ( M < N ) ? M : N;

    ldl = max( 1, lda );
    ldu = max( 1, minMN );

    /* Calculates the dimensions according to the transposition */
    L = LU;
    U = malloc( minMN * N * sizeof(double) );

    /* Set before copying to get the non unit diagonal */
    LAPACKE_dlaset_work( LAPACK_COL_MAJOR, 'A', minMN, N, 0., 1.,  U, ldu );
    LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'U', minMN, N, LU, lda, U, ldu );

    /* Copy L before setting to force the diagonal to 1. */
    LAPACKE_dlaset_work( LAPACK_COL_MAJOR, 'U', M, minMN, 0., 1., L, ldl );

    Anorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'O', M, N, A, lda, NULL );

    /* Makes the multiplication with the core function */
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, minMN,
                 -1, L, ldl, U, ldu, 1., A, lda );

    /* Calculates the norm with the core function's result */
    Rnorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'M', M, N, A, lda, NULL );

    result = (Anorm * minMN * eps);
    if ( result > 0. ) {
        result = Rnorm / result;
    }

    /* Verifies if the result is inside a threshold */
    if (  isnan(Rnorm) || isinf(Rnorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
        /* fprintf(stderr, "M= %4d, N= %4d, Anorm= %le, Rnorm= %le, result= %le\n", */
        /*         M, N, Anorm, Rnorm, result ); */
        info_solution = 1;
    }
    else {
        info_solution = 0;
    }

    free( U );

    return info_solution;
}

int
testone_dgetrf( dgetrf_fct_t dgetrf,
                int M, int N, int check )
{
    int     hres = 0;
    double *A;
    int     lda;
    int     seedA = random();
    perf_t  start, stop;
    int minMN = ( M < N ) ? M : N;

    double gflops;
    double flops = flops_dgetrf( M, N );

    /* Create the matrices */
    lda = max( M, 1 );
    A = malloc( lda * N * sizeof(double) );

    /* Fill the matrices with random values */
    CORE_dplrnt( minMN, M, N, A, lda, M, 0, 0, seedA );

    /* Calculate the product */
    perf( &start );
    dgetrf( CblasColMajor, M, N, A, lda );
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
        double *Ainit = malloc( lda * N  * sizeof(double) );
        CORE_dplrnt( minMN, M, N, Ainit, lda, M, 0, 0, seedA );

        hres += check_dgetrf( M, N, A, Ainit, lda );

        free( Ainit );
    }
    else {
        printf( "M= %4d N= %4d : %le GFlop/s\n",
                M, N, gflops );
    }

    free( A );

    return hres;
}

int
testall_dgetrf( dgetrf_fct_t tested_dgetrf )
{
    int all_M[] = { 0, 5, 8, 17, 57 };
    int all_N[] = { 0, 3, 5, 17, 64 };

    int nb_M = sizeof( all_M ) / sizeof( int );
    int nb_N = sizeof( all_N ) / sizeof( int );

    int im, in, m, n;
    int nbfailed = 0;
    int nbpassed = 0;
    int nbtests = nb_M * nb_N;

    for( im = 0; im < nb_M; im ++ ) {
        m = all_M[im];
        for( in = 0; in < nb_N; in ++ ) {
            n = all_N[in];

            nbfailed += testone_dgetrf( tested_dgetrf, m, n, 1 );
            nbpassed++;
            fprintf( stdout, "\r %4d / %4d", nbpassed, nbtests );
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
testone_dgetrf_tiled( dplrnt_tiled_fct_t dplrnt,
                      dgetrf_tiled_fct_t dgetrf,
                      int M, int N, int b, int check )
{
    int      hres = 0;
    double **Atile;
    int      lda;
    int      seedA = random();
    perf_t   start, stop;
    int minMN = ( M < N ) ? M : N;

    double gflops;
    double flops = flops_dgetrf( M, N );

    /* Create the matrices */
    lda = max( M, 1 );

    /* Fill the matrices with random values */
    Atile = lapack2tile( M, N, b, NULL, lda );
    dplrnt( minMN, M, N, b, Atile, seedA );

    /* Calculate the product */
    perf( &start );
    dgetrf( CblasColMajor, M, N, b, Atile );
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
        double *A     = malloc( lda * N * sizeof(double) );
        double *Ainit = malloc( lda * N * sizeof(double) );
        CORE_dplrnt( minMN, M, N, Ainit, lda, M, 0, 0, seedA );

        tile2lapack( M, N, b, (const double **)Atile, A, lda );

        hres += check_dgetrf( M, N, A, Ainit, lda );

        free( A );
        free( Ainit );
    }
    else {
        printf( "M= %4d N= %4d : %le GFlop/s\n",
                M, N, gflops );
    }

    tileFree( M, N, b, Atile );

    return hres;
}

int
testall_dgetrf_tiled( dplrnt_tiled_fct_t dplrnt,
                      dgetrf_tiled_fct_t tested_dgetrf )
{
    int all_M[] = { 0, 5, 8, 17, 57 };
    int all_N[] = { 0, 3, 5, 17, 64 };
    int all_b[] = { 1, 3, 7, 16 };

    int nb_M = sizeof( all_M ) / sizeof( int );
    int nb_N = sizeof( all_N ) / sizeof( int );
    int nb_b = sizeof( all_b ) / sizeof( int );

    int im, in, ib, m, n, b;
    int nbfailed = 0;
    int nbpassed = 0;
    int nbtests = nb_M * nb_N * nb_b;

    for( im = 0; im < nb_M; im ++ ) {
        m = all_M[im];
        for( in = 0; in < nb_N; in ++ ) {
            n = all_N[in];
            for( ib = 0; ib < nb_b; ib ++ ) {
                b = all_b[ib];

                nbfailed += testone_dgetrf_tiled( dplrnt, tested_dgetrf, m, n, b, 1 );
                nbpassed++;
                fprintf( stdout, "\r %4d / %4d", nbpassed, nbtests );
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
