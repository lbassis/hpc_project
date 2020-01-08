#include "algonum.h"
#include "codelets.h"

void
my_dgemm_tiled_starpu( CBLAS_LAYOUT layout,
                       CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                       int M, int N, int K, int b,
                       double alpha, const double **A,
                                     const double **B,
                       double beta,        double **C )
{
    starpu_data_handle_t *handlesA;
    starpu_data_handle_t *handlesB;
    starpu_data_handle_t *handlesC;
    starpu_data_handle_t hA, hB, hC;

    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int KT = my_iceil( K, b );
    int m, n, k;

    handlesA = calloc( MT * KT, sizeof(starpu_data_handle_t) );
    handlesB = calloc( KT * NT, sizeof(starpu_data_handle_t) );
    handlesC = calloc( MT * NT, sizeof(starpu_data_handle_t) );

    for( n=0; n<NT; n++) {
        int nn = n == (NT-1) ? N - n * b : b;

        for( m=0; m<MT; m++) {
            int mm = m == (MT-1) ? M - m * b : b;

            hC = get_starpu_handle( 2, handlesC, C, m, n, b, MT );

            if ( transA == CblasNoTrans ) {
                if ( transB == CblasNoTrans ) {

                    /* A: CblasNoTrans / B: CblasNoTrans */
                    for( k=0; k<KT; k++) {
                        int kk = k == (KT-1) ? K - k * b : b;
                        double lbeta = (k == 0) ? beta : 1.;

                        hA = get_starpu_handle( 0, handlesA, (double **)A, m, k, b, MT );
                        hB = get_starpu_handle( 1, handlesB, (double **)B, k, n, b, KT );

                        insert_dgemm( transA, transB, mm, nn, kk,
                                      alpha, hA, b, hB, b, lbeta, hC, b );
                    }
                }
                else {

                    /* A: Cblas[Conj]Trans / B: CblasNoTrans */
                    for( k=0; k<KT; k++) {
                        int kk = k == (KT-1) ? K - k * b : b;
                        double lbeta = (k == 0) ? beta : 1.;

                        hA = get_starpu_handle( 0, handlesA, (double **)A, k, m, b, KT );
                        hB = get_starpu_handle( 1, handlesB, (double **)B, k, n, b, KT );

                        insert_dgemm( transA, transB, mm, nn, kk,
                                      alpha, hA, b, hB, b, lbeta, hC, b );
                    }
                }
            }
            else {
                if ( transB == CblasNoTrans ) {

                    /* A: CblasNoTrans / B: Cblas[Conj]Trans */
                    for( k=0; k<KT; k++) {
                        int kk = k == (KT-1) ? K - k * b : b;
                        double lbeta = (k == 0) ? beta : 1.;

                        hA = get_starpu_handle( 0, handlesA, (double **)A, m, k, b, MT );
                        hB = get_starpu_handle( 1, handlesB, (double **)B, n, k, b, NT );

                        insert_dgemm( transA, transB, mm, nn, kk,
                                      alpha, hA, b, hB, b, lbeta, hC, b );
                    }
                }
                else {

                    /* A: Cblas[Conj]Trans / B: Cblas[Conj]Trans */
                    for( k=0; k<KT; k++) {
                        int kk = k == (KT-1) ? K - k * b : b;
                        double lbeta = (k == 0) ? beta : 1.;

                        hA = get_starpu_handle( 0, handlesA, (double **)A, k, m, b, KT );
                        hB = get_starpu_handle( 1, handlesB, (double **)B, n, k, b, NT );

                        insert_dgemm( transA, transB, mm, nn, kk,
                                      alpha, hA, b, hB, b, lbeta, hC, b );
                    }
                }
            }
        }
    }

    unregister_starpu_handle( MT * KT, handlesA );
    unregister_starpu_handle( KT * NT, handlesB );
    unregister_starpu_handle( MT * NT, handlesC );

    /* Let's wait for the end of all the tasks */
    starpu_task_wait_for_all();
#if defined(ENABLE_MPI)
    starpu_mpi_barrier(MPI_COMM_WORLD);
#endif

    free( handlesA );
    free( handlesB );
    free( handlesC );
}
