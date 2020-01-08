#include "algonum.h"
#include "codelets.h"

void
tile2lapack_starpu( int M, int N, int b,
                    const double **Atile,
                    double *Alapack, int lda )
{
    starpu_data_handle_t *handlesAt;
    starpu_data_handle_t *handlesAl;
    starpu_data_handle_t hAt, hAl;

    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int m, n;

    handlesAt = calloc( MT * NT, sizeof(starpu_data_handle_t) );
    handlesAl = calloc( MT * NT, sizeof(starpu_data_handle_t) );

    assert( lda >= M );

    /* Now, let's copy the tile one by one, in column major order */
    for( n=0; n<NT; n++) {
        for( m=0; m<MT; m++) {
            int mm = m == (MT-1) ? M - m * b : b;
            int nn = n == (NT-1) ? N - n * b : b;

            hAt = get_starpu_handle( 0, handlesAt, (double**)Atile, m, n, b, MT );
            hAl = get_starpu_handle_lap( 1, handlesAl + n * MT + m,
                                         m, n, mm, nn,
                                         Alapack + n * lda + m, lda, MT );

            /* Let's use LAPACKE to ease the copy */
            insert_dlacpy( mm, nn, hAt, b, hAl, lda );
        }
    }

    unregister_starpu_handle( MT * NT, handlesAt );
    unregister_starpu_handle( MT * NT, handlesAl );

    /* Let's wait for the end of all the tasks */
    starpu_task_wait_for_all();
#if defined(ENABLE_MPI)
    starpu_mpi_barrier(MPI_COMM_WORLD);
#endif

    free( handlesAt );
    free( handlesAl );
}
