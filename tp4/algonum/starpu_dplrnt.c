#include "algonum.h"
#include "codelets.h"

void
dplrnt_tiled_starpu( double bump, int M, int N, int b,
                     double **A, unsigned long long int seed )
{
    starpu_data_handle_t *handlesA;
    starpu_data_handle_t hA;

    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int m, n, mm, nn;

    handlesA = calloc( MT * NT, sizeof(starpu_data_handle_t) );

    for( m=0; m<MT; m++ ) {
        mm = m == (MT-1) ? M - m * b : b;

        for( n=0; n<NT; n++ ) {
            nn = n == (NT-1) ? N - n * b : b;

            hA = get_starpu_handle( 0, handlesA, A, m, n, b, MT );

            insert_dplrnt( bump, mm, nn, hA, b,
                           M, m * b, n * b, seed );
        }
    }

    unregister_starpu_handle( MT * NT, handlesA );

    /* Let's wait for the end of all the tasks */
    starpu_task_wait_for_all();
#if defined(ENABLE_MPI)
    starpu_mpi_barrier(MPI_COMM_WORLD);
#endif

    free( handlesA );
}
