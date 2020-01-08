#include <assert.h>
#include "algonum_int.h"

double **
lapack2tile( int M, int N, int b,
             const double *Alapack, int lda )
{
    /* Let's compute the total number of tiles with a *ceil* */
    int MT = (M + b - 1) / b;
    int NT = (N + b - 1) / b;
    int m, n;

    /* Allocate the array of pointers to the tiles */
    double **Atile = malloc( MT * NT * sizeof(double*) );

    /* Now, let's copy the tile one by one, in column major order */
    for( n=0; n<NT; n++) {
        for( m=0; m<MT; m++) {
            double *tile = malloc( b * b * sizeof(double) );
            int mm = m == (MT-1) ? M - m * b : b;
            int nn = n == (NT-1) ? N - n * b : b;

            /* Let's use LAPACKE to ease the copy */
            if ( Alapack != NULL ) {
                LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A', mm, nn,
                                     Alapack + lda * b * n + b * m, lda,
                                     tile, b );
            }
            Atile[ MT * n + m ] = tile;
        }
    }

    return Atile;
}

void
tile2lapack( int M, int N, int b,
             const double **Atile,
             double *Alapack, int lda )
{
    /* Let's compute the total number of tiles with a *ceil* */
    int MT = (M + b - 1) / b;
    int NT = (N + b - 1) / b;
    int m, n;

    assert( lda >= M );

    /* Now, let's copy the tile one by one, in column major order */
    for( n=0; n<NT; n++) {
        for( m=0; m<MT; m++) {
            const double *tile = Atile[ MT * n + m ];
            int mm = m == (MT-1) ? M - m * b : b;
            int nn = n == (NT-1) ? N - n * b : b;

            /* Let's use LAPACKE to ease the copy */
            LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A', mm, nn,
                                 tile, b,
                                 Alapack + lda * b * n + b * m, lda );
        }
    }
}

void
tileFree( int M, int N, int b, double **A )
{
    /* Let's compute the total number of tiles with a *ceil* */
    int MT = (M + b - 1) / b;
    int NT = (N + b - 1) / b;
    int m, n;

    /* Now, let's copy the tile one by one, in column major order */
    for( n=0; n<NT; n++) {
        for( m=0; m<MT; m++) {
            free( A[ MT * n + m ] );
        }
    }
    free( A );
}
