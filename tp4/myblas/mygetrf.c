#include "algonum.h"

// legacy code

void
my_dgetrf_seq( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
{
    int m, n, k;
    int K = ( M > N ) ? N : M;

    for( k=0; k<K; k++ ) {
        for( m=k+1; m<M; m++ ) {
            A[ lda * k + m ] = A[ lda * k + m ] / A[ lda * k + k ];
            for( n=k+1; n<N; n++ ) {
                A[ lda * n + m ] = A[ lda * n + m ] - A[ lda * k + m ] * A[ lda * n + k ];
            }
        }
    }
}

void
my_dgetrf_openmp( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
{
    my_dgetrf_seq( layout, M, N, A, lda );
}

void my_dgetrf_tiled_openmp( CBLAS_LAYOUT layout,
                             int M, int N, int b, double **A )
{
    int m, n, k;
    int K = ( M > N ) ? N : M;

    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int KT = my_iceil( K, b );

    for( k=0; k<KT; k++) {
        int kk = k == (KT-1) ? M - k * b : b;

        my_dgetrf_seq( CblasColMajor, kk, kk, A[ MT * k + k ], b );

        for( n=k+1; n<NT; n++) {
            int nn = n == (NT-1) ? N - n * b : b;

	    cblas_dtrsm( CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
			 kk, nn, 1.,
			 A[ MT * k + k ], b,
			 A[ MT * n + k ], b );
        }

        for( m=k+1; m<MT; m++) {
            int mm = m == (MT-1) ? M - m * b : b;

	    cblas_dtrsm( CblasColMajor, CblasRight, CblasUnit, CblasNoTrans, CblasNonUnit,
			 mm, kk, 1.,
			 A[ MT * k + k ], b,
			 A[ MT * k + m ], b );

            for( n=k+1; n<NT; n++) {
                int nn = n == (NT-1) ? N - n * b : b;

		cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, mm, nn, kk,
			     -1., A[ MT * k + m ], b,
			          A[ MT * n + k ], b,
			      1., A[ MT * n + m ], b );
            }
        }
    }
}

/* To make sure we use the right prototype */
static dgetrf_fct_t valig_mygetrf __attribute__ ((unused)) = my_dgetrf_seq;
