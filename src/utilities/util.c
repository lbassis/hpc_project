#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mkl.h>
#include "util.h"
#include "perf.h"

#define M_MAX 1e9
#define MAX_REPS 1e5
#define FLOPS_DDOT 4

void affiche(unsigned long m, unsigned long n, double *a, unsigned long lda, FILE *flux) {

  unsigned long i, j;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      fprintf(flux, "%.0lf ", a[i + j * lda]);
    }
    printf("\n");
  }
}

double *alloc_mat(unsigned long m, unsigned long n) {
  return calloc(m*n, sizeof(double));
}

double *alloc_vec(unsigned long n) {
  return alloc_mat(1, n);
}

int init_random(unsigned long m, unsigned long n, double *a, unsigned int seed) {

  unsigned long i, j;
  long long int   ISEED[4] = {0,0,0,seed};   /* initial seed for zlarnv() */
  LAPACKE_dlarnv_work(1, ISEED, m*n, a);

  srand(seed);
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      if (i == j) { // si on est dans une diagonale, on s'assure qu'elle est dominante
	a[m*j+i] += m;
      }
    }
  }
  return 0;
}

int init_identity(unsigned long m, unsigned long n, double **a) {

  unsigned long i, j;
  double *mat = *a;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      if (i == j) {
	mat[i*m+j] = 1;
      }
      else {
	mat[i*m+j] = 0;
      }
    }
  }
  return 0;
}

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
            double *tile = calloc( b * b, sizeof(double) );
            int mm = m == (MT-1) ? M - m * b : b;
            int nn = n == (NT-1) ? N - n * b : b;

            /* Let's use LAPACKE to ease the copy */
            LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A', mm, nn,
                                 Alapack+( lda * b * n + b * m ), lda,
                                 tile, b );

            Atile[ MT * n + m ] = tile;
        }
    }
    return Atile;
}

void
tile2lapack( int M, int N, int b,
             const double **Atile,
             double *A, int lda )
{
    /* Let's compute the total number of tiles with a *ceil* */
    int MT = (M + b - 1) / b;
    int NT = (N + b - 1) / b;
    int m, n;

    //assert( lda >= M );

    /* Now, let's copy the tile one by one, in column major order */
    for( n=0; n<NT; n++) {
        for( m=0; m<MT; m++) {
	    const double *tile = Atile[ MT * n + m ];
            int mm = m == (MT-1) ? M - m * b : b;
            int nn = n == (NT-1) ? N - n * b : b;

            /* Let's use LAPACKE to ease the copy */
            LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A', mm, nn,
                                 tile, b,
                                 A+( lda * b * n + b * m ), lda );
            //free(tile);
        }
    }
}
