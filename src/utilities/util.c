#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "util.h"
#include "perf.h"

#define M_MAX 1e9
#define MAX_REPS 1e5
#define FLOPS_DDOT 4

void affiche(unsigned long m, unsigned long n, double *a, unsigned long lda, FILE *flux) {

  unsigned long i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      fprintf(flux, "%f ", a[lda*j + i]);
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
  srand(seed);
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      a[m*j+i] = (double)rand() / (double)((unsigned)RAND_MAX);

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
