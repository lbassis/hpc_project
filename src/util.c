#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "util.h"
#include "perf.h"
#include "ddot.h"

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

int init_random(unsigned long m, unsigned long n, double **a) {

  unsigned long i, j;
  double *mat = *a;
  
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      mat[m*j+i] = (double)rand() / (double)((unsigned)RAND_MAX);
    }
  }
  return 0;
}

void ddot_warm() {

  int current_rep, current_max = MAX_REPS;
  unsigned long temp_m = 256;
  double r;
  double *a = alloc_vec(M_MAX);
  double performance;
  perf_t start, stop;

  init_random(1, M_MAX, &a);
  while (temp_m < M_MAX) {
    current_rep = 0;

    perf(&start);
    while (current_rep < current_max) {
      r = my_ddot(temp_m, a, 1, a, 1);
      current_rep++;
    }
    perf(&stop);

    perf_diff(&start, &stop);
    performance = current_max*perf_mflops(&stop, FLOPS_DDOT*temp_m);
    printf("%lu;%lf\n", temp_m, performance);

    temp_m *= 2;
    current_max = (current_max < 10)? 5:current_max/2;
  }

  free(a);
}
