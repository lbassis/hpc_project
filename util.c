#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "perf.h"
#include "ddot.h"

#define RAND_MAX 1
#define M_MAX 1e9
#define FLOPS_DDOT 4

void affiche(int m, int n, double *a, int lda, FILE *flux) {

  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      fprintf(flux, "%f ", a[lda*j + i]);
    }
    printf("\n");
  }
}

double *alloc_mat(int m, int n) {  
  return calloc(m*n, sizeof(double));
}

double *alloc_vec(int n) {
  return alloc_mat(1, n);
}

int init_random(int m, int n, double **a) {

  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      (*a)[m*j+i] = (double)rand() / (double)((unsigned)RAND_MAX + 1);
    }
  }
  return 0;
}

void boucle_ddot_wcache() {

  double r;
  long temp_m = 150;
  double *a = alloc_vec(M_MAX);
  double performance;
  perf_t start, stop;

  while (temp_m < M_MAX) {
    perf(&start);
    r = my_ddot(temp_m, a, 1, a, 1);
    perf(&stop);

    perf_diff(&start, &stop);
    performance = perf_mflops(&stop, FLOPS_DDOT*temp_m);
    printf("%d;%lf\n", temp_m, performance);


    temp_m = round(pow(temp_m, 1.1));
  }

  free(a);
}
