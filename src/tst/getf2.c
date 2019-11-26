#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl.h>

#include "util.h"
#include "perf.h"
#include "my_lib.h"


int main(void){
  printf("%s: \n", __FILE__);

  double *a, *b;
  int i;
  int IONE = 1;
  long long int   ISEED[4] = {0,0,0,1};   /* initial seed for zlarnv() */

  int m = 150;
  int n = 150;
  int lda = m;
  long long int* ipiv = (long long int*)malloc(m * sizeof(long long int));

  a = alloc_mat(lda, n);
  b = alloc_mat(lda, n);
  init_random(m, n, a, 1);

  for (i = 0; i < lda*n; i++) b[i] = a[i];

  my_dgetf2(CblasColMajor, m, n, a, lda, NULL);
  LAPACKE_dgetf2(LAPACK_COL_MAJOR, m, n, b, lda, ipiv);

  for (i = 0; i < lda*n; i++) b[i] -= a[i];

  printf("||LAPACKE_dgetf2 - my_dgetf2||1 = %lf\n", LAPACKE_dlange(CblasColMajor, 'M', m, n, b, lda));
  printf("||ipiv||1 = %lf\n", LAPACKE_dlange(CblasColMajor, 'M', m, 1, (double*)ipiv, m));
  printf("_____\n");


  free(a);
  free(b);
  free(ipiv);
  return 0;
}
