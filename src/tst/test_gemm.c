#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl.h>

#include "util.h"
#include "perf.h"
#include "my_lib.h"

int main(void){

  double *a, *b, *c, *d;
  int i;
  int IONE = 1;
  long long int   ISEED[4] = {0,0,0,1};   /* initial seed for zlarnv() */

  double alpha = 1.;
  double beta = 1.3;
  int m =   150;
  int n =   150;
  int k =   150;
  int lda = m;
  int ldb = k;
  int ldc = m;

  a = alloc_mat(lda, k);
  b = alloc_mat(ldb, n);
  c = alloc_mat(ldc, n);
  d = alloc_mat(ldc, n);

  int tr[2] = {CblasTrans, CblasNoTrans};
  char* tr_name[2] = {"Trans", "NoTrans"};
  int ta, tb;
  for(ta = 0; ta < 2; ta++){
    for(tb = 0; tb < 2; tb++){
      printf("A : %s, B : %s\n", tr_name[ta], tr_name[tb]);
      /* random a b and c = d */
      LAPACKE_dlarnv_work(IONE, ISEED, lda*k, a);
      LAPACKE_dlarnv_work(IONE, ISEED, ldb*n, b);
      LAPACKE_dlarnv_work(IONE, ISEED, ldc*n, c);
      for (i = 0; i < ldc*n; i++) d[i] = c[i];

      /* affiche(m, n, d, ldc, stdout); */
      /* c = a*b */
      my_dgemm_omp(CblasColMajor, tr[ta], tr[tb], m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      /* d = a*b */
      cblas_dgemm(CblasColMajor, tr[ta], tr[tb], m, n, k, alpha, a, lda, b, ldb, beta, d, ldc);
      /* print d-c */
      for (i = 0; i < ldc*n; i++) {
        d[i] -= c[i];
      }
      printf("||cblas_gemm - my_gemm||1 = %lf\n", LAPACKE_dlange(CblasColMajor, 'M', m, n, d, ldc));
      //affiche(m, n, d, ldc, stdout);
      printf("_____\n");
    }
  }

  free(a);
  free(b);
  free(c);
  free(d);
  return 0;
}
