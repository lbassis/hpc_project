#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl.h>

#include "util.h"
#include "perf.h"
#include "my_lib.h"

int main(void){
  printf("%s: \n", __FILE__);

  double *a, *b, *c;
  int i;
  int IONE = 1;
  long long int   ISEED[4] = {0,0,0,1};   /* initial seed for zlarnv() */

  double alpha = 4.;
  int m =   150;
  int n =   150;
  int lda = m;
  int ldb = n;

  a = alloc_mat(lda, n);
  b = alloc_mat(ldb, m);
  c = alloc_mat(ldb, m);

  int tr[2] = {CblasTrans, CblasNoTrans};
  char* tr_name[2] = {"Trans", "NoTrans"};
  int u[2] = {CblasUnit, CblasNonUnit};
  char* u_name[2] = {"Unit", "NoUnit"};
  int l[2] = {CblasLower, CblasUpper};
  char* l_name[2] = {"Lower", "Upper"};
  int r[2] = {CblasLeft, CblasRight};
  char* r_name[2] = {"Left", "Right"};
  int ta, ui, li, ri;
  for(ta = 0; ta < 2; ta++){
    for(ui = 0; ui < 2; ui++){
      for(li = 0; li < 2; li++){
        for(ri = 0; ri < 2; ri++){

          printf("%s, %s, %s, %s\n", tr_name[ta], u_name[ui], l_name[li], r_name[ri]);
          /* random a b and c = d */
          LAPACKE_dlarnv_work(IONE, ISEED, lda*n, a);
          LAPACKE_dlarnv_work(IONE, ISEED, ldb*m, b);

          for (i = 0; i < ldb * m; i++) c[i] = b[i];
          //affiche(m, n, b, lda, stdout);

          my_dtrsm(/*int *Layout*/   CblasColMajor,
                  /*int side*/      r[ri],
                  /*int uplo*/      l[li],
                  /*int transA*/    tr[ta],
                  /*int diag*/      u[ui],
                  /*int m*/         m,
                  /*int n*/         n,
                  /*double alpha*/  alpha,
                  /*double *a*/     a,
                  /*int lda*/       lda,
                  /*double *b*/     b,
                  /*int ldb*/       ldb);

          cblas_dtrsm(/*int *Layout*/ CblasColMajor,
                      /*int side*/      r[ri],
                      /*int uplo*/      l[li],
                      /*int transA*/    tr[ta],
                      /*int diag*/      u[ui],
                      /*int m */        m,
                      /*int n */        n,
                      /*double alpha*/  alpha,
                      /*double *a*/     a,
                      /*int lda*/       lda,
                      /*double *b*/     c,
                      /*int ldb*/       ldb);

            //printf("\n");
            //affiche(m, n, c, ldb, stdout);
            //printf("\n");
            //affiche(m, n, b, ldb, stdout);
            for (i = 0; i < ldb * m; i++) c[i] -= b[i];

            printf("||cblas_dtrsm - my_dtrsm||1 = %lf\n", LAPACKE_dlange(CblasColMajor, 'M', n, m, c, ldb));
            printf("_____\n");
        }
      }
    }
  }

  free(a);
  free(b);
  free(c);
  return 0;
}
