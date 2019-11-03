#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util.h"
#include "ddot.h"

int main() {

  double *a = alloc_vec(5);
  init_random(5, 1, &a);
  affiche(5, 1, a, 1, stdout);
  //boucle_ddot_wcache();
  
}

void test() {

  int i, j, m, n, lda;
  double *a, b;
  m = 4;
  n = 4;
  lda = 7;
  a = alloc_mat(lda, n);

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      a[lda*i+j] = m*i+j;
    }
  }

  affiche(4, 4, a, 7, stdout);
  b = my_ddot(2, a, 3, a, 1);
  printf("b: %f\n", b);
}
