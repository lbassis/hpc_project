#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util.h"
#include "ddot.h"
#include "perf.h"

int main() {

  double a[] = {2, -1, 3, 4};
  double b[] = {3, 5, 3, 2};
  double *c = my_dgemm_scalaire(2, a, b);
  affiche(2, 2, a, 2, stdout);
  printf("x\n");
  affiche(2, 2, b, 2, stdout);
  printf("=\n");
  affiche(2, 2, c, 2, stdout);

  return 0;
}
