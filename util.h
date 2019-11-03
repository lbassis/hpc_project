void affiche(int m, int n, double *a, int lda, FILE *flux);
double *alloc_mat(int m, int n);
double *alloc_vec(int n);
int init_random(int m, int n, double **a);
void boucle_ddot_wcache();
