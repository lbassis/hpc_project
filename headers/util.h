void affiche(unsigned long m, unsigned long n, double *a, unsigned long lda, FILE *flux);
double *alloc_mat(unsigned long m, unsigned long n);
double *alloc_vec(unsigned long n);
int init_random(unsigned long m, unsigned long n, double *a, unsigned int seed);

double **
lapack2tile( int M, int N, int b,
             const double *Alapack, int lda );
void
tile2lapack( int M, int N, int b,
             const double **Atile,
             double *A, int lda );
