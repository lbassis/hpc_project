double my_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);
void my_daxpy (int n, double a, double *x, int incX, double *y, int incY);
void my_dgemv(int transA, int m, int n, double alpha, double *A, int lda, double *X, int incX, double beta, double *Y, int incY);
void my_dger(int m, int n, double alpha, double *X, int incX, double *Y, int incY, double *A, int lda);
void my_dgetf2( int m, int n, double* a, int lda, int* ipiv );
void my_dtrsm (int *Layout, int side, int uplo, int transA, int diag, int m,
               int n, double alpha, double *a, int lda, double *b, int ldb);

