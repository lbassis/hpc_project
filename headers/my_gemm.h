void my_dgemm_scalaire(int m, double *a, double *b, double* c);
void my_dgemm_scalaire_kij(int m, double *a, double *b, double* c);
void my_dgemm_scalaire_ijk(int m, double *a, double *b, double* c);
void my_dgemm_scalaire_jik(int m, double *a, double *b, double* c);
void my_dgemm_seq(int transA, int transB, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc);
void my_dgemm(int transA, int transB, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc);
