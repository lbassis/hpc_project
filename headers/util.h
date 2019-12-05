#ifndef UTIL_H
#define UTIL_H


#include <mpi.h>

void affiche(unsigned long m, unsigned long n, double *a, unsigned long lda, FILE *flux);

double *alloc_mat(unsigned long m, unsigned long n);

double *alloc_vec(unsigned long n);

int init_random(unsigned long m, unsigned long n, double *a, unsigned int seed);

double** lapack2tile(int M, int N, int b, const double *Alapack, int lda);

void tile2lapack(int M, int N, int b, const double** Atile, double *A, int lda );

void scatter_matrix(const int m,
                    const int n,
                    const double** in,
                    double*** out,
                    const int nb_proc,
                    const int me,
                    const int dim[2],
                    const MPI_Comm comm);

void gather_matrix(const int m,
                    const int n,
                    const double** in,
                    double** out,
                    const int nb_proc,
                    const int me,
                    const int dim[2],
                    const MPI_Comm comm);


#endif
