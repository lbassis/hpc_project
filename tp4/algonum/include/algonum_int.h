#ifndef _algonum_int_h
#define _algonum_int_h

#include <stdio.h>
#include <stdlib.h>
#include "algonum.h"

static inline int
max( int M, int N )
{
    return ( M > N ) ? M : N;
}

extern dgemm_fct_t dgemm_seq, dgemm_omp;

void CORE_dplrnt( double bump, int m, int n, double *A, int lda,
                  int bigM, int m0, int n0, unsigned long long int seed );

#endif /* _algonum_int_h */
