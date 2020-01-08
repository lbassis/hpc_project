/**
 *
 * @file core_dplrnt.c
 *
 * @copyright 2019-2019 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon core_dplrnt CPU kernel
 *
 * @version 0.9.2
 * @author Mathieu Faverge
 * @author Emmanuel Agullo
 * @date 2019-11-13
 *
 */
#include "algonum_int.h"

/*
  Rnd64seed is a global variable but it doesn't spoil thread safety. All matrix
  generating threads only read Rnd64seed. It is safe to set Rnd64seed before
  and after any calls to create_tile(). The only problem can be caused if
  Rnd64seed is changed during the matrix generation time.
*/

//static unsigned long long int Rnd64seed = 100;
#define Rnd64_A 6364136223846793005ULL
#define Rnd64_C 1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20

static unsigned long long int
Rnd64_jump(unsigned long long int n, unsigned long long int seed ) {
    unsigned long long int a_k, c_k, ran;
    int i;

    a_k = Rnd64_A;
    c_k = Rnd64_C;

    ran = seed;
    for (i = 0; n; n >>= 1, ++i) {
	if (n & 1)
	    ran = a_k * ran + c_k;
	c_k *= (a_k + 1);
	a_k *= a_k;
    }

    return ran;
}

void CORE_dplrnt( double bump, int m, int n, double *A, int lda,
                  int bigM, int m0, int n0, unsigned long long int seed )
{
    double *tmp = A;
    int64_t i, j;
    unsigned long long int ran, jump;

    jump = (unsigned long long int)m0 + (unsigned long long int)n0 * (unsigned long long int)bigM;

    for (j=0; j<n; ++j ) {
        ran = Rnd64_jump( jump, seed );
        for (i = 0; i < m; ++i) {
            *tmp = 0.5f - ran * RndF_Mul;
            ran  = Rnd64_A * ran + Rnd64_C;

	    if ( (i + m0) == (j + n0) ) {
		*tmp += bump;
	    }

            tmp++;
        }
        tmp  += lda-i;
        jump += bigM;
    }
}

void dplrnt_tiled( double bump, int M, int N, int b,
                   double **A, unsigned long long int seed )
{
    int MT = (M + b - 1) / b;
    int NT = (N + b - 1) / b;
    int m, n;
    int mm, nn;

    for( m=0; m<MT; m++ ) {
        mm = m == (MT-1) ? M - m * b : b;

        for( n=0; n<NT; n++ ) {
            nn = n == (NT-1) ? N - n * b : b;

            CORE_dplrnt( bump, mm, nn, A[ MT * n + m ], b,
                         M, m * b, n * b, seed );
        }
    }
}
