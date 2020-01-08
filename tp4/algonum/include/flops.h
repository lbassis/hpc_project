/**
 *
 * @file flops.h
 *
 * @copyright 2009-2014 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2019 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 *  File provided by Univ. of Tennessee,
 *
 * @version 0.9.2
 * @author Mathieu Faverge
 * @author Cedric Castagnede
 * @date 2014-11-16
 *
 */
/*
 * This file provide the flops formula for all Level 3 BLAS and some
 * Lapack routines.  Each macro uses the same size parameters as the
 * function associated and provide one formula for additions and one
 * for multiplications. Example to use these macros:
 *
 *    FLOPS_ZGEMM( m, n, k )
 *
 * All the formula are reported in the LAPACK Lawn 41:
 *     http://www.netlib.org/lapack/lawns/lawn41.ps
 */
#ifndef _flops_h_
#define _flops_h_

/**
 *           Generic formula coming from LAWN 41
 */

/*
 * Level 2 BLAS
 */
#define FMULS_GEMV(__m, __n) ((double)(__m) * (double)(__n) + 2. * (double)(__m))
#define FADDS_GEMV(__m, __n) ((double)(__m) * (double)(__n)                     )

#define FMULS_SYMV(__n) FMULS_GEMV( (__n), (__n) )
#define FADDS_SYMV(__n) FADDS_GEMV( (__n), (__n) )
#define FMULS_HEMV FMULS_SYMV
#define FADDS_HEMV FADDS_SYMV

/*
 * Level 3 BLAS
 */
#define FMULS_GEMM(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))
#define FADDS_GEMM(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))

#define FMULS_SYMM(__side, __m, __n) ( ( (__side) == CblasLeft ) ? FMULS_GEMM((__m), (__m), (__n)) : FMULS_GEMM((__m), (__n), (__n)) )
#define FADDS_SYMM(__side, __m, __n) ( ( (__side) == CblasLeft ) ? FADDS_GEMM((__m), (__m), (__n)) : FADDS_GEMM((__m), (__n), (__n)) )
#define FMULS_HEMM FMULS_SYMM
#define FADDS_HEMM FADDS_SYMM

#define FMULS_SYRK(__k, __n) (0.5 * (double)(__k) * (double)(__n) * ((double)(__n)+1.))
#define FADDS_SYRK(__k, __n) (0.5 * (double)(__k) * (double)(__n) * ((double)(__n)+1.))
#define FMULS_HERK FMULS_SYRK
#define FADDS_HERK FADDS_SYRK

#define FMULS_SYR2K(__k, __n) ((double)(__k) * (double)(__n) * (double)(__n)                )
#define FADDS_SYR2K(__k, __n) ((double)(__k) * (double)(__n) * (double)(__n) + (double)(__n))
#define FMULS_HER2K FMULS_SYR2K
#define FADDS_HER2K FADDS_SYR2K

#define FMULS_TRMM_2(__m, __n) (0.5 * (double)(__n) * (double)(__m) * ((double)(__m)+1.))
#define FADDS_TRMM_2(__m, __n) (0.5 * (double)(__n) * (double)(__m) * ((double)(__m)-1.))


#define FMULS_TRMM(__side, __m, __n) ( ( (__side) == CblasLeft ) ? FMULS_TRMM_2((__m), (__n)) : FMULS_TRMM_2((__n), (__m)) )
#define FADDS_TRMM(__side, __m, __n) ( ( (__side) == CblasLeft ) ? FADDS_TRMM_2((__m), (__n)) : FADDS_TRMM_2((__n), (__m)) )

#define FMULS_TRSM FMULS_TRMM
#define FADDS_TRSM FMULS_TRMM

/*
 * Lapack
 */
#define FMULS_GETRF(__m, __n) ( ((__m) < (__n)) ? (0.5 * (double)(__m) * ((double)(__m) * ((double)(__n) - (1./3.) * (__m) - 1. ) + (double)(__n)) + (2. / 3.) * (__m)) \
                                :                 (0.5 * (double)(__n) * ((double)(__n) * ((double)(__m) - (1./3.) * (__n) - 1. ) + (double)(__m)) + (2. / 3.) * (__n)) )
#define FADDS_GETRF(__m, __n) ( ((__m) < (__n)) ? (0.5 * (double)(__m) * ((double)(__m) * ((double)(__n) - (1./3.) * (__m)      ) - (double)(__n)) + (1. / 6.) * (__m)) \
                                :                 (0.5 * (double)(__n) * ((double)(__n) * ((double)(__m) - (1./3.) * (__n)      ) - (double)(__m)) + (1. / 6.) * (__n)) )

#define FMULS_GETRI(__n) ( (double)(__n) * ((5. / 6.) + (double)(__n) * ((2. / 3.) * (double)(__n) + 0.5)) )
#define FADDS_GETRI(__n) ( (double)(__n) * ((5. / 6.) + (double)(__n) * ((2. / 3.) * (double)(__n) - 1.5)) )

#define FMULS_GETRS(__n, __nrhs) ((double)(__nrhs) * (double)(__n) *  (double)(__n)       )
#define FADDS_GETRS(__n, __nrhs) ((double)(__nrhs) * (double)(__n) * ((double)(__n) - 1. ))

#define FMULS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n) + 0.5) * (double)(__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n)      ) * (double)(__n) - (1. / 6.)))

#define FMULS_POTRI(__n) ( (double)(__n) * ((2. / 3.) + (double)(__n) * ((1. / 3.) * (double)(__n) + 1. )) )
#define FADDS_POTRI(__n) ( (double)(__n) * ((1. / 6.) + (double)(__n) * ((1. / 3.) * (double)(__n) - 0.5)) )

#define FMULS_POTRS(__n, __nrhs) ((double)(__nrhs) * (double)(__n) * ((double)(__n) + 1. ))
#define FADDS_POTRS(__n, __nrhs) ((double)(__nrhs) * (double)(__n) * ((double)(__n) - 1. ))

//SPBTRF
//SPBTRS

#define FMULS_SYTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n) + 0.5) * (double)(__n) + (1. / 3.)))
#define FADDS_SYTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n)      ) * (double)(__n) - (1. / 6.)))

//SSYTRI
//SSYTRS

#define FMULS_GEQRF(__m, __n) (((__m) > (__n)) ? ((double)(__n) * ((double)(__n) * (  0.5-(1./3.) * (double)(__n) + (double)(__m)) +    (double)(__m) + 23. / 6.)) \
                               :                 ((double)(__m) * ((double)(__m) * ( -0.5-(1./3.) * (double)(__m) + (double)(__n)) + 2.*(double)(__n) + 23. / 6.)) )
#define FADDS_GEQRF(__m, __n) (((__m) > (__n)) ? ((double)(__n) * ((double)(__n) * (  0.5-(1./3.) * (double)(__n) + (double)(__m))                    +  5. / 6.)) \
                               :                 ((double)(__m) * ((double)(__m) * ( -0.5-(1./3.) * (double)(__m) + (double)(__n)) +    (double)(__n) +  5. / 6.)) )

#define FMULS_GEQLF(__m, __n) FMULS_GEQRF(__m, __n)
#define FADDS_GEQLF(__m, __n) FADDS_GEQRF(__m, __n)

#define FMULS_GERQF(__m, __n) (((__m) > (__n)) ? ((double)(__n) * ((double)(__n) * (  0.5-(1./3.) * (double)(__n) + (double)(__m)) +    (double)(__m) + 29. / 6.)) \
                               :                 ((double)(__m) * ((double)(__m) * ( -0.5-(1./3.) * (double)(__m) + (double)(__n)) + 2.*(double)(__n) + 29. / 6.)) )
#define FADDS_GERQF(__m, __n) (((__m) > (__n)) ? ((double)(__n) * ((double)(__n) * ( -0.5-(1./3.) * (double)(__n) + (double)(__m)) +    (double)(__m) +  5. / 6.)) \
                               :                 ((double)(__m) * ((double)(__m) * (  0.5-(1./3.) * (double)(__m) + (double)(__n)) +                  +  5. / 6.)) )

#define FMULS_GELQF(__m, __n) FMULS_GERQF(__m, __n)
#define FADDS_GELQF(__m, __n) FADDS_GERQF(__m, __n)

#define FMULS_UNGQR(__m, __n, __k) ((double)(__k) * (2.* (double)(__m) * (double)(__n) +            2. * (double)(__n) - 5./3. + (double)(__k) * ( 2./3. * (double)(__k) - ((double)(__m) + (double)(__n)) - 1.)))
#define FADDS_UNGQR(__m, __n, __k) ((double)(__k) * (2.* (double)(__m) * (double)(__n) + (double)(__n) - (double)(__m) + 1./3. + (double)(__k) * ( 2./3. * (double)(__k) - ((double)(__m) + (double)(__n))     )))
#define FMULS_UNGQL FMULS_UNGQR
#define FMULS_ORGQR FMULS_UNGQR
#define FMULS_ORGQL FMULS_UNGQR
#define FADDS_UNGQL FADDS_UNGQR
#define FADDS_ORGQR FADDS_UNGQR
#define FADDS_ORGQL FADDS_UNGQR

#define FMULS_UNGRQ(__m, __n, __k) ((double)(__k) * (2.* (double)(__m) * (double)(__n) + (double)(__m) + (double)(__n) - 2./3. + (double)(__k) * ( 2./3. * (double)(__k) - ((double)(__m) + (double)(__n)) - 1.)))
#define FADDS_UNGRQ(__m, __n, __k) ((double)(__k) * (2.* (double)(__m) * (double)(__n) + (double)(__m) - (double)(__n) + 1./3. + (double)(__k) * ( 2./3. * (double)(__k) - ((double)(__m) + (double)(__n))     )))
#define FMULS_UNGLQ FMULS_UNGRQ
#define FMULS_ORGRQ FMULS_UNGRQ
#define FMULS_ORGLQ FMULS_UNGRQ
#define FADDS_UNGLQ FADDS_UNGRQ
#define FADDS_ORGRQ FADDS_UNGRQ
#define FADDS_ORGLQ FADDS_UNGRQ

#define FMULS_GEQRS(__m, __n, __nrhs) ((double)(__nrhs) * ((double)(__n) * ( 2.* (double)(__m) - 0.5 * (double)(__n) + 2.5)))
#define FADDS_GEQRS(__m, __n, __nrhs) ((double)(__nrhs) * ((double)(__n) * ( 2.* (double)(__m) - 0.5 * (double)(__n) + 0.5)))

#define FMULS_UNMQR(__side, __m, __n, __k) ( ((__side) == CblasLeft ) ? ((double)(__k) *  (double)(__n) * ( 2.* (double)(__m) - (double)(__k) + 2.)) \
                                             :                         ((double)(__k) * ((double)(__m) * ( 2.* (double)(__n) - (double)(__k) + 1.) + (double)(__n) - .5 * (double)(__k) + .5)) )
#define FADDS_UNMQR(__side, __m, __n, __k) ( ((__side) == CblasLeft ) ? ((double)(__k) *  (double)(__n) * ( 2.* (double)(__m) - (double)(__k) + 1.)) \
                                             :                         ((double)(__k) *  (double)(__m) * ( 2.* (double)(__n) - (double)(__k) + 1.)) )

#define FMULS_UNMLQ FMULS_UNMQR
#define FADDS_UNMLQ FADDS_UNMQR

//UNMQR, UNMLQ, UNMQL, UNMRQ (Left)
//UNMQR, UNMLQ, UNMQL, UNMRQ (Right)

#define FMULS_TRTRI(__n) ((double)(__n) * ((double)(__n) * ( 1./6. * (double)(__n) + 0.5 ) + 1./3.))
#define FADDS_TRTRI(__n) ((double)(__n) * ((double)(__n) * ( 1./6. * (double)(__n) - 0.5 ) + 1./3.))

#define FMULS_GEHRD(__n) ( (double)(__n) * ((double)(__n) * (5./3. *(double)(__n) + 0.5) - 7./6.) - 13. )
#define FADDS_GEHRD(__n) ( (double)(__n) * ((double)(__n) * (5./3. *(double)(__n) - 1. ) - 2./3.) -  8. )

#define FMULS_SYTRD(__n) ( (double)(__n) *  ( (double)(__n) * ( 2./3. * (double)(__n) + 2.5 ) - 1./6. ) - 15.)
#define FADDS_SYTRD(__n) ( (double)(__n) *  ( (double)(__n) * ( 2./3. * (double)(__n) + 1.  ) - 8./3. ) -  4.)
#define FMULS_HETRD FMULS_SYTRD
#define FADDS_HETRD FADDS_SYTRD

#define FMULS_GEBRD(__m, __n) ( ((__m) >= (__n)) ? ((double)(__n) * ((double)(__n) * (2. * (double)(__m) - 2./3. * (double)(__n) + 2. )                 + 20./3.)) \
                                :                  ((double)(__m) * ((double)(__m) * (2. * (double)(__n) - 2./3. * (double)(__m) + 2. )                 + 20./3.)) )
#define FADDS_GEBRD(__m, __n) ( ((__m) >= (__n)) ? ((double)(__n) * ((double)(__n) * (2. * (double)(__m) - 2./3. * (double)(__n) + 1. ) - (double)(__m) +  5./3.)) \
                                :                  ((double)(__m) * ((double)(__m) * (2. * (double)(__n) - 2./3. * (double)(__m) + 1. ) - (double)(__n) +  5./3.)) )


/**
 *               Users functions
 */
/*
 * Level 2 BLAS
 */
static inline double flops_zgemv( double __m, double __n) { double flops =  (6. * FMULS_GEMV((__m), (__n)) + 2.0 * FADDS_GEMV((__m), (__n)) ); return flops; }
static inline double flops_cgemv( double __m, double __n) { double flops =  (6. * FMULS_GEMV((__m), (__n)) + 2.0 * FADDS_GEMV((__m), (__n)) ); return flops; }
static inline double flops_dgemv( double __m, double __n) { double flops =  (     FMULS_GEMV((__m), (__n)) +       FADDS_GEMV((__m), (__n)) ); return flops; }
static inline double flops_sgemv( double __m, double __n) { double flops =  (     FMULS_GEMV((__m), (__n)) +       FADDS_GEMV((__m), (__n)) ); return flops; }

static inline double flops_zhemv( double __n) { double flops =  (6. * FMULS_HEMV((__n)) + 2.0 * FADDS_HEMV((__n)) ); return flops; }
static inline double flops_chemv( double __n) { double flops =  (6. * FMULS_HEMV((__n)) + 2.0 * FADDS_HEMV((__n)) ); return flops; }

static inline double flops_zsymv( double __n) { double flops =  (6. * FMULS_SYMV((__n)) + 2.0 * FADDS_SYMV((__n)) ); return flops; }
static inline double flops_csymv( double __n) { double flops =  (6. * FMULS_SYMV((__n)) + 2.0 * FADDS_SYMV((__n)) ); return flops; }
static inline double flops_dsymv( double __n) { double flops =  (     FMULS_SYMV((__n)) +       FADDS_SYMV((__n)) ); return flops; }
static inline double flops_ssymv( double __n) { double flops =  (     FMULS_SYMV((__n)) +       FADDS_SYMV((__n)) ); return flops; }

/*
 * Level 3 BLAS
 */
static inline double flops_zgemm( double __m, double __n, double __k) { double flops =  (6. * FMULS_GEMM((__m), (__n), (__k)) + 2.0 * FADDS_GEMM((__m), (__n), (__k)) ); return flops; }
static inline double flops_cgemm( double __m, double __n, double __k) { double flops =  (6. * FMULS_GEMM((__m), (__n), (__k)) + 2.0 * FADDS_GEMM((__m), (__n), (__k)) ); return flops; }
static inline double flops_dgemm( double __m, double __n, double __k) { double flops =  (     FMULS_GEMM((__m), (__n), (__k)) +       FADDS_GEMM((__m), (__n), (__k)) ); return flops; }
static inline double flops_sgemm( double __m, double __n, double __k) { double flops =  (     FMULS_GEMM((__m), (__n), (__k)) +       FADDS_GEMM((__m), (__n), (__k)) ); return flops; }

static inline double flops_zhemm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (6. * FMULS_HEMM(__side, (__m), (__n)) + 2.0 * FADDS_HEMM(__side, (__m), (__n)) ); return flops; }
static inline double flops_chemm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (6. * FMULS_HEMM(__side, (__m), (__n)) + 2.0 * FADDS_HEMM(__side, (__m), (__n)) ); return flops; }

static inline double flops_zsymm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (6. * FMULS_SYMM(__side, (__m), (__n)) + 2.0 * FADDS_SYMM(__side, (__m), (__n)) ); return flops; }
static inline double flops_csymm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (6. * FMULS_SYMM(__side, (__m), (__n)) + 2.0 * FADDS_SYMM(__side, (__m), (__n)) ); return flops; }
static inline double flops_dsymm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (     FMULS_SYMM(__side, (__m), (__n)) +       FADDS_SYMM(__side, (__m), (__n)) ); return flops; }
static inline double flops_ssymm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (     FMULS_SYMM(__side, (__m), (__n)) +       FADDS_SYMM(__side, (__m), (__n)) ); return flops; }

static inline double flops_zherk( double __k, double __n) { double flops =  (6. * FMULS_HERK((__k), (__n)) + 2.0 * FADDS_HERK((__k), (__n)) ); return flops; }
static inline double flops_cherk( double __k, double __n) { double flops =  (6. * FMULS_HERK((__k), (__n)) + 2.0 * FADDS_HERK((__k), (__n)) ); return flops; }

static inline double flops_zsyrk( double __k, double __n) { double flops =  (6. * FMULS_SYRK((__k), (__n)) + 2.0 * FADDS_SYRK((__k), (__n)) ); return flops; }
static inline double flops_csyrk( double __k, double __n) { double flops =  (6. * FMULS_SYRK((__k), (__n)) + 2.0 * FADDS_SYRK((__k), (__n)) ); return flops; }
static inline double flops_dsyrk( double __k, double __n) { double flops =  (     FMULS_SYRK((__k), (__n)) +       FADDS_SYRK((__k), (__n)) ); return flops; }
static inline double flops_ssyrk( double __k, double __n) { double flops =  (     FMULS_SYRK((__k), (__n)) +       FADDS_SYRK((__k), (__n)) ); return flops; }

static inline double flops_zher2k( double __k, double __n) { double flops =  (6. * FMULS_HER2K((__k), (__n)) + 2.0 * FADDS_HER2K((__k), (__n)) ); return flops; }
static inline double flops_cher2k( double __k, double __n) { double flops =  (6. * FMULS_HER2K((__k), (__n)) + 2.0 * FADDS_HER2K((__k), (__n)) ); return flops; }

static inline double flops_zsyr2k( double __k, double __n) { double flops =  (6. * FMULS_SYR2K((__k), (__n)) + 2.0 * FADDS_SYR2K((__k), (__n)) ); return flops; }
static inline double flops_csyr2k( double __k, double __n) { double flops =  (6. * FMULS_SYR2K((__k), (__n)) + 2.0 * FADDS_SYR2K((__k), (__n)) ); return flops; }
static inline double flops_dsyr2k( double __k, double __n) { double flops =  (     FMULS_SYR2K((__k), (__n)) +       FADDS_SYR2K((__k), (__n)) ); return flops; }
static inline double flops_ssyr2k( double __k, double __n) { double flops =  (     FMULS_SYR2K((__k), (__n)) +       FADDS_SYR2K((__k), (__n)) ); return flops; }

static inline double flops_ztrmm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (6. * FMULS_TRMM(__side, (__m), (__n)) + 2.0 * FADDS_TRMM(__side, (__m), (__n)) ); return flops; }
static inline double flops_ctrmm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (6. * FMULS_TRMM(__side, (__m), (__n)) + 2.0 * FADDS_TRMM(__side, (__m), (__n)) ); return flops; }
static inline double flops_dtrmm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (     FMULS_TRMM(__side, (__m), (__n)) +       FADDS_TRMM(__side, (__m), (__n)) ); return flops; }
static inline double flops_strmm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (     FMULS_TRMM(__side, (__m), (__n)) +       FADDS_TRMM(__side, (__m), (__n)) ); return flops; }

static inline double flops_ztrsm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (6. * FMULS_TRSM(__side, (__m), (__n)) + 2.0 * FADDS_TRSM(__side, (__m), (__n)) ); return flops; }
static inline double flops_ctrsm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (6. * FMULS_TRSM(__side, (__m), (__n)) + 2.0 * FADDS_TRSM(__side, (__m), (__n)) ); return flops; }
static inline double flops_dtrsm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (     FMULS_TRSM(__side, (__m), (__n)) +       FADDS_TRSM(__side, (__m), (__n)) ); return flops; }
static inline double flops_strsm( CBLAS_SIDE __side, double __m, double __n) { double flops =  (     FMULS_TRSM(__side, (__m), (__n)) +       FADDS_TRSM(__side, (__m), (__n)) ); return flops; }

/*
 * Lapack
 */
static inline double flops_zgetrf( double __m, double __n) { double flops =  (6. * FMULS_GETRF((__m), (__n)) + 2.0 * FADDS_GETRF((__m), (__n)) ); return flops; }
static inline double flops_cgetrf( double __m, double __n) { double flops =  (6. * FMULS_GETRF((__m), (__n)) + 2.0 * FADDS_GETRF((__m), (__n)) ); return flops; }
static inline double flops_dgetrf( double __m, double __n) { double flops =  (     FMULS_GETRF((__m), (__n)) +       FADDS_GETRF((__m), (__n)) ); return flops; }
static inline double flops_sgetrf( double __m, double __n) { double flops =  (     FMULS_GETRF((__m), (__n)) +       FADDS_GETRF((__m), (__n)) ); return flops; }

static inline double flops_zgetri( double __n) { double flops =  (6. * FMULS_GETRI((__n)) + 2.0 * FADDS_GETRI((__n)) ); return flops; }
static inline double flops_cgetri( double __n) { double flops =  (6. * FMULS_GETRI((__n)) + 2.0 * FADDS_GETRI((__n)) ); return flops; }
static inline double flops_dgetri( double __n) { double flops =  (     FMULS_GETRI((__n)) +       FADDS_GETRI((__n)) ); return flops; }
static inline double flops_sgetri( double __n) { double flops =  (     FMULS_GETRI((__n)) +       FADDS_GETRI((__n)) ); return flops; }

static inline double flops_zgetrs( double __n, double __nrhs) { double flops =  (6. * FMULS_GETRS((__n), (__nrhs)) + 2.0 * FADDS_GETRS((__n), (__nrhs)) ); return flops; }
static inline double flops_cgetrs( double __n, double __nrhs) { double flops =  (6. * FMULS_GETRS((__n), (__nrhs)) + 2.0 * FADDS_GETRS((__n), (__nrhs)) ); return flops; }
static inline double flops_dgetrs( double __n, double __nrhs) { double flops =  (     FMULS_GETRS((__n), (__nrhs)) +       FADDS_GETRS((__n), (__nrhs)) ); return flops; }
static inline double flops_sgetrs( double __n, double __nrhs) { double flops =  (     FMULS_GETRS((__n), (__nrhs)) +       FADDS_GETRS((__n), (__nrhs)) ); return flops; }

static inline double flops_zpotrf( double __n) { double flops =  (6. * FMULS_POTRF((__n)) + 2.0 * FADDS_POTRF((__n)) ); return flops; }
static inline double flops_cpotrf( double __n) { double flops =  (6. * FMULS_POTRF((__n)) + 2.0 * FADDS_POTRF((__n)) ); return flops; }
static inline double flops_dpotrf( double __n) { double flops =  (     FMULS_POTRF((__n)) +       FADDS_POTRF((__n)) ); return flops; }
static inline double flops_spotrf( double __n) { double flops =  (     FMULS_POTRF((__n)) +       FADDS_POTRF((__n)) ); return flops; }

static inline double flops_zpotri( double __n) { double flops =  (6. * FMULS_POTRI((__n)) + 2.0 * FADDS_POTRI((__n)) ); return flops; }
static inline double flops_cpotri( double __n) { double flops =  (6. * FMULS_POTRI((__n)) + 2.0 * FADDS_POTRI((__n)) ); return flops; }
static inline double flops_dpotri( double __n) { double flops =  (     FMULS_POTRI((__n)) +       FADDS_POTRI((__n)) ); return flops; }
static inline double flops_spotri( double __n) { double flops =  (     FMULS_POTRI((__n)) +       FADDS_POTRI((__n)) ); return flops; }

static inline double flops_zpotrs( double __n, double __nrhs) { double flops =  (6. * FMULS_POTRS((__n), (__nrhs)) + 2.0 * FADDS_POTRS((__n), (__nrhs)) ); return flops; }
static inline double flops_cpotrs( double __n, double __nrhs) { double flops =  (6. * FMULS_POTRS((__n), (__nrhs)) + 2.0 * FADDS_POTRS((__n), (__nrhs)) ); return flops; }
static inline double flops_dpotrs( double __n, double __nrhs) { double flops =  (     FMULS_POTRS((__n), (__nrhs)) +       FADDS_POTRS((__n), (__nrhs)) ); return flops; }
static inline double flops_spotrs( double __n, double __nrhs) { double flops =  (     FMULS_POTRS((__n), (__nrhs)) +       FADDS_POTRS((__n), (__nrhs)) ); return flops; }

static inline double flops_zgeqrf( double __m, double __n) { double flops =  (6. * FMULS_GEQRF((__m), (__n)) + 2.0 * FADDS_GEQRF((__m), (__n)) ); return flops; }
static inline double flops_cgeqrf( double __m, double __n) { double flops =  (6. * FMULS_GEQRF((__m), (__n)) + 2.0 * FADDS_GEQRF((__m), (__n)) ); return flops; }
static inline double flops_dgeqrf( double __m, double __n) { double flops =  (     FMULS_GEQRF((__m), (__n)) +       FADDS_GEQRF((__m), (__n)) ); return flops; }
static inline double flops_sgeqrf( double __m, double __n) { double flops =  (     FMULS_GEQRF((__m), (__n)) +       FADDS_GEQRF((__m), (__n)) ); return flops; }

static inline double flops_zgeqlf( double __m, double __n) { double flops =  (6. * FMULS_GEQLF((__m), (__n)) + 2.0 * FADDS_GEQLF((__m), (__n)) ); return flops; }
static inline double flops_cgeqlf( double __m, double __n) { double flops =  (6. * FMULS_GEQLF((__m), (__n)) + 2.0 * FADDS_GEQLF((__m), (__n)) ); return flops; }
static inline double flops_dgeqlf( double __m, double __n) { double flops =  (     FMULS_GEQLF((__m), (__n)) +       FADDS_GEQLF((__m), (__n)) ); return flops; }
static inline double flops_sgeqlf( double __m, double __n) { double flops =  (     FMULS_GEQLF((__m), (__n)) +       FADDS_GEQLF((__m), (__n)) ); return flops; }

static inline double flops_zgerqf( double __m, double __n) { double flops =  (6. * FMULS_GERQF((__m), (__n)) + 2.0 * FADDS_GERQF((__m), (__n)) ); return flops; }
static inline double flops_cgerqf( double __m, double __n) { double flops =  (6. * FMULS_GERQF((__m), (__n)) + 2.0 * FADDS_GERQF((__m), (__n)) ); return flops; }
static inline double flops_dgerqf( double __m, double __n) { double flops =  (     FMULS_GERQF((__m), (__n)) +       FADDS_GERQF((__m), (__n)) ); return flops; }
static inline double flops_sgerqf( double __m, double __n) { double flops =  (     FMULS_GERQF((__m), (__n)) +       FADDS_GERQF((__m), (__n)) ); return flops; }

static inline double flops_zgelqf( double __m, double __n) { double flops =  (6. * FMULS_GELQF((__m), (__n)) + 2.0 * FADDS_GELQF((__m), (__n)) ); return flops; }
static inline double flops_cgelqf( double __m, double __n) { double flops =  (6. * FMULS_GELQF((__m), (__n)) + 2.0 * FADDS_GELQF((__m), (__n)) ); return flops; }
static inline double flops_dgelqf( double __m, double __n) { double flops =  (     FMULS_GELQF((__m), (__n)) +       FADDS_GELQF((__m), (__n)) ); return flops; }
static inline double flops_sgelqf( double __m, double __n) { double flops =  (     FMULS_GELQF((__m), (__n)) +       FADDS_GELQF((__m), (__n)) ); return flops; }

static inline double flops_zungqr( double __m, double __n, double __k) { double flops =  (6. * FMULS_UNGQR((__m), (__n), (__k)) + 2.0 * FADDS_UNGQR((__m), (__n), (__k)) ); return flops; }
static inline double flops_cungqr( double __m, double __n, double __k) { double flops =  (6. * FMULS_UNGQR((__m), (__n), (__k)) + 2.0 * FADDS_UNGQR((__m), (__n), (__k)) ); return flops; }
static inline double flops_dorgqr( double __m, double __n, double __k) { double flops =  (     FMULS_UNGQR((__m), (__n), (__k)) +       FADDS_UNGQR((__m), (__n), (__k)) ); return flops; }
static inline double flops_sorgqr( double __m, double __n, double __k) { double flops =  (     FMULS_UNGQR((__m), (__n), (__k)) +       FADDS_UNGQR((__m), (__n), (__k)) ); return flops; }

static inline double flops_zungql( double __m, double __n, double __k) { double flops =  (6. * FMULS_UNGQL((__m), (__n), (__k)) + 2.0 * FADDS_UNGQL((__m), (__n), (__k)) ); return flops; }
static inline double flops_cungql( double __m, double __n, double __k) { double flops =  (6. * FMULS_UNGQL((__m), (__n), (__k)) + 2.0 * FADDS_UNGQL((__m), (__n), (__k)) ); return flops; }
static inline double flops_dorgql( double __m, double __n, double __k) { double flops =  (     FMULS_UNGQL((__m), (__n), (__k)) +       FADDS_UNGQL((__m), (__n), (__k)) ); return flops; }
static inline double flops_sorgql( double __m, double __n, double __k) { double flops =  (     FMULS_UNGQL((__m), (__n), (__k)) +       FADDS_UNGQL((__m), (__n), (__k)) ); return flops; }

static inline double flops_zungrq( double __m, double __n, double __k) { double flops =  (6. * FMULS_UNGRQ((__m), (__n), (__k)) + 2.0 * FADDS_UNGRQ((__m), (__n), (__k)) ); return flops; }
static inline double flops_cungrq( double __m, double __n, double __k) { double flops =  (6. * FMULS_UNGRQ((__m), (__n), (__k)) + 2.0 * FADDS_UNGRQ((__m), (__n), (__k)) ); return flops; }
static inline double flops_dorgrq( double __m, double __n, double __k) { double flops =  (     FMULS_UNGRQ((__m), (__n), (__k)) +       FADDS_UNGRQ((__m), (__n), (__k)) ); return flops; }
static inline double flops_sorgrq( double __m, double __n, double __k) { double flops =  (     FMULS_UNGRQ((__m), (__n), (__k)) +       FADDS_UNGRQ((__m), (__n), (__k)) ); return flops; }

static inline double flops_zunglq( double __m, double __n, double __k) { double flops =  (6. * FMULS_UNGLQ((__m), (__n), (__k)) + 2.0 * FADDS_UNGLQ((__m), (__n), (__k)) ); return flops; }
static inline double flops_cunglq( double __m, double __n, double __k) { double flops =  (6. * FMULS_UNGLQ((__m), (__n), (__k)) + 2.0 * FADDS_UNGLQ((__m), (__n), (__k)) ); return flops; }
static inline double flops_dorglq( double __m, double __n, double __k) { double flops =  (     FMULS_UNGLQ((__m), (__n), (__k)) +       FADDS_UNGLQ((__m), (__n), (__k)) ); return flops; }
static inline double flops_sorglq( double __m, double __n, double __k) { double flops =  (     FMULS_UNGLQ((__m), (__n), (__k)) +       FADDS_UNGLQ((__m), (__n), (__k)) ); return flops; }

static inline double flops_zunmqr( CBLAS_SIDE side, double __m, double __n, double __k) { double flops =  (6. * FMULS_UNMQR(side, (__m), (__n), (__k)) + 2.0 * FADDS_UNMQR(side, (__m), (__n), (__k)) ); return flops; }
static inline double flops_cunmqr( CBLAS_SIDE side, double __m, double __n, double __k) { double flops =  (6. * FMULS_UNMQR(side, (__m), (__n), (__k)) + 2.0 * FADDS_UNMQR(side, (__m), (__n), (__k)) ); return flops; }
static inline double flops_dormqr( CBLAS_SIDE side, double __m, double __n, double __k) { double flops =  (     FMULS_UNMQR(side, (__m), (__n), (__k)) +       FADDS_UNMQR(side, (__m), (__n), (__k)) ); return flops; }
static inline double flops_sormqr( CBLAS_SIDE side, double __m, double __n, double __k) { double flops =  (     FMULS_UNMQR(side, (__m), (__n), (__k)) +       FADDS_UNMQR(side, (__m), (__n), (__k)) ); return flops; }

static inline double flops_zunmlq( CBLAS_SIDE side, double __m, double __n, double __k) { double flops =  (6. * FMULS_UNMLQ(side, (__m), (__n), (__k)) + 2.0 * FADDS_UNMLQ(side, (__m), (__n), (__k)) ); return flops; }
static inline double flops_cunmlq( CBLAS_SIDE side, double __m, double __n, double __k) { double flops =  (6. * FMULS_UNMLQ(side, (__m), (__n), (__k)) + 2.0 * FADDS_UNMLQ(side, (__m), (__n), (__k)) ); return flops; }
static inline double flops_dormlq( CBLAS_SIDE side, double __m, double __n, double __k) { double flops =  (     FMULS_UNMLQ(side, (__m), (__n), (__k)) +       FADDS_UNMLQ(side, (__m), (__n), (__k)) ); return flops; }
static inline double flops_sormlq( CBLAS_SIDE side, double __m, double __n, double __k) { double flops =  (     FMULS_UNMLQ(side, (__m), (__n), (__k)) +       FADDS_UNMLQ(side, (__m), (__n), (__k)) ); return flops; }

static inline double flops_zgeqrs( double __m, double __n, double __nrhs) { double flops =  (6. * FMULS_GEQRS((__m), (__n), (__nrhs)) + 2.0 * FADDS_GEQRS((__m), (__n), (__nrhs)) ); return flops; }
static inline double flops_cgeqrs( double __m, double __n, double __nrhs) { double flops =  (6. * FMULS_GEQRS((__m), (__n), (__nrhs)) + 2.0 * FADDS_GEQRS((__m), (__n), (__nrhs)) ); return flops; }
static inline double flops_dgeqrs( double __m, double __n, double __nrhs) { double flops =  (     FMULS_GEQRS((__m), (__n), (__nrhs)) +       FADDS_GEQRS((__m), (__n), (__nrhs)) ); return flops; }
static inline double flops_sgeqrs( double __m, double __n, double __nrhs) { double flops =  (     FMULS_GEQRS((__m), (__n), (__nrhs)) +       FADDS_GEQRS((__m), (__n), (__nrhs)) ); return flops; }

static inline double flops_ztrtri( double __n) { double flops =  (6. * FMULS_TRTRI((__n)) + 2.0 * FADDS_TRTRI((__n)) ); return flops; }
static inline double flops_ctrtri( double __n) { double flops =  (6. * FMULS_TRTRI((__n)) + 2.0 * FADDS_TRTRI((__n)) ); return flops; }
static inline double flops_dtrtri( double __n) { double flops =  (     FMULS_TRTRI((__n)) +       FADDS_TRTRI((__n)) ); return flops; }
static inline double flops_strtri( double __n) { double flops =  (     FMULS_TRTRI((__n)) +       FADDS_TRTRI((__n)) ); return flops; }

static inline double flops_zgehrd( double __n) { double flops =  (6. * FMULS_GEHRD((__n)) + 2.0 * FADDS_GEHRD((__n)) ); return flops; }
static inline double flops_cgehrd( double __n) { double flops =  (6. * FMULS_GEHRD((__n)) + 2.0 * FADDS_GEHRD((__n)) ); return flops; }
static inline double flops_dgehrd( double __n) { double flops =  (     FMULS_GEHRD((__n)) +       FADDS_GEHRD((__n)) ); return flops; }
static inline double flops_sgehrd( double __n) { double flops =  (     FMULS_GEHRD((__n)) +       FADDS_GEHRD((__n)) ); return flops; }

static inline double flops_zhetrd( double __n) { double flops =  (6. * FMULS_HETRD((__n)) + 2.0 * FADDS_HETRD((__n)) ); return flops; }
static inline double flops_chetrd( double __n) { double flops =  (6. * FMULS_HETRD((__n)) + 2.0 * FADDS_HETRD((__n)) ); return flops; }

static inline double flops_zsytrd( double __n) { double flops =  (6. * FMULS_SYTRD((__n)) + 2.0 * FADDS_SYTRD((__n)) ); return flops; }
static inline double flops_csytrd( double __n) { double flops =  (6. * FMULS_SYTRD((__n)) + 2.0 * FADDS_SYTRD((__n)) ); return flops; }
static inline double flops_dsytrd( double __n) { double flops =  (     FMULS_SYTRD((__n)) +       FADDS_SYTRD((__n)) ); return flops; }
static inline double flops_ssytrd( double __n) { double flops =  (     FMULS_SYTRD((__n)) +       FADDS_SYTRD((__n)) ); return flops; }

static inline double flops_zgebrd( double __m, double __n) { double flops =  (6. * FMULS_GEBRD((__m), (__n)) + 2.0 * FADDS_GEBRD((__m), (__n)) ); return flops; }
static inline double flops_cgebrd( double __m, double __n) { double flops =  (6. * FMULS_GEBRD((__m), (__n)) + 2.0 * FADDS_GEBRD((__m), (__n)) ); return flops; }
static inline double flops_dgebrd( double __m, double __n) { double flops =  (     FMULS_GEBRD((__m), (__n)) +       FADDS_GEBRD((__m), (__n)) ); return flops; }
static inline double flops_sgebrd( double __m, double __n) { double flops =  (     FMULS_GEBRD((__m), (__n)) +       FADDS_GEBRD((__m), (__n)) ); return flops; }

/*
 * Norms
 */
#define FMULS_LANGE(__m, __n) ((double)(__m) * (double)(__n))
#define FADDS_LANGE(__m, __n) ((double)(__m) * (double)(__n))

#endif /* _flops_h_ */
