#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <strings.h>
#include <assert.h>
#include "algonum.h"
#if defined(ENABLE_STARPU)
#include "codelets.h"
#endif

dgetrf_fct_t dgetrf_seq = my_dgetrf_seq;
dgetrf_fct_t dgetrf_omp = NULL;

void dgetrf_mkl( CBLAS_LAYOUT layout,
                 int m, int n, double *a, int lda )
{
    int minMN = ( m < n ) ? m : n;
    int *ipiv = malloc( minMN * sizeof(int) );
    int rc;

    rc = LAPACKE_dgetrf_work( LAPACK_COL_MAJOR, m, n, a, lda, ipiv );
    assert( rc == 0 );

#if !defined(NDEBUG)
    {
        int i;
        for(i=0; i<minMN; i++) {
            assert( ipiv[i] == i+1 );
        }
    }
#endif
    free( ipiv );
}

#define GETOPT_STRING "hv:M:N:K:AB"
static struct option long_options[] =
{
    {"help",          no_argument,       0,      'h'},
    {"v",             required_argument, 0,      'v'},
    // Matrix parameters
    {"M",             required_argument, 0,      'M'},
    {"N",             required_argument, 0,      'N'},
    {"nb",            required_argument, 0,      'b'},
    {0, 0, 0, 0}
};

void
print_usage()
{
    printf( "Options:\n"
            "  -h --help  Show this help\n"
            "  -v --v=xxx Select the version to test among: seq, omp, mkl\n"
            "  -M x       Set the M value\n"
            "  -N x       Set the N value\n"
            "  -b x       Set the block size b value\n" );
    exit(1);
}

int main( int argc, char **argv )
{
    dgetrf_fct_t tested_dgetrf = dgetrf_seq;
    dgetrf_tiled_fct_t tested_tiled_dgetrf = NULL;
    dplrnt_tiled_fct_t tested_tiled_dplrnt = NULL;
    int starpu = 0;
    int opt;

    while ((opt = getopt_long(argc, argv, GETOPT_STRING, long_options, NULL)) != -1)
    {
        switch(opt) {
        case 'h':
            print_usage(argv[0]);
            exit(0);

        case 'v':
            if ( strcasecmp( optarg, "mkl" ) == 0 ) {
                printf( "Test MKL version\n" );
                tested_dgetrf = dgetrf_mkl;
            }
            else if ( strcasecmp( optarg, "seq" ) == 0 ) {
                printf( "Test Sequential version\n" );
                tested_dgetrf = my_dgetrf_seq;
            }
            else if ( strcasecmp( optarg, "omp" ) == 0 ) {
                printf( "Test OpenMP version\n" );
                tested_dgetrf = my_dgetrf_openmp;
            }
            else if ( strcasecmp( optarg, "tiled_omp" ) == 0 ) {
                printf( "Test tiled OpenMP version\n" );
                tested_tiled_dgetrf = my_dgetrf_tiled_openmp;
                tested_tiled_dplrnt = dplrnt_tiled;
            }
#if defined(ENABLE_STARPU)
            else if ( strcasecmp( optarg, "tiled_starpu" ) == 0 ) {
                printf( "Test tiled StarPU version\n" );
                starpu = 1;
                tested_tiled_dgetrf = my_dgetrf_tiled_starpu;
                tested_tiled_dplrnt = dplrnt_tiled_starpu;
            }
#endif
            else {
                printf( "Test Seq version\n" );
                tested_dgetrf = dgetrf_seq;
            }
            break;

        case '?': /* error from getopt[_long] */
            exit(1);
            break;

        default:
            print_usage();
            exit(1);
        }
    }

#if defined(ENABLE_STARPU)
    if ( starpu ) {
        my_starpu_init();
    }
#endif

    if ( tested_dgetrf ) {
        testall_dgetrf( tested_dgetrf );
    }
    else if ( tested_tiled_dgetrf ) {
        testall_dgetrf_tiled( tested_tiled_dplrnt, tested_tiled_dgetrf );
    }

#if defined(ENABLE_STARPU)
    if ( starpu ) {
        my_starpu_exit();
    }
#endif

    return EXIT_SUCCESS;
}

