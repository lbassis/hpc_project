#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <strings.h>
#include "algonum.h"
#if defined(ENABLE_STARPU)
#include "codelets.h"
#endif

dgemm_fct_t dgemm_seq = my_dgemm_seq;
dgemm_fct_t dgemm_omp = NULL;

#define GETOPT_STRING "hv:M:N:K:AB"
static struct option long_options[] =
{
    {"help",          no_argument,       0,      'h'},
    {"v",             required_argument, 0,      'v'},
    // Matrix parameters
    {"M",             required_argument, 0,      'M'},
    {"N",             required_argument, 0,      'N'},
    {"K",             required_argument, 0,      'K'},
    {"nb",            required_argument, 0,      'b'},
    // Check/prints
    {"transA",        no_argument,       0,      'A'},
    {"transB",        no_argument,       0,      'B'},
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
            "  -K x       Set the K value\n"
            "  -b x       Set the block size b value\n"
            "  -A         Switch transA to CblasTrans\n"
            "  -B         Switch transB to CblasTrans\n" );
    exit(1);
}

int main( int argc, char **argv )
{
    dgemm_fct_t        tested_dgemm = NULL;
    dgemm_tiled_fct_t  tested_tiled_dgemm = NULL;
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
                tested_dgemm = cblas_dgemm;
            }
            else if ( strcasecmp( optarg, "seq" ) == 0 ) {
                printf( "Test Sequential version\n" );
                tested_dgemm = my_dgemm_seq;
            }
            else if ( strcasecmp( optarg, "scal_omp" ) == 0 ) {
                printf( "Test Scalar OpenMP version\n" );
                tested_dgemm = my_dgemm_scal_openmp;
            }
            else if ( strcasecmp( optarg, "bloc_omp" ) == 0 ) {
                printf( "Test block OpenMP version\n" );
                tested_dgemm = my_dgemm_bloc_openmp;
            }
            else if ( strcasecmp( optarg, "tiled_omp" ) == 0 ) {
                printf( "Test tiled OpenMP version\n" );
                tested_tiled_dgemm = my_dgemm_tiled_openmp;
                tested_tiled_dplrnt = dplrnt_tiled;
            }
#if defined(ENABLE_STARPU)
            else if ( strcasecmp( optarg, "tiled_starpu" ) == 0 ) {
                printf( "Test tiled StarPU version\n" );
                starpu = 1;
                tested_tiled_dgemm = my_dgemm_tiled_starpu;
                tested_tiled_dplrnt = dplrnt_tiled_starpu;
            }
#endif
            else {
                printf( "Test Seq version\n" );
                tested_dgemm = my_dgemm_seq;
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

    if ( tested_dgemm ) {
        testall_dgemm( tested_dgemm );
    }
    else if ( tested_tiled_dgemm ) {
        testall_dgemm_tiled( tested_tiled_dplrnt, tested_tiled_dgemm );
    }

#if defined(ENABLE_STARPU)
    if ( starpu ) {
        my_starpu_exit();
    }
#endif

    return EXIT_SUCCESS;
}

