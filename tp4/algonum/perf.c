#include "algonum.h"
#include "perf.h"
#include <stdio.h>
#include <stdlib.h>
#if defined(ENABLE_MPI)
#include <mpi.h>
#endif

void perf(perf_t *p) {
#if defined(ENABLE_MPI)
    MPI_Barrier( MPI_COMM_WORLD );
#endif
    gettimeofday( p, NULL );
}

void perf_diff(const perf_t *begin, perf_t *end) {
    end->tv_sec = end->tv_sec - begin->tv_sec;
    end->tv_usec = end->tv_usec - begin->tv_usec;
    if (end->tv_usec < 0) {
	(end->tv_sec)--;
	end->tv_usec += 1.e6;
    }
}

void perf_printh(const perf_t *p) {
    long m = p->tv_sec / 60;
    long s = p->tv_sec - m * 60;
    long ms = p->tv_usec / 1.e3;
    long micros = p->tv_usec - ms * 1.e3;

    //  printf("%ld sec %ld usec\n", p->tv_sec, p->tv_usec);
    printf("%ld:%ld:%ld:%ld\n", m, s, ms, micros);
}

void perf_printmicro(const perf_t *p) {
    printf("%le\n", p->tv_usec + (p->tv_sec * 1.e6));
}

double perf_mflops(const perf_t *p, const double nb_op) {
    return nb_op / (p->tv_sec * 1.e6 + p->tv_usec);
}

double perf_gflops(const perf_t *p, const double nb_op) {
    return (nb_op / (p->tv_sec * 1.e6 + p->tv_usec)) * 1.e-3;
}
