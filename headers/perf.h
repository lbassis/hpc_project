#ifndef PERF_H
#define PERF_H
#include <sys/time.h>

typedef struct timeval perf_t;

void perf(perf_t *p);

void perf_diff(const perf_t *begin, perf_t *end);

void perf_printh(const perf_t *p);

void perf_printmicro(const perf_t *p);

double perf_mflops(const perf_t *p, const long long nb_op);

void perf_print_time(const perf_t *p, long div);


void perf_add(const perf_t *begin, perf_t *end);

#endif
