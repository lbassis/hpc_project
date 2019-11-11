#!/usr/bin/env bash

#./bin/test_perf 10000000000 > data/ddot.data
./bin/test_perf > data/ddot.data
./bin/test_perf_my_dgemm_scalaire > data/my_dgemm_scalaire.data

R CMD BATCH graph/display.R
