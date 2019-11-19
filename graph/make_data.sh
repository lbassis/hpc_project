#!/usr/bin/env bash

#./bin/test_perf.exe > data/ddot.data
./bin/test_perf_gemm_cache.exe > data/gemm_cache.data
R CMD BATCH graph/display.R gemm_cache
