#!/usr/bin/env bash

echo "n, ms" > data
./test_perf >> data
R CMD BATCH display.R
