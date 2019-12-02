echo "version,threads,Mflops/s" > scalability.data
for i in 1 2 4 8 16 32
do
    export OMP_NUM_THREADS=$i
    ./bin/perf/scalability_gemm.exe $i >> scalability_gemm.data
done
