for n in {500..1500..100} 
do
    echo 'op,imp,t' > results$n.data
    for rep in {1..5};
    do
	./bin/perf/getrf_split.exe $n >> results$n.data
    done
done

