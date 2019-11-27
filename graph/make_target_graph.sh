if [[ "$#" -gt 0 ]]; then

  ./bin/perf/$1.exe > data/$1.data
  #Rscript graph/gen_graph.R data/$1.data "" $2 $3 pdf/$1.pdf ; \

fi
