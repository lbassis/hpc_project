HPC project for the Calcul Intensif et Science des Données Master at ENSEIRB-MATMECA
=====

### make install
Create all necessary folders to the project

### make uninstall
Delete all folders created by 'make install'

### make or make -jxx
Compile all src files including the library, unitary tests et performances tests

### make lib
Compile the only library

### make test
Compile and execute all unitary tests

### make graph
Genarate all the data and generate graphs, this may take awhile
Output will be in pdf/

### make graph_from_data
Generate graphs from the data in data/

### make clean
Delete all object / binary files execpt the library

### make clean_lib
Delete only the library

### make clean_graph
Delete all data and graphs

### make clean_all
Do 'make clean', 'make clean_lib' and 'make clean_graph'

For more details about tp4/ see tp4/README.md
