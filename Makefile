%.o: 	
	gcc -c util.c ddot.c perf.c -lm
all:	util.o ddot.o perf.o
	gcc tp1.c util.o ddot.o perf.o -o tp1 -lm
clean:
	rm *.o
