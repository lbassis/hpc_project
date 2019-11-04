%.o: 	
	gcc -c util.c ddot.c perf.c
all:	%.o
	gcc tp1.c util.o ddot.o perf.o -o tp1
clean:
	rm *.o
