
PROGRAMS = tp1 driver test_perf test_perf_my_dgemm_scalaire test_getrf test_getf2 #$(basename $(notdir $(wildcard src/*.c)))
CC = gcc

bin_prog = $(addprefix bin/,$(PROGRAMS))
# Must be the first rule
.PHONY: default
default: $(bin_prog)

.PHONY: install uninstall
install:
	mkdir -p bin dep lib obj pdf data

uninstall:
	rm -rf bin dep lib obj pdf data

CFLAGS = -O3 -Wall -Wextra
CFLAGS += -I./headers
CFLAGS +=  -DMKL_ILP64 -m64 -I${MKLROOT}/include

LDLIBS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

deps:
	for p in ${PROGRAMS} ; do \
		$(CC) $(CFLAGS) -MM src/$$p.c | sed -e 's|\(.*\)\.o:|bin/\1:|g' | sed -e 's|\.c |\.o |g' | sed -e 's|\.h|\.o|g' \
		| sed -e 's|headers|obj|g' | sed -e 's|src|obj|g' > dep/$$p.d ; \
		echo '	$$(CC) -o $$@ $$^ $$(LDLIBS)' >> dep/$$p.d ; \
	done

-include $(wildcard dep/*.d)

obj/%.o: src/%.c
	$(CC) -o $@ $(CFLAGS) -c $<

.PHONY: lib
lib: src/ddot.c
	$(CC) $(CFLAGS) -shared -o lib/libmyblas.so -fPIC src/ddot.c


.PHONY: graph
graph:
	bash graph/make_data.sh

.PHONY: clean clean_deps clean_graph clean_all

clean_all: clean clean_deps clean_graph

clean:
	rm -f bin/* obj/*.o

clean_deps:
	rm -f dep/*.d

clean_graph:
	rm -f data/*.data pdf/*.pdf
