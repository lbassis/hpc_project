
PROGRAMS = tp1 driver test_perf
CC = gcc

bin_prog = bin/tp1 bin/driver bin/test_perf
# Must be the first rule
.PHONY: default
default: $(bin_prog)



CFLAGS = -O3 -Wall -Wextra
CFLAGS += -I./headers
#CFLAGS +=  -DMKL_ILP64 -m64 -I${MKLROOT}/include

#LDLIBS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl




obj/%.o: src/%.c
	$(CC) -o $@ $(CFLAGS) -c $<

dep/%.d: src/%.c
	$(CC) $(CFLAGS) -MM $< | sed -e 's|\(.*\)\.o:|bin/\1:|g' | sed -e 's|\.c |\.o |g' | sed -e 's|\.h|\.o|g' \
	| sed -e 's|headers|obj|g' | sed -e 's|src|obj|g' > $@

deps:
	for p in ${PROGRAMS} ; do \
		$(CC) $(CFLAGS) -MM src/$$p.c | sed -e 's|\(.*\)\.o:|bin/\1:|g' | sed -e 's|\.c |\.o |g' | sed -e 's|\.h|\.o|g' \
		| sed -e 's|headers|obj|g' | sed -e 's|src|obj|g' > dep/$$p.d ; \
		echo '	$$(CC) -o $$@ $$^ $$(LDLIBS)' >> dep/$$p.d ; \
	done

-include $(wildcard dep/*.d)

.PHONY: clean clean_deps
clean:
	rm -f bin/* obj/*.o

clean_deps:
	rm -f dep/*.d
