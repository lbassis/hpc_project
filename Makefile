CC = gcc

TESTS_SRC = $(notdir $(wildcard src/tst/*.c))
BIN_TESTS = $(addprefix bin/tst/,$(TESTS_SRC:.c=.exe))

PERF_SRC = $(notdir $(wildcard src/perf/*.c))
BIN_PERF = $(addprefix bin/perf/,$(PERF_SRC:.c=.exe)) bin/perf/getrf_split.exe

BIN = $(BIN_TESTS) $(BIN_PERF)

LIB_SRC = $(wildcard src/mylib/*.c)
LIB_OBJ = $(addprefix obj/mylib/,$(notdir $(LIB_SRC:.c=.o)))
LIB_PERF_OBJ = $(addprefix obj/mylibperf/,$(notdir $(LIB_SRC:.c=.o)))

UTILS_SRC = $(wildcard src/utilities/*.c)
UTILS_OBJ = $(addprefix obj/utilities/,$(notdir $(UTILS_SRC:.c=.o)))

LIB_DIR = lib2

.PHONY: default
default: start lib $(BIN)
	@echo -e '\033[0;36mCompilation successful \033[0m'

start:
	@echo -e '\033[0;36mStart of compilation \033[0m'
	rm -f $(LIB_DIR)/*.so.

.PHONY: install uninstall
install:
	mkdir -p bin bin/tst bin/perf $(LIB_DIR) obj obj/mylib obj/mylibperf obj/utilities obj/tst obj/perf pdf data

uninstall:
	rm -rf bin bin/tst bin/perf $(LIB_DIR) obj obj/mylib obj/mylibperf obj/utilities obj/tst obj/perf pdf data

CFLAGS = -O3 -Wall -Wextra
CFLAGS += -I./headers
#CFLAGS += -I/home/cisd-simonin/myblas
CFLAGS +=  -DMKL_ILP64 -m64 -I${MKLROOT}/include
CFLAGS += -fopenmp

LDLIBS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
#LDLIBS += -L/home/cisd-simonin/myblas/build

#bin/test_trsm: obj/test_trsm.o $(UTILS_OBJ) lib/mylib.so
#	$(CC) -o $@ $(CFLAGS) $^ $(LDLIBS)


obj/mylibperf/%.o: src/mylib/%.c
	@$(CC) -o $@ $(CFLAGS) -c $< -fPIC -DPERF

obj/mylib/%.o: src/mylib/%.c
	@$(CC) -o $@ $(CFLAGS) -c $< -fPIC

obj/utilities/%.o: src/utilities/%.c
	@$(CC) -o $@ $(CFLAGS) -c $<

obj/tst/%.o: src/tst/%.c
	@$(CC) -o $@ $(CFLAGS) -c $<

obj/perf/getrf_split.o: src/getrf_split.c
	@$(CC) -o $@ $(CFLAGS) $^ $(LDLIBS)

obj/perf/%.o: src/perf/%.c
	@$(CC) -o $@ $(CFLAGS) -c $<

bin/tst/%.exe: obj/tst/%.o $(UTILS_OBJ) $(LIB_DIR)/libmyblas.so
	@$(CC) -o $@ $(CFLAGS) $^ $(LDLIBS)


bin/perf/getrf_split.exe: obj/perf/getrf_split.o $(UTILS_OBJ) $(LIB_DIR)/libmyblasperf.so
	@$(CC) -o $@ $(CFLAGS) $^ $(LDLIBS)

bin/perf/%.exe: obj/perf/%.o $(UTILS_OBJ) $(LIB_DIR)/libmyblas.so
	@$(CC) -o $@ $(CFLAGS) $^ $(LDLIBS)


$(LIB_DIR)/libmyblas.so: $(LIB_OBJ)
	@$(CC) $(CFLAGS) -shared -o $@ $^ $(LDLIBS)

$(LIB_DIR)/libmyblasperf.so: $(LIB_PERF_OBJ)
	@$(CC) $(CFLAGS) -shared -o $@ $^ $(LDLIBS)


### COMMANDS ###

.PHONY: lib test check graph
lib: $(LIB_DIR)/libmyblas.so $(LIB_DIR)/libmyblasperf.so

graph: lib $(BIN_PERF)
	@for perf in $(basename $(notdir $(BIN_PERF))); do \
		echo $$perf ; \
		./bin/perf/$$perf.exe > data/$$perf.data; \
		Rscript graph/gen_graph.R data/$$perf.data "" n Mflops pdf/$$perf\_flop.pdf ; \
		Rscript graph/gen_graph.R data/$$perf.data "" n us pdf/$$perf\_time.pdf ; \
	done

test: $(BIN_TESTS)
	for test in $(BIN_TESTS); do \
		./$$test ; \
	done


.PHONY: clean clean_deps clean_graph clean_all

clean_all: clean clean_graph clean_lib

clean:
	rm -f bin/*.exe bin/tst/*.exe bin/perf/*.exe obj/*.o obj/mylib/*.o obj/utilities/*.o

clean_lib:
	rm -f $(LIB_DIR)/*.so

clean_graph:
	rm -f data/*.data pdf/*.pdf
