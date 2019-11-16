CC = gcc

PROGRAMS = $(notdir $(wildcard src/tst/*.c))
BIN = $(addprefix bin/,$(PROGRAMS:.c=.exe))

LIB_SRC = $(wildcard src/mylib/*.c)
LIB_OBJ = $(addprefix obj/mylib/,$(notdir $(LIB_SRC:.c=.o)))

UTILS_SRC = $(wildcard src/utilities/*.c)
UTILS_OBJ = $(addprefix obj/utilities/,$(notdir $(UTILS_SRC:.c=.o)))



.PHONY: default
default: $(BIN)


.PHONY: install uninstall
install:
	mkdir -p bin lib obj obj/mylib obj/utilities pdf data

uninstall:
	rm -rf bin lib obj obj/mylib obj/utilities pdf data

CFLAGS = -O3 -Wall -Wextra
CFLAGS += -I./headers
#CFLAGS += -I/home/cisd-simonin/myblas
CFLAGS +=  -DMKL_ILP64 -m64 -I${MKLROOT}/include

LDLIBS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
#LDLIBS += -L/home/cisd-simonin/myblas/build

#bin/test_trsm: obj/test_trsm.o $(UTILS_OBJ) lib/mylib.so
#	$(CC) -o $@ $(CFLAGS) $^ $(LDLIBS)


obj/mylib/%.o: src/mylib/%.c
	@$(CC) -o $@ $(CFLAGS) -c $< -fPIC

obj/utilities/%.o: src/utilities/%.c
	@$(CC) -o $@ $(CFLAGS) -c $<

obj/%.o: src/tst/%.c
	@$(CC) -o $@ $(CFLAGS) -c $<

bin/%.exe: obj/%.o $(UTILS_OBJ) lib/mylib.so
	@$(CC) -o $@ $(CFLAGS) $^ $(LDLIBS)

lib/mylib.so: $(LIB_OBJ)
	@$(CC) $(CFLAGS) -shared -o lib/mylib.so $^ $(LDLIBS)

lib: lib/mylib.so

.PHONY: graph
graph:
	bash graph/make_data.sh

.PHONY: clean clean_deps clean_graph clean_all

clean_all: clean clean_graph

clean:
	rm -f bin/* obj/*.o

clean_graph:
	rm -f data/*.data pdf/*.pdf
