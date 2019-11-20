CC = gcc

PROGRAMS = $(notdir $(wildcard src/tst/*.c))
BIN = $(addprefix bin/,$(PROGRAMS:.c=.exe))

LIB_SRC = $(wildcard src/mylib/*.c)
LIB_OBJ = $(addprefix obj/mylib/,$(notdir $(LIB_SRC:.c=.o)))

UTILS_SRC = $(wildcard src/utilities/*.c)
UTILS_OBJ = $(addprefix obj/utilities/,$(notdir $(UTILS_SRC:.c=.o)))

LIB_DIR = lib2

.PHONY: default
default: start lib $(BIN)
	@echo -e '\033[0;36mCompilation successful \033[0m'

start:
	@echo -e '\033[0;36mStart of compilation \033[0m'
	rm -f $(LIB_DIR)/libmyblas.so.

.PHONY: install uninstall
install:
	mkdir -p bin $(LIB_DIR) obj obj/mylib obj/utilities pdf data

uninstall:
	rm -rf bin $(LIB_DIR) obj obj/mylib obj/utilities pdf data

CFLAGS = -O3 -Wall -Wextra
CFLAGS += -I./headers
#CFLAGS += -I/home/cisd-simonin/myblas
CFLAGS +=  -DMKL_ILP64 -m64 -I${MKLROOT}/include
CFLAGS += -fopenmp

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

bin/%.exe: obj/%.o $(UTILS_OBJ) $(LIB_DIR)/libmyblas.so
	@$(CC) -o $@ $(CFLAGS) $^ $(LDLIBS)

$(LIB_DIR)/libmyblas.so: $(LIB_OBJ)
	@$(CC) $(CFLAGS) -shared -o $(LIB_DIR)/libmyblas.so $^ $(LDLIBS)

.PHONY: lib
lib: $(LIB_DIR)/libmyblas.so

.PHONY: graph
graph:
	bash graph/make_data.sh

.PHONY: clean clean_deps clean_graph clean_all

clean_all: clean clean_graph

clean:
	rm -f bin/* obj/*.o

clean_graph:
	rm -f data/*.data pdf/*.pdf
