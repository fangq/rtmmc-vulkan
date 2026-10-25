#################################################################
#  Host-side unit tests for vkmmc (no GPU needed)
#
#  make            build ./test_vkmmc_unit
#  make run        build and run the tests
#  make asan       build with AddressSanitizer + UBSan and run
#  make coverage   build with --coverage -O0, run, print gcov line coverage
#                  of the vkmmc_*.h headers
#  make clean      remove build products
#
#  Run from the repository root with: make -C tests/unit run
#################################################################

CXX      ?= g++
CC       ?= gcc
GCOV     ?= gcov

SRCDIR   := ../../src
ZMATDIR  := $(SRCDIR)/zmat

TEST_SRC := test_vkmmc_unit.cpp
HEADERS  := $(SRCDIR)/vkmmc_io.h $(SRCDIR)/vkmmc_shapes.h $(SRCDIR)/vkmmc_curvature.h $(SRCDIR)/vkmmc_host.h

WARN     := -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-unused-function
CXXINC   := -I$(SRCDIR) -I$(ZMATDIR)
ZMATDEFS := -DNO_LZMA -DNO_LZ4 -DNO_BLOSC2 -DNO_ZSTD -DNO_ZLIB -I$(ZMATDIR) -I$(ZMATDIR)/miniz
LDLIBS   := -lpthread

# build flavors: each has its own object/binary names so they can coexist
OPT_release  := -O2 -g
OPT_asan     := -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined
OPT_coverage := -O0 -g --coverage

FLAVOR   ?= release
SUFFIX_release  :=
SUFFIX_asan     := _asan
SUFFIX_coverage := _cov
SFX      := $(SUFFIX_$(FLAVOR))
OPT      := $(OPT_$(FLAVOR))

BIN      := test_vkmmc_unit$(SFX)
OBJS     := test_vkmmc_unit$(SFX).o zmatlib$(SFX).o miniz$(SFX).o

.PHONY: all run asan coverage clean build

all:
	@$(MAKE) --no-print-directory build FLAVOR=release

build: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(OPT) -o $@ $(OBJS) $(LDLIBS)

test_vkmmc_unit$(SFX).o: $(TEST_SRC) $(HEADERS)
	$(CXX) -std=c++11 $(OPT) $(WARN) $(CXXINC) -c -o $@ $<

zmatlib$(SFX).o: $(ZMATDIR)/zmatlib.c $(ZMATDIR)/zmatlib.h
	$(CC) $(OPT) $(ZMATDEFS) -c -o $@ $<

miniz$(SFX).o: $(ZMATDIR)/miniz/miniz.c $(ZMATDIR)/miniz/miniz.h
	$(CC) $(OPT) -c -o $@ $<

run: all
	./test_vkmmc_unit

asan:
	@$(MAKE) --no-print-directory build FLAVOR=asan
	ASAN_OPTIONS=detect_leaks=1:abort_on_error=0 UBSAN_OPTIONS=print_stacktrace=1 ./test_vkmmc_unit_asan

# the summary is printed even when some tests fail (the run's status is reported)
coverage:
	@rm -f *_cov.gcda
	@$(MAKE) --no-print-directory build FLAVOR=coverage
	-./test_vkmmc_unit_cov
	@echo "==== gcov line coverage of the vkmmc headers ===="
	@$(GCOV) -n -o test_vkmmc_unit_cov.o $(TEST_SRC) 2>/dev/null | \
	    awk '/^File /{f=$$2; gsub(/\x27/, "", f)} \
	         /^Lines executed/{ if (f ~ /vkmmc_(io|shapes|curvature|host)\.h$$/) { n = split(f, p, "/"); printf "  %-20s %s\n", p[n], $$0 } }'

clean:
	rm -f test_vkmmc_unit test_vkmmc_unit_asan test_vkmmc_unit_cov *.o *.gcda *.gcno *.gcov
