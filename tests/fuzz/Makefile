#################################################################
#  libFuzzer harnesses for the vkmmc host code (no GPU needed)
#
#  make               build build/fuzz_{input,jdata,cmdline} (clang, ASan+UBSan)
#  make seeds         (re)generate the seed corpora
#  make run T=300     run each harness for T seconds (default 60) on all cores
#  make regress       replay regressions/ (past findings) through every harness
#  make clean
#
#  clang <= 16 ASan crashes at random on kernels with high mmap ASLR entropy,
#  so the harnesses run under 'setarch -R' (ASLR off) when it is available.
#
#  Crashes, leaks and timeouts are written to findings/; reproduce one with
#  build/fuzz_input findings/crash-<hash>
#################################################################

CXX      := clang++
CC       := clang
SRC      := ../../src
SAN      := -fsanitize=address,undefined,float-cast-overflow -fno-sanitize-recover=undefined,float-cast-overflow
FLAGS    := -g -O1 -fno-omit-frame-pointer $(SAN)
CXXFLAGS := -std=c++11 $(FLAGS) -I$(SRC) -I$(SRC)/zmat
ZMATFLAGS:= $(FLAGS) -DNO_LZMA -DNO_LZ4 -DNO_BLOSC2 -DNO_ZSTD -DNO_ZLIB -I$(SRC)/zmat -I$(SRC)/zmat/miniz
HEADERS  := $(wildcard $(SRC)/vkmmc_*.h) fuzz_common.h
FUZZERS  := fuzz_input fuzz_jdata fuzz_cmdline
T        ?= 60
JOBS     ?= $(shell nproc)

NOASLR   := $(shell setarch -R true 2>/dev/null && echo "setarch -R")

.PHONY: all seeds run regress clean

all: $(addprefix build/,$(FUZZERS))

build/zmatlib.o: $(SRC)/zmat/zmatlib.c
	@mkdir -p build
	$(CC) $(ZMATFLAGS) -c -o $@ $<

build/miniz.o: $(SRC)/zmat/miniz/miniz.c
	@mkdir -p build
	$(CC) $(FLAGS) -c -o $@ $<

build/%: %.cpp $(HEADERS) build/zmatlib.o build/miniz.o
	$(CXX) $(CXXFLAGS) -fsanitize=fuzzer -o $@ $< build/zmatlib.o build/miniz.o -lpthread

seeds:
	rm -rf corpus_input corpus_jdata corpus_cmdline
	python3 mkseeds.py corpus_input corpus_jdata corpus_cmdline

corpus_input:
	python3 mkseeds.py corpus_input corpus_jdata corpus_cmdline

run: all corpus_input
	@mkdir -p findings
	@for f in $(FUZZERS); do \
	    c=corpus_$${f#fuzz_}; d=; [ $$f = fuzz_input ] && d=-dict=vkmmc.dict; \
	    echo "=== $$f for $(T) s"; \
	    UBSAN_OPTIONS=print_stacktrace=1 $(NOASLR) ./build/$$f $$c $$d -max_total_time=$(T) -timeout=10 \
	        -rss_limit_mb=4096 -max_len=65536 -jobs=$(JOBS) -workers=$(JOBS) -artifact_prefix=findings/ \
	        -print_final_stats=1 > findings/$$f.log 2>&1; \
	    grep -h "stat::number_of_executed_units\|ERROR\|SUMMARY" fuzz-*.log findings/$$f.log 2>/dev/null | sort | uniq -c | head -20; \
	    rm -f fuzz-*.log; \
	done
	@ls findings | grep -v "\.log$$" || echo "no findings"

regress: all
	@for f in $(FUZZERS); do $(NOASLR) ./build/$$f -runs=1 regressions/* > build/regress_$$f.log 2>&1 \
	    && echo "$$f: regressions pass" || { echo "$$f: FAILED, see build/regress_$$f.log"; exit 1; }; done

clean:
	rm -rf build findings fuzz-*.log
