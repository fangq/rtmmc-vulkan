# Makefile for vkmmc_optix

CUDA_DIR = /usr/local/cuda
OPTIX_DIR = /usr/local/optix
SRC_DIR = ../src

NVCC = /usr/bin/nvcc
CXX = g++

NVCC_FLAGS = -ptx --expt-relaxed-constexpr --use_fast_math -std=c++11
NVCC_INCLUDES = -I$(OPTIX_DIR)/include -I$(OPTIX_DIR)/SDK

CXX_FLAGS = -std=c++11 -O2
CXX_INCLUDES = -I$(SRC_DIR) -I$(SRC_DIR)/miniz -I$(OPTIX_DIR)/include -I$(CUDA_DIR)/include
CXX_LIBS = -L$(CUDA_DIR)/lib64 -lcudart -ldl

PTX = vkmmc_optix_core.ptx
EXEC = vkmmc_optix

all: $(EXEC)

$(PTX): vkmmc_optix_core.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDES) -o $@ $<

$(EXEC): vkmmc_optix.cpp $(PTX) $(SRC_DIR)/miniz/miniz.c
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDES) -o $@ vkmmc_optix.cpp $(SRC_DIR)/miniz/miniz.c $(CXX_LIBS)

clean:
	rm -f $(EXEC) $(PTX)

pretty:
	astyle \
	    --style=attach \
	    --indent=spaces=4 \
	    --indent-modifiers \
	    --indent-switches \
	    --indent-preproc-block \
	    --indent-preproc-define \
	    --indent-col1-comments \
	    --pad-oper \
	    --pad-header \
	    --align-pointer=type \
	    --align-reference=type \
	    --add-brackets \
	    --convert-tabs \
	    --lineend=linux \
	    --preserve-date \
	    --suffix=none \
	    --formatted \
	    --break-blocks \
	   "*.cpp" "*.cu"

.PHONY: all clean pretty