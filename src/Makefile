# Vulkan RT-MMC Makefile
# Usage:
#   make            — build host + shader (portable, CAS atomic)
#   make nvidia     — build host + shader with native atomicAdd(float)
#   make shader     — compile shader only (portable)
#   make host       — compile host only
#   make clean      — remove build artifacts
#   make listgpu    — list available Vulkan GPUs
#   make run        — run with default input

# ==================== Configuration ====================

CXX      ?= g++
CXXFLAGS  = -std=c++11 -O2 -Wall -I. -Iminiz
LDFLAGS   = -lvulkan

GLSLC    ?= glslangValidator
GLSLFLAGS = --target-env vulkan1.2 -e main

BINARY    = vkmmc
SHADER    = vkmmc_core.spv
SOURCES   = vkmmc.cpp miniz/miniz.c
HEADERS   = vkmmc_io.h vkmmc_shapes.h nlohmann/json.hpp
COMP_SRC  = vkmmc_core.comp

# Default input for 'make run'
INPUT    ?= input.json
NPHOTON  ?= 100000
GPUID    ?= 1

# ==================== Targets ====================

.PHONY: all nvidia host shader clean listgpu run help

all: $(BINARY) $(SHADER)

nvidia: $(BINARY) shader-nvidia

help:
	@echo "Vulkan RT-MMC Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make            build host + portable shader"
	@echo "  make nvidia     build host + NVIDIA shader (native atomicAdd float)"
	@echo "  make host       build host binary only"
	@echo "  make shader     build portable shader only"
	@echo "  make clean      remove build artifacts"
	@echo "  make listgpu    list Vulkan GPU devices"
	@echo "  make run        run simulation (INPUT=file NPHOTON=n GPUID=id)"
	@echo ""
	@echo "Variables:"
	@echo "  CXX=$(CXX)  GLSLC=$(GLSLC)"
	@echo "  INPUT=$(INPUT)  NPHOTON=$(NPHOTON)  GPUID=$(GPUID)"

# ---- Host binary ----

$(BINARY): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES) $(LDFLAGS)
	@echo "Built: $@"

host: $(BINARY)

# ---- Shader (portable — CAS atomic fallback) ----

$(SHADER): $(COMP_SRC)
	$(GLSLC) $(GLSLFLAGS) -o $@ $<
	@echo "Built: $@ (portable)"

shader: $(SHADER)

# ---- Shader (NVIDIA — native atomicAdd float) ----

shader-nvidia: $(COMP_SRC)
	sed 's|^//#define USE_ATOMIC_FLOAT|#define USE_ATOMIC_FLOAT|' $< > _mc_nvidia.comp
	$(GLSLC) $(GLSLFLAGS) -o $(SHADER) _mc_nvidia.comp
	@rm -f _mc_nvidia.comp
	@echo "Built: $(SHADER) (NVIDIA atomicAdd float)"

# ---- Utilities ----

listgpu: $(BINARY)
	./$(BINARY) -L

run: $(BINARY) $(SHADER)
	./$(BINARY) -f $(INPUT) -n $(NPHOTON) -G $(GPUID)

clean:
	rm -f $(BINARY) $(SHADER) _mc_nvidia.comp *.jdat *.o

# ---- Dependencies ----

miniz/miniz.c:
	@echo "Error: miniz.c not found. Download from https://github.com/richgel999/miniz"
	@echo "  mkdir -p miniz"
	@echo "  curl -sL https://raw.githubusercontent.com/richgel999/miniz/master/miniz.c -o miniz/miniz.c"
	@echo "  curl -sL https://raw.githubusercontent.com/richgel999/miniz/master/miniz.h -o miniz/miniz.h"
	@exit 1

nlohmann/json.hpp:
	@echo "Error: nlohmann/json.hpp not found. Download with:"
	@echo "  mkdir -p nlohmann"
	@echo "  curl -sL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -o nlohmann/json.hpp"
	@exit 1
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
	   "*.h" "*.cpp" "*.comp"

.PHONY: clean
