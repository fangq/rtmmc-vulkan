# Vulkan RT-MMC Makefile
#
# Build variants:
#   make                       — mesh-only (fastest, no CSG/curvature)
#   make CSG=1                 — CSG shapes, no curvature
#   make CSG=1 CURV=1          — CSG + curvature
#   make nvidia                — NVIDIA native atomicAdd (combine with CSG/CURV)
#   make CSG=1 CURV=1 nvidia   — full featured NVIDIA build

CXX      ?= g++
CXXFLAGS  = -std=c++11 -O2 -Wall -I. -Iminiz
LDFLAGS   = -lvulkan

GLSLC    ?= glslangValidator
GLSLFLAGS = --target-env vulkan1.2 -e main

BINARY    = vkmmc
SHADER    = vkmmc_core.spv
SOURCES   = vkmmc.cpp miniz/miniz.c
HEADERS   = vkmmc_io.h vkmmc_shapes.h vkmmc_curvature.h nlohmann/json.hpp
COMP_SRC  = vkmmc_core.comp

INPUT    ?= input.json
NPHOTON  ?= 100000
GPUID    ?= 1

# CSG support
ifdef CSG
    GLSLFLAGS += -DUSE_CSG
    CXXFLAGS  += -DUSE_CSG
endif

# Curvature support (requires CSG)
ifdef CURV
    ifndef CSG
        $(error CURV=1 requires CSG=1)
    endif
    GLSLFLAGS += -DUSE_CURVATURE
    CXXFLAGS  += -DUSE_CURVATURE
endif

.PHONY: all nvidia host shader shader-nvidia clean listgpu run help pretty

all: $(BINARY) $(SHADER)

nvidia: $(BINARY) shader-nvidia

help:
	@echo "Vulkan RT-MMC Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make                 mesh-only (fastest)"
	@echo "  make CSG=1           CSG shapes, no curvature"
	@echo "  make CSG=1 CURV=1    CSG + curvature"
	@echo "  make nvidia          NVIDIA atomicAdd (combine with CSG/CURV)"
	@echo "  make host            host binary only"
	@echo "  make shader          shader only"
	@echo "  make clean           remove artifacts"
	@echo "  make listgpu         list Vulkan GPUs"
	@echo "  make run             run simulation"

$(BINARY): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES) $(LDFLAGS)
	@echo "Built: $@"

host: $(BINARY)

# Force shader rebuild when flags change
SHADER_FLAGS_FILE = .shader_flags
CURRENT_FLAGS := $(GLSLFLAGS)

.PHONY: check_flags
check_flags:
	@echo '$(CURRENT_FLAGS)' | cmp -s - $(SHADER_FLAGS_FILE) || echo '$(CURRENT_FLAGS)' > $(SHADER_FLAGS_FILE)

$(SHADER): $(COMP_SRC) check_flags $(SHADER_FLAGS_FILE)
	$(GLSLC) $(GLSLFLAGS) -o $@ $<
	@echo "Built: $@ (flags: $(GLSLFLAGS))"

shader: $(SHADER)

shader-nvidia: $(COMP_SRC)
	sed 's|^//#define USE_ATOMIC_FLOAT|#define USE_ATOMIC_FLOAT|' $< > _mc_nvidia.comp
	$(GLSLC) $(GLSLFLAGS) -o $(SHADER) _mc_nvidia.comp
	@rm -f _mc_nvidia.comp
	@echo "Built: $(SHADER) (NVIDIA atomicAdd float)"

listgpu: $(BINARY)
	./$(BINARY) -L

run: $(BINARY) $(SHADER)
	./$(BINARY) -f $(INPUT) -n $(NPHOTON) -G $(GPUID)

clean:
	rm -f $(BINARY) $(SHADER) _mc_nvidia.comp .shader_flags *.jdat *.o

miniz/miniz.c:
	@echo "Error: miniz.c not found. Download from https://github.com/richgel999/miniz"
	@exit 1

nlohmann/json.hpp:
	@echo "Error: nlohmann/json.hpp not found."
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