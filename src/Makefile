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

BINARY        = vkmmc
SHADER        = vkmmc_core.spv
SHADER_HEADER = vkmmc_core_spv.h
SOURCES       = vkmmc.cpp miniz/miniz.c
HEADERS       = vkmmc_io.h vkmmc_shapes.h vkmmc_curvature.h nlohmann/json.hpp $(SHADER_HEADER)
COMP_SRC      = vkmmc_core.comp

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

# Interactive viewer support
GLFW_INC  ?= $(shell pkg-config --cflags glfw3 2>/dev/null || echo "-I/usr/local/optix/SDK/support/GLFW/include")
GLFW_LIB  ?= $(shell pkg-config --libs   glfw3 2>/dev/null || echo "-L/usr/local/optix/SDK/lib -Wl,-rpath,/usr/local/optix/SDK/lib -lglfw")
ifdef SHOW
    CXXFLAGS  += -DUSE_SHOW $(GLFW_INC)
    LDFLAGS   += $(GLFW_LIB)
    SHOW_DEPS  = vkmmc_show_vert_spv.h vkmmc_show_frag_spv.h \
                 vkmmc_show_vol_vert_spv.h vkmmc_show_vol_frag_spv.h
    HEADERS   += $(SHOW_DEPS)
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

$(SHADER_HEADER): $(SHADER)
	xxd -i $< | sed 's/^unsigned char/alignas(4) unsigned char/' > $@
	@echo "Built: $@"

vkmmc_show_vert.spv: vkmmc_show.vert
	glslangValidator --target-env vulkan1.2 -e main -S vert -o $@ $<

vkmmc_show_frag.spv: vkmmc_show.frag
	glslangValidator --target-env vulkan1.2 -e main -S frag -o $@ $<

vkmmc_show_vert_spv.h: vkmmc_show_vert.spv
	xxd -i $< | sed 's/^unsigned char/alignas(4) unsigned char/' > $@

vkmmc_show_frag_spv.h: vkmmc_show_frag.spv
	xxd -i $< | sed 's/^unsigned char/alignas(4) unsigned char/' > $@

vkmmc_show_vol_vert.spv: vkmmc_show_vol.vert
	glslangValidator --target-env vulkan1.2 -e main -S vert -o $@ $<

vkmmc_show_vol_frag.spv: vkmmc_show_vol.frag
	glslangValidator --target-env vulkan1.2 -e main -S frag -o $@ $<

vkmmc_show_vol_vert_spv.h: vkmmc_show_vol_vert.spv
	xxd -i $< | sed 's/^unsigned char/alignas(4) unsigned char/' > $@

vkmmc_show_vol_frag_spv.h: vkmmc_show_vol_frag.spv
	xxd -i $< | sed 's/^unsigned char/alignas(4) unsigned char/' > $@

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
	xxd -i $(SHADER) | sed 's/^unsigned char/alignas(4) unsigned char/' > $(SHADER_HEADER)
	@echo "Built: $(SHADER) (NVIDIA atomicAdd float)"

listgpu: $(BINARY)
	./$(BINARY) -L

run: $(BINARY) $(SHADER)
	./$(BINARY) -f $(INPUT) -n $(NPHOTON) -G $(GPUID)

clean:
	rm -f $(BINARY) $(SHADER) $(SHADER_HEADER) _mc_nvidia.comp .shader_flags *.jdat *.o \
	    vkmmc_show_vert.spv vkmmc_show_frag.spv vkmmc_show_vert_spv.h vkmmc_show_frag_spv.h \
	    vkmmc_show_vol_vert.spv vkmmc_show_vol_frag.spv \
	    vkmmc_show_vol_vert_spv.h vkmmc_show_vol_frag_spv.h

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