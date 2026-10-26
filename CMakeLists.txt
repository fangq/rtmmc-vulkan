#################################################################
#  CMake build for vkmmc (Vulkan) and vkmmc_optix (OptiX)
#
#    cmake -B build [options]
#    cmake --build build -j
#    ctest --test-dir build -LE gpu       # tests that need no GPU
#    ctest --test-dir build               # all tests, incl. the GPU suites
#
#  Options:
#    -DVKMMC_BUILD_VULKAN=ON|OFF    the Vulkan backend (needs Vulkan, glslangValidator)
#    -DVKMMC_BUILD_OPTIX=AUTO|ON|OFF  the OptiX backend (needs CUDA and the OptiX
#                                   headers; AUTO builds it when both are found)
#    -DOPTIX_ROOT=dir               OptiX headers: dir/include/optix.h, e.g. a
#                                   clone of https://github.com/NVIDIA/optix-dev
#    -DVKMMC_SHOW=ON                the --show viewer (needs GLFW 3)
#    -DVKMMC_USE_SYSTEM_ZLIB=ON     link zlib instead of the bundled miniz
#    -DVKMMC_STATIC_RUNTIME=ON      link libstdc++/libgcc statically
#    -DVKMMC_BUILD_TESTS=OFF        skip the unit tests
#
#  The Makefiles in src/ and optix/ build the same binaries.
#################################################################

cmake_minimum_required(VERSION 3.17)

project(vkmmc VERSION 0.5.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

option(VKMMC_BUILD_VULKAN "build the Vulkan backend (vkmmc)" ON)
set(VKMMC_BUILD_OPTIX AUTO CACHE STRING "build the OptiX backend (vkmmc_optix): AUTO, ON or OFF")
set_property(CACHE VKMMC_BUILD_OPTIX PROPERTY STRINGS AUTO ON OFF)
option(VKMMC_SHOW "build the --show viewer into vkmmc (needs GLFW 3)" OFF)
option(VKMMC_USE_SYSTEM_ZLIB "link the system zlib instead of the bundled miniz" OFF)
option(VKMMC_STATIC_RUNTIME "link libstdc++ and libgcc statically" OFF)
option(VKMMC_BUILD_TESTS "build the unit tests and register the test suites" ON)

set(OPTIX_ROOT "$ENV{OPTIX_ROOT}" CACHE PATH "OptiX SDK or optix-dev clone (containing include/optix.h)")

set(SRC ${PROJECT_SOURCE_DIR}/src)
set(GEN ${PROJECT_BINARY_DIR}/generated)
set(EMBED ${PROJECT_SOURCE_DIR}/cmake/embed_files.cmake)
file(MAKE_DIRECTORY ${GEN})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

find_package(Threads REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(VKMMC_WARNINGS -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
endif()

if(MSVC)
    # the JSON library needs more object sections; silence the "unsafe CRT" notes
    add_compile_options(/bigobj /utf-8)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE NOMINMAX)
    set(VKMMC_WARNINGS /W3)
endif()

# ---------------------------------------------------------------
# zmat (zlib/gzip only) with miniz or the system zlib
# ---------------------------------------------------------------
add_library(vkmmc_zmat STATIC ${SRC}/zmat/zmatlib.c)
target_include_directories(vkmmc_zmat PUBLIC ${SRC}/zmat)
target_compile_definitions(vkmmc_zmat PRIVATE NO_LZMA NO_LZ4 NO_BLOSC2 NO_ZSTD)
target_link_libraries(vkmmc_zmat PUBLIC Threads::Threads)
set_target_properties(vkmmc_zmat PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(VKMMC_USE_SYSTEM_ZLIB)
    find_package(ZLIB REQUIRED)
    target_link_libraries(vkmmc_zmat PUBLIC ZLIB::ZLIB)
else()
    target_sources(vkmmc_zmat PRIVATE ${SRC}/zmat/miniz/miniz.c)
    target_include_directories(vkmmc_zmat PRIVATE ${SRC}/zmat/miniz)
    target_compile_definitions(vkmmc_zmat PRIVATE NO_ZLIB)
endif()

# the host code shared by both backends (header-only)
add_library(vkmmc_host INTERFACE)
target_include_directories(vkmmc_host INTERFACE ${SRC})
target_link_libraries(vkmmc_host INTERFACE vkmmc_zmat)

function(vkmmc_static_runtime target)
    if(VKMMC_STATIC_RUNTIME AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_link_options(${target} PRIVATE -static-libstdc++ -static-libgcc)
    endif()
endfunction()

# ---------------------------------------------------------------
# Vulkan backend: compile every kernel variant to SPIR-V and embed it
# ---------------------------------------------------------------
if(VKMMC_BUILD_VULKAN)
    find_package(Vulkan REQUIRED)
    find_program(GLSLANG_VALIDATOR NAMES glslangValidator glslang
                 HINTS ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} $ENV{VULKAN_SDK}/bin REQUIRED)

    set(GLSLFLAGS --target-env vulkan1.2 -e main)
    set(COMP ${SRC}/vkmmc_core.comp)
    set(SPVS "")

    # variant name, then its defines
    foreach(v mesh csg curv)
        if(v STREQUAL "mesh")
            set(defs "")
        elseif(v STREQUAL "csg")
            set(defs -DUSE_CSG)
        else()
            set(defs -DUSE_CSG -DUSE_CURVATURE)
        endif()

        foreach(atom "" "_i64")
            set(spv ${GEN}/vkmmc_core_${v}${atom}.spv)
            set(extra "")

            if(atom STREQUAL "_i64")
                set(extra -DUSE_ATOMIC_INT64)
            endif()

            add_custom_command(OUTPUT ${spv}
                COMMAND ${GLSLANG_VALIDATOR} ${GLSLFLAGS} ${defs} ${extra} -o ${spv} ${COMP}
                DEPENDS ${COMP} COMMENT "SPIR-V: vkmmc_core_${v}${atom}.spv" VERBATIM)
            list(APPEND SPVS ${spv})
        endforeach()
    endforeach()

    add_custom_command(OUTPUT ${GEN}/vkmmc_core_spv.h
        COMMAND ${CMAKE_COMMAND} -DOUTPUT=${GEN}/vkmmc_core_spv.h "-DINPUTS=${SPVS}"
                "-DQUALIFIER=static const" -DALIGN=1 -P ${EMBED}
        DEPENDS ${SPVS} ${EMBED} COMMENT "embedding the Vulkan kernels" VERBATIM)

    add_executable(vkmmc ${SRC}/vkmmc.cpp ${GEN}/vkmmc_core_spv.h)
    target_include_directories(vkmmc PRIVATE ${GEN})
    target_link_libraries(vkmmc PRIVATE vkmmc_host Vulkan::Vulkan)
    target_compile_options(vkmmc PRIVATE ${VKMMC_WARNINGS})
    vkmmc_static_runtime(vkmmc)

    if(VKMMC_SHOW)
        find_package(glfw3 3.3 QUIET)

        if(NOT glfw3_FOUND)
            find_package(PkgConfig REQUIRED)
            pkg_check_modules(GLFW REQUIRED IMPORTED_TARGET glfw3)
            add_library(glfw ALIAS PkgConfig::GLFW)
        endif()

        set(SHOWHDRS "")

        foreach(s vert:vert frag:frag vol_vert:vert vol_frag:frag)
            string(REPLACE ":" ";" parts ${s})
            list(GET parts 0 name)
            list(GET parts 1 stage)
            string(REPLACE "_vert" ".vert" fname "vkmmc_show_${name}")
            string(REPLACE "_frag" ".frag" fname "${fname}")
            set(spv ${GEN}/vkmmc_show_${name}.spv)
            add_custom_command(OUTPUT ${spv}
                COMMAND ${GLSLANG_VALIDATOR} ${GLSLFLAGS} -S ${stage} -o ${spv} ${SRC}/${fname}
                DEPENDS ${SRC}/${fname} VERBATIM)
            add_custom_command(OUTPUT ${GEN}/vkmmc_show_${name}_spv.h
                COMMAND ${CMAKE_COMMAND} -DOUTPUT=${GEN}/vkmmc_show_${name}_spv.h -DINPUTS=${spv} -DALIGN=1 -P ${EMBED}
                DEPENDS ${spv} ${EMBED} VERBATIM)
            list(APPEND SHOWHDRS ${GEN}/vkmmc_show_${name}_spv.h)
        endforeach()

        target_sources(vkmmc PRIVATE ${SHOWHDRS})
        target_compile_definitions(vkmmc PRIVATE USE_SHOW)
        target_link_libraries(vkmmc PRIVATE glfw)
    endif()

    install(TARGETS vkmmc RUNTIME DESTINATION bin)
endif()

# ---------------------------------------------------------------
# OptiX backend: PTX for every kernel variant, embedded; static CUDA runtime
# ---------------------------------------------------------------
set(BUILD_OPTIX OFF)

if(NOT VKMMC_BUILD_OPTIX STREQUAL "OFF")
    find_package(CUDAToolkit QUIET)
    find_path(OPTIX_INCLUDE_DIR optix.h HINTS ${OPTIX_ROOT}/include /usr/local/optix/include)

    if(CUDAToolkit_FOUND AND OPTIX_INCLUDE_DIR)
        set(BUILD_OPTIX ON)
    elseif(VKMMC_BUILD_OPTIX STREQUAL "ON")
        message(FATAL_ERROR "VKMMC_BUILD_OPTIX=ON needs the CUDA toolkit and the OptiX headers (set OPTIX_ROOT)")
    else()
        message(STATUS "vkmmc_optix: skipped (CUDA toolkit or OptiX headers not found; set OPTIX_ROOT)")
    endif()
endif()

if(BUILD_OPTIX)
    message(STATUS "vkmmc_optix: OptiX headers in ${OPTIX_INCLUDE_DIR}, nvcc ${CUDAToolkit_NVCC_EXECUTABLE}")
    set(CU ${PROJECT_SOURCE_DIR}/optix/vkmmc_optix_core.cu)
    set(PTXS "")

    foreach(v mesh csg curv)
        if(v STREQUAL "mesh")
            set(defs "")
        elseif(v STREQUAL "csg")
            set(defs -DUSE_CSG)
        else()
            set(defs -DUSE_CSG -DUSE_CURVATURE)
        endif()

        set(ptx ${GEN}/vkmmc_optix_${v}.ptx)
        add_custom_command(OUTPUT ${ptx}
            COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -ptx --expt-relaxed-constexpr --use_fast_math -std=c++11
                    -I${OPTIX_INCLUDE_DIR} ${defs} -o ${ptx} ${CU}
            DEPENDS ${CU} ${PROJECT_SOURCE_DIR}/optix/vkmmc_vec_math.h COMMENT "PTX: vkmmc_optix_${v}.ptx" VERBATIM)
        list(APPEND PTXS ${ptx})
    endforeach()

    add_custom_command(OUTPUT ${GEN}/vkmmc_optix_ptx.h
        COMMAND ${CMAKE_COMMAND} -DOUTPUT=${GEN}/vkmmc_optix_ptx.h "-DINPUTS=${PTXS}" "-DQUALIFIER=static const" -P ${EMBED}
        DEPENDS ${PTXS} ${EMBED} COMMENT "embedding the OptiX kernels" VERBATIM)

    add_executable(vkmmc_optix ${PROJECT_SOURCE_DIR}/optix/vkmmc_optix.cpp ${GEN}/vkmmc_optix_ptx.h)
    target_include_directories(vkmmc_optix PRIVATE ${GEN} ${OPTIX_INCLUDE_DIR})
    target_link_libraries(vkmmc_optix PRIVATE vkmmc_host CUDA::cudart_static ${CMAKE_DL_LIBS})
    target_compile_options(vkmmc_optix PRIVATE ${VKMMC_WARNINGS})
    vkmmc_static_runtime(vkmmc_optix)
    install(TARGETS vkmmc_optix RUNTIME DESTINATION bin)
endif()

# ---------------------------------------------------------------
# Tests: unit tests (no GPU); GPU suites carry the label "gpu"
# ---------------------------------------------------------------
if(VKMMC_BUILD_TESTS)
    enable_testing()
    add_executable(vkmmc_unittest ${PROJECT_SOURCE_DIR}/tests/unit/test_vkmmc_unit.cpp)
    target_link_libraries(vkmmc_unittest PRIVATE vkmmc_host)
    add_test(NAME unit COMMAND vkmmc_unittest)

    find_program(BASH bash)
    find_package(Python3 COMPONENTS Interpreter)

    if(BASH AND Python3_FOUND)
        if(TARGET vkmmc)
            add_test(NAME gpu_vulkan COMMAND ${BASH} ${PROJECT_SOURCE_DIR}/tests/test_vkmmc.sh $<TARGET_FILE:vkmmc>)
            set_tests_properties(gpu_vulkan PROPERTIES LABELS gpu)
        endif()

        if(TARGET vkmmc_optix)
            add_test(NAME gpu_optix COMMAND ${BASH} ${PROJECT_SOURCE_DIR}/tests/test_vkmmc.sh $<TARGET_FILE:vkmmc_optix>)
            set_tests_properties(gpu_optix PROPERTIES LABELS gpu)
        endif()
    endif()

    # front-end smoke tests that stop before the GPU
    foreach(t vkmmc vkmmc_optix)
        if(TARGET ${t})
            add_test(NAME ${t}_version COMMAND ${t} -v)
            add_test(NAME ${t}_dumpjson COMMAND ${t} ${PROJECT_SOURCE_DIR}/tests/fuzz/regressions/input-flat-lens-cap-timeout-1436.json
                                               --dumpjson)
        endif()
    endforeach()
endif()
