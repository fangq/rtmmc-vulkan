/*
 * vkmmc_optix.cpp — OptiX 7 host for surface-based MC photon transport
 *
 * Reuses vkmmc_io.h, vkmmc_shapes.h, vkmmc_curvature.h for JSON I/O,
 * shape triangulation, and curvature computation.
 *
 * Build (adjust paths as needed):
 *   nvcc -ptx -o vkmmc_optix_core.ptx vkmmc_optix_core.cu \
 *        -I/path/to/optix/include -I/path/to/sutil
 *   g++ -std=c++14 -O2 -o vkmmc_optix vkmmc_optix.cpp miniz.c \
 *        -I. -Iminiz -I/path/to/optix/include \
 *        -lcuda -lcudart -loptix
 *
 * Usage: same as vkmmc
 *   ./vkmmc_optix input.json
 *   ./vkmmc_optix -f input.json -n 1e7 -m 48 -c 1
 */

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <sstream>

#include "vkmmc_io.h"
#include "vkmmc_shapes.h"
#include "vkmmc_curvature.h"

/* ================================================================ */
/*                  CUDA / OptiX error checking                     */
/* ================================================================ */

#define CUDA_CHECK(call) do { \
        cudaError_t rc = call; \
        if (rc != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(rc), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

#define OPTIX_CHECK(call) do { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            fprintf(stderr, "OptiX error %d at %s:%d\n", (int)res, __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

/* ================================================================ */
/*                   GPU data structures                            */
/* ================================================================ */

struct Medium_GPU {
    float mua, mus, g, n;
};

struct NodeCurv_GPU {
    float4 vnorm_k1;
    float4 pdir_k2;
    float4 node_pos;
};

#define MAX_PROP 256

struct VKMMCParam {
    OptixTraversableHandle gashandle;
    CUdeviceptr  facebuf;
    CUdeviceptr  curvbuf;
    CUdeviceptr  outputbuf;
    CUdeviceptr  seedbuf;
    int     srctype;
    float3  srcpos, srcdir;
    float4  srcparam1, srcparam2;
    float3  grid_min, grid_extent;
    uint4   grid_stride;
    float   voxel_scale;
    float   tstart, tend;
    float   inv_timestep;
    int     maxgate;
    uint32_t initial_medium;
    uint32_t do_reflect;
    int     output_type;
    uint32_t num_media;
    uint32_t do_csg;
    uint32_t has_curvature;
    int     threadphoton, oddphoton;
    float   minenergy;
    float   roulettesize;
    Medium_GPU media[MAX_PROP];
};

/* SBT records */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

/* Simple CUDA buffer helper */
struct CUDABuf {
    void* ptr = nullptr;
    size_t sz = 0;
    void alloc(size_t n) {
        sz = n;
        CUDA_CHECK(cudaMalloc(&ptr, n));
    }
    void free() {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            sz = 0;
        }
    }
    void upload(const void* src, size_t n) {
        CUDA_CHECK(cudaMemcpy(ptr, src, n, cudaMemcpyHostToDevice));
    }
    void download(void* dst, size_t n) {
        CUDA_CHECK(cudaMemcpy(dst, ptr, n, cudaMemcpyDeviceToHost));
    }
    CUdeviceptr d_ptr() const {
        return (CUdeviceptr)ptr;
    }
    template<typename T> void alloc_upload(const T* data, size_t count) {
        alloc(count * sizeof(T));
        upload(data, count * sizeof(T));
    }
    template<typename T> void alloc_upload(const std::vector<T>& v) {
        alloc_upload(v.data(), v.size());
    }
};

/* ================================================================ */
/*                       CLI parsing                                */
/* ================================================================ */

struct CmdOverrides {
    uint64_t nphoton, batch_size;
    uint32_t rng_seed, totalthread;
    float unitinmm;
    int outputtype, isreflect, isnormalize, gpuid, meshres, docurv;
    bool listgpu, dumpjson, dumpmesh, debugcurv;
    std::string session_id, inputfile;
    CmdOverrides(): nphoton(0), batch_size(UINT64_MAX), rng_seed(0), totalthread(0),
        unitinmm(0), outputtype(-1), isreflect(-1), isnormalize(-1), gpuid(0),
        meshres(0), docurv(-1), listgpu(false), dumpjson(false), dumpmesh(false),
        debugcurv(false) {}
};

void printhelp(const char* n) {
    printf("vkmmc_optix - OptiX ray-tracing accelerated surface MC\n"
           "Usage: %s input.json  OR  %s -f input.json [flags]\n\n"
           "Flags:\n -f/--input\tJSON file\n -n/--photon\tphoton number\n"
           " -s/--session\tsession name\n -u/--unitinmm\tvoxel size [1]\n"
           " -E/--seed\tRNG seed\n -O/--outputtype\tx:energy,f:flux,l:fluence\n"
           " -b/--reflect\tmismatch [1]\n -U/--normalize\t[1]\n"
           " -t/--thread\tGPU threads [65536]\n -G/--gpuid\tdevice [0]\n"
           " -m/--meshres\tshape mesh resolution [24]\n"
           " -c/--curv\tcurvature [1=on,0=off,-1=auto]\n"
           " -L/--listgpu\tlist GPUs\n --dumpjson\tdump config\n"
           " --dumpmesh\tsave mesh JSON\n --debugcurv\texport normal error\n"
           " -h/--help\n", n, n);
    exit(0);
}

SimConfig parse_cmdline(int argc, char** argv, CmdOverrides& ovr) {
    std::string inputfile;

    if (argc < 2) {
        printhelp(argv[0]);
    }

    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);

        if ((a == "-f" || a == "--input") && i + 1 < argc) {
            inputfile = argv[++i];
        } else if ((a == "-n" || a == "--photon") && i + 1 < argc) {
            ovr.nphoton = (uint64_t)atof(argv[++i]);
        } else if ((a == "-s" || a == "--session") && i + 1 < argc) {
            ovr.session_id = argv[++i];
        } else if ((a == "-u" || a == "--unitinmm") && i + 1 < argc) {
            ovr.unitinmm = (float)atof(argv[++i]);
        } else if ((a == "-E" || a == "--seed") && i + 1 < argc) {
            ovr.rng_seed = (uint32_t)atoi(argv[++i]);
        } else if ((a == "-O" || a == "--outputtype") && i + 1 < argc) {
            char c2 = argv[++i][0];
            ovr.outputtype = (c2 == 'f' ? 0 : c2 == 'l' ? 1 : 2);
        } else if ((a == "-b" || a == "--reflect") && i + 1 < argc) {
            ovr.isreflect = atoi(argv[++i]);
        } else if ((a == "-U" || a == "--normalize") && i + 1 < argc) {
            ovr.isnormalize = atoi(argv[++i]);
        } else if ((a == "-t" || a == "--thread") && i + 1 < argc) {
            ovr.totalthread = (uint32_t)atoi(argv[++i]);
        } else if ((a == "-G" || a == "--gpuid") && i + 1 < argc) {
            ovr.gpuid = atoi(argv[++i]);
        } else if ((a == "-m" || a == "--meshres") && i + 1 < argc) {
            ovr.meshres = atoi(argv[++i]);
        } else if ((a == "-c" || a == "--curv") && i + 1 < argc) {
            ovr.docurv = atoi(argv[++i]);
        } else if (a == "-L" || a == "--listgpu") {
            ovr.listgpu = true;
        } else if (a == "--dumpjson") {
            ovr.dumpjson = true;
        } else if (a == "--dumpmesh") {
            ovr.dumpmesh = true;
        } else if (a == "--debugcurv") {
            ovr.debugcurv = true;
        } else if (a == "-h" || a == "--help") {
            printhelp(argv[0]);
        } else if (a[0] != '-' && inputfile.empty()) {
            inputfile = a;
        }
    }

    if (inputfile.empty()) {
        fprintf(stderr, "No input JSON\n");
        printhelp(argv[0]);
    }

    ovr.inputfile = inputfile;
    SimConfig cfg = load_json_input(inputfile.c_str());

    if (ovr.nphoton > 0) {
        cfg.nphoton = ovr.nphoton;
    }

    if (ovr.rng_seed > 0) {
        cfg.rng_seed = ovr.rng_seed;
    }

    if (!ovr.session_id.empty()) {
        cfg.session_id = ovr.session_id;
    }

    if (ovr.unitinmm > 0) {
        cfg.unitinmm = ovr.unitinmm;
    }

    if (ovr.outputtype >= 0) {
        cfg.output_type = ovr.outputtype;
    }

    if (ovr.isreflect >= 0) {
        cfg.do_mismatch = (ovr.isreflect != 0);
    }

    if (ovr.isnormalize >= 0) {
        cfg.do_normalize = (ovr.isnormalize != 0);
    }

    if (ovr.meshres > 0) {
        cfg.mesh_res = ovr.meshres;
    }

    return cfg;
}

/* ================================================================ */
/*                         List GPUs                                */
/* ================================================================ */

void list_gpus() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    printf("========================== CUDA GPU Devices ==========================\n");

    for (int i = 0; i < count; i++) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, i));
        printf("Device %d: %s (SM %d.%d, %zu MB)\n", i, p.name,
               p.major, p.minor, p.totalGlobalMem / (1024 * 1024));
    }
}

/* ================================================================ */
/*                     OptiX pipeline setup                         */
/* ================================================================ */

struct OptixState {
    OptixDeviceContext context;
    OptixModule module;
    OptixPipeline pipeline;
    OptixPipelineCompileOptions pipeCompileOpts;
    OptixProgramGroup raygenPG, missPG, hitgroupPG;
    OptixShaderBindingTable sbt;
    CUDABuf raygenSBT, missSBT, hitgroupSBT;
    CUDABuf gasBuf, tempBuf;
    OptixTraversableHandle gasHandle;
};

void optix_log_callback(unsigned int level, const char* tag, const char* msg, void*) {
    if (level < 4) {
        fprintf(stderr, "[OptiX %u][%s]: %s\n", level, tag, msg);
    }
}

void setup_optix(OptixState& st, int gpuid, const char* ptxfile) {
    CUDA_CHECK(cudaSetDevice(gpuid));
    CUDA_CHECK(cudaFree(0));  /* init CUDA */
    OPTIX_CHECK(optixInit());

    /* Context */
    CUcontext cuCtx = 0;
    OptixDeviceContextOptions ctxOpts = {};
    ctxOpts.logCallbackFunction = optix_log_callback;
    ctxOpts.logCallbackLevel = 3;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &ctxOpts, &st.context));

    /* Load PTX */
    std::ifstream ptxf(ptxfile);

    if (!ptxf) {
        throw std::runtime_error(std::string("Cannot open PTX: ") + ptxfile);
    }

    std::string ptxstr((std::istreambuf_iterator<char>(ptxf)),
                       std::istreambuf_iterator<char>());

    /* Module */
    OptixModuleCompileOptions modOpts = {};
    modOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    modOpts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    modOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    modOpts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    modOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

    st.pipeCompileOpts = {};
    st.pipeCompileOpts.usesMotionBlur = false;
    st.pipeCompileOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    st.pipeCompileOpts.numPayloadValues = 16;  /* 12 photon + 4 RNG */
    st.pipeCompileOpts.numAttributeValues = 2;
    st.pipeCompileOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    st.pipeCompileOpts.pipelineLaunchParamsVariableName = "gcfg";

    char log[2048];
    size_t logsz = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(st.context, &modOpts, &st.pipeCompileOpts,
                                         ptxstr.c_str(), ptxstr.size(), log, &logsz, &st.module));

    if (logsz > 1) {
        printf("Module log: %s\n", log);
    }

    /* Program groups */
    OptixProgramGroupOptions pgOpts = {};

    OptixProgramGroupDesc rgDesc = {};
    rgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgDesc.raygen.module = st.module;
    rgDesc.raygen.entryFunctionName = "__raygen__rg";
    logsz = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(st.context, &rgDesc, 1, &pgOpts, log, &logsz, &st.raygenPG));

    OptixProgramGroupDesc msDesc = {};
    msDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msDesc.miss.module = st.module;
    msDesc.miss.entryFunctionName = "__miss__ms";
    logsz = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(st.context, &msDesc, 1, &pgOpts, log, &logsz, &st.missPG));

    OptixProgramGroupDesc hgDesc = {};
    hgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hgDesc.hitgroup.moduleCH = st.module;
    hgDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    logsz = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(st.context, &hgDesc, 1, &pgOpts, log, &logsz, &st.hitgroupPG));

    /* Pipeline */
    OptixProgramGroup pgs[] = { st.raygenPG, st.missPG, st.hitgroupPG };
    OptixPipelineLinkOptions linkOpts = {};
    linkOpts.maxTraceDepth = 1;
    logsz = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(st.context, &st.pipeCompileOpts, &linkOpts,
                                    pgs, 3, log, &logsz, &st.pipeline));

    /* SBT */
    RaygenRecord rg;
    OPTIX_CHECK(optixSbtRecordPackHeader(st.raygenPG, &rg));
    st.raygenSBT.alloc_upload(&rg, 1);

    MissRecord ms;
    OPTIX_CHECK(optixSbtRecordPackHeader(st.missPG, &ms));
    st.missSBT.alloc_upload(&ms, 1);

    HitgroupRecord hg;
    OPTIX_CHECK(optixSbtRecordPackHeader(st.hitgroupPG, &hg));
    st.hitgroupSBT.alloc_upload(&hg, 1);

    st.sbt = {};
    st.sbt.raygenRecord = st.raygenSBT.d_ptr();
    st.sbt.missRecordBase = st.missSBT.d_ptr();
    st.sbt.missRecordStrideInBytes = sizeof(MissRecord);
    st.sbt.missRecordCount = 1;
    st.sbt.hitgroupRecordBase = st.hitgroupSBT.d_ptr();
    st.sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    st.sbt.hitgroupRecordCount = 1;
}

/* ================================================================ */
/*                Build acceleration structure                      */
/* ================================================================ */

void build_gas(OptixState& st, const std::vector<Vec3>& nodes,
               const std::vector<std::array<uint32_t, 3>>& faces) {
    CUDABuf vertBuf, idxBuf;
    vertBuf.alloc_upload(nodes.data(), nodes.size());

    std::vector<uint3> idx3(faces.size());

    for (size_t i = 0; i < faces.size(); i++) {
        idx3[i] = make_uint3(faces[i][0], faces[i][1], faces[i][2]);
    }

    idxBuf.alloc_upload(idx3.data(), idx3.size());

    OptixBuildInput triInput = {};
    triInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    CUdeviceptr dv = vertBuf.d_ptr(), di = idxBuf.d_ptr();
    triInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triInput.triangleArray.vertexStrideInBytes = sizeof(Vec3);
    triInput.triangleArray.numVertices = (unsigned)nodes.size();
    triInput.triangleArray.vertexBuffers = &dv;
    triInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triInput.triangleArray.indexStrideInBytes = sizeof(uint3);
    triInput.triangleArray.numIndexTriplets = (unsigned)faces.size();
    triInput.triangleArray.indexBuffer = di;
    uint32_t flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    triInput.triangleArray.flags = &flags;
    triInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOpts = {};
    accelOpts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOpts.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(st.context, &accelOpts, &triInput, 1, &bufSizes));

    CUDABuf tempBuf, outBuf, compactSizeBuf;
    tempBuf.alloc(bufSizes.tempSizeInBytes);
    outBuf.alloc(bufSizes.outputSizeInBytes);
    compactSizeBuf.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactSizeBuf.d_ptr();

    OPTIX_CHECK(optixAccelBuild(st.context, 0, &accelOpts, &triInput, 1,
                                tempBuf.d_ptr(), tempBuf.sz, outBuf.d_ptr(), outBuf.sz,
                                &st.gasHandle, &emitDesc, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t compactSize;
    compactSizeBuf.download(&compactSize, sizeof(uint64_t));
    st.gasBuf.alloc(compactSize);
    OPTIX_CHECK(optixAccelCompact(st.context, 0, st.gasHandle,
                                  st.gasBuf.d_ptr(), st.gasBuf.sz, &st.gasHandle));
    CUDA_CHECK(cudaDeviceSynchronize());

    tempBuf.free();
    outBuf.free();
    compactSizeBuf.free();
    vertBuf.free();
    idxBuf.free();

    printf("GAS: %u triangles, %lu KB (compacted)\n",
           (unsigned)faces.size(), (unsigned long)(compactSize / 1024));
}

/* ================================================================ */
/*                            Main                                  */
/* ================================================================ */

int main(int argc, char** argv) {
    const char* ptxFile = "vkmmc_optix_core.ptx";

    /* Check for PTX file override */
    if (argc > 2) {
        std::string la(argv[argc - 1]);

        if (la.size() > 4 && la.substr(la.size() - 4) == ".ptx") {
            ptxFile = argv[--argc];
        }
    }

    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);

        if (a == "-L" || a == "--listgpu") {
            list_gpus();
            return 0;
        }
    }

    CmdOverrides ovr;
    SimConfig cfg = parse_cmdline(argc, argv, ovr);

    /* ---- Phase 1: Generate CSG mesh ---- */
    std::vector<NodeCurvature> curvData;
    bool has_curvature = false;
    bool want_curvature = (ovr.docurv < 0) ? true : (ovr.docurv != 0);

    if (cfg.is_csg) {
        std::ifstream sf(ovr.inputfile.c_str());
        json jroot;
        sf >> jroot;

        if (jroot.contains("Shapes") && jroot["Shapes"].is_array()) {
            float ext[6] = {0, 60, 0, 60, 0, 60};
            ShapeMesh sm = parse_shapes(jroot["Shapes"], ext, cfg.mesh_res);
            cfg.nodes = sm.nodes;
            cfg.faces = sm.faces;
            cfg.facedata = sm.facedata;
            cfg.face_shape_id = sm.shape_id;
            update_bbox(cfg);

            if (want_curvature) {
                auto origins = extract_shape_origins(jroot["Shapes"]);
                curvData = compute_curvature(sm, origins);
                has_curvature = true;
                printf("Curvature: %zu nodes, %zu shapes\n", curvData.size(), origins.size());
            }
        }

        if (cfg.nodes.empty()) {
            fprintf(stderr, "CSG: no shapes\n");
            return 1;
        }
    }

    /* ---- Phase 2: Setup OptiX ---- */
    CUDA_CHECK(cudaSetDevice(ovr.gpuid));
    {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, ovr.gpuid));
        printf("GPU: %s (SM %d.%d)\n", p.name, p.major, p.minor);
    }

    OptixState optix;
    setup_optix(optix, ovr.gpuid, ptxFile);

    /* Build GAS */
    build_gas(optix, cfg.nodes, cfg.faces);

    /* ---- Phase 3: Prepare buffers ---- */

    /* Face buffer: interleaved */
    struct FaceGPU {
        float nx, ny, nz, pm, v0f, v1f, v2f, pad;
    };
    std::vector<FaceGPU> faceGpu(cfg.faces.size());

    for (size_t i = 0; i < cfg.faces.size(); i++) {
        faceGpu[i].nx = cfg.facedata[i].nx;
        faceGpu[i].ny = cfg.facedata[i].ny;
        faceGpu[i].nz = cfg.facedata[i].nz;
        faceGpu[i].pm = cfg.facedata[i].packed_media;
        uint32_t v0 = cfg.faces[i][0], v1 = cfg.faces[i][1], v2 = cfg.faces[i][2];
        memcpy(&faceGpu[i].v0f, &v0, 4);
        memcpy(&faceGpu[i].v1f, &v1, 4);
        memcpy(&faceGpu[i].v2f, &v2, 4);
        faceGpu[i].pad = 0;
    }

    CUDABuf faceBuf;
    faceBuf.alloc_upload(faceGpu);

    // Curvature buffer - ALWAYS allocate cfg.nodes.size() elements
    /* Curvature buffer */
    struct GpuNodeCurv {
        float nx, ny, nz, k1, px, py, pz, k2, posx, posy, posz, pad;
    };
    std::vector<GpuNodeCurv> gpuCurv(cfg.nodes.size());

    printf("DEBUG: Allocating curvature buffer: %zu nodes\n", cfg.nodes.size());

    if (has_curvature && !curvData.empty()) {
        printf("DEBUG: Filling with curvature data: %zu entries\n", curvData.size());

        for (size_t i = 0; i < curvData.size(); i++) {
            gpuCurv[i] = {curvData[i].nx, curvData[i].ny, curvData[i].nz, curvData[i].k1,
                          curvData[i].px, curvData[i].py, curvData[i].pz, curvData[i].k2,
                          cfg.nodes[i].x, cfg.nodes[i].y, cfg.nodes[i].z, 0
                         };
        }
    } else {
        printf("DEBUG: Filling with zeros\n");
        memset(gpuCurv.data(), 0, gpuCurv.size() * sizeof(GpuNodeCurv));
    }

    printf("DEBUG: Curvature buffer size: %zu elements = %zu bytes\n",
           gpuCurv.size(), gpuCurv.size() * sizeof(GpuNodeCurv));

    CUDABuf curvBuf;
    curvBuf.alloc_upload(gpuCurv);

    printf("DEBUG: Curvature buffer uploaded to GPU at %p\n", curvBuf.ptr);

    // DEBUG: Verify face indices are in bounds
    printf("DEBUG: Checking face vertex indices...\n");

    for (size_t i = 0; i < std::min(size_t(10), cfg.faces.size()); i++) {
        uint32_t v0 = cfg.faces[i][0];
        uint32_t v1 = cfg.faces[i][1];
        uint32_t v2 = cfg.faces[i][2];
        printf("  Face[%zu]: v=(%u,%u,%u)", i, v0, v1, v2);

        if (v0 >= cfg.nodes.size() || v1 >= cfg.nodes.size() || v2 >= cfg.nodes.size()) {
            printf(" *** OUT OF BOUNDS! ***\n");
        } else {
            printf(" OK\n");
        }
    }

    /* Output grid */
    float vs = cfg.unitinmm;

    if (cfg.has_steps) {
        vs = cfg.steps[0];
    }

    float ge = vs * 0.5f;
    float gmin[3] = {cfg.nmin.x - ge, cfg.nmin.y - ge, cfg.nmin.z - ge};
    float gmax[3] = {cfg.nmax.x + ge, cfg.nmax.y + ge, cfg.nmax.z + ge};
    uint32_t nx = (uint32_t)ceil((gmax[0] - gmin[0]) / vs);
    uint32_t ny = (uint32_t)ceil((gmax[1] - gmin[1]) / vs);
    uint32_t nz = (uint32_t)ceil((gmax[2] - gmin[2]) / vs);
    uint32_t crop0w = nx * ny * nz * cfg.maxgate, outSz = crop0w * 2 + 16;

    CUDABuf outBuf;
    outBuf.alloc(outSz * sizeof(float));
    CUDA_CHECK(cudaMemset(outBuf.ptr, 0, outSz * sizeof(float)));
    printf("Grid: %ux%ux%u x %d gates, voxel=%.3fmm\n", nx, ny, nz, cfg.maxgate, vs);

    /* Thread count */
    uint32_t tt = (ovr.totalthread > 0) ? ovr.totalthread : 65536;

    if (cfg.nphoton < tt) {
        tt = ((uint32_t)cfg.nphoton + 63) / 64 * 64;

        if (!tt) {
            tt = 64;
        }
    }

    /* RNG seeds */
    srand(cfg.rng_seed > 0 ? cfg.rng_seed : (uint32_t)time(0));
    std::vector<uint4> seeds(tt);

    for (uint32_t i = 0; i < tt; i++) {
        seeds[i] = make_uint4((uint32_t)rand(), (uint32_t)rand(), (uint32_t)rand(), (uint32_t)rand());
    }

    CUDABuf seedBuf;
    seedBuf.alloc_upload(seeds);

    /* ---- Phase 4: Fill launch params ---- */
    VKMMCParam params;
    memset(&params, 0, sizeof(params));
    params.gashandle = optix.gasHandle;
    params.facebuf = faceBuf.d_ptr();
    params.curvbuf = curvBuf.d_ptr();
    params.outputbuf = outBuf.d_ptr();
    params.seedbuf = seedBuf.d_ptr();
    params.srctype = cfg.srctype;
    params.srcpos = make_float3(cfg.srcpos[0], cfg.srcpos[1], cfg.srcpos[2]);
    params.srcdir = make_float3(cfg.srcdir[0], cfg.srcdir[1], cfg.srcdir[2]);
    params.srcparam1 = make_float4(cfg.srcparam1[0], cfg.srcparam1[1], cfg.srcparam1[2], cfg.srcparam1[3]);
    params.srcparam2 = make_float4(cfg.srcparam2[0], cfg.srcparam2[1], cfg.srcparam2[2], cfg.srcparam2[3]);
    params.grid_min = make_float3(gmin[0], gmin[1], gmin[2]);
    params.grid_extent = make_float3(gmax[0] - gmin[0], gmax[1] - gmin[1], gmax[2] - gmin[2]);
    params.grid_stride = make_uint4(nx, nx * ny, nx * ny * nz, crop0w);
    params.voxel_scale = 1.0f / vs;
    params.tstart = cfg.t0;
    params.tend = cfg.t1;
    params.inv_timestep = 1.0f / cfg.dt;
    params.maxgate = cfg.maxgate;
    params.initial_medium = cfg.mediumid0;
    params.do_reflect = cfg.do_mismatch ? 1u : 0u;
    params.output_type = cfg.output_type;
    params.num_media = (uint32_t)cfg.media.size();
    params.do_csg = cfg.is_csg ? 1u : 0u;
    params.has_curvature = has_curvature ? 1u : 0u;
    params.minenergy = cfg.minenergy;
    params.roulettesize = cfg.roulettesize;
    // Also add debug output:
    printf("DEBUG: has_curvature=%d, params.has_curvature=%u\n",
           has_curvature, params.has_curvature);

    /* Media */
    for (size_t i = 0; i < cfg.media.size() && i < MAX_PROP; i++) {
        params.media[i] = {cfg.media[i].mua, cfg.media[i].mus, cfg.media[i].g, cfg.media[i].n};
    }

    CUDABuf paramBuf;
    paramBuf.alloc(sizeof(VKMMCParam));

    /* ---- Phase 5: Launch ---- */
    uint64_t batchsz = 500000;

    if (ovr.batch_size == 0) {
        batchsz = cfg.nphoton;
    } else if (ovr.batch_size != UINT64_MAX) {
        batchsz = ovr.batch_size;
    }

    uint64_t pdone = 0;
    int batch = 0;
    printf("Threads: %u, batch: %lu photons\n", tt, (unsigned long)batchsz);
    // ========== ADD THESE DEBUG LINES ==========
    printf("DEBUG: Mesh mode configuration:\n");
    printf("  cfg.is_csg = %d\n", cfg.is_csg);
    printf("  params.do_csg = %u\n", params.do_csg);
    printf("  params.initial_medium = %u (0x%x)\n", params.initial_medium, params.initial_medium);
    printf("  cfg.init_elem = %d\n", cfg.init_elem);
    printf("  cfg.mediumid0 = %u\n", cfg.mediumid0);
    printf("  params.has_curvature = %u\n", params.has_curvature);
    printf("  params.num_media = %u\n", params.num_media);
    printf("  Media properties:\n");

    for (size_t i = 0; i < cfg.media.size() && i < 5; i++) {
        printf("    Media[%zu]: mua=%.6f mus=%.6f g=%.4f n=%.4f\n",
               i, cfg.media[i].mua, cfg.media[i].mus, cfg.media[i].g, cfg.media[i].n);
    }

    // ========================================
    typedef std::chrono::high_resolution_clock Clk;
    double kms = 0;
    Clk::time_point t0 = Clk::now();

    while (pdone < cfg.nphoton) {
        uint64_t rem = cfg.nphoton - pdone, bp = (rem < batchsz) ? rem : batchsz;
        params.threadphoton = (int)(bp / tt);
        params.oddphoton = (int)(bp - (uint64_t)params.threadphoton * tt);
        paramBuf.upload(&params, sizeof(params));

        Clk::time_point ks = Clk::now();
        OPTIX_CHECK(optixLaunch(optix.pipeline, 0,
                                paramBuf.d_ptr(), sizeof(VKMMCParam),
                                &optix.sbt, tt, 1, 1));
        CUDA_CHECK(cudaDeviceSynchronize());
        kms += std::chrono::duration<double, std::milli>(Clk::now() - ks).count();

        pdone += bp;
        batch++;
        printf("  batch %d: %lu photons (%lu/%lu)\n",
               batch, (unsigned long)bp, (unsigned long)pdone, (unsigned long)cfg.nphoton);
    }

    double tms = std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
    printf("Done (%d batches), kernel: %.3f ms, total: %.3f ms\n", batch, kms, tms);
    printf("Speed: %.2f photon/ms\n", (double)cfg.nphoton / kms);

    /* ---- Phase 6: Readback ---- */
    std::vector<float> raw(outSz);
    outBuf.download(raw.data(), outSz * sizeof(float));
    std::vector<float> fluence(crop0w);
    double absorbed = 0;

    for (uint32_t i = 0; i < crop0w; i++) {
        fluence[i] = raw[i] + raw[i + crop0w];
        absorbed += fluence[i];
    }

    printf("absorbed: %.5f%%\n", absorbed / (double)cfg.nphoton * 100.0);

    if (cfg.do_normalize) {
        float vv = vs * vs * vs;

        for (uint32_t i = 0; i < crop0w; i++) {
            fluence[i] /= (float)cfg.nphoton * vv;
        }
    }

    /* Save JData */
    {
        std::vector<size_t> dims = {nx, ny, nz};

        if (cfg.maxgate > 1) {
            dims.push_back((size_t)cfg.maxgate);
        }

        json root;
        root["Session"] = {{"ID", cfg.session_id}, {"Photons", cfg.nphoton}};
        root["Forward"] = {{"T0", cfg.t0}, {"T1", cfg.t1}, {"Dt", cfg.dt}};
        root["Domain"] = {{"LengthUnit", cfg.unitinmm}, {"VoxelSize", vs},
            {"Dim", {nx, ny, nz}}, {"Origin", {gmin[0], gmin[1], gmin[2]}}
        };
        root["Fluence"] = jdata_encode("single", dims, fluence.data(), crop0w * sizeof(float));
        std::string outname = cfg.session_id + ".jdat";
        std::ofstream of(outname.c_str());
        of << root.dump(2) << std::endl;
        printf("Output: %s (%ux%ux%u)\n", outname.c_str(), nx, ny, nz);
    }

    /* ---- Cleanup ---- */
    faceBuf.free();
    curvBuf.free();
    outBuf.free();
    seedBuf.free();
    paramBuf.free();
    optix.gasBuf.free();
    optix.raygenSBT.free();
    optix.missSBT.free();
    optix.hitgroupSBT.free();
    optixPipelineDestroy(optix.pipeline);
    optixProgramGroupDestroy(optix.raygenPG);
    optixProgramGroupDestroy(optix.missPG);
    optixProgramGroupDestroy(optix.hitgroupPG);
    optixModuleDestroy(optix.module);
    optixDeviceContextDestroy(optix.context);

    return 0;
}