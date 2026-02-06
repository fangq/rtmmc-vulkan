/*
 * vkmmc_io.h — Load/save MMC JSON input/output with JData-encoded arrays
 *
 * Dependencies:
 *   - nlohmann/json (json.hpp): https://github.com/nlohmann/json
 *   - miniz.c: https://github.com/richgel999/miniz
 *
 * JData spec: binary arrays are row-major, zlib-compressed, base64-encoded
 *   _ArrayType_    : element type ("single","int32",etc.)
 *   _ArraySize_    : dimensions array
 *   _ArrayZipType_ : "zlib"
 *   _ArrayZipSize_ : total element count (product of _ArraySize_)
 *   _ArrayZipData_ : base64(zlib(raw_bytes))
 *
 * Input arrays:
 *   MeshNode: [nn,3] float32 — node coordinates
 *   MeshElem: [ne,5] int32   — [n1,n2,n3,n4,region] tetrahedral (1-based)
 *   MeshSurf: [nf,4] int32   — [n1,n2,n3,region] surface triangles (1-based)
 *     If MeshSurf is present, it is used directly (no tet extraction needed).
 *     "region" is the enclosed region ID; the neighbor across each face must
 *     be determined by face-pairing or set to ambient(0) for exterior faces.
 *
 * Output:
 *   3D fluence array saved as JData inside a JSON file.
 */

#ifndef VK_RTMMC_IO_H
#define VK_RTMMC_IO_H

#include <nlohmann/json.hpp>
#include <miniz.h>

#include <vector>
#include <string>
#include <array>
#include <map>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <fstream>

using json = nlohmann::json;

// ============================================================================
// Data structures (shared with vkmmc.cpp)
// ============================================================================

struct Vec3 { float x, y, z; };
struct Medium { float mua, mus, g, n; };
struct FaceData { float nx, ny, nz, packed_media; }; // matches OptiX float4 fnorm

struct SimConfig {
    std::string session_id;
    uint64_t nphoton = 1000000;
    uint32_t rng_seed = 29012391;
    bool do_mismatch = true, do_normalize = true;
    int output_type = 2; // 0=flux,1=fluence,2=energy
    float t0 = 0, t1 = 5e-9f, dt = 5e-9f;
    int maxgate = 1;
    float unitinmm = 1.0f;
    std::vector<Medium> media;
    int srctype = 0;
    float srcpos[3]={}, srcdir[4]={0,0,1,0};
    float srcparam1[4]={}, srcparam2[4]={};
    std::vector<Vec3> nodes;
    std::vector<std::array<uint32_t,3>> faces;
    std::vector<FaceData> facedata;
    Vec3 nmin{}, nmax{};
    int init_elem = -1;
    uint32_t mediumid0 = 0xFFFFFFFFu;
    // Grid dimensions from Domain.Dim (if provided, overrides bbox-derived)
    uint32_t grid_dim[3] = {0, 0, 0};
    bool has_grid_dim = false;
    // Voxel size from Domain.Steps (overrides LengthUnit if present)
    float steps[3] = {0, 0, 0};
    bool has_steps = false;
};

// ============================================================================
// Base64 encode / decode
// ============================================================================

static const char b64_enc[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static const uint8_t b64_dec[256] = {
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,62,64,64,64,63,
    52,53,54,55,56,57,58,59,60,61,64,64,64, 0,64,64,64, 0, 1, 2, 3, 4, 5, 6,
     7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,64,64,64,64,64,
    64,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
    49,50,51,64,64,64,64,64, 64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64
};

std::vector<uint8_t> base64_decode(const std::string& in) {
    std::string cl; cl.reserve(in.size());
    for (char c : in) if (b64_dec[(uint8_t)c] < 64 || c == '=') cl += c;
    size_t len = cl.size();
    if (len % 4) throw std::runtime_error("Invalid base64 length");
    size_t ol = len / 4 * 3;
    if (len >= 1 && cl[len-1] == '=') ol--;
    if (len >= 2 && cl[len-2] == '=') ol--;
    std::vector<uint8_t> out(ol); size_t j = 0;
    for (size_t i = 0; i < len; i += 4) {
        uint32_t a=b64_dec[(uint8_t)cl[i]], b=b64_dec[(uint8_t)cl[i+1]];
        uint32_t c=(i+2<len)?b64_dec[(uint8_t)cl[i+2]]:0;
        uint32_t d=(i+3<len)?b64_dec[(uint8_t)cl[i+3]]:0;
        uint32_t t=(a<<18)|(b<<12)|(c<<6)|d;
        if(j<ol)out[j++]=(t>>16)&0xFF;
        if(j<ol)out[j++]=(t>>8)&0xFF;
        if(j<ol)out[j++]=t&0xFF;
    }
    return out;
}

std::string base64_encode(const uint8_t* data, size_t len) {
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t b = ((uint32_t)data[i]) << 16;
        if (i + 1 < len) b |= ((uint32_t)data[i+1]) << 8;
        if (i + 2 < len) b |= (uint32_t)data[i+2];
        out += b64_enc[(b >> 18) & 0x3F];
        out += b64_enc[(b >> 12) & 0x3F];
        out += (i + 1 < len) ? b64_enc[(b >> 6) & 0x3F] : '=';
        out += (i + 2 < len) ? b64_enc[b & 0x3F] : '=';
    }
    return out;
}

// ============================================================================
// JData decode: base64 → zlib decompress → raw bytes
// ============================================================================

static size_t jdata_elem_size(const std::string& t) {
    if (t=="single"||t=="float32"||t=="int32"||t=="uint32") return 4;
    if (t=="double"||t=="float64"||t=="int64"||t=="uint64") return 8;
    if (t=="int16"||t=="uint16") return 2;
    if (t=="int8"||t=="uint8") return 1;
    return 4;
}

std::vector<uint8_t> jdata_decode(const json& jd) {
    if (jd.value("_ArrayZipType_","") != "zlib")
        throw std::runtime_error("Unsupported _ArrayZipType_");
    auto comp = base64_decode(jd["_ArrayZipData_"].get<std::string>());
    size_t nelems = jd["_ArrayZipSize_"].get<size_t>();
    size_t esz = jdata_elem_size(jd["_ArrayType_"].get<std::string>());
    size_t usz = nelems * esz;
    std::vector<uint8_t> out(usz);
    mz_ulong dlen = (mz_ulong)usz;
    if (mz_uncompress(out.data(), &dlen, comp.data(), (mz_ulong)comp.size()) != MZ_OK)
        throw std::runtime_error("zlib decompression failed");
    if (dlen != usz)
        throw std::runtime_error("Decompressed size mismatch");
    return out;
}

// ============================================================================
// JData encode: raw bytes → zlib compress → base64
// ============================================================================

json jdata_encode(const std::string& arr_type,
                  const std::vector<size_t>& arr_size,
                  const void* data, size_t data_bytes)
{
    // zlib compress
    mz_ulong comp_bound = mz_compressBound((mz_ulong)data_bytes);
    std::vector<uint8_t> comp(comp_bound);
    mz_ulong comp_len = comp_bound;
    if (mz_compress(comp.data(), &comp_len,
                    (const unsigned char*)data, (mz_ulong)data_bytes) != MZ_OK)
        throw std::runtime_error("zlib compression failed");
    comp.resize(comp_len);

    // total element count
    size_t total = 1;
    for (auto d : arr_size) total *= d;

    json jd;
    jd["_ArrayType_"] = arr_type;
    jd["_ArraySize_"] = arr_size;
    jd["_ArrayOrder_"] = "c";
    jd["_ArrayZipType_"] = "zlib";
    jd["_ArrayZipSize_"] = total;
    jd["_ArrayZipData_"] = base64_encode(comp.data(), comp.size());
    return jd;
}

// ============================================================================
// Vec3 helpers
// ============================================================================

static Vec3 v3cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
static float v3len(Vec3 v) { return std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z); }

// Pack front/back media into float (matching OptiX buildSBT)
static float pack_media(uint32_t front, uint32_t back) {
    uint32_t p = (front << 16) | (back & 0xFFFF);
    float f; std::memcpy(&f, &p, 4); return f;
}

// Compute face normal and build FaceData
static FaceData make_facedata(const Vec3* nodes, uint32_t i0, uint32_t i1,
                               uint32_t i2, uint32_t front, uint32_t back)
{
    Vec3 a=nodes[i0], b=nodes[i1], c=nodes[i2];
    Vec3 e1={b.x-a.x,b.y-a.y,b.z-a.z}, e2={c.x-a.x,c.y-a.y,c.z-a.z};
    Vec3 N = v3cross(e1, e2);
    float l = v3len(N);
    if (l > 0.f) { N.x/=l; N.y/=l; N.z/=l; }
    return {N.x, N.y, N.z, pack_media(front, back)};
}

// ============================================================================
// Load MeshSurf: [nf, 4] int32 — n1, n2, n3, region (1-based nodes)
//
// "region" = the medium label of the region this face encloses.
// To determine front/back, we pair faces: if two faces share the same
// sorted vertex triplet, one encloses region A and the other region B.
// Unpaired faces are exterior (neighbor = ambient 0).
// ============================================================================

struct FaceKey {
    uint32_t v[3];
    bool operator<(const FaceKey& o) const {
        if(v[0]!=o.v[0]) return v[0]<o.v[0];
        if(v[1]!=o.v[1]) return v[1]<o.v[1];
        return v[2]<o.v[2];
    }
};

void load_mesh_surf(const std::vector<Vec3>& nodes,
                    const int32_t* sdata, size_t nf,
                    std::vector<std::array<uint32_t,3>>& out_faces,
                    std::vector<FaceData>& out_fd)
{
    // First pass: group faces by sorted vertex key to find neighbors
    struct Entry { uint32_t v[3]; uint32_t region; size_t idx; };
    std::map<FaceKey, std::vector<Entry>> fmap;

    for (size_t i = 0; i < nf; i++) {
        const int32_t* row = sdata + i * 4;
        Entry e;
        e.v[0] = (uint32_t)(row[0]-1);
        e.v[1] = (uint32_t)(row[1]-1);
        e.v[2] = (uint32_t)(row[2]-1);
        e.region = (uint32_t)row[3];
        e.idx = i;
        FaceKey k;
        k.v[0]=e.v[0]; k.v[1]=e.v[1]; k.v[2]=e.v[2];
        std::sort(k.v, k.v+3);
        fmap[k].push_back(e);
    }

    out_faces.clear(); out_fd.clear();
    out_faces.reserve(fmap.size());
    out_fd.reserve(fmap.size());

    for (auto it = fmap.begin(); it != fmap.end(); ++it) {
        auto& entries = it->second;
        auto& e0 = entries[0];
        uint32_t front = e0.region;
        uint32_t back = 0; // ambient by default (exterior face)

        if (entries.size() >= 2) {
            back = entries[1].region;
        }

        // Skip internal faces where both sides have the same region
        if (front == back) continue;

        out_faces.push_back({{e0.v[0], e0.v[1], e0.v[2]}});
        out_fd.push_back(make_facedata(nodes.data(), e0.v[0], e0.v[1], e0.v[2],
                                        front, back));
    }

    std::cout << "MeshSurf: " << nf << " input faces -> "
              << out_faces.size() << " unique surface triangles\n";
}

// ============================================================================
// Extract surface from MeshElem (tetrahedral): [ne, 5] int32
// ============================================================================

void extract_surface_from_tet(const std::vector<Vec3>& nodes,
                               const int32_t* elems, size_t ne,
                               std::vector<std::array<uint32_t,3>>& out_faces,
                               std::vector<FaceData>& out_fd)
{
    static const int ftab[4][3] = {{0,1,2},{0,1,3},{0,2,3},{1,2,3}};

    struct FInfo { uint32_t v[3]; int r1, r2; };
    std::map<FaceKey, FInfo> fmap;

    for (size_t e = 0; e < ne; e++) {
        const int32_t* row = elems + e * 5;
        uint32_t n[4] = {(uint32_t)(row[0]-1),(uint32_t)(row[1]-1),
                         (uint32_t)(row[2]-1),(uint32_t)(row[3]-1)};
        int reg = row[4];
        for (int f = 0; f < 4; f++) {
            uint32_t fv[3] = {n[ftab[f][0]], n[ftab[f][1]], n[ftab[f][2]]};
            FaceKey k; k.v[0]=fv[0]; k.v[1]=fv[1]; k.v[2]=fv[2];
            std::sort(k.v, k.v+3);
            auto it = fmap.find(k);
            if (it == fmap.end())
                fmap[k] = {fv[0],fv[1],fv[2], reg, -1};
            else
                it->second.r2 = reg;
        }
    }

    out_faces.clear(); out_fd.clear();
    out_faces.reserve(fmap.size());
    out_fd.reserve(fmap.size());

    for (auto it = fmap.begin(); it != fmap.end(); ++it) {
        auto& fi = it->second;
        uint32_t front = (uint32_t)fi.r1;
        uint32_t back = (fi.r2 >= 0) ? (uint32_t)fi.r2 : 0u;
        // Skip internal faces where both sides have the same region
        if (front == back) continue;
        out_faces.push_back({{fi.v[0], fi.v[1], fi.v[2]}});
        out_fd.push_back(make_facedata(nodes.data(), fi.v[0], fi.v[1], fi.v[2],
                                        front, back));
    }

    std::cout << "Extracted " << out_faces.size() << " surface triangles from "
              << ne << " tetrahedra\n";
    for (size_t i = 0; i < out_faces.size(); i++) {
        auto& f = out_faces[i];
        auto& fd = out_fd[i];
        uint32_t pk = 0;
        std::memcpy(&pk, &fd.packed_media, 4);
    }
}

// ============================================================================
// Parse helpers
// ============================================================================

static int parse_srctype(const std::string& s) {
    if(s=="pencil")return 0; if(s=="isotropic")return 1; if(s=="cone")return 2;
    if(s=="gaussian")return 3; if(s=="planar")return 4; if(s=="pattern")return 5;
    if(s=="fourier")return 6; if(s=="arcsine")return 7; if(s=="disk")return 8;
    if(s=="zgaussian")return 11; if(s=="line")return 12; if(s=="slit")return 13;
    return 0;
}

static int parse_outputtype(const std::string& s) {
    if(s.empty()) return 2;
    switch(s[0]){case 'f':return 0;case 'l':return 1;case 'x':return 2;case 'j':return 3;}
    return 2;
}

// ============================================================================
// Load JSON input
// ============================================================================

SimConfig load_json_input(const char* filepath) {
    std::ifstream f(filepath);
    if (!f) throw std::runtime_error(std::string("Cannot open: ") + filepath);
    json j; f >> j;
    SimConfig cfg;

    // Session
    if (j.contains("Session")) {
        auto& s = j["Session"];
        cfg.session_id   = s.value("ID", "default");
        cfg.nphoton      = s.value("Photons", (uint64_t)1000000);
        cfg.rng_seed     = s.value("RNGSeed", (uint32_t)29012391);
        cfg.do_mismatch  = s.value("DoMismatch", true);
        cfg.do_normalize = s.value("DoNormalize", true);
        cfg.output_type  = parse_outputtype(s.value("OutputType", "x"));
    }

    // Forward
    if (j.contains("Forward")) {
        auto& fw = j["Forward"];
        cfg.t0 = fw.value("T0", 0.0f);
        cfg.t1 = fw.value("T1", 5e-9f);
        cfg.dt = fw.value("Dt", 5e-9f);
        cfg.maxgate = std::max(1, (int)((cfg.t1 - cfg.t0) / cfg.dt + 0.5f));
    }

    // Domain
    if (j.contains("Domain")) {
        auto& d = j["Domain"];
        cfg.unitinmm = d.value("LengthUnit", 1.0f);
        if (d.contains("Media")) {
            cfg.media.clear();
            for (auto& m : d["Media"])
                cfg.media.push_back({m.value("mua",0.f), m.value("mus",0.f),
                                     m.value("g",1.f), m.value("n",1.f)});
        }
        // Read grid dimensions if provided
        if (d.contains("Dim")) {
            auto dm = d["Dim"];
            cfg.grid_dim[0] = dm[0].get<uint32_t>();
            cfg.grid_dim[1] = dm[1].get<uint32_t>();
            cfg.grid_dim[2] = dm[2].get<uint32_t>();
            cfg.has_grid_dim = true;
        }
        // Read voxel step size if provided
        if (d.contains("Steps")) {
            auto st = d["Steps"];
            for (int i = 0; i < (int)st.size() && i < 3; i++)
                cfg.steps[i] = st[i].get<float>();
            cfg.has_steps = true;
        }
    }

    // Optode / Source
    if (j.contains("Optode") && j["Optode"].contains("Source")) {
        auto& src = j["Optode"]["Source"];
        cfg.srctype = parse_srctype(src.value("Type", "pencil"));
        if (src.contains("Pos")) {
            auto p=src["Pos"]; cfg.srcpos[0]=p[0]; cfg.srcpos[1]=p[1]; cfg.srcpos[2]=p[2];
        }
        if (src.contains("Dir")) {
            auto d=src["Dir"]; for(int i=0;i<(int)d.size()&&i<4;i++) cfg.srcdir[i]=d[i];
        }
        if (src.contains("Param1")) {
            auto p=src["Param1"]; for(int i=0;i<(int)p.size()&&i<4;i++) cfg.srcparam1[i]=p[i];
        }
        if (src.contains("Param2")) {
            auto p=src["Param2"]; for(int i=0;i<(int)p.size()&&i<4;i++) cfg.srcparam2[i]=p[i];
        }
    }

    // Shapes — mesh data
    if (!j.contains("Shapes"))
        throw std::runtime_error("JSON missing 'Shapes'");
    auto& sh = j["Shapes"];

    // Decode MeshNode: [nn, 3] float32
    if (!sh.contains("MeshNode"))
        throw std::runtime_error("JSON missing 'Shapes.MeshNode'");
    {
        auto& mn = sh["MeshNode"];
        auto dims = mn["_ArraySize_"].get<std::vector<size_t>>();
        size_t nn = dims[0];
        auto raw = jdata_decode(mn);
        const float* fd = reinterpret_cast<const float*>(raw.data());
        cfg.nodes.resize(nn);
        cfg.nmin = {1e30f, 1e30f, 1e30f};
        cfg.nmax = {-1e30f, -1e30f, -1e30f};
        for (size_t i = 0; i < nn; i++) {
            float x=fd[i*3], y=fd[i*3+1], z=fd[i*3+2];
            cfg.nodes[i] = {x, y, z};
            cfg.nmin.x=std::min(cfg.nmin.x,x); cfg.nmin.y=std::min(cfg.nmin.y,y);
            cfg.nmin.z=std::min(cfg.nmin.z,z);
            cfg.nmax.x=std::max(cfg.nmax.x,x); cfg.nmax.y=std::max(cfg.nmax.y,y);
            cfg.nmax.z=std::max(cfg.nmax.z,z);
        }
        std::cout << "MeshNode: " << nn << " nodes\n";
    }

    // Decode mesh faces: prefer MeshSurf, fall back to MeshElem
    if (sh.contains("MeshSurf")) {
        auto& ms = sh["MeshSurf"];
        auto dims = ms["_ArraySize_"].get<std::vector<size_t>>();
        size_t nf = dims[0], nc = dims[1];
        if (nc != 4)
            throw std::runtime_error("MeshSurf must have 4 columns [n1 n2 n3 region]");
        auto raw = jdata_decode(ms);
        const int32_t* sd = reinterpret_cast<const int32_t*>(raw.data());
        load_mesh_surf(cfg.nodes, sd, nf, cfg.faces, cfg.facedata);
    } else if (sh.contains("MeshElem")) {
        auto& me = sh["MeshElem"];
        auto dims = me["_ArraySize_"].get<std::vector<size_t>>();
        size_t ne = dims[0], nc = dims[1];
        if (nc != 5)
            throw std::runtime_error("MeshElem must have 5 columns [n1 n2 n3 n4 region]");
        auto raw = jdata_decode(me);
        const int32_t* ed = reinterpret_cast<const int32_t*>(raw.data());
        std::cout << "MeshElem: " << ne << " tetrahedra\n";
        extract_surface_from_tet(cfg.nodes, ed, ne, cfg.faces, cfg.facedata);
    } else {
        throw std::runtime_error("JSON needs either 'MeshSurf' or 'MeshElem'");
    }

    // InitElem
    if (sh.contains("InitElem")) {
        cfg.init_elem = sh["InitElem"].get<int>();
        cfg.mediumid0 = 0xFFFFFFFFu; // runtime detection for Single-AS
    }

    return cfg;
}

// ============================================================================
// Save fluence output as JData JSON
//
// Saves a 3D (or 4D with time gates) float array:
//   dim = [nx, ny, nz] or [nx, ny, nz, maxgate]
//   type = "single" (float32)
// ============================================================================

void save_json_output(const char* filepath,
                      const SimConfig& cfg,
                      const float* data,
                      uint32_t nx, uint32_t ny, uint32_t nz,
                      int maxgate)
{
    size_t total = (size_t)nx * ny * nz * maxgate;
    size_t data_bytes = total * sizeof(float);

    // Flat buffer index = ix + iy*nx + iz*nx*ny + igate*nx*ny*nz
    // _ArrayOrder_: "c" tells decoder data is C row-major
    // _ArraySize_ lists logical dimensions: [nx, ny, nz, maxgate]
    std::vector<size_t> dims;
    if (maxgate > 1)
        dims = {(size_t)nx, (size_t)ny, (size_t)nz, (size_t)maxgate};
    else
        dims = {(size_t)nx, (size_t)ny, (size_t)nz};

    json root;

    // Copy forward metadata
    root["Session"]["ID"] = cfg.session_id;
    root["Session"]["Photons"] = cfg.nphoton;

    root["Forward"]["T0"] = cfg.t0;
    root["Forward"]["T1"] = cfg.t1;
    root["Forward"]["Dt"] = cfg.dt;

    // Grid info
    root["Domain"]["LengthUnit"] = cfg.unitinmm;
    root["Domain"]["Dim"] = {nx, ny, nz};
    root["Domain"]["Origin"] = {cfg.nmin.x, cfg.nmin.y, cfg.nmin.z};

    // Fluence data as JData array
    root["Fluence"] = jdata_encode("single", dims, data, data_bytes);

    // Write JSON
    std::ofstream f(filepath);
    if (!f) throw std::runtime_error(std::string("Cannot write: ") + filepath);
    f << root.dump(2) << std::endl;

    std::cout << "Output saved to " << filepath << " (";
    for (size_t i = 0; i < dims.size(); i++)
        std::cout << (i ? "x" : "") << dims[i];
    std::cout << " float32, " << data_bytes << " bytes uncompressed)\n";
}

#endif // VK_RTMMC_IO_H