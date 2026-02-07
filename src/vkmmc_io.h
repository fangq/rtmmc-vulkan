/*
 * vkmmc_io.h — Load/save MMC JSON input/output with JData-encoded arrays
 * Supports mesh mode (MeshNode/MeshElem/MeshSurf) and CSG shape mode
 */
#ifndef VKMMC_IO_H
#define VKMMC_IO_H

#include <nlohmann/json.hpp>
#include <miniz.h>
#include <vector>
#include <string>
#include <array>
#include <map>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <fstream>

using json = nlohmann::json;

// ============================================================================
// Data structures
// ============================================================================
struct Vec3 {
    float x, y, z;
};
struct Medium {
    float mua, mus, g, n;
};
struct FaceData {
    float nx, ny, nz, packed_media;
};

struct SimConfig {
    std::string session_id;
    uint64_t nphoton;
    uint32_t rng_seed;
    bool do_mismatch, do_normalize;
    int output_type;
    float t0, t1, dt;
    int maxgate;
    float unitinmm;
    std::vector<Medium> media;
    int srctype;
    float srcpos[3], srcdir[4], srcparam1[4], srcparam2[4];
    std::vector<Vec3> nodes;
    std::vector<std::array<uint32_t, 3> > faces;
    std::vector<FaceData> facedata;
    Vec3 nmin, nmax;
    int init_elem;
    uint32_t mediumid0;
    uint32_t grid_dim[3];
    bool has_grid_dim;
    float steps[3];
    bool has_steps;
    bool is_csg;
    std::vector<uint32_t> face_shape_id;

    SimConfig() : nphoton(1000000), rng_seed(29012391), do_mismatch(true),
        do_normalize(true), output_type(2), t0(0), t1(5e-9f), dt(5e-9f),
        maxgate(1), unitinmm(1.0f), srctype(0), init_elem(-1),
        mediumid0(0xFFFFFFFFu), has_grid_dim(false), has_steps(false), is_csg(false) {
        memset(srcpos, 0, sizeof(srcpos));
        memset(srcdir, 0, sizeof(srcdir));
        srcdir[2] = 1.0f;
        memset(srcparam1, 0, sizeof(srcparam1));
        memset(srcparam2, 0, sizeof(srcparam2));
        memset(grid_dim, 0, sizeof(grid_dim));
        memset(steps, 0, sizeof(steps));
        nmin.x = nmin.y = nmin.z = 1e30f;
        nmax.x = nmax.y = nmax.z = -1e30f;
    }
};

// ============================================================================
// Base64
// ============================================================================
static const uint8_t b64_dec[256] = {
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 0, 64, 64, 64, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64,
    64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64
};
static const char b64_enc[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::vector<uint8_t> base64_decode(const std::string& in) {
    std::string cl;
    cl.reserve(in.size());

    for (size_t i = 0; i < in.size(); i++) if (b64_dec[(uint8_t)in[i]] < 64 || in[i] == '=') {
            cl += in[i];
        }

    size_t len = cl.size();

    if (len % 4) {
        throw std::runtime_error("bad base64");
    }

    size_t ol = len / 4 * 3;

    if (len >= 1 && cl[len - 1] == '=') {
        ol--;
    }

    if (len >= 2 && cl[len - 2] == '=') {
        ol--;
    }

    std::vector<uint8_t> out(ol);
    size_t j = 0;

    for (size_t i = 0; i < len; i += 4) {
        uint32_t a = b64_dec[(uint8_t)cl[i]], b = b64_dec[(uint8_t)cl[i + 1]];
        uint32_t c = (i + 2 < len) ? b64_dec[(uint8_t)cl[i + 2]] : 0, d = (i + 3 < len) ? b64_dec[(uint8_t)cl[i + 3]] : 0;
        uint32_t t = (a << 18) | (b << 12) | (c << 6) | d;

        if (j < ol) {
            out[j++] = (t >> 16) & 0xFF;
        }

        if (j < ol) {
            out[j++] = (t >> 8) & 0xFF;
        }

        if (j < ol) {
            out[j++] = t & 0xFF;
        }
    }

    return out;
}

static std::string base64_encode(const uint8_t* data, size_t len) {
    std::string out;
    out.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len; i += 3) {
        uint32_t b = ((uint32_t)data[i]) << 16;

        if (i + 1 < len) {
            b |= ((uint32_t)data[i + 1]) << 8;
        }

        if (i + 2 < len) {
            b |= (uint32_t)data[i + 2];
        }

        out += b64_enc[(b >> 18) & 0x3F];
        out += b64_enc[(b >> 12) & 0x3F];
        out += (i + 1 < len) ? b64_enc[(b >> 6) & 0x3F] : '=';
        out += (i + 2 < len) ? b64_enc[b & 0x3F] : '=';
    }

    return out;
}

// ============================================================================
// JData decode/encode
// ============================================================================
static size_t jdata_elem_size(const std::string& t) {
    if (t == "single" || t == "float32" || t == "int32" || t == "uint32") {
        return 4;
    }

    if (t == "double" || t == "float64") {
        return 8;
    }

    if (t == "int16" || t == "uint16") {
        return 2;
    }

    return 4;
}

static std::vector<uint8_t> jdata_decode(const json& jd) {
    if (jd.value("_ArrayZipType_", "") != "zlib") {
        throw std::runtime_error("need zlib");
    }

    std::vector<uint8_t> comp = base64_decode(jd["_ArrayZipData_"].get<std::string>());
    size_t ne = jd["_ArrayZipSize_"].get<size_t>();
    size_t esz = jdata_elem_size(jd["_ArrayType_"].get<std::string>());
    size_t usz = ne * esz;
    std::vector<uint8_t> out(usz);
    mz_ulong dl = (mz_ulong)usz;

    if (mz_uncompress(out.data(), &dl, comp.data(), (mz_ulong)comp.size()) != MZ_OK) {
        throw std::runtime_error("zlib fail");
    }

    return out;
}

static json jdata_encode(const std::string& atype, const std::vector<size_t>& asize,
                         const void* data, size_t bytes, bool row_major = true) {
    mz_ulong cb = mz_compressBound((mz_ulong)bytes);
    std::vector<uint8_t> comp(cb);
    mz_ulong cl = cb;

    if (mz_compress(comp.data(), &cl, (const unsigned char*)data, (mz_ulong)bytes) != MZ_OK) {
        throw std::runtime_error("zlib compress fail");
    }

    comp.resize(cl);
    size_t total = 1;

    for (size_t i = 0; i < asize.size(); i++) {
        total *= asize[i];
    }

    json jd;
    jd["_ArrayType_"] = atype;
    jd["_ArraySize_"] = asize;

    if (row_major) {
        jd["_ArrayOrder_"] = "c";
    }

    jd["_ArrayZipType_"] = "zlib";
    jd["_ArrayZipSize_"] = total;
    jd["_ArrayZipData_"] = base64_encode(comp.data(), comp.size());
    return jd;
}

// ============================================================================
// Helpers
// ============================================================================
static Vec3 v3cross(Vec3 a, Vec3 b) {
    return {a.y* b.z - a.z * b.y, a.z* b.x - a.x * b.z, a.x* b.y - a.y * b.x};
}
static float v3len(Vec3 v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

static float pack_media(uint32_t front, uint32_t back) {
    uint32_t p = (front << 16) | (back & 0xFFFF);
    float f;
    memcpy(&f, &p, 4);
    return f;
}

struct FaceKey {
    uint32_t v[3];
    bool operator<(const FaceKey& o) const {
        if (v[0] != o.v[0]) {
            return v[0] < o.v[0];
        }

        if (v[1] != o.v[1]) {
            return v[1] < o.v[1];
        }

        return v[2] < o.v[2];
    }
};

// ============================================================================
// Load MeshSurf
// ============================================================================
static void load_mesh_surf(const std::vector<Vec3>& nodes, const int32_t* sd, size_t nf,
                           std::vector<std::array<uint32_t, 3> >& out_f, std::vector<FaceData>& out_d) {
    struct Entry {
        uint32_t v[3];
        uint32_t region;
    };
    std::map<FaceKey, std::vector<Entry> > fmap;

    for (size_t i = 0; i < nf; i++) {
        const int32_t* r = sd + i * 4;
        Entry e;
        e.v[0] = (uint32_t)(r[0] - 1);
        e.v[1] = (uint32_t)(r[1] - 1);
        e.v[2] = (uint32_t)(r[2] - 1);
        e.region = (uint32_t)r[3];
        FaceKey k;
        k.v[0] = e.v[0];
        k.v[1] = e.v[1];
        k.v[2] = e.v[2];
        std::sort(k.v, k.v + 3);
        fmap[k].push_back(e);
    }

    out_f.clear();
    out_d.clear();

    for (std::map<FaceKey, std::vector<Entry> >::iterator it = fmap.begin(); it != fmap.end(); ++it) {
        std::vector<Entry>& ents = it->second;
        uint32_t front = (ents.size() >= 2) ? ents[1].region : 0u, back = ents[0].region;

        if (front == back) {
            continue;
        }

        Entry& e0 = ents[0];
        Vec3 a = nodes[e0.v[0]], b = nodes[e0.v[1]], c = nodes[e0.v[2]];
        Vec3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z}, e2 = {c.x - a.x, c.y - a.y, c.z - a.z};
        Vec3 N = v3cross(e1, e2);
        float l = v3len(N);

        if (l > 0) {
            N.x /= l;
            N.y /= l;
            N.z /= l;
        }

        out_f.push_back({{e0.v[0], e0.v[1], e0.v[2]}});
        out_d.push_back({N.x, N.y, N.z, pack_media(front, back)});
    }

    std::cout << "MeshSurf: " << nf << " -> " << out_f.size() << " unique triangles\n";
}

// ============================================================================
// Extract surface from tetrahedral mesh
// ============================================================================
static void extract_surface_from_tet(const std::vector<Vec3>& nodes, const int32_t* elems,
                                     size_t ne, std::vector<std::array<uint32_t, 3> >& out_f, std::vector<FaceData>& out_d) {
    static const int ftab[4][3] = {{0, 3, 1}, {3, 2, 1}, {0, 2, 3}, {0, 1, 2}};
    struct FInfo {
        uint32_t v[3];
        int r1, r2;
    };
    std::map<FaceKey, FInfo> fmap;

    for (size_t e = 0; e < ne; e++) {
        const int32_t* row = elems + e * 5;
        uint32_t n[4] = {(uint32_t)(row[0] - 1), (uint32_t)(row[1] - 1), (uint32_t)(row[2] - 1), (uint32_t)(row[3] - 1)};
        int reg = row[4];

        for (int f = 0; f < 4; f++) {
            uint32_t fv[3] = {n[ftab[f][0]], n[ftab[f][1]], n[ftab[f][2]]};
            FaceKey k;
            k.v[0] = fv[0];
            k.v[1] = fv[1];
            k.v[2] = fv[2];
            std::sort(k.v, k.v + 3);
            std::map<FaceKey, FInfo>::iterator it = fmap.find(k);

            if (it == fmap.end()) {
                FInfo fi;
                fi.v[0] = fv[0];
                fi.v[1] = fv[1];
                fi.v[2] = fv[2];
                fi.r1 = reg;
                fi.r2 = -1;
                fmap[k] = fi;
            } else {
                it->second.r2 = reg;
            }
        }
    }

    out_f.clear();
    out_d.clear();

    for (std::map<FaceKey, FInfo>::iterator it = fmap.begin(); it != fmap.end(); ++it) {
        FInfo& fi = it->second;

        if (fi.r2 >= 0 && fi.r1 == fi.r2) {
            continue;
        }

        Vec3 a = nodes[fi.v[0]], b = nodes[fi.v[1]], c = nodes[fi.v[2]];
        Vec3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z}, e2 = {c.x - a.x, c.y - a.y, c.z - a.z};
        Vec3 N = v3cross(e1, e2);
        float l = v3len(N);

        if (l > 0) {
            N.x /= l;
            N.y /= l;
            N.z /= l;
        }

        uint32_t front = (fi.r2 >= 0) ? (uint32_t)fi.r2 : 0u, back = (uint32_t)fi.r1;
        out_f.push_back({{fi.v[0], fi.v[1], fi.v[2]}});
        out_d.push_back({N.x, N.y, N.z, pack_media(front, back)});
    }

    std::cout << "Extracted " << out_f.size() << " surface triangles from " << ne << " tetrahedra\n";
}

// ============================================================================
// Parse helpers
// ============================================================================
static int parse_srctype(const std::string& s) {
    if (s == "pencil") {
        return 0;
    }

    if (s == "isotropic") {
        return 1;
    }

    if (s == "cone") {
        return 2;
    }

    if (s == "gaussian") {
        return 3;
    }

    if (s == "planar") {
        return 4;
    }

    if (s == "pattern") {
        return 5;
    }

    if (s == "fourier") {
        return 6;
    }

    if (s == "arcsine") {
        return 7;
    }

    if (s == "disk") {
        return 8;
    }

    if (s == "zgaussian") {
        return 11;
    }

    if (s == "line") {
        return 12;
    }

    if (s == "slit") {
        return 13;
    }

    return 0;
}

static int parse_outputtype(const std::string& s) {
    if (s.empty()) {
        return 2;
    }

    switch (s[0]) {
        case 'f':
            return 0;

        case 'l':
            return 1;

        case 'x':
            return 2;
    }

    return 2;
}

static void update_bbox(SimConfig& cfg) {
    cfg.nmin.x = cfg.nmin.y = cfg.nmin.z = 1e30f;
    cfg.nmax.x = cfg.nmax.y = cfg.nmax.z = -1e30f;

    for (size_t i = 0; i < cfg.nodes.size(); i++) {
        Vec3& v = cfg.nodes[i];

        if (v.x < cfg.nmin.x) {
            cfg.nmin.x = v.x;
        }

        if (v.y < cfg.nmin.y) {
            cfg.nmin.y = v.y;
        }

        if (v.z < cfg.nmin.z) {
            cfg.nmin.z = v.z;
        }

        if (v.x > cfg.nmax.x) {
            cfg.nmax.x = v.x;
        }

        if (v.y > cfg.nmax.y) {
            cfg.nmax.y = v.y;
        }

        if (v.z > cfg.nmax.z) {
            cfg.nmax.z = v.z;
        }
    }
}

// ============================================================================
// Load nodes (JData or inline JSON array)
// ============================================================================
static void load_nodes(const json& mn, SimConfig& cfg) {
    if (mn.contains("_ArraySize_")) {
        std::vector<size_t> dims;
        json da = mn["_ArraySize_"];

        for (size_t i = 0; i < da.size(); i++) {
            dims.push_back(da[i].get<size_t>());
        }

        size_t nn = dims[0];
        std::vector<uint8_t> raw = jdata_decode(mn);
        const float* fd = reinterpret_cast<const float*>(raw.data());
        cfg.nodes.resize(nn);

        for (size_t i = 0; i < nn; i++) cfg.nodes[i] = {fd[i * 3], fd[i * 3 + 1], fd[i * 3 + 2]};

        std::cout << "MeshNode: " << nn << " nodes\n";
    } else if (mn.is_array()) {
        size_t nn = mn.size();
        cfg.nodes.resize(nn);

        for (size_t i = 0; i < nn; i++)
            cfg.nodes[i] = {mn[i][0].get<float>(), mn[i][1].get<float>(), mn[i][2].get<float>()};

        std::cout << "MeshNode: " << nn << " nodes (inline)\n";
    }

    update_bbox(cfg);
}

// ============================================================================
// Load elements (JData or inline), returns raw int32 buffer + count/cols
// ============================================================================
static void load_elems(const json& me, std::vector<uint8_t>& raw_out,
                       const int32_t*& data_out, size_t& count_out, size_t& cols_out) {
    if (me.contains("_ArraySize_")) {
        json da = me["_ArraySize_"];
        count_out = da[0].get<size_t>();
        cols_out = da[1].get<size_t>();
        raw_out = jdata_decode(me);
        data_out = reinterpret_cast<const int32_t*>(raw_out.data());
    } else if (me.is_array()) {
        count_out = me.size();
        cols_out = me[0].size();
        raw_out.resize(count_out * cols_out * sizeof(int32_t));
        int32_t* buf = reinterpret_cast<int32_t*>(raw_out.data());

        for (size_t i = 0; i < count_out; i++)
            for (size_t j = 0; j < cols_out; j++) {
                buf[i * cols_out + j] = me[i][j].get<int32_t>();
            }

        data_out = buf;
    }
}

// ============================================================================
// Main JSON loader
// ============================================================================
static SimConfig load_json_input(const char* filepath) {
    std::ifstream f(filepath);

    if (!f) {
        throw std::runtime_error(std::string("Cannot open: ") + filepath);
    }

    json j;
    f >> j;
    SimConfig cfg;

    // Session
    if (j.contains("Session")) {
        json& s = j["Session"];
        cfg.session_id = s.value("ID", "default");
        cfg.nphoton = s.value("Photons", (uint64_t)1000000);
        cfg.rng_seed = s.value("RNGSeed", (uint32_t)29012391);
        cfg.do_mismatch = s.value("DoMismatch", true);
        cfg.do_normalize = s.value("DoNormalize", true);
        cfg.output_type = parse_outputtype(s.value("OutputType", "x"));
    }

    // Forward
    if (j.contains("Forward")) {
        json& fw = j["Forward"];
        double d0 = fw.value("T0", 0.0), d1 = fw.value("T1", 5e-9), dd = fw.value("Dt", 5e-9);
        cfg.t0 = (float)d0;
        cfg.t1 = (float)d1;
        cfg.dt = (float)dd;
        cfg.maxgate = (int)((d1 - d0) / dd + 0.5);

        if (cfg.maxgate < 1) {
            cfg.maxgate = 1;
        }

        printf("Forward: T0=%.4e T1=%.4e Dt=%.4e maxgate=%d\n", d0, d1, dd, cfg.maxgate);
    }

    // Domain
    if (j.contains("Domain")) {
        json& d = j["Domain"];
        cfg.unitinmm = d.value("LengthUnit", 1.0f);

        if (d.contains("Media")) {
            cfg.media.clear();
            json ma = d["Media"];

            if (ma.is_array() && ma.size() > 0 && ma[0].is_array()) {
                ma = ma[0];    // unwrap [[...]]
            }

            for (size_t i = 0; i < ma.size(); i++) {
                json& m = ma[i];
                Medium med = {m.value("mua", 0.f)* cfg.unitinmm, m.value("mus", 0.f)* cfg.unitinmm,
                              m.value("g", 1.f), m.value("n", 1.f)
                             };
                cfg.media.push_back(med);
                printf("  Media[%zu]: mua=%.6f mus=%.6f g=%.4f n=%.4f\n", i, med.mua, med.mus, med.g, med.n);
            }
        }

        if (d.contains("Dim")) {
            json dm = d["Dim"];
            cfg.grid_dim[0] = dm[0].get<uint32_t>();
            cfg.grid_dim[1] = dm[1].get<uint32_t>();
            cfg.grid_dim[2] = dm[2].get<uint32_t>();
            cfg.has_grid_dim = true;
        }

        if (d.contains("Steps")) {
            json st = d["Steps"];

            for (int i = 0; i < (int)st.size() && i < 3; i++) {
                cfg.steps[i] = st[i].get<float>();
            }

            cfg.has_steps = true;
        }
    }

    // Optode / Source
    if (j.contains("Optode") && j["Optode"].contains("Source")) {
        json& src = j["Optode"]["Source"];
        cfg.srctype = parse_srctype(src.value("Type", "pencil"));

        if (src.contains("Pos")) {
            json p = src["Pos"];
            cfg.srcpos[0] = p[0];
            cfg.srcpos[1] = p[1];
            cfg.srcpos[2] = p[2];
        }

        if (src.contains("Dir")) {
            json d = src["Dir"];

            for (int i = 0; i < (int)d.size() && i < 4; i++) {
                cfg.srcdir[i] = d[i];
            }
        }

        if (src.contains("Param1")) {
            json p = src["Param1"];

            for (int i = 0; i < (int)p.size() && i < 4; i++) {
                cfg.srcparam1[i] = p[i];
            }
        }

        if (src.contains("Param2")) {
            json p = src["Param2"];

            for (int i = 0; i < (int)p.size() && i < 4; i++) {
                cfg.srcparam2[i] = p[i];
            }
        }
    }

    // Shapes
    if (!j.contains("Shapes")) {
        throw std::runtime_error("JSON missing 'Shapes'");
    }

    json shapes = j["Shapes"];

    if (shapes.is_array()) {
        // ---- CSG shape mode ----
        cfg.is_csg = true;
        cfg.mediumid0 = 0xFFFFFFFFu;
        // Shape mesh generation is handled externally via vkmmc_shapes.h
        // Store the shapes JSON for later processing
        // For now, we just note it's CSG mode; the caller will invoke parse_shapes()
        printf("Shapes: CSG mode detected (array of shape constructs)\n");
    } else {
        // ---- Mesh mode ----
        cfg.is_csg = false;

        // Load nodes
        if (!shapes.contains("MeshNode")) {
            throw std::runtime_error("Mesh mode requires MeshNode");
        }

        load_nodes(shapes["MeshNode"], cfg);

        // Load faces
        std::vector<uint8_t> elem_raw;
        const int32_t* elem_data = NULL;
        size_t elem_count = 0, elem_cols = 0;

        if (shapes.contains("MeshSurf")) {
            json& ms = shapes["MeshSurf"];

            if (ms.contains("_ArraySize_")) {
                json da = ms["_ArraySize_"];
                size_t nf = da[0].get<size_t>();
                std::vector<uint8_t> raw = jdata_decode(ms);
                load_mesh_surf(cfg.nodes, reinterpret_cast<const int32_t*>(raw.data()), nf, cfg.faces, cfg.facedata);
            }
        } else if (shapes.contains("MeshElem")) {
            load_elems(shapes["MeshElem"], elem_raw, elem_data, elem_count, elem_cols);

            if (elem_data && elem_cols == 5) {
                std::cout << "MeshElem: " << elem_count << " tetrahedra\n";
                extract_surface_from_tet(cfg.nodes, elem_data, elem_count, cfg.faces, cfg.facedata);
            }
        } else {
            throw std::runtime_error("Mesh mode requires MeshSurf or MeshElem");
        }

        // InitElem
        if (shapes.contains("InitElem")) {
            cfg.init_elem = shapes["InitElem"].get<int>();

            if (elem_data && cfg.init_elem > 0 && cfg.init_elem <= (int)elem_count) {
                int region = elem_data[(cfg.init_elem - 1) * elem_cols + (elem_cols - 1)];
                cfg.mediumid0 = (uint32_t)region;
                printf("InitElem: %d, medium type: %u\n", cfg.init_elem, cfg.mediumid0);
            }
        }
    }

    return cfg;
}

#endif // VKMMC_IO_H