/*
 * vkmmc_io.h — Load/save MMC JSON input/output with JData-encoded arrays
 * Supports mesh mode (MeshNode/MeshElem/MeshSurf) and CSG shape mode
 *
 * All input is validated here: a malformed or out-of-range input raises a
 * std::runtime_error with a message naming the offending field, so that the
 * GPU kernels can assume every index, label and property they read is valid.
 */
#ifndef VKMMC_IO_H
#define VKMMC_IO_H

#include <nlohmann/json.hpp>
#include "zmatlib.h"
#include <vector>
#include <string>
#include <array>
#include <map>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <cctype>

using json = nlohmann::json;

// ============================================================================
// Limits
// ============================================================================
#define VKMMC_MAX_MEDIA      256         /* OptiX keeps media in constant memory */
#define VKMMC_MAX_CSG_TAG    64          /* CSG enclosure walk tracks 64 tags */
#define VKMMC_MAX_MESH_LABEL 0x3FFFu     /* front labels have 14 bits */

/* packed_media bit layout of a triangle (stored as float bits for the GPU):
   bit 31: flat (no curvature correction), bit 30: coplanar with another CSG
   surface (the kernel re-derives the enclosing shapes after crossing it),
   bits 29-16: front medium, bits 15-0: back medium */
#define VKMMC_FACE_FLAT     0x80000000u
#define VKMMC_FACE_COPLANAR 0x40000000u
#define VKMMC_MIN_MESHRES    4
#define VKMMC_MAX_MESHRES    2048
#define VKMMC_MAX_GATES      100000
#define VKMMC_MAX_VOXELS     ((uint64_t)1 << 30)  /* voxels x gates; kernel indexes with uint32 */
#define VKMMC_MAX_JDATA_DIM  8
#define VKMMC_MAX_COORD      1e6         /* |coordinates| in length units; float precision */
#define VKMMC_MAX_SOURCES    1024        /* rows of a multi-source list */
#define VKMMC_MAX_PATTERNS   1024        /* SrcNum: patterns sharing the photons */
#define VKMMC_MAX_PATTERN_LEN (1u << 26) /* pattern values (pixels x SrcNum) */
#define VKMMC_MAX_SRC_WEIGHT 1000.0      /* launch weight; keeps the 64-bit fixed-point sums in range */
#define VKMMC_MAX_TIME       1.0         /* seconds */

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

/* the source types and numbering of MCX/MMC */
enum SrcType {
    SRC_PENCIL = 0, SRC_ISOTROPIC = 1, SRC_CONE = 2, SRC_GAUSSIAN = 3, SRC_PLANAR = 4,
    SRC_PATTERN = 5, SRC_FOURIER = 6, SRC_ARCSINE = 7, SRC_DISK = 8, SRC_FOURIERX = 9,
    SRC_FOURIERX2D = 10, SRC_ZGAUSSIAN = 11, SRC_LINE = 12, SRC_SLIT = 13, SRC_PENCILARRAY = 14,
    SRC_PATTERN3D = 15, SRC_HYPERBOLOID = 16, SRC_RING = 17, SRC_TYPE_COUNT = 18
};

/* one source of a multi-source list: pos (.w = weight), dir (.w = focal length), param1, param2 */
struct ExtraSrc {
    float pos[4], dir[4], param1[4], param2[4];
};

enum OutputType { OT_FLUX = 0, OT_FLUENCE = 1, OT_ENERGY = 2 };

struct SimConfig {
    std::string session_id;
    uint64_t nphoton;
    uint32_t rng_seed;
    bool do_mismatch, do_normalize;
    int output_type;
    float t0, t1, dt;
    int maxgate;
    float unitinmm;
    std::vector<Medium> media;          /* as given in the input, in 1/mm */
    int srctype;
    float srcpos[3], srcdir[4], srcparam1[4], srcparam2[4];   /* srcdir[3]: focal length */
    int srcnum;                         /* patterns sharing the photons of a pattern source */
    std::vector<float> srcpattern;      /* pattern intensities, srcnum per pixel (pixel-major) */
    std::vector<ExtraSrc> extrasrc;     /* multi-source list (Pos with several rows) */
    int srcid;                          /* multi-source: -1 a field per source, k > 0 source k only */
    std::vector<Vec3> nodes;
    std::vector<std::array<uint32_t, 3> > faces;
    std::vector<FaceData> facedata;
    Vec3 nmin, nmax;
    int init_elem;
    uint32_t mediumid0;
    float steps[3];
    bool has_steps;
    bool is_csg;
    json shapes;                        /* CSG mode: the Shapes array */
    std::vector<uint32_t> face_shape_id;
    int mesh_res;
    float minenergy;
    float roulettesize;
    uint32_t debuglevel;    /* bitmask; bit 13 (8192 = dlTraj) enables trajectory */
    uint32_t maxjumpdebug;  /* max trajectory events to record */

    SimConfig() : session_id("default"), nphoton(1000000), rng_seed(29012391), do_mismatch(true),
        do_normalize(true), output_type(OT_ENERGY), t0(0), t1(5e-9f), dt(5e-9f),
        maxgate(1), unitinmm(1.0f), srctype(SRC_PENCIL), srcnum(1), srcid(0), init_elem(-1),
        mediumid0(0xFFFFFFFFu), has_steps(false), is_csg(false), mesh_res(24), minenergy(1e-6f),
        roulettesize(10.0f), debuglevel(0), maxjumpdebug(10000000) {
        memset(srcpos, 0, sizeof(srcpos));
        memset(srcdir, 0, sizeof(srcdir));
        srcdir[2] = 1.0f;
        memset(srcparam1, 0, sizeof(srcparam1));
        memset(srcparam2, 0, sizeof(srcparam2));
        memset(steps, 0, sizeof(steps));
        nmin.x = nmin.y = nmin.z = 1e30f;
        nmax.x = nmax.y = nmax.z = -1e30f;
    }
};

// ============================================================================
// Error helpers
// ============================================================================
static void vkmmc_fail(const std::string& msg) {
    throw std::runtime_error(msg);
}

static void vkmmc_require(bool cond, const std::string& msg) {
    if (!cond) {
        vkmmc_fail(msg);
    }
}

/* Read a finite number from obj[key], or return def when the key is absent */
static double json_num(const json& obj, const char* key, double def, const std::string& ctx) {
    if (!obj.is_object() || !obj.contains(key) || obj[key].is_null()) {
        return def;
    }

    const json& v = obj[key];

    if (v.is_boolean()) {
        return v.get<bool>() ? 1.0 : 0.0;
    }

    vkmmc_require(v.is_number(), ctx + "." + key + " must be a number");
    double d = v.get<double>();
    vkmmc_require(std::isfinite(d), ctx + "." + key + " must be finite");
    return d;
}

static double json_num_range(const json& obj, const char* key, double def, double lo, double hi,
                             const std::string& ctx) {
    double d = json_num(obj, key, def, ctx);

    if (!(d >= lo && d <= hi)) {
        std::ostringstream ss;
        ss << ctx << "." << key << "=" << d << " is out of range [" << lo << ", " << hi << "]";
        vkmmc_fail(ss.str());
    }

    return d;
}

static bool json_bool(const json& obj, const char* key, bool def, const std::string& ctx) {
    if (!obj.is_object() || !obj.contains(key) || obj[key].is_null()) {
        return def;
    }

    const json& v = obj[key];

    if (v.is_boolean()) {
        return v.get<bool>();
    }

    vkmmc_require(v.is_number(), ctx + "." + key + " must be a boolean or a number");
    return v.get<double>() != 0.0;
}

static std::string json_str(const json& obj, const char* key, const std::string& def, const std::string& ctx) {
    if (!obj.is_object() || !obj.contains(key) || obj[key].is_null()) {
        return def;
    }

    vkmmc_require(obj[key].is_string(), ctx + "." + key + " must be a string");
    return obj[key].get<std::string>();
}

/* Read an array of at least nmin and at most nmax finite numbers */
static std::vector<double> json_vec(const json& v, size_t nmin, size_t nmax, const std::string& ctx) {
    vkmmc_require(v.is_array(), ctx + " must be an array");
    vkmmc_require(v.size() >= nmin, ctx + " needs at least " + std::to_string(nmin) + " numbers");
    vkmmc_require(v.size() <= nmax, ctx + " has more than " + std::to_string(nmax) + " numbers");
    std::vector<double> out;

    for (size_t i = 0; i < v.size() && i < nmax; i++) {
        vkmmc_require(v[i].is_number(), ctx + "[" + std::to_string(i) + "] must be a number");
        double d = v[i].get<double>();
        vkmmc_require(std::isfinite(d), ctx + "[" + std::to_string(i) + "] must be finite");
        out.push_back(d);
    }

    return out;
}

/* obj[key] as a fixed-length vector; key must be present */
static std::vector<double> json_vec_key(const json& obj, const char* key, size_t n, const std::string& ctx) {
    vkmmc_require(obj.is_object() && obj.contains(key), ctx + " requires '" + key + "'");
    return json_vec(obj[key], n, n, ctx + "." + key);
}

// ============================================================================
// Base64
// ============================================================================
static const uint8_t b64_dec[256] = {
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 64, 64, 64, 64, 0, 1, 2, 3, 4, 5, 6,
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

/*
 * Decode base64, skipping whitespace and line breaks. Padding is optional;
 * any other character, a misplaced '=' or a dangling 6-bit group is an error.
 */
static std::vector<uint8_t> base64_decode(const std::string& in) {
    std::vector<uint8_t> out;
    out.reserve(in.size() / 4 * 3 + 3);
    uint32_t acc = 0;
    int nbits = 0, npad = 0;
    size_t nsym = 0;

    for (size_t i = 0; i < in.size(); i++) {
        uint8_t c = (uint8_t)in[i];

        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            continue;
        }

        if (c == '=') {
            npad++;
            continue;
        }

        if (npad || b64_dec[c] >= 64) {
            throw std::runtime_error("bad base64: invalid character or data after padding");
        }

        acc = (acc << 6) | b64_dec[c];
        nbits += 6;
        nsym++;

        if (nbits >= 8) {
            nbits -= 8;
            out.push_back((uint8_t)((acc >> nbits) & 0xFF));
        }
    }

    if (nsym % 4 == 1 || npad > 2 || (npad && (nsym + npad) % 4)) {
        throw std::runtime_error("bad base64: truncated input");
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

/* threads used by zmat to deflate/inflate zlib/gzip blocks in parallel */
static int jdata_nthread() {
    unsigned int n = std::thread::hardware_concurrency();
    return (int)std::max(1u, std::min(n, 16u));
}

/* element size in bytes of a JData _ArrayType_, 0 if unsupported */
static size_t jdata_elem_size(const std::string& t) {
    if (t == "single" || t == "float32" || t == "int32" || t == "uint32") {
        return 4;
    }

    if (t == "double" || t == "float64" || t == "int64" || t == "uint64") {
        return 8;
    }

    if (t == "int16" || t == "uint16") {
        return 2;
    }

    if (t == "int8" || t == "uint8" || t == "logical" || t == "bool") {
        return 1;
    }

    return 0;
}

/* A decoded JData N-D array: raw little-endian bytes in row-major order */
struct JDataArray {
    std::string type;
    std::vector<size_t> dims;
    size_t esize;
    std::vector<uint8_t> raw;

    size_t numel() const {
        size_t n = 1;

        for (size_t i = 0; i < dims.size(); i++) {
            n *= dims[i];
        }

        return n;
    }

    /* element i converted to double */
    double at(size_t i) const {
        const uint8_t* p = raw.data() + i * esize;
        const std::string& t = type;

        if (t == "single" || t == "float32") {
            return get<float>(p);
        } else if (t == "double" || t == "float64") {
            return get<double>(p);
        } else if (t == "int8") {
            return get<int8_t>(p);
        } else if (t == "uint8" || t == "logical" || t == "bool") {
            return get<uint8_t>(p);
        } else if (t == "int16") {
            return get<int16_t>(p);
        } else if (t == "uint16") {
            return get<uint16_t>(p);
        } else if (t == "int32") {
            return get<int32_t>(p);
        } else if (t == "uint32") {
            return get<uint32_t>(p);
        } else if (t == "int64") {
            return (double)get<int64_t>(p);
        } else if (t == "uint64") {
            return (double)get<uint64_t>(p);
        }

        return 0.0;
    }

    template<class T> static T get(const uint8_t* p) {
        T v;
        memcpy(&v, p, sizeof(T));
        return v;
    }
};

/* product of dims with overflow check */
static size_t jdata_numel(const std::vector<size_t>& dims, const std::string& ctx) {
    size_t n = 1;

    for (size_t i = 0; i < dims.size(); i++) {
        if (dims[i] && n > ((size_t)1 << 40) / dims[i]) {
            vkmmc_fail(ctx + ": array is too large");
        }

        n *= dims[i];
    }

    return n;
}

static std::vector<size_t> jdata_dims(const json& v, const std::string& ctx) {
    std::vector<size_t> dims;

    if (v.is_number_unsigned() || (v.is_number_integer() && v.get<int64_t>() >= 0)) {
        dims.push_back(v.get<size_t>());
        return dims;
    }

    vkmmc_require(v.is_array() && v.size() >= 1 && v.size() <= VKMMC_MAX_JDATA_DIM,
                  ctx + " must be a non-negative integer or an array of 1-" + std::to_string(VKMMC_MAX_JDATA_DIM) + " of them");

    for (size_t i = 0; i < v.size(); i++) {
        vkmmc_require(v[i].is_number_unsigned() || (v[i].is_number_integer() && v[i].get<int64_t>() >= 0),
                      ctx + " entries must be non-negative integers");
        dims.push_back(v[i].get<size_t>());
    }

    return dims;
}

/* read the optional _ArrayZipOffsets_ block index: [[comp, uncomp], ...] or flat */
static std::vector<size_t> jdata_zipoffsets(const json& jd) {
    std::vector<size_t> off;

    if (!jd.contains("_ArrayZipOffsets_") || !jd["_ArrayZipOffsets_"].is_array()) {
        return off;
    }

    const json& v = jd["_ArrayZipOffsets_"];

    for (size_t i = 0; i < v.size(); i++) {
        if (v[i].is_array()) {
            for (size_t k = 0; k < v[i].size(); k++) {
                if (!v[i][k].is_number() || v[i][k].get<double>() < 0) {
                    return std::vector<size_t>();
                }

                off.push_back(v[i][k].get<size_t>());
            }
        } else if (v[i].is_number() && v[i].get<double>() >= 0) {
            off.push_back(v[i].get<size_t>());
        } else {
            return std::vector<size_t>();
        }
    }

    /* the index is only a speed hint; zmat verifies it against the stream */
    if (off.size() < 4 || (off.size() & 1)) {
        off.clear();
    }

    return off;
}

/* transpose a column-major N-D buffer to row-major in place */
static void jdata_col2row(JDataArray& a) {
    size_t nd = a.dims.size();

    if (nd < 2) {
        return;
    }

    size_t n = a.numel(), es = a.esize;
    std::vector<uint8_t> out(a.raw.size());
    std::vector<size_t> rstride(nd), cstride(nd), idx(nd, 0);
    rstride[nd - 1] = 1;
    cstride[0] = 1;

    for (size_t d = nd - 1; d > 0; d--) {
        rstride[d - 1] = rstride[d] * a.dims[d];
    }

    for (size_t d = 1; d < nd; d++) {
        cstride[d] = cstride[d - 1] * a.dims[d - 1];
    }

    for (size_t i = 0; i < n; i++) {
        size_t src = 0, dst = 0;

        for (size_t d = 0; d < nd; d++) {
            src += idx[d] * cstride[d];
            dst += idx[d] * rstride[d];
        }

        memcpy(&out[dst * es], &a.raw[src * es], es);

        for (size_t d = nd; d-- > 0;) {
            if (++idx[d] < a.dims[d]) {
                break;
            }

            idx[d] = 0;
        }
    }

    a.raw.swap(out);
}

/*
 * Decode a JData-annotated array: {"_ArrayType_", "_ArraySize_", then either
 * "_ArrayZipType_" (zlib/gzip/base64) + "_ArrayZipSize_" + "_ArrayZipData_",
 * or an uncompressed "_ArrayData_" (numeric array or base64 string)}.
 * The result is always row-major; "_ArrayOrder_":"c"/"col" input is transposed.
 */
static JDataArray jdata_decode_array(const json& jd, const std::string& ctx = "JData array") {
    vkmmc_require(jd.is_object(), ctx + " must be a JData object");
    vkmmc_require(jd.contains("_ArrayType_") && jd["_ArrayType_"].is_string(), ctx + " needs a string _ArrayType_");
    JDataArray a;
    a.type = jd["_ArrayType_"].get<std::string>();
    a.esize = jdata_elem_size(a.type);
    vkmmc_require(a.esize > 0, ctx + ": unsupported _ArrayType_ '" + a.type + "'");
    vkmmc_require(jd.contains("_ArraySize_"), ctx + " needs _ArraySize_");
    a.dims = jdata_dims(jd["_ArraySize_"], ctx + "._ArraySize_");
    size_t numel = jdata_numel(a.dims, ctx), nbytes = numel * a.esize;

    if (jd.contains("_ArrayZipData_")) {
        std::string ztype = json_str(jd, "_ArrayZipType_", "zlib", ctx);
        vkmmc_require(jd["_ArrayZipData_"].is_string(), ctx + "._ArrayZipData_ must be a base64 string");

        if (jd.contains("_ArrayZipSize_")) {
            std::vector<size_t> zdims = jdata_dims(jd["_ArrayZipSize_"], ctx + "._ArrayZipSize_");
            vkmmc_require(jdata_numel(zdims, ctx) == numel, ctx + ": _ArrayZipSize_ does not match _ArraySize_");
        }

        std::vector<uint8_t> comp = base64_decode(jd["_ArrayZipData_"].get<std::string>());

        if (ztype == "base64") {
            a.raw.swap(comp);
        } else {
            vkmmc_require(ztype == "zlib" || ztype == "gzip",
                          ctx + ": unsupported _ArrayZipType_ '" + ztype + "' (zlib, gzip or base64)");
            vkmmc_require(!comp.empty(), ctx + ": empty compressed stream");
            std::vector<size_t> off = jdata_zipoffsets(jd);
            size_t* poff = off.empty() ? NULL : off.data();
            size_t noff = off.size();
            unsigned char* outbuf = NULL;
            size_t outlen = 0;
            int status = 0;
            TZMatFlags flags;
            flags.iscompress = 0;
            flags.param.nthread = (char)jdata_nthread();
            int ret = zmat_run_indexed(comp.size(), comp.data(), &outlen, &outbuf,
                                       ztype == "zlib" ? zmZlib : zmGzip, &status, flags.iscompress,
                                       poff ? &poff : NULL, poff ? &noff : NULL);

            if (ret != 0 || outbuf == NULL) {
                free(outbuf);
                vkmmc_fail(ctx + ": failed to decompress " + ztype + " data");
            }

            a.raw.assign(outbuf, outbuf + outlen);
            free(outbuf);
        }
    } else if (jd.contains("_ArrayData_")) {
        const json& d = jd["_ArrayData_"];

        if (d.is_string()) {
            a.raw = base64_decode(d.get<std::string>());
        } else {
            /* plain numeric array, possibly nested; store as double */
            std::vector<double> vals;

            vkmmc_require(d.is_array(), ctx + "._ArrayData_ must be an array or a base64 string");

            /* flatten in order (iterative to avoid deep recursion on hostile input) */
            std::vector<std::pair<const json*, size_t> > it;
            it.push_back(std::make_pair(&d, (size_t)0));

            while (!it.empty()) {
                const json* cur = it.back().first;
                size_t& k = it.back().second;

                if (k >= cur->size()) {
                    it.pop_back();
                    continue;
                }

                const json& e = (*cur)[k++];

                if (e.is_array()) {
                    vkmmc_require(it.size() < VKMMC_MAX_JDATA_DIM, ctx + "._ArrayData_ is nested too deeply");
                    it.push_back(std::make_pair(&e, (size_t)0));
                } else {
                    vkmmc_require(e.is_number() || e.is_boolean(), ctx + "._ArrayData_ must contain only numbers");
                    vals.push_back(e.is_boolean() ? (e.get<bool>() ? 1.0 : 0.0) : e.get<double>());
                }

                vkmmc_require(vals.size() <= numel, ctx + ": _ArrayData_ has more elements than _ArraySize_");
            }

            vkmmc_require(vals.size() == numel, ctx + ": _ArrayData_ has fewer elements than _ArraySize_");
            a.type = "double";
            a.esize = 8;
            nbytes = numel * 8;
            a.raw.resize(nbytes);

            if (nbytes) {
                memcpy(a.raw.data(), vals.data(), nbytes);
            }
        }
    } else {
        vkmmc_fail(ctx + " needs _ArrayZipData_ or _ArrayData_");
    }

    if (a.raw.size() != nbytes) {
        vkmmc_fail(ctx + ": decoded " + std::to_string(a.raw.size()) + " bytes, expected " +
                   std::to_string(nbytes) + " (" + std::to_string(numel) + " x " + a.type + ")");
    }

    std::string order = json_str(jd, "_ArrayOrder_", "r", ctx);

    if (!order.empty() && (order[0] == 'c' || order[0] == 'C' || order[0] == 'f' || order[0] == 'F')) {
        jdata_col2row(a);
    }

    return a;
}

/* legacy helper: raw decoded bytes of a JData array (row-major) */
static inline std::vector<uint8_t> jdata_decode(const json& jd) {
    return jdata_decode_array(jd).raw;
}

/*
 * Encode an array as JData. Data larger than 4 MB is deflated in parallel
 * blocks (zmat); the stream is a standard zlib stream and its block index is
 * stored in _ArrayZipOffsets_ so that readers can inflate it in parallel too.
 * row_major=false marks data stored with the first index fastest.
 */
static json jdata_encode(const std::string& atype, const std::vector<size_t>& asize,
                         const void* data, size_t bytes, bool row_major = true) {
    size_t total = 1;

    for (size_t i = 0; i < asize.size(); i++) {
        total *= asize[i];
    }

    json jd;
    jd["_ArrayType_"] = atype;
    jd["_ArraySize_"] = asize;

    if (!row_major) {
        jd["_ArrayOrder_"] = "c";
    }

    jd["_ArrayZipType_"] = "zlib";
    jd["_ArrayZipSize_"] = total;

    if (bytes == 0) {
        jd["_ArrayZipData_"] = "eJwDAAAAAAE=";   /* zlib stream of zero bytes */
        return jd;
    }

    unsigned char* comp = NULL;
    size_t clen = 0, *off = NULL, noff = 0;
    int status = 0;
    TZMatFlags flags;
    flags.iscompress = 0;
    flags.param.clevel = 1;
    flags.param.nthread = (char)jdata_nthread();
    int ret = zmat_run_indexed(bytes, (unsigned char*)data, &clen, &comp, zmZlib, &status,
                               flags.iscompress, &off, &noff);

    if (ret != 0 || comp == NULL) {
        free(comp);
        free(off);
        throw std::runtime_error("zlib compression failed");
    }

    jd["_ArrayZipData_"] = base64_encode(comp, clen);

    if (off && noff >= 6) {
        json offsets = json::array();

        for (size_t i = 0; i + 1 < noff; i += 2) {
            offsets.push_back({off[i], off[i + 1]});
        }

        jd["_ArrayZipOffsets_"] = offsets;
    }

    free(comp);
    free(off);
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
    uint32_t p = ((front & VKMMC_MAX_MESH_LABEL) << 16) | (back & 0xFFFF);
    float f;
    memcpy(&f, &p, 4);
    return f;
}

static void unpack_media(float pm, uint32_t& front, uint32_t& back, bool& flat) {
    uint32_t p;
    memcpy(&p, &pm, 4);
    flat = (p & VKMMC_FACE_FLAT) != 0;
    front = (p >> 16) & VKMMC_MAX_MESH_LABEL;
    back = p & 0xFFFFu;
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
/*
 * sd: nf rows of [v1, v2, v3, label] with 1-based node indices, already
 * range-checked. A triangle listed once is an exterior face (front=0); listed
 * twice it is an interface between the two labels.
 */
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

        if (!(l > 0)) {
            continue;   /* degenerate triangle: cannot be hit */
        }

        N.x /= l;
        N.y /= l;
        N.z /= l;

        out_f.push_back({{e0.v[0], e0.v[1], e0.v[2]}});
        out_d.push_back({N.x, N.y, N.z, pack_media(front, back)});
    }

    printf("MeshSurf: %zu -> %zu unique triangles\n", nf, out_f.size());
}

// ============================================================================
// Extract surface from tetrahedral mesh
// ============================================================================

/**
 * Extract surface triangles from a tetrahedral mesh.
 *
 * Convention: the geometric normal of each output triangle points toward the
 * FRONT medium side. In packed_media, front is stored in bits [30:16] and
 * back in bits [15:0].
 *
 * For a photon traveling from medium A to medium B through this triangle:
 *   - If the ray hits the front face (ff=true), it means the ray is coming
 *     from the front-medium side → crossing INTO back medium.
 *     So new_mid = back_med.
 *   - If the ray hits the back face (ff=false), it means the ray is coming
 *     from the back-medium side → crossing INTO front medium.
 *     So new_mid = front_med.
 *
 * This is exactly what the shader does: new_mid = ff ? bm : fm;
 *
 * For exterior faces (front=0, back=region), the normal points outward from
 * the mesh (toward medium 0/exterior):
 *   - Photon inside mesh heading out → hits back face → new_mid = 0 (exit)
 *   - Photon outside heading in → hits front face → new_mid = region (enter)
 *
 * Faces shared by more than two tetrahedra (a non-manifold mesh) are rejected.
 */
static void extract_surface_from_tet(const std::vector<Vec3>& nodes, const int32_t* elems,
                                     size_t ne, std::vector<std::array<uint32_t, 3> >& out_f,
                                     std::vector<FaceData>& out_d) {
    static const int ftab[4][3] = {{0, 3, 1}, {3, 2, 1}, {0, 2, 3}, {0, 1, 2}};

    struct FInfo {
        uint32_t v[3];      /* vertex indices (winding from first tet) */
        int r1;             /* region of first tet */
        int r2;             /* region of second tet (-1 if exterior) */
        int tet1;           /* index of first tet */
        int count;          /* number of tets sharing this face */
    };

    std::map<FaceKey, FInfo> fmap;

    for (size_t e = 0; e < ne; e++) {
        const int32_t* row = elems + e * 5;
        uint32_t n[4] = {(uint32_t)(row[0] - 1), (uint32_t)(row[1] - 1),
                         (uint32_t)(row[2] - 1), (uint32_t)(row[3] - 1)
                        };
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
                fi.tet1 = (int)e;
                fi.count = 1;
                fmap[k] = fi;
            } else {
                if (++it->second.count > 2) {
                    vkmmc_fail("MeshElem: a triangle is shared by more than two tetrahedra (element " +
                               std::to_string(e + 1) + ")");
                }

                it->second.r2 = reg;
            }
        }
    }

    out_f.clear();
    out_d.clear();

    for (std::map<FaceKey, FInfo>::iterator it = fmap.begin(); it != fmap.end(); ++it) {
        FInfo& fi = it->second;

        /* Skip faces with the same medium on both sides (incl. label-0 elements on the exterior) */
        if ((fi.r2 >= 0 && fi.r1 == fi.r2) || (fi.r2 < 0 && fi.r1 == 0)) {
            continue;
        }

        Vec3 a = nodes[fi.v[0]], b = nodes[fi.v[1]], c = nodes[fi.v[2]];
        Vec3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z};
        Vec3 e2 = {c.x - a.x, c.y - a.y, c.z - a.z};
        Vec3 N = v3cross(e1, e2);
        float l = v3len(N);

        if (!(l > 0)) {
            continue;   /* degenerate triangle: cannot be hit */
        }

        N.x /= l;
        N.y /= l;
        N.z /= l;

        /* Find the 4th vertex of tet1 (the one not on this face) */
        const int32_t* row1 = elems + (size_t)fi.tet1 * 5;
        uint32_t tet1_verts[4] = {(uint32_t)(row1[0] - 1), (uint32_t)(row1[1] - 1),
                                  (uint32_t)(row1[2] - 1), (uint32_t)(row1[3] - 1)
                                 };
        Vec3 opp = {0, 0, 0};  /* opposite vertex */

        for (int vi = 0; vi < 4; vi++) {
            if (tet1_verts[vi] != fi.v[0] &&
                    tet1_verts[vi] != fi.v[1] &&
                    tet1_verts[vi] != fi.v[2]) {
                opp = nodes[tet1_verts[vi]];
                break;
            }
        }

        /* Vector from face centroid to opposite vertex of tet1 */
        float cx = (a.x + b.x + c.x) / 3.0f;
        float cy = (a.y + b.y + c.y) / 3.0f;
        float cz = (a.z + b.z + c.z) / 3.0f;
        float dx = opp.x - cx, dy = opp.y - cy, dz = opp.z - cz;
        float dot = N.x * dx + N.y * dy + N.z * dz;

        /* the normal must point away from tet1 (region r1, the back medium) */
        if (dot > 0) {
            N.x = -N.x;
            N.y = -N.y;
            N.z = -N.z;
            uint32_t tmp = fi.v[1];
            fi.v[1] = fi.v[2];
            fi.v[2] = tmp;
        }

        uint32_t back_med = (uint32_t)fi.r1;
        uint32_t front_med = (fi.r2 >= 0) ? (uint32_t)fi.r2 : 0u;

        out_f.push_back({{fi.v[0], fi.v[1], fi.v[2]}});
        out_d.push_back({N.x, N.y, N.z, pack_media(front_med, back_med)});
    }

    printf("Extracted %zu surface triangles from %zu tetrahedra\n", out_f.size(), ne);
}

// ============================================================================
// Parse helpers
// ============================================================================

static const char* const srctype_names[SRC_TYPE_COUNT] = {
    "pencil", "isotropic", "cone", "gaussian", "planar", "pattern", "fourier", "arcsine", "disk",
    "fourierx", "fourierx2d", "zgaussian", "line", "slit", "pencilarray", "pattern3d", "hyperboloid", "ring"
};

/* source type id; -1 if unknown */
static int parse_srctype(const std::string& name) {
    std::string s(name);

    for (size_t i = 0; i < s.size(); i++) {
        s[i] = (char)tolower((unsigned char)s[i]);
    }

    for (int i = 0; i < SRC_TYPE_COUNT; i++) {
        if (s == srctype_names[i]) {
            return i;
        }
    }

    return -1;
}

static const char* srctype_name(int t) {
    return (t >= 0 && t < SRC_TYPE_COUNT) ? srctype_names[t] : "unknown";
}

/* output fields: one per pattern of a shared pattern source, or one per source of a
   multi-source list simulated at once; 1 otherwise */
static uint32_t src_fields(const SimConfig& cfg) {
    return (!cfg.extrasrc.empty() && cfg.srcid < 0) ? (uint32_t)cfg.extrasrc.size() : (uint32_t)cfg.srcnum;
}

/* output type from a name or its first letter: flux, fluence, energy; -1 if unknown */
static int parse_outputtype(const std::string& name) {
    std::string s(name);

    for (size_t i = 0; i < s.size(); i++) {
        s[i] = (char)tolower((unsigned char)s[i]);
    }

    if (s.empty()) {
        return OT_ENERGY;
    }

    switch (s[0]) {
        case 'f':
            return (s.compare(0, 5, "fluen") == 0) ? OT_FLUENCE : OT_FLUX;

        case 'l':
            return OT_FLUENCE;

        case 'x':
        case 'e':
            return OT_ENERGY;
    }

    return -1;
}

static void update_bbox(SimConfig& cfg) {
    cfg.nmin.x = cfg.nmin.y = cfg.nmin.z = 1e30f;
    cfg.nmax.x = cfg.nmax.y = cfg.nmax.z = -1e30f;

    for (size_t i = 0; i < cfg.nodes.size(); i++) {
        Vec3& v = cfg.nodes[i];
        cfg.nmin.x = std::min(cfg.nmin.x, v.x);
        cfg.nmin.y = std::min(cfg.nmin.y, v.y);
        cfg.nmin.z = std::min(cfg.nmin.z, v.z);
        cfg.nmax.x = std::max(cfg.nmax.x, v.x);
        cfg.nmax.y = std::max(cfg.nmax.y, v.y);
        cfg.nmax.z = std::max(cfg.nmax.z, v.z);
    }
}

// ============================================================================
// Load a numeric table (JData or inline JSON array of rows) as row-major doubles
// ============================================================================
static std::vector<double> load_table(const json& v, size_t mincols, size_t& rows, size_t& cols,
                                      const std::string& ctx) {
    std::vector<double> out;

    if (v.is_object()) {
        JDataArray a = jdata_decode_array(v, ctx);
        vkmmc_require(a.dims.size() == 2, ctx + " must be a 2-D array");
        rows = a.dims[0];
        cols = a.dims[1];
        size_t n = a.numel();
        out.resize(n);

        for (size_t i = 0; i < n; i++) {
            out[i] = a.at(i);
        }
    } else {
        vkmmc_require(v.is_array(), ctx + " must be a JData object or an array of rows");
        rows = v.size();
        cols = rows ? v[0].size() : 0;
        out.reserve(rows * cols);

        for (size_t i = 0; i < rows; i++) {
            vkmmc_require(v[i].is_array() && v[i].size() == cols,
                          ctx + " row " + std::to_string(i + 1) + " must have " + std::to_string(cols) + " numbers");

            for (size_t j = 0; j < cols; j++) {
                vkmmc_require(v[i][j].is_number(), ctx + " must contain only numbers");
                out.push_back(v[i][j].get<double>());
            }
        }
    }

    vkmmc_require(rows > 0, ctx + " is empty");
    vkmmc_require(cols >= mincols, ctx + " needs at least " + std::to_string(mincols) + " columns");

    for (size_t i = 0; i < out.size(); i++) {
        vkmmc_require(std::isfinite(out[i]), ctx + " contains a NaN or Inf");
    }

    return out;
}

// ============================================================================
// Load nodes (JData or inline JSON array)
// ============================================================================
static void load_nodes(const json& mn, SimConfig& cfg) {
    size_t nn = 0, cols = 0;
    std::vector<double> d = load_table(mn, 3, nn, cols, "Shapes.MeshNode");
    vkmmc_require(nn <= 0x7FFFFFFFu, "Shapes.MeshNode: too many nodes");
    cfg.nodes.resize(nn);

    for (size_t i = 0; i < nn; i++) {
        cfg.nodes[i] = {(float)d[i * cols], (float)d[i * cols + 1], (float)d[i * cols + 2]};

        if (!(std::fabs(cfg.nodes[i].x) <= VKMMC_MAX_COORD && std::fabs(cfg.nodes[i].y) <= VKMMC_MAX_COORD &&
                std::fabs(cfg.nodes[i].z) <= VKMMC_MAX_COORD)) {
            vkmmc_fail("Shapes.MeshNode: node " + std::to_string(i + 1) + " is beyond +-1e6 length units");
        }
    }

    printf("MeshNode: %zu nodes\n", nn);
    update_bbox(cfg);
}

/*
 * Load elements (JData or inline) as int32 rows of ncols: node indices are
 * checked against 1..nn and the label (last column) against 0..maxlabel.
 */
static std::vector<int32_t> load_elems(const json& me, size_t nnodes, size_t nnode_cols, size_t maxlabel,
                                       size_t& count, const std::string& ctx) {
    size_t cols = 0;
    std::vector<double> d = load_table(me, nnode_cols + 1, count, cols, ctx);
    std::vector<int32_t> out(count * (nnode_cols + 1));
    size_t nneg = 0;

    for (size_t i = 0; i < count; i++) {
        for (size_t j = 0; j < nnode_cols; j++) {
            double v = d[i * cols + j];

            if (!(v >= 1 && v <= (double)nnodes) || v != std::floor(v)) {
                vkmmc_fail(ctx + " row " + std::to_string(i + 1) + ": node index " + std::to_string(v) +
                           " is not an integer in [1, " + std::to_string(nnodes) + "]");
            }

            out[i * (nnode_cols + 1) + j] = (int32_t)v;
        }

        double lab = d[i * cols + cols - 1];

        /* negative labels (MMC's wide-field source/detector domains, -1/-2) are
           not tissue: they are treated as the background, medium 0 */
        if (!(lab >= -65535 && lab <= (double)maxlabel) || lab != std::floor(lab)) {
            vkmmc_fail(ctx + " row " + std::to_string(i + 1) + ": label " + std::to_string(lab) +
                       " is not an integer in [0, " + std::to_string(maxlabel) + "] (or negative for the background)");
        }

        if (lab < 0) {
            nneg++;
        }

        out[i * (nnode_cols + 1) + nnode_cols] = (int32_t)std::max(lab, 0.0);
    }

    if (nneg) {
        printf("%s: %zu rows with a negative label are treated as the background (medium 0)\n", ctx.c_str(), nneg);
    }

    return out;
}

// ============================================================================
// Main JSON loader
// ============================================================================

/* largest label a medium list of this size can resolve */
static size_t max_label(const SimConfig& cfg) {
    return cfg.media.empty() ? 0 : std::min(cfg.media.size() - 1, (size_t)VKMMC_MAX_MESH_LABEL);
}


// ============================================================================
// Source
// ============================================================================
static void normalize_srcdir(float* d, const std::string& ctx) {
    double l = std::sqrt((double)d[0] * d[0] + (double)d[1] * d[1] + (double)d[2] * d[2]);
    vkmmc_require(l > 1e-10, ctx + " must be a non-zero vector");

    for (int i = 0; i < 3; i++) {
        d[i] = (float)(d[i] / l);
    }
}

/* a number, or for the focal length (Dir[3]) the JData strings "_NaN_", "_Inf_", "-_Inf_" */
static double source_num(const json& v, bool special, const std::string& ctx) {
    if (special && v.is_string()) {
        std::string t = v.get<std::string>();

        if (t == "_NaN_") {
            return NAN;
        } else if (t == "_Inf_") {
            return INFINITY;
        } else if (t == "-_Inf_") {
            return -INFINITY;
        }
    }

    vkmmc_require(v.is_number(), ctx + " must be a number" + (special ? " (or \"_NaN_\"/\"_Inf_\"/\"-_Inf_\")" : ""));
    double d = v.get<double>();
    vkmmc_require(std::isfinite(d), ctx + " must be finite");
    return d;
}

/*
 * A source vector field: one row [a,b,...] or, for a multi-source list, a 2-D
 * array of rows. Each row has nmin..4 numbers; absent entries keep def.
 */
static std::vector<std::array<float, 4> > source_rows(const json& v, size_t nmin, const float* def, bool special,
        double maxabs, const std::string& ctx) {
    vkmmc_require(v.is_array() && !v.empty(), ctx + " must be a non-empty array");
    bool multi = v[0].is_array();
    size_t nrow = multi ? v.size() : 1;
    vkmmc_require(nrow <= VKMMC_MAX_SOURCES, ctx + " has more than " + std::to_string(VKMMC_MAX_SOURCES) + " rows");
    std::vector<std::array<float, 4> > rows(nrow);

    for (size_t r = 0; r < nrow; r++) {
        const json& row = multi ? v[r] : v;
        std::string rc = multi ? ctx + "[" + std::to_string(r) + "]" : ctx;
        vkmmc_require(row.is_array(), rc + " must be an array");
        vkmmc_require(row.size() >= nmin, rc + " needs at least " + std::to_string(nmin) + " numbers");
        vkmmc_require(row.size() <= 4, rc + " has more than 4 numbers");

        for (size_t i = 0; i < 4; i++) {
            double d = (i < row.size()) ? source_num(row[i], special && i == 3, rc + "[" + std::to_string(i) + "]") : def[i];
            vkmmc_require(!(std::fabs(d) > maxabs) || (special && i == 3), rc + " is beyond +-1e6");
            rows[r][i] = (float)d;
        }
    }

    return rows;
}

/*
 * Optode.Source.Pattern: a JData array (row-major [SrcNum,] Nx, Ny[, Nz], read
 * with the first index fastest as MMC does), or {"Nx","Ny"[,"Nz"],"Data"} with
 * Data a flat array or the name of a float32 binary file
 */
static void load_pattern(const json& v, SimConfig& cfg) {
    const std::string ctx = "Optode.Source.Pattern";
    std::vector<double> data;
    vkmmc_require(v.is_object(), ctx + " must be a JData array or an object with Nx, Ny and Data");

    if (v.contains("_ArrayType_")) {
        JDataArray a = jdata_decode_array(v, ctx);
        size_t n = a.numel();
        vkmmc_require(a.dims.size() >= 1 && a.dims.size() <= 4, ctx + " must have 1 to 4 dimensions");
        vkmmc_require(n <= VKMMC_MAX_PATTERN_LEN, ctx + " is too large");

        if (a.dims.size() == 3 && a.dims[0] > 1 && a.dims[2] > 1 && cfg.srctype == SRC_PATTERN) {   /* as MMC */
            cfg.srcnum = (int)std::min<size_t>(a.dims[0], VKMMC_MAX_PATTERNS + 1);
        }

        /* row-major -> first index fastest */
        size_t nd = a.dims.size();
        std::vector<size_t> idx(nd, 0);
        data.resize(n);

        for (size_t i = 0; i < n; i++) {
            size_t o = 0, st = 1;

            for (size_t d = 0; d < nd; d++) {
                o += idx[d] * st;
                st *= a.dims[d];
            }

            data[o] = a.at(i);

            for (size_t d = nd; d-- > 0;) {
                if (++idx[d] < a.dims[d]) {
                    break;
                }

                idx[d] = 0;
            }
        }
    } else {
        double nx = json_num_range(v, "Nx", 0, 1, 1e7, ctx), ny = json_num_range(v, "Ny", 0, 1, 1e7, ctx);
        double nz = json_num_range(v, "Nz", 1, 1, 1e7, ctx);
        double n = nx * ny * nz * cfg.srcnum;
        vkmmc_require(n <= VKMMC_MAX_PATTERN_LEN, ctx + " is too large");
        vkmmc_require(v.contains("Data"), ctx + " requires 'Data'");
        const json& d = v["Data"];

        if (d.is_string()) {
            std::ifstream f(d.get<std::string>().c_str(), std::ios::binary | std::ios::ate);
            vkmmc_require((bool)f, ctx + ".Data: cannot open " + d.get<std::string>());
            vkmmc_require((double)f.tellg() >= n * sizeof(float), ctx + ".Data: the file is too short");
            f.seekg(0);
            std::vector<float> buf((size_t)n);
            f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)(buf.size() * sizeof(float)));
            vkmmc_require(f.gcount() == (std::streamsize)(buf.size() * sizeof(float)), ctx + ".Data: the file is too short");
            data.assign(buf.begin(), buf.end());
        } else {
            vkmmc_require(d.is_array() && d.size() == (size_t)n, ctx + ".Data must hold Nx*Ny*Nz*SrcNum numbers");
            data.resize((size_t)n);

            for (size_t i = 0; i < data.size(); i++) {
                vkmmc_require(d[i].is_number(), ctx + ".Data must contain only numbers");
                data[i] = d[i].get<double>();
            }
        }
    }

    vkmmc_require(data.size() <= VKMMC_MAX_PATTERN_LEN, ctx + " is too large");
    cfg.srcpattern.resize(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        vkmmc_require(data[i] >= 0 && data[i] <= VKMMC_MAX_SRC_WEIGHT,
                      ctx + " values must be in [0, " + std::to_string((int)VKMMC_MAX_SRC_WEIGHT) + "]");
        cfg.srcpattern[i] = (float)data[i];
    }
}

/* parameter checks that keep the kernels' source sampling well defined */
static void check_source(const SimConfig& cfg) {
    const float* p1 = cfg.srcparam1, *p2 = cfg.srcparam2;
    const std::string ctx = std::string("Optode.Source (") + srctype_name(cfg.srctype) + ")";

    switch (cfg.srctype) {
        case SRC_CONE:
            vkmmc_require(p1[0] > 0, ctx + ": Param1[0] (the cone half angle) must be positive");
            break;

        case SRC_DISK:
        case SRC_RING:
            vkmmc_require(p1[0] >= 0 && p1[1] >= 0, ctx + ": Param1[0..1] (radii) must be non-negative");
            break;

        case SRC_GAUSSIAN:
        case SRC_ZGAUSSIAN:
            vkmmc_require(p1[0] >= 0, ctx + ": Param1[0] (the width) must be non-negative");
            break;

        case SRC_HYPERBOLOID:
            vkmmc_require(p1[0] >= 0 && p1[2] >= 0, ctx + ": Param1[0] (waist) and Param1[2] (Rayleigh range) must be non-negative");
            break;

        case SRC_PENCILARRAY:
            vkmmc_require(p1[3] >= 1 && p2[3] >= 1 && p1[3] <= 1e6 && p2[3] <= 1e6 && p1[3] == std::floor(p1[3]) && p2[3] == std::floor(p2[3]),
                          ctx + ": Param1[3] and Param2[3] (the beam counts) must be positive integers");
            break;

        case SRC_LINE:
        case SRC_SLIT:
            vkmmc_require(p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2] > 0, ctx + ": Param1[0..2] (the segment) must be a non-zero vector");
            vkmmc_require(p2[0] >= 0 && p2[1] >= 0, ctx + ": Param2[0..1] must be non-negative");
            break;

        case SRC_FOURIERX:
        case SRC_FOURIERX2D:
            vkmmc_require(p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2] > 0, ctx + ": Param1[0..2] must be a non-zero vector");
            vkmmc_require(cfg.srctype == SRC_FOURIERX2D || (p2[3] >= 0 && p2[3] <= 2),
                          ctx + ": Param2[3] (the modulation depth M) must be in [0, 2]");
            break;

        case SRC_PATTERN:
        case SRC_PATTERN3D: {
            double nx = (cfg.srctype == SRC_PATTERN) ? p1[3] : p1[0], ny = (cfg.srctype == SRC_PATTERN) ? p2[3] : p1[1];
            double nz = (cfg.srctype == SRC_PATTERN) ? 1 : p1[2];
            vkmmc_require(nx >= 1 && ny >= 1 && nz >= 1 && nx == std::floor(nx) && ny == std::floor(ny) && nz == std::floor(nz),
                          ctx + ((cfg.srctype == SRC_PATTERN) ? ": Param1[3] and Param2[3] (the pattern size)" : ": Param1[0..2] (the pattern size)")
                          + " must be positive integers");
            vkmmc_require(!cfg.srcpattern.empty(), ctx + " requires Optode.Source.Pattern");
            vkmmc_require((double)cfg.srcpattern.size() == nx * ny * nz * cfg.srcnum,
                          ctx + ": Optode.Source.Pattern must hold " + std::to_string((uint64_t)(nx * ny * nz)) + " x SrcNum values");
            break;
        }

        default:
            break;
    }

    vkmmc_require(cfg.srcnum == 1 || cfg.srctype == SRC_PATTERN || cfg.srctype == SRC_PATTERN3D,
                  "Optode.Source.SrcNum > 1 requires a pattern or pattern3d source");
    vkmmc_require(cfg.srcnum == 1 || cfg.extrasrc.empty(),
                  "Optode.Source.SrcNum > 1 (shared patterns) and multiple sources cannot be combined");
}

static void load_source(const json& src, SimConfig& cfg) {
    vkmmc_require(src.is_object(), "Optode.Source must be an object");
    std::string tname = json_str(src, "Type", "pencil", "Optode.Source");
    cfg.srctype = parse_srctype(tname);
    vkmmc_require(cfg.srctype >= 0, "Optode.Source.Type '" + tname + "' is not a known source type");

    static const float zero[4] = {0, 0, 0, 0}, one_w[4] = {0, 0, 0, 1}, zdir[4] = {0, 0, 1, 0};
    std::vector<std::array<float, 4> > pos(1), dir(1), p1(1), p2(1);
    memcpy(pos[0].data(), one_w, sizeof(one_w));
    memcpy(dir[0].data(), cfg.srcdir, sizeof(zdir));
    memcpy(p1[0].data(), cfg.srcparam1, sizeof(zero));
    memcpy(p2[0].data(), cfg.srcparam2, sizeof(zero));

    if (src.contains("Pos")) {
        pos = source_rows(src["Pos"], 3, one_w, false, VKMMC_MAX_COORD, "Optode.Source.Pos");
    }

    if (src.contains("Dir")) {
        dir = source_rows(src["Dir"], 3, zdir, true, 1e30, "Optode.Source.Dir");
    }

    if (src.contains("Param1")) {
        p1 = source_rows(src["Param1"], 1, zero, false, VKMMC_MAX_COORD, "Optode.Source.Param1");
    }

    if (src.contains("Param2")) {
        p2 = source_rows(src["Param2"], 1, zero, false, VKMMC_MAX_COORD, "Optode.Source.Param2");
    }

    /* multi-source: any field with several rows; a single row applies to every source */
    size_t nsrc = std::max(std::max(pos.size(), dir.size()), std::max(p1.size(), p2.size()));
    bool multi = src.contains("Pos") && src["Pos"].is_array() && !src["Pos"].empty() && src["Pos"][0].is_array();
    std::vector<std::array<float, 4> >* fields[4] = {&pos, &dir, &p1, &p2};

    for (int f = 0; f < 4; f++) {
        vkmmc_require(fields[f]->size() == 1 || fields[f]->size() == nsrc,
                      "Optode.Source: Pos, Dir, Param1 and Param2 must have the same number of rows");
        fields[f]->resize(nsrc, (*fields[f])[0]);
    }

    for (size_t r = 0; r < nsrc; r++) {
        std::string rc = "Optode.Source.Dir" + (nsrc > 1 ? "[" + std::to_string(r) + "]" : std::string());
        normalize_srcdir(dir[r].data(), rc);
    }

    for (int i = 0; i < 4; i++) {
        if (i < 3) {
            cfg.srcpos[i] = pos[0][i];
        }

        cfg.srcdir[i] = dir[0][i];
        cfg.srcparam1[i] = p1[0][i];
        cfg.srcparam2[i] = p2[0][i];
    }

    cfg.extrasrc.clear();

    if (multi || nsrc > 1) {
        cfg.extrasrc.resize(nsrc);

        for (size_t r = 0; r < nsrc; r++) {
            ExtraSrc& e = cfg.extrasrc[r];
            vkmmc_require(pos[r][3] >= 0 && pos[r][3] <= VKMMC_MAX_SRC_WEIGHT,
                          "Optode.Source.Pos[" + std::to_string(r) + "][3] (the source weight) must be in [0, "
                          + std::to_string((int)VKMMC_MAX_SRC_WEIGHT) + "]");
            vkmmc_require(p1[r][0] >= 0, "Optode.Source.Param1[" + std::to_string(r) + "][0] (the disk radius) must be non-negative");
            memcpy(e.pos, pos[r].data(), sizeof(e.pos));
            memcpy(e.dir, dir[r].data(), sizeof(e.dir));
            memcpy(e.param1, p1[r].data(), sizeof(e.param1));
            memcpy(e.param2, p2[r].data(), sizeof(e.param2));
        }
    }

    /* the source of a multi-source run: 0 or -1 a field per source, k > 0 source k only */
    double id = json_num(src, "ID", 0, "Optode.Source");
    vkmmc_require(id == std::floor(id) && id >= -1 && id <= VKMMC_MAX_SOURCES, "Optode.Source.ID must be -1, 0 or a source number");
    vkmmc_require(id <= (double)cfg.extrasrc.size() || cfg.extrasrc.empty(), "Optode.Source.ID is larger than the number of sources");
    cfg.srcid = cfg.extrasrc.empty() ? 0 : ((id <= 0) ? -1 : (int)id);

    cfg.srcnum = (int)json_num_range(src, "SrcNum", 1, 1, VKMMC_MAX_PATTERNS, "Optode.Source");
    cfg.srcpattern.clear();

    if (src.contains("Pattern") && (cfg.srctype == SRC_PATTERN || cfg.srctype == SRC_PATTERN3D)) {
        load_pattern(src["Pattern"], cfg);
        vkmmc_require(cfg.srcnum <= VKMMC_MAX_PATTERNS, "Optode.Source.Pattern has more than "
                      + std::to_string(VKMMC_MAX_PATTERNS) + " patterns");
    }

    check_source(cfg);
}

static SimConfig load_json_config(const json& j) {
    SimConfig cfg;

    vkmmc_require(j.is_object(), "input must be a JSON object");

    // Session
    if (j.contains("Session")) {
        const json& s = j["Session"];
        vkmmc_require(s.is_object(), "Session must be an object");
        cfg.session_id = json_str(s, "ID", "default", "Session");
        vkmmc_require(!cfg.session_id.empty() && cfg.session_id.size() < 256, "Session.ID must have 1-255 characters");
        cfg.nphoton = (uint64_t)json_num_range(s, "Photons", 1e6, 1, 9.2e18, "Session");
        cfg.rng_seed = (uint32_t)(int64_t)json_num_range(s, "RNGSeed", 29012391, -2147483648.0, 4294967295.0, "Session");
        cfg.do_mismatch = json_bool(s, "DoMismatch", true, "Session");
        cfg.do_normalize = json_bool(s, "DoNormalize", true, "Session");
        cfg.output_type = parse_outputtype(json_str(s, "OutputType", "x", "Session"));
        vkmmc_require(cfg.output_type >= 0, "Session.OutputType must be flux, fluence or energy (f/l/x)");
        cfg.minenergy = (float)json_num_range(s, "RussianRoulette", 1e-6, 0, 1, "Session");
        cfg.roulettesize = (float)json_num_range(s, "RussianRouletteSize", 10, 1, 1e6, "Session");
        cfg.debuglevel = (uint32_t)json_num_range(s, "DebugFlag", 0, 0, 4294967295.0, "Session");
        cfg.maxjumpdebug = (uint32_t)json_num_range(s, "MaxJumpDebug", 1e7, 0, 1e9, "Session");
    }

    // Forward
    if (j.contains("Forward")) {
        const json& fw = j["Forward"];
        vkmmc_require(fw.is_object(), "Forward must be an object");
        double d0 = json_num(fw, "T0", 0.0, "Forward"), d1 = json_num(fw, "T1", 5e-9, "Forward");
        double dd = json_num(fw, "Dt", d1 - d0, "Forward");
        vkmmc_require(d0 >= 0 && d1 > d0, "Forward: need 0 <= T0 < T1");
        vkmmc_require(d1 <= VKMMC_MAX_TIME, "Forward.T1 must not exceed 1 s");
        vkmmc_require(dd > 0, "Forward.Dt must be positive");
        dd = std::min(dd, d1 - d0);     /* a gate never extends past T1 */
        vkmmc_require(dd >= 1e-15, "Forward: the time gate width (Dt, or T1-T0) must be at least 1e-15 s");
        double ng = std::floor((d1 - d0) / dd + 0.5);
        vkmmc_require(ng <= VKMMC_MAX_GATES, "Forward: (T1-T0)/Dt gives more than " + std::to_string(VKMMC_MAX_GATES) + " time gates");
        cfg.t0 = (float)d0;
        cfg.t1 = (float)d1;
        cfg.dt = (float)dd;
        cfg.maxgate = std::max(1, (int)ng);
        printf("Forward: T0=%.4e T1=%.4e Dt=%.4e maxgate=%d\n", d0, d1, dd, cfg.maxgate);
    }

    // Domain
    vkmmc_require(j.contains("Domain") && j["Domain"].is_object(), "input requires a Domain object");
    const json& d = j["Domain"];
    cfg.unitinmm = (float)json_num_range(d, "LengthUnit", 1.0, 1e-6, 1e6, "Domain");

    vkmmc_require(d.contains("Media"), "Domain requires Media");
    {
        json ma = d["Media"];

        /* unwrap [[{...}, ...]]; a list of 4-number rows [[mua,mus,g,n], ...] is kept */
        if (ma.is_array() && ma.size() == 1 && ma[0].is_array() && !ma[0].empty() && !ma[0][0].is_number()) {
            ma = ma[0];
        }

        vkmmc_require(ma.is_array() && ma.size() >= 1, "Domain.Media must be a non-empty array");
        vkmmc_require(ma.size() <= VKMMC_MAX_MEDIA, "Domain.Media: at most " + std::to_string(VKMMC_MAX_MEDIA) + " media are supported");

        for (size_t i = 0; i < ma.size(); i++) {
            std::string ctx = "Domain.Media[" + std::to_string(i) + "]";
            const json& m = ma[i];
            Medium med;

            if (m.is_array()) {
                std::vector<double> p = json_vec(m, 4, 4, ctx);
                med = {(float)p[0], (float)p[1], (float)p[2], (float)p[3]};
            } else {
                vkmmc_require(m.is_object(), ctx + " must be an object {mua,mus,g,n}");
                med = {(float)json_num(m, "mua", 0, ctx), (float)json_num(m, "mus", 0, ctx),
                       (float)json_num(m, "g", 1, ctx), (float)json_num(m, "n", 1, ctx)
                      };
            }

            vkmmc_require(med.mua >= 0 && med.mua < 1e6f, ctx + ".mua must be in [0, 1e6)");
            vkmmc_require(med.mus >= 0 && med.mus < 1e6f, ctx + ".mus must be in [0, 1e6)");
            vkmmc_require(med.g >= -1 && med.g <= 1, ctx + ".g must be in [-1, 1]");
            vkmmc_require(med.n >= 0.01f && med.n <= 100, ctx + ".n must be in [0.01, 100]");
            cfg.media.push_back(med);
            printf("  Media[%zu]: mua=%.6f mus=%.6f g=%.4f n=%.4f\n", i, med.mua, med.mus, med.g, med.n);
        }
    }

    if (d.contains("Step")) {
        std::vector<double> st = json_vec(d["Step"], 1, 3, "Domain.Step");

        for (size_t i = 0; i < 3; i++) {
            cfg.steps[i] = (float)st[std::min(i, st.size() - 1)];
            vkmmc_require(cfg.steps[i] > 1e-6f && cfg.steps[i] < 1e6f, "Domain.Step must be in (1e-6, 1e6)");
        }

        cfg.has_steps = true;
    }

    if (d.contains("MeshRes")) {
        cfg.mesh_res = (int)json_num_range(d, "MeshRes", 24, 1, VKMMC_MAX_MESHRES, "Domain");
        printf("MeshRes: %d\n", cfg.mesh_res);
    }

    // Optode / Source
    if (j.contains("Optode") && j["Optode"].is_object() && j["Optode"].contains("Source")) {
        load_source(j["Optode"]["Source"], cfg);
    } else {
        normalize_srcdir(cfg.srcdir, "Optode.Source.Dir");
    }

    // Shapes
    vkmmc_require(j.contains("Shapes"), "input requires 'Shapes' (a shape array or a mesh object)");
    const json& shapes = j["Shapes"];

    if (shapes.is_array()) {
        // ---- CSG shape mode: meshes are generated by build_shape_mesh() ----
        vkmmc_require(shapes.size() > 0, "Shapes is an empty array");
        cfg.is_csg = true;
        cfg.mediumid0 = 0xFFFFFFFFu;
        cfg.shapes = shapes;
        printf("Shapes: CSG mode detected (array of shape constructs)\n");
    } else {
        // ---- Mesh mode ----
        vkmmc_require(shapes.is_object(), "Shapes must be an array of shapes or a mesh object");
        cfg.is_csg = false;

        vkmmc_require(shapes.contains("MeshNode"), "Mesh mode requires Shapes.MeshNode");
        load_nodes(shapes["MeshNode"], cfg);

        std::vector<int32_t> elem;
        size_t elem_count = 0;

        if (shapes.contains("MeshSurf")) {
            size_t nf = 0;
            std::vector<int32_t> sd = load_elems(shapes["MeshSurf"], cfg.nodes.size(), 3, max_label(cfg), nf,
                                                 "Shapes.MeshSurf");
            load_mesh_surf(cfg.nodes, sd.data(), nf, cfg.faces, cfg.facedata);
        } else if (shapes.contains("MeshElem")) {
            vkmmc_require(!shapes.contains("MeshROI"), "Shapes.MeshROI (implicit MMC edges, nodes and faces) is not supported");
            elem = load_elems(shapes["MeshElem"], cfg.nodes.size(), 4, max_label(cfg), elem_count,
                              "Shapes.MeshElem");
            printf("MeshElem: %zu tetrahedra\n", elem_count);
            extract_surface_from_tet(cfg.nodes, elem.data(), elem_count, cfg.faces, cfg.facedata);
        } else {
            vkmmc_fail("Mesh mode requires Shapes.MeshSurf or Shapes.MeshElem");
        }

        vkmmc_require(!cfg.faces.empty(), "the mesh has no surface triangles between different media");

        if (shapes.contains("InitElem")) {
            double ie = json_num(shapes, "InitElem", -1, "Shapes");
            vkmmc_require(ie == std::floor(ie) && ie >= -1 && ie <= 2147483647.0, "Shapes.InitElem must be an element index");
            cfg.init_elem = (int)ie;

            if (cfg.init_elem > 0) {
                vkmmc_require(!elem.empty(), "Shapes.InitElem requires Shapes.MeshElem");
                vkmmc_require(cfg.init_elem <= (int)elem_count, "Shapes.InitElem is larger than the element count");
                cfg.mediumid0 = (uint32_t)elem[(size_t)(cfg.init_elem - 1) * 5 + 4];
                printf("InitElem: %d, medium type: %u\n", cfg.init_elem, cfg.mediumid0);

                /* a background element (e.g. MMC's wide-field source domain): the
                   photon is launched outside the tissue and enters the first
                   surface it points at, as with no InitElem */
                if (cfg.mediumid0 == 0) {
                    cfg.mediumid0 = 0xFFFFFFFFu;
                }
            }
        }
    }

    return cfg;
}

/* parse a JSON file; a JSON string passed with -j/--json is merged on top */
static json load_json_file(const char* filepath, const std::string& patch = std::string()) {
    std::ifstream f(filepath, std::ios::binary);

    if (!f) {
        throw std::runtime_error(std::string("cannot open input file: ") + filepath);
    }

    json j;

    try {
        j = json::parse(f);

        if (!patch.empty()) {
            j.merge_patch(json::parse(patch));
        }
    } catch (const json::parse_error& e) {
        throw std::runtime_error(std::string("JSON syntax error in ") + (patch.empty() ? filepath : "input or -j") +
                                 ": " + e.what());
    }

    return j;
}

static inline SimConfig load_json_input(const char* filepath) {
    return load_json_config(load_json_file(filepath));
}

#endif // VKMMC_IO_H
