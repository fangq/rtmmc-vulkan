/*
 * test_vkmmc_unit.cpp — host-side (no GPU) unit tests for vkmmc
 *
 * Covers the header-only host logic: vkmmc_io.h (JSON validation, base64,
 * JData, mesh surface extraction), vkmmc_shapes.h (shape triangulation,
 * coplanar flagging), vkmmc_curvature.h (per-node curvature) and
 * vkmmc_host.h (command line, geometry preparation, output normalization and
 * file output).
 *
 * Build and run: make -C tests/unit run   (see the Makefile for asan/coverage)
 * Pass -v to see the diagnostic output printed by the headers.
 *
 * Tests whose comment starts with "Regression (fixed):" guard bugs found
 * while writing these tests.
 */

#include "vkmmc_io.h"
#include "vkmmc_shapes.h"
#include "vkmmc_curvature.h"
#include "vkmmc_host.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <set>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <fcntl.h>
#include <ftw.h>
#include <sys/stat.h>
#include <unistd.h>

// ============================================================================
// Minimal test framework
// ============================================================================
static FILE* g_out = stderr;      /* report stream (stdout is silenced by default) */
static int g_checks = 0, g_failed_checks = 0, g_test_failed = 0;
static std::string g_test;

#define REPORT_FAIL(...) do { \
        g_failed_checks++; g_test_failed++; \
        fprintf(g_out, "  FAIL [%s] %s:%d: ", g_test.c_str(), __FILE__, __LINE__); \
        fprintf(g_out, __VA_ARGS__); fprintf(g_out, "\n"); \
    } while (0)

#define CHECK(cond) do { \
        g_checks++; \
        bool ok_ = false; std::string err_; \
        try { ok_ = (cond); } catch (const std::exception& e_) { err_ = e_.what(); } \
        if (!ok_) { \
            if (err_.empty()) REPORT_FAIL("%s", #cond); \
            else REPORT_FAIL("%s threw: %s", #cond, err_.c_str()); \
        } \
    } while (0)

#define CHECK_CLOSE(a, b, tol) do { \
        g_checks++; \
        double va_ = (double)(a), vb_ = (double)(b), vt_ = (double)(tol); \
        if (!(std::fabs(va_ - vb_) <= vt_)) \
            REPORT_FAIL("%s = %.9g, expected %s = %.9g (tol %.3g)", #a, va_, #b, vb_, vt_); \
    } while (0)

#define CHECK_THROWS(expr, substr) do { \
        g_checks++; \
        bool thrown_ = false; std::string msg_; \
        try { (void)(expr); } catch (const std::exception& e_) { thrown_ = true; msg_ = e_.what(); } \
        if (!thrown_) REPORT_FAIL("%s did not throw (expected \"%s\")", #expr, substr); \
        else if (msg_.find(substr) == std::string::npos) \
            REPORT_FAIL("%s threw \"%s\", expected a message containing \"%s\"", #expr, msg_.c_str(), substr); \
    } while (0)

#define CHECK_NOTHROW(expr) do { \
        g_checks++; \
        try { (void)(expr); } catch (const std::exception& e_) { REPORT_FAIL("%s threw: %s", #expr, e_.what()); } \
    } while (0)

typedef void (*TestFn)();
struct TestEntry {
    const char* name;
    TestFn fn;
};
static std::vector<TestEntry>& test_registry() {
    static std::vector<TestEntry> r;
    return r;
}
struct TestRegistrar {
    TestRegistrar(const char* n, TestFn f) {
        test_registry().push_back({n, f});
    }
};
#define TEST(name) \
    static void name(); \
    static TestRegistrar reg_##name(#name, name); \
    static void name()

// ============================================================================
// Helpers
// ============================================================================
static json J(const char* s) {
    return json::parse(s);
}

/* base with an RFC 7386 merge patch on top */
static json patched(json base, const char* patch) {
    base.merge_patch(json::parse(patch));
    return base;
}

static json base_csg() {
    return J(R"({
        "Session": {"ID": "t", "Photons": 1000},
        "Domain": {"Media": [{"mua":0,"mus":0,"g":1,"n":1}, {"mua":0.01,"mus":1,"g":0.9,"n":1.37},
                             {"mua":0.02,"mus":2,"g":0.8,"n":1.5}]},
        "Optode": {"Source": {"Type": "pencil", "Pos": [30, 30, 0], "Dir": [0, 0, 1]}},
        "Shapes": [{"Grid": {"Size": [60, 60, 60], "Tag": 1}}]
    })");
}

/* unit cube split into 6 tetrahedra (Kuhn); node id = x + 2y + 4z (0-based) */
static std::vector<Vec3> cube_nodes() {
    std::vector<Vec3> n;

    for (int i = 0; i < 8; i++) {
        n.push_back({(float)(i & 1), (float)((i >> 1) & 1), (float)((i >> 2) & 1)});
    }

    return n;
}

/* rows [n1,n2,n3,n4,label] (1-based); label 1 if x comes before y in the
   axis permutation (region x>y), else label2 */
static std::vector<int32_t> cube_elems(int label2) {
    static const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    std::vector<int32_t> e;

    for (int p = 0; p < 6; p++) {
        int v1 = 1 << perms[p][0], v2 = v1 | (1 << perms[p][1]);
        int xpos = 0, ypos = 0;

        for (int k = 0; k < 3; k++) {
            if (perms[p][k] == 0) {
                xpos = k;
            }

            if (perms[p][k] == 1) {
                ypos = k;
            }
        }

        int lab = (xpos < ypos) ? 1 : label2;
        int row[5] = {1, v1 + 1, v2 + 1, 8, lab};
        e.insert(e.end(), row, row + 5);
    }

    return e;
}

static Vec3 sub(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
static double dot(Vec3 a, Vec3 b) {
    return (double)a.x * b.x + (double)a.y * b.y + (double)a.z * b.z;
}
static Vec3 centroid(const std::vector<Vec3>& n, const std::array<uint32_t, 3>& f) {
    Vec3 a = n[f[0]], b = n[f[1]], c = n[f[2]];
    return {(a.x + b.x + c.x) / 3, (a.y + b.y + c.y) / 3, (a.z + b.z + c.z) / 3};
}
static Vec3 normalized(Vec3 v) {
    float l = v3len(v);
    return {v.x / l, v.y / l, v.z / l};
}
static uint32_t pm_bits(float pm) {
    uint32_t p;
    memcpy(&p, &pm, 4);
    return p;
}

struct MeshCheck {
    bool watertight;      /* each undirected edge used by exactly 2 triangles */
    bool oriented;        /* each directed edge used exactly once */
    size_t bad_normals;   /* stored normal not unit or not matching the winding */
    size_t degenerate;    /* zero-area triangles */
    double volume;        /* enclosed signed volume (divergence theorem) */
};

static MeshCheck check_mesh(const std::vector<Vec3>& nodes, const std::vector<std::array<uint32_t, 3> >& faces,
                            const std::vector<FaceData>& fd, size_t f0 = 0, size_t f1 = (size_t) -1) {
    MeshCheck r = {true, true, 0, 0, 0.0};
    f1 = std::min(f1, faces.size());
    std::map<std::pair<uint32_t, uint32_t>, int> und, dir;

    for (size_t i = f0; i < f1; i++) {
        const std::array<uint32_t, 3>& f = faces[i];

        for (int k = 0; k < 3; k++) {
            uint32_t a = f[k], b = f[(k + 1) % 3];
            und[std::make_pair(std::min(a, b), std::max(a, b))]++;
            dir[std::make_pair(a, b)]++;
        }

        Vec3 A = nodes[f[0]], B = nodes[f[1]], C = nodes[f[2]];
        double e1[3] = {(double)B.x - A.x, (double)B.y - A.y, (double)B.z - A.z};
        double e2[3] = {(double)C.x - A.x, (double)C.y - A.y, (double)C.z - A.z};
        double cr[3] = {e1[1]* e2[2] - e1[2] * e2[1], e1[2]* e2[0] - e1[0] * e2[2], e1[0]* e2[1] - e1[1] * e2[0]};
        double cl = std::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
        r.volume += ((double)A.x * (B.y * (double)C.z - B.z * (double)C.y) +
                     (double)A.y * (B.z * (double)C.x - B.x * (double)C.z) +
                     (double)A.z * (B.x * (double)C.y - B.y * (double)C.x)) / 6.0;

        if (cl < 1e-9) {
            r.degenerate++;
            continue;
        }

        double nl = std::sqrt((double)fd[i].nx * fd[i].nx + (double)fd[i].ny * fd[i].ny + (double)fd[i].nz * fd[i].nz);
        double c = (cr[0] * fd[i].nx + cr[1] * fd[i].ny + cr[2] * fd[i].nz) / cl;

        if (std::fabs(nl - 1) > 1e-4 || c < 0.999) {
            r.bad_normals++;
        }
    }

    for (std::map<std::pair<uint32_t, uint32_t>, int>::iterator it = und.begin(); it != und.end(); ++it) {
        if (it->second != 2) {
            r.watertight = false;
        }
    }

    for (std::map<std::pair<uint32_t, uint32_t>, int>::iterator it = dir.begin(); it != dir.end(); ++it) {
        if (it->second != 1) {
            r.oriented = false;
        }
    }

    return r;
}

static size_t count_flat(const ShapeMesh& m, bool flat) {
    size_t n = 0;

    for (size_t i = 0; i < m.facedata.size(); i++) {
        n += (((pm_bits(m.facedata[i].packed_media) & VKMMC_FACE_FLAT) != 0) == flat);
    }

    return n;
}

static bool all_tags(const ShapeMesh& m, uint32_t tag, size_t f0 = 0, size_t f1 = (size_t) -1) {
    f1 = std::min(f1, m.facedata.size());

    for (size_t i = f0; i < f1; i++) {
        uint32_t fr, bk;
        bool fl;
        unpack_media(m.facedata[i].packed_media, fr, bk, fl);

        if (fr != 0 || bk != tag) {
            return false;
        }
    }

    return true;
}

static ShapeMesh shapes(const char* js, int res = 24, uint32_t maxtag = 64, std::vector<ShapeOrigin>* origins = NULL) {
    float ext[6] = {0, 60, 0, 60, 0, 60};
    return parse_shapes(J(js), ext, res, maxtag, origins);
}

/* analytic volume of a Lens (same rim radius rule as gen_lens), by quadrature */
static double lens_volume(double R, double Df, double Rf, double Db, double Rb) {
    double af = Df - Rf, Rcf = std::fabs(Rf), sf = Rf >= 0 ? 1 : -1;
    double ab = Rb - Db, Rcb = std::fabs(Rb), sb = Rb >= 0 ? -1 : 1;
    double re = std::min(R, std::min(Rcf, Rcb));

    if (Rf > 0 && Rb > 0 && std::fabs(ab - af) > 1e-10) {
        double zi = (Rcf * Rcf - Rcb * Rcb - af * af + ab * ab) / (2 * (ab - af));
        double r2 = Rcf * Rcf - (zi - af) * (zi - af);

        if (r2 > 0 && std::sqrt(r2) < R) {
            re = std::min(re, std::sqrt(r2));
        }
    }

    const int N = 200000;
    double v = 0, dr = re / N;

    for (int i = 0; i < N; i++) {
        double r = (i + 0.5) * dr;
        double zf = af + sf * std::sqrt(std::max(0.0, Rcf * Rcf - r * r));
        double zb = ab + sb * std::sqrt(std::max(0.0, Rcb * Rcb - r * r));
        v += 2 * M_PI * r * (zf - zb) * dr;
    }

    return v;
}

/* temporary directory for file-writing tests, removed at exit */
static std::string g_tmpdir;

static int rm_entry(const char* path, const struct stat*, int, struct FTW*) {
    return remove(path);
}

static const std::string& tmpdir() {
    if (g_tmpdir.empty()) {
        const char* base = getenv("TMPDIR");
        std::string tpl = std::string(base && *base ? base : "/tmp") + "/vkmmc_unit.XXXXXX";
        std::vector<char> buf(tpl.begin(), tpl.end());
        buf.push_back('\0');

        if (!mkdtemp(buf.data())) {
            fprintf(stderr, "cannot create a temporary directory\n");
            exit(2);
        }

        g_tmpdir = buf.data();
    }

    return g_tmpdir;
}

static std::string tmpfile_path(const std::string& name) {
    return tmpdir() + "/" + name;
}

static void write_text(const std::string& path, const std::string& s) {
    FILE* f = fopen(path.c_str(), "wb");

    if (f) {
        fwrite(s.data(), 1, s.size(), f);
        fclose(f);
    }
}

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

/* run fn with stdout redirected into a string */
template<class F> static std::string capture_stdout(F fn) {
    std::string path = tmpfile_path("stdout_capture.txt");
    fflush(stdout);
    int saved = dup(1);
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    dup2(fd, 1);
    close(fd);

    try {
        fn();
    } catch (...) {
        fflush(stdout);
        dup2(saved, 1);
        close(saved);
        throw;
    }

    fflush(stdout);
    dup2(saved, 1);
    close(saved);
    std::ifstream in(path.c_str(), std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static bool run_cmdline(std::vector<std::string> args, CmdOverrides& o, int defgpu = 0) {
    std::vector<char*> argv;

    for (size_t i = 0; i < args.size(); i++) {
        argv.push_back(&args[i][0]);
    }

    argv.push_back(NULL);
    return parse_cmdline((int)args.size(), argv.data(), o, "test", defgpu, ".spv");
}

static int run_frontend(std::vector<std::string> args, CmdOverrides& o, SimConfig& cfg, SimSetup& st) {
    std::vector<char*> argv;

    for (size_t i = 0; i < args.size(); i++) {
        argv.push_back(&args[i][0]);
    }

    argv.push_back(NULL);
    return vkmmc_frontend((int)args.size(), argv.data(), "test", 0, ".spv", o, cfg, st);
}

// ============================================================================
// Base64
// ============================================================================
TEST(base64_round_trip_random_lengths) {
    std::mt19937 rng(1234);
    bool all_ok = true;

    for (size_t len = 0; len <= 300; len++) {
        std::vector<uint8_t> d(len);

        for (size_t i = 0; i < len; i++) {
            d[i] = (uint8_t)(rng() & 0xFF);
        }

        std::string enc = base64_encode(d.data(), len);
        all_ok &= (enc.size() == (len + 2) / 3 * 4);
        all_ok &= (base64_decode(enc) == d);

        /* padding is optional */
        std::string nopad = enc;

        while (!nopad.empty() && nopad.back() == '=') {
            nopad.pop_back();
        }

        all_ok &= (base64_decode(nopad) == d);
    }

    CHECK(all_ok);
    CHECK(base64_encode((const uint8_t*)"Man", 3) == "TWFu");
    CHECK(base64_encode((const uint8_t*)"Ma", 2) == "TWE=");
    CHECK(base64_encode((const uint8_t*)"M", 1) == "TQ==");
    CHECK(base64_encode(NULL, 0).empty());
    CHECK(base64_decode("").empty());
}

TEST(base64_whitespace_and_errors) {
    std::vector<uint8_t> man = {'M', 'a', 'n', 'M', 'a'};
    CHECK(base64_decode("TWFu\nTWE=") == man);
    CHECK(base64_decode(" T W F u\r\n\tT W E = ") == man);
    CHECK(base64_decode("TWFuTWE") == man);
    CHECK_THROWS(base64_decode("TWF*"), "invalid character");
    CHECK_THROWS(base64_decode("TW\x80u"), "invalid character");
    CHECK_THROWS(base64_decode("TQ==TQ=="), "data after padding");
    CHECK_THROWS(base64_decode("TQ=a"), "data after padding");
    CHECK_THROWS(base64_decode("T"), "truncated");
    CHECK_THROWS(base64_decode("TWFuT"), "truncated");
    CHECK_THROWS(base64_decode("TQ="), "truncated");
    CHECK_THROWS(base64_decode("TQ==="), "truncated");
    CHECK_THROWS(base64_decode("TWE=="), "truncated");
}

// ============================================================================
// JData
// ============================================================================
template<class T> static void jdata_type_round_trip(const char* tname, bool& ok) {
    std::vector<T> v(60);

    for (size_t i = 0; i < v.size(); i++) {
        v[i] = (T)((i * 37 + 3) % 120);

        if (std::is_signed<T>::value && (i & 1)) {
            v[i] = (T)(0 - (T)v[i]);
        }
    }

    std::vector<size_t> dims = {3, 4, 5};
    json jd = jdata_encode(tname, dims, v.data(), v.size() * sizeof(T));
    JDataArray a = jdata_decode_array(jd);
    ok &= (a.type == tname && a.esize == sizeof(T) && a.dims == dims && a.numel() == 60);
    ok &= (a.raw.size() == v.size() * sizeof(T) && memcmp(a.raw.data(), v.data(), a.raw.size()) == 0);

    for (size_t i = 0; i < v.size(); i++) {
        ok &= (a.at(i) == (double)v[i]);
    }
}

TEST(jdata_round_trip_all_types) {
    struct {
        const char* name;
        size_t es;
    } types[] = {{"single", 4}, {"float32", 4}, {"double", 8}, {"float64", 8}, {"int8", 1}, {"uint8", 1},
        {"int16", 2}, {"uint16", 2}, {"int32", 4}, {"uint32", 4}, {"int64", 8}, {"uint64", 8},
        {"logical", 1}, {"bool", 1}
    };

    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        CHECK(jdata_elem_size(types[i].name) == types[i].es);
    }

    CHECK(jdata_elem_size("complex") == 0);

    bool ok;
    ok = true;
    jdata_type_round_trip<float>("single", ok);
    jdata_type_round_trip<float>("float32", ok);
    CHECK(ok);
    ok = true;
    jdata_type_round_trip<double>("double", ok);
    jdata_type_round_trip<double>("float64", ok);
    CHECK(ok);
    ok = true;
    jdata_type_round_trip<int8_t>("int8", ok);
    CHECK(ok);
    ok = true;
    jdata_type_round_trip<uint8_t>("uint8", ok);
    jdata_type_round_trip<uint8_t>("logical", ok);
    jdata_type_round_trip<uint8_t>("bool", ok);
    CHECK(ok);
    ok = true;
    jdata_type_round_trip<int16_t>("int16", ok);
    CHECK(ok);
    ok = true;
    jdata_type_round_trip<uint16_t>("uint16", ok);
    CHECK(ok);
    ok = true;
    jdata_type_round_trip<int32_t>("int32", ok);
    CHECK(ok);
    ok = true;
    jdata_type_round_trip<uint32_t>("uint32", ok);
    CHECK(ok);
    ok = true;
    jdata_type_round_trip<int64_t>("int64", ok);
    CHECK(ok);
    ok = true;
    jdata_type_round_trip<uint64_t>("uint64", ok);
    CHECK(ok);

    /* extreme values survive */
    int64_t big[2] = {INT64_MIN, INT64_MAX};
    JDataArray a = jdata_decode_array(jdata_encode("int64", {2}, big, sizeof(big)));
    CHECK(memcmp(a.raw.data(), big, sizeof(big)) == 0);

    /* legacy helper returns the raw bytes */
    float f3[3] = {1.5f, -2.f, 3.25f};
    std::vector<uint8_t> raw = jdata_decode(jdata_encode("single", {3}, f3, sizeof(f3)));
    CHECK(raw.size() == 12 && memcmp(raw.data(), f3, 12) == 0);

    /* unknown type in a hand-made JDataArray converts to 0 */
    JDataArray u;
    u.type = "complex";
    u.esize = 4;
    u.raw.assign(4, 1);
    u.dims = {1};
    CHECK(u.at(0) == 0.0);
}

TEST(jdata_encoded_fields) {
    float v[6] = {1, 2, 3, 4, 5, 6};
    json jd = jdata_encode("single", {2, 3}, v, sizeof(v));
    CHECK(jd["_ArrayType_"] == "single");
    CHECK(jd["_ArraySize_"] == J("[2,3]"));
    CHECK(jd["_ArrayZipType_"] == "zlib");
    CHECK(jd["_ArrayZipSize_"] == 6);
    CHECK(!jd.contains("_ArrayOrder_"));
    CHECK(!jd.contains("_ArrayZipOffsets_"));
    json jc = jdata_encode("single", {2, 3}, v, sizeof(v), false);
    CHECK(jc["_ArrayOrder_"] == "c");
}

TEST(jdata_empty_array) {
    /* Regression (fixed): jdata_encode() of a zero-element array (e.g. an empty trajectory in
       save_trajectory, or any {0,N} table) throws "zlib compression failed"
       because zmat refuses a zero-length input; it should encode an empty
       stream that decodes back to zero elements */
    json jd;
    CHECK_NOTHROW(jd = jdata_encode("single", {0, 3}, NULL, 0));

    /* decoding a valid zlib stream of zero bytes (78 9c 03 00 00 00 00 01) */
    json jz = J(R"({"_ArrayType_":"single","_ArraySize_":[0,3],"_ArrayZipType_":"zlib","_ArrayZipData_":"eJwDAAAAAAE="})");
    JDataArray a;
    bool ok = true;

    try {
        a = jdata_decode_array(jz);
    } catch (const std::exception& e) {
        ok = false;
        fprintf(g_out, "  note: empty zlib array decode threw: %s\n", e.what());
    }

    CHECK(ok);
    CHECK(a.numel() == 0 && a.raw.empty());

    JDataArray b = jdata_decode_array(J(R"({"_ArrayType_":"double","_ArraySize_":[0],"_ArrayData_":[]})"));
    CHECK(b.numel() == 0 && b.raw.empty());
    JDataArray c = jdata_decode_array(J(R"({"_ArrayType_":"int8","_ArraySize_":[2,0],"_ArrayZipType_":"base64","_ArrayZipData_":""})"));
    CHECK(c.numel() == 0 && c.dims.size() == 2);
}

TEST(jdata_large_block_parallel) {
    /* > 4 MB: deflated in parallel blocks with an _ArrayZipOffsets_ index */
    const size_t n = 1500000;   /* 6 MB of float */
    std::vector<float> v(n);
    std::mt19937 rng(42);

    for (size_t i = 0; i < n; i++) {
        v[i] = (float)(rng() % 1000) * 0.25f + (float)i;
    }

    json jd = jdata_encode("single", {1000, 1500}, v.data(), n * sizeof(float));
    CHECK(jd.contains("_ArrayZipOffsets_"));
    CHECK(jd["_ArrayZipOffsets_"].is_array() && jd["_ArrayZipOffsets_"].size() >= 3);
    CHECK(jd["_ArrayZipOffsets_"][0].is_array() && jd["_ArrayZipOffsets_"][0].size() == 2);

    std::vector<size_t> off = jdata_zipoffsets(jd);
    CHECK(off.size() >= 6 && off.size() % 2 == 0);

    JDataArray a = jdata_decode_array(jd);
    CHECK(a.dims.size() == 2 && a.dims[0] == 1000 && a.dims[1] == 1500);
    CHECK(a.raw.size() == n * 4 && memcmp(a.raw.data(), v.data(), n * 4) == 0);

    /* without the index (serial path) the result is identical */
    json js = jd;
    js.erase("_ArrayZipOffsets_");
    JDataArray b = jdata_decode_array(js);
    CHECK(b.raw.size() == n * 4 && memcmp(b.raw.data(), v.data(), n * 4) == 0);

    /* a stale/odd index is only a hint and never breaks decoding */
    json jbad = jd;
    jbad["_ArrayZipOffsets_"] = J("[1, 2, 3]");
    JDataArray c = jdata_decode_array(jbad);
    CHECK(c.raw.size() == n * 4 && memcmp(c.raw.data(), v.data(), n * 4) == 0);
    jbad["_ArrayZipOffsets_"] = J("[[0, 0], [10, 4194304], [20, 6000000]]");
    JDataArray d = jdata_decode_array(jbad);
    CHECK(d.raw.size() == n * 4 && memcmp(d.raw.data(), v.data(), n * 4) == 0);
}

TEST(jdata_zipoffsets_parsing) {
    CHECK(jdata_zipoffsets(J(R"({})")).empty());
    CHECK(jdata_zipoffsets(J(R"({"_ArrayZipOffsets_": 5})")).empty());
    CHECK(jdata_zipoffsets(J(R"({"_ArrayZipOffsets_": [1,2,3,4]})")).size() == 4);
    CHECK(jdata_zipoffsets(J(R"({"_ArrayZipOffsets_": [[1,2],[3,4],[5,6]]})")).size() == 6);
    CHECK(jdata_zipoffsets(J(R"({"_ArrayZipOffsets_": [[1,2],[3,-4]]})")).empty());
    CHECK(jdata_zipoffsets(J(R"({"_ArrayZipOffsets_": [1,2,"x",4]})")).empty());
    CHECK(jdata_zipoffsets(J(R"({"_ArrayZipOffsets_": [1,2]})")).empty());
    CHECK(jdata_zipoffsets(J(R"({"_ArrayZipOffsets_": [1,2,3,4,5]})")).empty());
}

TEST(jdata_gzip_and_base64_ziptypes) {
    int32_t v[12];

    for (int i = 0; i < 12; i++) {
        v[i] = i * i - 50;
    }

    unsigned char* gz = NULL;
    size_t gzlen = 0;
    int status = 0;
    int ret = zmat_run(sizeof(v), (unsigned char*)v, &gzlen, &gz, zmGzip, &status, 1);
    CHECK(ret == 0 && gz != NULL && gzlen > 10);
    CHECK(gz && gz[0] == 0x1f && gz[1] == 0x8b);    /* gzip magic */

    json jd;
    jd["_ArrayType_"] = "int32";
    jd["_ArraySize_"] = J("[3,4]");
    jd["_ArrayZipType_"] = "gzip";
    jd["_ArrayZipSize_"] = J("[3,4]");    /* array form */
    jd["_ArrayZipData_"] = base64_encode(gz, gzlen);
    free(gz);
    JDataArray a = jdata_decode_array(jd);
    CHECK(a.type == "int32" && a.raw.size() == sizeof(v) && memcmp(a.raw.data(), v, sizeof(v)) == 0);

    jd["_ArrayZipSize_"] = 12;            /* scalar form */
    CHECK(jdata_decode_array(jd).raw.size() == sizeof(v));
    jd["_ArrayZipSize_"] = J("[2,6]");    /* same count, other shape: accepted */
    CHECK(jdata_decode_array(jd).raw.size() == sizeof(v));
    jd["_ArrayZipSize_"] = 13;
    CHECK_THROWS(jdata_decode_array(jd), "_ArrayZipSize_ does not match _ArraySize_");
    jd["_ArrayZipSize_"] = J("[4,4]");
    CHECK_THROWS(jdata_decode_array(jd), "_ArrayZipSize_ does not match _ArraySize_");
    jd["_ArrayZipSize_"] = "12";
    CHECK_THROWS(jdata_decode_array(jd), "_ArrayZipSize_ must be");

    /* base64 "compression": raw bytes */
    json jb;
    jb["_ArrayType_"] = "int32";
    jb["_ArraySize_"] = 12;
    jb["_ArrayZipType_"] = "base64";
    jb["_ArrayZipData_"] = base64_encode((const uint8_t*)v, sizeof(v));
    JDataArray b = jdata_decode_array(jb);
    CHECK(b.dims.size() == 1 && b.dims[0] == 12);
    CHECK(memcmp(b.raw.data(), v, sizeof(v)) == 0);

    /* _ArrayZipType_ defaults to zlib */
    json jz = jdata_encode("int32", {12}, v, sizeof(v));
    jz.erase("_ArrayZipType_");
    CHECK(memcmp(jdata_decode_array(jz).raw.data(), v, sizeof(v)) == 0);
}

TEST(jdata_uncompressed_arraydata) {
    JDataArray a = jdata_decode_array(J(R"({"_ArrayType_":"int32","_ArraySize_":[2,3],"_ArrayData_":[[1,2,3],[4,5,6]]})"));
    CHECK(a.type == "double" && a.esize == 8 && a.numel() == 6);
    bool ok = true;

    for (size_t i = 0; i < 6; i++) {
        ok &= (a.at(i) == (double)(i + 1));
    }

    CHECK(ok);

    JDataArray f = jdata_decode_array(J(R"({"_ArrayType_":"uint8","_ArraySize_":4,"_ArrayData_":[1, 2.5, true, false]})"));
    CHECK(f.at(0) == 1 && f.at(1) == 2.5 && f.at(2) == 1 && f.at(3) == 0);

    /* deeply but legally nested (8 levels) */
    JDataArray d8 = jdata_decode_array(J(R"({"_ArrayType_":"double","_ArraySize_":1,"_ArrayData_":[[[[[[[[7]]]]]]]]})"));
    CHECK(d8.at(0) == 7);

    /* base64 string form keeps the declared type */
    uint16_t u[4] = {1, 65535, 300, 7};
    json jb;
    jb["_ArrayType_"] = "uint16";
    jb["_ArraySize_"] = J("[2,2]");
    jb["_ArrayData_"] = base64_encode((const uint8_t*)u, sizeof(u));
    JDataArray b = jdata_decode_array(jb);
    CHECK(b.type == "uint16" && b.at(1) == 65535 && b.at(2) == 300);
}

TEST(jdata_column_major_transpose) {
    /* 2-D: column-major input, element (i,j) stored at i + j*2 */
    double cm[6];

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cm[i + j * 2] = 10 * i + j;
        }
    }

    json jd = jdata_encode("double", {2, 3}, cm, sizeof(cm), false);
    JDataArray a = jdata_decode_array(jd);
    bool ok = true;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            ok &= (a.at(i * 3 + j) == 10 * i + j);
        }
    }

    CHECK(ok);
    CHECK(a.at(0) == 0 && a.at(1) == 1 && a.at(2) == 2 && a.at(3) == 10 && a.at(4) == 11 && a.at(5) == 12);

    /* 3-D [2,3,4]: in[i + 2j + 6k] -> out[(i*3+j)*4+k] */
    int32_t c3[24];

    for (int i = 0; i < 24; i++) {
        c3[i] = i;
    }

    JDataArray b = jdata_decode_array(jdata_encode("int32", {2, 3, 4}, c3, sizeof(c3), false));
    ok = true;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                ok &= (b.at((i * 3 + j) * 4 + k) == c3[i + 2 * j + 6 * k]);
            }
        }
    }

    CHECK(ok);
    CHECK(b.at(1) == 6 && b.at(4) == 2);   /* (0,0,1) -> 6; (0,1,0) -> 2 */

    /* "F"/"col" spellings and row-major "r" */
    json jf = jdata_encode("int32", {2, 3, 4}, c3, sizeof(c3));
    jf["_ArrayOrder_"] = "F";
    CHECK(jdata_decode_array(jf).at(1) == 6);
    jf["_ArrayOrder_"] = "col";
    CHECK(jdata_decode_array(jf).at(1) == 6);
    jf["_ArrayOrder_"] = "r";
    CHECK(jdata_decode_array(jf).at(1) == 1);
    jf["_ArrayOrder_"] = 1;
    CHECK_THROWS(jdata_decode_array(jf), "_ArrayOrder_ must be a string");

    /* nested _ArrayData_ in column-major order is transposed too */
    JDataArray n = jdata_decode_array(J(R"({"_ArrayType_":"double","_ArraySize_":[2,3],"_ArrayOrder_":"c","_ArrayData_":[1,2,3,4,5,6]})"));
    CHECK(n.at(0) == 1 && n.at(1) == 3 && n.at(2) == 5 && n.at(3) == 2 && n.at(4) == 4 && n.at(5) == 6);

    /* 1-D: nothing to transpose */
    JDataArray o = jdata_decode_array(J(R"({"_ArrayType_":"double","_ArraySize_":3,"_ArrayOrder_":"c","_ArrayData_":[1,2,3]})"));
    CHECK(o.at(0) == 1 && o.at(2) == 3);
}

TEST(jdata_errors) {
    float v[4] = {1, 2, 3, 4};
    json good = jdata_encode("single", {4}, v, sizeof(v));

    CHECK_THROWS(jdata_decode_array(J("[1,2]")), "must be a JData object");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArraySize_":4,"_ArrayData_":[1,2,3,4]})")), "needs a string _ArrayType_");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":5,"_ArraySize_":4,"_ArrayData_":[1,2,3,4]})")), "needs a string _ArrayType_");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"complex","_ArraySize_":4,"_ArrayData_":[1,2,3,4]})")),
                 "unsupported _ArrayType_ 'complex'");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"single","_ArrayData_":[1,2,3,4]})")), "needs _ArraySize_");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"single","_ArraySize_":4})")), "needs _ArrayZipData_ or _ArrayData_");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"single","_ArraySize_":-4,"_ArrayData_":[1]})")),
                 "must be a non-negative integer");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"single","_ArraySize_":[2,-2],"_ArrayData_":[1]})")),
                 "entries must be non-negative integers");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"single","_ArraySize_":[2,1.5],"_ArrayData_":[1]})")),
                 "entries must be non-negative integers");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"single","_ArraySize_":[],"_ArrayData_":[1]})")),
                 "must be a non-negative integer or an array of 1-8");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"single","_ArraySize_":[1,1,1,1,1,1,1,1,1],"_ArrayData_":[1]})")),
                 "array of 1-8");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"single","_ArraySize_":[2097152,2097152],"_ArrayData_":[1]})")),
                 "array is too large");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"single","_ArraySize_":[65536,65536,65536],"_ArrayZipData_":"eJw="})")),
                 "array is too large");

    /* size mismatch between the payload and _ArraySize_ */
    json m = good;
    m["_ArraySize_"] = 5;
    m["_ArrayZipSize_"] = 5;
    CHECK_THROWS(jdata_decode_array(m), "decoded 16 bytes, expected 20");
    json mb;
    mb["_ArrayType_"] = "single";
    mb["_ArraySize_"] = 3;
    mb["_ArrayData_"] = base64_encode((const uint8_t*)v, sizeof(v));
    CHECK_THROWS(jdata_decode_array(mb), "decoded 16 bytes, expected 12");

    /* corrupt or bogus compressed streams */
    json c = good;
    c["_ArrayZipData_"] = base64_encode((const uint8_t*)"this is not a zlib stream", 25);
    CHECK_THROWS(jdata_decode_array(c), "failed to decompress zlib data");
    c["_ArrayZipType_"] = "gzip";
    CHECK_THROWS(jdata_decode_array(c), "failed to decompress gzip data");
    std::vector<uint8_t> z = base64_decode(good["_ArrayZipData_"].get<std::string>());
    z.resize(z.size() / 2);    /* truncated stream */
    c = good;
    c["_ArrayZipData_"] = base64_encode(z.data(), z.size());
    CHECK_THROWS(jdata_decode_array(c), "JData array");
    c["_ArrayZipData_"] = "";
    CHECK_THROWS(jdata_decode_array(c), "empty compressed stream");
    c["_ArrayZipData_"] = "!!!!";
    CHECK_THROWS(jdata_decode_array(c), "bad base64");
    c["_ArrayZipData_"] = 5;
    CHECK_THROWS(jdata_decode_array(c), "_ArrayZipData_ must be a base64 string");
    c = good;
    c["_ArrayZipType_"] = "lzma";
    CHECK_THROWS(jdata_decode_array(c), "unsupported _ArrayZipType_ 'lzma'");
    c["_ArrayZipType_"] = 3;
    CHECK_THROWS(jdata_decode_array(c), "_ArrayZipType_ must be a string");

    /* uncompressed _ArrayData_ */
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"double","_ArraySize_":3,"_ArrayData_":[1,2,3,4]})")),
                 "more elements than _ArraySize_");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"double","_ArraySize_":3,"_ArrayData_":[1,2]})")),
                 "fewer elements than _ArraySize_");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"double","_ArraySize_":2,"_ArrayData_":[1,"a"]})")),
                 "must contain only numbers");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"double","_ArraySize_":2,"_ArrayData_":{"a":1}})")),
                 "must be an array or a base64 string");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"double","_ArraySize_":1,"_ArrayData_":[[[[[[[[[1]]]]]]]]]})")),
                 "nested too deeply");
    CHECK_THROWS(jdata_decode_array(J(R"({"_ArrayType_":"double","_ArraySize_":2,"_ArrayData_":[1]})"), "MyCtx"), "MyCtx: _ArrayData_ has fewer");
    CHECK(jdata_nthread() >= 1 && jdata_nthread() <= 16);
}

// ============================================================================
// JSON helpers, packing, small parsers
// ============================================================================
TEST(json_helpers) {
    json o = J(R"({"a": 2.5, "b": true, "c": "s", "n": null, "v": [1, 2, 3, 4], "bad": "x"})");
    CHECK(json_num(o, "a", 0, "X") == 2.5);
    CHECK(json_num(o, "b", 0, "X") == 1.0);
    CHECK(json_num(o, "missing", 7, "X") == 7);
    CHECK(json_num(o, "n", 7, "X") == 7);
    CHECK(json_num(J("[1]"), "a", 3, "X") == 3);
    CHECK_THROWS(json_num(o, "c", 0, "X"), "X.c must be a number");
    json nan;
    nan["a"] = NAN;
    CHECK_THROWS(json_num(nan, "a", 0, "X"), "X.a must be finite");
    nan["a"] = INFINITY;
    CHECK_THROWS(json_num(nan, "a", 0, "X"), "X.a must be finite");
    CHECK(json_num_range(o, "a", 0, 0, 3, "X") == 2.5);
    CHECK_THROWS(json_num_range(o, "a", 0, 3, 4, "X"), "X.a=2.5 is out of range [3, 4]");
    CHECK(json_bool(o, "b", false, "X") == true);
    CHECK(json_bool(o, "a", false, "X") == true);
    CHECK(json_bool(o, "missing", true, "X") == true);
    CHECK_THROWS(json_bool(o, "c", false, "X"), "X.c must be a boolean or a number");
    CHECK(json_str(o, "c", "d", "X") == "s");
    CHECK(json_str(o, "missing", "d", "X") == "d");
    CHECK_THROWS(json_str(o, "a", "d", "X"), "X.a must be a string");
    CHECK(json_vec(o["v"], 1, 4, "V").size() == 4);
    CHECK_THROWS(json_vec(o["v"], 1, 2, "V"), "V has more than 2 numbers");
    CHECK_THROWS(json_vec(o["a"], 1, 4, "V"), "V must be an array");
    CHECK_THROWS(json_vec(o["v"], 5, 6, "V"), "V needs at least 5 numbers");
    CHECK_THROWS(json_vec(J(R"([1, "x"])"), 1, 4, "V"), "V[1] must be a number");
    json vn = json::array({1.0, NAN});
    CHECK_THROWS(json_vec(vn, 1, 4, "V"), "V[1] must be finite");
    CHECK(json_vec_key(o, "v", 4, "O").size() == 4);
    CHECK_THROWS(json_vec_key(o, "w", 3, "O"), "O requires 'w'");

    /* Regression (fixed): json_vec documents "at least nmin and at most nmax numbers", but
       silently truncates longer arrays instead of rejecting them, so e.g.
       Optode.Source.Pos = [1,2,3,4] or Shapes...O = [1,2,3,4] is accepted */
    CHECK_THROWS(json_vec(o["v"], 3, 3, "V"), "V");
}

TEST(pack_unpack_media) {
    uint32_t fr, bk;
    bool fl;
    unpack_media(pack_media(5, 7), fr, bk, fl);
    CHECK(fr == 5 && bk == 7 && !fl);
    unpack_media(pack_media(0x3FFF, 0xFFFF), fr, bk, fl);
    CHECK(fr == 0x3FFF && bk == 0xFFFF && !fl);
    unpack_media(pack_media(0x4001, 2), fr, bk, fl);   /* front keeps 14 bits */
    CHECK(fr == 1 && bk == 2);
    CHECK((pm_bits(pack_media(0x7FFF, 0)) & (VKMMC_FACE_FLAT | VKMMC_FACE_COPLANAR)) == 0);

    uint32_t p = pm_bits(pack_media(3, 4)) | VKMMC_FACE_FLAT | VKMMC_FACE_COPLANAR;
    float pf;
    memcpy(&pf, &p, 4);
    unpack_media(pf, fr, bk, fl);
    CHECK(fr == 3 && bk == 4 && fl);

    unpack_media(pack_tag(9), fr, bk, fl);
    CHECK(fr == 0 && bk == 9 && !fl);
    unpack_media(pack_tag_flat(9), fr, bk, fl);
    CHECK(fr == 0 && bk == 9 && fl);
    FaceData fd = {0, 0, 1, pack_tag_flat(2)};
    CHECK(face_is_flat(fd));
    fd.packed_media = pack_tag(2);
    CHECK(!face_is_flat(fd));
}

TEST(small_parsers) {
    CHECK(parse_outputtype("flux") == OT_FLUX);
    CHECK(parse_outputtype("f") == OT_FLUX);
    CHECK(parse_outputtype("F") == OT_FLUX);
    CHECK(parse_outputtype("fluence") == OT_FLUENCE);
    CHECK(parse_outputtype("l") == OT_FLUENCE);
    CHECK(parse_outputtype("L") == OT_FLUENCE);
    CHECK(parse_outputtype("energy") == OT_ENERGY);
    CHECK(parse_outputtype("x") == OT_ENERGY);
    CHECK(parse_outputtype("E") == OT_ENERGY);
    CHECK(parse_outputtype("") == OT_ENERGY);
    CHECK(parse_outputtype("q") == -1);
    /* Regression (fixed): the first letter is matched case-insensitively but "fluen" is not,
       so a capitalized "Fluence" (or "FLUENCE") silently selects flux */
    CHECK(parse_outputtype("Fluence") == OT_FLUENCE);

    CHECK(parse_srctype("pencil") == SRC_PENCIL);
    CHECK(parse_srctype("planar") == SRC_PLANAR);
    CHECK(parse_srctype("disk") == SRC_DISK);
    CHECK(parse_srctype("zgaussian") == SRC_ZGAUSSIAN);
    CHECK(parse_srctype("slit") == SRC_SLIT);
    CHECK(parse_srctype("") == -1);
    CHECK(parse_srctype("laser") == -1);
    CHECK(srctype_supported(SRC_PENCIL) && srctype_supported(SRC_PLANAR) && srctype_supported(SRC_DISK));
    CHECK(!srctype_supported(SRC_ISOTROPIC) && !srctype_supported(SRC_CONE));
    CHECK(std::string(srctype_name(SRC_PENCIL)) == "pencil");
    CHECK(std::string(srctype_name(SRC_PLANAR)) == "planar");
    CHECK(std::string(srctype_name(SRC_DISK)) == "disk");
    CHECK(std::string(srctype_name(SRC_CONE)) == "unknown");
    CHECK(std::string(kernel_variant_name(KV_MESH)) == "mesh");
    CHECK(std::string(kernel_variant_name(KV_CSG_CURV)) == "csg+curvature");
    CHECK(std::string(kernel_variant_name(7)) == "unknown");
    CHECK(std::string(output_type_name(OT_FLUX)) == "flux");
    CHECK(std::string(output_type_name(OT_FLUENCE)) == "fluence");
    CHECK(std::string(output_type_name(OT_ENERGY)) == "energy");
}

// ============================================================================
// load_json_config / load_json_file
// ============================================================================
TEST(config_minimal_csg_defaults) {
    SimConfig c = load_json_config(J(R"({"Domain": {"Media": [{"mua":0,"mus":0}, {"mua":0.01,"mus":1,"g":0.9,"n":1.37}]},
                                         "Shapes": [{"Grid": {"Size": [60, 60, 60], "Tag": 1}}]})"));
    CHECK(c.is_csg);
    CHECK(c.shapes.is_array() && c.shapes.size() == 1);
    CHECK(c.session_id == "default");
    CHECK(c.nphoton == 1000000);
    CHECK(c.rng_seed == 29012391u);
    CHECK(c.do_mismatch && c.do_normalize);
    CHECK(c.output_type == OT_ENERGY);
    CHECK_CLOSE(c.t0, 0, 0);
    CHECK_CLOSE(c.t1, 5e-9, 1e-15);
    CHECK_CLOSE(c.dt, 5e-9, 1e-15);
    CHECK(c.maxgate == 1);
    CHECK(c.unitinmm == 1.0f);
    CHECK(c.srctype == SRC_PENCIL);
    CHECK(c.srcdir[0] == 0 && c.srcdir[1] == 0 && c.srcdir[2] == 1);
    CHECK(c.mesh_res == 24);
    CHECK(!c.has_steps);
    CHECK(c.mediumid0 == 0xFFFFFFFFu);
    CHECK(c.media.size() == 2);
    CHECK(c.media[0].g == 1 && c.media[0].n == 1);      /* object defaults */
    CHECK_CLOSE(c.media[1].mua, 0.01, 1e-7);
    CHECK_CLOSE(c.media[1].n, 1.37, 1e-6);
    CHECK(c.nodes.empty() && c.faces.empty());
}

TEST(config_session_forward_domain) {
    json j = patched(base_csg(), R"({"Session": {"ID": "abc", "Photons": 1e7, "RNGSeed": 7, "DoMismatch": false,
        "DoNormalize": 0, "OutputType": "fluence", "RussianRoulette": 0.01, "RussianRouletteSize": 20,
        "DebugFlag": 8192, "MaxJumpDebug": 500},
        "Forward": {"T0": 0, "T1": 5e-9, "Dt": 1e-9},
        "Domain": {"LengthUnit": 0.5, "Step": [2], "MeshRes": 16}})");
    SimConfig c = load_json_config(j);
    CHECK(c.session_id == "abc");
    CHECK(c.nphoton == 10000000ull);
    CHECK(c.rng_seed == 7);
    CHECK(!c.do_mismatch && !c.do_normalize);
    CHECK(c.output_type == OT_FLUENCE);
    CHECK_CLOSE(c.minenergy, 0.01, 1e-8);
    CHECK_CLOSE(c.roulettesize, 20, 0);
    CHECK(c.debuglevel == 8192 && c.maxjumpdebug == 500);
    CHECK(c.maxgate == 5);
    CHECK_CLOSE(c.dt, 1e-9, 1e-16);
    CHECK(c.unitinmm == 0.5f);
    CHECK(c.has_steps && c.steps[0] == 2 && c.steps[1] == 2 && c.steps[2] == 2);
    CHECK(c.mesh_res == 16);

    c = load_json_config(patched(base_csg(), R"({"Domain": {"Step": [1, 2, 3]}, "Forward": {"T0": 1e-9, "T1": 3e-9}})"));
    CHECK(c.steps[0] == 1 && c.steps[1] == 2 && c.steps[2] == 3);
    CHECK(c.maxgate == 1);    /* Dt defaults to T1-T0 */
    CHECK_CLOSE(c.dt, 2e-9, 1e-16);

    c = load_json_config(patched(base_csg(), R"({"Session": {"OutputType": "f"}, "Forward": {"T1": 1e-8, "Dt": 3e-9}})"));
    CHECK(c.output_type == OT_FLUX);
    CHECK(c.maxgate == 3);    /* round((T1-T0)/Dt) */
    c = load_json_config(patched(base_csg(), R"({"Session": {"OutputType": "x", "DoMismatch": true}})"));
    CHECK(c.output_type == OT_ENERGY && c.do_mismatch);
}

TEST(config_media_forms) {
    json j = base_csg();
    j["Domain"]["Media"] = J(R"([[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.1,"mus":10,"g":0.9,"n":1.4}]])");
    SimConfig c = load_json_config(j);
    CHECK(c.media.size() == 2 && c.media[1].mus == 10.0f);

    /* a single wrapped 4-element-array list is unwrapped once */
    j["Domain"]["Media"] = J(R"([[[0,0,1,1],[0.1,10,0.9,1.4]]])");
    c = load_json_config(j);
    CHECK(c.media.size() == 2 && c.media[1].mus == 10.0f && c.media[1].n == 1.4f);

    /* Regression (fixed): Media given as a list of [mua,mus,g,n] arrays is mistaken for the
       [[...]] wrapper (ma[0] is an array, so it is unwrapped to its first
       medium [0,0,1,1] and the numbers are then rejected as media) */
    j["Domain"]["Media"] = J(R"([[0,0,1,1],[0.1,10,0.9,1.4],[0.2,5,0.8,1.3]])");
    bool ok = true;

    try {
        c = load_json_config(j);
    } catch (const std::exception& e) {
        ok = false;
        fprintf(g_out, "  note: Media as 4-element arrays threw: %s\n", e.what());
    }

    CHECK(ok);
    CHECK(ok && c.media.size() == 3 && c.media[2].mua == 0.2f && c.media[1].g == 0.9f);
}

#define CHECK_CFG_THROWS(base, patch, substr) CHECK_THROWS(load_json_config(patched(base, patch)), substr)

TEST(config_validation_errors) {
    json b = base_csg();
    CHECK_THROWS(load_json_config(J("[1,2]")), "input must be a JSON object");
    CHECK_THROWS(load_json_config(J("3")), "input must be a JSON object");
    CHECK_CFG_THROWS(b, R"({"Session": 5})", "Session must be an object");
    CHECK_CFG_THROWS(b, R"({"Session": {"Photons": 0}})", "Session.Photons=0 is out of range");
    CHECK_CFG_THROWS(b, R"({"Session": {"Photons": 0.5}})", "Session.Photons");
    CHECK_CFG_THROWS(b, R"({"Session": {"Photons": "many"}})", "Session.Photons must be a number");
    CHECK_CFG_THROWS(b, R"({"Session": {"ID": ""}})", "Session.ID must have 1-255 characters");
    CHECK_CFG_THROWS(b, R"({"Session": {"ID": 5}})", "Session.ID must be a string");
    CHECK_CFG_THROWS(b, R"({"Session": {"OutputType": "q"}})", "Session.OutputType must be flux, fluence or energy");
    CHECK_CFG_THROWS(b, R"({"Session": {"DoMismatch": "yes"}})", "Session.DoMismatch must be a boolean or a number");
    CHECK_CFG_THROWS(b, R"({"Session": {"RussianRoulette": 2}})", "Session.RussianRoulette");
    CHECK_CFG_THROWS(b, R"({"Session": {"RNGSeed": 1e10}})", "Session.RNGSeed");
    CHECK_CFG_THROWS(b, R"({"Forward": 1})", "Forward must be an object");
    CHECK_CFG_THROWS(b, R"({"Forward": {"T0": 5e-9, "T1": 5e-9}})", "Forward: need 0 <= T0 < T1");
    CHECK_CFG_THROWS(b, R"({"Forward": {"T0": 5e-9, "T1": 1e-9}})", "Forward: need 0 <= T0 < T1");
    CHECK_CFG_THROWS(b, R"({"Forward": {"T0": -1e-9, "T1": 1e-9}})", "Forward: need 0 <= T0 < T1");
    CHECK_CFG_THROWS(b, R"({"Forward": {"T1": 1e-9, "Dt": 0}})", "Forward.Dt must be positive");
    CHECK_CFG_THROWS(b, R"({"Forward": {"T1": 1e-9, "Dt": -1e-10}})", "Forward.Dt must be positive");
    CHECK_CFG_THROWS(b, R"({"Forward": {"T1": 1, "Dt": 1e-9}})", "time gates");
    CHECK_NOTHROW(load_json_config(patched(b, R"({"Forward": {"T1": 1e-4, "Dt": 1e-9}})")));   /* exactly 100000 gates */
    CHECK_CFG_THROWS(b, R"({"Domain": null})", "input requires a Domain object");
    CHECK_CFG_THROWS(b, R"({"Domain": 4})", "input requires a Domain object");
    CHECK_CFG_THROWS(b, R"({"Domain": {"LengthUnit": 0}})", "Domain.LengthUnit=0 is out of range");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Media": null}})", "Domain requires Media");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Media": []}})", "Domain.Media must be a non-empty array");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Media": {"mua": 1}}})", "Domain.Media must be a non-empty array");
    json many = b;
    many["Domain"]["Media"] = json::array();

    for (int i = 0; i < 257; i++) {
        many["Domain"]["Media"].push_back(J(R"({"mua":0,"mus":0,"g":1,"n":1})"));
    }

    CHECK_THROWS(load_json_config(many), "at most 256 media");
    many["Domain"]["Media"].erase(256);
    CHECK_NOTHROW(load_json_config(many));
    CHECK_CFG_THROWS(b, R"({"Domain": {"Media": [{"mua": -0.1}]}})", "Domain.Media[0].mua must be in [0, 1e6)");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Media": [{"mus": -1}]}})", "Domain.Media[0].mus must be in [0, 1e6)");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Media": [{"mua": 0}, {"g": 1.5}]}})", "Domain.Media[1].g must be in [-1, 1]");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Media": [{"mua": 0}, {"g": -1.01}]}})", "Domain.Media[1].g must be in [-1, 1]");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Media": [{"n": 0}]}})", "Domain.Media[0].n must be in [0.01, 100]");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Media": [{"n": -1.3}]}})", "Domain.Media[0].n must be in [0.01, 100]");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Media": [{"mua": "x"}]}})", "Domain.Media[0].mua must be a number");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Media": ["water"]}})", "Domain.Media[0] must be an object");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Media": [[[0, 0, 1]]]}})", "Domain.Media[0] needs at least 4 numbers");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Step": 0}})", "Domain.Step must be an array");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Step": []}})", "Domain.Step needs at least 1 numbers");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Step": [0]}})", "Domain.Step must be in (1e-6, 1e6)");
    CHECK_CFG_THROWS(b, R"({"Domain": {"Step": [1, -2, 1]}})", "Domain.Step must be in (1e-6, 1e6)");
    CHECK_CFG_THROWS(b, R"({"Domain": {"MeshRes": 0}})", "Domain.MeshRes=0 is out of range [1, 2048]");
    CHECK_CFG_THROWS(b, R"({"Domain": {"MeshRes": 4096}})", "Domain.MeshRes=4096 is out of range");
    CHECK_CFG_THROWS(b, R"({"Optode": {"Source": 3}})", "Optode.Source must be an object");
    CHECK_CFG_THROWS(b, R"({"Optode": {"Source": {"Type": "laser"}}})", "Optode.Source.Type 'laser' is not a known source type");
    CHECK_CFG_THROWS(b, R"({"Optode": {"Source": {"Type": "isotropic"}}})", "'isotropic' is not supported yet");
    CHECK_CFG_THROWS(b, R"({"Optode": {"Source": {"Type": "cone"}}})", "not supported yet");
    CHECK_CFG_THROWS(b, R"({"Optode": {"Source": {"Pos": [1, 2]}}})", "Optode.Source.Pos needs at least 3 numbers");
    CHECK_CFG_THROWS(b, R"({"Optode": {"Source": {"Pos": [1, 2, "z"]}}})", "Optode.Source.Pos[2] must be a number");
    CHECK_CFG_THROWS(b, R"({"Optode": {"Source": {"Dir": [0, 0, 0]}}})", "Optode.Source.Dir must be a non-zero vector");
    CHECK_CFG_THROWS(b, R"({"Optode": {"Source": {"Dir": [1, 0]}}})", "Optode.Source.Dir needs at least 3 numbers");
    CHECK_CFG_THROWS(b, R"({"Optode": {"Source": {"Param1": []}}})", "Optode.Source.Param1 needs at least 1 numbers");
    CHECK_CFG_THROWS(b, R"({"Optode": {"Source": {"Param2": "x"}}})", "Optode.Source.Param2 must be an array");
    CHECK_CFG_THROWS(b, R"({"Optode": {"Source": {"Type": "disk", "Param1": [-1]}}})", "disk radius) must be non-negative");
    CHECK_CFG_THROWS(b, R"({"Optode": {"Source": {"Type": 5}}})", "Optode.Source.Type must be a string");
    CHECK_CFG_THROWS(b, R"({"Shapes": null})", "input requires 'Shapes'");
    CHECK_CFG_THROWS(b, R"({"Shapes": []})", "Shapes is an empty array");
    CHECK_CFG_THROWS(b, R"({"Shapes": "box"})", "Shapes must be an array of shapes or a mesh object");

    /* NaN/Inf can only come from a programmatically built json */
    json n = b;
    n["Domain"]["LengthUnit"] = NAN;
    CHECK_THROWS(load_json_config(n), "Domain.LengthUnit must be finite");
    n = b;
    n["Session"]["Photons"] = INFINITY;
    CHECK_THROWS(load_json_config(n), "Session.Photons must be finite");
    n = b;
    n["Domain"]["Media"][1]["mua"] = NAN;
    CHECK_THROWS(load_json_config(n), "Domain.Media[1].mua must be finite");
    n = b;
    n["Optode"]["Source"]["Pos"][0] = -INFINITY;
    CHECK_THROWS(load_json_config(n), "Optode.Source.Pos[0] must be finite");
}

TEST(config_source) {
    json b = base_csg();
    SimConfig c = load_json_config(patched(b, R"({"Optode": {"Source": {"Type": "disk", "Pos": [1, 2, 3],
        "Dir": [0, 3, 4, 7], "Param1": [2.5], "Param2": [1, 2, 3, 4]}}})"));
    CHECK(c.srctype == SRC_DISK);
    CHECK(c.srcpos[0] == 1 && c.srcpos[1] == 2 && c.srcpos[2] == 3);
    CHECK_CLOSE(c.srcdir[0], 0, 1e-7);
    CHECK_CLOSE(c.srcdir[1], 0.6, 1e-6);
    CHECK_CLOSE(c.srcdir[2], 0.8, 1e-6);
    CHECK(c.srcdir[3] == 7);    /* 4th element kept, not normalized */
    CHECK(c.srcparam1[0] == 2.5f && c.srcparam1[1] == 0);
    CHECK(c.srcparam2[3] == 4);
    c = load_json_config(patched(b, R"({"Optode": {"Source": {"Type": "planar", "Dir": [-2, 0, 0]}}})"));
    CHECK(c.srctype == SRC_PLANAR && c.srcdir[0] == -1.0f);
    c = load_json_config(patched(b, R"({"Optode": {"Source": {"Type": "disk"}}})"));
    CHECK(c.srctype == SRC_DISK && c.srcparam1[0] == 0);
    c = load_json_config(patched(b, R"({"Optode": {"Detector": []}})"));
    CHECK(c.srcpos[0] == 30 && c.srcpos[2] == 0);
    json es = b;
    es["Optode"]["Source"] = json::object();
    c = load_json_config(es);
    CHECK(c.srctype == SRC_PENCIL && c.srcpos[0] == 0 && c.srcdir[2] == 1);
}

static json cube_mesh_json(int label2 = 2) {
    std::vector<Vec3> n = cube_nodes();
    std::vector<int32_t> e = cube_elems(label2);
    json j = base_csg();
    j["Shapes"] = json::object();
    json& nodes = j["Shapes"]["MeshNode"];
    nodes = json::array();

    for (size_t i = 0; i < n.size(); i++) {
        nodes.push_back({n[i].x, n[i].y, n[i].z});
    }

    json& el = j["Shapes"]["MeshElem"];
    el = json::array();

    for (size_t i = 0; i < e.size(); i += 5) {
        el.push_back({e[i], e[i + 1], e[i + 2], e[i + 3], e[i + 4]});
    }

    return j;
}

TEST(config_mesh_mode) {
    json j = cube_mesh_json();
    SimConfig c = load_json_config(j);
    CHECK(!c.is_csg);
    CHECK(c.nodes.size() == 8);
    CHECK(c.faces.size() == 14 && c.facedata.size() == 14);
    CHECK(c.nmin.x == 0 && c.nmin.y == 0 && c.nmin.z == 0 && c.nmax.x == 1 && c.nmax.y == 1 && c.nmax.z == 1);
    CHECK(c.mediumid0 == 0xFFFFFFFFu && c.init_elem == -1);

    j["Shapes"]["InitElem"] = 3;   /* perm (1,0,2): y before x -> label 2 */
    c = load_json_config(j);
    CHECK(c.init_elem == 3 && c.mediumid0 == 2);
    j["Shapes"]["InitElem"] = 1;
    c = load_json_config(j);
    CHECK(c.mediumid0 == 1);
    j["Shapes"]["InitElem"] = 0;   /* <= 0: not used */
    c = load_json_config(j);
    CHECK(c.mediumid0 == 0xFFFFFFFFu);

    /* a JData-encoded element table goes through the same checks */
    std::vector<int32_t> e = cube_elems(2);
    json jd = cube_mesh_json();
    jd["Shapes"]["MeshElem"] = jdata_encode("int32", {6, 5}, e.data(), e.size() * 4);
    std::vector<float> nf;
    std::vector<Vec3> nv = cube_nodes();

    for (size_t i = 0; i < nv.size(); i++) {
        nf.push_back(nv[i].x);
        nf.push_back(nv[i].y);
        nf.push_back(nv[i].z);
    }

    jd["Shapes"]["MeshNode"] = jdata_encode("single", {8, 3}, nf.data(), nf.size() * 4);
    c = load_json_config(jd);
    CHECK(c.nodes.size() == 8 && c.faces.size() == 14);

    /* MeshSurf input */
    json js = base_csg();
    js["Shapes"] = J(R"({"MeshNode": [[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
                         "MeshSurf": [[1,3,2,1],[1,2,4,1],[1,4,3,1],[2,3,4,1]]})");
    c = load_json_config(js);
    CHECK(c.faces.size() == 4);
}

TEST(config_mesh_errors) {
    json j = cube_mesh_json();
    json bad = j;
    bad["Shapes"]["MeshElem"][0][1] = 9;
    CHECK_THROWS(load_json_config(bad), "Shapes.MeshElem row 1: node index 9.000000 is not an integer in [1, 8]");
    bad = j;
    bad["Shapes"]["MeshElem"][2][0] = 0;
    CHECK_THROWS(load_json_config(bad), "Shapes.MeshElem row 3: node index");
    bad = j;
    bad["Shapes"]["MeshElem"][0][2] = 2.5;
    CHECK_THROWS(load_json_config(bad), "is not an integer in [1, 8]");
    bad = j;
    bad["Shapes"]["MeshElem"][1][4] = 3;    /* 3 media: labels 0..2 */
    CHECK_THROWS(load_json_config(bad), "Shapes.MeshElem row 2: label 3.000000 is not an integer in [0, 2]");
    bad = j;
    bad["Shapes"]["MeshElem"][1][4] = -1;
    CHECK_THROWS(load_json_config(bad), "label");
    bad = j;
    bad["Shapes"]["MeshElem"][1][4] = 1.5;
    CHECK_THROWS(load_json_config(bad), "label 1.500000 is not an integer");
    bad = j;
    bad["Shapes"]["MeshElem"][1] = J("[1, 2, 3]");
    CHECK_THROWS(load_json_config(bad), "Shapes.MeshElem row 2 must have 5 numbers");
    bad = j;
    bad["Shapes"]["MeshElem"] = J("[[1, 2, 3, 4]]");
    CHECK_THROWS(load_json_config(bad), "Shapes.MeshElem needs at least 5 columns");
    bad = j;
    bad["Shapes"]["MeshElem"] = J("[]");
    CHECK_THROWS(load_json_config(bad), "Shapes.MeshElem is empty");
    bad = j;
    bad["Shapes"]["MeshElem"] = J("5");
    CHECK_THROWS(load_json_config(bad), "Shapes.MeshElem must be a JData object or an array of rows");
    bad = j;
    bad["Shapes"]["MeshElem"][0][0] = "a";
    CHECK_THROWS(load_json_config(bad), "Shapes.MeshElem must contain only numbers");
    bad = j;
    bad["Shapes"]["MeshElem"] = J(R"({"_ArrayType_":"int32","_ArraySize_":5,"_ArrayData_":[1,2,3,4,1]})");
    CHECK_THROWS(load_json_config(bad), "Shapes.MeshElem must be a 2-D array");
    bad = j;
    bad["Shapes"].erase("MeshNode");
    CHECK_THROWS(load_json_config(bad), "Mesh mode requires Shapes.MeshNode");
    bad = j;
    bad["Shapes"].erase("MeshElem");
    CHECK_THROWS(load_json_config(bad), "Mesh mode requires Shapes.MeshSurf or Shapes.MeshElem");
    bad = j;
    bad["Shapes"]["MeshNode"] = J("[[0, 0], [1, 1]]");
    CHECK_THROWS(load_json_config(bad), "Shapes.MeshNode needs at least 3 columns");
    bad = j;
    bad["Shapes"]["MeshNode"][3][1] = 1e39;
    CHECK_THROWS(load_json_config(bad), "Shapes.MeshNode: node 4 is outside the float range");
    bad = j;
    bad["Shapes"]["MeshNode"][3][1] = NAN;
    CHECK_THROWS(load_json_config(bad), "Shapes.MeshNode contains a NaN or Inf");
    bad = j;
    bad["Shapes"]["InitElem"] = 7;
    CHECK_THROWS(load_json_config(bad), "Shapes.InitElem is larger than the element count");
    bad = j;
    bad["Shapes"]["InitElem"] = "1";
    CHECK_THROWS(load_json_config(bad), "Shapes.InitElem must be a number");

    /* Regression (fixed): InitElem is converted with (int) before the range check, so a value
       beyond INT_MAX (undefined behavior; INT_MIN on x86) is silently ignored
       instead of being rejected as larger than the element count */
    bad = j;
    bad["Shapes"]["InitElem"] = 1e10;
    CHECK_THROWS(load_json_config(bad), "Shapes.InitElem");

    /* Regression (fixed): every element in medium 0 (the background) has no surface between
       different media, but extract_surface_from_tet() still emits the
       exterior faces as front=0/back=0 triangles (load_mesh_surf drops
       front==back faces), so this check never fires for MeshElem input */
    bad = j;

    for (size_t i = 0; i < 6; i++) {
        bad["Shapes"]["MeshElem"][i][4] = 0;
    }

    CHECK_THROWS(load_json_config(bad), "the mesh has no surface triangles between different media");

    json js = base_csg();
    js["Shapes"] = J(R"({"MeshNode": [[0,0,0],[1,0,0],[0,1,0],[0,0,1]], "MeshSurf": [[1,3,2,1]], "InitElem": 1})");
    CHECK_THROWS(load_json_config(js), "Shapes.InitElem requires Shapes.MeshElem");
    js["Shapes"]["MeshSurf"] = J("[[1,3,5,1]]");
    CHECK_THROWS(load_json_config(js), "Shapes.MeshSurf row 1: node index 5.000000");
    js["Shapes"]["MeshSurf"] = J("[[1,3,2,3]]");
    CHECK_THROWS(load_json_config(js), "Shapes.MeshSurf row 1: label 3.000000");
    js["Shapes"]["MeshSurf"] = J("[[1,3,2,0]]");
    CHECK_THROWS(load_json_config(js), "no surface triangles");
}

TEST(load_json_file_and_patch) {
    CHECK_THROWS(load_json_file(tmpfile_path("does_not_exist.json").c_str()), "cannot open input file");
    std::string bad = tmpfile_path("bad.json");
    write_text(bad, "{\"a\": 1,, }");
    CHECK_THROWS(load_json_file(bad.c_str()), "JSON syntax error in");
    CHECK_THROWS(load_json_file(bad.c_str()), "bad.json");

    std::string good = tmpfile_path("good.json");
    write_text(good, R"({"a": 1, "b": {"c": 2, "d": 3}, "arr": [1, 2, 3]})");
    json j = load_json_file(good.c_str());
    CHECK(j["a"] == 1 && j["b"]["d"] == 3);

    j = load_json_file(good.c_str(), R"({"a": 5, "b": {"c": null, "e": 4}, "arr": [9], "new": "x"})");
    CHECK(j["a"] == 5);
    CHECK(!j["b"].contains("c"));     /* null removes */
    CHECK(j["b"]["d"] == 3);          /* untouched members kept */
    CHECK(j["b"]["e"] == 4);
    CHECK(j["arr"] == J("[9]"));      /* arrays are replaced, not merged */
    CHECK(j["new"] == "x");
    CHECK_THROWS(load_json_file(good.c_str(), "{oops"), "JSON syntax error in input or -j");

    /* load_json_input = file + config */
    std::string in = tmpfile_path("in.json");
    write_text(in, base_csg().dump());
    SimConfig c = load_json_input(in.c_str());
    CHECK(c.is_csg && c.nphoton == 1000 && c.session_id == "t");
}

// ============================================================================
// Mesh surface extraction
// ============================================================================
TEST(extract_surface_single_region_cube) {
    std::vector<Vec3> n = cube_nodes();
    std::vector<int32_t> e = cube_elems(1);    /* all region 1 */
    std::vector<std::array<uint32_t, 3> > f;
    std::vector<FaceData> d;
    extract_surface_from_tet(n, e.data(), 6, f, d);
    CHECK(f.size() == 12 && d.size() == 12);
    Vec3 cc = {0.5f, 0.5f, 0.5f};
    bool outward = true, labels = true, unit = true, axis = true;

    for (size_t i = 0; i < f.size(); i++) {
        Vec3 N = {d[i].nx, d[i].ny, d[i].nz};
        outward &= dot(N, sub(centroid(n, f[i]), cc)) > 0;
        unit &= std::fabs(v3len(N) - 1) < 1e-6;
        /* exterior normals of a cube are axis-aligned */
        axis &= (std::fabs(std::fabs(N.x) + std::fabs(N.y) + std::fabs(N.z) - 1) < 1e-6);
        uint32_t fr, bk;
        bool fl;
        unpack_media(d[i].packed_media, fr, bk, fl);
        labels &= (fr == 0 && bk == 1 && !fl);
    }

    CHECK(outward);
    CHECK(unit);
    CHECK(axis);
    CHECK(labels);
    MeshCheck mc = check_mesh(n, f, d);
    CHECK(mc.watertight && mc.oriented && mc.bad_normals == 0);
    CHECK_CLOSE(mc.volume, 1.0, 1e-6);
}

TEST(extract_surface_two_regions) {
    std::vector<Vec3> n = cube_nodes();
    std::vector<int32_t> e = cube_elems(2);
    std::vector<std::array<uint32_t, 3> > f;
    std::vector<FaceData> d;
    extract_surface_from_tet(n, e.data(), 6, f, d);
    CHECK(f.size() == 14);
    Vec3 cc = {0.5f, 0.5f, 0.5f};
    size_t nint = 0, next = 0;
    bool ext_ok = true, int_ok = true;

    for (size_t i = 0; i < f.size(); i++) {
        uint32_t fr, bk;
        bool fl;
        unpack_media(d[i].packed_media, fr, bk, fl);
        Vec3 N = {d[i].nx, d[i].ny, d[i].nz};
        Vec3 c = centroid(n, f[i]);

        if (fr == 0) {
            next++;
            /* the exterior label is the region of the tet on the inside */
            uint32_t expect = (c.x > c.y) ? 1 : 2;
            ext_ok &= (bk == expect) && dot(N, sub(c, cc)) > 0;
        } else {
            nint++;
            /* interface on the plane x=y; normal points from back toward front */
            int_ok &= std::fabs(c.x - c.y) < 1e-6 && (fr == 1 || fr == 2) && (bk == 1 || bk == 2) && fr != bk;
            Vec3 toward_front = (fr == 1) ? Vec3{1, 0, 0.5f} : Vec3{0, 1, 0.5f};   /* a point in region fr */
            int_ok &= dot(N, sub(toward_front, c)) > 0;
            int_ok &= std::fabs(std::fabs(N.x) - std::sqrt(0.5)) < 1e-5 && std::fabs(N.z) < 1e-6;
        }
    }

    CHECK(next == 12);
    CHECK(nint == 2);
    CHECK(ext_ok);
    CHECK(int_ok);
}

TEST(extract_surface_errors_and_degenerate) {
    /* three tets on one triangle (1,2,3) */
    std::vector<Vec3> n = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, -1}, {1, 1, 1}};
    int32_t e3[15] = {1, 2, 3, 4, 1, 1, 2, 3, 5, 1, 1, 3, 2, 6, 1};
    std::vector<std::array<uint32_t, 3> > f;
    std::vector<FaceData> d;
    CHECK_THROWS(extract_surface_from_tet(n, e3, 3, f, d), "a triangle is shared by more than two tetrahedra (element 3)");

    /* tet with three collinear nodes: that face has zero area and is skipped */
    std::vector<Vec3> c = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}};
    int32_t ec[5] = {1, 2, 3, 4, 1};
    extract_surface_from_tet(c, ec, 1, f, d);
    CHECK(f.size() == 3);

    /* a single regular tet: 4 outward faces */
    int32_t e1[5] = {1, 2, 3, 4, 2};
    extract_surface_from_tet(n, e1, 1, f, d);
    CHECK(f.size() == 4);
    Vec3 tc = {0.25f, 0.25f, 0.25f};
    bool out = true;

    for (size_t i = 0; i < f.size(); i++) {
        out &= dot(Vec3{d[i].nx, d[i].ny, d[i].nz}, sub(centroid(n, f[i]), tc)) > 0;
    }

    CHECK(out);
    /* listing order (handedness) of the tet nodes does not matter */
    int32_t e1r[5] = {1, 3, 2, 4, 2};
    extract_surface_from_tet(n, e1r, 1, f, d);
    out = true;

    for (size_t i = 0; i < f.size(); i++) {
        out &= dot(Vec3{d[i].nx, d[i].ny, d[i].nz}, sub(centroid(n, f[i]), tc)) > 0;
    }

    CHECK(f.size() == 4 && out);
}

TEST(load_mesh_surf_labels) {
    std::vector<Vec3> n = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {2, 0, 0}};
    int32_t sd[] = {
        1, 2, 3, 1,       /* single: exterior, front 0 back 1 */
        1, 2, 4, 1,       /* double: interface 1|2 */
        1, 4, 2, 2,
        2, 3, 4, 1,       /* double with the same label: dropped */
        4, 3, 2, 1,
        1, 2, 5, 1        /* collinear: degenerate, skipped */
    };
    std::vector<std::array<uint32_t, 3> > f;
    std::vector<FaceData> d;
    load_mesh_surf(n, sd, 6, f, d);
    CHECK(f.size() == 2);
    bool found_ext = false, found_int = false;

    for (size_t i = 0; i < f.size(); i++) {
        uint32_t fr, bk;
        bool fl;
        unpack_media(d[i].packed_media, fr, bk, fl);
        std::array<uint32_t, 3> k = f[i];
        std::sort(k.begin(), k.end());

        if (k[0] == 0 && k[1] == 1 && k[2] == 2) {
            found_ext = (fr == 0 && bk == 1);
            /* winding (1,2,3) as listed: normal +z */
            CHECK_CLOSE(d[i].nz, 1, 1e-6);
            CHECK(f[i][0] == 0 && f[i][1] == 1 && f[i][2] == 2);
        } else if (k[0] == 0 && k[1] == 1 && k[2] == 3) {
            found_int = (fr == 2 && bk == 1);
            /* winding of the first listing (1,2,4): normal -y */
            CHECK_CLOSE(d[i].ny, -1, 1e-6);
        }
    }

    CHECK(found_ext);
    CHECK(found_int);
}

TEST(load_table_and_elems_direct) {
    size_t r = 0, c = 0;
    std::vector<double> t = load_table(J("[[1,2,3],[4,5,6]]"), 3, r, c, "T");
    CHECK(r == 2 && c == 3 && t.size() == 6 && t[4] == 5);
    CHECK_THROWS(load_table(J("[1,2,3]"), 1, r, c, "T"), "T row 1 must have");
    CHECK_THROWS(load_table(J("[[1,2]]"), 3, r, c, "T"), "T needs at least 3 columns");
    double rows[6] = {1, 2, 3, 4, 5, 6};
    json cm = jdata_encode("double", {2, 3}, rows, sizeof(rows), false);
    t = load_table(cm, 3, r, c, "T");    /* column-major JData becomes row-major */
    CHECK(r == 2 && c == 3 && t[0] == 1 && t[1] == 3 && t[3] == 2);

    size_t cnt = 0;
    std::vector<int32_t> el = load_elems(J("[[1,2,3,9,1],[2,3,4,1,0]]"), 9, 4, 1, cnt, "E");
    CHECK(cnt == 2 && el.size() == 10 && el[3] == 9 && el[9] == 0);
    /* extra columns: the label is the last column */
    el = load_elems(J("[[1,2,3,7,1]]"), 9, 3, 7, cnt, "E");
    CHECK(cnt == 1 && el.size() == 4 && el[3] == 1);
}

// ============================================================================
// Shape generators
// ============================================================================
TEST(shape_box) {
    ShapeMesh m;
    gen_box(m, 1, 2, 3, 4, 5, 6, 3);
    CHECK(m.nodes.size() == 8 && m.faces.size() == 12);
    MeshCheck mc = check_mesh(m.nodes, m.faces, m.facedata);
    CHECK(mc.watertight && mc.oriented && mc.bad_normals == 0 && mc.degenerate == 0);
    CHECK_CLOSE(mc.volume, 120.0, 1e-4);
    CHECK(count_flat(m, true) == 12);
    CHECK(all_tags(m, 3));
    CHECK(flag_coplanar_faces(m) == 0);   /* the two halves of a quad only share the diagonal */

    /* a second box appended keeps indices consistent */
    gen_box(m, 10, 10, 10, 1, 1, 1, 4);
    mc = check_mesh(m.nodes, m.faces, m.facedata, 12, 24);
    CHECK(mc.watertight && mc.oriented);
    CHECK_CLOSE(mc.volume, 1.0, 1e-4);
    CHECK(all_tags(m, 4, 12, 24));
}

TEST(shape_sphere) {
    const double R = 7.5;
    double prev_err = 1e9;
    bool monotonic = true, radius_ok = true;
    int counts_ok = 1;

    for (int res : {4, 8, 16, 24, 64}) {
        std::vector<ShapeOrigin> org;
        std::string js = "[{\"Sphere\": {\"O\": [30, 31, 32], \"R\": 7.5, \"Tag\": 2}}]";
        ShapeMesh m = shapes(js.c_str(), res, 64, &org);
        int nlon = res, nlat = std::max(res * 2 / 3, 8);
        counts_ok &= (m.faces.size() == (size_t)(2 * nlon * (nlat - 1)));
        counts_ok &= (m.nodes.size() == (size_t)(nlon * (nlat - 1) + 2));
        MeshCheck mc = check_mesh(m.nodes, m.faces, m.facedata);
        CHECK(mc.watertight && mc.oriented && mc.bad_normals == 0 && mc.degenerate == 0);
        double exact = 4.0 / 3.0 * M_PI * R * R * R;
        double err = (exact - mc.volume) / exact;
        CHECK(err > 0);       /* inscribed polyhedron */
        monotonic &= (err < prev_err);
        prev_err = err;

        for (size_t i = 0; i < m.nodes.size(); i++) {
            Vec3 d = sub(m.nodes[i], Vec3{30, 31, 32});
            radius_ok &= std::fabs(v3len(d) - R) < 1e-4 * R;
        }

        CHECK(count_flat(m, false) == m.faces.size());
        CHECK(all_tags(m, 2));
        CHECK(org.size() == 1 && org[0].type == 1 && org[0].R == 7.5f && org[0].cy == 31);
    }

    CHECK(counts_ok);
    CHECK(monotonic);
    CHECK(prev_err < 0.005);    /* res 64 */
    CHECK(radius_ok);
}

static void check_cylinder(const char* label, Vec3 c0, Vec3 c1, float R, int res) {
    ShapeMesh m;
    gen_cylinder(m, c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, R, 5, res);
    CHECK(m.faces.size() == (size_t)(4 * res));
    CHECK(m.nodes.size() == (size_t)(2 * res + 2));
    MeshCheck mc = check_mesh(m.nodes, m.faces, m.facedata);
    CHECK(mc.watertight && mc.oriented && mc.bad_normals == 0 && mc.degenerate == 0);
    double L = v3len(sub(c1, c0));
    double prism = 0.5 * res * R * R * std::sin(2 * M_PI / res) * L;
    CHECK_CLOSE(mc.volume, prism, 1e-4 * prism);
    double exact = M_PI * R * R * L;
    CHECK(mc.volume < exact && (exact - mc.volume) / exact < 2 * M_PI * M_PI / (3.0 * res * res) + 1e-6);
    CHECK(count_flat(m, true) == (size_t)(2 * res));    /* caps */
    CHECK(count_flat(m, false) == (size_t)(2 * res));   /* side */
    CHECK(all_tags(m, 5));

    /* side normals are perpendicular to the axis, caps are along it */
    Vec3 ax = normalized(sub(c1, c0));
    bool side_ok = true, cap_ok = true;

    for (size_t i = 0; i < m.faces.size(); i++) {
        Vec3 N = {m.facedata[i].nx, m.facedata[i].ny, m.facedata[i].nz};

        if (pm_bits(m.facedata[i].packed_media) & VKMMC_FACE_FLAT) {
            cap_ok &= std::fabs(std::fabs(dot(N, ax)) - 1) < 1e-4;
        } else {
            side_ok &= std::fabs(dot(N, ax)) < 1e-4;
        }
    }

    CHECK(side_ok);
    CHECK(cap_ok);

    /* ring nodes are at distance R from the axis */
    bool rad = true;

    for (int i = 0; i < 2 * res; i++) {
        Vec3 d = sub(m.nodes[i], c0);
        double p = dot(d, ax);
        Vec3 r = {(float)(d.x - p * ax.x), (float)(d.y - p * ax.y), (float)(d.z - p * ax.z)};
        rad &= std::fabs(v3len(r) - R) < 1e-4 * R;
    }

    if (!rad) {
        fprintf(g_out, "  (cylinder %s)\n", label);
    }

    CHECK(rad);
}

TEST(shape_cylinder) {
    check_cylinder("z", {30, 30, 10}, {30, 30, 50}, 5, 24);
    check_cylinder("x", {10, 30, 30}, {50, 30, 30}, 4, 16);
    check_cylinder("y", {30, 50, 30}, {30, 10, 30}, 3, 8);
    check_cylinder("oblique", {10, 12, 14}, {40, 35, 30}, 6, 32);
    check_cylinder("coarse", {0, 0, 0}, {0, 0, 1}, 1, 4);

    ShapeMesh m;
    gen_cylinder(m, 1, 1, 1, 1, 1, 1, 2, 1, 16);    /* zero length: nothing */
    CHECK(m.faces.empty() && m.nodes.empty());

    /* the tessellation error shrinks with the resolution */
    double e_prev = 1e9;
    bool mono = true;

    for (int res : {8, 16, 32, 64}) {
        std::string js = "[{\"Cylinder\": {\"C0\": [30,30,10], \"C1\": [30,30,50], \"R\": 5, \"Tag\": 1}}]";
        ShapeMesh s = shapes(js.c_str(), res);
        double exact = M_PI * 25 * 40;
        double err = std::fabs(check_mesh(s.nodes, s.faces, s.facedata).volume - exact) / exact;
        mono &= err < e_prev;
        e_prev = err;
    }

    CHECK(mono);
    CHECK(e_prev < 0.002);
}

TEST(shape_slabs_and_layers) {
    float ext[6] = {0, 60, 0, 40, 0, 20};
    ShapeMesh m;
    gen_slab(m, 0, 10, 15, ext, 1);
    gen_slab(m, 1, 5, 25, ext, 2);
    gen_slab(m, 2, 2, 4, ext, 3);
    CHECK(m.faces.size() == 36);
    CHECK_CLOSE(check_mesh(m.nodes, m.faces, m.facedata, 0, 12).volume, 5 * 40 * 20, 1e-2);
    CHECK_CLOSE(check_mesh(m.nodes, m.faces, m.facedata, 12, 24).volume, 60 * 20 * 20, 1e-2);
    CHECK_CLOSE(check_mesh(m.nodes, m.faces, m.facedata, 24, 36).volume, 60 * 40 * 2, 1e-2);
    CHECK(count_flat(m, true) == 36);
    CHECK(all_tags(m, 2, 12, 24));

    /* single [lo, hi] pair, against the default extent [0,60]^3 */
    ShapeMesh s = shapes(R"([{"XSlabs": {"Bound": [10, 20], "Tag": 1}}])");
    CHECK(s.faces.size() == 12);
    MeshCheck mc = check_mesh(s.nodes, s.faces, s.facedata);
    CHECK(mc.watertight && mc.oriented && mc.bad_normals == 0);
    CHECK_CLOSE(mc.volume, 10 * 60 * 60, 1e-1);
    bool xr = true;

    for (size_t i = 0; i < s.nodes.size(); i++) {
        xr &= (s.nodes[i].x == 10 || s.nodes[i].x == 20) && (s.nodes[i].y == 0 || s.nodes[i].y == 60);
    }

    CHECK(xr);
    s = shapes(R"([{"YSlabs": {"Bound": [[1, 2], [5, 9]], "Tag": 2}}])");
    CHECK(s.faces.size() == 24);
    CHECK_CLOSE(check_mesh(s.nodes, s.faces, s.facedata).volume, 5 * 3600, 1e-1);

    /* Grid sets the extent of later slabs/layers */
    s = shapes(R"([{"Grid": {"Size": [10, 20, 30], "Tag": 1}}, {"ZSlabs": {"Bound": [5, 6], "Tag": 2}}])");
    CHECK_CLOSE(check_mesh(s.nodes, s.faces, s.facedata, 12, 24).volume, 10 * 20 * 1, 1e-3);

    /* MCX layers are 1-based and inclusive: [5,10] spans z in [4,10] */
    s = shapes(R"([{"ZLayers": [5, 10, 3]}])");
    CHECK(s.faces.size() == 12);
    float zmin = 1e9f, zmax = -1e9f;

    for (size_t i = 0; i < s.nodes.size(); i++) {
        zmin = std::min(zmin, s.nodes[i].z);
        zmax = std::max(zmax, s.nodes[i].z);
    }

    CHECK(zmin == 4 && zmax == 10);
    CHECK(all_tags(s, 3));
    s = shapes(R"([{"XLayers": [[1, 10, 1], [11, 20, 2], [21, 21, 3]]}])");
    CHECK(s.faces.size() == 36);
    CHECK(all_tags(s, 2, 12, 24));
    CHECK_CLOSE(check_mesh(s.nodes, s.faces, s.facedata, 24, 36).volume, 3600, 1e-2);
    s = shapes(R"([{"YLayers": [[1, 30, 1]]}])");
    CHECK_CLOSE(check_mesh(s.nodes, s.faces, s.facedata).volume, 30 * 3600, 1);
}

struct LensCase {
    const char* name;
    double R, Df, Rf, Db, Rb;
    Vec3 dir;
};

static void check_lens(const LensCase& lc, int res) {
    char js[512];
    snprintf(js, sizeof(js), "[{\"Lens\": {\"O\": [30, 30, 30], \"Dir\": [%g, %g, %g], \"R\": %g, \"Tag\": 1, "
             "\"Front\": {\"D\": %g, \"R\": %g}, \"Back\": {\"D\": %g, \"R\": %g}}}]",
             lc.dir.x, lc.dir.y, lc.dir.z, lc.R, lc.Df, lc.Rf, lc.Db, lc.Rb);
    std::vector<ShapeOrigin> org;
    ShapeMesh m = shapes(js, res, 64, &org);
    MeshCheck mc = check_mesh(m.nodes, m.faces, m.facedata);
    Vec3 n = normalized(lc.dir);
    double exact = lens_volume(lc.R, lc.Df, lc.Rf, lc.Db, lc.Rb);
    bool tri_ok = mc.watertight && mc.oriented && mc.bad_normals == 0;
    bool vol_ok = std::fabs(mc.volume - exact) < 0.02 * exact;

    /* front cap normals point along +Dir, back cap along -Dir, rim radial */
    size_t bad_front = 0, bad_back = 0, bad_rim = 0, nflat = 0;
    bool ids_ok = m.shape_id.size() == m.faces.size();

    for (size_t i = 0; i < m.faces.size(); i++) {
        Vec3 N = {m.facedata[i].nx, m.facedata[i].ny, m.facedata[i].nz};
        bool flat = (pm_bits(m.facedata[i].packed_media) & VKMMC_FACE_FLAT) != 0;
        uint32_t sid = i < m.shape_id.size() ? m.shape_id[i] : 0;
        ids_ok &= (sid == 1 || sid == 2) && !(flat && sid != 1);

        if (flat) {
            nflat++;
            bad_rim += (std::fabs(dot(N, n)) > 0.5);
        } else if (sid == 1) {
            bad_front += (dot(N, n) <= 0);
        } else {
            bad_back += (dot(N, n) >= 0);
        }
    }

    if (!(tri_ok && vol_ok && bad_front == 0 && bad_back == 0)) {
        fprintf(g_out, "  lens '%s' res=%d: watertight=%d oriented=%d bad_normals=%zu volume=%.5g (exact %.5g) "
                "inward front/back faces=%zu/%zu\n", lc.name, res, mc.watertight, mc.oriented, mc.bad_normals,
                mc.volume, exact, bad_front, bad_back);
    }

    CHECK(mc.watertight);
    CHECK(mc.oriented);
    CHECK(mc.bad_normals == 0);
    CHECK(vol_ok);
    CHECK(bad_front == 0);
    CHECK(bad_back == 0);
    CHECK(bad_rim == 0);
    CHECK(nflat == (size_t)(2 * res));    /* the rim band */
    CHECK(ids_ok);
    CHECK(all_tags(m, 1));
    CHECK(org.size() == 2 && org[0].type == 3 && org[1].type == 3);
    CHECK(org.size() == 2 && org[0].R == (float)lc.Rf && org[1].R == (float)lc.Rb);
    /* front sphere center O + n*(Df-Rf), back O + n*(Rb-Db) */
    CHECK(org.size() == 2 && std::fabs(org[0].cz - (30 + n.z * (lc.Df - lc.Rf))) < 1e-3);
    CHECK(org.size() == 2 && std::fabs(org[1].cx - (30 + n.x * (lc.Rb - lc.Db))) < 1e-3);
}

TEST(shape_lens) {
    LensCase convex = {"biconvex", 5, 1, 20, 1.5, 15, {0, 0, 1}};
    LensCase oblique = {"biconvex oblique", 5, 1, 20, 1.5, 15, {1, 1, 0}};
    LensCase flatish = {"near-flat", 5, 1, 1e4, 1, -1e4, {0, 1, 0}};
    LensCase planoconcave = {"plano-concave (mild)", 5, 1, -40, 1, 1e5, {0, 0, 1}};
    check_lens(convex, 32);
    check_lens(convex, 64);
    check_lens(oblique, 48);
    check_lens(flatish, 32);
    check_lens(planoconcave, 32);

    /* Regression (fixed): gen_lens orients each triangle by add_tri_outward() against the
       lens origin O, which is wrong for a strongly concave cap: near the rim
       the outward normal of a concave cap tilts toward the axis and
       dot(n, centroid - O) < 0, so those triangles are flipped inward. A
       biconcave lens R=5, Front/Back {D:1, R:-10} gives an inconsistently
       oriented (non-manifold-oriented) surface with inward rim triangles. */
    LensCase concave = {"biconcave", 5, 1, -10, 1, -10, {0, 0, 1}};
    check_lens(concave, 32);

    /* nrings == 1, nseg < 3 and zero direction branches of gen_lens */
    ShapeMesh m;
    gen_lens(m, 0, 0, 0, 0, 0, 1, 5, 1, 20, 1, 20, 1, 1, 2, 8, 1);
    MeshCheck mc = check_mesh(m.nodes, m.faces, m.facedata);
    CHECK(m.faces.size() == 8 * 4 && mc.watertight && mc.oriented && mc.bad_normals == 0);
    CHECK(m.shape_id.size() == m.faces.size());
    ShapeMesh e;
    gen_lens(e, 0, 0, 0, 0, 0, 1, 5, 1, 20, 1, 20, 1, 1, 2, 2, 4);
    gen_lens(e, 0, 0, 0, 0, 0, 0, 5, 1, 20, 1, 20, 1, 1, 2, 8, 4);
    gen_lens(e, 0, 0, 0, 0, 0, 1, 1e-7f, 1, 20, 1, 20, 1, 1, 2, 8, 4);
    CHECK(e.faces.empty() && e.nodes.empty());

    /* biconvex lens whose spheres intersect inside R: the rim is clamped to
       the intersection circle; the surface stays closed */
    ShapeMesh x;
    gen_lens(x, 0, 0, 0, 0, 0, 1, 5, 2, 4, 2, 4, 1, 1, 2, 16, 6);
    mc = check_mesh(x.nodes, x.faces, x.facedata);
    CHECK(mc.watertight && mc.bad_normals == 0);
    /* the rim band collapses to zero-area triangles (all flat-flagged) */
    CHECK(mc.degenerate == 32 && count_flat(x, true) == 32);
    CHECK_CLOSE(mc.volume, lens_volume(5, 2, 4, 2, 4), 0.05 * lens_volume(5, 2, 4, 2, 4));
}

TEST(parse_shapes_ids_and_offsets) {
    std::vector<ShapeOrigin> org;
    ShapeMesh m = shapes(R"([{"Name": "phantom"},
        {"Grid": {"Size": [60, 60, 60], "Tag": 1}},
        {"Sphere": {"O": [30, 30, 30], "R": 5, "Tag": 2}},
        {"Box": {"O": [5, 5, 5], "Size": [4, 4, 4], "Tag": 3}},
        {"Cylinder": {"C0": [10, 40, 10], "C1": [10, 40, 50], "R": 3, "Tag": 2}},
        {"Lens": {"O": [45, 45, 30], "Dir": [0, 0, 1], "R": 4, "Tag": 3,
                  "Front": {"D": 1, "R": 20}, "Back": {"D": 1, "R": 20}}},
        {"Sphere": {"O": [45, 15, 30], "R": 3, "Tag": 1}}])", 16, 3, &org);
    /* Name 1, Grid 2, Sphere 3, Box 4, Cylinder 5, Lens 6+7, Sphere 8 */
    CHECK(org.size() == 8);
    CHECK(m.shape_id.size() == m.faces.size());
    uint32_t mx = 0;
    std::set<uint32_t> ids;

    for (size_t i = 0; i < m.shape_id.size(); i++) {
        mx = std::max(mx, m.shape_id[i]);
        ids.insert(m.shape_id[i]);
    }

    CHECK(mx == org.size());
    CHECK(ids == std::set<uint32_t>({2, 3, 4, 5, 6, 7, 8}));
    int types[8] = {0, 0, 1, 0, 2, 3, 3, 1};
    bool tok = true;

    for (size_t i = 0; i < org.size() && i < 8; i++) {
        tok &= org[i].type == types[i];
    }

    CHECK(tok);
    CHECK(m.shape_id[0] == 2 && m.shape_id[11] == 2 && m.shape_id[12] == 3);
    CHECK(extract_shape_origins(J(R"([{"Grid": {"Size": [60, 60, 60], "Tag": 1}},
        {"Lens": {"O": [45, 45, 30], "Dir": [0, 0, 1], "R": 4, "Tag": 1, "Front": {"D": 1, "R": 20}, "Back": {"D": 1, "R": 20}}},
        {"Cylinder": {"C0": [10, 40, 10], "C1": [10, 40, 50], "R": 3}}])"), 16).size() == 4);

    /* the cylinder origin record */
    CHECK(org[4].cx == 10 && org[4].cy == 40 && org[4].cz == 10);
    CHECK(org[4].az == 1 && org[4].length == 40 && org[4].R == 3);

    /* Origin is subtracted from the positions of later shapes */
    std::vector<ShapeOrigin> o2;
    ShapeMesh s = shapes(R"([{"Origin": [10, 20, 30]}, {"Sphere": {"O": [40, 40, 40], "R": 5}},
                             {"Box": {"O": [10, 20, 30], "Size": [1, 1, 1]}},
                             {"Cylinder": {"C0": [10, 20, 30], "C1": [10, 20, 40], "R": 1}}])", 16, 64, &o2);
    CHECK(o2.size() == 4);
    CHECK(o2[1].cx == 30 && o2[1].cy == 20 && o2[1].cz == 10);
    CHECK(o2[3].cx == 0 && o2[3].cy == 0 && o2[3].cz == 0 && o2[3].length == 10);
    float bmin = 1e9f;
    size_t nsphere = 2 * 16 * (std::max(16 * 2 / 3, 8) - 1);

    for (size_t i = 0; i < s.faces.size(); i++) {
        if (i >= nsphere && i < nsphere + 12) {
            for (int k = 0; k < 3; k++) {
                const Vec3& v = s.nodes[s.faces[i][k]];
                bmin = std::min(bmin, std::min(v.x, std::min(v.y, v.z)));
            }
        }
    }

    CHECK(bmin == 0);   /* box at [10,20,30] - Origin = [0,0,0] */
    CHECK(s.shape_id[0] == 2);

    /* Subgrid uses MCX 1-based voxel origins */
    ShapeMesh g = shapes(R"([{"Subgrid": {"O": [1, 1, 1], "Size": [10, 10, 10], "Tag": 1}}])");
    float lo = 1e9f, hi = -1e9f;

    for (size_t i = 0; i < g.nodes.size(); i++) {
        lo = std::min(lo, std::min(g.nodes[i].x, std::min(g.nodes[i].y, g.nodes[i].z)));
        hi = std::max(hi, std::max(g.nodes[i].x, std::max(g.nodes[i].y, g.nodes[i].z)));
    }

    CHECK(lo == 0 && hi == 10);

    /* Grid writes the extent */
    float ext[6] = {0, 60, 0, 60, 0, 60};
    parse_shapes(J(R"([{"Grid": {"Size": [11, 12, 13]}}])"), ext, 24, 64, NULL);
    CHECK(ext[1] == 11 && ext[3] == 12 && ext[5] == 13 && ext[0] == 0);

    /* flat shapes accept any mesh resolution >= 1 */
    CHECK_NOTHROW(shapes(R"([{"Box": {"O": [0, 0, 0], "Size": [1, 1, 1]}}, {"ZLayers": [1, 2, 1]}])", 1));
}

#define CHECK_SHAPES_THROWS(js, res, maxtag, substr) CHECK_THROWS(shapes(js, res, maxtag), substr)

TEST(parse_shapes_errors) {
    float ext[6] = {0, 60, 0, 60, 0, 60};
    CHECK_THROWS(parse_shapes(J("{}"), ext, 24, 64, NULL), "Shapes must be an array");
    CHECK_SHAPES_THROWS(R"([{"Box": {"O": [0,0,0], "Size": [1,1,1]}}])", 0, 64, "mesh resolution must be in [4, 2048]");
    CHECK_SHAPES_THROWS(R"([{"Box": {"O": [0,0,0], "Size": [1,1,1]}}])", 4096, 64, "mesh resolution must be in");
    CHECK_SHAPES_THROWS(R"([{"Torus": {"R": 1}}])", 24, 64, "Shapes[0].Torus: unknown shape type");
    CHECK_SHAPES_THROWS(R"([5])", 24, 64, "Shapes[0] must be an object with a single key");
    CHECK_SHAPES_THROWS(R"([{"Box": {"O": [0,0,0], "Size": [1,1,1]}, "Sphere": {}}])", 24, 64, "with a single key");
    CHECK_SHAPES_THROWS(R"([{"Name": "x"}, {}])", 24, 64, "Shapes[1] must be an object with a single key");
    CHECK_SHAPES_THROWS(R"([{"Sphere": 5}])", 24, 64, "Shapes[0].Sphere must be an object");
    CHECK_SHAPES_THROWS(R"([{"Box": [0, 0, 0]}])", 24, 64, "Shapes[0].Box must be an object");
    CHECK_SHAPES_THROWS(R"([{"Sphere": {"O": [1,1,1], "R": 1, "Tag": 0}}])", 24, 64, "Shapes[0].Sphere.Tag=0.000000 must be an integer in [1, 64]");
    CHECK_SHAPES_THROWS(R"([{"Sphere": {"O": [1,1,1], "R": 1, "Tag": 1.5}}])", 24, 64, "must be an integer in [1, 64]");
    CHECK_SHAPES_THROWS(R"([{"Sphere": {"O": [1,1,1], "R": 1, "Tag": 3}}])", 24, 2, "must be an integer in [1, 2]");
    CHECK_SHAPES_THROWS(R"([{"Sphere": {"O": [1,1,1], "R": 1, "Tag": 65}}])", 24, 1000, "must be an integer in [1, 64]");
    CHECK_SHAPES_THROWS(R"([{"Sphere": {"O": [1,1,1], "R": 1, "Tag": "a"}}])", 24, 64, "Tag must be a number");
    CHECK_SHAPES_THROWS(R"([{"Sphere": {"O": [1,1,1], "R": 0}}])", 24, 64, "Shapes[0].Sphere.R must be positive");
    CHECK_SHAPES_THROWS(R"([{"Sphere": {"O": [1,1,1], "R": -2}}])", 24, 64, "R must be positive");
    CHECK_SHAPES_THROWS(R"([{"Sphere": {"O": [1,1,1]}}])", 24, 64, "Shapes[0].Sphere requires 'R'");
    CHECK_SHAPES_THROWS(R"([{"Sphere": {"R": 1}}])", 24, 64, "Shapes[0].Sphere requires 'O'");
    CHECK_SHAPES_THROWS(R"([{"Sphere": {"O": [1,1], "R": 1}}])", 24, 64, "Shapes[0].Sphere.O needs at least 3 numbers");
    CHECK_SHAPES_THROWS(R"([{"Sphere": {"O": [1,1,1], "R": 1}}])", 3, 64, "the mesh resolution (MeshRes/-m) must be at least 4");
    CHECK_SHAPES_THROWS(R"([{"Cylinder": {"C0": [1,1,1], "C1": [1,1,1], "R": 1}}])", 24, 64, "C0 and C1 must be distinct");
    CHECK_SHAPES_THROWS(R"([{"Cylinder": {"C0": [1,1,1], "C1": [1,1,5], "R": 1}}])", 2, 64, "must be at least 4 for curved shapes");
    CHECK_SHAPES_THROWS(R"([{"Cylinder": {"C0": [1,1,1], "C1": [1,1,5]}}])", 24, 64, "Cylinder requires 'R'");
    CHECK_SHAPES_THROWS(R"([{"Box": {"O": [0,0,0], "Size": [1,0,1]}}])", 24, 64, "Shapes[0].Box.Size must be positive");
    CHECK_SHAPES_THROWS(R"([{"Grid": {"Size": [10,10,-1]}}])", 24, 64, "Shapes[0].Grid.Size must be positive");
    CHECK_SHAPES_THROWS(R"([{"Subgrid": {"O": [1,1,1]}}])", 24, 64, "Subgrid requires 'Size'");
    CHECK_SHAPES_THROWS(R"([{"ZSlabs": {"Tag": 1}}])", 24, 64, "Shapes[0].ZSlabs requires a 'Bound' array");
    CHECK_SHAPES_THROWS(R"([{"ZSlabs": {"Bound": [5, 5]}}])", 24, 64, "Bound: need lower < upper bound");
    CHECK_SHAPES_THROWS(R"([{"XSlabs": {"Bound": [[1, 2], [3]]}}])", 24, 64, "Shapes[0].XSlabs.Bound[1] needs at least 2 numbers");
    CHECK_SHAPES_THROWS(R"([{"ZLayers": {"a": 1}}])", 24, 64, "Shapes[0].ZLayers must be an array of [start, end, tag] layers");
    CHECK_SHAPES_THROWS(R"([{"ZLayers": [[5, 4, 1]]}])", 24, 64, "Shapes[0].ZLayers[0]: need start <= end");
    CHECK_SHAPES_THROWS(R"([{"ZLayers": [[1, 4, 0]]}])", 24, 64, "Shapes[0].ZLayers[0].Tag=0.000000 must be an integer");
    CHECK_SHAPES_THROWS(R"([{"YLayers": [[1, 4]]}])", 24, 64, "Shapes[0].YLayers[0] needs at least 3 numbers");
    CHECK_SHAPES_THROWS(R"([{"Origin": [1, 2]}])", 24, 64, "Shapes[0].Origin needs at least 3 numbers");
    CHECK_SHAPES_THROWS(R"([{"Origin": 1}])", 24, 64, "Shapes[0].Origin must be an array");
    const char* lens_ok = R"({"O": [0,0,0], "Dir": [0,0,1], "R": 2, "Front": {"D": 1, "R": 5}, "Back": {"D": 1, "R": 5}})";
    json L = J(lens_ok);
    auto lens_with = [&](const char* patch) -> std::string {
        json s = json::array();
        s.push_back({{"Lens", patched(L, patch)}});
        return s.dump();
    };
    CHECK_NOTHROW(shapes(lens_with("{}").c_str()));
    CHECK_THROWS(shapes(lens_with(R"({"Front": {"D": 0}, "Back": {"D": 0}})").c_str()),
                 "the lens thickness Front.D + Back.D must be positive");
    CHECK_THROWS(shapes(lens_with(R"({"Front": {"D": -2}})").c_str()), "lens thickness");
    CHECK_THROWS(shapes(lens_with(R"({"Front": {"R": 0}})").c_str()), "Front.R and Back.R must be non-zero");
    CHECK_THROWS(shapes(lens_with(R"({"Back": {"R": null}})").c_str()), "must be non-zero");
    CHECK_THROWS(shapes(lens_with(R"({"Dir": [0, 0, 0]})").c_str()), "Shapes[0].Lens.Dir must be a non-zero vector");
    CHECK_THROWS(shapes(lens_with(R"({"Dir": null})").c_str()), "Shapes[0].Lens requires 'Dir'");
    CHECK_THROWS(shapes(lens_with(R"({"Front": null})").c_str()), "requires a 'Front' object {D, R}");
    CHECK_THROWS(shapes(lens_with(R"({"Back": 1})").c_str()), "requires a 'Back' object {D, R}");
    CHECK_THROWS(shapes(lens_with(R"({"R": 0})").c_str()), "Shapes[0].Lens.R must be positive");
    CHECK_THROWS(shapes(lens_with(R"({"R": 1e-7})").c_str()), "the lens parameters give an empty surface");
    CHECK_THROWS(shapes(lens_with("{}").c_str(), 3), "must be at least 4 for curved shapes");
    CHECK_THROWS(shapes(lens_with(R"({"Tag": 9})").c_str(), 24, 4), "Shapes[0].Lens.Tag=9.000000");
}

// ============================================================================
// Coplanar overlap flagging
// ============================================================================
static bool face_on_plane(const ShapeMesh& m, size_t i, int axis, float v) {
    for (int k = 0; k < 3; k++) {
        if ((&m.nodes[m.faces[i][k]].x)[axis] != v) {
            return false;
        }
    }

    return true;
}

static bool face_coplanar(const ShapeMesh& m, size_t i) {
    return (pm_bits(m.facedata[i].packed_media) & VKMMC_FACE_COPLANAR) != 0;
}

TEST(coplanar_layers_in_grid) {
    ShapeMesh m = shapes(R"([{"Grid": {"Size": [60, 60, 60], "Tag": 1}}, {"ZLayers": [[1, 10, 1], [11, 20, 2]]}])");
    size_t n10 = 0, f10 = 0, n0 = 0, f0 = 0, n60 = 0, f60 = 0, n20 = 0, f20 = 0;

    for (size_t i = 0; i < m.faces.size(); i++) {
        if (face_on_plane(m, i, 2, 10)) {
            n10++;
            f10 += face_coplanar(m, i);
        } else if (face_on_plane(m, i, 2, 0)) {
            n0++;
            f0 += face_coplanar(m, i);
        } else if (face_on_plane(m, i, 2, 60)) {
            n60++;
            f60 += face_coplanar(m, i);
        } else if (face_on_plane(m, i, 2, 20)) {
            n20++;
            f20 += face_coplanar(m, i);
        }
    }

    CHECK(n10 == 4 && f10 == 4);    /* top of layer 1 == bottom of layer 2 */
    CHECK(n0 == 4 && f0 == 4);      /* bottom of layer 1 == bottom of the grid */
    CHECK(n60 == 2 && f60 == 0);    /* grid top: alone */
    CHECK(n20 == 2 && f20 == 0);    /* layer 2 top: alone */
    /* flagging keeps the flat bit and labels */
    CHECK(count_flat(m, true) == m.faces.size());
    CHECK(all_tags(m, 1, 0, 12) && all_tags(m, 2, 24, 36));
}

TEST(coplanar_lone_and_separate) {
    CHECK(count_flat(shapes(R"([{"Box": {"O": [1, 2, 3], "Size": [4, 5, 6]}}])"), true) == 12);
    ShapeMesh b = shapes(R"([{"Box": {"O": [1, 2, 3], "Size": [4, 5, 6]}}])");
    size_t nb = 0;

    for (size_t i = 0; i < b.faces.size(); i++) {
        nb += face_coplanar(b, i);
    }

    CHECK(nb == 0);

    /* two boxes stacked with a gap: parallel planes z=10 / z=20 are distinct,
       and their coplanar side walls (x=0, ...) do not overlap */
    ShapeMesh s = shapes(R"([{"Box": {"O": [0, 0, 0], "Size": [10, 10, 10]}}, {"Box": {"O": [0, 0, 20], "Size": [10, 10, 10]}}])");
    CHECK(flag_coplanar_faces(s) == 0);

    /* boxes touching only along an edge are not overlapping either */
    ShapeMesh e = shapes(R"([{"Box": {"O": [0, 0, 0], "Size": [10, 10, 10]}}, {"Box": {"O": [10, 10, 0], "Size": [10, 10, 10]}}])");
    CHECK(flag_coplanar_faces(e) == 0);

    /* curved shapes alone: nothing flagged */
    ShapeMesh sp = shapes(R"([{"Sphere": {"O": [30, 30, 30], "R": 5}}, {"Cylinder": {"C0": [5,5,5], "C1": [5,5,20], "R": 2}}])");
    CHECK(flag_coplanar_faces(sp) == 0);
}

TEST(coplanar_box_flush_with_grid) {
    ShapeMesh m = shapes(R"([{"Grid": {"Size": [60, 60, 60], "Tag": 1}}, {"Box": {"O": [0, 10, 10], "Size": [10, 10, 10], "Tag": 2}}])");
    size_t grid_x0 = 0, box_x0 = 0, box_x10 = 0, other = 0;

    for (size_t i = 0; i < m.faces.size(); i++) {
        bool f = face_coplanar(m, i);

        if (i < 12 && face_on_plane(m, i, 0, 0)) {
            grid_x0 += f;
        } else if (i >= 12 && face_on_plane(m, i, 0, 0)) {
            box_x0 += f;
        } else if (i >= 12 && face_on_plane(m, i, 0, 10)) {
            box_x10 += f;
        } else {
            other += f;
        }
    }

    CHECK(box_x0 == 2);       /* box wall flush with the grid wall */
    CHECK(grid_x0 >= 1);      /* at least the grid triangle(s) under the box */
    CHECK(box_x10 == 0);
    CHECK(other == 0);
}

TEST(coplanar_tris_overlap_direct) {
    Vec3 A[3] = {{0, 0, 5}, {1, 0, 5}, {0, 1, 5}};
    Vec3 inside[3] = {{0.1f, 0.1f, 5}, {0.5f, 0.1f, 5}, {0.1f, 0.5f, 5}};
    Vec3 overl[3] = {{0.2f, 0.2f, 5}, {1, 0.2f, 5}, {0.2f, 1, 5}};
    Vec3 edge[3] = {{1, 0, 5}, {0, 1, 5}, {1, 1, 5}};       /* shares the hypotenuse */
    Vec3 vertex[3] = {{1, 0, 5}, {2, 0, 5}, {2, 1, 5}};     /* shares one vertex */
    Vec3 disj[3] = {{2, 2, 5}, {3, 2, 5}, {2, 3, 5}};
    Vec3 degen[3] = {{0.2f, 0.2f, 5}, {0.2f, 0.2f, 5}, {0.3f, 0.3f, 5}};
    CHECK(coplanar_tris_overlap(A, A, 0, 1, 1e-5f));
    CHECK(coplanar_tris_overlap(A, inside, 0, 1, 1e-5f));
    CHECK(coplanar_tris_overlap(inside, A, 0, 1, 1e-5f));
    CHECK(coplanar_tris_overlap(A, overl, 0, 1, 1e-5f));
    CHECK(!coplanar_tris_overlap(A, edge, 0, 1, 1e-5f));
    CHECK(!coplanar_tris_overlap(A, vertex, 0, 1, 1e-5f));
    CHECK(!coplanar_tris_overlap(A, disj, 0, 1, 1e-5f));
    CHECK(!coplanar_tris_overlap(disj, A, 0, 1, 1e-5f));
    /* a degenerate sliver inside A: edges of zero length are skipped */
    CHECK(coplanar_tris_overlap(A, degen, 0, 1, 1e-5f) == coplanar_tris_overlap(degen, A, 0, 1, 1e-5f));
    /* projections onto other axis pairs (plane x = 5) */
    Vec3 Px[3] = {{5, 0, 0}, {5, 1, 0}, {5, 0, 1}};
    Vec3 Qx[3] = {{5, 0.2f, 0.2f}, {5, 0.6f, 0.2f}, {5, 0.2f, 0.6f}};
    CHECK(coplanar_tris_overlap(Px, Qx, 1, 2, 1e-5f));
    CHECK(!coplanar_tris_overlap(Px, Qx, 0, 1, 1e-5f));   /* both collapse to lines */
}

// ============================================================================
// Curvature
// ============================================================================
TEST(curvature_sphere_nodes) {
    std::vector<ShapeOrigin> org;
    ShapeMesh m = shapes(R"([{"Sphere": {"O": [30, 30, 30], "R": 8, "Tag": 1}}])", 16, 64, &org);
    std::vector<NodeCurvature> c = compute_curvature(m, org);
    CHECK(c.size() == m.nodes.size());
    bool ok = true, pd = true;

    for (size_t i = 0; i < c.size(); i++) {
        Vec3 r = normalized(sub(m.nodes[i], Vec3{30, 30, 30}));
        ok &= dot(Vec3{c[i].nx, c[i].ny, c[i].nz}, r) > 1 - 1e-6;
        ok &= std::fabs(c[i].k1 - 0.125f) < 1e-6 && std::fabs(c[i].k2 - 0.125f) < 1e-6;
        Vec3 p = {c[i].px, c[i].py, c[i].pz};
        pd &= std::fabs(v3len(p) - 1) < 1e-5 && std::fabs(dot(p, r)) < 1e-5;   /* tangent */
    }

    CHECK(ok);
    CHECK(pd);
}

TEST(curvature_cylinder_nodes) {
    std::vector<ShapeOrigin> org;
    ShapeMesh m = shapes(R"([{"Cylinder": {"C0": [10, 12, 14], "C1": [40, 35, 30], "R": 4, "Tag": 1}}])", 16, 64, &org);
    std::vector<NodeCurvature> c = compute_curvature(m, org);
    Vec3 c0 = {10, 12, 14}, ax = normalized(Vec3{30, 23, 16});
    size_t nside = 0, ncap = 0;
    bool side_ok = true, cap_ok = true;

    for (size_t i = 0; i < c.size(); i++) {
        Vec3 d = sub(m.nodes[i], c0);
        double p = dot(d, ax);
        Vec3 r = {(float)(d.x - p * ax.x), (float)(d.y - p * ax.y), (float)(d.z - p * ax.z)};
        Vec3 N = {c[i].nx, c[i].ny, c[i].nz}, P = {c[i].px, c[i].py, c[i].pz};

        if (v3len(r) > 1e-3) {
            nside++;
            r = normalized(r);
            side_ok &= dot(N, r) > 1 - 1e-5 && std::fabs(dot(N, ax)) < 1e-5;
            side_ok &= std::fabs(c[i].k1 - 0.25f) < 1e-6 && c[i].k2 == 0;
            side_ok &= std::fabs(dot(P, ax)) < 1e-5 && std::fabs(dot(P, N)) < 1e-5 && std::fabs(v3len(P) - 1) < 1e-5;
        } else {
            ncap++;    /* cap centers: flat, normal along -axis at C0, +axis at C1 */
            cap_ok &= c[i].k1 == 0 && c[i].k2 == 0;
            cap_ok &= std::fabs(dot(N, ax) - (p > 1 ? 1 : -1)) < 1e-5;
        }
    }

    CHECK(nside == 32);
    CHECK(ncap == 2);
    CHECK(side_ok);
    CHECK(cap_ok);
}

TEST(curvature_box_and_misc) {
    std::vector<ShapeOrigin> org;
    ShapeMesh m = shapes(R"([{"Box": {"O": [10, 10, 10], "Size": [10, 20, 30], "Tag": 1}}])", 24, 64, &org);
    std::vector<NodeCurvature> c = compute_curvature(m, org);
    bool ok = true;
    Vec3 cc = {15, 20, 25};

    for (size_t i = 0; i < c.size(); i++) {
        Vec3 N = {c[i].nx, c[i].ny, c[i].nz};
        ok &= c[i].k1 == 0 && c[i].k2 == 0;
        ok &= std::fabs(v3len(N) - 1) < 1e-5 && dot(N, sub(m.nodes[i], cc)) > 0;   /* averaged, outward */
    }

    CHECK(c.size() == 8);
    CHECK(ok);

    /* faces without shape ids leave the nodes at the default */
    ShapeMesh nid = m;
    nid.shape_id.clear();
    c = compute_curvature(nid, org);
    CHECK(c[0].nz == 1 && c[0].k1 == 0 && c[0].px == 1);

    /* lens cap: convex outward = (P-C)/|P-C|, concave flipped */
    ShapeOrigin so = make_origin(3);
    so.cz = -10;
    so.R = 10;
    NodeCurvature nc;
    CHECK(compute_node_curvature_for_shape(so, 0, 0, 0, nc));
    CHECK(nc.nz == 1 && nc.k1 == 0.1f && nc.k2 == 0.1f);
    so.R = -10;
    CHECK(compute_node_curvature_for_shape(so, 0, 0, 0, nc));
    CHECK(nc.nz == -1 && nc.k1 == 0.1f);
    so.type = 0;
    CHECK(!compute_node_curvature_for_shape(so, 1, 2, 3, nc));
    CHECK(nc.k1 == 0 && nc.nz == 1);
    so = make_origin(1);
    so.R = 2;
    CHECK(compute_node_curvature_for_shape(so, 0, 0, 0, nc));    /* at the center: no NaN */
    CHECK(std::isfinite(nc.nx) && std::isfinite(nc.ny) && std::isfinite(nc.nz));

    float tx, ty, tz;
    make_tangent(0, 0, 1, tx, ty, tz);
    CHECK(std::fabs(tx * 0 + ty * 0 + tz * 1) < 1e-6 && std::fabs(std::sqrt(tx * tx + ty * ty + tz * tz) - 1) < 1e-6);
    make_tangent(1, 0, 0, tx, ty, tz);
    CHECK(std::fabs(tx) < 1e-6 && std::fabs(std::sqrt(tx * tx + ty * ty + tz * tz) - 1) < 1e-6);
}

/* mean (1 - cos) error of the curvature-corrected and flat normals at sample points of curved triangles */
static void normal_errors(const char* shapes_js, int res, double& curv_err, double& flat_err, size_t& nsample) {
    json j = base_csg();
    j["Shapes"] = J(shapes_js);
    j["Domain"]["MeshRes"] = res;
    SimConfig cfg = load_json_config(j);
    CmdOverrides ovr;
    SimSetup st = prepare_geometry(cfg, ovr);
    std::vector<ShapeOrigin> org = extract_shape_origins(cfg.shapes, cfg.mesh_res);
    curv_err = flat_err = 0;
    nsample = 0;
    const double bary[4][3] = {{1.0 / 3, 1.0 / 3, 1.0 / 3}, {0.7, 0.15, 0.15}, {0.15, 0.7, 0.15}, {0.15, 0.15, 0.7}};

    for (size_t fi = 0; fi < cfg.faces.size(); fi++) {
        if (face_is_flat(cfg.facedata[fi])) {
            Vec3 P = centroid(cfg.nodes, cfg.faces[fi]);
            Vec3 Nc = curvature_normal_at(cfg, st.curv, P, fi);
            CHECK(Nc.x == cfg.facedata[fi].nx && Nc.y == cfg.facedata[fi].ny && Nc.z == cfg.facedata[fi].nz);
            continue;
        }

        const ShapeOrigin& so = org[cfg.face_shape_id[fi] - 1];

        for (int s = 0; s < 4; s++) {
            Vec3 A = cfg.nodes[cfg.faces[fi][0]], B = cfg.nodes[cfg.faces[fi][1]], C = cfg.nodes[cfg.faces[fi][2]];
            Vec3 P = {(float)(bary[s][0] * A.x + bary[s][1] * B.x + bary[s][2] * C.x),
                      (float)(bary[s][0] * A.y + bary[s][1] * B.y + bary[s][2] * C.y),
                      (float)(bary[s][0] * A.z + bary[s][1] * B.z + bary[s][2] * C.z)
                     };
            Vec3 Nf = {cfg.facedata[fi].nx, cfg.facedata[fi].ny, cfg.facedata[fi].nz};
            Vec3 Na = analytic_normal(so, P, Nf);
            Vec3 Nc = curvature_normal_at(cfg, st.curv, P, fi);
            curv_err += 1 - dot(Nc, Na);
            flat_err += 1 - dot(Nf, Na);
            nsample++;
        }
    }

    if (nsample) {
        curv_err /= nsample;
        flat_err /= nsample;
    }
}

TEST(curvature_corrected_normals_beat_flat) {
    double ce, fe;
    size_t n;
    normal_errors(R"([{"Grid": {"Size": [60, 60, 60], "Tag": 1}}, {"Sphere": {"O": [30, 30, 30], "R": 10, "Tag": 2}}])", 8, ce, fe, n);
    CHECK(n > 0);
    CHECK(fe > 1e-4);
    CHECK(ce < 0.1 * fe);
    fprintf(g_out, "  sphere res 8: mean normal error curvature=%.3e flat=%.3e (%zu samples)\n", ce, fe, n);
    normal_errors(R"([{"Grid": {"Size": [60, 60, 60], "Tag": 1}}, {"Cylinder": {"C0": [30, 30, 10], "C1": [30, 30, 50], "R": 8, "Tag": 2}}])", 6, ce, fe, n);
    CHECK(n > 0);
    CHECK(fe > 1e-4);
    CHECK(ce < 0.1 * fe);
    fprintf(g_out, "  cylinder res 6: mean normal error curvature=%.3e flat=%.3e (%zu samples)\n", ce, fe, n);

    /* analytic_normal: box falls back, axis point falls back */
    ShapeOrigin b = make_origin(0);
    Vec3 fb = {0, 1, 0};
    Vec3 r = analytic_normal(b, Vec3{1, 2, 3}, fb);
    CHECK(r.y == 1);
    ShapeOrigin cyl = make_origin(2);
    cyl.az = 1;
    r = analytic_normal(cyl, Vec3{0, 0, 3}, fb);
    CHECK(r.y == 1);
    ShapeOrigin lens = make_origin(3);
    lens.R = -5;
    r = analytic_normal(lens, Vec3{0, 0, 5}, fb);
    CHECK(r.z == -1);
}

// ============================================================================
// Command line
// ============================================================================
TEST(cmdline_defaults_and_options) {
    CmdOverrides o;
    CHECK(run_cmdline({"vkmmc", "in.json"}, o, 3));
    CHECK(o.inputfile == "in.json");
    CHECK(o.gpuid == 3);
    CHECK(o.nphoton == 0 && o.batch_size == UINT64_MAX && !o.has_seed && o.unitinmm == 0);
    CHECK(o.outputtype == -1 && o.isreflect == -1 && o.isnormalize == -1);
    CHECK(o.meshres == 0 && o.docurv == -1 && o.debuglevel == -1 && o.maxjumpdebug == -1);
    CHECK(!o.listgpu && !o.dumpjson && !o.dumpmesh && !o.debugcurv && !o.show);
    CHECK(o.kernelfile.empty() && o.json_str.empty() && o.session_id.empty());

    CmdOverrides a;
    CHECK(run_cmdline({"vkmmc", "-f", "x.json", "-n", "1e7", "-s", "sess", "-u", "0.5", "-E", "123", "-O", "l",
                       "-b", "0", "-U", "0", "-t", "128", "-B", "1000", "-G", "2", "-j", "{\"a\":1}", "--dumpjson",
                       "--dumpmesh", "--debugcurv", "-m", "16", "-c", "1", "-D", "M", "--maxjumpdebug", "100",
                       "kern.spv"
                      }, a));
    CHECK(a.inputfile == "x.json");
    CHECK(a.nphoton == 10000000ull);
    CHECK(a.session_id == "sess");
    CHECK(a.unitinmm == 0.5f);
    CHECK(a.rng_seed == 123 && a.has_seed);
    CHECK(a.outputtype == OT_FLUENCE);
    CHECK(a.isreflect == 0 && a.isnormalize == 0);
    CHECK(a.totalthread == 128 && a.batch_size == 1000 && a.gpuid == 2);
    CHECK(a.json_str == "{\"a\":1}");
    CHECK(a.dumpjson && a.dumpmesh && a.debugcurv);
    CHECK(a.meshres == 16 && a.docurv == 1);
    CHECK(a.debuglevel == (int)VKMMC_DEBUG_TRAJ);
    CHECK(a.maxjumpdebug == 100);
    CHECK(a.kernelfile == "kern.spv");

    CmdOverrides l;
    CHECK(run_cmdline({"vkmmc", "--input", "y.json", "--photon", "42", "--session", "s2", "--unitinmm", "2",
                       "--seed", "-1", "--outputtype", "flux", "--reflect", "1", "--normalize", "1", "--thread", "64",
                       "--batch", "0", "--gpuid", "1", "--json", "{}", "--meshres", "8", "--curv", "-1",
                       "--debug", "8193", "--listgpu"
                      }, l));
    CHECK(l.inputfile == "y.json" && l.nphoton == 42 && l.session_id == "s2" && l.unitinmm == 2);
    CHECK(l.rng_seed == 0xFFFFFFFFu && l.has_seed);
    CHECK(l.outputtype == OT_FLUX && l.isreflect == 1 && l.isnormalize == 1);
    CHECK(l.totalthread == 64 && l.batch_size == 0 && l.gpuid == 1 && l.meshres == 8 && l.docurv == -1);
    CHECK(l.debuglevel == 8193 && l.listgpu);

    CmdOverrides d;
    CHECK(run_cmdline({"vkmmc", "in.json", "-D", "mx"}, d));
    CHECK(d.debuglevel == (int)((1u << 13) | (1u << 7)));
    CmdOverrides g;
    CHECK(run_cmdline({"vkmmc", "-L"}, g));    /* listing GPUs needs no input */
    CHECK(g.listgpu && g.inputfile.empty());

    CmdOverrides h, v;
    CHECK(!run_cmdline({"vkmmc", "-h"}, h));
    CHECK(!run_cmdline({"vkmmc", "--help"}, h));
    CHECK(!run_cmdline({"vkmmc", "-v"}, v));
    CHECK(!run_cmdline({"vkmmc", "in.json", "--version"}, v));
}

#define CHECK_CMD_THROWS(substr, ...) do { CmdOverrides o_; CHECK_THROWS(run_cmdline(__VA_ARGS__, o_), substr); } while (0)

TEST(cmdline_errors) {
    CHECK_CMD_THROWS("option -n expects a number, got 'abc'", {"vkmmc", "in.json", "-n", "abc"});
    CHECK_CMD_THROWS("option -n expects a number, got '1e7x'", {"vkmmc", "in.json", "-n", "1e7x"});
    CHECK_CMD_THROWS("option -n expects a number", {"vkmmc", "in.json", "-n", ""});
    CHECK_CMD_THROWS("option -n expects a number", {"vkmmc", "in.json", "-n", "inf"});
    CHECK_CMD_THROWS("option -n expects a number", {"vkmmc", "in.json", "-n", "1e400"});
    CHECK_CMD_THROWS("option -n value 0 is out of range", {"vkmmc", "in.json", "-n", "0"});
    CHECK_CMD_THROWS("option -b value 2 is out of range [0, 1]", {"vkmmc", "in.json", "-b", "2"});
    CHECK_CMD_THROWS("option -b expects an integer, got '0.5'", {"vkmmc", "in.json", "-b", "0.5"});
    CHECK_CMD_THROWS("option -U value", {"vkmmc", "in.json", "-U", "-1"});
    CHECK_CMD_THROWS("option -t value 0 is out of range", {"vkmmc", "in.json", "-t", "0"});
    CHECK_CMD_THROWS("option -m value 0 is out of range [1, 2048]", {"vkmmc", "in.json", "-m", "0"});
    CHECK_CMD_THROWS("option -c value 2 is out of range", {"vkmmc", "in.json", "-c", "2"});
    CHECK_CMD_THROWS("option -u value", {"vkmmc", "in.json", "-u", "0"});
    CHECK_CMD_THROWS("option -E value", {"vkmmc", "in.json", "-E", "5e9"});
    CHECK_CMD_THROWS("option -G expects an integer", {"vkmmc", "in.json", "-G", "1.5"});
    CHECK_CMD_THROWS("option --maxjumpdebug value", {"vkmmc", "in.json", "--maxjumpdebug", "-5"});
    CHECK_CMD_THROWS("option -D: unknown debug flag 'Q'", {"vkmmc", "in.json", "-D", "MQ"});
    CHECK_CMD_THROWS("option -D expects a number", {"vkmmc", "in.json", "-D", "8x"});
    CHECK_CMD_THROWS("option -O expects x (energy), f (flux) or l (fluence)", {"vkmmc", "in.json", "-O", "q"});
    CHECK_CMD_THROWS("option -s: session name must have 1-255 characters", {"vkmmc", "in.json", "-s", ""});
    CHECK_CMD_THROWS("option -n requires a value", {"vkmmc", "in.json", "-n"});
    CHECK_CMD_THROWS("option --json requires a value", {"vkmmc", "in.json", "--json"});
    CHECK_CMD_THROWS("unknown option '-Z' (see --help)", {"vkmmc", "in.json", "-Z"});
    CHECK_CMD_THROWS("unknown option '--photons'", {"vkmmc", "in.json", "--photons", "5"});
    CHECK_CMD_THROWS("unexpected argument 'b.json' (input file already given: a.json)", {"vkmmc", "a.json", "b.json"});
    CHECK_CMD_THROWS("unexpected argument", {"vkmmc", "-f", "a.json", "b.json"});
    CHECK_CMD_THROWS("no input file given", {"vkmmc", "-n", "100"});
    CHECK_CMD_THROWS("no input file given", {"vkmmc", "kernel.spv"});
    CHECK_CMD_THROWS("--show is not available", {"vkmmc", "in.json", "--show"});
}

TEST(apply_overrides_fields) {
    SimConfig c = load_json_config(base_csg());
    CmdOverrides none;
    SimConfig c0 = c;
    apply_overrides(c, none);
    CHECK(c.nphoton == c0.nphoton && c.rng_seed == c0.rng_seed && c.session_id == c0.session_id);
    CHECK(c.unitinmm == c0.unitinmm && c.output_type == c0.output_type && c.do_mismatch == c0.do_mismatch);
    CHECK(c.mesh_res == c0.mesh_res && c.debuglevel == c0.debuglevel && c.maxjumpdebug == c0.maxjumpdebug);

    CmdOverrides o;
    CHECK(run_cmdline({"vkmmc", "in.json", "-n", "55", "-E", "0", "-s", "new", "-u", "0.1", "-O", "f", "-b", "0",
                       "-U", "0", "-m", "12", "-D", "0", "--maxjumpdebug", "0"
                      }, o));
    apply_overrides(c, o);
    CHECK(c.nphoton == 55);
    CHECK(c.rng_seed == 0);       /* has_seed: 0 is applied (clock seed) */
    CHECK(c.session_id == "new");
    CHECK(c.unitinmm == 0.1f);
    CHECK(c.output_type == OT_FLUX);
    CHECK(!c.do_mismatch && !c.do_normalize);
    CHECK(c.mesh_res == 12);
    CHECK(c.debuglevel == 0 && c.maxjumpdebug == 0);
    CmdOverrides on;
    CHECK(run_cmdline({"vkmmc", "in.json", "-b", "1", "-U", "1"}, on));
    apply_overrides(c, on);
    CHECK(c.do_mismatch && c.do_normalize);
}

// ============================================================================
// Geometry preparation, grid, kernel parameters
// ============================================================================
TEST(prepare_geometry_variants) {
    json j = base_csg();
    j["Shapes"] = J(R"([{"Grid": {"Size": [60, 60, 60], "Tag": 1}}, {"Sphere": {"O": [30, 30, 30], "R": 10, "Tag": 2}}])");
    j["Domain"]["MeshRes"] = 8;
    SimConfig c = load_json_config(j);
    CmdOverrides o;
    SimSetup st = prepare_geometry(c, o);
    CHECK(st.variant == KV_CSG_CURV && st.has_curvature);
    CHECK(st.curv.size() == c.nodes.size());
    CHECK(c.faces.size() == 12 + 2 * 8 * 7);
    CHECK(c.face_shape_id.size() == c.faces.size());
    CHECK(c.nmin.x == 0 && c.nmax.z == 60);

    CmdOverrides off;
    off.docurv = 0;
    c = load_json_config(j);
    st = prepare_geometry(c, off);
    CHECK(st.variant == KV_CSG && !st.has_curvature && st.curv.empty());

    /* only flat shapes: auto mode disables curvature, -c 1 forces it */
    json jb = base_csg();
    c = load_json_config(jb);
    st = prepare_geometry(c, o);
    CHECK(st.variant == KV_CSG);
    CmdOverrides on;
    on.docurv = 1;
    c = load_json_config(jb);
    st = prepare_geometry(c, on);
    CHECK(st.variant == KV_CSG_CURV && st.curv.size() == 8);

    /* mesh input */
    c = load_json_config(cube_mesh_json());
    st = prepare_geometry(c, o);
    CHECK(st.variant == KV_MESH && !st.has_curvature);
    CHECK(c.faces.size() == 14);
}

TEST(prepare_geometry_errors) {
    CmdOverrides o;
    json j = base_csg();
    j["Domain"]["Media"] = J(R"([{"mua": 0}])");
    SimConfig c = load_json_config(j);
    CHECK_THROWS(prepare_geometry(c, o), "CSG mode needs at least 2 media");

    j = base_csg();
    j["Domain"]["Media"].erase(2);    /* 2 media: tags 1..1 */
    j["Shapes"] = J(R"([{"Sphere": {"O": [30, 30, 30], "R": 10, "Tag": 2}}])");
    c = load_json_config(j);
    CHECK_THROWS(prepare_geometry(c, o), "Shapes[0].Sphere.Tag=2.000000 must be an integer in [1, 1]");

    j = base_csg();
    j["Shapes"] = J(R"([{"Name": "nothing"}])");
    c = load_json_config(j);
    CHECK_THROWS(prepare_geometry(c, o), "Shapes: no surface was generated");

    j = base_csg();
    j["Shapes"] = J(R"([{"Sphere": {"O": [30, 30, 30], "R": 10}}])");
    j["Domain"]["MeshRes"] = 2;
    c = load_json_config(j);
    CHECK_THROWS(prepare_geometry(c, o), "must be at least 4 for curved shapes");

    CmdOverrides on;
    on.docurv = 1;
    c = load_json_config(cube_mesh_json());
    CHECK_THROWS(prepare_geometry(c, on), "curvature (-c 1) requires a CSG shape input");

    /* -j can shrink the media list after the mesh labels were checked */
    c = load_json_config(cube_mesh_json());
    c.media.resize(2);
    CHECK_THROWS(prepare_geometry(c, o), "mesh label 2 has no entry in Domain.Media");
    c = load_json_config(cube_mesh_json(1));
    c.mediumid0 = 5;
    CHECK_THROWS(prepare_geometry(c, o), "the InitElem label has no entry in Domain.Media");
}

TEST(compute_grid_dims) {
    SimConfig c;
    c.nodes = {{0, 0, 0}, {60, 40, 20}};
    update_bbox(c);
    OutputGrid g = compute_grid(c);
    CHECK(g.vs == 1.0f);
    CHECK(g.nx == 61 && g.ny == 41 && g.nz == 21);
    CHECK(g.gmin[0] == -0.5f && g.gmax[0] == 60.5f && g.gmin[2] == -0.5f);
    CHECK(g.nvox == 61ull * 41 * 21 && g.nelem == g.nvox && g.maxgate == 1);

    c.has_steps = true;
    c.steps[0] = c.steps[1] = c.steps[2] = 2;
    c.maxgate = 4;
    g = compute_grid(c);
    CHECK(g.vs == 2 && g.nx == 31 && g.ny == 21 && g.nz == 11);
    CHECK(g.gmin[1] == -1 && g.gmax[1] == 41);
    CHECK(g.nelem == g.nvox * 4);

    /* still Step 2: x [-4,-1.5] -> 2, y [4,6.2] -> 2, z [6,10] -> 2 */
    c.nodes = {{-3, 5, 7}, {-2.5f, 5.2f, 9}};
    update_bbox(c);
    g = compute_grid(c);
    CHECK(g.nx == 2 && g.ny == 2 && g.nz == 2);
    CHECK(g.gmin[0] == -4.0f);

    SimConfig big;
    big.nodes = {{0, 0, 0}, {2000, 2000, 2000}};
    update_bbox(big);
    CHECK_THROWS(compute_grid(big), "the output grid 2001x2001x2001x1 exceeds 1073741824 elements");
    big.nodes = {{0, 0, 0}, {1e8f, 1, 1}};
    update_bbox(big);
    CHECK_THROWS(compute_grid(big), "the output grid is too large; increase Domain.Step");
    SimConfig gates;
    gates.nodes = {{0, 0, 0}, {99, 99, 99}};
    update_bbox(gates);
    gates.maxgate = 1074;    /* 100^3 * 1074 > 2^30 */
    CHECK_THROWS(compute_grid(gates), "increase Domain.Step or Forward.Dt");
    SimConfig empty;
    CHECK_THROWS(compute_grid(empty), "the mesh has no nodes");
}

TEST(kernel_parameters) {
    SimConfig c;
    c.media = {{0, 0, 1, 1}, {0.02f, 10, 0.9f, 1.37f}};
    c.unitinmm = 0.5f;
    c.t0 = 1e-9f;
    c.t1 = 5e-9f;
    c.dt = 1e-9f;
    std::vector<Medium> km = kernel_media(c);
    CHECK(km.size() == 2);
    CHECK_CLOSE(km[1].mua, 0.01, 1e-8);
    CHECK_CLOSE(km[1].mus, 5, 1e-6);
    CHECK(km[1].g == 0.9f && km[1].n == 1.37f);
    CHECK(c.media[1].mua == 0.02f);    /* input untouched */
    float ts, te, idt;
    kernel_time_gates(c, ts, te, idt);
    CHECK_CLOSE(ts, 2e-9, 1e-16);
    CHECK_CLOSE(te, 1e-8, 1e-16);
    CHECK_CLOSE(idt, 5e8, 1e2);
    c.unitinmm = 1;
    kernel_time_gates(c, ts, te, idt);
    CHECK_CLOSE(idt, 1e9, 1e2);

    CmdOverrides o;
    c.nphoton = 1000000;
    CHECK(thread_count(c, o) == 65536);
    o.totalthread = 100;
    CHECK(thread_count(c, o) == 128);
    o.totalthread = 64;
    CHECK(thread_count(c, o) == 64);
    o.totalthread = 1;
    CHECK(thread_count(c, o) == 64);
    o.totalthread = 0;
    c.nphoton = 10;
    CHECK(thread_count(c, o) == 64);
    c.nphoton = 100;
    CHECK(thread_count(c, o) == 128);
    c.nphoton = 65537;
    CHECK(thread_count(c, o) == 65536);

    c.nphoton = 1234567;
    CmdOverrides b;
    CHECK(batch_size(c, b) == 500000);
    b.batch_size = 0;
    CHECK(batch_size(c, b) == 1234567);
    b.batch_size = 1000;
    CHECK(batch_size(c, b) == 1000);

    c.rng_seed = 77;
    CHECK(base_seed(c) == 77);
    c.rng_seed = 0;
    CHECK(base_seed(c) != 0);
}

TEST(make_seeds_properties) {
    std::vector<uint32_t> s1, s2, s3, s4;
    make_seeds(12345, 0, 256, s1);
    make_seeds(12345, 0, 256, s2);
    CHECK(s1.size() == 1024);
    CHECK(s1 == s2);
    bool nz = true;

    for (size_t i = 0; i < s1.size(); i++) {
        nz &= s1[i] != 0;
    }

    CHECK(nz);
    CHECK(std::set<uint32_t>(s1.begin(), s1.end()).size() == s1.size());
    make_seeds(12345, 500000, 256, s3);
    make_seeds(12346, 0, 256, s4);
    CHECK(s3 != s1 && s4 != s1);

    /* per-thread 128-bit states never repeat across batches or threads */
    std::set<std::array<uint32_t, 4> > states;
    std::vector<uint32_t> s;

    for (int batch = 0; batch < 8; batch++) {
        make_seeds(29012391, (uint64_t)batch * 500000, 4096, s);

        for (size_t t = 0; t < 4096; t++) {
            states.insert({{s[4 * t], s[4 * t + 1], s[4 * t + 2], s[4 * t + 3]}});
        }
    }

    CHECK(states.size() == 8 * 4096u);
    make_seeds(1, 0, 0, s);
    CHECK(s.empty());
}

// ============================================================================
// Output normalization and files
// ============================================================================
static OutputGrid small_grid(uint32_t nx, uint32_t ny, uint32_t nz, int nt, float vs) {
    OutputGrid g;
    g.nx = nx;
    g.ny = ny;
    g.nz = nz;
    g.maxgate = nt;
    g.vs = vs;
    g.gmin[0] = g.gmin[1] = g.gmin[2] = -0.5f;
    g.gmax[0] = g.gmax[1] = g.gmax[2] = 10;
    g.nvox = (uint64_t)nx * ny * nz;
    g.nelem = g.nvox * nt;
    return g;
}

TEST(accumulate_fixed_point) {
    /* the kernels deposit 64-bit fixed-point sums, 2^32 units per photon weight */
    std::vector<uint64_t> raw = {0, (uint64_t)1 << 32, (uint64_t)3 << 31, 1, ((uint64_t)1 << 63) + 5};
    std::vector<double> acc;
    accumulate_fixed(acc, raw.data(), raw.size());
    CHECK(acc.size() == raw.size());
    CHECK(acc[0] == 0.0 && acc[1] == 1.0 && acc[2] == 1.5);
    CHECK_CLOSE(acc[3], 1.0 / 4294967296.0, 1e-20);
    CHECK_CLOSE(acc[4], 2147483648.0, 1e-3);
    accumulate_fixed(acc, raw.data(), raw.size());     /* batches add up */
    CHECK(acc[1] == 2.0 && acc[2] == 3.0);
}

TEST(finalize_output_normalization) {
    OutputGrid g = small_grid(2, 3, 2, 1, 2.0f);
    std::vector<double> acc(g.nelem);

    for (size_t i = 0; i < g.nelem; i++) {
        acc[i] = (double)i + 0.5;
    }

    acc[3] = NAN;
    acc[5] = INFINITY;
    acc[7] = -INFINITY;

    SimConfig c;
    c.nphoton = 1000;
    c.unitinmm = 0.25f;
    c.dt = 2e-9f;
    c.has_steps = true;
    c.do_normalize = false;
    c.output_type = OT_ENERGY;
    double absorbed = 0;
    std::vector<float> o = finalize_output(c, g, acc, &absorbed);
    CHECK(o.size() == g.nelem);
    CHECK(o[0] == 0.5f && o[10] == 10.5f);
    CHECK(o[3] == 0 && o[5] == 0 && o[7] == 0);    /* NaN/Inf zeroed */
    double total = 0;

    for (size_t i = 0; i < o.size(); i++) {
        total += o[i];
    }

    CHECK_CLOSE(absorbed, total / 1000.0, 1e-9);

    /* energy: 1 / (N * Vmm^3), Vmm = vs * unitinmm = 0.5 mm */
    c.do_normalize = true;
    o = finalize_output(c, g, acc);
    double s_energy = 1.0 / (1000.0 * 0.125);
    CHECK_CLOSE(o[10], 10.5 * s_energy, 1e-5);
    CHECK(o[3] == 0);

    /* fluence: additionally * unitinmm */
    c.output_type = OT_FLUENCE;
    o = finalize_output(c, g, acc, &absorbed);
    CHECK_CLOSE(o[10], 10.5 * s_energy * 0.25, 1e-6);
    CHECK(absorbed == -1);

    /* flux: additionally / dt */
    c.output_type = OT_FLUX;
    o = finalize_output(c, g, acc);
    CHECK_CLOSE(o[10] / (10.5 * s_energy * 0.25 / 2e-9), 1.0, 1e-6);

    /* unit length unit and step; a short accumulator is zero padded */
    SimConfig u;
    u.nphoton = 4;
    u.do_normalize = true;
    OutputGrid g1 = small_grid(1, 1, 2, 1, 1.0f);
    std::vector<double> a1 = {4};
    o = finalize_output(u, g1, a1);
    CHECK(o[0] == 1.0f && o[1] == 0.0f);
}

TEST(save_fluence_round_trip) {
    for (int nt : {1, 2}) {
        OutputGrid g = small_grid(3, 4, 5, nt, 1.5f);
        std::vector<float> vol(g.nelem);

        for (size_t i = 0; i < vol.size(); i++) {
            vol[i] = (float)i * 1.25f + 0.5f;
        }

        SimConfig c;
        c.session_id = tmpfile_path("fluence_" + std::to_string(nt));
        c.nphoton = 321;
        c.output_type = OT_FLUENCE;
        save_fluence(c, g, vol);
        std::string fn = c.session_id + ".jdat";
        CHECK(file_exists(fn));
        json j = load_json_file(fn.c_str());
        CHECK(j["Session"]["Photons"] == 321 && j["Session"]["OutputType"] == "fluence");
        CHECK(j["Domain"]["Dim"] == J("[3,4,5]"));
        CHECK(j["Domain"]["VoxelSize"] == 1.5);
        CHECK(j["Fluence"]["_ArrayOrder_"] == "c");
        JDataArray a = jdata_decode_array(j["Fluence"]);
        std::vector<size_t> dims = {3, 4, 5};

        if (nt > 1) {
            dims.push_back(nt);
        }

        CHECK(a.dims == dims);
        CHECK(a.type == "single");
        bool ok = true;

        for (int it = 0; it < nt; it++) {
            for (int iz = 0; iz < 5; iz++) {
                for (int iy = 0; iy < 4; iy++) {
                    for (int ix = 0; ix < 3; ix++) {
                        size_t rm = (((size_t)ix * 4 + iy) * 5 + iz) * nt + it;
                        ok &= a.at(rm) == vol[ix + iy * 3 + iz * 12 + it * 60];
                    }
                }
            }
        }

        CHECK(ok);
    }

    SimConfig bad;
    bad.session_id = tmpfile_path("no_such_dir/out");
    OutputGrid g = small_grid(1, 1, 1, 1, 1);
    CHECK_THROWS(save_fluence(bad, g, std::vector<float>(1, 0.f)), "cannot write output file");
}

TEST(trajectory_round_trip) {
    uint32_t rec[2 * 6];
    float p[2][4] = {{1.5f, 2.5f, 3.5f, 0.75f}, {-1, -2, -3, 0.25f}};

    for (int i = 0; i < 2; i++) {
        rec[i * 6] = 10 + i;
        memcpy(&rec[i * 6 + 1], p[i], 16);
        rec[i * 6 + 5] = 100 + i;
    }

    TrajData t = unpack_trajectory(rec, 2);
    CHECK(t.ids.size() == 2 && t.ids[1] == 11 && t.evts[0] == 100);
    CHECK(t.pts[3] == -1 && t.pts[2] == 3.5f && t.w[0] == 0.75f);

    SimConfig c;
    c.session_id = tmpfile_path("traj");
    c.media = {{0, 0, 1, 1}, {0.1f, 1, 0.9f, 1.4f}};
    save_trajectory(c, t);
    json j = load_json_file((c.session_id + "_traj.jdat").c_str());
    CHECK(j["MCXData"]["Info"]["MediaNum"] == 2 && j["MCXData"]["Info"]["SavedPhoton"] == 2);
    JDataArray pa = jdata_decode_array(j["MCXData"]["Trajectory"]["p"]);
    CHECK(pa.dims == std::vector<size_t>({2, 3}) && pa.at(4) == -2);
    JDataArray ids = jdata_decode_array(j["MCXData"]["Trajectory"]["photonid"]);
    CHECK(ids.type == "uint32" && ids.at(0) == 10);
}

static std::string sphere_input(const std::string& sess) {
    json j = base_csg();
    j["Session"]["ID"] = sess;
    j["Domain"]["MeshRes"] = 8;
    j["Shapes"] = J(R"([{"Grid": {"Size": [60, 60, 60], "Tag": 1}}, {"Sphere": {"O": [30, 30, 30], "R": 10, "Tag": 2}}])");
    return j.dump();
}

TEST(dump_mesh_reload) {
    json j = J(sphere_input(tmpfile_path("dm")).c_str());
    SimConfig c = load_json_config(j);
    CmdOverrides o;
    prepare_geometry(c, o);
    dump_mesh(c);
    std::string fn = c.session_id + "_mesh.json";
    CHECK(file_exists(fn));
    SimConfig r = load_json_config(load_json_file(fn.c_str()));
    CHECK(!r.is_csg);
    CHECK(r.nodes.size() == c.nodes.size());
    CHECK(r.faces.size() == c.faces.size());
    CHECK(r.media.size() == c.media.size() && r.media[1].mus == c.media[1].mus);
    CHECK(r.nphoton == c.nphoton && r.srcpos[0] == c.srcpos[0]);

    /* same triangles, same labels, same orientation */
    std::map<std::array<uint32_t, 3>, std::pair<uint32_t, Vec3> > ref;

    for (size_t i = 0; i < c.faces.size(); i++) {
        std::array<uint32_t, 3> k = c.faces[i];
        std::sort(k.begin(), k.end());
        uint32_t fr, bk;
        bool fl;
        unpack_media(c.facedata[i].packed_media, fr, bk, fl);
        ref[k] = std::make_pair(fr << 16 | bk, Vec3{c.facedata[i].nx, c.facedata[i].ny, c.facedata[i].nz});
    }

    bool match = true;

    for (size_t i = 0; i < r.faces.size(); i++) {
        std::array<uint32_t, 3> k = r.faces[i];
        std::sort(k.begin(), k.end());
        uint32_t fr, bk;
        bool fl;
        unpack_media(r.facedata[i].packed_media, fr, bk, fl);
        auto it = ref.find(k);
        match &= it != ref.end() && it->second.first == (fr << 16 | bk) &&
                 dot(it->second.second, Vec3{r.facedata[i].nx, r.facedata[i].ny, r.facedata[i].nz}) > 0.9999;
    }

    CHECK(match);
    bool nodes_eq = true;

    for (size_t i = 0; i < r.nodes.size() && i < c.nodes.size(); i++) {
        nodes_eq &= r.nodes[i].x == c.nodes[i].x && r.nodes[i].y == c.nodes[i].y && r.nodes[i].z == c.nodes[i].z;
    }

    CHECK(nodes_eq);

    /* an interface face (front != 0) is written twice and read back with both labels */
    SimConfig two = load_json_config(cube_mesh_json());
    two.session_id = tmpfile_path("dm2");
    dump_mesh(two);
    SimConfig r2 = load_json_config(load_json_file((two.session_id + "_mesh.json").c_str()));
    CHECK(r2.faces.size() == 14);
    size_t nint = 0;

    for (size_t i = 0; i < r2.facedata.size(); i++) {
        uint32_t fr, bk;
        bool fl;
        unpack_media(r2.facedata[i].packed_media, fr, bk, fl);
        nint += (fr != 0);
    }

    CHECK(nint == 2);

    SimConfig none;
    CHECK_THROWS(dump_mesh(none), "no mesh to dump");
}

TEST(dump_config_json_output) {
    SimConfig c = load_json_config(J(sphere_input("dj").c_str()));
    CmdOverrides o;
    SimSetup st = prepare_geometry(c, o);
    std::string out = capture_stdout([&]() {
        dump_config_json(c, st);
    });
    json d;
    bool parsed = true;

    try {
        d = json::parse(out);
    } catch (...) {
        parsed = false;
    }

    CHECK(parsed);
    CHECK(parsed && d["Session"]["ID"] == "dj" && d["Session"]["Photons"] == 1000);
    CHECK(parsed && d["Mesh"]["Mode"] == "csg" && d["Mesh"]["Faces"] == c.faces.size());
    CHECK(parsed && d["Mesh"]["Kernel"] == "csg+curvature" && d["Mesh"]["Curvature"] == true);
    CHECK(parsed && d["Domain"]["Media"].size() == 3 && d["Forward"]["Gates"] == 1);
    CHECK(parsed && d["Optode"]["Source"]["Type"] == "pencil");
}

TEST(debugcurv_output) {
    SimConfig c = load_json_config(J(sphere_input(tmpfile_path("dc")).c_str()));
    CmdOverrides o;
    SimSetup st = prepare_geometry(c, o);
    run_debugcurv(c, st);
    std::string fn = c.session_id + "_curv.json";
    CHECK(file_exists(fn));
    json j = load_json_file(fn.c_str());
    CHECK(j["CoarseMesh"]["MeshRes"] == 8 && j["FineMesh"]["MeshRes"] == 32);
    CHECK(j["ErrorStats"]["CurvMean"].get<double>() < j["ErrorStats"]["FlatMean"].get<double>());
    JDataArray mn = jdata_decode_array(j["Shapes"]["MeshNode"]);
    CHECK(mn.dims.size() == 2 && mn.dims[1] == 5 && mn.dims[0] == j["FineMesh"]["Nodes"].get<size_t>());

    SimSetup flat;
    CHECK_THROWS(run_debugcurv(c, flat), "--debugcurv requires a CSG input with curvature");
    SimConfig boxes = load_json_config(base_csg());
    boxes.session_id = tmpfile_path("boxes");
    CmdOverrides on;
    on.docurv = 1;
    SimSetup bs = prepare_geometry(boxes, on);
    CHECK_NOTHROW(run_debugcurv(boxes, bs));    /* flat nodes are evaluated with the face normal */
}

TEST(frontend_flow) {
    std::string in = tmpfile_path("fe.json");
    write_text(in, sphere_input(tmpfile_path("fe")));
    CmdOverrides o;
    SimConfig c;
    SimSetup st;
    CHECK(run_frontend({"vkmmc"}, o, c, st) == 0);
    CmdOverrides o1;
    CHECK(run_frontend({"vkmmc", "-h"}, o1, c, st) == 0);
    CmdOverrides o2;
    CHECK(run_frontend({"vkmmc", "-L"}, o2, c, st) == 1);
    CmdOverrides o3;
    CHECK(run_frontend({"vkmmc", in, "-n", "77", "-j", "{\"Session\": {\"RNGSeed\": 5}}"}, o3, c, st) == 1);
    CHECK(c.nphoton == 77 && c.rng_seed == 5 && st.variant == KV_CSG_CURV && !c.faces.empty());
    CmdOverrides o4;
    std::string js = capture_stdout([&]() {
        CHECK(run_frontend({"vkmmc", in, "--dumpjson"}, o4, c, st) == 0);
    });
    CHECK(js.find("\"Kernel\": \"csg+curvature\"") != std::string::npos);
    CmdOverrides o5;
    CHECK(run_frontend({"vkmmc", in, "--dumpmesh", "-s", tmpfile_path("fe_dump")}, o5, c, st) == 0);
    CHECK(file_exists(tmpfile_path("fe_dump_mesh.json")));
    CmdOverrides o6;
    CHECK(run_frontend({"vkmmc", in, "--debugcurv"}, o6, c, st) == 0);
    CHECK(file_exists(tmpfile_path("fe_curv.json")));
    CmdOverrides o7;
    CHECK_THROWS(run_frontend({"vkmmc", in, "-j", "{\"Domain\": {\"Media\": []}}"}, o7, c, st),
                 "Domain.Media must be a non-empty array");
    CmdOverrides o8;
    CHECK_THROWS(run_frontend({"vkmmc", tmpfile_path("missing.json")}, o8, c, st), "cannot open input file");
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    bool verbose = false;
    const char* only = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-v")) {
            verbose = true;
        } else {
            only = argv[i];    /* run tests whose name contains this string */
        }
    }

    /* the headers print diagnostics to stdout; keep the report readable */
    int rep = dup(1);
    g_out = fdopen(rep, "w");
    setvbuf(g_out, NULL, _IONBF, 0);

    if (!verbose) {
        int devnull = open("/dev/null", O_WRONLY);
        fflush(stdout);
        dup2(devnull, 1);
        close(devnull);
    }

    int npass = 0, nfail = 0, nrun = 0;

    for (size_t i = 0; i < test_registry().size(); i++) {
        const TestEntry& t = test_registry()[i];

        if (only && !strstr(t.name, only)) {
            continue;
        }

        g_test = t.name;
        g_test_failed = 0;
        nrun++;

        try {
            t.fn();
        } catch (const std::exception& e) {
            REPORT_FAIL("uncaught exception: %s", e.what());
        }

        if (g_test_failed) {
            nfail++;
            fprintf(g_out, "[FAIL] %s (%d failed checks)\n", t.name, g_test_failed);
        } else {
            npass++;
            fprintf(g_out, "[ OK ] %s\n", t.name);
        }
    }

    fprintf(g_out, "\n%d tests: %d passed, %d failed; %d checks: %d passed, %d failed\n", nrun, npass, nfail,
            g_checks, g_checks - g_failed_checks, g_failed_checks);

    if (!g_tmpdir.empty()) {
        nftw(g_tmpdir.c_str(), rm_entry, 16, FTW_DEPTH | FTW_PHYS);
    }

    fflush(stdout);
    return g_failed_checks ? 1 : 0;
}
