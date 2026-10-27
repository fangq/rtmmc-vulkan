/*
 * vkmmc_host.h — host-side logic shared by the Vulkan (vkmmc) and OptiX
 * (vkmmc_optix) front-ends: command line, geometry preparation, output grid,
 * RNG seeding, result post-processing and file output.
 *
 * Units: all coordinates (shapes, mesh nodes, source, Domain.Step) are in
 * "length units" of Domain.LengthUnit (or -u) millimeters each. Optical
 * properties are given in 1/mm. The kernels work in length units, so the host
 * scales mua/mus and the time gates accordingly.
 */
#ifndef VKMMC_HOST_H
#define VKMMC_HOST_H

#include "vkmmc_io.h"
#include "vkmmc_shapes.h"
#include "vkmmc_curvature.h"
#include <cerrno>
#include <climits>
#include <ctime>

#define VKMMC_VERSION    "0.5.0"
#define VKMMC_CODENAME   "Kylin"
#define VKMMC_URL        "https://github.com/fangq/vkmmc"

#define VKMMC_DEBUG_TRAJ 8192u      /* -D M: save photon trajectories */
#define VKMMC_C0         299792458000.0   /* speed of light in vacuum, mm/s */

/* kernel variants; each binary embeds all of them and picks one per input */
enum KernelVariant { KV_MESH = 0, KV_CSG = 1, KV_CSG_CURV = 2, KV_COUNT = 3 };

static const char* kernel_variant_name(int v) {
    static const char* names[] = {"mesh", "csg", "csg+curvature"};
    return (v >= 0 && v < KV_COUNT) ? names[v] : "unknown";
}

// ============================================================================
// Command line
// ============================================================================
struct CmdOverrides {
    uint64_t nphoton, batch_size;
    uint32_t rng_seed, totalthread;
    bool has_seed;
    float unitinmm;
    int outputtype, isreflect, isspecular, isnormalize, gpuid;
    bool listgpu, dumpjson, dumpmesh;
    int meshres;
    int docurv;         // -1 = auto (enable for CSG), 0 = off, 1 = on
    bool debugcurv;
    int debuglevel;     // -1 = not set
    int64_t maxjumpdebug;   // -1 = not set
    bool show;          // launch interactive viewer
    std::string traj_file;  // optional _traj.jdat to display without simulating
    std::string session_id, json_str, inputfile;
    std::string kernelfile; // optional .spv/.ptx kernel overriding the embedded one
    CmdOverrides(): nphoton(0), batch_size(UINT64_MAX), rng_seed(0), totalthread(0), has_seed(false), unitinmm(0),
        outputtype(-1), isreflect(-1), isspecular(-1), isnormalize(-1), gpuid(-1), listgpu(false), dumpjson(false), dumpmesh(false),
        meshres(0), docurv(-1), debugcurv(false), debuglevel(-1), maxjumpdebug(-1), show(false) {}
};

static void print_version(const char* prog, const char* backend) {
    printf("%s %s (%s, beta), %s backend - ray-tracing accelerated mesh/shape-based Monte Carlo\n", prog,
           VKMMC_VERSION, VKMMC_CODENAME, backend);
}

static void printhelp(const char* prog, const char* backend, int defgpu) {
    print_version(prog, backend);
    printf("Usage: %s input.json [flags]  OR  %s -f input.json [flags]\n\n"
           "Flags:\n"
           " -f/--input  file    JSON input file\n"
           " -j/--json   str     JSON string merged on top of the input file\n"
           " -n/--photon num     photon number (accepts 1e7)\n"
           " -s/--session name   session name, used as output file name prefix\n"
           " -u/--unitinmm mm    length of one coordinate unit in mm [Domain.LengthUnit]\n"
           " -E/--seed   int     RNG seed (>0: reproducible; 0: from the clock)\n"
           " -O/--outputtype c   x: energy deposit, f: flux, l: fluence [x]\n"
           " -b/--reflect 0|1    refractive index mismatch at boundaries [1]\n"
           " -V/--specular 0|1   Fresnel reflection where photons from outside enter [0]\n"
           " -U/--normalize 0|1  normalize the output [1]\n"
           " -t/--thread num     total GPU threads, a multiple of 64 [131072]\n"
           " -B/--batch  num     photons per kernel launch [auto: about 0.5 s each, 0: all at once]\n"
           " -G/--gpuid  id      GPU device [%d]; -L lists them\n"
           " -m/--meshres num    shape mesh resolution [24], %d-%d\n"
           " -c/--curv   -1|0|1  curvature-corrected normals [-1: auto, on for curved shapes]\n"
           " -D/--debug  flags   debug flags as letters (M: save trajectories) or a bitmask\n"
           " --maxjumpdebug num  maximum trajectory events to save [10000000]\n"
           " --debugcurv         export the curvature normal error map and exit\n"
           " --dumpjson          print the parsed input as JSON and exit\n"
           " --dumpmesh          save the generated surface mesh to <session>_mesh.json and exit\n"
#ifdef USE_SHOW
           " --show [file.jdat]  interactive viewer; with a .jdat, display a saved result\n"
#endif
           " -L/--listgpu        list GPUs\n"
           " -v/--version        print version\n"
           " -h/--help           print this help\n"
           "A trailing .spv (vkmmc) or .ptx (vkmmc_optix) file overrides the embedded kernel.\n"
           "URL: %s\n", prog, prog, defgpu, VKMMC_MIN_MESHRES, VKMMC_MAX_MESHRES, VKMMC_URL);
}

/* strict numeric parsing: the whole argument must be a number within [lo, hi] */
static double parse_num_arg(const char* opt, const char* s, double lo, double hi) {
    char* end = NULL;
    errno = 0;
    double v = strtod(s, &end);

    if (end == s || *end != '\0' || errno == ERANGE || !std::isfinite(v)) {
        vkmmc_fail(std::string("option ") + opt + " expects a number, got '" + s + "'");
    }

    if (!(v >= lo && v <= hi)) {
        std::ostringstream ss;
        ss << "option " << opt << " value " << s << " is out of range [" << lo << ", " << hi << "]";
        vkmmc_fail(ss.str());
    }

    return v;
}

static int parse_int_arg(const char* opt, const char* s, double lo, double hi) {
    double v = parse_num_arg(opt, s, lo, hi);
    vkmmc_require(v == std::floor(v), std::string("option ") + opt + " expects an integer, got '" + s + "'");
    return (int)v;
}

/*
 * Parse argv into ovr. Returns false when the program should exit without
 * running (help, version); throws on invalid arguments.
 */
static bool parse_cmdline(int argc, char** argv, CmdOverrides& ovr, const char* backend, int defgpu,
                          const char* kernel_ext) {
    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        bool hasval = (i + 1 < argc);
        auto val = [&]() -> const char* {
            if (!hasval) {
                vkmmc_fail("option " + a + " requires a value");
            }

            hasval = false;
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            printhelp(argv[0], backend, defgpu);
            return false;
        } else if (a == "-v" || a == "--version") {
            print_version(argv[0], backend);
            return false;
        } else if (a == "-f" || a == "--input") {
            ovr.inputfile = val();
        } else if (a == "-n" || a == "--photon") {
            ovr.nphoton = (uint64_t)parse_num_arg(a.c_str(), val(), 1, 9.2e18);
        } else if (a == "-s" || a == "--session") {
            ovr.session_id = val();
            vkmmc_require(!ovr.session_id.empty() && ovr.session_id.size() < 256, "option -s: session name must have 1-255 characters");
        } else if (a == "-u" || a == "--unitinmm") {
            ovr.unitinmm = (float)parse_num_arg(a.c_str(), val(), 1e-6, 1e6);
        } else if (a == "-E" || a == "--seed") {
            ovr.rng_seed = (uint32_t)(int64_t)parse_num_arg(a.c_str(), val(), -2147483648.0, 4294967295.0);
            ovr.has_seed = true;
        } else if (a == "-O" || a == "--outputtype") {
            ovr.outputtype = parse_outputtype(val());
            vkmmc_require(ovr.outputtype >= 0, "option -O expects x (energy), f (flux) or l (fluence)");
        } else if (a == "-b" || a == "--reflect") {
            ovr.isreflect = parse_int_arg(a.c_str(), val(), 0, 1);
        } else if (a == "-V" || a == "--specular") {
            ovr.isspecular = parse_int_arg(a.c_str(), val(), 0, 1);
        } else if (a == "-U" || a == "--normalize") {
            ovr.isnormalize = parse_int_arg(a.c_str(), val(), 0, 1);
        } else if (a == "-t" || a == "--thread") {
            ovr.totalthread = (uint32_t)parse_int_arg(a.c_str(), val(), 1, 1 << 24);
        } else if (a == "-B" || a == "--batch") {
            ovr.batch_size = (uint64_t)parse_num_arg(a.c_str(), val(), 0, 9.2e18);
        } else if (a == "-G" || a == "--gpuid") {
            ovr.gpuid = parse_int_arg(a.c_str(), val(), 0, 1024);
        } else if (a == "-L" || a == "--listgpu") {
            ovr.listgpu = true;
        } else if (a == "-j" || a == "--json") {
            ovr.json_str = val();
        } else if (a == "--dumpjson") {
            ovr.dumpjson = true;
        } else if (a == "--dumpmesh") {
            ovr.dumpmesh = true;
        } else if (a == "--debugcurv") {
            ovr.debugcurv = true;
        } else if (a == "-m" || a == "--meshres") {
            ovr.meshres = parse_int_arg(a.c_str(), val(), 1, VKMMC_MAX_MESHRES);
        } else if (a == "-c" || a == "--curv") {
            ovr.docurv = parse_int_arg(a.c_str(), val(), -1, 1);
        } else if (a == "-D" || a == "--debug") {
            const char* darg = val();
            ovr.debuglevel = isalpha((unsigned char)darg[0]) ? (int)parse_debugopt(darg, "option -D")
                             : parse_int_arg("-D", darg, 0, INT_MAX);
        } else if (a == "--maxjumpdebug") {
            ovr.maxjumpdebug = (int64_t)parse_num_arg(a.c_str(), val(), 0, 1e9);
        } else if (a == "--show") {
#ifdef USE_SHOW
            ovr.show = true;

            if (i + 1 < argc) {
                std::string next = argv[i + 1];

                if (next.size() > 5 && next.compare(next.size() - 5, 5, ".jdat") == 0) {
                    ovr.traj_file = argv[++i];
                }
            }

#else
            vkmmc_fail("--show is not available: rebuild with 'make SHOW=1'");
#endif
        } else if (a.size() > 1 && a[0] == '-') {
            vkmmc_fail("unknown option '" + a + "' (see --help)");
        } else if (kernel_ext && a.size() > 4 && a.compare(a.size() - 4, 4, kernel_ext) == 0) {
            ovr.kernelfile = a;
        } else if (ovr.inputfile.empty()) {
            ovr.inputfile = a;
        } else {
            vkmmc_fail("unexpected argument '" + a + "' (input file already given: " + ovr.inputfile + ")");
        }
    }

    if (ovr.listgpu) {
        return true;
    }

    if (ovr.inputfile.empty() && !(ovr.show && !ovr.traj_file.empty())) {
        printhelp(argv[0], backend, defgpu);
        vkmmc_fail("no input file given");
    }

    if (ovr.gpuid < 0) {
        ovr.gpuid = defgpu;
    }

    return true;
}

/* apply command line overrides on top of the input file */
static void apply_overrides(SimConfig& cfg, const CmdOverrides& ovr) {
    if (ovr.nphoton > 0) {
        cfg.nphoton = ovr.nphoton;
    }

    if (ovr.has_seed) {
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

    if (ovr.isspecular >= 0) {
        cfg.do_specular = (ovr.isspecular != 0);
    }

    if (ovr.isnormalize >= 0) {
        cfg.do_normalize = (ovr.isnormalize != 0);
    }

    if (ovr.meshres > 0) {
        cfg.mesh_res = ovr.meshres;
    }

    if (ovr.debuglevel >= 0) {
        cfg.debuglevel = (uint32_t)ovr.debuglevel;
    }

    if (ovr.maxjumpdebug >= 0) {
        cfg.maxjumpdebug = (uint32_t)ovr.maxjumpdebug;
    }
}

// ============================================================================
// Geometry preparation
// ============================================================================
struct SimSetup {
    std::vector<NodeCurvature> curv;
    bool has_curvature;
    int variant;          /* KernelVariant */
    SimSetup(): has_curvature(false), variant(KV_MESH) {}
};

/*
 * Build the surface mesh (CSG mode), compute curvature, and check that every
 * label and the initial medium address a defined medium.
 */
#define VKMMC_SAFETY_DIST 0.001f   /* SAFETY_DIST of the kernels */

static int kernel_focus_mode(float f);

static void cross3(const double* a, const double* b, double* c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

static double dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/*
 * A source on a face is inside the tissue there, as in MCX. The kernels move
 * such photons off the face; a fixed pencil beam is moved here once instead:
 * back before the face within VKMMC_SAFETY_DIST of Pos along Dir if the side
 * behind it is a medium, else past it. Face normals point to the front medium.
 */
static void snap_pencil_source(SimConfig& cfg) {
    if (cfg.srctype != SRC_PENCIL || !cfg.extrasrc.empty() || kernel_focus_mode(cfg.srcdir[3]) != 0) {
        return;
    }

    const double d[3] = {cfg.srcdir[0], cfg.srcdir[1], cfg.srcdir[2]};
    float tbest = 2 * VKMMC_SAFETY_DIST;
    size_t best = 0;

    for (size_t i = 0; i < cfg.faces.size(); i++) {   /* Moller-Trumbore, t in [-SAFETY_DIST, SAFETY_DIST] */
        const Vec3& a = cfg.nodes[cfg.faces[i][0]], &b = cfg.nodes[cfg.faces[i][1]], &c = cfg.nodes[cfg.faces[i][2]];
        double e1[3] = {b.x - a.x, b.y - a.y, b.z - a.z}, e2[3] = {c.x - a.x, c.y - a.y, c.z - a.z};
        double tv[3] = {cfg.srcpos[0] - a.x, cfg.srcpos[1] - a.y, cfg.srcpos[2] - a.z}, pv[3], qv[3];
        cross3(d, e2, pv);
        double det = dot3(e1, pv);

        if (std::fabs(det) < 1e-12) {
            continue;
        }

        cross3(tv, e1, qv);
        double u = dot3(tv, pv) / det, v = dot3(d, qv) / det, t = dot3(e2, qv) / det;

        if (u >= 0 && v >= 0 && u + v <= 1 && std::fabs(t) <= VKMMC_SAFETY_DIST && t < tbest) {
            tbest = (float)t;
            best = i;
        }
    }

    if (tbest > VKMMC_SAFETY_DIST) {
        return;
    }

    const FaceData& f = cfg.facedata[best];
    uint32_t front, back;
    bool flat;
    unpack_media(f.packed_media, front, back, flat);
    bool ff = (f.nx * d[0] + f.ny * d[1] + f.nz * d[2]) < 0;   /* from the front side to the back side */
    float s = tbest + (((ff ? front : back) != 0) ? -VKMMC_SAFETY_DIST : VKMMC_SAFETY_DIST);

    for (int i = 0; i < 3; i++) {
        cfg.srcpos[i] += s * d[i];
    }
}

static SimSetup prepare_geometry(SimConfig& cfg, const CmdOverrides& ovr) {
    SimSetup st;

    if (cfg.is_csg) {
        float ext[6] = {0, 60, 0, 60, 0, 60};
        std::vector<ShapeOrigin> origins;
        vkmmc_require(cfg.media.size() >= 2, "CSG mode needs at least 2 media (0: background, 1..: shape tags)");
        ShapeMesh sm = parse_shapes(cfg.shapes, ext, cfg.mesh_res, (uint32_t)(cfg.media.size() - 1), &origins);
        vkmmc_require(!sm.faces.empty(), "Shapes: no surface was generated");
        cfg.nodes = sm.nodes;
        cfg.faces = sm.faces;
        cfg.facedata = sm.facedata;
        cfg.face_shape_id = sm.shape_id;
        update_bbox(cfg);

        bool curved = false;

        for (size_t i = 0; i < origins.size(); i++) {
            curved |= (origins[i].type != 0);
        }

        bool want = (ovr.docurv < 0) ? curved : (ovr.docurv != 0);

        if (want) {
            st.curv = compute_curvature(sm, origins);
            st.has_curvature = true;
            printf("Curvature: computed for %zu nodes, %zu shapes\n", st.curv.size(), origins.size());
        } else {
            printf("Curvature: disabled\n");
        }

        st.variant = st.has_curvature ? KV_CSG_CURV : KV_CSG;
    } else {
        vkmmc_require(ovr.docurv <= 0, "curvature (-c 1) requires a CSG shape input");
        st.variant = KV_MESH;

        /* labels were checked against the media count when the mesh was loaded,
           but -j may have changed the media list since */
        for (size_t i = 0; i < cfg.facedata.size(); i++) {
            uint32_t front, back;
            bool flat;
            unpack_media(cfg.facedata[i].packed_media, front, back, flat);
            vkmmc_require(front < cfg.media.size() && back < cfg.media.size(),
                          "mesh label " + std::to_string(std::max(front, back)) + " has no entry in Domain.Media");
        }

        if (cfg.mediumid0 != 0xFFFFFFFFu) {
            vkmmc_require(cfg.mediumid0 < cfg.media.size(), "the InitElem label has no entry in Domain.Media");
        }
    }

    vkmmc_require(cfg.faces.size() < (1u << 30), "the mesh has too many triangles");
    snap_pencil_source(cfg);
    return st;
}

// ============================================================================
// Output grid
// ============================================================================
struct OutputGrid {
    uint32_t nx, ny, nz;
    int maxgate;
    float vs;             /* voxel size in length units */
    float gmin[3], gmax[3];
    uint64_t nvox;        /* nx*ny*nz */
    uint64_t nelem;       /* nvox*maxgate: one field */
    uint32_t nfield;      /* fields: patterns of a shared pattern source or sources of a multi-source run */
    uint32_t nlaunch;     /* launched-energy sums after the fields: one per pattern */
    uint64_t nout;        /* kernel output elements: nelem*nfield + nlaunch */
};

/* voxel grid covering the mesh bounding box, half a voxel of margin per side */
static OutputGrid compute_grid(const SimConfig& cfg) {
    OutputGrid g;
    g.vs = cfg.has_steps ? cfg.steps[0] : 1.0f;
    g.maxgate = cfg.maxgate;
    vkmmc_require(!cfg.nodes.empty(), "the mesh has no nodes");

    float ge = g.vs * 0.5f;
    float lo[3] = {cfg.nmin.x, cfg.nmin.y, cfg.nmin.z}, hi[3] = {cfg.nmax.x, cfg.nmax.y, cfg.nmax.z};
    double n[3];

    for (int i = 0; i < 3; i++) {
        g.gmin[i] = lo[i] - ge;
        g.gmax[i] = hi[i] + ge;
        n[i] = std::ceil(((double)g.gmax[i] - g.gmin[i]) / g.vs);
        vkmmc_require(std::isfinite(n[i]) && n[i] >= 1 && n[i] < 1e7, "the output grid is too large; increase Domain.Step");
    }

    g.nx = (uint32_t)n[0];
    g.ny = (uint32_t)n[1];
    g.nz = (uint32_t)n[2];
    g.nvox = (uint64_t)g.nx * g.ny * g.nz;
    g.nelem = g.nvox * (uint64_t)g.maxgate;
    g.nfield = src_fields(cfg);
    g.nlaunch = (uint32_t)cfg.srcnum;
    g.nout = g.nelem * g.nfield + g.nlaunch;

    if (g.nelem * g.nfield > VKMMC_MAX_VOXELS) {
        std::ostringstream ss;
        ss << "the output grid " << g.nx << "x" << g.ny << "x" << g.nz << "x" << g.maxgate;

        if (g.nfield > 1) {
            ss << "x" << g.nfield;
        }

        ss << " exceeds " << VKMMC_MAX_VOXELS << " elements; increase Domain.Step or Forward.Dt";
        vkmmc_fail(ss.str());
    }

    return g;
}

/* media with mua/mus scaled from 1/mm to 1/length-unit, as the kernels use */
static std::vector<Medium> kernel_media(const SimConfig& cfg) {
    std::vector<Medium> m = cfg.media;

    for (size_t i = 0; i < m.size(); i++) {
        m[i].mua *= cfg.unitinmm;
        m[i].mus *= cfg.unitinmm;
    }

    return m;
}

/* the kernels accumulate time of flight in "length-unit time"; scale the gates to match */
static void kernel_time_gates(const SimConfig& cfg, float& tstart, float& tend, float& inv_dt) {
    double s = 1.0 / cfg.unitinmm;
    tstart = (float)(cfg.t0 * s);
    tend = (float)(cfg.t1 * s);
    inv_dt = (float)(1.0 / (cfg.dt * s));
}

/* the kernels' focus mode of a focal length: 0 none (0 or +Inf), 1 focused, 2 isotropic (NaN), 3 Lambertian (-Inf) */
static int kernel_focus_mode(float f) {
    if (std::isnan(f)) {
        return 2;
    }

    if (std::isinf(f)) {
        return (f < 0) ? 3 : 0;
    }

    return (f != 0) ? 1 : 0;
}

/*
 * The initial medium the kernels start every photon in, or 0xFFFFFFFF to
 * find it per photon by a ray probe along the launch direction. InitElem
 * applies to a single source; with random launch directions the probe is
 * used, as a source on the surface (e.g. an isotropic one) must lose the
 * photons pointing out of the mesh.
 */
static uint32_t kernel_initial_medium(const SimConfig& cfg) {
    int fm = kernel_focus_mode(cfg.srcdir[3]);
    bool randdir = cfg.srctype == SRC_ISOTROPIC || cfg.srctype == SRC_CONE || cfg.srctype == SRC_ARCSINE
                   || cfg.srctype == SRC_ZGAUSSIAN || cfg.srctype == SRC_LINE || fm == 2 || fm == 3;
    return (cfg.extrasrc.empty() && !randdir) ? cfg.mediumid0 : 0xFFFFFFFFu;
}

/* Fresnel reflection at the entry: DoSpecular, or for MMC's wide-field meshes,
   whose sources are in the -1 domain, the tissue surface as any boundary */
static bool kernel_specular(const SimConfig& cfg) {
    return cfg.do_specular || (cfg.wide_field && cfg.do_mismatch);
}

/* the kernels tally the launched energy only for sources with varying launch
   weights; for the others it is the photon count */
static bool kernel_tallies_launch(const SimConfig& cfg) {
    int t = cfg.srctype;
    return cfg.srcnum > 1 || !cfg.extrasrc.empty() || t == SRC_PATTERN || t == SRC_PATTERN3D || t == SRC_FOURIER
           || t == SRC_FOURIERX || t == SRC_FOURIERX2D;
}

/* the multi-source list as the kernels read it, 4 vec4 per source (a dummy if empty); a
   non-finite focal length is ignored there */
static std::vector<float> kernel_extrasrc(const SimConfig& cfg) {
    std::vector<float> v(std::max<size_t>(cfg.extrasrc.size(), 1) * 16, 0.0f);

    for (size_t i = 0; i < cfg.extrasrc.size(); i++) {
        memcpy(&v[i * 16], &cfg.extrasrc[i], 16 * sizeof(float));

        if (!std::isfinite(v[i * 16 + 7])) {
            v[i * 16 + 7] = 0;
        }
    }

    return v;
}

/* the pattern intensities as the kernels read them (a dummy if none) */
static std::vector<float> kernel_pattern(const SimConfig& cfg) {
    return cfg.srcpattern.empty() ? std::vector<float>(1, 0.0f) : cfg.srcpattern;
}

/* GPU thread count: a multiple of 64, at most one photon per thread for small runs */
static uint32_t thread_count(const SimConfig& cfg, const CmdOverrides& ovr) {
    uint64_t tt = (ovr.totalthread > 0) ? ovr.totalthread : 131072;

    if (cfg.nphoton < tt) {
        tt = cfg.nphoton;
    }

    tt = (tt + 63) / 64 * 64;
    return (uint32_t)std::max<uint64_t>(tt, 64);
}

#define VKMMC_BATCH_MS 500.0                  /* auto batches: target kernel time, below GPU watchdogs */
#define VKMMC_BATCH_MAX ((uint64_t)1 << 24)    /* auto batches: photons, keeps the fixed-point sums far from overflow */

/*
 * Photons in the next kernel launch, remaining still to run. -B n fixes the
 * batch (0: all at once). By default the first launch runs 4 photons per
 * thread and each later one is sized from the speed of the previous launch
 * (last photons in last_ms) to take about VKMMC_BATCH_MS, at most
 * VKMMC_BATCH_MAX photons. A photon's random
 * numbers depend only on its index, so the batch sizes do not change the
 * output.
 */
static uint64_t next_batch(const CmdOverrides& ovr, uint64_t remaining, uint32_t nthread, uint64_t last, double last_ms) {
    uint64_t n;

    if (ovr.batch_size != UINT64_MAX) {
        n = (ovr.batch_size == 0) ? remaining : ovr.batch_size;
    } else if (last == 0) {
        n = (uint64_t)nthread * 4;
    } else {
        double rate = (double)last / std::max(last_ms, 1e-3);
        n = (uint64_t)std::min(rate * VKMMC_BATCH_MS, (double)last * 8);
        n = std::min(std::max(n, (uint64_t)nthread), VKMMC_BATCH_MAX);

        if (remaining - std::min(n, remaining) < n / 4) {
            n = remaining;     /* no short last launch */
        }
    }

    /* the kernels claim photons with a 32-bit counter, which every thread passes once at the end */
    return std::min(std::min(n, remaining), (uint64_t)0xFFFFFFFFu - nthread);
}

/* base RNG seed; 0 picks one from the clock */
static uint32_t base_seed(const SimConfig& cfg) {
    return (cfg.rng_seed > 0) ? cfg.rng_seed : (uint32_t)time(0);
}

// ============================================================================
// Results
// ============================================================================
struct TrajData {
    std::vector<uint32_t> ids, evts;
    std::vector<float> pts, w;
};

#define VKMMC_FIXED_SCALE 4294967296.0   /* kernel output: 64-bit fixed point, 2^32 per unit weight */

/*
 * Add one batch of kernel output (nelem unsigned 64-bit fixed-point sums) to
 * the double-precision accumulator acc. Reading back and clearing the GPU
 * buffer after every batch keeps the integer sums far from overflow.
 */
static void accumulate_fixed(std::vector<double>& acc, const uint64_t* raw, uint64_t nelem) {
    acc.resize(nelem, 0.0);

    for (uint64_t i = 0; i < nelem; i++) {
        acc[i] += (double)raw[i] * (1.0 / VKMMC_FIXED_SCALE);
    }
}

/* read back the output after every batch when the volume is at most this large */
#define VKMMC_BATCH_READBACK_BYTES ((uint64_t)512 << 20)

/*
 * Zero NaN/Inf, report the absorbed fraction and normalize the accumulated
 * output (acc: g.nelem values per field, then the launched energy of each
 * pattern, tallied by the kernels for sources with varying launch weights;
 * otherwise every photon is taken to start with weight 1).
 * Each field is normalized by the launched energy of its pattern; the fields
 * of a multi-source run share the total, as in MMC.
 *   energy (x):  1/mm^3 per unit of launched energy
 *   fluence (l): 1/mm^2 per unit, integrated over each time gate
 *   flux (f):    1/(mm^2 s) per unit, fluence divided by the gate width
 */
static std::vector<float> finalize_output(const SimConfig& cfg, const OutputGrid& g, const std::vector<double>& acc,
        double* absorbed_out = NULL) {
    uint64_t nvol = g.nelem * g.nfield;
    std::vector<float> out(nvol);
    std::vector<double> launched(g.nlaunch, (double)cfg.nphoton / g.nlaunch), fieldsum(g.nfield, 0.0);
    double ltot = 0;
    size_t bad = 0;

    if (acc.size() >= nvol + g.nlaunch && kernel_tallies_launch(cfg)) {
        for (uint32_t k = 0; k < g.nlaunch; k++) {
            launched[k] = acc[nvol + k];
        }
    }

    for (uint32_t k = 0; k < g.nlaunch; k++) {
        ltot += launched[k];
    }

    double unit = 1.0;

    if (cfg.do_normalize) {
        double vmm = (double)g.vs * cfg.unitinmm;
        unit = 1.0 / (vmm * vmm * vmm);

        if (cfg.output_type != OT_ENERGY) {
            unit *= cfg.unitinmm;    /* path length: length units -> mm */
        }

        if (cfg.output_type == OT_FLUX) {
            unit /= cfg.dt;
        }
    }

    for (uint32_t f = 0; f < g.nfield; f++) {
        double e = (g.nlaunch > 1) ? launched[f] : ltot, scale = unit;

        if (cfg.do_normalize) {
            scale = (e > 0) ? unit / e : 0.0;
        }

        for (uint64_t i = f * g.nelem; i < (f + 1) * g.nelem; i++) {
            double v = (i < acc.size()) ? acc[i] : 0.0;

            if (!std::isfinite(v)) {
                bad++;
                v = 0;
            }

            fieldsum[f] += v;
            out[i] = (float)(v * scale);
        }
    }

    double total = 0;

    for (uint32_t f = 0; f < g.nfield; f++) {
        total += fieldsum[f];
    }

    if (bad) {
        printf("WARNING: %zu NaN/Inf voxels zeroed\n", bad);
    }

    double absorbed = (ltot > 0) ? total / ltot : 0.0;

    if (cfg.output_type == OT_ENERGY) {
        printf("simulated %llu photons, absorbed: %.5f%%\n", (unsigned long long)cfg.nphoton, absorbed * 100.0);

        if (ltot != (double)cfg.nphoton) {
            printf("  launched energy: %.6g\n", ltot);
        }

        for (uint32_t f = 0; g.nfield > 1 && f < g.nfield; f++) {
            double e = (g.nlaunch > 1) ? launched[f] : ltot;
            printf("  %s %u: absorbed: %.5f%%\n", (g.nlaunch > 1) ? "pattern" : "source", f + 1,
                   (e > 0) ? fieldsum[f] / e * 100.0 : 0.0);
        }
    } else {
        printf("simulated %llu photons\n", (unsigned long long)cfg.nphoton);
    }

    if (absorbed_out) {
        *absorbed_out = (cfg.output_type == OT_ENERGY) ? absorbed : -1;
    }

    return out;
}

static const char* output_type_name(int t) {
    return t == OT_FLUX ? "flux" : (t == OT_FLUENCE ? "fluence" : "energy");
}

static void write_json_file(const std::string& fname, const json& root, int indent) {
    std::ofstream of(fname.c_str());
    vkmmc_require((bool)of, "cannot write output file " + fname);
    of << root.dump(indent) << std::endl;
    vkmmc_require((bool)of, "failed to write output file " + fname);
}

/* save the volume as <session>.jdat; data is x-fastest, i.e. column-major [nx,ny,nz,nt] */
static void save_fluence(const SimConfig& cfg, const OutputGrid& g, const std::vector<float>& vol) {
    std::vector<size_t> dims = {g.nx, g.ny, g.nz};

    if (g.maxgate > 1 || g.nfield > 1) {
        dims.push_back((size_t)g.maxgate);
    }

    if (g.nfield > 1) {
        dims.push_back((size_t)g.nfield);   /* one volume per pattern or source, as MMC saves them */
    }

    json root;
    root["Session"] = {{"ID", cfg.session_id}, {"Photons", cfg.nphoton}, {"OutputType", output_type_name(cfg.output_type)},
        {"DoNormalize", cfg.do_normalize}, {"Version", VKMMC_VERSION}
    };
    root["Forward"] = {{"T0", cfg.t0}, {"T1", cfg.t1}, {"Dt", cfg.dt}};
    root["Domain"] = {{"LengthUnit", cfg.unitinmm}, {"VoxelSize", g.vs}, {"Dim", {g.nx, g.ny, g.nz}},
        {"Origin", {g.gmin[0], g.gmin[1], g.gmin[2]}}
    };
    root["Fluence"] = jdata_encode("single", dims, vol.data(), vol.size() * sizeof(float), false);
    std::string outname = cfg.session_id + ".jdat";
    write_json_file(outname, root, 2);
    printf("Output: %s (%ux%ux%u", outname.c_str(), g.nx, g.ny, g.nz);

    for (size_t i = 3; i < dims.size(); i++) {
        printf("x%zu", dims[i]);
    }

    printf(")\n");
}

/* save trajectory events as <session>_traj.jdat in the MCX trajectory layout */
static inline void save_trajectory(const SimConfig& cfg, const TrajData& t) {
    uint32_t n = (uint32_t)t.ids.size();
    json troot;
    json& mdata = troot["MCXData"];
    json media_arr = json::array();

    for (size_t i = 0; i < cfg.media.size(); i++) {
        const Medium& m = cfg.media[i];
        media_arr.push_back({{"mua", m.mua}, {"mus", m.mus}, {"g", m.g}, {"n", m.n}});
    }

    mdata["Info"] = {
        {"Version",        1},
        {"MediaNum",       (int)cfg.media.size()},
        {"DetNum",         0},
        {"ColumnNum",      5},
        {"TotalPhoton",    cfg.nphoton},
        {"DetectedPhoton", n},
        {"SavedPhoton",    n},
        {"LengthUnit",     cfg.unitinmm},
        {"SeedByte",       0},
        {"Normalizer",     1.0},
        {"Repeat",         1},
        {"SrcNum",         1},
        {"SaveDetFlag",    0},
        {"Media",          media_arr}
    };

    json& tdat = mdata["Trajectory"];
    tdat["photonid"] = jdata_encode("uint32", {n, 1}, t.ids.data(), n * sizeof(uint32_t));
    tdat["p"]        = jdata_encode("single", {n, 3}, t.pts.data(), (size_t)n * 3 * sizeof(float));
    tdat["w0"]       = jdata_encode("single", {n, 1}, t.w.data(), n * sizeof(float));
    tdat["e0"]       = jdata_encode("uint32", {n, 1}, t.evts.data(), n * sizeof(uint32_t));

    std::string tname = cfg.session_id + "_traj.jdat";
    write_json_file(tname, troot, -1);
    printf("Trajectory saved: %s\n", tname.c_str());
}

/* unpack kernel trajectory records: 6 uint32 per event (id, x, y, z, w, event) */
static inline TrajData unpack_trajectory(const uint32_t* rec, uint32_t n) {
    TrajData t;
    t.ids.resize(n);
    t.evts.resize(n);
    t.pts.resize((size_t)n * 3);
    t.w.resize(n);

    for (uint32_t i = 0; i < n; i++) {
        const uint32_t* r = rec + (size_t)i * 6;
        t.ids[i] = r[0];
        memcpy(&t.pts[(size_t)i * 3], &r[1], 12);
        memcpy(&t.w[i], &r[4], 4);
        t.evts[i] = r[5];
    }

    return t;
}

// ============================================================================
// --dumpjson / --dumpmesh / --debugcurv
// ============================================================================

/* a focal length; NaN and Inf as the JData strings the input accepts */
static json focal_json(float f) {
    if (std::isnan(f)) {
        return "_NaN_";
    }

    if (std::isinf(f)) {
        return (f > 0) ? "_Inf_" : "-_Inf_";
    }

    return f;
}

/* Optode.Source as an input file has it; with_pattern adds the pattern data, else its size */
static json source_json(const SimConfig& cfg, bool with_pattern) {
    json o;
    o["Type"] = srctype_name(cfg.srctype);

    if (cfg.extrasrc.empty()) {
        o["Pos"] = {cfg.srcpos[0], cfg.srcpos[1], cfg.srcpos[2]};
        o["Dir"] = {cfg.srcdir[0], cfg.srcdir[1], cfg.srcdir[2], focal_json(cfg.srcdir[3])};
        o["Param1"] = {cfg.srcparam1[0], cfg.srcparam1[1], cfg.srcparam1[2], cfg.srcparam1[3]};
        o["Param2"] = {cfg.srcparam2[0], cfg.srcparam2[1], cfg.srcparam2[2], cfg.srcparam2[3]};
    } else {
        const char* key[4] = {"Pos", "Dir", "Param1", "Param2"};

        for (int k = 0; k < 4; k++) {
            o[key[k]] = json::array();
        }

        for (size_t i = 0; i < cfg.extrasrc.size(); i++) {
            const ExtraSrc& e = cfg.extrasrc[i];
            o["Pos"].push_back({e.pos[0], e.pos[1], e.pos[2], e.pos[3]});
            o["Dir"].push_back({e.dir[0], e.dir[1], e.dir[2], focal_json(e.dir[3])});
            o["Param1"].push_back({e.param1[0], e.param1[1], e.param1[2], e.param1[3]});
            o["Param2"].push_back({e.param2[0], e.param2[1], e.param2[2], e.param2[3]});
        }

        o["ID"] = cfg.srcid < 0 ? 0 : cfg.srcid;
    }

    o["SrcNum"] = cfg.srcnum;

    if (!cfg.srcpattern.empty()) {
        std::vector<size_t> dims;

        if (cfg.srcnum > 1) {
            dims.push_back((size_t)cfg.srcnum);
        }

        if (cfg.srctype == SRC_PATTERN) {
            dims.push_back((size_t)cfg.srcparam1[3]);
            dims.push_back((size_t)cfg.srcparam2[3]);
        } else {
            for (int i = 0; i < 3; i++) {
                dims.push_back((size_t)cfg.srcparam1[i]);
            }
        }

        if (with_pattern) {
            /* stored with the first index fastest */
            o["Pattern"] = jdata_encode("single", dims, cfg.srcpattern.data(), cfg.srcpattern.size() * sizeof(float), false);
        } else {
            o["PatternSize"] = dims;
        }
    }

    return o;
}

static void dump_config_json(const SimConfig& cfg, const SimSetup& st) {
    json d;
    d["Session"] = {{"ID", cfg.session_id}, {"Photons", cfg.nphoton}, {"RNGSeed", cfg.rng_seed},
        {"DoMismatch", cfg.do_mismatch}, {"DoSpecular", cfg.do_specular}, {"DoNormalize", cfg.do_normalize},
        {"OutputType", output_type_name(cfg.output_type)}, {"DebugFlag", cfg.debuglevel}
    };
    d["Forward"] = {{"T0", cfg.t0}, {"T1", cfg.t1}, {"Dt", cfg.dt}, {"Gates", cfg.maxgate}};
    d["Domain"]["LengthUnit"] = cfg.unitinmm;
    d["Domain"]["Media"] = json::array();

    for (size_t i = 0; i < cfg.media.size(); i++) {
        d["Domain"]["Media"].push_back({{"mua", cfg.media[i].mua}, {"mus", cfg.media[i].mus},
            {"g", cfg.media[i].g}, {"n", cfg.media[i].n}
        });
    }

    d["Optode"]["Source"] = source_json(cfg, false);
    d["Mesh"] = {{"Mode", cfg.is_csg ? "csg" : "mesh"}, {"Nodes", cfg.nodes.size()}, {"Faces", cfg.faces.size()},
        {"Kernel", kernel_variant_name(st.variant)}, {"Curvature", st.has_curvature},
        {"BBoxMin", {cfg.nmin.x, cfg.nmin.y, cfg.nmin.z}}, {"BBoxMax", {cfg.nmax.x, cfg.nmax.y, cfg.nmax.z}}
    };
    printf("%s\n", d.dump(2).c_str());
}

/* save the surface mesh as <session>_mesh.json, a mesh-mode input file */
static void dump_mesh(const SimConfig& cfg) {
    size_t nn = cfg.nodes.size(), nf = cfg.faces.size();
    vkmmc_require(nn > 0 && nf > 0, "no mesh to dump");

    json dump;
    dump["Session"] = {{"ID", cfg.session_id}, {"Photons", cfg.nphoton}, {"DoMismatch", cfg.do_mismatch},
        {"DoSpecular", cfg.do_specular}
    };
    dump["Forward"] = {{"T0", cfg.t0}, {"T1", cfg.t1}, {"Dt", cfg.dt}};
    dump["Domain"]["LengthUnit"] = cfg.unitinmm;
    dump["Domain"]["Media"] = json::array();

    for (size_t i = 0; i < cfg.media.size(); i++) {
        dump["Domain"]["Media"].push_back({{"mua", cfg.media[i].mua}, {"mus", cfg.media[i].mus},
            {"g", cfg.media[i].g}, {"n", cfg.media[i].n}
        });
    }

    dump["Optode"]["Source"] = source_json(cfg, true);

    std::vector<float> nd(nn * 3);

    for (size_t i = 0; i < nn; i++) {
        nd[i * 3] = cfg.nodes[i].x;
        nd[i * 3 + 1] = cfg.nodes[i].y;
        nd[i * 3 + 2] = cfg.nodes[i].z;
    }

    dump["Shapes"]["MeshNode"] = jdata_encode("single", {nn, 3}, nd.data(), nn * 3 * sizeof(float));

    /* MeshSurf rows [v1,v2,v3,label]: an exterior face is listed once with its
       back label; an interface is listed twice, once per side, which is how
       load_mesh_surf() recovers both labels */
    std::vector<int32_t> sd;
    sd.reserve(nf * 4);

    for (size_t i = 0; i < nf; i++) {
        uint32_t front, back;
        bool flat;
        unpack_media(cfg.facedata[i].packed_media, front, back, flat);
        int32_t v[3] = {(int32_t)cfg.faces[i][0] + 1, (int32_t)cfg.faces[i][1] + 1, (int32_t)cfg.faces[i][2] + 1};
        sd.insert(sd.end(), {v[0], v[1], v[2], (int32_t)back});

        if (front != 0) {
            sd.insert(sd.end(), {v[0], v[1], v[2], (int32_t)front});
        }
    }

    dump["Shapes"]["MeshSurf"] = jdata_encode("int32", {sd.size() / 4, 4}, sd.data(), sd.size() * sizeof(int32_t));
    std::string mf = cfg.session_id + "_mesh.json";
    write_json_file(mf, dump, 2);
    printf("Mesh saved to %s (%zu nodes, %zu faces, %s mode)\n", mf.c_str(), nn, nf, cfg.is_csg ? "CSG" : "mesh");
}

/* curvature-predicted normal at point P on triangle fi (same formula as the kernels) */
static Vec3 curvature_normal_at(const SimConfig& cfg, const std::vector<NodeCurvature>& curv, Vec3 P, size_t fi) {
    const FaceData& fd = cfg.facedata[fi];

    if (face_is_flat(fd)) {
        return {fd.nx, fd.ny, fd.nz};
    }

    double NB[3] = {0, 0, 0};

    for (int vi = 0; vi < 3; vi++) {
        uint32_t idx = cfg.faces[fi][vi];
        const NodeCurvature& c = curv[idx];
        float vdir[3] = {c.py* c.nz - c.pz * c.ny, c.pz* c.nx - c.px * c.nz, c.px* c.ny - c.py * c.nx};
        float di[3] = {P.x - cfg.nodes[idx].x, P.y - cfg.nodes[idx].y, P.z - cfg.nodes[idx].z};
        float du = di[0] * c.px + di[1] * c.py + di[2] * c.pz;
        float dv = di[0] * vdir[0] + di[1] * vdir[1] + di[2] * vdir[2];
        NB[0] += c.nx + c.k1 * du * c.px + c.k2 * dv * vdir[0];
        NB[1] += c.ny + c.k1 * du * c.py + c.k2 * dv * vdir[1];
        NB[2] += c.nz + c.k1 * du * c.pz + c.k2 * dv * vdir[2];
    }

    double l = std::sqrt(NB[0] * NB[0] + NB[1] * NB[1] + NB[2] * NB[2]);

    if (l > 1e-10) {
        NB[0] /= l;
        NB[1] /= l;
        NB[2] /= l;
    }

    if (NB[0] * fd.nx + NB[1] * fd.ny + NB[2] * fd.nz < 0) {
        NB[0] = -NB[0];
        NB[1] = -NB[1];
        NB[2] = -NB[2];
    }

    return {(float)NB[0], (float)NB[1], (float)NB[2]};
}

/* analytic outward normal of shape so at P; fallback if undefined */
static Vec3 analytic_normal(const ShapeOrigin& so, Vec3 P, Vec3 fallback) {
    double dx = P.x - so.cx, dy = P.y - so.cy, dz = P.z - so.cz;

    if (so.type == 2) {
        double proj = dx * so.ax + dy * so.ay + dz * so.az;
        dx -= proj * so.ax;
        dy -= proj * so.ay;
        dz -= proj * so.az;
    } else if (so.type != 1 && so.type != 3) {
        return fallback;
    }

    double l = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (l < 1e-10) {
        return fallback;
    }

    double s = (so.type == 3 && so.R < 0) ? -1.0 / l : 1.0 / l;
    return {(float)(dx * s), (float)(dy * s), (float)(dz * s)};
}

/*
 * --debugcurv: compare, at the nodes of a 4x finer mesh, the curvature
 * corrected normal and the flat face normal of the coarse mesh against the
 * analytic normal; saves <session>_curv.json (MeshNode columns: x, y, z,
 * curvature error, flat error; error = 1 - cos(angle)).
 */
static void run_debugcurv(const SimConfig& cfg, const SimSetup& st) {
    vkmmc_require(cfg.is_csg && st.has_curvature && !st.curv.empty(), "--debugcurv requires a CSG input with curvature (-c 1)");

    std::vector<ShapeOrigin> origins;
    float ext[6] = {0, 60, 0, 60, 0, 60};
    int fine_res = std::min(cfg.mesh_res * 4, VKMMC_MAX_MESHRES);
    ShapeMesh fine = parse_shapes(cfg.shapes, ext, fine_res, (uint32_t)(cfg.media.size() - 1), &origins);
    size_t nn_fine = fine.nodes.size(), nf_fine = fine.faces.size();
    printf("debugcurv: coarse mesh = %zu nodes, %zu faces (mesh_res=%d)\n", cfg.nodes.size(), cfg.faces.size(), cfg.mesh_res);
    printf("debugcurv: fine mesh = %zu nodes, %zu faces (mesh_res=%d)\n", nn_fine, nf_fine, fine_res);

    std::vector<int> fine_node_shape(nn_fine, -1);

    for (size_t fi = 0; fi < nf_fine; fi++) {
        int si = (fi < fine.shape_id.size()) ? (int)fine.shape_id[fi] - 1 : -1;

        for (int k = 0; k < 3; k++) {
            uint32_t ni = fine.faces[fi][k];

            if (ni < nn_fine && fine_node_shape[ni] < 0) {
                fine_node_shape[ni] = si;
            }
        }
    }

    /* coarse triangles grouped by shape, with centroids */
    std::map<int, std::vector<size_t> > byshape;
    std::vector<Vec3> cent(cfg.faces.size());

    for (size_t fi = 0; fi < cfg.faces.size(); fi++) {
        int csi = (fi < cfg.face_shape_id.size()) ? (int)cfg.face_shape_id[fi] - 1 : -1;
        byshape[csi].push_back(fi);
        Vec3 A = cfg.nodes[cfg.faces[fi][0]], B = cfg.nodes[cfg.faces[fi][1]], C = cfg.nodes[cfg.faces[fi][2]];
        cent[fi] = {(A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3, (A.z + B.z + C.z) / 3};
    }

    auto bary_err = [&](Vec3 P, size_t fi) -> float {
        Vec3 A = cfg.nodes[cfg.faces[fi][0]], B = cfg.nodes[cfg.faces[fi][1]], C = cfg.nodes[cfg.faces[fi][2]];
        double v0[3] = {B.x - A.x, B.y - A.y, B.z - A.z}, v1[3] = {C.x - A.x, C.y - A.y, C.z - A.z};
        double v2[3] = {P.x - A.x, P.y - A.y, P.z - A.z};
        double d00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2], d01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
        double d11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2], d20 = v2[0] * v0[0] + v2[1] * v0[1] + v2[2] * v0[2];
        double d21 = v2[0] * v1[0] + v2[1] * v1[1] + v2[2] * v1[2], den = d00 * d11 - d01 * d01;

        if (std::fabs(den) < 1e-20) {
            return 1e30f;
        }

        double v = (d11 * d20 - d01 * d21) / den, w = (d00 * d21 - d01 * d20) / den, u = 1 - v - w;
        return (float)((u < 0 ? u * u : 0) + (v < 0 ? v * v : 0) + (w < 0 ? w * w : 0));
    };

    std::vector<float> out_nodes(nn_fine * 5, 0.f);
    double curv_sum = 0, curv_max = 0, flat_sum = 0, flat_max = 0;
    int counted = 0;

    for (size_t ni = 0; ni < nn_fine; ni++) {
        Vec3 P = fine.nodes[ni];
        int si = fine_node_shape[ni];
        out_nodes[ni * 5 + 0] = P.x;
        out_nodes[ni * 5 + 1] = P.y;
        out_nodes[ni * 5 + 2] = P.z;

        std::map<int, std::vector<size_t> >::iterator it = byshape.find(si);

        if (si < 0 || si >= (int)origins.size() || it == byshape.end()) {
            continue;
        }

        /* nearest coarse centroid of the same shape, refined by barycentric containment */
        int best = -1;
        float bestd = 1e30f;

        for (size_t k = 0; k < it->second.size(); k++) {
            size_t fi = it->second[k];
            float dx = P.x - cent[fi].x, dy = P.y - cent[fi].y, dz = P.z - cent[fi].z, d2 = dx * dx + dy * dy + dz * dz;

            if (d2 < bestd) {
                bestd = d2;
                best = (int)fi;
            }
        }

        float r2 = bestd * 4.0f + 1.0f, beste = 1e30f;
        int refined = best;

        for (size_t k = 0; k < it->second.size(); k++) {
            size_t fi = it->second[k];
            float dx = P.x - cent[fi].x, dy = P.y - cent[fi].y, dz = P.z - cent[fi].z;

            if (dx * dx + dy * dy + dz * dz > r2) {
                continue;
            }

            float e = bary_err(P, fi);

            if (e < beste) {
                beste = e;
                refined = (int)fi;
            }
        }

        best = refined;
        const FaceData& fd = cfg.facedata[best];
        Vec3 Nf = {fd.nx, fd.ny, fd.nz};
        Vec3 Na = analytic_normal(origins[si], P, Nf);
        Vec3 Nc = curvature_normal_at(cfg, st.curv, P, (size_t)best);

        if (Na.x * Nf.x + Na.y * Nf.y + Na.z * Nf.z < 0) {
            Na.x = -Na.x;
            Na.y = -Na.y;
            Na.z = -Na.z;
        }

        float cerr = 1.0f - std::min(1.0f, std::max(-1.0f, Nc.x * Na.x + Nc.y * Na.y + Nc.z * Na.z));
        float ferr = 1.0f - std::min(1.0f, std::max(-1.0f, Nf.x * Na.x + Nf.y * Na.y + Nf.z * Na.z));
        out_nodes[ni * 5 + 3] = cerr;
        out_nodes[ni * 5 + 4] = ferr;
        curv_sum += cerr;
        flat_sum += ferr;
        curv_max = std::max(curv_max, (double)cerr);
        flat_max = std::max(flat_max, (double)ferr);
        counted++;
    }

    vkmmc_require(counted > 0, "--debugcurv: no curved surface nodes to evaluate");
    printf("debugcurv: %d nodes evaluated\n", counted);
    printf("  curvature normal error - mean=%.6e  max=%.6e\n", curv_sum / counted, curv_max);
    printf("  flat normal error      - mean=%.6e  max=%.6e\n", flat_sum / counted, flat_max);
    printf("  improvement ratio      - mean=%.2fx  max=%.2fx\n",
           (flat_sum / counted) / (curv_sum / counted + 1e-30), flat_max / (curv_max + 1e-30));

    json dump;
    dump["Session"] = {{"ID", cfg.session_id + "_curv"}};
    dump["CoarseMesh"] = {{"Nodes", cfg.nodes.size()}, {"Faces", cfg.faces.size()}, {"MeshRes", cfg.mesh_res}};
    dump["FineMesh"] = {{"Nodes", nn_fine}, {"Faces", nf_fine}, {"MeshRes", fine_res}};
    dump["ErrorStats"] = {{"CurvMean", curv_sum / counted}, {"CurvMax", curv_max},
        {"FlatMean", flat_sum / counted}, {"FlatMax", flat_max}
    };
    dump["Shapes"]["MeshNode"] = jdata_encode("single", {nn_fine, 5}, out_nodes.data(), nn_fine * 5 * sizeof(float));

    std::vector<int32_t> surf(nf_fine * 4);

    for (size_t fi = 0; fi < nf_fine; fi++) {
        surf[fi * 4 + 0] = (int32_t)(fine.faces[fi][0] + 1);
        surf[fi * 4 + 1] = (int32_t)(fine.faces[fi][1] + 1);
        surf[fi * 4 + 2] = (int32_t)(fine.faces[fi][2] + 1);
        surf[fi * 4 + 3] = (fi < fine.shape_id.size()) ? (int32_t)fine.shape_id[fi] : 1;
    }

    dump["Shapes"]["MeshSurf"] = jdata_encode("int32", {nf_fine, 4}, surf.data(), nf_fine * 4 * sizeof(int32_t));
    std::string outf = cfg.session_id + "_curv.json";
    write_json_file(outf, dump, 2);
    printf("debugcurv: saved to %s\n", outf.c_str());
}

/*
 * Common front-end: parse the command line, load and validate the input and
 * build the geometry. Returns 1 to run a simulation, 0 to exit successfully
 * (help, version, dump commands); throws on error.
 */
static int vkmmc_frontend(int argc, char** argv, const char* backend, int defgpu, const char* kernel_ext,
                          CmdOverrides& ovr, SimConfig& cfg, SimSetup& st) {
    if (argc < 2) {
        printhelp(argv[0], backend, defgpu);
        return 0;
    }

    if (!parse_cmdline(argc, argv, ovr, backend, defgpu, kernel_ext)) {
        return 0;
    }

    if (ovr.listgpu || ovr.inputfile.empty()) {
        return 1;
    }

    cfg = load_json_config(load_json_file(ovr.inputfile.c_str(), ovr.json_str));
    apply_overrides(cfg, ovr);
    st = prepare_geometry(cfg, ovr);

    if (ovr.dumpjson) {
        dump_config_json(cfg, st);
        return 0;
    }

    if (ovr.dumpmesh) {
        dump_mesh(cfg);
        return 0;
    }

    if (ovr.debugcurv) {
        run_debugcurv(cfg, st);
        return 0;
    }

    return 1;
}

#endif /* VKMMC_HOST_H */
