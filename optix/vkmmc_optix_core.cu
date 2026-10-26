/*
 * vkmmc_optix_core.cu — OptiX 7 MC photon transport
 *
 * Compile-time flags:
 *   USE_CSG       — enable CSG shape-based media state tracking
 *   USE_CURVATURE — enable curvature-corrected normals (requires USE_CSG)
 */

#include <optix.h>
#include <optix_device.h>
#include "vkmmc_vec_math.h"
#include <limits>
#include <math.h>
#include <stdint.h>

#if defined(USE_CURVATURE) && !defined(USE_CSG)
    #error "USE_CURVATURE requires USE_CSG"
#endif

struct Medium {
    float mua, mus, g, n;
};

#ifdef USE_CURVATURE
struct NodeCurv {
    float4 vnorm_k1;
    float4 pdir_k2;
    float4 node_pos;
};
#endif

#define MAX_PROP 256

struct VKMMCParam {
    OptixTraversableHandle gashandle;
    CUdeviceptr facebuf;
    CUdeviceptr curvbuf;
    CUdeviceptr outputbuf;
    CUdeviceptr seedbuf;
    int srctype;
    float3 srcpos, srcdir;
    float4 srcparam1, srcparam2;
    float3 grid_min, grid_extent;
    uint4 grid_stride;
    float voxel_scale, tstart, tend, inv_timestep;
    int maxgate;
    unsigned int initial_medium, do_reflect;
    int output_type;
    unsigned int num_media;
    unsigned int do_csg;
    unsigned int has_curvature;
    int threadphoton, oddphoton;
    float minenergy, roulettesize;
    CUdeviceptr patternbuf;       /* pattern source: src_num intensities per pixel */
    CUdeviceptr extrasrcbuf;      /* multi-source list: 4 float4 per source */
    CUdeviceptr slotbuf;          /* per thread: the output field or pattern pixel of its photon */
    unsigned int src_num;         /* patterns sharing the photons */
    int src_id;                   /* multi-source: -1 a random source per photon, k > 0 source k */
    unsigned int extra_src_len, num_field;
    float focal;                  /* focal length (focus_mode 1) */
    int focus_mode;               /* 0 none, 1 focused, 2 isotropic, 3 Lambertian */
    unsigned int do_tally;        /* tally the launched energy (sources with varying weights) */
    unsigned int do_specular;     /* Fresnel reflection where a photon from outside enters */
    Medium media[MAX_PROP];
};

extern "C" {
    __constant__ VKMMCParam gcfg;
}

#define INV_C0 3.335640951981520e-12f
#define SAFETY_DIST 0.001f
#define TWO_PI 6.28318530717959f
#define ONE_PI 3.14159265358979f
#define FEPS 1.19209290E-07f
#define MEDIUM_UNKNOWN 0xFFFFFFFFu
#define MEDIUM_DEAD 0xFFFFFFFEu
#define MEDIUM_AMBIENT 0u
#define MEDIUM_PROBE 0xFFFFFFFDu      /* probe ray: report the hit, no transport */
#define MEDIUM_PROBE_HIT 0xFFFFFFFCu
#define RAY_TMIN 1e-5f
#define OUT_FLUX 0
#define OUT_FLUENCE 1
#define OUT_ENERGY 2

#ifdef USE_CURVATURE
    #define FLAT_FLAG 0x80000000u
#endif

/* ================================================================ */
/*                             RNG                                  */
/* ================================================================ */

__device__ __forceinline__ float rand01(uint4& st) {
    union {
        unsigned long long i;
        float f[2];
        unsigned int u[2];
    } s1;
    const unsigned long long s0 = ((unsigned long long)st.z << 32) | st.w;
    s1.i = ((unsigned long long)st.x << 32) | st.y;
    st.x = st.z;
    st.y = st.w;
    s1.i ^= s1.i << 23;
    unsigned long long ns = s1.i ^ s0 ^ (s1.i >> 18) ^ (s0 >> 5);
    st.z = (unsigned int)(ns >> 32);
    st.w = (unsigned int)ns;
    s1.i = ns + s0;
    s1.u[0] = 0x3F800000U | (s1.u[0] >> 9);
    return s1.f[0] - 1.0f;
}

__device__ __forceinline__ float rand_scatlen(uint4& rng) {
    return -logf(rand01(rng) + FEPS);
}

/* ================================================================ */
/*                           Payload                                */
/* ================================================================ */

struct Photon {
    float3 p0, dir;
    float slen, weight, tof;
    unsigned int mid;
#ifdef USE_CSG
    uint2 mstate;
    unsigned int pending;   /* crossed a coplanar triangle: raygen re-derives mstate */
    unsigned int primid;    /* ... and that triangle */
#endif
};

__device__ __forceinline__ Photon getPhoton() {
    Photon ph;
    ph.p0 = make_float3(__uint_as_float(optixGetPayload_0()),
                        __uint_as_float(optixGetPayload_1()),
                        __uint_as_float(optixGetPayload_2()));
    ph.dir = make_float3(__uint_as_float(optixGetPayload_3()),
                         __uint_as_float(optixGetPayload_4()),
                         __uint_as_float(optixGetPayload_5()));
    ph.slen = __uint_as_float(optixGetPayload_6());
    ph.weight = __uint_as_float(optixGetPayload_7());
    ph.tof = __uint_as_float(optixGetPayload_8());
    ph.mid = optixGetPayload_9();
#ifdef USE_CSG
    ph.mstate = make_uint2(optixGetPayload_10(), optixGetPayload_11());
    ph.pending = optixGetPayload_16();
    ph.primid = optixGetPayload_17();
#endif
    return ph;
}

__device__ __forceinline__ void setPhoton(const Photon& ph) {
    optixSetPayload_0(__float_as_uint(ph.p0.x));
    optixSetPayload_1(__float_as_uint(ph.p0.y));
    optixSetPayload_2(__float_as_uint(ph.p0.z));
    optixSetPayload_3(__float_as_uint(ph.dir.x));
    optixSetPayload_4(__float_as_uint(ph.dir.y));
    optixSetPayload_5(__float_as_uint(ph.dir.z));
    optixSetPayload_6(__float_as_uint(ph.slen));
    optixSetPayload_7(__float_as_uint(ph.weight));
    optixSetPayload_8(__float_as_uint(ph.tof));
    optixSetPayload_9(ph.mid);
#ifdef USE_CSG
    optixSetPayload_10(ph.mstate.x);
    optixSetPayload_11(ph.mstate.y);
    optixSetPayload_16(ph.pending);
    optixSetPayload_17(ph.primid);
#endif
}

__device__ __forceinline__ uint4 getRNG() {
#ifdef USE_CSG
    return make_uint4(optixGetPayload_12(), optixGetPayload_13(),
                      optixGetPayload_14(), optixGetPayload_15());
#else
    return make_uint4(optixGetPayload_10(), optixGetPayload_11(),
                      optixGetPayload_12(), optixGetPayload_13());
#endif
}

__device__ __forceinline__ void setRNG(const uint4& s) {
#ifdef USE_CSG
    optixSetPayload_12(s.x);
    optixSetPayload_13(s.y);
    optixSetPayload_14(s.z);
    optixSetPayload_15(s.w);
#else
    optixSetPayload_10(s.x);
    optixSetPayload_11(s.y);
    optixSetPayload_12(s.z);
    optixSetPayload_13(s.w);
#endif
}

__device__ __forceinline__ void traceRay(Photon& ph, uint4& rng, float tmax, float tmin = 0.0f) {
    unsigned int p0 = __float_as_uint(ph.p0.x), p1 = __float_as_uint(ph.p0.y), p2 = __float_as_uint(ph.p0.z);
    unsigned int p3 = __float_as_uint(ph.dir.x), p4 = __float_as_uint(ph.dir.y), p5 = __float_as_uint(ph.dir.z);
    unsigned int p6 = __float_as_uint(ph.slen), p7 = __float_as_uint(ph.weight), p8 = __float_as_uint(ph.tof);
    unsigned int p9 = ph.mid;
#ifdef USE_CSG
    unsigned int p10 = ph.mstate.x, p11 = ph.mstate.y;
    unsigned int p12 = rng.x, p13 = rng.y, p14 = rng.z, p15 = rng.w;
    unsigned int p16 = ph.pending, p17 = ph.primid;
    optixTrace(gcfg.gashandle, ph.p0, ph.dir, tmin, tmax, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0,
               p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17);
    ph.mstate = make_uint2(p10, p11);
    ph.pending = p16;
    ph.primid = p17;
    rng = make_uint4(p12, p13, p14, p15);
#else
    unsigned int p10 = rng.x, p11 = rng.y, p12 = rng.z, p13 = rng.w;
    optixTrace(gcfg.gashandle, ph.p0, ph.dir, tmin, tmax, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0,
               p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
    rng = make_uint4(p10, p11, p12, p13);
#endif
    ph.p0 = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
    ph.dir = make_float3(__uint_as_float(p3), __uint_as_float(p4), __uint_as_float(p5));
    ph.slen = __uint_as_float(p6);
    ph.weight = __uint_as_float(p7);
    ph.tof = __uint_as_float(p8);
    ph.mid = p9;
}

/* ================================================================ */
/*                     CSG media state                              */
/* ================================================================ */

#ifdef USE_CSG
__device__ __forceinline__ unsigned int ms_get(uint2 ms, int slot) {
    unsigned int w = (slot < 4) ? ms.x : ms.y;
    return (w >> ((slot & 3) * 8)) & 0xFFu;
}
__device__ __forceinline__ uint2 ms_set(uint2 ms, int slot, unsigned int val) {
    unsigned int sh = (slot & 3) * 8, mk = 0xFFu << sh;

    if (slot < 4) {
        ms.x = (ms.x & ~mk) | ((val & 0xFFu) << sh);
    } else {
        ms.y = (ms.y & ~mk) | ((val & 0xFFu) << sh);
    }

    return ms;
}
__device__ __forceinline__ uint2 ms_enter(uint2 ms, unsigned int bid) {
    if (!bid) {
        return ms;
    }

    unsigned int cm = ms_get(ms, 0), nm = max(cm, bid), dm = (nm == bid) ? cm : bid;
    ms = ms_set(ms, 0, nm);

    if (dm) for (int i = 1; i < 8; i++) if (!ms_get(ms, i)) {
                ms = ms_set(ms, i, dm);
                break;
            }

    return ms;
}
__device__ __forceinline__ uint2 ms_leave(uint2 ms, unsigned int bid) {
    if (!bid) {
        return ms;
    }

    for (int i = 0; i < 8; i++) if (ms_get(ms, i) == bid) {
            ms = ms_set(ms, i, 0);
            break;
        }

    unsigned int mx = 0;
    int ms2 = -1;

    for (int i = 0; i < 8; i++) {
        unsigned int v = ms_get(ms, i);

        if (v > mx) {
            mx = v;
            ms2 = i;
        }
    }

    if (ms2 > 0) {
        ms = ms_set(ms, ms2, 0);
        ms = ms_set(ms, 0, mx);
    } else if (ms2 < 0) {
        ms = ms_set(ms, 0, 0);
    }

    return ms;
}
#endif

/* ================================================================ */
/*                       Face data helpers                          */
/* ================================================================ */

__device__ __forceinline__ void unpack_media(float pw, unsigned int& fm, unsigned int& bm) {
    unsigned int p = __float_as_uint(pw);   /* bit 31: flat, 30: coplanar, 29-16: front, 15-0: back */
    fm = (p >> 16) & 0x3FFFu;
    bm = p & 0xFFFFu;
}

__device__ __forceinline__ bool is_coplanar(float pw) {
    return (__float_as_uint(pw) & 0x40000000u) != 0u;
}

#ifdef USE_CURVATURE
__device__ __forceinline__ float3 get_curvature_normal(int tri_id, float3 hit_pos) {
    float4* fb = (float4*)gcfg.facebuf;
    float4 fn4 = fb[tri_id * 2];
    unsigned int packed = __float_as_uint(fn4.w);

    if (gcfg.has_curvature == 0u || (packed & FLAT_FLAG)) {
        return make_float3(fn4.x, fn4.y, fn4.z);
    }

    float4 vidx_raw = fb[tri_id * 2 + 1];
    float3 NB = make_float3(0, 0, 0);
    float* vr = (float*)&vidx_raw;

    for (int vi = 0; vi < 3; vi++) {
        unsigned int idx = __float_as_uint(vr[vi]);
        NodeCurv* cb = (NodeCurv*)gcfg.curvbuf;
        NodeCurv nc = cb[idx];
        float3 Ni = make_float3(nc.vnorm_k1.x, nc.vnorm_k1.y, nc.vnorm_k1.z);
        float k1 = nc.vnorm_k1.w;
        float3 ui = make_float3(nc.pdir_k2.x, nc.pdir_k2.y, nc.pdir_k2.z);
        float k2 = nc.pdir_k2.w;
        float3 Pi = make_float3(nc.node_pos.x, nc.node_pos.y, nc.node_pos.z);
        float3 vi2 = cross(ui, Ni);
        float3 di = hit_pos - Pi;
        float du = dot(di, ui), dv = dot(di, vi2);
        NB = NB + Ni + k1 * du * ui + k2 * dv * vi2;
    }

    NB = normalize(NB);

    if (dot(NB, make_float3(fn4.x, fn4.y, fn4.z)) < 0.0f) {
        NB = -NB;
    }

    return NB;
}
#endif

/* ================================================================ */
/*                    Surface normal helper                         */
/* ================================================================ */

__device__ __forceinline__ float3 get_normal(int primid, float3 hit_pos, float4 fn4) {
#ifdef USE_CURVATURE
    return get_curvature_normal(primid, hit_pos);
#else
    return make_float3(fn4.x, fn4.y, fn4.z);
#endif
}

/* ================================================================ */
/*                         Output                                   */
/* ================================================================ */

#define FIXED_SCALE 4294967296.f

/* add a deposit as an unsigned 64-bit fixed-point sum, see vkmmc_core.comp */
__device__ __forceinline__ void atomic_add_out(unsigned int idx, float val) {
    atomicAdd((unsigned long long*)gcfg.outputbuf + idx, (unsigned long long)(val * FIXED_SCALE + 0.5f));
}

/* the photon's slot: its output field (multi-source) or its pattern pixel (shared patterns) */
__device__ __forceinline__ unsigned int photon_slot() {
    return ((unsigned int*)gcfg.slotbuf)[optixGetLaunchIndex().x];
}

/* a deposit of a photon shared by src_num patterns: one per pattern field */
__device__ __noinline__ void deposit_shared(unsigned int idx, float val) {
    const float* pat = (const float*)gcfg.patternbuf + photon_slot() * gcfg.src_num;

    for (unsigned int k = 0; k < gcfg.src_num; k++) {
        if (pat[k] > 0.f) {
            atomic_add_out(k * gcfg.grid_stride.w + idx, val * pat[k]);
        }
    }
}

/* a deposit of the current photon at element idx of its field(s); see deposit() in vkmmc_core.comp.
   The rare shared-pattern path is not inlined: it would cost the common path registers */
__device__ __forceinline__ void deposit(unsigned int idx, float val) {
    if (gcfg.src_num > 1) {
        deposit_shared(idx, val);
    } else {
        atomic_add_out(((gcfg.extra_src_len > 0 && gcfg.src_id < 0) ? photon_slot() * gcfg.grid_stride.w : 0u) + idx, val);
    }
}

/* exact voxel/gate traversal; see accumulate() in vkmmc_core.comp */
__device__ void accumulate(const Photon& ph, const Medium& prop, float L) {
    if (!(L > 0.f)) {
        return;
    }

    const uint4 gs = gcfg.grid_stride;
    int3 dim = make_int3((int)gs.x, (int)(gs.y / gs.x), (int)(gs.z / gs.y));
    float3 s = (ph.p0 - gcfg.grid_min) * gcfg.voxel_scale;
    float3 d = ph.dir;
    float Lv = L * gcfg.voxel_scale, lscale = 1.f / gcfg.voxel_scale, w = ph.weight;
    int3 iv = make_int3(min(max((int)floorf(s.x), 0), dim.x - 1), min(max((int)floorf(s.y), 0), dim.y - 1),
                        min(max((int)floorf(s.z), 0), dim.z - 1));
    float g0 = (ph.tof - gcfg.tstart) * gcfg.inv_timestep;
    float gstep = INV_C0 * prop.n * lscale * gcfg.inv_timestep;
    float gi = floorf(g0);

    /* fast path: a step that starts and ends in the same voxel and gate */
    {
        float3 e = s + d * Lv;
        int3 ie = make_int3(min(max((int)floorf(e.x), 0), dim.x - 1), min(max((int)floorf(e.y), 0), dim.y - 1),
                            min(max((int)floorf(e.z), 0), dim.z - 1));

        if (ie.x == iv.x && ie.y == iv.y && ie.z == iv.z && floorf(g0 + Lv * gstep) == gi) {
            if (gi >= 0.f && gi < (float)gcfg.maxgate) {
                float x = prop.mua * L;
                float frac = (x < 1e-3f) ? x * (1.f - 0.5f * x) : 1.f - expf(-x);
                deposit((unsigned int)gi * gs.z + (unsigned int)iv.z * gs.y + (unsigned int)iv.y * gs.x + (unsigned int)iv.x,
                        (gcfg.output_type == OUT_ENERGY) ? w * frac : w * ((x < 1e-3f) ? L * (1.f - 0.5f * x) : frac / prop.mua));
            }

            return;
        }
    }

    float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    float3 tdel = make_float3(1.f / fmaxf(ad.x, 1e-20f), 1.f / fmaxf(ad.y, 1e-20f), 1.f / fmaxf(ad.z, 1e-20f));
    const float BIG = 3.402823466e+38f;
    float3 tm;
    tm.x = (ad.x < 1e-20f) ? BIG : fmaxf(((d.x > 0) ? (iv.x + 1) - s.x : s.x - iv.x) * tdel.x, 0.f);
    tm.y = (ad.y < 1e-20f) ? BIG : fmaxf(((d.y > 0) ? (iv.y + 1) - s.y : s.y - iv.y) * tdel.y, 0.f);
    tm.z = (ad.z < 1e-20f) ? BIG : fmaxf(((d.z > 0) ? (iv.z + 1) - s.z : s.z - iv.z) * tdel.z, 0.f);
    int3 dir = make_int3(d.x > 0 ? 1 : -1, d.y > 0 ? 1 : -1, d.z > 0 ? 1 : -1);
    float tg = (gi + 1.f - g0) / gstep, t = 0.f;

    for (int guard = 0; guard < 65536 && t < Lv; guard++) {
        if (gi >= (float)gcfg.maxgate) {
            break;
        }

        float tv = fminf(fminf(tm.x, tm.y), tm.z);
        float te = fminf(fminf(tv, tg), Lv);
        float x = prop.mua * (te - t) * lscale;
        float frac = (x < 1e-3f) ? x * (1.f - 0.5f * x) : 1.f - expf(-x);

        if (gi >= 0.f) {
            float dep = (gcfg.output_type == OUT_ENERGY) ? w * frac
                        : w * ((x < 1e-3f) ? (te - t) * lscale * (1.f - 0.5f * x) : frac / prop.mua);
            deposit((unsigned int)gi * gs.z + (unsigned int)iv.z * gs.y + (unsigned int)iv.y * gs.x + (unsigned int)iv.x, dep);
        }

        w *= 1.f - frac;
        t = te;

        if (te == tg) {
            gi += 1.f;
            tg = (gi + 1.f - g0) / gstep;
        }

        if (te == tv && te < Lv) {
            if (tm.x <= tm.y && tm.x <= tm.z) {
                iv.x = min(max(iv.x + dir.x, 0), dim.x - 1);
                tm.x += tdel.x;
            } else if (tm.y <= tm.z) {
                iv.y = min(max(iv.y + dir.y, 0), dim.y - 1);
                tm.y += tdel.y;
            } else {
                iv.z = min(max(iv.z + dir.z, 0), dim.z - 1);
                tm.z += tdel.z;
            }
        }
    }
}

/* ================================================================ */
/*                        Scattering                                */
/* ================================================================ */

__device__ float3 scatter_dir(float3 d, float g, uint4& rng) {
    float ct;

    if (fabsf(g) > FEPS) {
        float t = (1 - g * g) / fmaxf(1 - g + 2 * g * rand01(rng), FEPS);
        ct = fmaxf(-1.f, fminf(1.f, (1 + g * g - t * t) / (2 * g)));
    } else {
        ct = 2 * rand01(rng) - 1;
    }

    float st = sqrtf(fmaxf(1 - ct * ct, 0.f));
    float sp, cp;
    sincosf(TWO_PI * rand01(rng), &sp, &cp);

    if (d.z > -1 + FEPS && d.z < 1 - FEPS) {
        float s2 = 1 - d.z * d.z, isc = st * rsqrtf(s2);
        return isc * (cp * make_float3(d.x * d.z, d.y * d.z, -s2) + sp * make_float3(-d.y, d.x, 0)) + ct * d;
    }

    return make_float3(st * cp, st * sp, d.z > 0 ? ct : -ct);
}

/* ================================================================ */
/*                          Fresnel                                 */
/* ================================================================ */

/* Fresnel reflection/refraction; see do_reflection() in vkmmc_core.comp.
   N: shading normal, Ng: facet normal (either orientation); returns true if reflected */
__device__ bool do_reflection(float3 N, float3 Ng, float n1, float n2, uint4& rng, float3 hit, Photon& ph) {
    float3 d0 = ph.dir;
    float3 Nin = (dot(d0, Ng) < 0) ? Ng : -Ng;
    float ci = -dot(d0, N);

    if (ci < 0) {
        N = -N;
        ci = -ci;
    }

    float n12 = n1 * n1, n22 = n2 * n2, ct2 = 1 - n12 / n22 * (1 - ci * ci);
    bool reflect = true;

    if (ct2 > 0) {
        float ct = sqrtf(ct2);
        float re = n12 * ci * ci + n22 * ct2, im = 2 * n1 * n2 * ci * ct;
        float rp = (re - im) / (re + im);
        re = n22 * ci * ci + n12 * ct * ct;
        float rt = (rp + (re - im) / (re + im)) * 0.5f;

        if (rand01(rng) > rt) {
            float r = n1 / n2;
            ph.dir = r * d0 + (r * ci - ct) * N;

            if (dot(ph.dir, Nin) >= 0) {
                float cig = -dot(d0, Nin), ctg2 = 1 - r * r * (1 - cig * cig);

                if (ctg2 > 0) {
                    ph.dir = r * d0 + (r * cig - sqrtf(ctg2)) * Nin;
                } else {
                    ph.dir = d0 + 2 * cig * Nin;
                    ph.p0 = hit + Nin * SAFETY_DIST;
                    ph.dir = ph.dir * rsqrtf(dot(ph.dir, ph.dir));
                    return true;
                }
            }

            ph.p0 = hit - Nin * SAFETY_DIST;
            reflect = false;
        }
    }

    if (reflect) {
        ph.dir = d0 + 2 * ci * N;

        if (dot(ph.dir, Nin) <= 0) {
            ph.dir = d0 - 2 * dot(d0, Nin) * Nin;
        }

        ph.p0 = hit + Nin * SAFETY_DIST;
    }

    ph.dir = ph.dir * rsqrtf(dot(ph.dir, ph.dir));
    return reflect;
}

/* ================================================================ */
/*                       Launch photon                              */
/* ================================================================ */

/* source types, see vkmmc_core.comp */
enum {
    SRC_PENCIL = 0, SRC_ISOTROPIC = 1, SRC_CONE = 2, SRC_GAUSSIAN = 3, SRC_PLANAR = 4,
    SRC_PATTERN = 5, SRC_FOURIER = 6, SRC_ARCSINE = 7, SRC_DISK = 8, SRC_FOURIERX = 9,
    SRC_FOURIERX2D = 10, SRC_ZGAUSSIAN = 11, SRC_LINE = 12, SRC_SLIT = 13,
    SRC_PENCILARRAY = 14, SRC_PATTERN3D = 15, SRC_HYPERBOLOID = 16, SRC_RING = 17
};

/* rotate v by the zenith (sin, cos) z and azimuth (sin, cos) a */
__device__ __forceinline__ float3 rotate_dir(float3 v, float2 z, float2 a) {
    if (v.z > -1 + FEPS && v.z < 1 - FEPS) {
        float s2 = 1 - v.z * v.z, sc = z.x * rsqrtf(s2);
        return sc * (a.y * make_float3(v.x * v.z, v.y * v.z, -s2) + a.x * make_float3(-v.y, v.x, 0)) + z.y * v;
    }

    return make_float3(z.x * a.y, z.x * a.x, v.z > 0 ? z.y : -z.y);
}

__device__ __forceinline__ float3 rotate_ct(float3 v, float ct, float phi) {
    float sp, cp;
    sincosf(phi, &sp, &cp);
    return rotate_dir(v, make_float2(sqrtf(fmaxf(1.f - ct * ct, 0.f)), ct), make_float2(sp, cp));
}

/* offset of radius r0 at azimuth phi in the plane perpendicular to sd */
__device__ __forceinline__ float3 disk_offset(float3 sd, float r0, float phi) {
    float sp, cp;
    sincosf(phi, &sp, &cp);

    if (sd.z > -1 + FEPS && sd.z < 1 - FEPS) {
        float t0 = 1 - sd.z * sd.z, t1 = r0 * rsqrtf(t0);
        return make_float3(t1 * (sd.x * sd.z * cp - sd.y * sp), t1 * (sd.y * sd.z * cp + sd.x * sp), -t1 * t0 * cp);
    }

    return make_float3(r0 * cp, r0 * sp, 0);
}

__device__ __forceinline__ float3 focus_dir(float3 p, float3 d, float3 origin, float f) {
    float3 o = origin + f * d;
    return normalize((f < 0) ? p - o : o - p);
}

/* sample the position, direction and weight of a photon; see sample_source() in vkmmc_core.comp */
__device__ void sample_source(Photon& ph, uint4& rng, unsigned int& slot) {
    float3 spos = gcfg.srcpos, sdir = gcfg.srcdir;
    float4 p1 = gcfg.srcparam1, p2 = gcfg.srcparam2;
    float focal = (gcfg.focus_mode == 1) ? gcfg.focal : 0.f, sw = 1.f;
    unsigned int field = 0;
    slot = 0;

    if (gcfg.extra_src_len > 0) {
        /* each source is of Optode.Source.Type with its own Pos (.w weight), Dir (.w focal length), Param1, Param2 */
        unsigned int k = (unsigned int)max(gcfg.src_id - 1, 0);

        if (gcfg.src_id < 0) {
            k = min((unsigned int)(rand01(rng) * gcfg.extra_src_len), gcfg.extra_src_len - 1);
            field = k;
        }

        const float4* es = (const float4*)gcfg.extrasrcbuf + k * 4;
        float4 sp = es[0], sd = es[1];
        spos = make_float3(sp.x, sp.y, sp.z);
        sdir = make_float3(sd.x, sd.y, sd.z);
        p1 = es[2];
        p2 = es[3];
        focal = sd.w;
        sw = sp.w;
        ph.p0 = spos;
        ph.dir = sdir;

        /* a pencil beam, on a disk of radius Param1[0] perpendicular to Dir (as mmc-optix) */
        if (gcfg.srctype == SRC_PENCIL) {
            ph.weight = sw;
            slot = field;

            if (p1.x > 0) {
                float phi = TWO_PI * rand01(rng);
                ph.p0 += disk_offset(ph.dir, sqrtf(rand01(rng)) * p1.x, phi);
            }

            if (focal != 0) {
                ph.dir = focus_dir(ph.p0, ph.dir, spos, focal);
            }

            return;
        }
    }

    float3 v1 = make_float3(p1.x, p1.y, p1.z), v2 = make_float3(p2.x, p2.y, p2.z);
    const int st = gcfg.srctype;
    float3 origin = ph.p0;
    bool canfocus = true;

    if (st == SRC_PLANAR || st == SRC_PATTERN || st == SRC_FOURIER || st == SRC_PENCILARRAY || st == SRC_PATTERN3D) {
        float rx = rand01(rng), ry = rand01(rng);

        if (st == SRC_PATTERN3D) {
            float rz = rand01(rng);
            int nx = (int)p1.x, ny = (int)p1.y, nz = (int)p1.z;
            ph.p0 = spos + make_float3(rx * p1.x, ry * p1.y, rz * p1.z);
            slot = (unsigned int)((min((int)(rz * nz), nz - 1) * ny + min((int)(ry * ny), ny - 1)) * nx + min((int)(rx * nx), nx - 1));
            ph.weight = (gcfg.src_num > 1) ? 1.f : ((const float*)gcfg.patternbuf)[slot];
            origin += 0.5f * v1;
        } else {
            if (st == SRC_PENCILARRAY) {
                float ax = (p1.w > 1) ? floorf(rx * p1.w) / (p1.w - 1) : 0.f;
                float ay = (p2.w > 1) ? floorf(ry * p2.w) / (p2.w - 1) : 0.f;
                ph.p0 = spos + ax * v1 + ay * v2;
            } else {
                ph.p0 = spos + rx * v1 + ry * v2;
            }

            if (st == SRC_PATTERN) {
                int xs = (int)p1.w, ys = (int)p2.w;
                slot = (unsigned int)(min((int)(ry * ys), ys - 1) * xs + min((int)(rx * xs), xs - 1));
                ph.weight = (gcfg.src_num > 1) ? 1.f : ((const float*)gcfg.patternbuf)[slot];
            } else if (st == SRC_FOURIER) {
                ph.weight = (cosf((floorf(p1.w) * rx + floorf(p2.w) * ry + p1.w - floorf(p1.w)) * TWO_PI)
                             * (1.f - p2.w + floorf(p2.w)) + 1.f) * 0.5f;
            }

            origin += 0.5f * (v1 + v2);
        }
    } else if (st == SRC_FOURIERX || st == SRC_FOURIERX2D) {
        float rx = rand01(rng), ry = rand01(rng);
        float3 v2x = p1.w * rsqrtf(dot(v1, v1)) * cross(sdir, v1);
        ph.p0 = spos + rx * v1 + ry * v2x;

        if (st == SRC_FOURIERX2D) {
            ph.weight = (sinf((p2.x * rx + p2.z) * TWO_PI) * sinf((p2.y * ry + p2.w) * TWO_PI) + 1.f) * 0.5f;
        } else {
            ph.weight = (cosf((p2.x * rx + p2.y * ry + p2.z) * TWO_PI) * (1.f - p2.w) + 1.f) * 0.5f;
        }

        origin += 0.5f * (v1 + v2x);
    } else if (st == SRC_DISK || st == SRC_RING || st == SRC_GAUSSIAN) {
        float phi, r0;

        if (st == SRC_RING && (p1.z > 0 || p1.w > 0)) {
            phi = fabsf(p1.z - p1.w) * rand01(rng) + fminf(p1.z, p1.w);
        } else {
            phi = TWO_PI * rand01(rng);
        }

        if (st != SRC_GAUSSIAN) {
            r0 = sqrtf(rand01(rng) * fabsf(p1.x * p1.x - p1.y * p1.y) + p1.y * p1.y);
        } else if (fabsf(focal) < 1e-5f || fabsf(p1.y) < 1e-5f) {
            r0 = sqrtf(rand_scatlen(rng)) * p1.x;
        } else {
            float z0 = p1.x * p1.x * ONE_PI / p1.y;
            r0 = sqrtf(rand_scatlen(rng) * (1.f + focal * focal / (z0 * z0))) * p1.x;
        }

        ph.p0 = spos + disk_offset(sdir, r0, phi);
    } else if (st == SRC_CONE || st == SRC_ISOTROPIC || st == SRC_ARCSINE) {
        float phi = TWO_PI * rand01(rng), ct;

        if (st == SRC_CONE) {
            ct = (p1.y > 0) ? cosf(rand01(rng) * fminf(p1.x, TWO_PI)) : 1.f - rand01(rng) * (1.f - cosf(fminf(p1.x, ONE_PI)));
        } else if (st == SRC_ISOTROPIC) {
            ct = 2.f * rand01(rng) - 1.f;
        } else {
            ct = cosf(ONE_PI * rand01(rng));
        }

        ph.dir = rotate_ct(ph.dir, ct, phi);
        canfocus = false;
    } else if (st == SRC_ZGAUSSIAN) {
        float phi = TWO_PI * rand01(rng);
        float ang = sqrtf(2.f * rand_scatlen(rng)) * (1.f - 2.f * rand01(rng)) * p1.x;
        ph.dir = rotate_dir(ph.dir, make_float2(sinf(ang), cosf(ang)), make_float2(sinf(phi), cosf(phi)));

        canfocus = false;
    } else if (st == SRC_HYPERBOLOID) {
        float sphi, cphi;
        sincosf(TWO_PI * rand01(rng), &sphi, &cphi);
        float r0 = sqrtf(0.5f * rand_scatlen(rng)) * p1.x;
        float tilt = (p1.z > 0) ? (-p1.y / p1.z) : 0.f;
        float3 bdir = normalize(make_float3(-r0 * sphi, r0 * cphi, p1.z));
        float dpx = r0 * (cphi - tilt * sphi), dpy = r0 * (sphi + tilt * cphi);
        float3 sd = sdir;

        if (sd.z > -1 + FEPS && sd.z < 1 - FEPS) {
            float stt = sqrtf(1 - sd.z * sd.z), cb = sd.x / stt, sb = sd.y / stt;
            ph.p0 = spos + make_float3(dpx * cb * sd.z - dpy * sb, dpx * sb * sd.z + dpy * cb, -dpx * stt);
            ph.dir = make_float3(bdir.x * cb * sd.z - bdir.y * sb + bdir.z * cb * stt,
                                 bdir.x * sb * sd.z + bdir.y * cb + bdir.z * sb * stt,
                                 -bdir.x * stt + bdir.z * sd.z);
        } else {
            ph.p0 = spos + make_float3(dpx, dpy, 0);
            ph.dir = make_float3(bdir.x, bdir.y, (sd.z > 0) ? bdir.z : -bdir.z);
        }

        ph.dir = normalize(ph.dir);
        canfocus = false;
    } else if (st == SRC_LINE || st == SRC_SLIT) {
        ph.p0 += rand01(rng) * v1;
        float3 axis = normalize(v1);

        if (st == SRC_LINE) {
            float3 dp = ph.dir - dot(ph.dir, axis) * axis;

            if (p2.x > 0 && dot(dp, dp) > 1e-12f) {
                float sp, cp;
                sincosf(p2.x * (2.f * rand01(rng) - 1.f), &sp, &cp);
                dp = normalize(dp);
                ph.dir = dp * cp + cross(axis, dp) * sp;
            } else {
                float sp, cp;
                sincosf(TWO_PI * rand01(rng), &sp, &cp);
                ph.dir = rotate_dir(axis, make_float2(1.f, 0.f), make_float2(sp, cp));
            }
        } else if (p2.x > 0 || p2.y > 0) {
            float sp, cp;
            sincosf(TWO_PI * rand01(rng), &sp, &cp);
            float r = sqrtf(2.f * rand_scatlen(rng));
            ph.dir = normalize(ph.dir + (cp * p2.x * r) * normalize(cross(v1, ph.dir)) + (sp * p2.y * r) * axis);
        }

        origin += 0.5f * v1;
        canfocus = (st == SRC_SLIT);
    }

    if (canfocus) {
        if (gcfg.focus_mode == 2) {
            float phi = TWO_PI * rand01(rng);
            ph.dir = rotate_ct(ph.dir, 2.f * rand01(rng) - 1.f, phi);
        } else if (gcfg.focus_mode == 3) {
            float phi = TWO_PI * rand01(rng), s = sqrtf(rand01(rng));
            ph.dir = rotate_ct(ph.dir, sqrtf(1.f - s * s), phi);
        } else if (focal != 0) {
            ph.dir = focus_dir(ph.p0, ph.dir, origin, focal);
        }
    }

    ph.weight *= sw;

    if (gcfg.extra_src_len > 0) {
        slot = field;
    }
}

/*
 * Launch a photon; w0 returns its launch weight. The slot buffer holds the
 * field or pattern pixel the hit programs deposit to.
 */
__device__ void launch_photon(Photon& ph, uint4& rng, float& w0) {
    ph.p0 = gcfg.srcpos;
    ph.dir = gcfg.srcdir;
    ph.weight = 1.0f;
    ph.tof = 0;
#ifdef USE_CSG
    ph.mstate = make_uint2(0, 0);
    ph.pending = 0;
    ph.primid = 0;
#endif
    unsigned int slot = 0;

    if (gcfg.srctype != SRC_PENCIL || gcfg.extra_src_len > 0 || gcfg.focus_mode != 0) {
        sample_source(ph, rng, slot);
    }
    w0 = ph.weight;

    if (gcfg.src_num > 1 || (gcfg.extra_src_len > 0 && gcfg.src_id < 0)) {
        ((unsigned int*)gcfg.slotbuf)[optixGetLaunchIndex().x] = slot;
    }

    ph.slen = rand_scatlen(rng);
    ph.mid = gcfg.initial_medium;
}

/* launched energy, see tally_launch() in vkmmc_core.comp */
__device__ __forceinline__ void tally_launch(float w0) {
    if (!gcfg.do_tally) {
        return;
    }

    if (gcfg.src_num > 1) {
        const float* pat = (const float*)gcfg.patternbuf + photon_slot() * gcfg.src_num;

        for (unsigned int k = 0; k < gcfg.src_num; k++) {
            atomic_add_out(gcfg.num_field * gcfg.grid_stride.w + k, pat[k]);
        }
    } else {
        atomic_add_out(gcfg.num_field * gcfg.grid_stride.w, w0);
    }
}


/* closest hit along a ray: distance, triangle and face orientation */
__device__ bool probe_ray(float3 o, float3 d, float tmin, uint4& rng, float& hd, int& tid, bool& ff) {
    Photon pr;
    pr.p0 = o;
    pr.dir = d;
    pr.slen = 0;
    pr.weight = 0;
    pr.tof = 0;
    pr.mid = MEDIUM_PROBE;
#ifdef USE_CSG
    pr.mstate = make_uint2(0, 0);
    pr.pending = 0;
    pr.primid = 0;
#endif
    traceRay(pr, rng, std::numeric_limits<float>::max(), tmin);

    if (pr.mid != MEDIUM_PROBE_HIT) {
        return false;
    }

    hd = pr.slen;
    tid = (int)__float_as_uint(pr.weight);
    ff = pr.tof > 0.5f;
    return true;
}

#ifdef USE_CSG
#define COPLANAR_EPS 2e-4f
#define COPLANAR_MAXHIT 8

__device__ __forceinline__ bool tagset_has(uint2 set, unsigned int tag) {
    return ((tag > 32 ? set.y : set.x) >> ((tag - 1) & 31)) & 1u;
}

__device__ __forceinline__ uint2 tagset_add(uint2 set, unsigned int tag) {
    unsigned int bit = 1u << ((tag - 1) & 31);
    return (tag > 32) ? make_uint2(set.x, set.y | bit) : make_uint2(set.x | bit, set.y);
}

/*
 * All CSG triangles where a photon moving along nfar crossed a coplanar
 * facet, see coplanar_crossings() in vkmmc_core.comp. The probe uses ray type
 * 1, whose any-hit program records every candidate and ignores it.
 * mode 0 updates ms, mode 1 the enclosure walk sets seen/inside.
 */
__device__ void coplanar_crossings(float3 p, float3 nfar, int mode, uint2& ms, uint2& seen, uint2& inside) {
    float4* fb = (float4*)gcfg.facebuf;
    unsigned int n = 0, h0 = 0, h1 = 0, h2 = 0, h3 = 0, h4 = 0, h5 = 0, h6 = 0, h7 = 0;
    optixTrace(gcfg.gashandle, p, -nfar, SAFETY_DIST - COPLANAR_EPS, SAFETY_DIST + COPLANAR_EPS, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 1, 1, 1, n, h0, h1, h2, h3, h4, h5, h6, h7);
    unsigned int hits[COPLANAR_MAXHIT] = {h0, h1, h2, h3, h4, h5, h6, h7};
    uint2 entered = make_uint2(0, 0), left = make_uint2(0, 0);

    for (unsigned int k = 0; k < n && k < COPLANAR_MAXHIT; k++) {
        unsigned int tid = hits[k] & 0x7FFFFFFFu, fm, bm;
        bool enter = !(hits[k] >> 31);     /* the probe sees the faces from the other side */
        unpack_media(fb[tid * 2].w, fm, bm);

        if (bm < 1 || bm > 64) {
            continue;
        }

        if (mode == 0) {
            if (enter && !tagset_has(entered, bm)) {
                entered = tagset_add(entered, bm);
                ms = ms_enter(ms, bm);
            } else if (!enter && !tagset_has(left, bm)) {
                left = tagset_add(left, bm);
                ms = ms_leave(ms, bm);
            }
        } else if (!tagset_has(seen, bm)) {
            seen = tagset_add(seen, bm);

            if (!enter) {
                inside = tagset_add(inside, bm);
            }
        }
    }
}

/* tags enclosing p (bit t-1 for tag t) and the first hit, see csg_enclosure() in vkmmc_core.comp */
__device__ uint2 csg_enclosure(float3 p, float3 d, uint4& rng, float& hd0, int& tid0, bool& ff0) {
    float4* fb = (float4*)gcfg.facebuf;
    uint2 seen = make_uint2(0, 0), inside = make_uint2(0, 0), ms = make_uint2(0, 0);
    float3 perp = (fabsf(d.z) < 0.999f) ? normalize(cross(d, make_float3(0, 0, 1)))
                  : normalize(cross(d, make_float3(1, 0, 0)));
    float3 wp = p + perp * 2e-4f;
    float hd;
    int tid;
    bool ff;
    unsigned int fm, bm;

    hd0 = -1.0f;
    tid0 = 0;
    ff0 = false;

    for (int wk = 0; wk < 64; wk++) {
        if (!probe_ray(wp, d, RAY_TMIN, rng, hd, tid, ff)) {
            break;
        }

        if (wk == 0) {
            hd0 = hd;
            tid0 = tid;
            ff0 = ff;
        }

        float4 fn = fb[tid * 2];
        float3 Ng = make_float3(fn.x, fn.y, fn.z);
        float3 nfar = (dot(d, Ng) > 0) ? Ng : -Ng;
        wp = wp + d * hd + nfar * SAFETY_DIST;

        if (is_coplanar(fn.w)) {
            coplanar_crossings(wp, nfar, 1, ms, seen, inside);
            continue;
        }

        unpack_media(fn.w, fm, bm);

        if (bm >= 1 && bm <= 64 && !tagset_has(seen, bm)) {
            seen = tagset_add(seen, bm);

            if (!ff) {
                inside = tagset_add(inside, bm);
            }
        }
    }

    return inside;
}

__device__ uint2 csg_state(uint2 inside) {
    uint2 ms = make_uint2(0, 0);

    for (unsigned int t = 1; t <= 64; t++)
        if (tagset_has(inside, t)) {
            ms = ms_enter(ms, t);
        }

    return ms;
}

/*
 * Finish a crossing of a coplanar triangle flagged by the closest-hit
 * program (which cannot trace): apply the medium change of every coincident
 * triangle, then Fresnel reflection between the old and the new medium.
 */
__device__ void finish_coplanar_crossing(Photon& ph, uint4& rng) {
    float4* fb = (float4*)gcfg.facebuf;
    float4 fn4 = fb[ph.primid * 2];
    float3 Ng = make_float3(fn4.x, fn4.y, fn4.z);
    float3 Nfar = (dot(ph.dir, Ng) > 0) ? Ng : -Ng;
    float3 hit = ph.p0 - Nfar * SAFETY_DIST;
    unsigned int old_mid = ph.mid;
    uint2 new_ms = ph.mstate, u0 = make_uint2(0, 0), u1 = make_uint2(0, 0);
    coplanar_crossings(ph.p0, Nfar, 0, new_ms, u0, u1);
    unsigned int new_mid = ms_get(new_ms, 0);
    bool reflected = false;
    ph.pending = 0;

    if (gcfg.do_reflect && new_mid != old_mid && new_mid < gcfg.num_media && old_mid < gcfg.num_media
            && gcfg.media[old_mid].n != gcfg.media[new_mid].n) {
        reflected = do_reflection(get_normal(ph.primid, hit, fn4), Ng, gcfg.media[old_mid].n, gcfg.media[new_mid].n,
                                  rng, hit, ph);
    }

    if (!reflected) {
        ph.mstate = new_ms;
        ph.mid = (new_mid == 0) ? MEDIUM_DEAD : new_mid;
    }
}
#endif

/* a photon from outside enters at hp, see enter_domain() in vkmmc_core.comp; false if reflected */
__device__ bool enter_domain(Photon& ph, uint4& rng, float3 hp, float hd, int tid) {
    float4* fb = (float4*)gcfg.facebuf;
    ph.p0 = hp + ph.dir * SAFETY_DIST;
    ph.tof += hd * INV_C0 * gcfg.media[0].n;

    return !(gcfg.do_specular && ph.mid < gcfg.num_media && gcfg.media[0].n != gcfg.media[ph.mid].n
             && do_reflection(get_normal(tid, hp, fb[tid * 2]), make_float3(fb[tid * 2]), gcfg.media[0].n,
                              gcfg.media[ph.mid].n, rng, hp, ph));
}

/*
 * Find the medium enclosing the source (same algorithm as vkmmc_core.comp).
 * CSG: a forward walk through all surfaces; the source is inside a shape when
 * the first crossing of its tag is an exit. A pencil source launches from one
 * point (moved off a face by the host), so the walk is cached per thread. A
 * source outside everything enters the first surface it points at. A source
 * on a face is inside the tissue there: the rays start SAFETY_DIST behind the
 * source and a face within 2 SAFETY_DIST is the one it sits on; the photon
 * stays on the side behind it, or moves forward if that is the background.
 * Returns false if the photon misses the domain.
 */
__device__ bool initial_medium(Photon& ph, uint4& rng, uint2& cache, bool& cached) {
    float4* fb = (float4*)gcfg.facebuf;
    float hd;
    int tid;
    bool ff;
    unsigned int fm, bm;
    float3 o = ph.p0 - ph.dir * SAFETY_DIST;

#ifdef USE_CSG

    if (gcfg.do_csg) {
        uint2 inside;
        bool fixed_beam = (gcfg.srctype == SRC_PENCIL && gcfg.extra_src_len == 0 && gcfg.focus_mode == 0);

        if (fixed_beam) {   /* already moved off a face by the host */
            if (!cached) {
                cache = csg_enclosure(ph.p0, ph.dir, rng, hd, tid, ff);
                cached = true;
            }

            inside = cache;
            hd = -1.0f;
        } else {
            inside = csg_enclosure(o, ph.dir, rng, hd, tid, ff);

            if (hd >= 0 && hd < 2.0f * SAFETY_DIST) {
                float3 hp = o + ph.dir * hd;

                if (inside.x || inside.y) {
                    ph.p0 = hp - ph.dir * SAFETY_DIST;
                } else {
                    ph.p0 = hp + ph.dir * SAFETY_DIST;
                    unpack_media(fb[tid * 2].w, fm, bm);

                    if (is_coplanar(fb[tid * 2].w)) {
                        inside = csg_enclosure(ph.p0, ph.dir, rng, hd, tid, ff);
                    } else if (ff && bm >= 1 && bm <= 64) {
                        inside = tagset_add(inside, bm);
                    }
                }

                hd = -1.0f;
            } else if (hd >= 0) {
                hd -= SAFETY_DIST;
            }
        }

        ph.mstate = csg_state(inside);
        ph.pending = 0;
        ph.mid = ms_get(ph.mstate, 0);

        if (ph.mid == 0) {
            if ((hd < 0 && !probe_ray(ph.p0, ph.dir, 0.0f, rng, hd, tid, ff)) || !ff) {
                ph.mid = MEDIUM_DEAD;
                return false;
            }

            unpack_media(fb[tid * 2].w, fm, bm);
            ph.mstate = ms_enter(ph.mstate, bm);
            ph.mid = ms_get(ph.mstate, 0);

            if (!enter_domain(ph, rng, ph.p0 + ph.dir * hd, hd, tid)) {
                ph.mstate = make_uint2(0, 0);
                ph.mid = MEDIUM_DEAD;
            }
        }

        return true;
    }

#endif

    if (ph.mid != MEDIUM_UNKNOWN) {
        return true;
    }

    if (!probe_ray(o, ph.dir, 0.0f, rng, hd, tid, ff)) {
        ph.mid = MEDIUM_DEAD;
        return false;
    }

    unpack_media(fb[tid * 2].w, fm, bm);
    float3 hp = o + ph.dir * hd;

    if (hd < 2.0f * SAFETY_DIST) {
        /* on the face: the photon goes from near to far (ff: front to back) */
        unsigned int near = ff ? fm : bm, far = ff ? bm : fm;
        ph.p0 = (near != MEDIUM_AMBIENT) ? hp - ph.dir * SAFETY_DIST : hp + ph.dir * SAFETY_DIST;
        ph.mid = (near != MEDIUM_AMBIENT) ? near : far;
        return ph.mid != MEDIUM_AMBIENT;
    }

    ph.mid = ff ? fm : bm;

    /* source outside the mesh, pointing at it: enter through the hit face */
    if (ph.mid == MEDIUM_AMBIENT) {
        if (!ff) {
            return false;
        }

        ph.mid = bm;

        if (!enter_domain(ph, rng, hp, hd - SAFETY_DIST, tid)) {
            ph.mid = MEDIUM_DEAD;
        }
    }

    return true;
}

/* launch a photon and find its medium; a photon missing the domain is launched again, see vkmmc_core.comp */
__device__ void start_photon(Photon& ph, uint4& rng, float& w0, uint2& cache, bool& cached) {
    if ((gcfg.srctype == SRC_PENCIL && gcfg.extra_src_len == 0 && gcfg.focus_mode == 0) || gcfg.initial_medium != MEDIUM_UNKNOWN) {
        launch_photon(ph, rng, w0);
        initial_medium(ph, rng, cache, cached);
        return;
    }

    for (int k = 0; k < 1000; k++) {
        launch_photon(ph, rng, w0);

        if (initial_medium(ph, rng, cache, cached)) {
            break;
        }
    }
}

/* ================================================================ */
/*                          Raygen                                  */
/* ================================================================ */

extern "C" __global__ void __raygen__rg() {
    unsigned int tid = optixGetLaunchIndex().x;
    uint4 rng = ((uint4*)gcfg.seedbuf)[tid];
    uint2 cache = make_uint2(0, 0);
    bool cached = false;

    int done = 0;
    int maxph = gcfg.threadphoton + (tid < (unsigned int)gcfg.oddphoton ? 1 : 0);
    float w0;

    Photon ph;
    start_photon(ph, rng, w0, cache, cached);

    if (maxph > 0) {
        tally_launch(w0);
    }

    while (done < maxph) {
#ifdef USE_CSG

        if (ph.pending) {
            finish_coplanar_crossing(ph, rng);
        }

#endif

        if (ph.mid != MEDIUM_DEAD && ph.mid != MEDIUM_AMBIENT && ph.tof < gcfg.tend) {
            float mus = gcfg.media[ph.mid].mus;
            traceRay(ph, rng, mus > 0.0f ? ph.slen / mus : std::numeric_limits<float>::max());
        } else {
            start_photon(ph, rng, w0, cache, cached);

            if (++done < maxph) {
                tally_launch(w0);
            }
        }
    }

    ((uint4*)gcfg.seedbuf)[tid] = rng;
}

/* ================================================================ */
/*                        Closest-hit                               */
/* ================================================================ */

extern "C" __global__ void __closesthit__ch() {
    Photon ph = getPhoton();
    uint4 rng = getRNG();

    float hitlen = optixGetRayTmax();
    int primid = optixGetPrimitiveIndex();
    bool ff = optixIsFrontFaceHit();

    float4* fb = (float4*)gcfg.facebuf;
    float4 fn4 = fb[primid * 2];
    unsigned int fm, bm;
    unpack_media(fn4.w, fm, bm);

    /* ---- Probe ray: report the hit only ---- */
    if (ph.mid == MEDIUM_PROBE) {
        ph.slen = hitlen;
        ph.weight = __uint_as_float((unsigned int)primid);
        ph.tof = ff ? 1.0f : 0.0f;
        ph.mid = MEDIUM_PROBE_HIT;
        setPhoton(ph);
        return;
    }

    /* ---- Normal transport ---- */
    Medium prop = gcfg.media[ph.mid];
    float L = fminf(hitlen, prop.mus > 0.0f ? ph.slen / prop.mus : hitlen);

    accumulate(ph, prop, L);

    float3 hit = ph.p0 + ph.dir * L;
    float3 Ng = make_float3(fn4.x, fn4.y, fn4.z);
    ph.p0 = hit + ((dot(ph.dir, Ng) > 0) ? Ng : -Ng) * SAFETY_DIST;   /* SAFETY_DIST past the facet */
    ph.weight *= expf(-prop.mua * L);
    ph.tof += L * INV_C0 * prop.n;
    ph.slen -= L * prop.mus;

#ifdef USE_CSG

    if (gcfg.do_csg && is_coplanar(fn4.w)) {
        ph.pending = 1;          /* raygen re-derives the enclosing shapes */
        ph.primid = (unsigned int)primid;
    } else if (gcfg.do_csg) {
        unsigned int old_mid = ph.mid;
        uint2 new_mstate = ff ? ms_enter(ph.mstate, bm) : ms_leave(ph.mstate, bm);
        unsigned int new_mid = ms_get(new_mstate, 0);
        bool reflected = false;

        if (gcfg.do_reflect && new_mid != old_mid &&
                new_mid < gcfg.num_media && old_mid < gcfg.num_media) {
            float n_in = gcfg.media[old_mid].n;
            float n_out = gcfg.media[new_mid].n;   /* media[0] is the background */

            if (n_in != n_out) {
                reflected = do_reflection(get_normal(primid, hit, fn4), Ng, n_in, n_out, rng, hit, ph);
            }
        }

        if (!reflected) {
            ph.mstate = new_mstate;
            ph.mid = new_mid;

            if (ph.mid == 0) {
                ph.mid = MEDIUM_DEAD;
            }
        }
    } else
#endif
    {
        unsigned int new_mid = ff ? bm : fm;

        if (gcfg.do_reflect && prop.n != gcfg.media[new_mid].n) {
            if (do_reflection(get_normal(primid, hit, fn4), Ng, prop.n, gcfg.media[new_mid].n, rng, hit, ph)) {
                new_mid = ph.mid;
            }
        }

        ph.mid = new_mid;

        if (ph.mid == MEDIUM_AMBIENT) {
            ph.mid = MEDIUM_DEAD;
        }
    }

    setPhoton(ph);
    setRNG(rng);
}

/* ================================================================ */
/*                            Miss                                  */
/* ================================================================ */

extern "C" __global__ void __miss__ms() {
    Photon ph = getPhoton();
    uint4 rng = getRNG();

    Medium prop = gcfg.media[ph.mid >= gcfg.num_media ? 0 : ph.mid];

    /* a probe that hit nothing, or a ray leaving a non-scattering medium */
    if (ph.mid >= gcfg.num_media || prop.mus <= 0.0f) {
        ph.mid = MEDIUM_DEAD;
        setPhoton(ph);
        return;
    }

    float L = ph.slen / prop.mus;

    accumulate(ph, prop, L);

    ph.p0 = ph.p0 + ph.dir * L;
    ph.weight *= expf(-prop.mua * L);
    ph.tof += L * INV_C0 * prop.n;

    // Russian roulette
    if (ph.weight < gcfg.minenergy) {
        if (rand01(rng) <= 1.0f / gcfg.roulettesize) {
            ph.weight *= gcfg.roulettesize;
        } else {
            ph.mid = MEDIUM_DEAD;
            setPhoton(ph);
            setRNG(rng);
            return;
        }
    }

    ph.dir = scatter_dir(ph.dir, prop.g, rng);
    ph.slen = rand_scatlen(rng);

    setPhoton(ph);
    setRNG(rng);
}

/* ================================================================ */
/*          Ray type 1: enumerate coplanar triangles                */
/* ================================================================ */

#define AH_STORE(k) case k: optixSetPayload_##k(rec); break;

extern "C" __global__ void __anyhit__coplanar() {
    unsigned int n = optixGetPayload_0();

    if (n < 8) {
        unsigned int rec = (unsigned int)optixGetPrimitiveIndex() | (optixIsFrontFaceHit() ? 0x80000000u : 0u);

        switch (n + 1) {
                AH_STORE(1)
                AH_STORE(2)
                AH_STORE(3)
                AH_STORE(4)
                AH_STORE(5)
                AH_STORE(6)
                AH_STORE(7)
                AH_STORE(8)
        }

        optixSetPayload_0(n + 1);
    }

    optixIgnoreIntersection();
}

extern "C" __global__ void __miss__coplanar() {
}
