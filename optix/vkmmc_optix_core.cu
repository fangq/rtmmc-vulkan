/*
 * vkmmc_optix_core.cu — OptiX 7 MC photon transport
 *
 * Compile-time flags:
 *   USE_CSG       — enable CSG shape-based media state tracking
 *   USE_CURVATURE — enable curvature-corrected normals (requires USE_CSG)
 */

#include <optix.h>
#include <optix_device.h>
#include <sutil/vec_math.h>
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
    Medium media[MAX_PROP];
};

extern "C" {
    __constant__ VKMMCParam gcfg;
}

#define INV_C0 3.335640951981520e-12f
#define SAFETY_DIST 0.001f
#define TWO_PI 6.28318530717959f
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
                atomic_add_out((unsigned int)gi * gs.z + (unsigned int)iv.z * gs.y + (unsigned int)iv.y * gs.x + (unsigned int)iv.x,
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
            atomic_add_out((unsigned int)gi * gs.z + (unsigned int)iv.z * gs.y + (unsigned int)iv.y * gs.x + (unsigned int)iv.x, dep);
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

__device__ void launch_photon(Photon& ph, uint4& rng) {
    ph.p0 = gcfg.srcpos;
    ph.dir = gcfg.srcdir;
    ph.weight = 1.0f;
    ph.tof = 0;
#ifdef USE_CSG
    ph.mstate = make_uint2(0, 0);
    ph.pending = 0;
    ph.primid = 0;
#endif

    if (gcfg.srctype == 8) { // disk
        float r0 = sqrtf(rand01(rng)) * gcfg.srcparam1.x;
        float phi = TWO_PI * rand01(rng);
        float cp, sp;
        sincosf(phi, &sp, &cp);
        float3 sd = gcfg.srcdir;

        if (sd.z > -1 + FEPS && sd.z < 1 - FEPS) {
            float t0 = 1 - sd.z * sd.z, t1 = r0 * rsqrtf(t0);
            ph.p0.x += t1 * (sd.x * sd.z * cp - sd.y * sp);
            ph.p0.y += t1 * (sd.y * sd.z * cp + sd.x * sp);
            ph.p0.z -= t1 * t0 * cp;
        } else {
            ph.p0.x += r0 * cp;
            ph.p0.y += r0 * sp;
        }
    } else if (gcfg.srctype == 4) { // planar
        float u = rand01(rng), v = rand01(rng);
        ph.p0 = gcfg.srcpos + u * make_float3(gcfg.srcparam1.x, gcfg.srcparam1.y, gcfg.srcparam1.z)
                + v * make_float3(gcfg.srcparam2.x, gcfg.srcparam2.y, gcfg.srcparam2.z);
    }

    ph.slen = rand_scatlen(rng);
    ph.mid = gcfg.initial_medium;
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

/* tags enclosing p (bit t-1 for tag t), see csg_enclosure() in vkmmc_core.comp */
__device__ uint2 csg_enclosure(float3 p, float3 d, uint4& rng) {
    float4* fb = (float4*)gcfg.facebuf;
    uint2 seen = make_uint2(0, 0), inside = make_uint2(0, 0), ms = make_uint2(0, 0);
    float3 perp = (fabsf(d.z) < 0.999f) ? normalize(cross(d, make_float3(0, 0, 1)))
                  : normalize(cross(d, make_float3(1, 0, 0)));
    float3 wp = p + perp * 2e-4f;
    float hd;
    int tid;
    bool ff;
    unsigned int fm, bm;

    for (int wk = 0; wk < 64; wk++) {
        if (!probe_ray(wp, d, RAY_TMIN, rng, hd, tid, ff)) {
            break;
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

/*
 * Find the medium enclosing the source (same algorithm as vkmmc_core.comp).
 * CSG: a forward walk through all surfaces; the source is inside a shape when
 * the first crossing of its tag is an exit. A pencil source launches from one
 * point, so the walk is cached per thread. A source outside everything enters
 * the first surface it points at, with Fresnel reflection.
 */
__device__ void initial_medium(Photon& ph, uint4& rng, uint2& cache, bool& cached) {
    float4* fb = (float4*)gcfg.facebuf;
    float hd;
    int tid;
    bool ff;
    unsigned int fm, bm;

#ifdef USE_CSG

    if (gcfg.do_csg) {
        uint2 inside = cache;

        if (gcfg.srctype != 0 || !cached) {
            inside = csg_enclosure(ph.p0, ph.dir, rng);
            cache = inside;
            cached = true;
        }

        ph.mstate = csg_state(inside);
        ph.pending = 0;
        ph.mid = ms_get(ph.mstate, 0);

        if (ph.mid == 0) {
            if (probe_ray(ph.p0, ph.dir, 0.0f, rng, hd, tid, ff) && ff) {
                unpack_media(fb[tid * 2].w, fm, bm);
                float3 hp = ph.p0 + ph.dir * hd;
                ph.p0 = hp + ph.dir * SAFETY_DIST;
                ph.mstate = ms_enter(ph.mstate, bm);
                ph.mid = ms_get(ph.mstate, 0);

                if (gcfg.do_reflect && ph.mid < gcfg.num_media && gcfg.media[0].n != gcfg.media[ph.mid].n
                        && do_reflection(get_normal(tid, hp, fb[tid * 2]), make_float3(fb[tid * 2]), gcfg.media[0].n, gcfg.media[ph.mid].n, rng, hp, ph)) {
                    ph.mstate = make_uint2(0, 0);
                    ph.mid = MEDIUM_DEAD;
                }
            } else {
                ph.mid = MEDIUM_DEAD;
            }
        }

        return;
    }

#endif

    if (ph.mid != MEDIUM_UNKNOWN) {
        return;
    }

    if (!probe_ray(ph.p0, ph.dir, 0.0f, rng, hd, tid, ff)) {
        ph.mid = MEDIUM_DEAD;
        return;
    }

    unpack_media(fb[tid * 2].w, fm, bm);
    ph.mid = ff ? fm : bm;

    /* source outside the mesh, pointing at it: enter through the hit face */
    if (ph.mid == MEDIUM_AMBIENT && ff) {
        float3 hp = ph.p0 + ph.dir * hd;
        ph.p0 = hp + ph.dir * SAFETY_DIST;
        ph.mid = bm;

        if (gcfg.do_reflect && gcfg.media[0].n != gcfg.media[bm].n
                && do_reflection(get_normal(tid, hp, fb[tid * 2]), make_float3(fb[tid * 2]), gcfg.media[0].n, gcfg.media[bm].n, rng, hp, ph)) {
            ph.mid = MEDIUM_DEAD;
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

    Photon ph;
    launch_photon(ph, rng);
    initial_medium(ph, rng, cache, cached);

    int done = 0;
    int maxph = gcfg.threadphoton + (tid < (unsigned int)gcfg.oddphoton ? 1 : 0);

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
            launch_photon(ph, rng);
            initial_medium(ph, rng, cache, cached);
            ++done;
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
