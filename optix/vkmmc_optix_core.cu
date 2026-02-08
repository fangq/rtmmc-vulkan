/*
 * vkmmc_optix_core.cu — OptiX 7 surface-based MC with CSG & curvature
 *
 * Payload layout (14 registers, matching existing MMC OptiX):
 *   0-2:  p0.xyz          (position)
 *   3-5:  dir.xyz         (direction)
 *   6:    slen            (remaining scattering length)
 *   7:    weight
 *   8:    photontimer     (time of flight)
 *   9:    mediumid | (mstate_hi << 16)  — pack mid + partial CSG state
 *  10-11: mstate.x, mstate.y           — CSG media state buffer
 *  12-15: rng (4 x uint32)
 *
 * Uses 16 payload values. Set numPayloadValues=16 in pipeline options.
 * For OptiX < 7.5, reduce to 14 by storing RNG in global memory.
 *
 * Compile:
 *   nvcc -ptx --expt-relaxed-constexpr -o vkmmc_optix_core.ptx \
 *        vkmmc_optix_core.cu -I$OPTIX_DIR/include -I$OPTIX_DIR/SDK \
 *        --use_fast_math -std=c++14
 */

#include <optix.h>
#include <optix_device.h>
#include <sutil/vec_math.h>
#include <limits>
#include <math.h>
#include <stdint.h>

/* ================================================================ */
/*                        Data structures                           */
/* ================================================================ */

struct Medium {
    float mua, mus, g, n;
};

struct NodeCurv {
    float4 vnorm_k1;   /* .xyz = vertex normal, .w = k1 */
    float4 pdir_k2;    /* .xyz = principal dir, .w = k2 */
    float4 node_pos;   /* .xyz = position, .w = unused */
};

#define MAX_PROP 256

struct VKMMCParam {
    OptixTraversableHandle gashandle;
    float4*    facebuf;       /* interleaved: [norm+media, vtx_indices] per tri */
    NodeCurv*  curvbuf;
    CUdeviceptr outputbuf;    /* output grid */
    CUdeviceptr seedbuf;      /* uint4 per thread */
    int     srctype;
    float3  srcpos, srcdir;
    float4  srcparam1, srcparam2;
    float3  grid_min, grid_extent;
    uint4   grid_stride;      /* .x=nx, .y=nx*ny, .z=nx*ny*nz, .w=crop0w */
    float   voxel_scale;
    float   tstart, tend;
    float   inv_timestep;
    int     maxgate;
    unsigned int initial_medium, do_reflect;
    int     output_type;
    unsigned int num_media, do_csg, has_curvature;
    int     threadphoton, oddphoton;
    Medium  media[MAX_PROP];
};

extern "C" { __constant__ VKMMCParam gcfg; }

/* ================================================================ */
/*                         Constants                                */
/* ================================================================ */

#define INV_C0       3.335640951981520e-12f
#define SAFETY_DIST  0.001f
#define DOUBLE_SAFETY 0.002f
#define TWO_PI       6.28318530717959f
#define FEPS         1.19209290E-07f
#define RAY_TMIN     1e-5f
#define MEDIUM_UNKNOWN 0xFFFFFFFFu
#define MEDIUM_DEAD    0xFFFFFFFEu
#define MEDIUM_AMBIENT 0u
#define OUT_FLUX     0
#define OUT_FLUENCE  1
#define OUT_ENERGY   2

/* ================================================================ */
/*                     RNG (xorshift128+)                           */
/* ================================================================ */

__device__ __forceinline__ float rand01(uint4& st) {
    union { unsigned long long i; float f[2]; unsigned int u[2]; } s1;
    const unsigned long long s0 = ((unsigned long long)st.z << 32) | st.w;
    s1.i = ((unsigned long long)st.x << 32) | st.y;
    st.x = st.z; st.y = st.w;
    s1.i ^= s1.i << 23;
    unsigned long long ns = s1.i ^ s0 ^ (s1.i >> 18) ^ (s0 >> 5);
    st.z = (unsigned int)(ns >> 32); st.w = (unsigned int)ns;
    s1.i = ns + s0;
    s1.u[0] = 0x3F800000U | (s1.u[0] >> 9);
    return s1.f[0] - 1.0f;
}

__device__ __forceinline__ float rand_scatlen(uint4& rng) {
    return -logf(rand01(rng) + FEPS);
}

/* ================================================================ */
/*                     Payload get/set                              */
/* ================================================================ */

struct Photon {
    float3   p0, dir;
    float    slen, weight, tof;
    unsigned int mid;
    uint2    mstate;
};

__device__ __forceinline__ Photon getPhoton() {
    Photon ph;
    ph.p0  = make_float3(__uint_as_float(optixGetPayload_0()),
                         __uint_as_float(optixGetPayload_1()),
                         __uint_as_float(optixGetPayload_2()));
    ph.dir = make_float3(__uint_as_float(optixGetPayload_3()),
                         __uint_as_float(optixGetPayload_4()),
                         __uint_as_float(optixGetPayload_5()));
    ph.slen   = __uint_as_float(optixGetPayload_6());
    ph.weight = __uint_as_float(optixGetPayload_7());
    ph.tof    = __uint_as_float(optixGetPayload_8());
    ph.mid    = optixGetPayload_9();
    ph.mstate = make_uint2(optixGetPayload_10(), optixGetPayload_11());
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
    optixSetPayload_10(ph.mstate.x);
    optixSetPayload_11(ph.mstate.y);
}

__device__ __forceinline__ uint4 getRNG() {
    return make_uint4(optixGetPayload_12(), optixGetPayload_13(),
                      optixGetPayload_14(), optixGetPayload_15());
}

__device__ __forceinline__ void setRNG(const uint4& s) {
    optixSetPayload_12(s.x); optixSetPayload_13(s.y);
    optixSetPayload_14(s.z); optixSetPayload_15(s.w);
}

/* Helper: trace with 16 payloads */
__device__ __forceinline__ void traceRay(Photon& ph, uint4& rng, float tmax) {
    unsigned int p0=__float_as_uint(ph.p0.x), p1=__float_as_uint(ph.p0.y), p2=__float_as_uint(ph.p0.z);
    unsigned int p3=__float_as_uint(ph.dir.x),p4=__float_as_uint(ph.dir.y),p5=__float_as_uint(ph.dir.z);
    unsigned int p6=__float_as_uint(ph.slen), p7=__float_as_uint(ph.weight),p8=__float_as_uint(ph.tof);
    unsigned int p9=ph.mid, p10=ph.mstate.x, p11=ph.mstate.y;
    unsigned int p12=rng.x, p13=rng.y, p14=rng.z, p15=rng.w;

    optixTrace(gcfg.gashandle, ph.p0, ph.dir, RAY_TMIN, tmax, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0,
               p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15);

    ph.p0  = make_float3(__uint_as_float(p0),__uint_as_float(p1),__uint_as_float(p2));
    ph.dir = make_float3(__uint_as_float(p3),__uint_as_float(p4),__uint_as_float(p5));
    ph.slen=__uint_as_float(p6); ph.weight=__uint_as_float(p7); ph.tof=__uint_as_float(p8);
    ph.mid=p9; ph.mstate=make_uint2(p10,p11);
    rng = make_uint4(p12,p13,p14,p15);
}

/* ================================================================ */
/*                     Media state (CSG)                            */
/* ================================================================ */

__device__ __forceinline__ unsigned int ms_get(uint2 ms, int slot) {
    unsigned int w = (slot < 4) ? ms.x : ms.y;
    return (w >> ((slot & 3) * 8)) & 0xFFu;
}
__device__ __forceinline__ uint2 ms_set(uint2 ms, int slot, unsigned int val) {
    unsigned int sh = (slot & 3) * 8, mk = 0xFFu << sh;
    if (slot < 4) ms.x = (ms.x & ~mk) | ((val & 0xFFu) << sh);
    else          ms.y = (ms.y & ~mk) | ((val & 0xFFu) << sh);
    return ms;
}
__device__ __forceinline__ uint2 ms_enter(uint2 ms, unsigned int bid) {
    if (!bid) return ms;
    unsigned int cm = ms_get(ms,0), nm = max(cm,bid), dm = (nm==bid)?cm:bid;
    ms = ms_set(ms,0,nm);
    if (dm) for (int i=1;i<8;i++) if (!ms_get(ms,i)){ms=ms_set(ms,i,dm);break;}
    return ms;
}
__device__ __forceinline__ uint2 ms_leave(uint2 ms, unsigned int bid) {
    if (!bid) return ms;
    for (int i=0;i<8;i++) if (ms_get(ms,i)==bid){ms=ms_set(ms,i,0);break;}
    unsigned int mx=0; int ms2=-1;
    for (int i=0;i<8;i++){unsigned int v=ms_get(ms,i);if(v>mx){mx=v;ms2=i;}}
    if (ms2>0){ms=ms_set(ms,ms2,0);ms=ms_set(ms,0,mx);}
    else if (ms2<0) ms=ms_set(ms,0,0);
    return ms;
}

/* ================================================================ */
/*                     Face buffer helpers                          */
/* ================================================================ */

__device__ __forceinline__ void unpack_media(float pw, unsigned int& fm, unsigned int& bm) {
    unsigned int p = __float_as_uint(pw); fm = p >> 16; bm = p & 0xFFFFu;
}

/* ================================================================ */
/*                    Curvature normal                              */
/* ================================================================ */

__device__ __forceinline__ float3 get_curvature_normal(int tri_id, float3 hit_pos) {
    float4 vidx_raw = gcfg.facebuf[tri_id * 2 + 1];
    float3 NB_sum = make_float3(0,0,0);
    float* vr = (float*)&vidx_raw;

    for (int vi = 0; vi < 3; vi++) {
        unsigned int idx = __float_as_uint(vr[vi]);
        NodeCurv nc = gcfg.curvbuf[idx];
        float3 Ni = make_float3(nc.vnorm_k1.x, nc.vnorm_k1.y, nc.vnorm_k1.z);
        float  k1 = nc.vnorm_k1.w;
        float3 ui = make_float3(nc.pdir_k2.x, nc.pdir_k2.y, nc.pdir_k2.z);
        float  k2 = nc.pdir_k2.w;
        float3 Pi = make_float3(nc.node_pos.x, nc.node_pos.y, nc.node_pos.z);
        float3 vi2 = cross(ui, Ni);
        float3 di = hit_pos - Pi;
        float du = dot(di, ui), dv = dot(di, vi2);
        NB_sum = NB_sum + Ni + k1*du*ui + k2*dv*vi2;
    }

    float3 NB = normalize(NB_sum);
    float4 fn4 = gcfg.facebuf[tri_id * 2];
    if (dot(NB, make_float3(fn4.x, fn4.y, fn4.z)) < 0.0f) NB = -NB;
    return NB;
}

/* ================================================================ */
/*                     Output accumulation                          */
/* ================================================================ */

__device__ __forceinline__ void atomic_add_out(unsigned int idx, float val) {
    float accum = atomicAdd(&((float*)gcfg.outputbuf)[idx], val);
    if (accum > 1000.f) {
        if (atomicAdd(&((float*)gcfg.outputbuf)[idx], -accum) < 0.0f)
            atomicAdd(&((float*)gcfg.outputbuf)[idx], accum);
        else
            atomicAdd(&((float*)gcfg.outputbuf)[idx + gcfg.grid_stride.w], accum);
    }
}

__device__ __forceinline__ unsigned int voxel_index(float3 rp) {
    unsigned int ix = rp.x>0 ? __float2uint_rd(fminf(rp.x,gcfg.grid_extent.x)*gcfg.voxel_scale) : 0;
    unsigned int iy = rp.y>0 ? __float2uint_rd(fminf(rp.y,gcfg.grid_extent.y)*gcfg.voxel_scale) : 0;
    unsigned int iz = rp.z>0 ? __float2uint_rd(fminf(rp.z,gcfg.grid_extent.z)*gcfg.voxel_scale) : 0;
    return iz*gcfg.grid_stride.y + iy*gcfg.grid_stride.x + ix;
}

__device__ __forceinline__ unsigned int time_offset(float tof) {
    return min(__float2int_rd((tof-gcfg.tstart)*gcfg.inv_timestep), gcfg.maxgate-1) * gcfg.grid_stride.z;
}

__device__ void accumulate(const Photon& ph, const Medium& prop, float L) {
    int sc = ((int)(L*gcfg.voxel_scale)+1)<<1;
    float sl = L/sc, decay = expf(-prop.mua*sl);
    float loss = (gcfg.output_type==OUT_ENERGY) ? ph.weight*(1-decay)
                 : (prop.mua>0 ? ph.weight*(1-decay)/prop.mua : 0);
    float3 step = sl*ph.dir;
    float3 sm = ph.p0 - gcfg.grid_min + 0.5f*step;
    float ct = ph.tof + sl*INV_C0*prop.n;
    unsigned int oe = time_offset(ct)+voxel_index(sm);
    float ow = loss;
    for (int i=1;i<sc;++i){
        loss*=decay; sm=sm+step; ct+=sl*INV_C0*prop.n;
        unsigned int ne=time_offset(ct)+voxel_index(sm);
        if (ne!=oe){atomic_add_out(oe,ow);oe=ne;ow=0;}
        ow+=loss;
    }
    atomic_add_out(oe,ow);
}

/* ================================================================ */
/*                     Scattering                                   */
/* ================================================================ */

__device__ float3 scatter_dir(float3 d, float g, uint4& rng) {
    float ct;
    if (fabsf(g)>FEPS){float t=(1-g*g)/(1-g+2*g*rand01(rng));
        ct=fmaxf(-1.f,fminf(1.f,(1+g*g-t*t)/(2*g)));}
    else ct=2*rand01(rng)-1;
    float st=sinf(acosf(ct));
    float phi=TWO_PI*rand01(rng);
    float sp,cp; sincosf(phi,&sp,&cp);
    if (d.z>-1+FEPS && d.z<1-FEPS){
        float s2=1-d.z*d.z, isc=st*rsqrtf(s2);
        return isc*(cp*make_float3(d.x*d.z,d.y*d.z,-s2)+sp*make_float3(-d.y,d.x,0))+ct*d;
    }
    return make_float3(st*cp, st*sp, d.z>0?ct:-ct);
}

/* ================================================================ */
/*                     Fresnel reflection                           */
/* ================================================================ */

__device__ bool do_reflection(float3 N, float n1, float n2, uint4& rng, Photon& ph) {
    float cid=dot(ph.dir,N);
    if (cid>0){N=-N;cid=-cid;} float ci=-cid;
    float n12=n1*n1, n22=n2*n2, ct2=1-n12/n22*(1-ci*ci);
    if (ct2>0){
        float ct=sqrtf(ct2);
        float re=n12*ci*ci+n22*ct2, im=2*n1*n2*ci*ct, rp=(re-im)/(re+im);
        re=n22*ci*ci+n12*ct*ct; float rt=(rp+(re-im)/(re+im))*0.5f;
        if (rand01(rng)<=rt){ph.p0=ph.p0-ph.dir*DOUBLE_SAFETY;ph.dir=ph.dir+2*ci*N;}
        else{float r=n1/n2;ph.dir=r*ph.dir+(r*ci-ct)*N;ph.dir=ph.dir*rsqrtf(dot(ph.dir,ph.dir));return false;}
    } else {ph.p0=ph.p0-ph.dir*DOUBLE_SAFETY;ph.dir=ph.dir+2*ci*N;}
    ph.dir=ph.dir*rsqrtf(dot(ph.dir,ph.dir)); return true;
}

/* ================================================================ */
/*                  Launch photon                                   */
/* ================================================================ */

__device__ void launch_photon(Photon& ph, uint4& rng) {
    ph.p0=gcfg.srcpos; ph.dir=gcfg.srcdir;
    ph.weight=1.0f; ph.tof=0; ph.mstate=make_uint2(0,0);

    if (gcfg.srctype==8){ /* disk */
        float r0=sqrtf(rand01(rng))*gcfg.srcparam1.x;
        float phi=TWO_PI*rand01(rng); float cp,sp; sincosf(phi,&sp,&cp);
        float3 sd=gcfg.srcdir;
        if (sd.z>-1+FEPS&&sd.z<1-FEPS){
            float t0=1-sd.z*sd.z, t1=r0*rsqrtf(t0);
            ph.p0.x+=t1*(sd.x*sd.z*cp-sd.y*sp);
            ph.p0.y+=t1*(sd.y*sd.z*cp+sd.x*sp);
            ph.p0.z-=t1*t0*cp;
        } else {ph.p0.x+=r0*cp;ph.p0.y+=r0*sp;}
    } else if (gcfg.srctype==4){ /* planar */
        float u=rand01(rng),v=rand01(rng);
        ph.p0=gcfg.srcpos+u*make_float3(gcfg.srcparam1.x,gcfg.srcparam1.y,gcfg.srcparam1.z)
                         +v*make_float3(gcfg.srcparam2.x,gcfg.srcparam2.y,gcfg.srcparam2.z);
    }

    ph.slen = rand_scatlen(rng);

    if (gcfg.do_csg) {
        /* CSG parity walk */
        unsigned int parity = 0;
        float3 walk_pos = ph.p0;
        for (int walk=0; walk<64; walk++){
            /* Probe trace: use a temporary photon to avoid corrupting state */
            Photon probe = ph; probe.mid = MEDIUM_UNKNOWN;
            uint4 probe_rng = rng; /* don't consume real RNG */
            traceRay(probe, probe_rng, std::numeric_limits<float>::max());
            if (probe.mid == MEDIUM_DEAD) break; /* no hit */
            /* Closest-hit sets mid = MEDIUM_UNKNOWN+1 as sentinel for probe, but
               we can't easily distinguish probe from real trace in CH.
               
               ALTERNATIVE: read face buffer directly using the hit triangle.
               But we don't have the hit triangle ID from the probe.
               
               PRACTICAL APPROACH: for CSG launch, trace from source along direction,
               skip forward to first front-face hit, enter that medium. */
            break;
        }

        /* Skip forward to first entry */
        ph.mid = 0; /* outside */
        Photon entry_ph = ph; entry_ph.mid = MEDIUM_UNKNOWN;
        traceRay(entry_ph, rng, std::numeric_limits<float>::max());
        
        if (entry_ph.mid != MEDIUM_DEAD && entry_ph.mid != MEDIUM_UNKNOWN) {
            ph = entry_ph; /* CH handled entry */
        } else {
            ph.mid = MEDIUM_DEAD;
        }
    } else {
        ph.mid = gcfg.initial_medium;
        if (ph.mid == MEDIUM_UNKNOWN) {
            traceRay(ph, rng, std::numeric_limits<float>::max());
            /* CH will set mid based on first hit */
        }
    }
}

/* ================================================================ */
/*                     Ray-gen shader                               */
/* ================================================================ */

extern "C" __global__ void __raygen__rg() {
    unsigned int tid = optixGetLaunchIndex().x;

    /* Init RNG from seed buffer */
    uint4 rng = ((uint4*)gcfg.seedbuf)[tid];

    /* Launch first photon */
    Photon ph;
    launch_photon(ph, rng);

    int done = 0;
    int maxph = gcfg.threadphoton + (tid < (unsigned int)gcfg.oddphoton ? 1 : 0);

    while (done < maxph) {
        if (ph.mid != MEDIUM_DEAD && ph.mid != MEDIUM_AMBIENT) {
            Medium prop = gcfg.media[ph.mid];
            float max_dist = (prop.mus < FEPS) ?
                std::numeric_limits<float>::max() : ph.slen / prop.mus;

            traceRay(ph, rng, max_dist);
            /* CH or MS updated ph and rng via payload */
        }

        if (ph.mid == MEDIUM_DEAD || ph.mid == MEDIUM_AMBIENT || ph.tof >= gcfg.tend) {
            launch_photon(ph, rng);
            ++done;
        }
    }
}

/* ================================================================ */
/*                   Closest-hit shader                             */
/* ================================================================ */

extern "C" __global__ void __closesthit__ch() {
    Photon ph = getPhoton();
    uint4 rng = getRNG();

    float hitlen = optixGetRayTmax();
    int primid = optixGetPrimitiveIndex();
    bool ff = optixIsFrontFaceHit();

    float4 fn4 = gcfg.facebuf[primid * 2];
    unsigned int fm, bm;
    unpack_media(fn4.w, fm, bm);

    /* Handle initial medium probe (used in launch_photon) */
    if (ph.mid == MEDIUM_UNKNOWN) {
        if (gcfg.do_csg) {
            ph.p0 = ph.p0 + ph.dir * (hitlen + SAFETY_DIST);
            if (ff) {
                ph.mstate = ms_enter(ph.mstate, bm);
                ph.mid = ms_get(ph.mstate, 0);
                /* Entry refraction */
                if (gcfg.do_reflect && ph.mid < gcfg.num_media) {
                    float n_in = 1.0f, n_out = gcfg.media[ph.mid].n;
                    if (n_in != n_out) {
                        float3 N;
                        if (gcfg.has_curvature) {
                            N = get_curvature_normal(primid, ph.p0 - ph.dir*SAFETY_DIST);
                        } else {
                            N = make_float3(fn4.x, fn4.y, fn4.z);
                        }
                        if (do_reflection(N, n_in, n_out, rng, ph)) {
                            ph.mstate = make_uint2(0,0); ph.mid = MEDIUM_DEAD;
                        }
                    }
                }
            } else {
                ph.mid = MEDIUM_DEAD; /* back face first = source behind surface */
            }
        } else {
            ph.mid = ff ? fm : bm;
        }
        setPhoton(ph); setRNG(rng); return;
    }

    Medium prop = gcfg.media[ph.mid];
    bool is_transparent = (prop.mus < FEPS);
    float L = is_transparent ? hitlen : fminf(hitlen, prop.mus>0 ? ph.slen/prop.mus : hitlen);

    accumulate(ph, prop, L);
    ph.p0 = ph.p0 + ph.dir * (L + SAFETY_DIST);
    ph.weight *= expf(-prop.mua * L);
    ph.tof += L * INV_C0 * prop.n;
    if (!is_transparent) ph.slen -= L * prop.mus;

    if (gcfg.do_csg) {
        unsigned int old_mid = ph.mid;
        uint2 new_mstate = ff ? ms_enter(ph.mstate, bm) : ms_leave(ph.mstate, bm);
        unsigned int new_mid = ms_get(new_mstate, 0);
        bool reflected = false;

        if (gcfg.do_reflect && new_mid != old_mid &&
                new_mid < gcfg.num_media && old_mid < gcfg.num_media) {
            float n_in = gcfg.media[old_mid].n;
            float n_out = (new_mid == 0) ? 1.0f : gcfg.media[new_mid].n;
            if (n_in != n_out) {
                float3 N;
                if (gcfg.has_curvature) {
                    N = get_curvature_normal(primid, ph.p0 - ph.dir*SAFETY_DIST);
                } else {
                    N = make_float3(fn4.x, fn4.y, fn4.z);
                }
                reflected = do_reflection(N, n_in, n_out, rng, ph);
            }
        }
        if (!reflected) {
            ph.mstate = new_mstate; ph.mid = new_mid;
            if (ph.mid == 0) ph.mid = MEDIUM_DEAD;
        }
    } else {
        unsigned int new_mid = ff ? bm : fm;
        if (gcfg.do_reflect && new_mid != MEDIUM_AMBIENT &&
                prop.n != gcfg.media[new_mid].n) {
            float3 N = make_float3(fn4.x, fn4.y, fn4.z);
            if (do_reflection(N, prop.n, gcfg.media[new_mid].n, rng, ph))
                new_mid = ff ? fm : bm;
        }
        ph.mid = new_mid;
        if (ph.mid == MEDIUM_AMBIENT) ph.mid = MEDIUM_DEAD;
    }

    setPhoton(ph); setRNG(rng);
}

/* ================================================================ */
/*                      Miss shader                                 */
/* ================================================================ */

extern "C" __global__ void __miss__ms() {
    Photon ph = getPhoton();
    uint4 rng = getRNG();

    /* Probe miss: no surface found */
    if (ph.mid == MEDIUM_UNKNOWN) {
        ph.mid = MEDIUM_DEAD;
        setPhoton(ph); setRNG(rng); return;
    }

    Medium prop = gcfg.media[ph.mid];

    if (prop.mus < FEPS) {
        ph.mid = MEDIUM_DEAD; /* transparent, no boundary */
    } else if (gcfg.do_csg && ms_get(ph.mstate, 0) == 0) {
        ph.mid = MEDIUM_DEAD;
    } else {
        float L = ph.slen / prop.mus;
        accumulate(ph, prop, L);
        ph.p0 = ph.p0 + ph.dir * L;
        ph.weight *= expf(-prop.mua * L);
        ph.tof += L * INV_C0 * prop.n;
        ph.dir = scatter_dir(ph.dir, prop.g, rng);
        ph.slen = rand_scatlen(rng);
    }

    setPhoton(ph); setRNG(rng);
}