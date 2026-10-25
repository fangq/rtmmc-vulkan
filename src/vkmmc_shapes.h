/*
 * vkmmc_shapes.h — Generate watertight triangle meshes from JSON shape constructs
 *
 * Each shape generates a closed surface. All triangles store front=0, back=Tag.
 * All triangle windings follow right-hand rule with normal pointing OUTWARD
 * from the shape interior.
 *
 * Box:      8 nodes, 12 triangles (6 quads)
 * Sphere:   lat/lon grid, (nlat-1)*nlon*2 triangles + 2*nlon cap triangles
 * Cylinder: side quads + center-fan caps
 *
 * Flat-flag: bit 31 of packed_media is set for geometrically flat triangles
 * (boxes, slabs, cylinder caps). When has_curvature=1, the shader skips
 * curvature normal correction for these triangles and uses the face normal.
 */
#ifndef VKMMC_SHAPES_H
#define VKMMC_SHAPES_H

#include "vkmmc_io.h"
#include <vector>
#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <cstdio>
#include <algorithm>
#include <map>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

struct ShapeMesh {
    std::vector<Vec3> nodes;
    std::vector<std::array<uint32_t, 3> > faces;
    std::vector<FaceData> facedata;
    std::vector<uint32_t> shape_id;
};

/* ---- helpers ---- */
static float pack_tag(uint32_t tag) {
    uint32_t pk = tag & 0xFFFFu;
    float f;
    std::memcpy(&f, &pk, 4);
    return f;
}

/* pack tag with bit-31 set to mark triangle as geometrically flat */
static float pack_tag_flat(uint32_t tag) {
    uint32_t pk = (tag & 0xFFFFu) | VKMMC_FACE_FLAT;
    float f;
    std::memcpy(&f, &pk, 4);
    return f;
}

/*
 * add_tri_toward: add one triangle, winding it so that its normal has a
 * non-negative component along the outward reference direction ref.
 * front=0, back=tag (CSG convention).
 */
static void add_tri_toward(ShapeMesh& m, uint32_t a, uint32_t b, uint32_t c,
                           uint32_t tag, Vec3 ref, bool flat = false) {
    Vec3 va = m.nodes[a], vb = m.nodes[b], vc = m.nodes[c];
    Vec3 e1 = {vb.x - va.x, vb.y - va.y, vb.z - va.z};
    Vec3 e2 = {vc.x - va.x, vc.y - va.y, vc.z - va.z};
    Vec3 n = v3cross(e1, e2);
    float l = v3len(n);

    if (l > 0) {
        n.x /= l;
        n.y /= l;
        n.z /= l;
    }

    if (n.x * ref.x + n.y * ref.y + n.z * ref.z < 0) {
        /* swap b and c to reverse winding */
        m.faces.push_back({{a, c, b}});
        n.x = -n.x;
        n.y = -n.y;
        n.z = -n.z;
    } else {
        m.faces.push_back({{a, b, c}});
    }

    FaceData fd;
    fd.nx = n.x;
    fd.ny = n.y;
    fd.nz = n.z;
    fd.packed_media = flat ? pack_tag_flat(tag) : pack_tag(tag);
    m.facedata.push_back(fd);
}

/*
 * add_tri_outward: add one triangle with its normal pointing away from
 * 'center' (correct for convex shapes such as boxes, spheres and cylinders).
 */
static void add_tri_outward(ShapeMesh& m, uint32_t a, uint32_t b, uint32_t c,
                            uint32_t tag, Vec3 center, bool flat = false) {
    Vec3 va = m.nodes[a], vb = m.nodes[b], vc = m.nodes[c];
    Vec3 ref = {(va.x + vb.x + vc.x) / 3.0f - center.x, (va.y + vb.y + vc.y) / 3.0f - center.y,
                (va.z + vb.z + vc.z) / 3.0f - center.z
               };
    add_tri_toward(m, a, b, c, tag, ref, flat);
}

/* add quad as 2 triangles with outward normals */
static void add_quad_outward(ShapeMesh& m, uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                             uint32_t tag, Vec3 center, bool flat = false) {
    add_tri_outward(m, a, b, c, tag, center, flat);
    add_tri_outward(m, a, c, d, tag, center, flat);
}

/* ================================================================ */
/*  Box: 8 nodes, 12 triangles, outward normals — all flat          */
/* ================================================================ */
static void gen_box(ShapeMesh& m, float ox, float oy, float oz,
                    float sx, float sy, float sz, uint32_t tag) {
    uint32_t b = (uint32_t)m.nodes.size();
    Vec3 v[8] = {{ox, oy, oz}, {ox + sx, oy, oz}, {ox + sx, oy + sy, oz}, {ox, oy + sy, oz},
        {ox, oy, oz + sz}, {ox + sx, oy, oz + sz}, {ox + sx, oy + sy, oz + sz}, {ox, oy + sy, oz + sz}
    };

    for (int i = 0; i < 8; i++) {
        m.nodes.push_back(v[i]);
    }

    /* Use box center for outward orientation check */
    Vec3 center = {ox + sx * 0.5f, oy + sy * 0.5f, oz + sz * 0.5f};

    add_quad_outward(m, b + 0, b + 3, b + 2, b + 1, tag, center, true); /* -Z */
    add_quad_outward(m, b + 4, b + 5, b + 6, b + 7, tag, center, true); /* +Z */
    add_quad_outward(m, b + 0, b + 1, b + 5, b + 4, tag, center, true); /* -Y */
    add_quad_outward(m, b + 2, b + 3, b + 7, b + 6, tag, center, true); /* +Y */
    add_quad_outward(m, b + 0, b + 4, b + 7, b + 3, tag, center, true); /* -X */
    add_quad_outward(m, b + 1, b + 2, b + 6, b + 5, tag, center, true); /* +X */
}

/* ================================================================ */
/*  Sphere: lat/lon subdivision, watertight — all curved            */
/*  All triangles wound with outward normals verified against center */
/* ================================================================ */
static void gen_sphere(ShapeMesh& m, float cx, float cy, float cz,
                       float R, uint32_t tag, int nlon = 24, int nlat = 16) {
    uint32_t b = (uint32_t)m.nodes.size();
    const float PI = 3.14159265358979f;
    Vec3 center = {cx, cy, cz};

    /* north pole */
    m.nodes.push_back({cx, cy, cz + R});

    /* latitude rings (1..nlat-1) */
    for (int i = 1; i < nlat; i++) {
        float phi = PI * (float)i / (float)nlat;
        float sp = std::sin(phi), cp = std::cos(phi);

        for (int j = 0; j < nlon; j++) {
            float theta = 2.0f * PI * (float)j / (float)nlon;
            m.nodes.push_back({cx + R* sp * std::cos(theta), cy + R* sp * std::sin(theta), cz + R * cp});
        }
    }

    /* south pole */
    uint32_t south_pole = (uint32_t)m.nodes.size();
    m.nodes.push_back({cx, cy, cz - R});

    /* north cap: fan from pole to first ring */
    for (int j = 0; j < nlon; j++) {
        uint32_t j1 = (j + 1) % nlon;
        add_tri_outward(m, b, b + 1 + j, b + 1 + j1, tag, center);
    }

    /* body: quads between adjacent rings */
    for (int i = 0; i < nlat - 2; i++) {
        uint32_t r0 = b + 1 + i * nlon;
        uint32_t r1 = b + 1 + (i + 1) * nlon;

        for (int j = 0; j < nlon; j++) {
            uint32_t j1 = (j + 1) % nlon;
            add_quad_outward(m, r0 + j, r1 + j, r1 + j1, r0 + j1, tag, center);
        }
    }

    /* south cap: fan from last ring to south pole */
    uint32_t last_ring = b + 1 + (nlat - 2) * nlon;

    for (int j = 0; j < nlon; j++) {
        uint32_t j1 = (j + 1) % nlon;
        add_tri_outward(m, last_ring + j, south_pole, last_ring + j1, tag, center);
    }
}

/* ================================================================ */
/*  Cylinder: side quads (curved) + center-fan caps (flat)          */
/*  All triangles wound with outward normals verified against center */
/* ================================================================ */
static void gen_cylinder(ShapeMesh& m, float c0x, float c0y, float c0z,
                         float c1x, float c1y, float c1z, float R,
                         uint32_t tag, int nseg = 32) {
    float ax = c1x - c0x, ay = c1y - c0y, az = c1z - c0z;
    float al = std::sqrt(ax * ax + ay * ay + az * az);

    if (al < 1e-10f) {
        return;
    }

    ax /= al;
    ay /= al;
    az /= al;

    /* Shape center for outward normal verification */
    Vec3 center = {(c0x + c1x) * 0.5f, (c0y + c1y) * 0.5f, (c0z + c1z) * 0.5f};

    /* perpendicular basis vectors u, v */
    float ux, uy, uz;

    if (std::fabs(az) > 0.9f) {
        ux = 1;
        uy = 0;
        uz = 0;
    } else {
        ux = 0;
        uy = 0;
        uz = 1;
    }

    /* u = normalize(cross(axis, up)) */
    float tx = ay * uz - az * uy, ty = az * ux - ax * uz, tz = ax * uy - ay * ux;
    float tl = std::sqrt(tx * tx + ty * ty + tz * tz);
    tx /= tl;
    ty /= tl;
    tz /= tl;
    /* v = cross(axis, u) */
    float vx = ay * tz - az * ty, vy = az * tx - ax * tz, vz = ax * ty - ay * tx;

    uint32_t b = (uint32_t)m.nodes.size();
    const float TWO_PI = 6.28318530718f;

    /* ring at c0, ring at c1 */
    for (int ring = 0; ring < 2; ring++) {
        float px = (ring == 0) ? c0x : c1x, py = (ring == 0) ? c0y : c1y, pz = (ring == 0) ? c0z : c1z;

        for (int i = 0; i < nseg; i++) {
            float a = TWO_PI * (float)i / (float)nseg;
            float ca = std::cos(a), sa = std::sin(a);
            m.nodes.push_back({px + R * (ca * tx + sa * vx), py + R * (ca * ty + sa * vy), pz + R * (ca * tz + sa * vz)});
        }
    }

    /* center of caps */
    uint32_t cap0 = (uint32_t)m.nodes.size();
    m.nodes.push_back({c0x, c0y, c0z});
    uint32_t cap1 = (uint32_t)m.nodes.size();
    m.nodes.push_back({c1x, c1y, c1z});

    /* side quads — curved surface */
    for (int i = 0; i < nseg; i++) {
        int n = (i + 1) % nseg;
        add_quad_outward(m, b + i, b + n, b + nseg + n, b + nseg + i, tag, center, false);
    }

    /* bottom cap (c0): flat */
    for (int i = 0; i < nseg; i++) {
        int n = (i + 1) % nseg;
        add_tri_outward(m, cap0, b + n, b + i, tag, center, true);
    }

    /* top cap (c1): flat */
    for (int i = 0; i < nseg; i++) {
        int n = (i + 1) % nseg;
        add_tri_outward(m, cap1, b + nseg + i, b + nseg + n, tag, center, true);
    }
}

/* ================================================================ */
/*  Slab: axis-aligned slab = box between two bounds (flat)         */
/* ================================================================ */
static void gen_slab(ShapeMesh& m, int dir, float lo, float hi,
                     float ext[6], uint32_t tag) {
    float ox = ext[0], sx = ext[1] - ext[0], oy = ext[2], sy = ext[3] - ext[2], oz = ext[4], sz = ext[5] - ext[4];

    if (dir == 0) {
        gen_box(m, lo, oy, oz, hi - lo, sy, sz, tag);
    } else if (dir == 1) {
        gen_box(m, ox, lo, oz, sx, hi - lo, sz, tag);
    } else {
        gen_box(m, ox, oy, lo, sx, sy, hi - lo, tag);
    }
}

/* Returns number of ShapeOrigin entries contributed by each shape type.
   Must stay in sync with extract_shape_origins() in vkmmc_curvature.h. */
static int shape_origin_count(const std::string& key) {
    if (key == "Lens") {
        return 2;    /* front spherical cap + back spherical cap */
    }

    return 1;
}

/* ================================================================ */
/*  Lens: two spherical caps + cylindrical edge ring                */
/*                                                                  */
/*  Front sphere center: O + n*(Ff-Df),  radius Rc_f = |Ff-2*Df|  */
/*  Back  sphere center: O - n*(Fb-Db),  radius Rc_b = |Fb-2*Db|  */
/*                                                                  */
/*  front_oid / back_oid: 1-based origin IDs written to shape_id   */
/* ================================================================ */
static void gen_lens(ShapeMesh& m,
                     float ox, float oy, float oz,
                     float ddx, float ddy, float ddz,
                     float R,
                     float Df, float Rf,   /* front: apex dist, sphere radius (signed) */
                     float Db, float Rb,   /* back:  apex dist, sphere radius (signed) */
                     uint32_t tag,
                     uint32_t front_oid, uint32_t back_oid,
                     int nseg = 32, int nrings = 12) {
    float dl = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);

    if (dl < 1e-10f || nseg < 3 || nrings < 1) {
        return;
    }

    float nx = ddx / dl, ny = ddy / dl, nz = ddz / dl;

    /* Perpendicular basis u, v */
    float ux, uy, uz;

    if (std::fabs(nz) < 0.9f) {
        float t = std::sqrt(nx * nx + ny * ny);
        ux = -ny / t;
        uy =  nx / t;
        uz = 0.0f;
    } else {
        float t = std::sqrt(ny * ny + nz * nz);
        ux = 0.0f;
        uy = -nz / t;
        uz =  ny / t;
    }

    float vx = ny * uz - nz * uy;
    float vy = nz * ux - nx * uz;
    float vz = nx * uy - ny * ux;

    /* Front sphere: center C_f = O + n*(Df-Rf), radius |Rf|
       sf = sign(Rf) so that zf_at(0) = (Df-Rf) + sf*|Rf| = Df */
    float af  = Df - Rf;
    float Rcf = std::fabs(Rf);
    float sf  = (Rf >= 0.0f) ? 1.0f : -1.0f;

    /* Back sphere: center C_b = O + n*(Rb-Db), radius |Rb|
       sb = -sign(Rb) so that zb_at(0) = (Rb-Db) + sb*|Rb| = -Db */
    float ab  = Rb - Db;
    float Rcb = std::fabs(Rb);
    float sb  = (Rb >= 0.0f) ? -1.0f : 1.0f;

    /* Edge radius: clamp to sphere radii */
    float r_edge = R;

    if (Rcf > 1e-6f) {
        r_edge = std::min(r_edge, Rcf);
    }

    if (Rcb > 1e-6f) {
        r_edge = std::min(r_edge, Rcb);
    }

    /* If both R positive (both faces convex outward), check sphere–sphere intersection */
    if (Rf > 0.0f && Rb > 0.0f && std::fabs(ab - af) > 1e-10f) {
        float z_int = (Rcf * Rcf - Rcb * Rcb - af * af + ab * ab) / (2.0f * (ab - af));
        float r2    = Rcf * Rcf - (z_int - af) * (z_int - af);

        if (r2 > 0.0f) {
            float r_int = std::sqrt(r2);

            if (r_int < R) {
                r_edge = std::min(r_edge, r_int);
            }
        }
    }

    if (r_edge < 1e-6f) {
        return;
    }

    /* z-offset from O (along n) for each face at lateral radius r */
    auto zf_at = [&](float r) -> float {
        return af + sf * std::sqrt(std::max(0.0f, Rcf * Rcf - r * r));
    };
    auto zb_at = [&](float r) -> float {
        return ab + sb * std::sqrt(std::max(0.0f, Rcb * Rcb - r * r));
    };

    float zeb = zb_at(r_edge);   /* back rim z (front rim computed per-node in ring loop) */

    /* Node at (lateral r, azimuth segment j, axial offset zoff from O) */
    const float TWO_PI = 6.28318530718f;
    auto make_node = [&](float r, int jseg, float zoff) -> Vec3 {
        float phi = TWO_PI * jseg / nseg;
        float cp  = std::cos(phi), sp2 = std::sin(phi);
        return {
            ox + nx* zoff + r* (cp * ux + sp2 * vx),
            oy + ny* zoff + r* (cp * uy + sp2 * vy),
            oz + nz* zoff + r* (cp * uz + sp2 * vz)};
    };

    uint32_t b = (uint32_t)m.nodes.size();

    /* ---- Nodes ---- */

    /* b+0: front apex */
    m.nodes.push_back({ox + nx * Df, oy + ny * Df, oz + nz * Df});

    /* b+1 .. b+nrings*nseg: front rings, ring i at r=r_edge*(i+1)/nrings */
    for (int i = 0; i < nrings; i++) {
        float r = r_edge * (i + 1) / nrings;
        float z = zf_at(r);

        for (int j = 0; j < nseg; j++) {
            m.nodes.push_back(make_node(r, j, z));
        }
    }

    /* b+1+nrings*nseg .. b+(nrings+1)*nseg: back rim (separate ring) */
    for (int j = 0; j < nseg; j++) {
        m.nodes.push_back(make_node(r_edge, j, zeb));
    }

    /* b+1+(nrings+1)*nseg .. b+2*nrings*nseg: back inner rings,
       ring 0 at r=r_edge*(nrings-1)/nrings down to r=r_edge/nrings */
    for (int i = nrings - 2; i >= 0; i--) {
        float r = r_edge * (i + 1) / nrings;
        float z = zb_at(r);

        for (int j = 0; j < nseg; j++) {
            m.nodes.push_back(make_node(r, j, z));
        }
    }

    /* b+1+2*nrings*nseg: back apex */
    uint32_t apex_b = (uint32_t)m.nodes.size();
    m.nodes.push_back({ox - nx * Db, oy - ny * Db, oz - nz * Db});

    /* Index helpers (seg wraps mod nseg) */
    auto f_apex = [&]() -> uint32_t { return b; };
    auto f_ring = [&](int ring, int seg) -> uint32_t {
        return b + 1 + (uint32_t)ring * nseg + (uint32_t)((seg % nseg + nseg) % nseg);
    };
    auto b_rim = [&](int seg) -> uint32_t {
        return b + 1 + (uint32_t)nrings * nseg + (uint32_t)((seg % nseg + nseg) % nseg);
    };
    auto b_ring = [&](int ring, int seg) -> uint32_t {
        return b + 1 + (uint32_t)(nrings + 1) * nseg
        + (uint32_t)ring * nseg + (uint32_t)((seg % nseg + nseg) % nseg);
    };

    /* ---- Faces ---- */

    size_t fi0 = m.faces.size();  /* start of front cap */

    /* Orientation references: a cap is a graph over the disk, so its outward
       normal always has a positive component along +n (front) or -n (back),
       also for concave caps; the rim band faces radially outward. */
    Vec3 nfront = {nx, ny, nz}, nback = {-nx, -ny, -nz};
    auto rim_quad = [&](uint32_t a, uint32_t b2, uint32_t c, uint32_t d) {
        const uint32_t v[4] = {a, b2, c, d};
        Vec3 r = {0, 0, 0};

        for (int k = 0; k < 4; k++) {
            r.x += m.nodes[v[k]].x * 0.25f - ox * 0.25f;
            r.y += m.nodes[v[k]].y * 0.25f - oy * 0.25f;
            r.z += m.nodes[v[k]].z * 0.25f - oz * 0.25f;
        }

        float ax = r.x * nx + r.y * ny + r.z * nz;   /* remove the axial component */
        Vec3 radial = {r.x - ax * nx, r.y - ax * ny, r.z - ax * nz};
        add_tri_toward(m, a, b2, c, tag, radial, true);
        add_tri_toward(m, a, c, d, tag, radial, true);
    };

    /* Front cap: fan apex → ring 0 */
    for (int j = 0; j < nseg; j++) {
        add_tri_toward(m, f_apex(), f_ring(0, j), f_ring(0, j + 1), tag, nfront);
    }

    /* Front cap: rings 0 → nrings-1 (front rim) */
    for (int i = 0; i < nrings - 1; i++) {
        for (int j = 0; j < nseg; j++) {
            add_tri_toward(m, f_ring(i, j), f_ring(i, j + 1), f_ring(i + 1, j + 1), tag, nfront);
            add_tri_toward(m, f_ring(i, j), f_ring(i + 1, j + 1), f_ring(i + 1, j), tag, nfront);
        }
    }

    /* Edge: front rim → back rim (flat) */
    for (int j = 0; j < nseg; j++) {
        rim_quad(f_ring(nrings - 1, j), f_ring(nrings - 1, j + 1), b_rim(j + 1), b_rim(j));
    }

    size_t fi2 = m.faces.size();  /* start of back cap */

    /* Back cap */
    if (nrings > 1) {
        /* back rim → back ring 0 */
        for (int j = 0; j < nseg; j++) {
            add_tri_toward(m, b_rim(j), b_rim(j + 1), b_ring(0, j + 1), tag, nback);
            add_tri_toward(m, b_rim(j), b_ring(0, j + 1), b_ring(0, j), tag, nback);
        }

        /* back rings 0 → nrings-2 */
        for (int i = 0; i < nrings - 2; i++) {
            for (int j = 0; j < nseg; j++) {
                add_tri_toward(m, b_ring(i, j), b_ring(i, j + 1), b_ring(i + 1, j + 1), tag, nback);
                add_tri_toward(m, b_ring(i, j), b_ring(i + 1, j + 1), b_ring(i + 1, j), tag, nback);
            }
        }

        /* fan: back ring nrings-2 → apex_b */
        for (int j = 0; j < nseg; j++) {
            add_tri_toward(m, b_ring(nrings - 2, j), b_ring(nrings - 2, j + 1), apex_b, tag, nback);
        }
    } else {
        /* nrings == 1: direct fan from back rim to apex */
        for (int j = 0; j < nseg; j++) {
            add_tri_toward(m, b_rim(j), b_rim(j + 1), apex_b, tag, nback);
        }
    }

    size_t fi3 = m.faces.size();

    /* Assign origin IDs to faces:
       front cap + edge → front_oid (edge is flat-flagged so curvature unused)
       back cap         → back_oid  */
    for (size_t fi = fi0; fi < fi2; fi++) {
        m.shape_id.push_back(front_oid);
    }

    for (size_t fi = fi2; fi < fi3; fi++) {
        m.shape_id.push_back(back_oid);
    }
}

/* ================================================================ */
/*  Coplanar overlapping faces                                      */
/* ================================================================ */

/* do two coplanar triangles overlap with a positive area? (2-D separating axis test) */
static bool coplanar_tris_overlap(const Vec3* A, const Vec3* B, int ax0, int ax1, float tol) {
    float a[3][2], b[3][2];

    for (int i = 0; i < 3; i++) {
        const float* pa = &A[i].x;
        const float* pb = &B[i].x;
        a[i][0] = pa[ax0];
        a[i][1] = pa[ax1];
        b[i][0] = pb[ax0];
        b[i][1] = pb[ax1];
    }

    for (int t = 0; t < 2; t++) {
        float (*P)[2] = t ? b : a;

        for (int e = 0; e < 3; e++) {
            float nx = -(P[(e + 1) % 3][1] - P[e][1]), ny = P[(e + 1) % 3][0] - P[e][0];
            float l = std::sqrt(nx * nx + ny * ny);

            if (l <= 0) {
                continue;
            }

            nx /= l;
            ny /= l;
            float amin = 1e30f, amax = -1e30f, bmin = 1e30f, bmax = -1e30f;

            for (int i = 0; i < 3; i++) {
                float pa = a[i][0] * nx + a[i][1] * ny, pb = b[i][0] * nx + b[i][1] * ny;
                amin = std::min(amin, pa);
                amax = std::max(amax, pa);
                bmin = std::min(bmin, pb);
                bmax = std::max(bmax, pb);
            }

            if (amax <= bmin + tol || bmax <= amin + tol) {
                return false;   /* separated, or only touching */
            }
        }
    }

    return true;
}

/*
 * Flag triangles that overlap a coplanar triangle (adjacent layers/slabs, a
 * box flush with the domain, ...). A ray tracer reports only one of two
 * coincident triangles, which would drop one of the two medium changes, so
 * the kernels re-derive the set of enclosing shapes after crossing a flagged
 * triangle. Returns the number of flagged triangles.
 */
static size_t flag_coplanar_faces(ShapeMesh& m) {
    float lo = 1e30f, hi = -1e30f;

    for (size_t i = 0; i < m.nodes.size(); i++) {
        lo = std::min(lo, std::min(m.nodes[i].x, std::min(m.nodes[i].y, m.nodes[i].z)));
        hi = std::max(hi, std::max(m.nodes[i].x, std::max(m.nodes[i].y, m.nodes[i].z)));
    }

    float tol = std::max(1e-5f, (hi - lo) * 1e-6f);

    /* key: canonical normal direction (sign chosen so the largest component is positive) */
    std::map<std::array<int, 3>, std::vector<std::pair<float, size_t> > > bynormal;

    for (size_t i = 0; i < m.faces.size(); i++) {
        float n[3] = {m.facedata[i].nx, m.facedata[i].ny, m.facedata[i].nz};
        int big = (std::fabs(n[0]) >= std::fabs(n[1]) && std::fabs(n[0]) >= std::fabs(n[2])) ? 0 :
                  (std::fabs(n[1]) >= std::fabs(n[2]) ? 1 : 2);
        float sg = (n[big] < 0) ? -1.f : 1.f;
        std::array<int, 3> key = {{(int)std::lround(n[0] * sg * 1e4f), (int)std::lround(n[1] * sg * 1e4f),
                                   (int)std::lround(n[2] * sg * 1e4f)
                                 }
                                };
        const Vec3& v = m.nodes[m.faces[i][0]];
        float d = (n[0] * v.x + n[1] * v.y + n[2] * v.z) * sg;
        bynormal[key].push_back(std::make_pair(d, i));
    }

    std::vector<bool> flag(m.faces.size(), false);

    for (std::map<std::array<int, 3>, std::vector<std::pair<float, size_t> > >::iterator it = bynormal.begin();
            it != bynormal.end(); ++it) {
        std::vector<std::pair<float, size_t> >& g = it->second;

        if (g.size() < 2) {
            continue;
        }

        std::sort(g.begin(), g.end());
        const std::array<int, 3>& k = it->first;
        int big = (std::abs(k[0]) >= std::abs(k[1]) && std::abs(k[0]) >= std::abs(k[2])) ? 0 : (std::abs(k[1]) >= std::abs(k[2]) ? 1 : 2);
        int ax0 = (big + 1) % 3, ax1 = (big + 2) % 3;

        /* sweep clusters of (nearly) equal plane offset */
        for (size_t s0 = 0; s0 < g.size();) {
            size_t s1 = s0 + 1;

            while (s1 < g.size() && g[s1].first - g[s1 - 1].first <= tol) {
                s1++;
            }

            for (size_t i = s0; i < s1; i++) {
                for (size_t j = i + 1; j < s1; j++) {
                    size_t fi = g[i].second, fj = g[j].second;

                    if (flag[fi] && flag[fj]) {
                        continue;
                    }

                    Vec3 A[3] = {m.nodes[m.faces[fi][0]], m.nodes[m.faces[fi][1]], m.nodes[m.faces[fi][2]]};
                    Vec3 B[3] = {m.nodes[m.faces[fj][0]], m.nodes[m.faces[fj][1]], m.nodes[m.faces[fj][2]]};

                    if (coplanar_tris_overlap(A, B, ax0, ax1, tol)) {
                        flag[fi] = flag[fj] = true;
                    }
                }
            }

            s0 = s1;
        }
    }

    size_t nflag = 0;

    for (size_t i = 0; i < m.faces.size(); i++) {
        if (flag[i]) {
            uint32_t pk;
            memcpy(&pk, &m.facedata[i].packed_media, 4);
            pk |= VKMMC_FACE_COPLANAR;
            memcpy(&m.facedata[i].packed_media, &pk, 4);
            nflag++;
        }
    }

    return nflag;
}

/* ================================================================ */
/*  Parse JSON Shapes array → combined mesh                         */
/* ================================================================ */

/* Per-shape analytic surface needed for curvature (see vkmmc_curvature.h).
   parse_shapes() emits one entry per shape (two for a Lens), indexed by the
   1-based ids stored in ShapeMesh::shape_id. */
struct ShapeOrigin {
    int type;             /* 0=box/slab, 1=sphere, 2=cylinder, 3=lens cap */
    float cx, cy, cz;     /* center (sphere/lens cap) or C0 (cylinder) */
    float ax, ay, az;     /* axis direction (cylinder only, unit vector) */
    float R;              /* radius (signed for lens caps: negative = concave) */
    float length;         /* cylinder axis length */
    uint32_t node_start, node_end;
};

static ShapeOrigin make_origin(int type) {
    ShapeOrigin so;
    memset(&so, 0, sizeof(so));
    so.type = type;
    return so;
}

/* CSG tag of a shape: 1..maxtag */
static uint32_t shape_tag(const json& obj, uint32_t maxtag, const std::string& ctx) {
    double t = json_num(obj, "Tag", 1, ctx);

    if (!(t >= 1 && t <= maxtag) || t != std::floor(t)) {
        vkmmc_fail(ctx + ".Tag=" + std::to_string(t) + " must be an integer in [1, " + std::to_string(maxtag) +
                   "] (a medium index; CSG supports up to " + std::to_string(VKMMC_MAX_CSG_TAG) + " tags)");
    }

    return (uint32_t)t;
}

static std::vector<double> shape_pos(const json& obj, const char* key, const double orig[3], const std::string& ctx) {
    std::vector<double> v = json_vec_key(obj, key, 3, ctx);

    for (int i = 0; i < 3; i++) {
        v[i] -= orig[i];
    }

    return v;
}

static std::vector<double> shape_size(const json& obj, const std::string& ctx) {
    std::vector<double> v = json_vec_key(obj, "Size", 3, ctx);

    for (int i = 0; i < 3; i++) {
        vkmmc_require(v[i] > 0, ctx + ".Size must be positive");
    }

    return v;
}

static double shape_radius(const json& obj, const std::string& ctx) {
    vkmmc_require(obj.contains("R"), ctx + " requires 'R'");
    double r = json_num(obj, "R", 0, ctx);
    vkmmc_require(r > 0, ctx + ".R must be positive");
    return r;
}

/*
 * Build a watertight triangle mesh from a JSON Shapes array (MCX shape syntax
 * plus Lens). Every shape is validated; invalid input throws. maxtag bounds the
 * Tag of each shape (number of media - 1). When origins is non-NULL it receives
 * the analytic surface of each shape, for curvature computation.
 */
static ShapeMesh parse_shapes(const json& arr, float ext[6], int mesh_res = 24,
                              uint32_t maxtag = VKMMC_MAX_CSG_TAG,
                              std::vector<ShapeOrigin>* origins = NULL) {
    ShapeMesh m;
    int origin_counter = 0;
    double orig[3] = {0, 0, 0};
    std::vector<ShapeOrigin> olist;

    vkmmc_require(arr.is_array(), "Shapes must be an array");
    vkmmc_require(mesh_res >= 1 && mesh_res <= VKMMC_MAX_MESHRES,
                  "mesh resolution must be in [" + std::to_string(VKMMC_MIN_MESHRES) + ", " +
                  std::to_string(VKMMC_MAX_MESHRES) + "]");

    /* curved shapes need a minimum tessellation; flat ones ignore mesh_res */
    auto need_res = [&](const std::string & ctx) {
        vkmmc_require(mesh_res >= VKMMC_MIN_MESHRES, ctx + ": the mesh resolution (MeshRes/-m) must be at least " +
                      std::to_string(VKMMC_MIN_MESHRES) + " for curved shapes");
    };
    maxtag = std::min(maxtag, (uint32_t)VKMMC_MAX_CSG_TAG);

    for (size_t si = 0; si < arr.size(); si++) {
        const json& shape = arr[si];
        std::string ctx = "Shapes[" + std::to_string(si) + "]";

        vkmmc_require(shape.is_object() && shape.size() == 1,
                      ctx + " must be an object with a single key, e.g. {\"Sphere\":{...}}");

        std::string key = shape.begin().key();
        const json& obj = shape.begin().value();
        ctx += "." + key;
        size_t faces_before = m.faces.size();
        ShapeOrigin so = make_origin(0);

        if (key == "Name") {
            /* label only */
        } else if (key == "Origin") {
            std::vector<double> o = json_vec(obj, 3, 3, ctx);
            orig[0] = o[0];
            orig[1] = o[1];
            orig[2] = o[2];
        } else {
            vkmmc_require(obj.is_object() || key.find("Layers") != std::string::npos, ctx + " must be an object");
        }

        if (key == "Grid") {
            std::vector<double> sz = shape_size(obj, ctx);
            uint32_t tag = shape_tag(obj, maxtag, ctx);
            ext[0] = 0;
            ext[1] = (float)sz[0];
            ext[2] = 0;
            ext[3] = (float)sz[1];
            ext[4] = 0;
            ext[5] = (float)sz[2];
            gen_box(m, 0, 0, 0, (float)sz[0], (float)sz[1], (float)sz[2], tag);
            printf("  Shape[%zu]: Grid [%.0f,%.0f,%.0f] tag=%u (%zu tri)\n", si, sz[0], sz[1], sz[2], tag,
                   m.faces.size() - faces_before);
        } else if (key == "Sphere") {
            std::vector<double> o = shape_pos(obj, "O", orig, ctx);
            double r = shape_radius(obj, ctx);
            uint32_t tag = shape_tag(obj, maxtag, ctx);
            need_res(ctx);
            int nlon = mesh_res, nlat = std::max(mesh_res * 2 / 3, 8);
            gen_sphere(m, (float)o[0], (float)o[1], (float)o[2], (float)r, tag, nlon, nlat);
            so = make_origin(1);
            so.cx = (float)o[0];
            so.cy = (float)o[1];
            so.cz = (float)o[2];
            so.R = (float)r;
            printf("  Shape[%zu]: Sphere O=[%.1f,%.1f,%.1f] R=%.1f tag=%u nlon=%d nlat=%d (%zu tri)\n",
                   si, o[0], o[1], o[2], r, tag, nlon, nlat, m.faces.size() - faces_before);
        } else if (key == "Box" || key == "Subgrid") {
            std::vector<double> o = shape_pos(obj, "O", orig, ctx);
            std::vector<double> sz = shape_size(obj, ctx);
            uint32_t tag = shape_tag(obj, maxtag, ctx);

            if (key == "Subgrid") {   /* MCX Subgrid: 1-based voxel origin */
                o[0] -= 1;
                o[1] -= 1;
                o[2] -= 1;
            }

            gen_box(m, (float)o[0], (float)o[1], (float)o[2], (float)sz[0], (float)sz[1], (float)sz[2], tag);
            printf("  Shape[%zu]: %s O=[%.1f,%.1f,%.1f] S=[%.1f,%.1f,%.1f] tag=%u (%zu tri)\n", si, key.c_str(),
                   o[0], o[1], o[2], sz[0], sz[1], sz[2], tag, m.faces.size() - faces_before);
        } else if (key == "Cylinder") {
            std::vector<double> c0 = shape_pos(obj, "C0", orig, ctx), c1 = shape_pos(obj, "C1", orig, ctx);
            double r = shape_radius(obj, ctx);
            uint32_t tag = shape_tag(obj, maxtag, ctx);
            double dx = c1[0] - c0[0], dy = c1[1] - c0[1], dz = c1[2] - c0[2];
            double l = std::sqrt(dx * dx + dy * dy + dz * dz);
            vkmmc_require(l > 1e-6, ctx + ": C0 and C1 must be distinct");
            need_res(ctx);
            gen_cylinder(m, (float)c0[0], (float)c0[1], (float)c0[2], (float)c1[0], (float)c1[1], (float)c1[2],
                         (float)r, tag, mesh_res);
            so = make_origin(2);
            so.cx = (float)c0[0];
            so.cy = (float)c0[1];
            so.cz = (float)c0[2];
            so.ax = (float)(dx / l);
            so.ay = (float)(dy / l);
            so.az = (float)(dz / l);
            so.length = (float)l;
            so.R = (float)r;
            printf("  Shape[%zu]: Cylinder R=%.1f tag=%u nseg=%d (%zu tri)\n",
                   si, r, tag, mesh_res, m.faces.size() - faces_before);
        } else if (key == "XSlabs" || key == "YSlabs" || key == "ZSlabs") {
            int dir = (key[0] == 'X') ? 0 : (key[0] == 'Y') ? 1 : 2;
            uint32_t tag = shape_tag(obj, maxtag, ctx);
            vkmmc_require(obj.contains("Bound") && obj["Bound"].is_array(), ctx + " requires a 'Bound' array");
            json bd = obj["Bound"];

            if (bd.size() == 2 && bd[0].is_number()) {
                bd = json::array({bd});    /* a single [lo, hi] pair */
            }

            for (size_t j = 0; j < bd.size(); j++) {
                std::vector<double> b = json_vec(bd[j], 2, 2, ctx + ".Bound[" + std::to_string(j) + "]");
                vkmmc_require(b[1] > b[0], ctx + ".Bound: need lower < upper bound");
                gen_slab(m, dir, (float)b[0], (float)b[1], ext, tag);
            }

            printf("  Shape[%zu]: %s (%zu tri)\n", si, key.c_str(), m.faces.size() - faces_before);
        } else if (key == "XLayers" || key == "YLayers" || key == "ZLayers") {
            int dir = (key[0] == 'X') ? 0 : (key[0] == 'Y') ? 1 : 2;
            vkmmc_require(obj.is_array(), ctx + " must be an array of [start, end, tag] layers");
            json layers = obj;

            if (layers.size() == 3 && layers[0].is_number()) {
                layers = json::array({layers});    /* a single layer */
            }

            for (size_t j = 0; j < layers.size(); j++) {
                std::string lctx = ctx + "[" + std::to_string(j) + "]";
                std::vector<double> l = json_vec(layers[j], 3, 3, lctx);
                vkmmc_require(l[1] >= l[0], lctx + ": need start <= end");
                json t;
                t["Tag"] = l[2];
                uint32_t lt = shape_tag(t, maxtag, lctx);
                gen_slab(m, dir, (float)(l[0] - 1.0), (float)l[1], ext, lt);
            }

            printf("  Shape[%zu]: %s (%zu tri)\n", si, key.c_str(), m.faces.size() - faces_before);
        } else if (key == "Lens") {
            std::vector<double> o = shape_pos(obj, "O", orig, ctx);
            std::vector<double> ld = json_vec_key(obj, "Dir", 3, ctx);
            double lR = shape_radius(obj, ctx);
            uint32_t tag = shape_tag(obj, maxtag, ctx);
            vkmmc_require(obj.contains("Front") && obj["Front"].is_object(), ctx + " requires a 'Front' object {D, R}");
            vkmmc_require(obj.contains("Back") && obj["Back"].is_object(), ctx + " requires a 'Back' object {D, R}");
            double Df = json_num(obj["Front"], "D", 0, ctx + ".Front"), Rf = json_num(obj["Front"], "R", 0, ctx + ".Front");
            double Db = json_num(obj["Back"], "D", 0, ctx + ".Back"), Rb = json_num(obj["Back"], "R", 0, ctx + ".Back");
            double dl = std::sqrt(ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2]);
            vkmmc_require(dl > 1e-10, ctx + ".Dir must be a non-zero vector");
            vkmmc_require(Df + Db > 0, ctx + ": the lens thickness Front.D + Back.D must be positive");
            vkmmc_require(Rf != 0 && Rb != 0, ctx + ": Front.R and Back.R must be non-zero");
            need_res(ctx);

            int nseg   = mesh_res;
            int nrings = std::max(mesh_res / 2, 4);
            uint32_t foid = (uint32_t)(origin_counter + 1);
            uint32_t boid = (uint32_t)(origin_counter + 2);
            gen_lens(m, (float)o[0], (float)o[1], (float)o[2], (float)ld[0], (float)ld[1], (float)ld[2], (float)lR,
                     (float)Df, (float)Rf, (float)Db, (float)Rb, tag, foid, boid, nseg, nrings);
            vkmmc_require(m.faces.size() > faces_before, ctx + ": the lens parameters give an empty surface");
            printf("  Shape[%zu]: Lens O=[%.1f,%.1f,%.1f] R=%.2f Df=%.2f Rf=%.2f Db=%.2f Rb=%.2f tag=%u (%zu tri)\n",
                   si, o[0], o[1], o[2], lR, Df, Rf, Db, Rb, tag, m.faces.size() - faces_before);

            /* front cap: center O + n*(Df-Rf); back cap: center O + n*(Rb-Db); radius kept signed */
            double nx = ld[0] / dl, ny = ld[1] / dl, nz = ld[2] / dl;
            ShapeOrigin sf2 = make_origin(3), sb2 = make_origin(3);
            sf2.cx = (float)(o[0] + nx * (Df - Rf));
            sf2.cy = (float)(o[1] + ny * (Df - Rf));
            sf2.cz = (float)(o[2] + nz * (Df - Rf));
            sf2.R = (float)Rf;
            sb2.cx = (float)(o[0] + nx * (Rb - Db));
            sb2.cy = (float)(o[1] + ny * (Rb - Db));
            sb2.cz = (float)(o[2] + nz * (Rb - Db));
            sb2.R = (float)Rb;
            olist.push_back(sf2);
            olist.push_back(sb2);
            origin_counter += shape_origin_count(key);
            continue;   /* gen_lens already pushed shape_ids */
        } else if (key != "Name" && key != "Origin") {
            vkmmc_fail(ctx + ": unknown shape type (Grid, Subgrid, Sphere, Box, Cylinder, XSlabs/YSlabs/ZSlabs, "
                       "XLayers/YLayers/ZLayers, Lens, Name, Origin)");
        }

        for (size_t fi = faces_before; fi < m.faces.size(); fi++) {
            m.shape_id.push_back((uint32_t)(origin_counter + 1));
        }

        olist.push_back(so);
        origin_counter += shape_origin_count(key);
    }

    printf("Shape mesh total: %zu nodes, %zu triangles\n", m.nodes.size(), m.faces.size());

    /* verify: print tag distribution */
    std::map<uint32_t, int> tagcount;
    int flatcount = 0;

    for (size_t i = 0; i < m.facedata.size(); i++) {
        uint32_t front, back;
        bool flat;
        unpack_media(m.facedata[i].packed_media, front, back, flat);
        flatcount += flat;
        tagcount[back]++;
    }

    printf("  Tag distribution:");

    for (std::map<uint32_t, int>::iterator it = tagcount.begin(); it != tagcount.end(); ++it) {
        printf(" [%u]=%d", it->first, it->second);
    }

    printf("\n  Flat-flagged triangles: %d / %zu\n", flatcount, m.facedata.size());

    size_t ncop = flag_coplanar_faces(m);

    if (ncop) {
        printf("  Coplanar-overlap triangles: %zu (enclosure re-derived on crossing)\n", ncop);
    }

    if (origins) {
        origins->swap(olist);
    }

    return m;
}

#endif
