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
    uint32_t pk = (tag & 0xFFFFu) | 0x80000000u;
    float f;
    std::memcpy(&f, &pk, 4);
    return f;
}

/*
 * add_tri_outward: add one triangle, ensuring its winding produces an outward
 * normal (i.e. pointing away from 'center'). If the computed cross-product
 * normal points toward center, the winding is flipped.
 * front=0, back=tag (CSG convention).
 */
static void add_tri_outward(ShapeMesh& m, uint32_t a, uint32_t b, uint32_t c,
                            uint32_t tag, Vec3 center, bool flat = false) {
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

    /* Triangle centroid */
    float cx = (va.x + vb.x + vc.x) / 3.0f;
    float cy = (va.y + vb.y + vc.y) / 3.0f;
    float cz = (va.z + vb.z + vc.z) / 3.0f;

    /* Vector from shape center to centroid */
    float dx = cx - center.x, dy = cy - center.y, dz = cz - center.z;

    /* If normal points inward (toward center), flip winding */
    if (n.x * dx + n.y * dy + n.z * dz < 0) {
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
    Vec3 center = {ox, oy, oz};

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

    /* Front cap: fan apex → ring 0 */
    for (int j = 0; j < nseg; j++) {
        add_tri_outward(m, f_apex(), f_ring(0, j), f_ring(0, j + 1), tag, center);
    }

    /* Front cap: rings 0 → nrings-1 (front rim) */
    for (int i = 0; i < nrings - 1; i++) {
        for (int j = 0; j < nseg; j++) {
            add_quad_outward(m, f_ring(i, j), f_ring(i, j + 1),
                             f_ring(i + 1, j + 1), f_ring(i + 1, j), tag, center);
        }
    }

    /* Edge: front rim → back rim (flat) */
    for (int j = 0; j < nseg; j++) {
        add_quad_outward(m, f_ring(nrings - 1, j), f_ring(nrings - 1, j + 1),
                         b_rim(j + 1), b_rim(j), tag, center, true);
    }

    size_t fi2 = m.faces.size();  /* start of back cap */

    /* Back cap */
    if (nrings > 1) {
        /* back rim → back ring 0 */
        for (int j = 0; j < nseg; j++) {
            add_quad_outward(m, b_rim(j), b_rim(j + 1),
                             b_ring(0, j + 1), b_ring(0, j), tag, center);
        }

        /* back rings 0 → nrings-2 */
        for (int i = 0; i < nrings - 2; i++) {
            for (int j = 0; j < nseg; j++) {
                add_quad_outward(m, b_ring(i, j), b_ring(i, j + 1),
                                 b_ring(i + 1, j + 1), b_ring(i + 1, j), tag, center);
            }
        }

        /* fan: back ring nrings-2 → apex_b */
        for (int j = 0; j < nseg; j++) {
            add_tri_outward(m, b_ring(nrings - 2, j), b_ring(nrings - 2, j + 1),
                            apex_b, tag, center);
        }
    } else {
        /* nrings == 1: direct fan from back rim to apex */
        for (int j = 0; j < nseg; j++) {
            add_tri_outward(m, b_rim(j), b_rim(j + 1), apex_b, tag, center);
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
/*  Parse JSON Shapes array → combined mesh                         */
/* ================================================================ */
static ShapeMesh parse_shapes(const json& arr, float ext[6], int mesh_res = 24) {
    ShapeMesh m;
    int origin_counter = 0;

    for (size_t si = 0; si < arr.size(); si++) {
        const json& shape = arr[si];

        if (!shape.is_object()) {
            continue;
        }

        std::string key;

        for (json::const_iterator it = shape.begin(); it != shape.end(); ++it) {
            key = it.key();
            break;
        }

        const json& obj = shape[key];
        size_t faces_before = m.faces.size();

        if (key == "Grid") {
            float sx = obj["Size"][0].get<float>(), sy = obj["Size"][1].get<float>(), sz = obj["Size"][2].get<float>();
            uint32_t tag = obj.value("Tag", 1u);
            ext[0] = 0;
            ext[1] = sx;
            ext[2] = 0;
            ext[3] = sy;
            ext[4] = 0;
            ext[5] = sz;
            gen_box(m, 0, 0, 0, sx, sy, sz, tag);
            printf("  Shape[%zu]: Grid [%.0f,%.0f,%.0f] tag=%u (%zu tri)\n", si, sx, sy, sz, tag, m.faces.size());
        } else if (key == "Sphere") {
            float cx = obj["O"][0].get<float>(), cy = obj["O"][1].get<float>(), cz = obj["O"][2].get<float>();
            float r = obj["R"].get<float>();
            uint32_t tag = obj.value("Tag", 1u);
            int nlon = mesh_res, nlat = std::max(mesh_res * 2 / 3, 8);
            gen_sphere(m, cx, cy, cz, r, tag, nlon, nlat);
            printf("  Shape[%zu]: Sphere O=[%.1f,%.1f,%.1f] R=%.1f tag=%u nlon=%d nlat=%d (%zu tri)\n",
                   si, cx, cy, cz, r, tag, nlon, nlat, m.faces.size());
        } else if (key == "Box") {
            float ox = obj["O"][0].get<float>(), oy = obj["O"][1].get<float>(), oz = obj["O"][2].get<float>();
            float sx = obj["Size"][0].get<float>(), sy = obj["Size"][1].get<float>(), sz = obj["Size"][2].get<float>();
            uint32_t tag = obj.value("Tag", 1u);
            gen_box(m, ox, oy, oz, sx, sy, sz, tag);
            printf("  Shape[%zu]: Box O=[%.1f,%.1f,%.1f] S=[%.1f,%.1f,%.1f] tag=%u (%zu tri)\n", si, ox, oy, oz, sx, sy, sz, tag, m.faces.size());
        } else if (key == "Subgrid") {
            float ox = obj["O"][0].get<float>() - 1, oy = obj["O"][1].get<float>() - 1, oz = obj["O"][2].get<float>() - 1;
            float sx = obj["Size"][0].get<float>(), sy = obj["Size"][1].get<float>(), sz = obj["Size"][2].get<float>();
            uint32_t tag = obj.value("Tag", 1u);
            gen_box(m, ox, oy, oz, sx, sy, sz, tag);
            printf("  Shape[%zu]: Subgrid tag=%u (%zu tri)\n", si, tag, m.faces.size());
        } else if (key == "Cylinder") {
            float x0 = obj["C0"][0].get<float>(), y0 = obj["C0"][1].get<float>(), z0 = obj["C0"][2].get<float>();
            float x1 = obj["C1"][0].get<float>(), y1 = obj["C1"][1].get<float>(), z1 = obj["C1"][2].get<float>();
            float r = obj["R"].get<float>();
            uint32_t tag = obj.value("Tag", 1u);
            gen_cylinder(m, x0, y0, z0, x1, y1, z1, r, tag, mesh_res);
            printf("  Shape[%zu]: Cylinder R=%.1f tag=%u nseg=%d (%zu tri)\n",
                   si, r, tag, mesh_res, m.faces.size());
        } else if (key == "XSlabs" || key == "YSlabs" || key == "ZSlabs") {
            int dir = (key[0] == 'X') ? 0 : (key[0] == 'Y') ? 1 : 2;

            if (obj.is_object()) {
                uint32_t tag = obj.value("Tag", 1u);

                if (obj.contains("Bound")) {
                    const json& bd = obj["Bound"];

                    for (size_t j = 0; j < bd.size(); j++) {
                        gen_slab(m, dir, bd[j][0].get<float>(), bd[j][1].get<float>(), ext, tag);
                    }
                }
            }

            printf("  Shape[%zu]: %s (%zu tri)\n", si, key.c_str(), m.faces.size());
        } else if (key == "XLayers" || key == "YLayers" || key == "ZLayers") {
            int dir = (key[0] == 'X') ? 0 : (key[0] == 'Y') ? 1 : 2;

            if (obj.is_array()) {
                for (size_t j = 0; j < obj.size(); j++) {
                    const json& layer = obj[j];

                    if (layer.is_array() && layer.size() >= 3) {
                        float lo = layer[0].get<float>() - 1.0f, hi = layer[1].get<float>();
                        uint32_t lt = layer[2].get<uint32_t>();
                        gen_slab(m, dir, lo, hi, ext, lt);
                    }
                }
            }

            printf("  Shape[%zu]: %s (%zu tri)\n", si, key.c_str(), m.faces.size());
        } else if (key == "Lens") {
            float lox = obj["O"][0].get<float>(), loy = obj["O"][1].get<float>(), loz = obj["O"][2].get<float>();
            float ldx = obj["Dir"][0].get<float>(), ldy = obj["Dir"][1].get<float>(), ldz = obj["Dir"][2].get<float>();
            float lR = obj["R"].get<float>();
            uint32_t tag = obj.value("Tag", 1u);
            float Df = obj["Front"]["D"].get<float>();
            float Rf = obj["Front"]["R"].get<float>();
            float Db = obj["Back"]["D"].get<float>();
            float Rb = obj["Back"]["R"].get<float>();
            int nseg   = mesh_res;
            int nrings = std::max(mesh_res / 2, 4);
            uint32_t foid = (uint32_t)(origin_counter + 1);
            uint32_t boid = (uint32_t)(origin_counter + 2);
            gen_lens(m, lox, loy, loz, ldx, ldy, ldz, lR, Df, Rf, Db, Rb, tag, foid, boid, nseg, nrings);
            printf("  Shape[%zu]: Lens O=[%.1f,%.1f,%.1f] R=%.2f Df=%.2f Rf=%.2f Db=%.2f Rb=%.2f tag=%u (%zu tri)\n",
                   si, lox, loy, loz, lR, Df, Rf, Db, Rb, tag, m.faces.size() - faces_before);
            origin_counter += shape_origin_count(key);
            continue;   /* gen_lens already pushed shape_ids */
        } else if (key != "Name" && key != "Origin") {
            printf("  Shape[%zu]: unknown '%s'\n", si, key.c_str());
        }

        for (size_t fi = faces_before; fi < m.faces.size(); fi++) {
            m.shape_id.push_back((uint32_t)(origin_counter + 1));
        }

        origin_counter += shape_origin_count(key);
    }

    printf("Shape mesh total: %zu nodes, %zu triangles\n", m.nodes.size(), m.faces.size());

    /* verify: print tag distribution */
    int tagcount[16] = {};
    int flatcount = 0;

    for (size_t i = 0; i < m.facedata.size(); i++) {
        uint32_t pk = 0;
        std::memcpy(&pk, &m.facedata[i].packed_media, 4);

        if (pk & 0x80000000u) {
            flatcount++;
        }

        uint32_t t = pk & 0xFFFFu;

        if (t < 16) {
            tagcount[t]++;
        }
    }

    printf("  Tag distribution:");

    for (int i = 0; i < 16; i++) if (tagcount[i]) {
            printf(" [%d]=%d", i, tagcount[i]);
        }

    printf("\n  Flat-flagged triangles: %d / %zu\n", flatcount, m.facedata.size());

    return m;
}

#endif