/*
 * vkmmc_shapes.h — Generate watertight triangle meshes from JSON shape constructs
 *
 * Each shape generates a closed surface. All triangles store front=0, back=Tag.
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

/* add one triangle with outward normal, front=0, back=tag */
static void add_tri(ShapeMesh& m, uint32_t a, uint32_t b, uint32_t c, uint32_t tag, bool flat = false) {
    m.faces.push_back({{a, b, c}});
    Vec3 va = m.nodes[a], vb = m.nodes[b], vc = m.nodes[c];
    Vec3 e1 = {vb.x - va.x, vb.y - va.y, vb.z - va.z}, e2 = {vc.x - va.x, vc.y - va.y, vc.z - va.z};
    Vec3 n = v3cross(e1, e2);
    float l = v3len(n);

    if (l > 0) {
        n.x /= l;
        n.y /= l;
        n.z /= l;
    }

    FaceData fd;
    fd.nx = n.x;
    fd.ny = n.y;
    fd.nz = n.z;
    fd.packed_media = flat ? pack_tag_flat(tag) : pack_tag(tag);
    m.facedata.push_back(fd);
}

/* add quad as 2 triangles */
static void add_quad(ShapeMesh& m, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t tag, bool flat = false) {
    add_tri(m, a, b, c, tag, flat);
    add_tri(m, a, c, d, tag, flat);
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

    add_quad(m, b + 0, b + 3, b + 2, b + 1, tag, true); /* -Z */
    add_quad(m, b + 4, b + 5, b + 6, b + 7, tag, true); /* +Z */
    add_quad(m, b + 0, b + 1, b + 5, b + 4, tag, true); /* -Y */
    add_quad(m, b + 2, b + 3, b + 7, b + 6, tag, true); /* +Y */
    add_quad(m, b + 0, b + 4, b + 7, b + 3, tag, true); /* -X */
    add_quad(m, b + 1, b + 2, b + 6, b + 5, tag, true); /* +X */
}

/* ================================================================ */
/*  Sphere: lat/lon subdivision, watertight — all curved            */
/*  nlon = longitude divisions, nlat = latitude divisions           */
/* ================================================================ */
static void gen_sphere(ShapeMesh& m, float cx, float cy, float cz,
                       float R, uint32_t tag, int nlon = 24, int nlat = 16) {
    uint32_t b = (uint32_t)m.nodes.size();
    const float PI = 3.14159265358979f;

    /* north pole */
    m.nodes.push_back({cx, cy, cz + R});

    /* latitude rings (1..nlat-1) */
    for (int i = 1; i < nlat; i++) {
        float phi = PI * (float)i / (float)nlat; /* 0=north, PI=south */
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
        add_tri(m, b, b + 1 + j, b + 1 + j1, tag);
    }

    /* body: quads between adjacent rings */
    for (int i = 0; i < nlat - 2; i++) {
        uint32_t r0 = b + 1 + i * nlon;
        uint32_t r1 = b + 1 + (i + 1) * nlon;

        for (int j = 0; j < nlon; j++) {
            uint32_t j1 = (j + 1) % nlon;
            add_quad(m, r0 + j, r1 + j, r1 + j1, r0 + j1, tag);
        }
    }

    /* south cap: fan from last ring to south pole */
    uint32_t last_ring = b + 1 + (nlat - 2) * nlon;

    for (int j = 0; j < nlon; j++) {
        uint32_t j1 = (j + 1) % nlon;
        add_tri(m, last_ring + j, south_pole, last_ring + j1, tag);
    }
}

/* ================================================================ */
/*  Cylinder: side quads (curved) + center-fan caps (flat)          */
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
        add_quad(m, b + i, b + n, b + nseg + n, b + nseg + i, tag, false);
    }

    /* bottom cap (c0): fan, normal toward -axis — flat */
    for (int i = 0; i < nseg; i++) {
        int n = (i + 1) % nseg;
        add_tri(m, cap0, b + n, b + i, tag, true);
    }

    /* top cap (c1): fan, normal toward +axis — flat */
    for (int i = 0; i < nseg; i++) {
        int n = (i + 1) % nseg;
        add_tri(m, cap1, b + nseg + i, b + nseg + n, tag, true);
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

/* ================================================================ */
/*  Parse JSON Shapes array → combined mesh                         */
/* ================================================================ */
static ShapeMesh parse_shapes(const json& arr, float ext[6], int mesh_res = 24) {
    ShapeMesh m;

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
        } else if (key != "Name" && key != "Origin") {
            printf("  Shape[%zu]: unknown '%s'\n", si, key.c_str());
        }

        for (size_t fi = faces_before; fi < m.faces.size(); fi++) {
            m.shape_id.push_back((uint32_t)(si + 1));
        }
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