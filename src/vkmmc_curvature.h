/*
 * vkmmc_curvature.h — Compute per-node principal curvature for CSG shapes
 *
 * For each node on a shape surface, we compute:
 *   - vnorm: vertex normal (outward)
 *   - k1, k2: principal curvatures
 *   - pdir1: first principal direction (pdir2 = cross(pdir1, vnorm))
 *
 * Sphere(R):  k1 = k2 = 1/R, pdir1 = any tangent
 * Cylinder(R, axis): k1 = 1/R (around axis), k2 = 0 (along axis)
 * Box/Slab: k1 = k2 = 0 (flat faces)
 */
#ifndef VKMMC_CURVATURE_H
#define VKMMC_CURVATURE_H

#include "vkmmc_io.h"
#include "vkmmc_shapes.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <map>
#include <set>

/* Per-node curvature data, packed for GPU upload */
struct NodeCurvature {
    float nx, ny, nz;    /* vertex normal */
    float k1, k2;        /* principal curvatures */
    float px, py, pz;    /* first principal direction */
};

/* Per-shape origin info needed for analytic curvature */
struct ShapeOrigin {
    int type;             /* 0=box/slab, 1=sphere, 2=cylinder */
    float cx, cy, cz;    /* center (sphere) or axis point (cylinder) */
    float ax, ay, az;    /* axis direction (cylinder only) */
    float R;              /* radius */
    uint32_t node_start, node_end; /* range in combined node list */
};

/* ================================================================ */
/*  Compute tangent basis for a normal vector                       */
/* ================================================================ */
static void make_tangent(float nx, float ny, float nz,
                         float& tx, float& ty, float& tz) {
    float ax, ay, az;

    if (std::fabs(nz) < 0.9f) {
        ax = 0;
        ay = 0;
        az = 1;
    } else {
        ax = 1;
        ay = 0;
        az = 0;
    }

    tx = ny * az - nz * ay;
    ty = nz * ax - nx * az;
    tz = nx * ay - ny * ax;
    float l = std::sqrt(tx * tx + ty * ty + tz * tz);

    if (l > 1e-10f) {
        tx /= l;
        ty /= l;
        tz /= l;
    }
}

/* ================================================================ */
/*  Compute analytic curvature for each node                        */
/* ================================================================ */
static std::vector<NodeCurvature> compute_curvature(
    const ShapeMesh& mesh,
    const std::vector<ShapeOrigin>& origins) {
    size_t nn = mesh.nodes.size();
    std::vector<NodeCurvature> curv(nn);
    memset(curv.data(), 0, nn * sizeof(NodeCurvature));

    /* Build node-to-shape mapping from shape_id on faces */
    std::vector<int> node_shape(nn, -1);

    for (size_t fi = 0; fi < mesh.faces.size(); fi++) {
        int si = (fi < mesh.shape_id.size()) ? (int)mesh.shape_id[fi] - 1 : -1;

        if (si < 0) {
            continue;
        }

        for (int k = 0; k < 3; k++) {
            uint32_t ni = mesh.faces[fi][k];

            if (ni < nn && node_shape[ni] < 0) {
                node_shape[ni] = si;
            }
        }
    }

    for (size_t i = 0; i < nn; i++) {
        int si = node_shape[i];

        if (si < 0 || si >= (int)origins.size()) {
            curv[i] = {0, 0, 1, 0, 0, 1, 0, 0};
            continue;
        }

        const ShapeOrigin& so = origins[si];
        float px = mesh.nodes[i].x, py = mesh.nodes[i].y, pz = mesh.nodes[i].z;

        if (so.type == 1) {
            /* ---- Sphere ---- */
            float dx = px - so.cx, dy = py - so.cy, dz = pz - so.cz;
            float l = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (l < 1e-10f) {
                l = 1e-10f;
            }

            float nx = dx / l, ny = dy / l, nz = dz / l;
            float k = 1.0f / so.R;
            float tx, ty, tz;
            make_tangent(nx, ny, nz, tx, ty, tz);
            curv[i].nx = nx;
            curv[i].ny = ny;
            curv[i].nz = nz;
            curv[i].k1 = k;
            curv[i].k2 = k;
            curv[i].px = tx;
            curv[i].py = ty;
            curv[i].pz = tz;

        } else if (so.type == 2) {
            /* ---- Cylinder ---- */
            float dx = px - so.cx, dy = py - so.cy, dz = pz - so.cz;
            float proj = dx * so.ax + dy * so.ay + dz * so.az;
            float rx = dx - proj * so.ax, ry = dy - proj * so.ay, rz = dz - proj * so.az;
            float rl = std::sqrt(rx * rx + ry * ry + rz * rz);

            if (rl < 1e-10f) {
                curv[i].nx = so.ax;
                curv[i].ny = so.ay;
                curv[i].nz = so.az;
                curv[i].k1 = 0;
                curv[i].k2 = 0;
                float tx, ty, tz;
                make_tangent(so.ax, so.ay, so.az, tx, ty, tz);
                curv[i].px = tx;
                curv[i].py = ty;
                curv[i].pz = tz;
            } else {
                float nx = rx / rl, ny = ry / rl, nz = rz / rl;
                float cx2 = so.ay * nz - so.az * ny, cy2 = so.az * nx - so.ax * nz, cz2 = so.ax * ny - so.ay * nx;
                float cl = std::sqrt(cx2 * cx2 + cy2 * cy2 + cz2 * cz2);

                if (cl > 1e-10f) {
                    cx2 /= cl;
                    cy2 /= cl;
                    cz2 /= cl;
                }

                curv[i].nx = nx;
                curv[i].ny = ny;
                curv[i].nz = nz;
                curv[i].k1 = 1.0f / so.R;
                curv[i].k2 = 0.0f;
                curv[i].px = cx2;
                curv[i].py = cy2;
                curv[i].pz = cz2;
            }

        } else {
            /* ---- Box / Slab (flat) ---- */
            curv[i].k1 = 0;
            curv[i].k2 = 0;
            curv[i].nx = 0;
            curv[i].ny = 0;
            curv[i].nz = 1;
            curv[i].px = 1;
            curv[i].py = 0;
            curv[i].pz = 0;
        }
    }

    /* For box/flat nodes, compute vertex normals by averaging incident face normals */
    std::vector<float> vnx(nn, 0), vny(nn, 0), vnz(nn, 0);
    std::vector<int> vcnt(nn, 0);

    for (size_t fi = 0; fi < mesh.faces.size(); fi++) {
        float fnx = mesh.facedata[fi].nx, fny = mesh.facedata[fi].ny, fnz = mesh.facedata[fi].nz;

        for (int k = 0; k < 3; k++) {
            uint32_t ni = mesh.faces[fi][k];

            if (ni < nn) {
                int si = node_shape[ni];

                if (si >= 0 && si < (int)origins.size() && origins[si].type == 0) {
                    vnx[ni] += fnx;
                    vny[ni] += fny;
                    vnz[ni] += fnz;
                    vcnt[ni]++;
                }
            }
        }
    }

    for (size_t i = 0; i < nn; i++) {
        if (vcnt[i] > 0) {
            float l = std::sqrt(vnx[i] * vnx[i] + vny[i] * vny[i] + vnz[i] * vnz[i]);

            if (l > 1e-10f) {
                curv[i].nx = vnx[i] / l;
                curv[i].ny = vny[i] / l;
                curv[i].nz = vnz[i] / l;
                make_tangent(curv[i].nx, curv[i].ny, curv[i].nz,
                             curv[i].px, curv[i].py, curv[i].pz);
            }
        }
    }

    return curv;
}

/* ================================================================ */
/*  Build ShapeOrigin list from JSON Shapes array                   */
/* ================================================================ */
static std::vector<ShapeOrigin> extract_shape_origins(const json& arr) {
    std::vector<ShapeOrigin> origins;

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
        ShapeOrigin so;
        memset(&so, 0, sizeof(so));

        if (key == "Sphere") {
            so.type = 1;
            so.cx = obj["O"][0].get<float>();
            so.cy = obj["O"][1].get<float>();
            so.cz = obj["O"][2].get<float>();
            so.R = obj["R"].get<float>();
        } else if (key == "Cylinder") {
            so.type = 2;
            float x0 = obj["C0"][0].get<float>(), y0 = obj["C0"][1].get<float>(), z0 = obj["C0"][2].get<float>();
            float x1 = obj["C1"][0].get<float>(), y1 = obj["C1"][1].get<float>(), z1 = obj["C1"][2].get<float>();
            so.cx = x0;
            so.cy = y0;
            so.cz = z0;
            float dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
            float l = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (l > 1e-10f) {
                so.ax = dx / l;
                so.ay = dy / l;
                so.az = dz / l;
            }

            so.R = obj["R"].get<float>();
        } else {
            so.type = 0;
        }

        origins.push_back(so);
    }

    return origins;
}

#endif /* VKMMC_CURVATURE_H */