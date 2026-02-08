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
 *
 * IMPORTANT: Nodes shared between flat and curved faces (e.g. cylinder rim
 * nodes shared between cap and side) are assigned the CURVED surface's
 * curvature. The flat-flag on cap/box triangles causes the shader to skip
 * curvature correction for those triangles entirely, so having curved
 * curvature at shared nodes is correct — it only gets used when the
 * triangle is curved.
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
    float cx, cy, cz;    /* center (sphere) or C0 (cylinder) */
    float ax, ay, az;    /* axis direction (cylinder only, unit vector) */
    float R;              /* radius */
    float length;         /* cylinder axis length */
    uint32_t node_start, node_end;
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
/*  Helper: is this face flat-flagged? (bit 31 of packed_media)     */
/* ================================================================ */
static bool face_is_flat(const FaceData& fd) {
    uint32_t pk = 0;
    std::memcpy(&pk, &fd.packed_media, 4);
    return (pk & 0x80000000u) != 0;
}

/* ================================================================ */
/*  Compute curvature for a node on a specific shape's surface      */
/*  Returns true if the node is on a curved part, false if flat     */
/* ================================================================ */
static bool compute_node_curvature_for_shape(
    const ShapeOrigin& so, float px, float py, float pz,
    NodeCurvature& out) {
    memset(&out, 0, sizeof(out));

    if (so.type == 1) {
        /* ---- Sphere: always curved ---- */
        float dx = px - so.cx, dy = py - so.cy, dz = pz - so.cz;
        float l = std::sqrt(dx * dx + dy * dy + dz * dz);

        if (l < 1e-10f) {
            l = 1e-10f;
        }

        float nx = dx / l, ny = dy / l, nz = dz / l;
        float k = 1.0f / so.R;
        float tx, ty, tz;
        make_tangent(nx, ny, nz, tx, ty, tz);
        out.nx = nx;
        out.ny = ny;
        out.nz = nz;
        out.k1 = k;
        out.k2 = k;
        out.px = tx;
        out.py = ty;
        out.pz = tz;
        return true;

    } else if (so.type == 2) {
        /* ---- Cylinder ---- */
        float dx = px - so.cx, dy = py - so.cy, dz = pz - so.cz;
        float proj = dx * so.ax + dy * so.ay + dz * so.az;
        float rx = dx - proj * so.ax, ry = dy - proj * so.ay, rz = dz - proj * so.az;
        float rl = std::sqrt(rx * rx + ry * ry + rz * rz);

        /* Always compute the curved-side curvature for this node.
           Even rim nodes (at cap boundaries) get radial curvature here.
           The flat-flag on cap triangles ensures the shader won't use
           curvature data when the hit triangle is a cap face. */
        if (rl > 1e-10f) {
            float nx = rx / rl, ny = ry / rl, nz = rz / rl;
            /* circumferential direction = cross(axis, radial_normal) */
            float cx2 = so.ay * nz - so.az * ny;
            float cy2 = so.az * nx - so.ax * nz;
            float cz2 = so.ax * ny - so.ay * nx;
            float cl = std::sqrt(cx2 * cx2 + cy2 * cy2 + cz2 * cz2);

            if (cl > 1e-10f) {
                cx2 /= cl;
                cy2 /= cl;
                cz2 /= cl;
            }

            out.nx = nx;
            out.ny = ny;
            out.nz = nz;
            out.k1 = 1.0f / so.R;  /* curvature around circumference */
            out.k2 = 0.0f;         /* zero along axis */
            out.px = cx2;
            out.py = cy2;
            out.pz = cz2;
            return true;
        } else {
            /* Node is exactly on the axis (cap center) — truly flat */
            float sign_proj = (proj >= so.length * 0.5f) ? 1.0f : -1.0f;
            out.nx = so.ax * sign_proj;
            out.ny = so.ay * sign_proj;
            out.nz = so.az * sign_proj;
            out.k1 = 0;
            out.k2 = 0;
            float tx, ty, tz;
            make_tangent(out.nx, out.ny, out.nz, tx, ty, tz);
            out.px = tx;
            out.py = ty;
            out.pz = tz;
            return false;
        }

    } else {
        /* ---- Box / Slab (flat) ---- */
        out.k1 = 0;
        out.k2 = 0;
        out.nx = 0;
        out.ny = 0;
        out.nz = 1;
        out.px = 1;
        out.py = 0;
        out.pz = 0;
        return false;
    }
}

/* ================================================================ */
/*  Compute analytic curvature for each node                        */
/*                                                                  */
/*  Strategy for nodes shared between flat and curved faces:        */
/*  1. First pass: for each node, collect all shapes it belongs to  */
/*     via face connectivity, preferring curved faces.              */
/*  2. Assign curvature from the curved shape if available.         */
/*  3. This ensures rim nodes (shared between cylinder cap and      */
/*     side) get the side-wall curvature, which is what the shader  */
/*     needs when a curved-side triangle is hit.                    */
/* ================================================================ */
static std::vector<NodeCurvature> compute_curvature(
    const ShapeMesh& mesh,
    const std::vector<ShapeOrigin>& origins) {
    size_t nn = mesh.nodes.size();
    std::vector<NodeCurvature> curv(nn);
    memset(curv.data(), 0, nn * sizeof(NodeCurvature));

    /* For each node, track:
       - best_shape: shape index to use for curvature (-1 = none)
       - is_curved: whether the best assignment is from a curved face */
    std::vector<int> best_shape(nn, -1);
    std::vector<bool> node_is_curved(nn, false);

    /* Scan all faces. For each node, prefer assignment from a curved
       (non-flat-flagged) face. If a node only touches flat faces,
       it keeps the flat shape assignment. */
    for (size_t fi = 0; fi < mesh.faces.size(); fi++) {
        int si = (fi < mesh.shape_id.size()) ? (int)mesh.shape_id[fi] - 1 : -1;

        if (si < 0) {
            continue;
        }

        bool this_face_flat = face_is_flat(mesh.facedata[fi]);

        for (int k = 0; k < 3; k++) {
            uint32_t ni = mesh.faces[fi][k];

            if (ni >= nn) {
                continue;
            }

            if (best_shape[ni] < 0) {
                /* First assignment */
                best_shape[ni] = si;
                node_is_curved[ni] = !this_face_flat;
            } else if (!node_is_curved[ni] && !this_face_flat) {
                /* Upgrade from flat-only to curved */
                best_shape[ni] = si;
                node_is_curved[ni] = true;
            }

            /* If already curved, keep it — don't downgrade */
        }
    }

    /* Compute curvature for each node using its assigned shape */
    for (size_t i = 0; i < nn; i++) {
        int si = best_shape[i];

        if (si < 0 || si >= (int)origins.size()) {
            curv[i] = {0, 0, 1, 0, 0, 1, 0, 0};
            continue;
        }

        float px = mesh.nodes[i].x, py = mesh.nodes[i].y, pz = mesh.nodes[i].z;
        compute_node_curvature_for_shape(origins[si], px, py, pz, curv[i]);
    }

    /* For box/flat-only nodes, compute vertex normals by averaging
       incident face normals (for completeness, though curvature is 0) */
    std::vector<float> vnx(nn, 0), vny(nn, 0), vnz(nn, 0);
    std::vector<int> vcnt(nn, 0);

    for (size_t fi = 0; fi < mesh.faces.size(); fi++) {
        float fnx = mesh.facedata[fi].nx, fny = mesh.facedata[fi].ny, fnz = mesh.facedata[fi].nz;

        for (int k = 0; k < 3; k++) {
            uint32_t ni = mesh.faces[fi][k];

            if (ni < nn) {
                int si = best_shape[ni];

                if (si >= 0 && si < (int)origins.size() && origins[si].type == 0 && !node_is_curved[ni]) {
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

    /* Print statistics */
    int n_curved = 0, n_flat = 0;

    for (size_t i = 0; i < nn; i++) {
        if (node_is_curved[i]) {
            n_curved++;
        } else {
            n_flat++;
        }
    }

    printf("Curvature: %d curved nodes, %d flat-only nodes (total %zu)\n",
           n_curved, n_flat, nn);

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
            so.length = l;

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