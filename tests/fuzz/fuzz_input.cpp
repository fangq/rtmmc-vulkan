/*
 * fuzz_input.cpp — fuzz the whole host input pipeline with JSON input files:
 * parse, load_json_config (validation, mesh loading, JData decoding),
 * prepare_geometry (shape meshes, coplanar flagging, curvature), compute_grid
 * and the kernel parameter derivation, i.e. everything vkmmc does before it
 * touches the GPU.
 */
#include "fuzz_common.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    fuzz_quiet();
    json j = json::parse(data, data + size, nullptr, false);

    if (j.is_discarded()) {
        return 0;
    }

    try {
        SimConfig cfg = load_json_config(j);
        CmdOverrides ovr;
        /* keep curved shapes cheap: the resolution is fuzzed via MeshRes anyway */
        cfg.mesh_res = std::min(cfg.mesh_res, 96);
        SimSetup st = prepare_geometry(cfg, ovr);
        OutputGrid g = compute_grid(cfg);
        std::vector<Medium> media = kernel_media(cfg);
        float t0, t1, idt;
        kernel_time_gates(cfg, t0, t1, idt);
        uint32_t tt = thread_count(cfg, ovr);
        (void)batch_size(cfg, ovr);
        (void)st;
        (void)media;
        (void)tt;

        /* invariants the GPU kernels rely on */
        for (size_t i = 0; i < cfg.faces.size(); i++) {
            for (int k = 0; k < 3; k++) {
                if (cfg.faces[i][k] >= cfg.nodes.size()) {
                    abort();
                }
            }

            uint32_t front, back;
            bool flat;
            unpack_media(cfg.facedata[i].packed_media, front, back, flat);

            if (front >= cfg.media.size() || back >= cfg.media.size()) {
                abort();
            }
        }

        if (st.has_curvature && st.curv.size() != cfg.nodes.size()) {
            abort();
        }

        if (g.nelem == 0 || g.nelem > VKMMC_MAX_VOXELS || tt % 64 || !(idt > 0) || !std::isfinite(idt)) {
            abort();
        }

        /* the source data the kernels index */
        if (g.nfield < 1 || g.nelem * g.nfield > VKMMC_MAX_VOXELS || g.nout != g.nelem * g.nfield + (uint64_t)cfg.srcnum
                || cfg.srcnum < 1 || cfg.extrasrc.size() > VKMMC_MAX_SOURCES
                || (cfg.srcid > 0 && (size_t)cfg.srcid > cfg.extrasrc.size())) {
            abort();
        }

        if (cfg.srctype == SRC_PATTERN || cfg.srctype == SRC_PATTERN3D) {
            double npix = (cfg.srctype == SRC_PATTERN) ? (double)cfg.srcparam1[3] * cfg.srcparam2[3]
                          : (double)cfg.srcparam1[0] * cfg.srcparam1[1] * cfg.srcparam1[2];

            if ((double)cfg.srcpattern.size() != npix * cfg.srcnum) {
                abort();
            }
        }

        for (size_t i = 0; i < cfg.srcpattern.size(); i++) {
            if (!(cfg.srcpattern[i] >= 0 && cfg.srcpattern[i] <= VKMMC_MAX_SRC_WEIGHT)) {
                abort();
            }
        }

        std::vector<float> es = kernel_extrasrc(cfg);

        for (size_t i = 0; i < es.size(); i++) {
            if (!std::isfinite(es[i])) {
                abort();
            }
        }

        (void)kernel_initial_medium(cfg);
        (void)source_json(cfg, false);
    } catch (const std::exception&) {
        /* rejected input */
    }

    return 0;
}
