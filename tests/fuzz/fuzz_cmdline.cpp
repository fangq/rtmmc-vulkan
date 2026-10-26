/*
 * fuzz_cmdline.cpp — fuzz the command line parser and the overrides
 *
 * The input is split at NUL/newline bytes into argv[1..]. Options that read
 * files or print help are allowed; their side effects are harmless here.
 */
#include "fuzz_common.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    fuzz_quiet();
    std::vector<std::string> args(1, "vkmmc");
    std::string cur;

    for (size_t i = 0; i < size && args.size() < 64; i++) {
        if (data[i] == 0 || data[i] == '\n') {
            args.push_back(cur);
            cur.clear();
        } else {
            cur += (char)data[i];
        }
    }

    if (!cur.empty()) {
        args.push_back(cur);
    }

    std::vector<char*> argv;

    for (size_t i = 0; i < args.size(); i++) {
        argv.push_back(&args[i][0]);
    }

    argv.push_back(NULL);

    try {
        CmdOverrides ovr;

        if (parse_cmdline((int)args.size(), argv.data(), ovr, "fuzz", 0, ".spv")) {
            SimConfig cfg;
            apply_overrides(cfg, ovr);

            if (ovr.totalthread > 0 && thread_count(cfg, ovr) % 64) {
                abort();
            }
        }
    } catch (const std::exception&) {
    }

    return 0;
}
