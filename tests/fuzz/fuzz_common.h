/*
 * fuzz_common.h — shared setup for the vkmmc libFuzzer harnesses
 *
 * The harnesses exercise the host code that runs before any GPU work. A
 * std::exception is the expected outcome for invalid input; anything else
 * (sanitizer report, other exception, timeout, excessive memory) is a bug.
 */
#ifndef VKMMC_FUZZ_COMMON_H
#define VKMMC_FUZZ_COMMON_H

#include "vkmmc_host.h"
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>

/* the headers print progress to stdout; silence it once per process */
static void fuzz_quiet() {
    static bool done = false;

    if (!done && getenv("VKMMC_FUZZ_VERBOSE") == NULL) {
        int fd = open("/dev/null", O_WRONLY);

        if (fd >= 0) {
            dup2(fd, 1);
            close(fd);
        }
    }

    done = true;
}

#endif
