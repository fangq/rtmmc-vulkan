/*
 * fuzz_jdata.cpp — structure-aware fuzzing of the JData decoder
 *
 * The input bytes are split into a header that picks _ArrayType_,
 * _ArrayZipType_, _ArrayOrder_ and dimensions, and a payload that becomes the
 * compressed stream (base64 encoded), an _ArrayZipOffsets_ index and/or an
 * _ArrayData_ array. Also checks base64 and JData encode/decode round trips.
 */
#include "fuzz_common.h"

static const char* TYPES[] = {"single", "double", "int8", "uint8", "int16", "uint16", "int32", "uint32",
                              "int64", "uint64", "logical", "float32", "bogus"
                             };
static const char* ZTYPES[] = {"zlib", "gzip", "base64", "lzma"};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    fuzz_quiet();

    if (size < 9) {
        return 0;
    }

    /* base64 round trip of the raw bytes */
    std::string b64 = base64_encode(data, size);

    if (base64_decode(b64) != std::vector<uint8_t>(data, data + size)) {
        abort();
    }

    try {
        (void)base64_decode(std::string((const char*)data, size));   /* arbitrary text */
    } catch (const std::exception&) {
    }

    uint8_t t = data[0] % 13, z = data[1] % 4, mode = data[2];
    size_t nd = 1 + data[3] % 4;
    std::vector<size_t> dims;

    for (size_t i = 0; i < nd; i++) {
        dims.push_back(data[4 + i % 4] % 9);
    }

    const uint8_t* payload = data + 8;
    size_t plen = size - 8;
    json jd;
    jd["_ArrayType_"] = TYPES[t];
    jd["_ArraySize_"] = (mode & 1) ? json(dims) : json(dims[0]);

    if (mode & 2) {
        jd["_ArrayOrder_"] = (mode & 4) ? "c" : "r";
    }

    if (mode & 8) {
        /* uncompressed numeric _ArrayData_, possibly nested */
        json arr = json::array();

        for (size_t i = 0; i < plen; i++) {
            if ((mode & 16) && i % 3 == 0) {
                arr.push_back(json::array({payload[i], -payload[i]}));
            } else {
                arr.push_back((double)payload[i] - 100.0);
            }
        }

        jd["_ArrayData_"] = arr;
    } else {
        jd["_ArrayZipType_"] = ZTYPES[z];

        if (mode & 32) {
            /* a real compressed stream of the payload, then corrupt a byte */
            json enc = jdata_encode("uint8", {plen}, payload, plen);
            std::vector<uint8_t> comp = base64_decode(enc["_ArrayZipData_"].get<std::string>());

            if ((mode & 64) && !comp.empty()) {
                comp[payload[0] % comp.size()] ^= payload[plen - 1] | 1;
            }

            jd["_ArrayZipData_"] = base64_encode(comp.data(), comp.size());
        } else {
            jd["_ArrayZipData_"] = base64_encode(payload, plen);
        }

        if (mode & 128) {
            json off = json::array();

            for (size_t i = 0; i + 1 < plen && i < 16; i += 2) {
                off.push_back({payload[i] * 1000u, payload[i + 1] * 4096u});
            }

            jd["_ArrayZipOffsets_"] = off;
        }

        if (mode & 16) {
            jd["_ArrayZipSize_"] = dims;
        }
    }

    try {
        JDataArray a = jdata_decode_array(jd, "fuzz");

        if (a.raw.size() != a.numel() * a.esize) {
            abort();
        }

        for (size_t i = 0; i < a.numel(); i++) {
            (void)a.at(i);
        }

        /* re-encode as stored and decode again: must be lossless */
        json re = jdata_encode(a.type, a.dims, a.raw.data(), a.raw.size());
        JDataArray b = jdata_decode_array(re, "fuzz2");

        if (b.raw != a.raw || b.dims != a.dims) {
            abort();
        }

        /* tables as used for MeshNode/MeshElem */
        size_t rows = 0, cols = 0;
        (void)load_table(jd, 1, rows, cols, "table");
    } catch (const std::exception&) {
    }

    return 0;
}
