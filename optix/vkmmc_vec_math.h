/*
 * vkmmc_vec_math.h — the few CUDA vector helpers the OptiX kernel uses
 *
 * Replaces the OptiX SDK's sutil/vec_math.h so that only the OptiX headers
 * (include/optix*.h, e.g. a clone of https://github.com/NVIDIA/optix-dev) and
 * the CUDA toolkit are needed to build.
 */
#ifndef VKMMC_VEC_MATH_H
#define VKMMC_VEC_MATH_H

#include <vector_functions.h>
#include <vector_types.h>

#define VKMMC_HD __host__ __device__ __forceinline__

VKMMC_HD float3 make_float3(const float4& v) {
    return make_float3(v.x, v.y, v.z);
}

VKMMC_HD float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

VKMMC_HD float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

VKMMC_HD float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

VKMMC_HD float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

VKMMC_HD float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

VKMMC_HD float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

VKMMC_HD float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

VKMMC_HD float3 normalize(const float3& v) {
    return v * rsqrtf(dot(v, v));
}

#undef VKMMC_HD

#endif
