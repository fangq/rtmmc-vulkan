#version 450
/*
 * Volume ray-marcher for rtmmc-vulkan viewer.
 * Renders a 3D texture (log10-normalised fluence) using front-to-back
 * alpha compositing (default) or Maximum Intensity Projection (use_mip!=0).
 * Colormap: "hot" — black → red → orange/yellow → white.
 */

layout(location = 0) in  vec2 vNdc;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D volTex;

/* Push constant block — 128 bytes, all vec4-aligned.
 * cam_logmin.xyz  = camera world position,   .w = log10 low clamp (maps to 0)
 * origin_logmax.xyz = volume AABB origin,    .w = log10 high clamp (maps to 1)
 * extent_opacity.xyz = volume AABB size,     .w = opacity scale factor
 * use_mip_pad.x  = 0: compositing, non-zero: MIP */
layout(push_constant) uniform VolPC {
    mat4  inv_mvp;
    vec4  cam_logmin;
    vec4  origin_logmax;
    vec4  extent_opacity;
    ivec4 use_mip_pad;
} pc;

/* Hot colormap: 0→black, 1/3→red, 2/3→yellow, 1→white */
vec3 hot(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3(t * 3.0, t * 3.0 - 1.0, t * 3.0 - 2.0), vec3(0.0), vec3(1.0));
}

/* Slab-method ray-AABB intersection. Returns (tNear, tFar). tFar < tNear → miss. */
vec2 ray_aabb(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
    vec3 t0 = (bmin - ro) / rd;
    vec3 t1 = (bmax - ro) / rd;
    return vec2(max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z)),
                min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z)));
}

void main() {
    /* Reconstruct world-space ray through this pixel using the inverse MVP.
     * The Y-flip viewport maps NDC y=+1 to screen top, so vNdc.y is already
     * in the correct convention for the inverse transform. */
    vec4 wn = pc.inv_mvp * vec4(vNdc, -1.0, 1.0);
    vec4 wf = pc.inv_mvp * vec4(vNdc,  1.0, 1.0);
    wn /= wn.w;
    wf /= wf.w;
    vec3 ro = wn.xyz;
    vec3 rd = normalize(wf.xyz - wn.xyz);

    vec3 bmin = pc.origin_logmax.xyz;
    vec3 bmax = pc.origin_logmax.xyz + pc.extent_opacity.xyz;
    vec2 t    = ray_aabb(ro, rd, bmin, bmax);

    if (t.y <= t.x || t.y <= 0.0) { discard; }

    float tStart    = max(t.x, 0.001);
    float tEnd      = t.y;
    float diagLen   = length(pc.extent_opacity.xyz);
    float step      = diagLen / 256.0;
    float opScale   = pc.extent_opacity.w;

    vec4  accum  = vec4(0.0);
    float maxVal = 0.0;

    for (float tm = tStart; tm < tEnd && accum.a < 0.99; tm += step) {
        vec3  uvw = (ro + rd * tm - pc.origin_logmax.xyz) / pc.extent_opacity.xyz;
        float v   = texture(volTex, uvw).r;   /* normalized log10 in [0,1] */

        if (pc.use_mip_pad.x != 0) {
            maxVal = max(maxVal, v);
        } else {
            float alpha = v * opScale * step / diagLen;
            alpha = clamp(alpha, 0.0, 0.08);
            if (alpha > 5e-4) {
                vec3 col = hot(v);
                accum.rgb += (1.0 - accum.a) * alpha * col;
                accum.a   += (1.0 - accum.a) * alpha;
            }
        }
    }

    if (pc.use_mip_pad.x != 0) {
        if (maxVal < 0.01) { discard; }
        outColor = vec4(hot(maxVal), 0.9);
    } else {
        if (accum.a < 0.005) { discard; }
        outColor = accum;
    }
}
