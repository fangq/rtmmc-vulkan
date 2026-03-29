#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;   /* per-vertex normal; (0,0,0) = unlit */

layout(push_constant) uniform PC {
    mat4 mvp;
    vec4 color;
} pc;

layout(location = 0) out vec4 vColor;

void main() {
    gl_Position = pc.mvp * vec4(pos, 1.0);

    float lit = 1.0;
    float nlen = dot(normal, normal);
    if (nlen > 1e-6) {
        /* Two-sided diffuse with a fixed world-space light */
        vec3 n = normalize(normal);
        vec3 light = normalize(vec3(1.0, 2.0, 1.5));
        float diff = abs(dot(n, light));   /* abs() = two-sided */
        lit = 0.30 + 0.70 * diff;
    }
    vColor = vec4(pc.color.rgb * lit, pc.color.a);
}
