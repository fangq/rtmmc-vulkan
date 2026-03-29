#version 450
/* Fullscreen quad — positions from gl_VertexIndex; no vertex buffer needed. */

const vec2 QUAD[6] = vec2[6](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
    vec2( 1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

layout(location = 0) out vec2 vNdc;

void main() {
    vec2 p  = QUAD[gl_VertexIndex];
    gl_Position = vec4(p, 0.0, 1.0);
    vNdc = p;
}
