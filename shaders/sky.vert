#version 450

// Fullscreen-triangle trick: three vertices at (-1,-1), (3,-1), (-1,3) cover
// the whole NDC square exactly once. z=1 puts it on the far plane so it never
// occludes terrain rendered into the same pass.

layout(location = 0) out vec2 vNdc;

void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vNdc = pos * 2.0 - 1.0;
    gl_Position = vec4(vNdc, 1.0, 1.0);
}
