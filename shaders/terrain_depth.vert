#version 450

// Depth-only pass for the sun's shadow map. Push constant carries the
// light-space view-projection — no UBO needed, no fragment shader needed
// (the pipeline is created with rasterizer-discard off and no color
// attachment, so the implicit depth output suffices).
//
// inNormal/inUV are part of the shared vertex layout but unused here.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(push_constant) uniform Push {
    mat4 lightViewProj;
} pc;

void main() {
    gl_Position = pc.lightViewProj * vec4(inPos, 1.0);
}
