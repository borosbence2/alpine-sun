#version 450

// Line-strip vertex shader for a GPX route. Vertices are already in ENU
// world space (built CPU-side from lat/lon and a DEM height sample, lifted
// a few metres so the line floats above terrain rather than z-fighting it).

layout(location = 0) in vec3 inPos;

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 lightViewProj;
    mat4 invViewProj;
    vec4 sunDirAndIrradiance;
    vec4 occlusionParams;
    vec4 sunHoursParams;
    vec4 toneParams;
    vec4 terrainAabb;
} cam;

void main() {
    gl_Position = cam.viewProj * vec4(inPos, 1.0);
}
