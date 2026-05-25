#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 lightViewProj;
    vec4 sunDirAndIrradiance;
    vec4 occlusionParams;
    vec4 terrainAabb;
} cam;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out float vHeight;
layout(location = 2) out vec3 vWorldPos;

void main() {
    gl_Position = cam.viewProj * vec4(inPos, 1.0);
    vNormal     = inNormal;
    vHeight     = inPos.z;
    vWorldPos   = inPos;
}
