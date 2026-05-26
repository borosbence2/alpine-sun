#version 450

// Atmosphere-rendered sky. Each fragment's view direction is reconstructed
// from invViewProj, converted to (azimuth, latitude) and sampled from the
// pre-baked sky-view LUT. The LUT was integrated for this frame's sun
// direction so the look updates with date/time changes.

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 lightViewProj;
    mat4 invViewProj;
    vec4 sunDirAndIrradiance;
    vec4 occlusionParams;
    vec4 sunHoursParams;
    vec4 toneParams;            // .x = exposure
    vec4 satParams;
    vec4 avalancheParams;
    vec4 terrainAabb;
} cam;

layout(set = 0, binding = 5) uniform sampler2D uSkyView;

layout(location = 0) in vec2 vNdc;
layout(location = 0) out vec4 outColor;

const float kPi    = 3.14159265358979;
const float kTwoPi = 6.28318530717958;

vec3 acesFilm(vec3 c) {
    return clamp((c * (2.51 * c + 0.03)) /
                 (c * (2.43 * c + 0.59) + 0.14),
                 0.0, 1.0);
}

void main() {
    // Reconstruct world-space view direction.
    vec4 nearW = cam.invViewProj * vec4(vNdc, 0.0, 1.0);
    vec4 farW  = cam.invViewProj * vec4(vNdc, 1.0, 1.0);
    vec3 viewDir = normalize(farW.xyz / farW.w - nearW.xyz / nearW.w);

    // ENU → (azimuth, latitude). Azimuth from +Y clockwise, matches the bake.
    float lat = asin(clamp(viewDir.z, -1.0, 1.0));
    float az  = atan(viewDir.x, viewDir.y);
    if (az < 0.0) az += kTwoPi;

    // Inverse of the horizon-biased v mapping in sky_view.comp:
    //   lat = sign(vc) * vc² * π/2,  vc ∈ [-1, 1],  v = (vc + 1) * 0.5
    float latNorm = clamp(lat / (kPi * 0.5), -1.0, 1.0);
    float vc = sign(latNorm) * sqrt(abs(latNorm));
    float u  = az / kTwoPi;
    float v  = vc * 0.5 + 0.5;

    vec3 sky = texture(uSkyView, vec2(u, v)).rgb;

    // Sun disk: punch a bright spot when the view direction lines up with the
    // sun. The sun subtends ~0.53° from Earth, so threshold cos(angular radius).
    vec3 sunDir = normalize(cam.sunDirAndIrradiance.xyz);
    float mu = dot(viewDir, sunDir);
    const float kSunCosRadius = 0.99995;      // ~0.57°
    float diskMask = smoothstep(kSunCosRadius, kSunCosRadius + 0.00002, mu);
    sky += vec3(20.0) * diskMask;

    outColor = vec4(acesFilm(sky * cam.toneParams.x), 1.0);
}
