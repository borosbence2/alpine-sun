#version 450

// Procedural sky. We back-project the fragment's NDC position to a world-space
// view direction via the inverse view-projection, then sample a 2-stop
// gradient (horizon → zenith) modulated by sun elevation for day/night and a
// horizon glow when the sun is low.
//
// Lives inside the same render pass as terrain; renders FIRST so terrain
// overdraws where it covers. Depth test is off in the pipeline.

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 lightViewProj;
    mat4 invViewProj;
    vec4 sunDirAndIrradiance;   // xyz = TO sun (ENU)
    vec4 occlusionParams;
    vec4 sunHoursParams;
    vec4 toneParams;            // .x = exposure (linear multiplier before ACES)
    vec4 terrainAabb;
} cam;

layout(location = 0) in vec2 vNdc;
layout(location = 0) out vec4 outColor;

// Krzysztof Narkowicz's ACES approximation — public domain. Maps HDR linear
// into a film-like [0,1] LDR. Output is linear; the swapchain's sRGB format
// handles gamma encoding on write.
vec3 acesFilm(vec3 c) {
    return clamp((c * (2.51 * c + 0.03)) /
                 (c * (2.43 * c + 0.59) + 0.14),
                 0.0, 1.0);
}

void main() {
    // Back-project: reconstruct a world-space view direction from the NDC by
    // taking the difference between a near-plane and far-plane unprojection.
    // Works regardless of perspective/ortho.
    vec4 nearWorld = cam.invViewProj * vec4(vNdc, 0.0, 1.0);
    vec4 farWorld  = cam.invViewProj * vec4(vNdc, 1.0, 1.0);
    vec3 viewDir   = normalize(farWorld.xyz / farWorld.w
                             - nearWorld.xyz / nearWorld.w);

    vec3 sunDir = cam.sunDirAndIrradiance.xyz;

    // Daytime gradient. zenith = saturated blue, horizon = pale near-white.
    // We bias upwards a touch so the horizon line itself sits a hair below the
    // 50/50 mix point (matches how the eye expects to see things).
    vec3 zenithCol  = vec3(0.28, 0.50, 0.85);
    vec3 horizonCol = vec3(0.78, 0.82, 0.85);
    float upT = clamp(viewDir.z * 1.4 + 0.05, 0.0, 1.0);
    vec3 daySky = mix(horizonCol, zenithCol, smoothstep(0.0, 1.0, upT));

    // Sunset/sunrise glow: warm tint along the horizon, strongest when the
    // sun is near the horizon and the look direction roughly aligns with the
    // sun's azimuth. Falls off rapidly with altitude.
    float sunLow      = smoothstep(0.30, -0.05, sunDir.z);   // 1 when sun near horizon
    vec2  flatViewDir = normalize(vec2(viewDir.x, viewDir.y) + 1e-5);
    vec2  flatSunDir  = normalize(vec2(sunDir.x, sunDir.y)  + 1e-5);
    float alignment   = max(0.0, dot(flatViewDir, flatSunDir));
    float horizonness = clamp(1.0 - viewDir.z * 2.5, 0.0, 1.0);
    vec3  warmGlow    = vec3(1.10, 0.55, 0.25) * sunLow * pow(alignment, 4.0) * horizonness;
    daySky += warmGlow;

    // Night sky: deep cool blue, near-uniform. Fade in as sun drops.
    vec3  nightSky  = vec3(0.015, 0.020, 0.040);
    float dayFactor = smoothstep(-0.10, 0.15, sunDir.z);
    vec3  sky       = mix(nightSky, daySky, dayFactor);

    outColor = vec4(acesFilm(sky * cam.toneParams.x), 1.0);
}
