#version 450

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    vec4 sunDirAndIrradiance;  // xyz = TO sun (ENU), w = direct beam W/m²
} cam;

layout(location = 0) in vec3 vNormal;
layout(location = 1) in float vHeight;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(vNormal);

    // Slope: 0 = flat ground, 1 = vertical wall.
    float slope = clamp(1.0 - N.z, 0.0, 1.0);

    // Height roughly normalised over a typical alpine range (500 m valley
    // floor → 4500 m summit).
    float h = clamp((vHeight - 500.0) / 4000.0, 0.0, 1.0);

    // Debug colour ramp: green valley → rocky slopes → snow caps.
    vec3 grass = vec3(0.32, 0.46, 0.20);
    vec3 rock  = vec3(0.45, 0.42, 0.38);
    vec3 snow  = vec3(0.94, 0.96, 0.98);

    vec3 albedo = mix(grass,  rock, smoothstep(0.30, 0.55, h));
    albedo      = mix(albedo, snow, smoothstep(0.55, 0.80, h));
    // Steep faces lean rock-ward regardless of altitude.
    albedo      = mix(albedo, rock, slope * 0.6);

    vec3  sunDir = cam.sunDirAndIrradiance.xyz;
    float ndotl  = max(0.0, dot(N, sunDir));

    // Sun below horizon → no direct contribution; smooth fade across the last
    // few degrees so dawn/dusk feel gradual instead of cliff-edged.
    float aboveHorizon = smoothstep(-0.05, 0.10, sunDir.z);
    float direct = ndotl * aboveHorizon;

    // Ambient also dims at night so dark side reads as actually dark, not
    // mid-grey. 0.20 → 0.05 floor at full night.
    float ambient = mix(0.05, 0.20, aboveHorizon);

    outColor = vec4(albedo * (ambient + 0.75 * direct), 1.0);
}
