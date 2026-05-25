#version 450

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

    // Placeholder lighting: fixed sun direction (SPA-driven sun arrives in
    // Phase 2). Lambert + 25% ambient floor.
    vec3 sunDir = normalize(vec3(0.4, 0.3, 1.0));
    float ndotl = max(0.0, dot(N, sunDir));
    vec3 lit = albedo * (0.25 + 0.75 * ndotl);

    outColor = vec4(lit, 1.0);
}
