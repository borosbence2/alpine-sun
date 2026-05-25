#version 450

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 lightViewProj;
    vec4 sunDirAndIrradiance;  // xyz = TO sun (ENU), w = direct beam W/m²
    vec4 occlusionParams;      // .x = shadow-map enabled, .y = horizon-map enabled
    vec4 terrainAabb;          // .xy = aabbMin, .zw = aabbMax (xy components only)
} cam;

layout(set = 0, binding = 1) uniform sampler2D      uShadowMap;
layout(set = 0, binding = 2) uniform sampler2DArray uHorizonMap;

layout(location = 0) in vec3 vNormal;
layout(location = 1) in float vHeight;
layout(location = 2) in vec3 vWorldPos;

layout(location = 0) out vec4 outColor;

const float TWO_PI = 6.28318530717958647692;
const float HALF_PI = 1.57079632679;
const int   HORIZON_BINS = 32;

// 3×3 PCF over the sun-POV shadow map.
float sampleShadow(vec4 lightClip) {
    vec3 ndc = lightClip.xyz / lightClip.w;
    if (ndc.z < 0.0 || ndc.z > 1.0) return 1.0;

    vec2 uv = ndc.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        return 1.0;
    }

    vec2 texelSize = 1.0 / vec2(textureSize(uShadowMap, 0));
    float refDepth = ndc.z;

    float lit = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            vec2 off = vec2(dx, dy) * texelSize;
            float occluder = texture(uShadowMap, uv + off).r;
            lit += refDepth <= occluder ? 1.0 : 0.0;
        }
    }
    return lit * (1.0 / 9.0);
}

// Horizon-map lookup. Compares the sun's elevation to the precomputed
// maximum-elevation profile of the terrain along the sun's azimuth. Returns
// visibility ∈ [0, 1].
float sampleHorizon(vec3 worldPos, vec3 sunDir) {
    vec2 aabbMin = cam.terrainAabb.xy;
    vec2 aabbMax = cam.terrainAabb.zw;
    vec2 extent  = aabbMax - aabbMin;

    // Map world XY → horizon UV. Mirrors the bake-time convention exactly
    // (v=0 at maxY, growing south).
    vec2 uv = vec2((worldPos.x - aabbMin.x) / extent.x,
                   (aabbMax.y - worldPos.y) / extent.y);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        return 1.0;
    }

    // Sun azimuth measured from +Y (north) clockwise — matches the bake.
    // atan(x, y) gives the angle of (x, y) measured from +Y clockwise.
    float az = atan(sunDir.x, sunDir.y);
    if (az < 0.0) az += TWO_PI;

    // Linear blend between the two adjacent bins, treated as continuous.
    float binFloat = az / TWO_PI * float(HORIZON_BINS);
    int   bin0     = int(floor(binFloat)) % HORIZON_BINS;
    int   bin1     = (bin0 + 1) % HORIZON_BINS;
    float frac     = fract(binFloat);

    float h0 = texture(uHorizonMap, vec3(uv, float(bin0))).r;
    float h1 = texture(uHorizonMap, vec3(uv, float(bin1))).r;
    float horizonAngle = mix(h0, h1, frac);

    float sunElev = asin(clamp(sunDir.z, -1.0, 1.0));
    // ±2° smoothstep band gives a soft horizon line. Tighter than this and
    // the transition looks like a binary mask along ridges; much wider and
    // shadows lose their shape.
    const float kBand = 0.0349;  // 2° in radians
    return smoothstep(-kBand, kBand, sunElev - horizonAngle);
}

void main() {
    vec3 N = normalize(vNormal);

    float slope = clamp(1.0 - N.z, 0.0, 1.0);
    float h     = clamp((vHeight - 500.0) / 4000.0, 0.0, 1.0);

    vec3 grass = vec3(0.32, 0.46, 0.20);
    vec3 rock  = vec3(0.45, 0.42, 0.38);
    vec3 snow  = vec3(0.94, 0.96, 0.98);

    vec3 albedo = mix(grass,  rock, smoothstep(0.30, 0.55, h));
    albedo      = mix(albedo, snow, smoothstep(0.55, 0.80, h));
    albedo      = mix(albedo, rock, slope * 0.6);

    vec3  sunDir = cam.sunDirAndIrradiance.xyz;
    float ndotl  = max(0.0, dot(N, sunDir));
    float aboveHorizon = smoothstep(-0.05, 0.10, sunDir.z);

    // Combine the two occlusion techniques: each contributes a visibility
    // factor in [0, 1] and we take the min (any source that says "shaded"
    // wins). When a source is disabled, its factor is 1.0 so it doesn't bias
    // the result.
    float vShadow  = cam.occlusionParams.x > 0.5
                   ? sampleShadow(cam.lightViewProj * vec4(vWorldPos, 1.0))
                   : 1.0;
    float vHorizon = cam.occlusionParams.y > 0.5
                   ? sampleHorizon(vWorldPos, sunDir)
                   : 1.0;
    float visibility = min(vShadow, vHorizon);

    float direct  = ndotl * aboveHorizon * visibility;
    float ambient = mix(0.05, 0.20, aboveHorizon);

    outColor = vec4(albedo * (ambient + 0.75 * direct), 1.0);
}
