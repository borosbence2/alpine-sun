#version 450

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 lightViewProj;
    mat4 invViewProj;
    vec4 sunDirAndIrradiance;  // xyz = TO sun (ENU), w = direct beam W/m²
    vec4 occlusionParams;      // .x = shadow-map enabled, .y = horizon-map enabled
    vec4 sunHoursParams;       // .x = show colormap, .y = max-hours scale
    vec4 toneParams;           // .x = exposure (linear multiplier before ACES)
    vec4 satParams;            // .x = use satellite albedo, .y = sat image is real
    vec4 avalancheParams;      // .x = overlay enabled, .y = solar-loading enabled
    vec4 cameraPos;            // world ENU position of the camera (.w unused)
    vec4 terrainAabb;          // .xy = aabbMin, .zw = aabbMax (xy components only)
} cam;

layout(set = 0, binding = 1) uniform sampler2D      uShadowMap;
layout(set = 0, binding = 2) uniform sampler2DArray uHorizonMap;
layout(set = 0, binding = 3) uniform sampler2D      uSunHours;
layout(set = 0, binding = 4) uniform sampler2D      uSatellite;
layout(set = 0, binding = 5) uniform sampler2D      uSkyView;

layout(location = 0) in vec3 vNormal;
layout(location = 1) in float vHeight;
layout(location = 2) in vec3 vWorldPos;

layout(location = 0) out vec4 outColor;

const float TWO_PI       = 6.28318530717958647692;
const float HALF_PI      = 1.57079632679;
const int   HORIZON_BINS = 32;

// Same ACES approximation as sky.frag — kept duplicated rather than including
// across stages because shader headers are awkward in our build.
vec3 acesFilm(vec3 c) {
    return clamp((c * (2.51 * c + 0.03)) /
                 (c * (2.43 * c + 0.59) + 0.14),
                 0.0, 1.0);
}

// Sky-view LUT lookup matching the parameterisation used by sky_view.comp.
// Direction is expected in ENU (+Z = up). Used here to drive ambient
// lighting: shaded faces pick up the colour of the sky they "see" along
// their surface normal — blue daytime, warm at sunset, near-black at night.
vec3 sampleSky(vec3 dir) {
    float lat = asin(clamp(dir.z, -1.0, 1.0));
    float az  = atan(dir.x, dir.y);
    if (az < 0.0) az += TWO_PI;
    float latNorm = clamp(lat / HALF_PI, -1.0, 1.0);
    float vc = sign(latNorm) * sqrt(abs(latNorm));
    float u  = az / TWO_PI;
    float v  = vc * 0.5 + 0.5;
    return texture(uSkyView, vec2(u, v)).rgb;
}

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

// World XY → horizon/sun-hours UV. Both maps share the same convention so we
// extract this once.
vec2 worldToHorizonUv(vec3 worldPos) {
    vec2 aabbMin = cam.terrainAabb.xy;
    vec2 aabbMax = cam.terrainAabb.zw;
    return vec2((worldPos.x - aabbMin.x) / (aabbMax.x - aabbMin.x),
                (aabbMax.y - worldPos.y) / (aabbMax.y - aabbMin.y));
}

float sampleHorizon(vec3 worldPos, vec3 sunDir) {
    vec2 uv = worldToHorizonUv(worldPos);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        return 1.0;
    }

    float az = atan(sunDir.x, sunDir.y);
    if (az < 0.0) az += TWO_PI;

    float binFloat = az / TWO_PI * float(HORIZON_BINS);
    int   bin0     = int(floor(binFloat)) % HORIZON_BINS;
    int   bin1     = (bin0 + 1) % HORIZON_BINS;
    float frac     = fract(binFloat);

    float h0 = texture(uHorizonMap, vec3(uv, float(bin0))).r;
    float h1 = texture(uHorizonMap, vec3(uv, float(bin1))).r;
    float horizonAngle = mix(h0, h1, frac);

    float sunElev = asin(clamp(sunDir.z, -1.0, 1.0));
    const float kBand = 0.0349;  // 2° smoothstep band
    return smoothstep(-kBand, kBand, sunElev - horizonAngle);
}

// Polynomial viridis approximation by Matt Zucker — visually indistinguishable
// from matplotlib's lookup at this scale, no LUT texture required.
vec3 viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.2777273272234177,  0.005407344544966578, 0.3340998053353061);
    const vec3 c1 = vec3(0.1050930431085774,  1.404613529898575,    1.384590162594685);
    const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213,    0.09509516302823659);
    const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585,   -19.33244095627987);
    const vec3 c4 = vec3(6.228269936347081,  14.17993336680509,    56.69055260068105);
    const vec3 c5 = vec3(4.776384997670288, -13.74514537774601,   -65.35303263337234);
    const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535,    26.3124352495832);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

void main() {
    vec3 N = normalize(vNormal);

    // --- Sun-hours visualisation mode ---
    // When enabled, the whole terrain reads as a viridis-mapped sun-hours/day.
    // We still apply a touch of Lambert so the shape stays legible, but the
    // dominant signal is the colormap. Skip the tonemap here so the colormap
    // values reach the screen unchanged — the legend matches without the
    // s-curve squashing the colours.
    if (cam.sunHoursParams.x > 0.5) {
        vec2 uv = worldToHorizonUv(vWorldPos);
        float hours = texture(uSunHours, uv).r;
        float maxH  = max(cam.sunHoursParams.y, 0.001);
        vec3  col   = viridis(hours / maxH);
        // Light shape cue: 0.85..1.0 multiplier from upward-facing-ness.
        float shape = mix(0.85, 1.0, max(0.0, N.z));
        outColor = vec4(col * shape, 1.0);
        return;
    }

    // --- Normal terrain shading ---
    float slope = clamp(1.0 - N.z, 0.0, 1.0);
    float h     = clamp((vHeight - 500.0) / 4000.0, 0.0, 1.0);

    vec3 grass = vec3(0.32, 0.46, 0.20);
    vec3 rock  = vec3(0.45, 0.42, 0.38);
    vec3 snow  = vec3(0.94, 0.96, 0.98);

    vec3 albedo = mix(grass,  rock, smoothstep(0.30, 0.55, h));
    albedo      = mix(albedo, snow, smoothstep(0.55, 0.80, h));
    albedo      = mix(albedo, rock, slope * 0.6);

    // Satellite override — only when the user enabled it AND the preset
    // actually has a bundled image (otherwise the placeholder 1x1 black would
    // wipe the terrain).
    if (cam.satParams.x > 0.5 && cam.satParams.y > 0.5) {
        vec2 satUv = worldToHorizonUv(vWorldPos);   // same world→UV mapping as horizon/sun-hours
        if (all(greaterThanEqual(satUv, vec2(0.0))) && all(lessThanEqual(satUv, vec2(1.0)))) {
            albedo = texture(uSatellite, satUv).rgb;
        }
    }

    vec3  sunDir = cam.sunDirAndIrradiance.xyz;
    float ndotl  = max(0.0, dot(N, sunDir));
    float aboveHorizon = smoothstep(-0.05, 0.10, sunDir.z);

    float vShadow  = cam.occlusionParams.x > 0.5
                   ? sampleShadow(cam.lightViewProj * vec4(vWorldPos, 1.0))
                   : 1.0;
    float vHorizon = cam.occlusionParams.y > 0.5
                   ? sampleHorizon(vWorldPos, sunDir)
                   : 1.0;
    float visibility = min(vShadow, vHorizon);

    float direct = ndotl * aboveHorizon * visibility;

    // Sky-tinted ambient: sample the Hillaire sky-view LUT in the direction
    // of the surface normal (clamped to the upper hemisphere so overhangs
    // still pick up some sky). The 0.06 scale brings the LUT's HDR units
    // into roughly the same magnitude as the old [0.05..0.20] flat ambient;
    // a small constant floor keeps shaded areas from going jet-black on
    // moonless nights.
    vec3 ambientDir = normalize(vec3(N.x, N.y, max(N.z, 0.1)));
    vec3 ambient = sampleSky(ambientDir) * 0.06 + vec3(0.012);

    vec3 linearColor = albedo * (ambient + vec3(0.75 * direct));

    // --- Aerial perspective ---
    // Cheap atmospheric haze: blend toward the sky colour in the view direction
    // as the distance from camera grows. Extinction is exponential with both
    // distance AND altitude (so high-altitude pixels haze less than valley
    // pixels at the same range). Real Hillaire AP uses a 3D LUT; this inline
    // approximation captures the dominant visual at a fraction of the cost.
    {
        vec3  toFrag = vWorldPos - cam.cameraPos.xyz;
        float dist   = length(toFrag);
        vec3  viewDir = toFrag / max(dist, 1e-3);
        // Mean altitude of the camera-fragment segment, in km.
        float meanH = max(0.0, (cam.cameraPos.z + vWorldPos.z) * 0.5) * 0.001;
        // Density falls off with altitude (mostly Mie-like haze; ~1.5 km scale).
        float density = exp(-meanH / 1.5);
        // 1 - transmittance over the path. Tuning constant chosen so a 20 km
        // path through valley air noticeably hazes.
        const float kHazeStrength = 0.00012;
        float fog = 1.0 - exp(-dist * density * kHazeStrength);
        vec3 hazeColor = sampleSky(viewDir);
        linearColor = mix(linearColor, hazeColor, clamp(fog, 0.0, 0.85));
    }

    vec3 lit = acesFilm(linearColor * cam.toneParams.x);

    // --- Avalanche terrain overlay ---
    // Heuristic only: blends an industry-standard slope-angle ramp over the
    // tonemapped image. The ramp peaks at 35–40°, fading to nothing below
    // 27° (snow rarely slides) and above 50° (snow doesn't accumulate).
    // When solar-loading is on, south-facing high-sun-hours slopes get a
    // saturation bump (wet-avalanche bias). The colour stays a UI decal —
    // applied after tonemap so exposure scrubbing doesn't shift the hazard
    // hue and ACES can't desaturate the warning reds.
    if (cam.avalancheParams.x > 0.5) {
        float slopeDeg = degrees(acos(clamp(N.z, -1.0, 1.0)));
        vec3  hazCol   = vec3(0.0);
        float hazW     = 0.0;
        if (slopeDeg >= 27.0 && slopeDeg <= 55.0) {
            // Discrete colour brackets matching FATMAP/CalTopo conventions.
            if      (slopeDeg < 30.0) hazCol = vec3(0.95, 0.90, 0.20);  // yellow
            else if (slopeDeg < 35.0) hazCol = vec3(1.00, 0.55, 0.10);  // orange
            else if (slopeDeg < 40.0) hazCol = vec3(0.95, 0.18, 0.18);  // red
            else if (slopeDeg < 45.0) hazCol = vec3(0.65, 0.08, 0.32);  // deep red
            else                       hazCol = vec3(0.40, 0.05, 0.30);  // dark purple
            // Triangular weight peaking at 38°; falls to 0 at the 27° and 55°
            // bracket edges. Smoothstep so the boundary isn't a hard line.
            float w = 1.0 - abs(slopeDeg - 38.0) / 17.0;
            hazW = clamp(w, 0.0, 1.0);
        }

        if (cam.avalancheParams.y > 0.5 && hazW > 0.0) {
            // Solar loading: south-facing slopes with lots of direct sun are
            // wet-slide-prone in spring afternoons. South in ENU = -Y.
            vec2 horiz = vec2(N.x, N.y);
            float horizLen = max(length(horiz), 1e-3);
            float southness = max(0.0, -N.y / horizLen);
            vec2 huv = worldToHorizonUv(vWorldPos);
            float sunH = texture(uSunHours, huv).r;
            float sunFactor = clamp(sunH / 8.0, 0.0, 1.0);
            float wetBoost = sunFactor * southness;
            // Push toward saturated red and raise weight so the overlay reads
            // as "warmer" hazard on the sun-loaded faces.
            hazCol = mix(hazCol, vec3(1.0, 0.08, 0.05), wetBoost * 0.55);
            hazW   = clamp(hazW + wetBoost * 0.25, 0.0, 1.0);
        }

        // Moderate opacity so terrain structure remains visible underneath.
        lit = mix(lit, hazCol, hazW * 0.65);
    }

    outColor = vec4(lit, 1.0);
}
