// alpine-sun — entry point.
//
// Phase 2: PSA-driven sun direction wired into the terrain shader, with an
// ImGui panel for location, date, and time. Camera is still the hand-rolled
// orbit from Phase 1.

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "dem_loader.h"
#include "geo_frame.h"
#include "terrain_mesh.h"
#include "sun_driver.h"
#include "gpx.h"
#include "device.h"        // forfun::createDevice / destroyDevice
#include "swapchain.h"     // forfun::createSwapchain / destroySwapchain
#include "frame_context.h" // forfun::createFrameContext / destroyFrameContext
#include "vk_helpers.h"    // transitionImage, createBufferGPU/HostMapped, etc.
#include "types.h"         // VK_CHECK, kFramesInFlight, Vertex, kDepthFormat

#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "terrain.vert.h"
#include "terrain.frag.h"
#include "terrain_depth.vert.h"
#include "horizon_map.comp.h"
#include "sun_hours.comp.h"
#include "sky.vert.h"
#include "sky.frag.h"
#include "route.vert.h"
#include "route.frag.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

// std140-ish camera block. All three matrices are written each frame; the
// shader uses viewProj, but view+proj are kept for future passes (horizon
// precompute) that may want them separately. lightViewProj projects world
// positions into the shadow map's clip space. sunDirAndIrradiance is xyz =
// unit direction TO sun (ENU frame), w = direct-beam W/m² (0 at night). vec4
// not vec3+float so std140 packing matches the C++ layout naturally.
struct alignas(16) CameraUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
    glm::mat4 lightViewProj;
    glm::mat4 invViewProj;           // sky.frag reconstructs view directions from this
    glm::vec4 sunDirAndIrradiance;
    glm::vec4 occlusionParams;       // .x = shadow-map enabled, .y = horizon-map enabled
    glm::vec4 sunHoursParams;        // .x = show sun-hours colormap, .y = max-hours scale
    glm::vec4 toneParams;            // .x = exposure (multiplier before ACES tonemap)
    glm::vec4 terrainAabb;           // .xy = (aabbMin.x, aabbMin.y), .zw = (aabbMax.x, aabbMax.y)
};

// Inputs to the SunDriver, owned by the UI. Defaults frame the Matterhorn
// (45.976°N, 7.658°E) at local noon today (CEST = UTC+2 in late May 2026).
struct SunUi {
    float latDeg    = 45.976f;
    float lonDeg    =  7.658f;
    int   year      = 2026;
    int   month     =    5;
    int   day       =   25;
    float localHour =   12.0f;
    float tzOffsetH =    2.0f;
};

// Occlusion controls. Shadow map = sun-POV depth pass, sharp near-terrain
// shadows but resolution-limited at this scale. Horizon map = per-texel
// precomputed apparent-elevation profile, captures large-scale ridges at O(1)
// per fragment; rebuilt only when the terrain changes (i.e. once at startup).
//
// Bias values forward straight to vkCmdSetDepthBias and only apply to the
// shadow map. Resolution of the shadow map comes from kShadowMapSize in
// forfun's types.h (currently 2048).
struct ShadowUi {
    bool  shadowMapEnabled  = true;
    bool  horizonMapEnabled = true;
    float depthBiasConstant = 1.25f;
    float depthBiasSlope    = 2.50f;
};

// Horizon-map dimensions. Picked to balance memory + bake cost.
//   512 × 512 × 32 × 2 bytes = 16 MB.
constexpr uint32_t kHorizonMapSize = 512;
constexpr uint32_t kHorizonBins    = 32;
// Raymarch params (also used by the compute shader via push constants).
constexpr float    kHorizonStepDistMeters = 150.0f;
constexpr int      kHorizonMaxSteps       = 128;

// Sun-hours-per-day visualisation. Output image matches the horizon map's
// XY layout so the terrain frag can reuse the same UV mapping. Compute time
// scales with kMaxSunSamples; 256 is comfortably enough for 5-min steps
// across 24 h while keeping the UBO around 4 KB.
constexpr uint32_t kSunHoursMapSize = 512;
constexpr uint32_t kMaxSunSamples   = 256;

struct SunHoursUi {
    bool  enabled       = false;
    float stepMinutes   = 10.0f;   // 144 samples/day at the default
    float maxHoursScale = 14.0f;   // normalises the viridis colormap
};

struct ToneUi {
    float exposure = 1.0f;
};

// Loaded GPX route. CPU keeps waypoints + ENU vertices around so we can build
// the waypoint sun-hours table; GPU holds a tightly-packed vertex buffer for
// the line-strip draw. Lifted a few metres above the sampled DEM height so
// the line floats above terrain instead of z-fighting it.
constexpr float kRouteLiftMetres = 15.0f;
struct Route {
    std::string                file;        // source path (for the UI status line)
    std::vector<gpx::Waypoint> waypoints;   // raw parser output (lat/lon/ele)
    std::vector<glm::vec3>     enu;         // world-space positions, post-lift
    std::vector<float>         sunHours;    // per-waypoint sun-hours/day (filled after bake)
    Buffer                     vb;
    uint32_t                   vertexCount = 0;
    bool                       visible     = true;
    float                      lengthKm    = 0.0f;
};

struct PendingGpxLoad {
    std::string path;
};

// Right-click picking. Cleared once the readback completes.
struct PickRequest {
    bool pending = false;
    int  pixelX  = 0;
    int  pixelY  = 0;
};
// Last successful pick — persists in the ImGui panel until replaced.
struct PickResult {
    bool      valid     = false;
    glm::vec3 worldPos  = glm::vec3(0.0f);
    double    latDeg    = 0.0;
    double    lonDeg    = 0.0;
    float     elevation = 0.0f;
    float     sunHours  = 0.0f;
};

// Matches the sun_hours.comp UBO layout in std140. The trailing pad keeps
// the struct size a multiple of 16 bytes so we can sizeof() it directly.
struct alignas(16) SunSamplesUBO {
    glm::vec4 dirs[kMaxSunSamples];
    int       count;
    float     hoursPerStep;
    float     _pad0;
    float     _pad1;
};

// Orbit camera: eye orbits around `target` on a sphere of radius `distance`.
// yaw is the compass angle from +X (east) measured CCW around +Z; pitch is the
// elevation angle above the horizontal plane through `target`.
struct OrbitCamera {
    glm::vec3 target      = glm::vec3(12000.0f, 53000.0f, 2500.0f);  // near Matterhorn
    float     yaw         = -2.356f;   // ~-135°: camera SW of target
    float     pitch       =  0.393f;   // ~22.5° above
    float     distance    = 30000.0f;
    float     minDistance =   500.0f;
    float     maxDistance = 250000.0f;
};

struct InputState {
    bool   leftDown      = false;
    bool   rightDown     = false;
    double lastX         = 0.0;
    double lastY         = 0.0;
    double scrollPending = 0.0;   // GLFW scroll deltas accumulate here; consumed each frame
};

// Globals so the GLFW scroll callback can reach them. We only ever have one
// window in this process, so a singleton is fine.
OrbitCamera g_camera;
InputState  g_input;
SunUi       g_sun;
ShadowUi    g_shadow;
SunHoursUi  g_sunHours;
ToneUi      g_tone;
PickRequest    g_pickRequest;
PickResult     g_pickResult;
PendingGpxLoad g_pendingGpx;     // populated by the GLFW drop callback; consumed at top of loop
Route          g_route;

void scrollCallback(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset) {
    // ImGui needs to see scroll over its widgets even though it installed its
    // own handler — when our callback was chained later, ImGui's chain helper
    // forwards for us. Still skip applying it to the camera when the cursor
    // is over UI.
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) return;
    g_input.scrollPending += yoffset;
}

glm::vec3 orbitEye(const OrbitCamera& c) {
    const float cp = std::cos(c.pitch);
    const float sp = std::sin(c.pitch);
    const float cy = std::cos(c.yaw);
    const float sy = std::sin(c.yaw);
    return c.target + c.distance * glm::vec3(cp * cy, cp * sy, sp);
}

// Fit an orthographic frustum to the terrain AABB rotated into light view
// space. The light "eye" looks down `-sunDir` toward the AABB centre; near/far
// span the projected depth range exactly, so the depth attachment never wastes
// resolution. Returns the combined lightProj * lightView so callers don't have
// to keep the two halves around.
//
// Caveat: glm::ortho with GLM_FORCE_DEPTH_ZERO_TO_ONE (defined in types.h)
// emits Vulkan-style Z∈[0,1]. We deliberately do NOT y-flip the projection
// here — the shadow map is sampled directly via the resulting clip-space
// position, so the Vulkan-vs-GL Y handedness cancels out as long as we treat
// both render and sample symmetrically.
glm::mat4 computeLightViewProj(const glm::vec3& sunDir,
                               const glm::vec3& aabbMin,
                               const glm::vec3& aabbMax) {
    const glm::vec3 centre = 0.5f * (aabbMin + aabbMax);
    // Distance along sunDir doesn't affect an ortho projection, but a non-zero
    // offset keeps lookAt() well-defined.
    const glm::vec3 lightEye = centre + sunDir;
    // Avoid the degenerate up vector when the sun sits near vertical.
    const glm::vec3 up = (std::abs(sunDir.z) > 0.99f)
        ? glm::vec3(0.0f, 1.0f, 0.0f)
        : glm::vec3(0.0f, 0.0f, 1.0f);
    const glm::mat4 lightView = glm::lookAt(lightEye, centre, up);

    const glm::vec3 corners[8] = {
        {aabbMin.x, aabbMin.y, aabbMin.z}, {aabbMax.x, aabbMin.y, aabbMin.z},
        {aabbMin.x, aabbMax.y, aabbMin.z}, {aabbMax.x, aabbMax.y, aabbMin.z},
        {aabbMin.x, aabbMin.y, aabbMax.z}, {aabbMax.x, aabbMin.y, aabbMax.z},
        {aabbMin.x, aabbMax.y, aabbMax.z}, {aabbMax.x, aabbMax.y, aabbMax.z},
    };
    constexpr float kBig = std::numeric_limits<float>::infinity();
    glm::vec3 lsMin( kBig,  kBig,  kBig);
    glm::vec3 lsMax(-kBig, -kBig, -kBig);
    for (const auto& c : corners) {
        const glm::vec3 lsC = glm::vec3(lightView * glm::vec4(c, 1.0f));
        lsMin = glm::min(lsMin, lsC);
        lsMax = glm::max(lsMax, lsC);
    }

    // In view space the camera looks down -Z, so a vertex with view-space Z
    // close to -lsMax.z is the nearest and -lsMin.z is the farthest.
    const glm::mat4 lightProj = glm::ortho(lsMin.x, lsMax.x,
                                           lsMin.y, lsMax.y,
                                           -lsMax.z, -lsMin.z);
    return lightProj * lightView;
}

// Bilinear sample of the DEM at a given lat/lon. Used to drape GPX waypoints
// onto the terrain so the rendered polyline tracks the surface even when a
// GPX file omits <ele> or has stale elevations. Out-of-tile inputs clamp to
// the nearest edge.
float sampleDemBilinear(const dem::Tile& tile, double lon, double lat) {
    if (tile.width < 2 || tile.height < 2) return 0.0f;
    double px = (lon - tile.extent.minLon) / tile.pixelSizeLon;
    double py = (tile.extent.maxLat - lat ) / tile.pixelSizeLat;   // N→S
    px = std::clamp(px, 0.0, double(tile.width  - 1));
    py = std::clamp(py, 0.0, double(tile.height - 1));
    const int x0 = static_cast<int>(std::floor(px));
    const int y0 = static_cast<int>(std::floor(py));
    const int x1 = std::min(x0 + 1, int(tile.width  - 1));
    const int y1 = std::min(y0 + 1, int(tile.height - 1));
    const float fx = static_cast<float>(px - x0);
    const float fy = static_cast<float>(py - y0);
    const float v00 = tile.elevation[size_t(y0) * tile.width + x0];
    const float v10 = tile.elevation[size_t(y0) * tile.width + x1];
    const float v01 = tile.elevation[size_t(y1) * tile.width + x0];
    const float v11 = tile.elevation[size_t(y1) * tile.width + x1];
    return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v10)
         +         fy  * ((1.0f - fx) * v01 + fx * v11);
}

// Populates `route.sunHours` by looking up each waypoint's world-XY in the
// CPU-side sun-hours grid. Cheap — one nearest-neighbour read per waypoint.
void updateRouteSunHoursFromReadback(Route& route,
                                     const float* sunHoursData,
                                     int sunHoursSize,
                                     const glm::vec3& aabbMin,
                                     const glm::vec3& aabbMax) {
    if (route.vertexCount == 0) {
        route.sunHours.clear();
        return;
    }
    const glm::vec2 extent(aabbMax.x - aabbMin.x, aabbMax.y - aabbMin.y);
    route.sunHours.resize(route.vertexCount);
    for (uint32_t i = 0; i < route.vertexCount; ++i) {
        const glm::vec3& p = route.enu[i];
        const float u = (p.x - aabbMin.x) / extent.x;
        const float v = (aabbMax.y - p.y) / extent.y;       // N→S
        const int x = std::clamp(int(u * sunHoursSize), 0, sunHoursSize - 1);
        const int y = std::clamp(int(v * sunHoursSize), 0, sunHoursSize - 1);
        route.sunHours[i] = sunHoursData[y * sunHoursSize + x];
    }
}

// GLFW drop callback — runs on the main thread during glfwPollEvents. We
// can't do GPU work here (no access to mesh/gpu) so we just record the path;
// the per-frame loop consumes it before recording commands.
void dropCallback(GLFWwindow*, int count, const char** paths) {
    for (int i = 0; i < count; ++i) {
        std::string p = paths[i];
        if (p.size() < 4) continue;
        std::string ext = p.substr(p.size() - 4);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (ext == ".gpx") {
            g_pendingGpx.path = std::move(p);
            return;
        }
    }
}

// Matt Zucker's polynomial viridis fit — same coefficients as the GLSL copy
// in terrain.frag so the ImGui legend matches the on-terrain colours exactly.
glm::vec3 viridisCpu(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    const glm::vec3 c0( 0.2777273272234177f,  0.005407344544966578f, 0.3340998053353061f);
    const glm::vec3 c1( 0.1050930431085774f,  1.404613529898575f,    1.384590162594685f);
    const glm::vec3 c2(-0.3308618287255563f,  0.214847559468213f,    0.09509516302823659f);
    const glm::vec3 c3(-4.634230498983486f,  -5.799100973351585f,  -19.33244095627987f);
    const glm::vec3 c4( 6.228269936347081f,  14.17993336680509f,    56.69055260068105f);
    const glm::vec3 c5( 4.776384997670288f, -13.74514537774601f,   -65.35303263337234f);
    const glm::vec3 c6(-5.435455855934631f,   4.645852612178535f,   26.3124352495832f);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// Populates `out` with one calendar day's worth of sun directions, sampled
// every `stepMinutes` from local midnight to midnight. Below-horizon samples
// (.z < 0) are stored as-is — the compute shader filters them out per thread.
// Returns the number of populated entries (clamped to kMaxSunSamples).
int fillSunSamples(SunSamplesUBO& out, const SunUi& sun, float stepMinutes) {
    const float clampedStep = std::max(1.0f, stepMinutes);
    int stepsPerDay = static_cast<int>(std::ceil(24.0f * 60.0f / clampedStep));
    if (stepsPerDay > static_cast<int>(kMaxSunSamples)) stepsPerDay = kMaxSunSamples;
    const float hoursPerStep = 24.0f / static_cast<float>(stepsPerDay);
    out.count        = stepsPerDay;
    out.hoursPerStep = hoursPerStep;
    out._pad0        = 0.0f;
    out._pad1        = 0.0f;
    for (int i = 0; i < stepsPerDay; ++i) {
        const float localHour = (static_cast<float>(i) + 0.5f) * hoursPerStep;
        const sun::Sample s   = sun::compute(sun.latDeg, sun.lonDeg,
                                             sun.year, sun.month, sun.day,
                                             localHour, sun.tzOffsetH);
        out.dirs[i] = glm::vec4(s.directionToSun, 0.0f);
    }
    for (int i = stepsPerDay; i < static_cast<int>(kMaxSunSamples); ++i) {
        out.dirs[i] = glm::vec4(0.0f);
    }
    return stepsPerDay;
}

struct LoadedTerrain {
    terrain::Mesh mesh;
    dem::Tile     tile;   // kept around so we can upload the heightmap to the GPU
};

// Built-in sample regions. Each entry names a Copernicus GLO-30 tile (downloaded
// at configure time by CMake) and a lat/lon crop window within it. Mesh stride
// is per-preset because cropped regions can afford finer mesh detail without
// blowing up the vertex count.
struct TerrainPreset {
    const char* name;
    const char* tilePath;       // path baked in by CMake
    double      minLon, maxLon;
    double      minLat, maxLat;
    uint32_t    meshStride;     // stride=1 = full DEM detail, stride=8 = 1/64 verts
    // Observer defaults: where the sun is computed from + local timezone.
    // The camera also frames itself on this point at startup.
    double      observerLat;
    double      observerLon;
    float       tzOffsetH;
    float       camDistanceM;
    // Optional GPX route auto-loaded with the preset. Toggleable in the UI
    // via the standard "Show on terrain" checkbox; nullptr means no built-in.
    const char* builtInGpxPath;
    const char* description;
};

const TerrainPreset kPresets[] = {
    // ~9 km × 10 km centred on Matterhorn (45.976°N, 7.658°E). The peak sits
    // close to the north edge of the source GLO-30 tile (N45), so we can't
    // extend much further north without stitching another tile in. Full
    // DEM detail; CEST is UTC+2 (DST in late May). Auto-loads the Hörnli
    // ridge route (hut → summit) as a built-in.
    {"matterhorn", ALPINE_SUN_MATTERHORN_TILE_PATH,
     7.60, 7.72, 45.91, 46.00, 1,
     45.976, 7.658, 2.0f, 12000.0f,
     ALPINE_SUN_ROUTE_MATTERHORN_HORNLI,
     "Matterhorn close-up (~9 km × 10 km) — full DEM detail"},
    // The original wide preset: full 1° tile, stride=8 to keep the mesh tractable.
    {"matterhorn-wide", ALPINE_SUN_MATTERHORN_TILE_PATH,
     7.0, 8.0, 45.0, 46.0, 8,
     45.976, 7.658, 2.0f, 30000.0f,
     nullptr,
     "Matterhorn region (1° × 1°, ~80 km × 110 km) — coarse mesh"},
    // ~15 km × 10 km centred on Everest (27.988°N, 86.925°E). Includes Lhotse,
    // Nuptse and the upper Khumbu icefall area. Same tile-edge constraint as
    // Matterhorn: the summit is near the north edge of N27. Nepal Standard
    // Time is UTC+5:45. Auto-loads the South-Col summit route (Camp 2 →
    // Lhotse face → South Col → summit) as a built-in.
    {"everest", ALPINE_SUN_EVEREST_TILE_PATH,
     86.85, 87.00, 27.91, 28.00, 1,
     27.988, 86.925, 5.75f, 14000.0f,
     ALPINE_SUN_ROUTE_EVEREST_SUMMIT,
     "Mt Everest close-up (~15 km × 10 km) — full DEM detail"},
};
constexpr int kPresetCount = sizeof(kPresets) / sizeof(kPresets[0]);

const TerrainPreset& findPresetOrDie(const char* name) {
    for (const auto& p : kPresets) {
        if (std::strcmp(p.name, name) == 0) return p;
    }
    std::fprintf(stderr, "alpine-sun: unknown terrain preset '%s'. Available:\n", name);
    for (const auto& p : kPresets) {
        std::fprintf(stderr, "  %-16s %s\n", p.name, p.description);
    }
    std::exit(EXIT_FAILURE);
}

LoadedTerrain loadAndBuildTerrain(const TerrainPreset& preset) {
    LoadedTerrain result;
    std::printf("preset: %s — %s\n", preset.name, preset.description);
    std::printf("dem: loading %s\n", preset.tilePath);
    if (!dem::loadGeoTIFF(preset.tilePath, result.tile)) {
        std::fprintf(stderr, "dem: load failed\n");
        std::exit(EXIT_FAILURE);
    }
    if (!dem::cropTile(result.tile,
                       preset.minLon, preset.maxLon,
                       preset.minLat, preset.maxLat)) {
        std::fprintf(stderr, "dem: crop failed\n");
        std::exit(EXIT_FAILURE);
    }
    const dem::Tile& tile = result.tile;
    std::printf("dem: %u x %u, lon [%.4f .. %.4f], lat [%.4f .. %.4f]\n",
                tile.width, tile.height,
                tile.extent.minLon, tile.extent.maxLon,
                tile.extent.minLat, tile.extent.maxLat);
    std::printf("dem: pixel size %.6f° lon, %.6f° lat\n",
                tile.pixelSizeLon, tile.pixelSizeLat);
    std::printf("dem: elevation [%.1f .. %.1f] m\n",
                static_cast<double>(tile.minElevation),
                static_cast<double>(tile.maxElevation));

    terrain::MeshOptions opts;
    opts.stride = preset.meshStride;
    result.mesh = terrain::makeMesh(result.tile, opts);
    const terrain::Mesh& mesh = result.mesh;
    std::printf("mesh: %zu verts, %zu tris, stride=%u\n",
                mesh.vertices.size(), mesh.indices.size() / 3, opts.stride);
    std::printf("mesh: AABB min=(%.0f, %.0f, %.0f) max=(%.0f, %.0f, %.0f) m\n",
                mesh.aabbMin.x, mesh.aabbMin.y, mesh.aabbMin.z,
                mesh.aabbMax.x, mesh.aabbMax.y, mesh.aabbMax.z);
    std::printf("mesh: ENU origin lon=%.4f lat=%.4f, m/deg lon=%.1f lat=%.1f\n",
                mesh.frame.centerLon, mesh.frame.centerLat,
                mesh.frame.metresPerDegreeLon, mesh.frame.metresPerDegreeLat);
    return result;
}

} // namespace

int main(int argc, char** argv) {
    // CLI: --terrain <name>. Default to the close Matterhorn view.
    const char* presetName = "matterhorn";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--terrain") == 0 && i + 1 < argc) {
            presetName = argv[++i];
        } else if (std::strcmp(argv[i], "--help") == 0
                || std::strcmp(argv[i], "-h") == 0) {
            std::printf("Usage: alpine_sun [--terrain <name>]\n\nPresets:\n");
            for (const auto& p : kPresets) {
                std::printf("  %-16s %s\n", p.name, p.description);
            }
            return EXIT_SUCCESS;
        }
    }
    const TerrainPreset& preset = findPresetOrDie(presetName);

    // Apply the preset's observer + camera defaults BEFORE loading so the
    // first sun-hours bake uses the right location/timezone.
    g_sun.latDeg    = static_cast<float>(preset.observerLat);
    g_sun.lonDeg    = static_cast<float>(preset.observerLon);
    g_sun.tzOffsetH = preset.tzOffsetH;

    LoadedTerrain loaded = loadAndBuildTerrain(preset);
    const terrain::Mesh& mesh = loaded.mesh;
    const dem::Tile&     tile = loaded.tile;

    // Frame the camera on the observer point at preset-defined distance. The
    // observer's (lat, lon) maps to (east, north) via the mesh's ENU frame;
    // we lift z to the AABB top so the target sits roughly on the peak.
    g_camera.target = glm::vec3(
        static_cast<float>(geo::lonToEast (mesh.frame, preset.observerLon)),
        static_cast<float>(geo::latToNorth(mesh.frame, preset.observerLat)),
        mesh.aabbMax.z);
    g_camera.distance = preset.camDistanceM;
    g_camera.maxDistance = std::max(g_camera.maxDistance, preset.camDistanceM * 4.0f);

    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit failed\n");
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "alpine-sun", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }

    forfun::Device gpu = forfun::createDevice({
        .appName          = "alpine-sun",
        .enableValidation = true,
        .createSurface    = [window](VkInstance inst) {
            VkSurfaceKHR s = VK_NULL_HANDLE;
            VK_CHECK(glfwCreateWindowSurface(inst, window, nullptr, &s));
            return s;
        },
    });
    std::printf("vulkan: device created, graphicsFamily=%u\n", gpu.graphicsFamily);

    forfun::Swapchain sc = forfun::createSwapchain(gpu, {
        .desiredExtent      = {1280, 720},
        .desiredFormat      = VK_FORMAT_B8G8R8A8_SRGB,
        .desiredColorSpace  = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .desiredPresentMode = VK_PRESENT_MODE_FIFO_KHR,
    });
    std::printf("vulkan: swapchain created, %ux%u, %zu images\n",
                sc.extent.width, sc.extent.height, sc.images.size());

    std::array<forfun::FrameContext, kFramesInFlight> frames{};
    for (auto& f : frames) f = forfun::createFrameContext(gpu);
    std::printf("vulkan: %u frame contexts ready (per-frame cmd pool + sync)\n",
                kFramesInFlight);

    // ---- Depth attachment ----
    // TRANSFER_SRC so the picking path can copy a single texel back to a
    // host buffer at the end of the frame.
    DepthImage depth = createDepthImage(gpu.allocator, gpu.device, sc.extent,
                                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    // ---- Sun-POV shadow map (depth-only, also sampled by terrain frag) ----
    DepthImage shadowMap = createDepthImage(
        gpu.allocator, gpu.device, {kShadowMapSize, kShadowMapSize},
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    // Sampler config mirrors forfun_core's createShadowSampler: nearest taps
    // (we PCF manually in the shader), CLAMP_TO_BORDER with white border so
    // shadow-frustum-relative out-of-bounds reads come back "fully lit".
    VkSampler shadowSampler = VK_NULL_HANDLE;
    {
        VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.magFilter    = VK_FILTER_NEAREST;
        ci.minFilter    = VK_FILTER_NEAREST;
        ci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        ci.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK(vkCreateSampler(gpu.device, &ci, nullptr, &shadowSampler));
    }

    // ---- Heightmap texture (full-resolution DEM, R32F) ----
    // Used only as input to the horizon-map compute bake — terrain rendering
    // itself uses the prebaked mesh. Layout ends up SHADER_READ_ONLY_OPTIMAL.
    VkImage         heightMapImage  = VK_NULL_HANDLE;
    VkImageView     heightMapView   = VK_NULL_HANDLE;
    VmaAllocation   heightMapAlloc  = VK_NULL_HANDLE;
    VkSampler       heightMapSampler = VK_NULL_HANDLE;
    {
        VkImageCreateInfo imageCi{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageCi.imageType     = VK_IMAGE_TYPE_2D;
        imageCi.format        = VK_FORMAT_R32_SFLOAT;
        imageCi.extent        = {tile.width, tile.height, 1};
        imageCi.mipLevels     = 1;
        imageCi.arrayLayers   = 1;
        imageCi.samples       = VK_SAMPLE_COUNT_1_BIT;
        imageCi.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imageCi.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageCi.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo allocCi{};
        allocCi.usage = VMA_MEMORY_USAGE_AUTO;
        VK_CHECK(vmaCreateImage(gpu.allocator, &imageCi, &allocCi,
                                &heightMapImage, &heightMapAlloc, nullptr));

        VkImageViewCreateInfo viewCi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewCi.image            = heightMapImage;
        viewCi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        viewCi.format           = VK_FORMAT_R32_SFLOAT;
        viewCi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(gpu.device, &viewCi, nullptr, &heightMapView));

        // Linear filter so the compute shader gets bilinear samples between
        // DEM texels — modestly smoother bake than nearest.
        VkSamplerCreateInfo sCi{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sCi.magFilter    = VK_FILTER_LINEAR;
        sCi.minFilter    = VK_FILTER_LINEAR;
        sCi.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sCi.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sCi.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sCi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(gpu.device, &sCi, nullptr, &heightMapSampler));
    }

    // ---- Horizon map (R16F, 2D array, layer = azimuth bin) ----
    // Storage-image-writable from the compute bake, then sampled by the
    // terrain frag.
    VkImage       horizonImage      = VK_NULL_HANDLE;
    VkImageView   horizonSampleView = VK_NULL_HANDLE;  // sampled as array
    VkImageView   horizonStorageView= VK_NULL_HANDLE;  // bound for write
    VmaAllocation horizonAlloc      = VK_NULL_HANDLE;
    VkSampler     horizonSampler    = VK_NULL_HANDLE;
    {
        VkImageCreateInfo imageCi{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageCi.imageType     = VK_IMAGE_TYPE_2D;
        imageCi.format        = VK_FORMAT_R16_SFLOAT;
        imageCi.extent        = {kHorizonMapSize, kHorizonMapSize, 1};
        imageCi.mipLevels     = 1;
        imageCi.arrayLayers   = kHorizonBins;
        imageCi.samples       = VK_SAMPLE_COUNT_1_BIT;
        imageCi.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imageCi.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageCi.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo allocCi{};
        allocCi.usage = VMA_MEMORY_USAGE_AUTO;
        VK_CHECK(vmaCreateImage(gpu.allocator, &imageCi, &allocCi,
                                &horizonImage, &horizonAlloc, nullptr));

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image            = horizonImage;
        vci.viewType         = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vci.format           = VK_FORMAT_R16_SFLOAT;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, kHorizonBins};
        VK_CHECK(vkCreateImageView(gpu.device, &vci, nullptr, &horizonSampleView));
        VK_CHECK(vkCreateImageView(gpu.device, &vci, nullptr, &horizonStorageView));

        // Linear filter for spatial XY, but layer (bin) sampling is always
        // nearest in Vulkan — the frag shader manually interpolates between
        // adjacent bins instead.
        VkSamplerCreateInfo sCi{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sCi.magFilter    = VK_FILTER_LINEAR;
        sCi.minFilter    = VK_FILTER_LINEAR;
        sCi.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sCi.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sCi.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sCi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(gpu.device, &sCi, nullptr, &horizonSampler));
    }

    // ---- Transient pool for one-time uploads + depth transition ----
    VkCommandPool uploadPool = VK_NULL_HANDLE;
    {
        VkCommandPoolCreateInfo poolCi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolCi.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCi.queueFamilyIndex = gpu.graphicsFamily;
        VK_CHECK(vkCreateCommandPool(gpu.device, &poolCi, nullptr, &uploadPool));
    }

    // Depth image lives in DEPTH_ATTACHMENT_OPTIMAL for its whole lifetime.
    // CLEAR loadOp handles the per-frame reset to far plane, so we don't need
    // to transition it again. The shadow map goes straight to
    // SHADER_READ_ONLY_OPTIMAL so the terrain frag's descriptor is always
    // valid — when shadows are toggled on we re-transition it through
    // DEPTH_ATTACHMENT_OPTIMAL each frame.
    {
        VkCommandBufferAllocateInfo cbAi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbAi.commandPool        = uploadPool;
        cbAi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbAi.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateCommandBuffers(gpu.device, &cbAi, &cmd));

        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
        transitionImage(cmd, depth.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        transitionImage(cmd, shadowMap.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sub.commandBufferCount = 1;
        sub.pCommandBuffers    = &cmd;
        VK_CHECK(vkQueueSubmit(gpu.graphicsQueue, 1, &sub, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(gpu.graphicsQueue));
        vkFreeCommandBuffers(gpu.device, uploadPool, 1, &cmd);
    }

    // ---- Upload terrain VB + IB ----
    const VkDeviceSize vbBytes = sizeof(Vertex)   * mesh.vertices.size();
    const VkDeviceSize ibBytes = sizeof(uint32_t) * mesh.indices.size();
    Buffer terrainVB = createBufferGPU(gpu.allocator, vbBytes,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Buffer terrainIB = createBufferGPU(gpu.allocator, ibBytes,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT  | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    uploadToBuffer(gpu.device, gpu.graphicsQueue, uploadPool, gpu.allocator,
                   mesh.vertices.data(), vbBytes, terrainVB.buffer);
    uploadToBuffer(gpu.device, gpu.graphicsQueue, uploadPool, gpu.allocator,
                   mesh.indices.data(),  ibBytes, terrainIB.buffer);
    std::printf("gpu: terrain uploaded — VB %.1f MB, IB %.1f MB\n",
                vbBytes / (1024.0 * 1024.0), ibBytes / (1024.0 * 1024.0));

    // ---- Upload DEM as a sampled R32F texture (for horizon-map bake) ----
    {
        const VkDeviceSize demBytes =
            static_cast<VkDeviceSize>(tile.width) * tile.height * sizeof(float);

        VkBufferCreateInfo stagingCi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        stagingCi.size  = demBytes;
        stagingCi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo stagingAllocCi{};
        stagingAllocCi.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocCi.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                             | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VkBuffer          stagingBuf  = VK_NULL_HANDLE;
        VmaAllocation     stagingAlloc = VK_NULL_HANDLE;
        VmaAllocationInfo stagingInfo{};
        VK_CHECK(vmaCreateBuffer(gpu.allocator, &stagingCi, &stagingAllocCi,
                                 &stagingBuf, &stagingAlloc, &stagingInfo));
        std::memcpy(stagingInfo.pMappedData, tile.elevation.data(),
                    static_cast<size_t>(demBytes));

        VkCommandBufferAllocateInfo cbAi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbAi.commandPool        = uploadPool;
        cbAi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbAi.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateCommandBuffers(gpu.device, &cbAi, &cmd));

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        transitionImage(cmd, heightMapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent      = {tile.width, tile.height, 1};
        vkCmdCopyBufferToImage(cmd, stagingBuf, heightMapImage,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        transitionImage(cmd, heightMapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        VK_CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sub.commandBufferCount = 1;
        sub.pCommandBuffers    = &cmd;
        VK_CHECK(vkQueueSubmit(gpu.graphicsQueue, 1, &sub, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(gpu.graphicsQueue));
        vkFreeCommandBuffers(gpu.device, uploadPool, 1, &cmd);
        vmaDestroyBuffer(gpu.allocator, stagingBuf, stagingAlloc);
        std::printf("gpu: heightmap uploaded — %u x %u R32F (%.1f MB)\n",
                    tile.width, tile.height, demBytes / (1024.0 * 1024.0));
    }

    // ---- Bake horizon map (one-time compute dispatch) ----
    // Build a self-contained compute pipeline + descriptor set, dispatch,
    // tear it all down. The horizon image keeps its result in
    // SHADER_READ_ONLY_OPTIMAL for the rest of the program.
    {
        VkDescriptorSetLayoutBinding bakeBindings[2]{};
        bakeBindings[0].binding         = 0;
        bakeBindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bakeBindings[0].descriptorCount = 1;
        bakeBindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bakeBindings[1].binding         = 1;
        bakeBindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bakeBindings[1].descriptorCount = 1;
        bakeBindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dslCi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslCi.bindingCount = 2;
        dslCi.pBindings    = bakeBindings;
        VkDescriptorSetLayout bakeSetLayout = VK_NULL_HANDLE;
        VK_CHECK(vkCreateDescriptorSetLayout(gpu.device, &dslCi, nullptr, &bakeSetLayout));

        VkDescriptorPoolSize bakePoolSizes[2]{};
        bakePoolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bakePoolSizes[0].descriptorCount = 1;
        bakePoolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bakePoolSizes[1].descriptorCount = 1;
        VkDescriptorPoolCreateInfo dpCi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpCi.maxSets       = 1;
        dpCi.poolSizeCount = 2;
        dpCi.pPoolSizes    = bakePoolSizes;
        VkDescriptorPool bakeDescPool = VK_NULL_HANDLE;
        VK_CHECK(vkCreateDescriptorPool(gpu.device, &dpCi, nullptr, &bakeDescPool));

        VkDescriptorSet bakeDescSet = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = bakeDescPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &bakeSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(gpu.device, &ai, &bakeDescSet));

        VkDescriptorImageInfo heightInfo{};
        heightInfo.sampler     = heightMapSampler;
        heightInfo.imageView   = heightMapView;
        heightInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo horizonStoreInfo{};
        horizonStoreInfo.imageView   = horizonStorageView;
        horizonStoreInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet bakeWrites[2]{};
        bakeWrites[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        bakeWrites[0].dstSet          = bakeDescSet;
        bakeWrites[0].dstBinding      = 0;
        bakeWrites[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bakeWrites[0].descriptorCount = 1;
        bakeWrites[0].pImageInfo      = &heightInfo;
        bakeWrites[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        bakeWrites[1].dstSet          = bakeDescSet;
        bakeWrites[1].dstBinding      = 1;
        bakeWrites[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bakeWrites[1].descriptorCount = 1;
        bakeWrites[1].pImageInfo      = &horizonStoreInfo;
        vkUpdateDescriptorSets(gpu.device, 2, bakeWrites, 0, nullptr);

        VkPushConstantRange bakePush{};
        bakePush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bakePush.size       = sizeof(glm::vec4) * 2;  // aabbMinMax + params

        VkPipelineLayoutCreateInfo bakePlCi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        bakePlCi.setLayoutCount         = 1;
        bakePlCi.pSetLayouts            = &bakeSetLayout;
        bakePlCi.pushConstantRangeCount = 1;
        bakePlCi.pPushConstantRanges    = &bakePush;
        VkPipelineLayout bakePipelineLayout = VK_NULL_HANDLE;
        VK_CHECK(vkCreatePipelineLayout(gpu.device, &bakePlCi, nullptr, &bakePipelineLayout));

        VkShaderModule bakeModule = createShaderModule(
            gpu.device, horizon_map_comp_spv, sizeof(horizon_map_comp_spv));

        VkPipelineShaderStageCreateInfo bakeStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        bakeStage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        bakeStage.module = bakeModule;
        bakeStage.pName  = "main";

        VkComputePipelineCreateInfo bakeCpCi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        bakeCpCi.stage  = bakeStage;
        bakeCpCi.layout = bakePipelineLayout;
        VkPipeline bakePipeline = VK_NULL_HANDLE;
        VK_CHECK(vkCreateComputePipelines(gpu.device, VK_NULL_HANDLE, 1, &bakeCpCi, nullptr, &bakePipeline));
        vkDestroyShaderModule(gpu.device, bakeModule, nullptr);

        // Record + submit the bake.
        VkCommandBufferAllocateInfo cbAi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbAi.commandPool        = uploadPool;
        cbAi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbAi.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateCommandBuffers(gpu.device, &cbAi, &cmd));

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        transitionImageRange(cmd, horizonImage, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             0, 1, 0, kHorizonBins);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bakePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                bakePipelineLayout, 0, 1, &bakeDescSet, 0, nullptr);

        // Push constants: (aabbMin.x, aabbMin.y, aabbMax.x, aabbMax.y),
        // (stepDistMeters, float(maxSteps), 0, 0).
        struct {
            glm::vec4 aabbMinMax;
            glm::vec4 params;
        } push{};
        push.aabbMinMax = glm::vec4(mesh.aabbMin.x, mesh.aabbMin.y,
                                    mesh.aabbMax.x, mesh.aabbMax.y);
        push.params     = glm::vec4(kHorizonStepDistMeters,
                                    static_cast<float>(kHorizonMaxSteps),
                                    0.0f, 0.0f);
        vkCmdPushConstants(cmd, bakePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);

        const uint32_t groupsX = (kHorizonMapSize + 7) / 8;
        const uint32_t groupsY = (kHorizonMapSize + 7) / 8;
        vkCmdDispatch(cmd, groupsX, groupsY, kHorizonBins);

        transitionImageRange(cmd, horizonImage, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                             0, 1, 0, kHorizonBins);

        VK_CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sub.commandBufferCount = 1;
        sub.pCommandBuffers    = &cmd;
        VK_CHECK(vkQueueSubmit(gpu.graphicsQueue, 1, &sub, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(gpu.graphicsQueue));
        vkFreeCommandBuffers(gpu.device, uploadPool, 1, &cmd);

        vkDestroyPipeline(gpu.device, bakePipeline, nullptr);
        vkDestroyPipelineLayout(gpu.device, bakePipelineLayout, nullptr);
        vkDestroyDescriptorPool(gpu.device, bakeDescPool, nullptr);
        vkDestroyDescriptorSetLayout(gpu.device, bakeSetLayout, nullptr);
        std::printf("gpu: horizon map baked — %u² × %u bins, %.0fm step × %d steps\n",
                    kHorizonMapSize, kHorizonBins,
                    static_cast<double>(kHorizonStepDistMeters), kHorizonMaxSteps);
    }

    // ---- Sun-hours accumulator (single-layer R32F image) ----
    VkImage         sunHoursImage      = VK_NULL_HANDLE;
    VkImageView     sunHoursStorageView= VK_NULL_HANDLE;
    VkImageView     sunHoursSampleView = VK_NULL_HANDLE;
    VmaAllocation   sunHoursAlloc      = VK_NULL_HANDLE;
    VkSampler       sunHoursSampler    = VK_NULL_HANDLE;
    {
        VkImageCreateInfo imageCi{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageCi.imageType     = VK_IMAGE_TYPE_2D;
        imageCi.format        = VK_FORMAT_R32_SFLOAT;
        imageCi.extent        = {kSunHoursMapSize, kSunHoursMapSize, 1};
        imageCi.mipLevels     = 1;
        imageCi.arrayLayers   = 1;
        imageCi.samples       = VK_SAMPLE_COUNT_1_BIT;
        imageCi.tiling        = VK_IMAGE_TILING_OPTIMAL;
        // TRANSFER_SRC so we can copy back to host every bake (waypoint table)
        // and copy single texels for right-click sampling.
        imageCi.usage         = VK_IMAGE_USAGE_STORAGE_BIT
                              | VK_IMAGE_USAGE_SAMPLED_BIT
                              | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageCi.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo allocCi{};
        allocCi.usage = VMA_MEMORY_USAGE_AUTO;
        VK_CHECK(vmaCreateImage(gpu.allocator, &imageCi, &allocCi,
                                &sunHoursImage, &sunHoursAlloc, nullptr));

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image            = sunHoursImage;
        vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        vci.format           = VK_FORMAT_R32_SFLOAT;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(gpu.device, &vci, nullptr, &sunHoursStorageView));
        VK_CHECK(vkCreateImageView(gpu.device, &vci, nullptr, &sunHoursSampleView));

        VkSamplerCreateInfo sCi{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sCi.magFilter    = VK_FILTER_LINEAR;
        sCi.minFilter    = VK_FILTER_LINEAR;
        sCi.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sCi.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sCi.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sCi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(gpu.device, &sCi, nullptr, &sunHoursSampler));
    }

    // ---- SunSamples UBO (host-mapped — rewritten each recompute) ----
    Buffer sunSamplesBuf = createBufferHostMapped(
        gpu.allocator, sizeof(SunSamplesUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // ---- Picking readback buffers ----
    // One float each: depth at cursor and sun-hours at picked texel. Host-mapped
    // so we can read straight from `mapped` after vkQueueWaitIdle.
    Buffer pickDepthBuf = createBufferHostMapped(
        gpu.allocator, sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Buffer pickHoursBuf = createBufferHostMapped(
        gpu.allocator, sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // ---- Sun-hours full-texture readback (for waypoint table) ----
    // 1 MB copy after every bake — populates a CPU-resident grid we can query
    // at any (worldX, worldY) without round-tripping through Vulkan again.
    const VkDeviceSize kSunHoursReadbackBytes =
        static_cast<VkDeviceSize>(kSunHoursMapSize) * kSunHoursMapSize * sizeof(float);
    Buffer sunHoursReadback = createBufferHostMapped(
        gpu.allocator, kSunHoursReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // ---- Sun-hours compute pipeline (persistent — re-dispatched on date change) ----
    VkDescriptorSetLayout sunHoursSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      sunHoursDescPool  = VK_NULL_HANDLE;
    VkDescriptorSet       sunHoursDescSet   = VK_NULL_HANDLE;
    VkPipelineLayout      sunHoursPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            sunHoursPipeline       = VK_NULL_HANDLE;
    {
        VkDescriptorSetLayoutBinding bindings[3]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[2].binding         = 2;
        bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dslCi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslCi.bindingCount = 3;
        dslCi.pBindings    = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(gpu.device, &dslCi, nullptr, &sunHoursSetLayout));

        VkDescriptorPoolSize poolSizes[3]{};
        poolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[1].descriptorCount = 1;
        poolSizes[2].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[2].descriptorCount = 1;
        VkDescriptorPoolCreateInfo dpCi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpCi.maxSets       = 1;
        dpCi.poolSizeCount = 3;
        dpCi.pPoolSizes    = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(gpu.device, &dpCi, nullptr, &sunHoursDescPool));

        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = sunHoursDescPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &sunHoursSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(gpu.device, &ai, &sunHoursDescSet));

        VkDescriptorImageInfo horizonReadInfo{};
        horizonReadInfo.sampler     = horizonSampler;
        horizonReadInfo.imageView   = horizonSampleView;
        horizonReadInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo sunHoursStoreInfo{};
        sunHoursStoreInfo.imageView   = sunHoursStorageView;
        sunHoursStoreInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorBufferInfo samplesInfo{};
        samplesInfo.buffer = sunSamplesBuf.buffer;
        samplesInfo.range  = sizeof(SunSamplesUBO);

        VkWriteDescriptorSet writes[3]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = sunHoursDescSet;
        writes[0].dstBinding      = 0;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo      = &horizonReadInfo;
        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = sunHoursDescSet;
        writes[1].dstBinding      = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo      = &sunHoursStoreInfo;
        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = sunHoursDescSet;
        writes[2].dstBinding      = 2;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].descriptorCount = 1;
        writes[2].pBufferInfo     = &samplesInfo;
        vkUpdateDescriptorSets(gpu.device, 3, writes, 0, nullptr);

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.size       = sizeof(glm::vec4);  // aabbMinMax

        VkPipelineLayoutCreateInfo plCi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plCi.setLayoutCount         = 1;
        plCi.pSetLayouts            = &sunHoursSetLayout;
        plCi.pushConstantRangeCount = 1;
        plCi.pPushConstantRanges    = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(gpu.device, &plCi, nullptr, &sunHoursPipelineLayout));

        VkShaderModule mod = createShaderModule(gpu.device,
            sun_hours_comp_spv, sizeof(sun_hours_comp_spv));
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";
        VkComputePipelineCreateInfo cpCi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpCi.stage  = stage;
        cpCi.layout = sunHoursPipelineLayout;
        VK_CHECK(vkCreateComputePipelines(gpu.device, VK_NULL_HANDLE, 1, &cpCi, nullptr, &sunHoursPipeline));
        vkDestroyShaderModule(gpu.device, mod, nullptr);
    }

    // Dedicated transient pool for sun-hours bakes — survives the program's
    // lifetime because we re-bake whenever the user changes date/location.
    VkCommandPool sunHoursPool = VK_NULL_HANDLE;
    {
        VkCommandPoolCreateInfo poolCi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolCi.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
                                | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolCi.queueFamilyIndex = gpu.graphicsFamily;
        VK_CHECK(vkCreateCommandPool(gpu.device, &poolCi, nullptr, &sunHoursPool));
    }

    // Tracks the parameter set baked last time so we only re-dispatch when
    // something the bake depends on actually changed.
    struct {
        float latDeg     = std::numeric_limits<float>::quiet_NaN();
        float lonDeg     = 0.0f;
        int   year       = 0;
        int   month      = 0;
        int   day        = 0;
        float tzOffsetH  = 0.0f;
        float stepMin    = 0.0f;
        bool initialized = false;
    } lastBakedParams;

    // Runs the sun-hours compute. Synchronous (waits for the queue) — the
    // dispatch is small enough (~few ms) that the brief stall is invisible.
    auto runSunHoursBake = [&]() {
        SunSamplesUBO* samples = static_cast<SunSamplesUBO*>(sunSamplesBuf.mapped);
        const int n = fillSunSamples(*samples, g_sun, g_sunHours.stepMinutes);

        VkCommandBufferAllocateInfo cbAi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbAi.commandPool        = sunHoursPool;
        cbAi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbAi.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateCommandBuffers(gpu.device, &cbAi, &cmd));

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        // Discard previous contents (UNDEFINED) — we overwrite every texel.
        transitionImage(cmd, sunHoursImage, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sunHoursPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                sunHoursPipelineLayout, 0, 1, &sunHoursDescSet, 0, nullptr);

        glm::vec4 push(mesh.aabbMin.x, mesh.aabbMin.y, mesh.aabbMax.x, mesh.aabbMax.y);
        vkCmdPushConstants(cmd, sunHoursPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);

        const uint32_t groupsX = (kSunHoursMapSize + 7) / 8;
        const uint32_t groupsY = (kSunHoursMapSize + 7) / 8;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        // Copy the just-baked accumulator to a host-visible buffer so the
        // route waypoint table can look up sun-hours per point without
        // round-tripping the GPU again. We go through TRANSFER_SRC because
        // imageStore writes need to be flushed before COPY can read them.
        transitionImage(cmd, sunHoursImage, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COPY_BIT,
                        VK_ACCESS_2_TRANSFER_READ_BIT);

        VkBufferImageCopy rbRegion{};
        rbRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        rbRegion.imageSubresource.layerCount = 1;
        rbRegion.imageExtent = {kSunHoursMapSize, kSunHoursMapSize, 1};
        vkCmdCopyImageToBuffer(cmd, sunHoursImage,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               sunHoursReadback.buffer, 1, &rbRegion);

        transitionImage(cmd, sunHoursImage, VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COPY_BIT,
                        VK_ACCESS_2_TRANSFER_READ_BIT,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        VK_CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sub.commandBufferCount = 1;
        sub.pCommandBuffers    = &cmd;
        VK_CHECK(vkQueueSubmit(gpu.graphicsQueue, 1, &sub, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(gpu.graphicsQueue));
        vkFreeCommandBuffers(gpu.device, sunHoursPool, 1, &cmd);

        lastBakedParams.latDeg      = g_sun.latDeg;
        lastBakedParams.lonDeg      = g_sun.lonDeg;
        lastBakedParams.year        = g_sun.year;
        lastBakedParams.month       = g_sun.month;
        lastBakedParams.day         = g_sun.day;
        lastBakedParams.tzOffsetH   = g_sun.tzOffsetH;
        lastBakedParams.stepMin     = g_sunHours.stepMinutes;
        lastBakedParams.initialized = true;
        std::printf("gpu: sun-hours baked — %d samples × %.1f h step\n",
                    n, static_cast<double>(samples->hoursPerStep));

        // Pull the freshly-baked accumulator into our route waypoint cache.
        // Safe to read now — the queue wait above guarantees the copy has
        // landed in sunHoursReadback.mapped.
        updateRouteSunHoursFromReadback(
            g_route,
            static_cast<const float*>(sunHoursReadback.mapped),
            int(kSunHoursMapSize),
            mesh.aabbMin, mesh.aabbMax);
    };

    // Initial bake so the descriptor binding is always backed by valid data.
    runSunHoursBake();

    // ---- Camera UBO + per-frame descriptor sets ----
    std::array<Buffer, kFramesInFlight> cameraUbos{};
    for (auto& b : cameraUbos) {
        b = createBufferHostMapped(gpu.allocator, sizeof(CameraUBO),
                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    }

    // Four bindings in one set:
    //   b0 = camera UBO (both stages),
    //   b1 = shadow map sampler2D (frag),
    //   b2 = horizon map sampler2DArray (frag),
    //   b3 = sun-hours sampler2D (frag).
    VkDescriptorSetLayoutBinding setBindings[4]{};
    setBindings[0].binding         = 0;
    setBindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    setBindings[0].descriptorCount = 1;
    setBindings[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    setBindings[1].binding         = 1;
    setBindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    setBindings[1].descriptorCount = 1;
    setBindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    setBindings[2].binding         = 2;
    setBindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    setBindings[2].descriptorCount = 1;
    setBindings[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    setBindings[3].binding         = 3;
    setBindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    setBindings[3].descriptorCount = 1;
    setBindings[3].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dslCi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCi.bindingCount = 4;
    dslCi.pBindings    = setBindings;
    VkDescriptorSetLayout cameraSetLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(gpu.device, &dslCi, nullptr, &cameraSetLayout));

    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = kFramesInFlight;
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = kFramesInFlight * 3;  // shadow + horizon + sun-hours per frame
    VkDescriptorPoolCreateInfo dpCi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCi.maxSets       = kFramesInFlight;
    dpCi.poolSizeCount = 2;
    dpCi.pPoolSizes    = poolSizes;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(gpu.device, &dpCi, nullptr, &descPool));

    std::array<VkDescriptorSet, kFramesInFlight> cameraDescSets{};
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = descPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &cameraSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(gpu.device, &ai, &cameraDescSets[i]));

        VkDescriptorBufferInfo bi{};
        bi.buffer = cameraUbos[i].buffer;
        bi.range  = sizeof(CameraUBO);

        VkDescriptorImageInfo shadowInfo{};
        shadowInfo.sampler     = shadowSampler;
        shadowInfo.imageView   = shadowMap.view;
        shadowInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo horizonInfo{};
        horizonInfo.sampler     = horizonSampler;
        horizonInfo.imageView   = horizonSampleView;
        horizonInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo sunHoursInfo{};
        sunHoursInfo.sampler     = sunHoursSampler;
        sunHoursInfo.imageView   = sunHoursSampleView;
        sunHoursInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet writes[4]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = cameraDescSets[i];
        writes[0].dstBinding      = 0;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo     = &bi;
        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = cameraDescSets[i];
        writes[1].dstBinding      = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo      = &shadowInfo;
        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = cameraDescSets[i];
        writes[2].dstBinding      = 2;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo      = &horizonInfo;
        writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet          = cameraDescSets[i];
        writes[3].dstBinding      = 3;
        writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].descriptorCount = 1;
        writes[3].pImageInfo      = &sunHoursInfo;
        vkUpdateDescriptorSets(gpu.device, 4, writes, 0, nullptr);
    }

    // ---- Terrain pipeline ----
    VkShaderModule vertModule = createShaderModule(gpu.device, terrain_vert_spv, sizeof(terrain_vert_spv));
    VkShaderModule fragModule = createShaderModule(gpu.device, terrain_frag_spv, sizeof(terrain_frag_spv));

    VkPipelineLayoutCreateInfo plCi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCi.setLayoutCount = 1;
    plCi.pSetLayouts    = &cameraSetLayout;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(gpu.device, &plCi, nullptr, &pipelineLayout));

    VkPipelineShaderStageCreateInfo stages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
    };
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName  = "main";
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName  = "main";

    VkVertexInputBindingDescription vbBinding{};
    vbBinding.binding   = 0;
    vbBinding.stride    = sizeof(Vertex);
    vbBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vbAttrs[3]{};
    vbAttrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
    vbAttrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
    vbAttrs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv)};

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &vbBinding;
    vi.vertexAttributeDescriptionCount = 3;
    vi.pVertexAttributeDescriptions    = vbAttrs;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_BACK_BIT;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments    = &cba;

    VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynStates;

    VkPipelineRenderingCreateInfo prCi{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    prCi.colorAttachmentCount    = 1;
    prCi.pColorAttachmentFormats = &sc.imageFormat;
    prCi.depthAttachmentFormat   = kDepthFormat;

    VkGraphicsPipelineCreateInfo gpCi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpCi.pNext               = &prCi;
    gpCi.stageCount          = 2;
    gpCi.pStages             = stages;
    gpCi.pVertexInputState   = &vi;
    gpCi.pInputAssemblyState = &ia;
    gpCi.pViewportState      = &vp;
    gpCi.pRasterizationState = &rs;
    gpCi.pMultisampleState   = &ms;
    gpCi.pDepthStencilState  = &ds;
    gpCi.pColorBlendState    = &cb;
    gpCi.pDynamicState       = &dyn;
    gpCi.layout              = pipelineLayout;
    VkPipeline terrainPipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(gpu.device, VK_NULL_HANDLE, 1, &gpCi, nullptr, &terrainPipeline));

    // Shader modules can be destroyed once the pipeline is built.
    vkDestroyShaderModule(gpu.device, fragModule, nullptr);
    vkDestroyShaderModule(gpu.device, vertModule, nullptr);

    std::printf("gpu: terrain pipeline ready\n");

    // ---- Shadow (depth-only) pipeline ----
    // Push constant carries the light-space VP matrix so the shadow pass
    // doesn't need a descriptor set. Same vertex layout as the terrain
    // pipeline (we draw the same VB/IB). VK_DYNAMIC_STATE_DEPTH_BIAS lets the
    // ImGui sliders retune slope/constant bias without rebuilding the pipeline.
    VkShaderModule depthVertModule = createShaderModule(
        gpu.device, terrain_depth_vert_spv, sizeof(terrain_depth_vert_spv));

    VkPushConstantRange shadowPush{};
    shadowPush.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    shadowPush.offset     = 0;
    shadowPush.size       = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo shadowPlCi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    shadowPlCi.pushConstantRangeCount = 1;
    shadowPlCi.pPushConstantRanges    = &shadowPush;
    VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(gpu.device, &shadowPlCi, nullptr, &shadowPipelineLayout));

    VkPipelineShaderStageCreateInfo shadowStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    shadowStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    shadowStage.module = depthVertModule;
    shadowStage.pName  = "main";

    VkPipelineRasterizationStateCreateInfo shadowRs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    shadowRs.polygonMode      = VK_POLYGON_MODE_FILL;
    shadowRs.cullMode         = VK_CULL_MODE_BACK_BIT;
    shadowRs.frontFace        = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    shadowRs.lineWidth        = 1.0f;
    shadowRs.depthBiasEnable  = VK_TRUE;  // actual values come via vkCmdSetDepthBias

    VkPipelineMultisampleStateCreateInfo shadowMs{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    shadowMs.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo shadowDs{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    shadowDs.depthTestEnable  = VK_TRUE;
    shadowDs.depthWriteEnable = VK_TRUE;
    shadowDs.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendStateCreateInfo shadowCb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    // colorAttachmentCount=0 — no color output.

    VkDynamicState shadowDynStates[3] = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS,
    };
    VkPipelineDynamicStateCreateInfo shadowDyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    shadowDyn.dynamicStateCount = 3;
    shadowDyn.pDynamicStates    = shadowDynStates;

    VkPipelineRenderingCreateInfo shadowPrCi{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    shadowPrCi.colorAttachmentCount = 0;
    shadowPrCi.depthAttachmentFormat = kDepthFormat;

    VkGraphicsPipelineCreateInfo shadowGpCi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    shadowGpCi.pNext               = &shadowPrCi;
    shadowGpCi.stageCount          = 1;
    shadowGpCi.pStages             = &shadowStage;
    shadowGpCi.pVertexInputState   = &vi;
    shadowGpCi.pInputAssemblyState = &ia;
    shadowGpCi.pViewportState      = &vp;
    shadowGpCi.pRasterizationState = &shadowRs;
    shadowGpCi.pMultisampleState   = &shadowMs;
    shadowGpCi.pDepthStencilState  = &shadowDs;
    shadowGpCi.pColorBlendState    = &shadowCb;
    shadowGpCi.pDynamicState       = &shadowDyn;
    shadowGpCi.layout              = shadowPipelineLayout;
    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(gpu.device, VK_NULL_HANDLE, 1, &shadowGpCi, nullptr, &shadowPipeline));

    vkDestroyShaderModule(gpu.device, depthVertModule, nullptr);

    std::printf("gpu: shadow pipeline ready (%ux%u depth map)\n",
                kShadowMapSize, kShadowMapSize);

    // ---- Sky pipeline (fullscreen triangle, runs first in the main pass) ----
    // Reuses the camera descriptor set layout: only binding 0 (UBO) is touched
    // by sky.frag; the other bindings (samplers) are bound but unread, which
    // is legal.
    VkPipeline       skyPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout skyPipelineLayout = VK_NULL_HANDLE;
    {
        VkShaderModule skyVertMod = createShaderModule(gpu.device,
            sky_vert_spv, sizeof(sky_vert_spv));
        VkShaderModule skyFragMod = createShaderModule(gpu.device,
            sky_frag_spv, sizeof(sky_frag_spv));

        VkPipelineLayoutCreateInfo skyPlCi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        skyPlCi.setLayoutCount = 1;
        skyPlCi.pSetLayouts    = &cameraSetLayout;
        VK_CHECK(vkCreatePipelineLayout(gpu.device, &skyPlCi, nullptr, &skyPipelineLayout));

        VkPipelineShaderStageCreateInfo skyStages[2]{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
        };
        skyStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        skyStages[0].module = skyVertMod;
        skyStages[0].pName  = "main";
        skyStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        skyStages[1].module = skyFragMod;
        skyStages[1].pName  = "main";

        // No vertex input — gl_VertexIndex drives the fullscreen triangle.
        VkPipelineVertexInputStateCreateInfo skyVi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

        VkPipelineInputAssemblyStateCreateInfo skyIa{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        skyIa.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo skyVp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        skyVp.viewportCount = 1;
        skyVp.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo skyRs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        skyRs.polygonMode = VK_POLYGON_MODE_FILL;
        skyRs.cullMode    = VK_CULL_MODE_NONE;
        skyRs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        skyRs.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo skyMs{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        skyMs.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Depth test off so we fill every pixel; depth write off so terrain
        // can claim depth normally afterwards.
        VkPipelineDepthStencilStateCreateInfo skyDs{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        skyDs.depthTestEnable  = VK_FALSE;
        skyDs.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState skyCba{};
        skyCba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                              | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo skyCb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        skyCb.attachmentCount = 1;
        skyCb.pAttachments    = &skyCba;

        VkDynamicState skyDynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo skyDyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        skyDyn.dynamicStateCount = 2;
        skyDyn.pDynamicStates    = skyDynStates;

        VkPipelineRenderingCreateInfo skyPrCi{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        skyPrCi.colorAttachmentCount    = 1;
        skyPrCi.pColorAttachmentFormats = &sc.imageFormat;
        skyPrCi.depthAttachmentFormat   = kDepthFormat;

        VkGraphicsPipelineCreateInfo skyGpCi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        skyGpCi.pNext               = &skyPrCi;
        skyGpCi.stageCount          = 2;
        skyGpCi.pStages             = skyStages;
        skyGpCi.pVertexInputState   = &skyVi;
        skyGpCi.pInputAssemblyState = &skyIa;
        skyGpCi.pViewportState      = &skyVp;
        skyGpCi.pRasterizationState = &skyRs;
        skyGpCi.pMultisampleState   = &skyMs;
        skyGpCi.pDepthStencilState  = &skyDs;
        skyGpCi.pColorBlendState    = &skyCb;
        skyGpCi.pDynamicState       = &skyDyn;
        skyGpCi.layout              = skyPipelineLayout;
        VK_CHECK(vkCreateGraphicsPipelines(gpu.device, VK_NULL_HANDLE, 1, &skyGpCi, nullptr, &skyPipeline));

        vkDestroyShaderModule(gpu.device, skyFragMod, nullptr);
        vkDestroyShaderModule(gpu.device, skyVertMod, nullptr);
    }
    std::printf("gpu: sky pipeline ready\n");

    // ---- Route pipeline (line strip over the terrain) ----
    // Shares the camera descriptor set layout (uses only binding 0). Single
    // vec3 vertex attribute, depth-tested so points behind ridges hide
    // correctly, no culling (lines have no front/back).
    VkPipeline       routePipeline       = VK_NULL_HANDLE;
    VkPipelineLayout routePipelineLayout = VK_NULL_HANDLE;
    {
        VkShaderModule rVertMod = createShaderModule(gpu.device,
            route_vert_spv, sizeof(route_vert_spv));
        VkShaderModule rFragMod = createShaderModule(gpu.device,
            route_frag_spv, sizeof(route_frag_spv));

        VkPipelineLayoutCreateInfo routePlCi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        routePlCi.setLayoutCount = 1;
        routePlCi.pSetLayouts    = &cameraSetLayout;
        VK_CHECK(vkCreatePipelineLayout(gpu.device, &routePlCi, nullptr, &routePipelineLayout));

        VkPipelineShaderStageCreateInfo rStages[2]{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
        };
        rStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        rStages[0].module = rVertMod;
        rStages[0].pName  = "main";
        rStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        rStages[1].module = rFragMod;
        rStages[1].pName  = "main";

        VkVertexInputBindingDescription rvbBinding{};
        rvbBinding.binding   = 0;
        rvbBinding.stride    = sizeof(glm::vec3);
        rvbBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription rvbAttr{};
        rvbAttr.location = 0;
        rvbAttr.binding  = 0;
        rvbAttr.format   = VK_FORMAT_R32G32B32_SFLOAT;
        rvbAttr.offset   = 0;

        VkPipelineVertexInputStateCreateInfo rVi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        rVi.vertexBindingDescriptionCount   = 1;
        rVi.pVertexBindingDescriptions      = &rvbBinding;
        rVi.vertexAttributeDescriptionCount = 1;
        rVi.pVertexAttributeDescriptions    = &rvbAttr;

        VkPipelineInputAssemblyStateCreateInfo rIa{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        rIa.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;

        VkPipelineViewportStateCreateInfo rVp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        rVp.viewportCount = 1;
        rVp.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo rRs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rRs.polygonMode = VK_POLYGON_MODE_FILL;
        rRs.cullMode    = VK_CULL_MODE_NONE;
        rRs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rRs.lineWidth   = 1.0f;  // overridden by dynamic state below

        VkPipelineMultisampleStateCreateInfo rMs{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        rMs.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo rDs{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        rDs.depthTestEnable  = VK_TRUE;
        rDs.depthWriteEnable = VK_TRUE;
        rDs.depthCompareOp   = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState rCba{};
        rCba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo rCb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        rCb.attachmentCount = 1;
        rCb.pAttachments    = &rCba;

        VkDynamicState rDynStates[3] = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH,
        };
        VkPipelineDynamicStateCreateInfo rDyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        rDyn.dynamicStateCount = 3;
        rDyn.pDynamicStates    = rDynStates;

        VkPipelineRenderingCreateInfo rPrCi{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rPrCi.colorAttachmentCount    = 1;
        rPrCi.pColorAttachmentFormats = &sc.imageFormat;
        rPrCi.depthAttachmentFormat   = kDepthFormat;

        VkGraphicsPipelineCreateInfo rGpCi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        rGpCi.pNext               = &rPrCi;
        rGpCi.stageCount          = 2;
        rGpCi.pStages             = rStages;
        rGpCi.pVertexInputState   = &rVi;
        rGpCi.pInputAssemblyState = &rIa;
        rGpCi.pViewportState      = &rVp;
        rGpCi.pRasterizationState = &rRs;
        rGpCi.pMultisampleState   = &rMs;
        rGpCi.pDepthStencilState  = &rDs;
        rGpCi.pColorBlendState    = &rCb;
        rGpCi.pDynamicState       = &rDyn;
        rGpCi.layout              = routePipelineLayout;
        VK_CHECK(vkCreateGraphicsPipelines(gpu.device, VK_NULL_HANDLE, 1, &rGpCi, nullptr, &routePipeline));

        vkDestroyShaderModule(gpu.device, rFragMod, nullptr);
        vkDestroyShaderModule(gpu.device, rVertMod, nullptr);
    }
    std::printf("gpu: route pipeline ready\n");

    // ---- Route loader lambda ----
    // Parses GPX → ENU world coords (DEM-sampled heights, lifted), uploads to
    // a freshly-sized host-mapped VB. Cheap enough to run synchronously on
    // each drop event.
    auto loadRoute = [&](const std::string& path) {
        std::vector<gpx::Waypoint> wps;
        if (!gpx::loadFile(path, wps) || wps.empty()) {
            std::fprintf(stderr, "route: failed to read or parse %s\n", path.c_str());
            return;
        }

        std::vector<glm::vec3> enu;
        enu.reserve(wps.size());
        float lengthM = 0.0f;
        for (size_t i = 0; i < wps.size(); ++i) {
            const auto& wp = wps[i];
            const float east  = static_cast<float>(geo::lonToEast (mesh.frame, wp.lon));
            const float north = static_cast<float>(geo::latToNorth(mesh.frame, wp.lat));
            // Always sample the DEM rather than trusting GPX <ele>: it keeps
            // the rendered line glued to *our* surface even if the GPX has
            // stale or barometric elevations.
            const float demH = sampleDemBilinear(tile, wp.lon, wp.lat);
            const glm::vec3 p(east, north, demH + kRouteLiftMetres);
            enu.push_back(p);
            if (i > 0) lengthM += glm::length(enu[i] - enu[i - 1]);
        }

        // Synchronise before destroying the old VB — any in-flight frame may
        // still be drawing from it.
        VK_CHECK(vkDeviceWaitIdle(gpu.device));
        if (g_route.vb.buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(gpu.allocator, g_route.vb.buffer, g_route.vb.allocation);
            g_route.vb = {};
        }
        const VkDeviceSize sz = sizeof(glm::vec3) * enu.size();
        g_route.vb = createBufferHostMapped(gpu.allocator, sz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        std::memcpy(g_route.vb.mapped, enu.data(), static_cast<size_t>(sz));

        g_route.file        = path;
        g_route.waypoints   = std::move(wps);
        g_route.enu         = std::move(enu);
        g_route.vertexCount = static_cast<uint32_t>(g_route.enu.size());
        g_route.lengthKm    = lengthM * 0.001f;
        g_route.visible     = true;

        // Sun-hours per waypoint comes from the latest readback. We may not
        // have ever baked when --terrain switches and the user immediately
        // drops a GPX, but sunHoursReadback is filled by the initial bake at
        // startup so this is always valid.
        updateRouteSunHoursFromReadback(
            g_route,
            static_cast<const float*>(sunHoursReadback.mapped),
            int(kSunHoursMapSize),
            mesh.aabbMin, mesh.aabbMax);

        std::printf("route: loaded %u waypoints, %.2f km from %s\n",
                    g_route.vertexCount, double(g_route.lengthKm), path.c_str());
    };

    glfwSetDropCallback(window, dropCallback);

    // Auto-load the preset's bundled route, if any. The user can still drop a
    // .gpx to replace it; the visibility checkbox toggles whatever's loaded.
    if (preset.builtInGpxPath != nullptr) {
        loadRoute(preset.builtInGpxPath);
    }

    // ---- ImGui setup ----
    // Generous pool — ImGui allocates one descriptor per font/texture and a
    // few more for its internals; helmet_demo's example uses 1000-of-each
    // and we mirror that to stay future-proof if we ever add image widgets.
    VkDescriptorPool imguiDescPool = VK_NULL_HANDLE;
    {
        VkDescriptorPoolSize imguiPoolSizes[] = {
            {VK_DESCRIPTOR_TYPE_SAMPLER,                1000},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,   1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,   1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,       1000},
        };
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        ci.maxSets       = 1000u * static_cast<uint32_t>(std::size(imguiPoolSizes));
        ci.poolSizeCount = static_cast<uint32_t>(std::size(imguiPoolSizes));
        ci.pPoolSizes    = imguiPoolSizes;
        VK_CHECK(vkCreateDescriptorPool(gpu.device, &ci, nullptr, &imguiDescPool));
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Register our scroll callback BEFORE ImGui hooks GLFW so its
    // `install_callbacks=true` path can chain to ours after handling UI scroll.
    glfwSetScrollCallback(window, scrollCallback);
    ImGui_ImplGlfw_InitForVulkan(window, true);

    VkPipelineRenderingCreateInfoKHR imguiRendering{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    imguiRendering.colorAttachmentCount    = 1;
    imguiRendering.pColorAttachmentFormats = &sc.imageFormat;
    // We draw ImGui inside the same render pass as the terrain (which has
    // depth bound), so its pipeline must declare a matching depth format
    // even though ImGui itself never reads/writes depth.
    imguiRendering.depthAttachmentFormat   = kDepthFormat;

    ImGui_ImplVulkan_InitInfo imguiInit{};
    imguiInit.Instance                    = gpu.instance;
    imguiInit.PhysicalDevice              = gpu.physicalDevice;
    imguiInit.Device                      = gpu.device;
    imguiInit.QueueFamily                 = gpu.graphicsFamily;
    imguiInit.Queue                       = gpu.graphicsQueue;
    imguiInit.DescriptorPool              = imguiDescPool;
    imguiInit.MinImageCount               = static_cast<uint32_t>(sc.images.size());
    imguiInit.ImageCount                  = static_cast<uint32_t>(sc.images.size());
    imguiInit.MSAASamples                 = VK_SAMPLE_COUNT_1_BIT;
    imguiInit.UseDynamicRendering         = true;
    imguiInit.PipelineRenderingCreateInfo = imguiRendering;
    ImGui_ImplVulkan_Init(&imguiInit);
    if (!ImGui_ImplVulkan_CreateFontsTexture()) {
        std::fprintf(stderr, "ImGui font upload failed\n");
        return EXIT_FAILURE;
    }
    std::printf("imgui: ready\n");

    // Initial sun position printout — handy for spot-checking PSA against an
    // almanac without launching the UI. Matterhorn local noon today: expect
    // elevation in the high 50s/low 60s (late May, ~46°N) and azimuth roughly
    // south (~180°, slightly east of meridian if before solar noon).
    {
        const sun::Sample s = sun::compute(
            g_sun.latDeg, g_sun.lonDeg,
            g_sun.year,   g_sun.month, g_sun.day,
            g_sun.localHour, g_sun.tzOffsetH);
        std::printf("sun:  %d-%02d-%02d %.2fh @ (%.3f, %.3f) → az=%.2f° el=%.2f° I=%.0f W/m²\n",
                    g_sun.year, g_sun.month, g_sun.day, g_sun.localHour,
                    g_sun.latDeg, g_sun.lonDeg,
                    s.azimuthDeg, s.elevationDeg, s.irradianceWm2);
    }

    std::printf("alpine-sun: window open. Left-drag to orbit, scroll to zoom, ESC to quit.\n");

    // Dawn-on-snow tint: pale warm pink. Distinctive enough to confirm the
    // clear actually happened (vs. driver-default black).
    constexpr VkClearColorValue kClearColor = {{0.95f, 0.82f, 0.78f, 1.0f}};

    uint32_t frameIndex = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // Drop callback may have queued a GPX file during glfwPollEvents.
        if (!g_pendingGpx.path.empty()) {
            loadRoute(g_pendingGpx.path);
            g_pendingGpx.path.clear();
        }

        // ---- Input → orbit camera ----
        // Skip drag-to-orbit when the cursor is over ImGui — otherwise
        // dragging a slider would also yank the camera. Scroll gating happens
        // inside scrollCallback.
        const bool uiCapturesMouse = ImGui::GetIO().WantCaptureMouse;
        {
            double mx = 0.0, my = 0.0;
            glfwGetCursorPos(window, &mx, &my);
            const bool leftHeld = !uiCapturesMouse
                && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

            if (leftHeld && g_input.leftDown) {
                const double dx = mx - g_input.lastX;
                const double dy = my - g_input.lastY;
                // 0.3° per pixel feels right for this scale.
                constexpr float kSensitivity = 0.005f;
                g_camera.yaw   -= static_cast<float>(dx) * kSensitivity;
                g_camera.pitch += static_cast<float>(dy) * kSensitivity;
                // Clamp pitch just shy of the poles to avoid lookAt singularity.
                constexpr float kPitchLimit = 1.4835f;  // ~85°
                g_camera.pitch = std::clamp(g_camera.pitch, -kPitchLimit, kPitchLimit);
            }
            g_input.leftDown = leftHeld;
            g_input.lastX    = mx;
            g_input.lastY    = my;

            if (g_input.scrollPending != 0.0) {
                // Each scroll tick scales distance by 0.9 (in) or 1/0.9 (out).
                const float zoom = std::pow(0.9f, static_cast<float>(g_input.scrollPending));
                g_camera.distance = std::clamp(g_camera.distance * zoom,
                                               g_camera.minDistance, g_camera.maxDistance);
                g_input.scrollPending = 0.0;
            }

            // Right-click anywhere on the terrain triggers a pick. We only
            // act on the rising edge so holding the button doesn't queue up
            // repeated reads.
            const bool rightHeld = !uiCapturesMouse
                && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
            if (rightHeld && !g_input.rightDown) {
                g_pickRequest.pending = true;
                g_pickRequest.pixelX  = static_cast<int>(mx);
                g_pickRequest.pixelY  = static_cast<int>(my);
            }
            g_input.rightDown = rightHeld;
        }

        forfun::FrameContext& frame = frames[frameIndex];

        // ---- ImGui: build the Sun + Shadows panel for this frame ----
        // Done before the command buffer opens so we can use sunSample +
        // lightViewProj when recording the shadow pass.
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const sun::Sample sunSample = sun::compute(
            g_sun.latDeg, g_sun.lonDeg,
            g_sun.year,   g_sun.month, g_sun.day,
            g_sun.localHour, g_sun.tzOffsetH);

        ImGui::SetNextWindowSize(ImVec2(320.0f, 0.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Sun")) {
            ImGui::TextUnformatted("Location");
            ImGui::DragFloat("Latitude°",  &g_sun.latDeg, 0.01f, -89.0f,  89.0f, "%.4f");
            ImGui::DragFloat("Longitude°", &g_sun.lonDeg, 0.01f, -180.0f, 180.0f, "%.4f");
            ImGui::Separator();
            ImGui::TextUnformatted("Date");
            ImGui::DragInt("Year",  &g_sun.year,  0.25f, 1970, 2099);
            ImGui::DragInt("Month", &g_sun.month, 0.05f, 1, 12);
            ImGui::DragInt("Day",   &g_sun.day,   0.10f, 1, 31);
            ImGui::Separator();
            ImGui::TextUnformatted("Time");
            ImGui::SliderFloat("Local hour", &g_sun.localHour, 0.0f, 24.0f, "%.2f h");
            ImGui::DragFloat ("UTC offset",  &g_sun.tzOffsetH, 0.25f, -12.0f, 14.0f, "%.2f h");
            ImGui::Separator();
            ImGui::Text("Azimuth     %6.2f°", sunSample.azimuthDeg);
            ImGui::Text("Elevation   %6.2f°", sunSample.elevationDeg);
            ImGui::Text("Irradiance  %6.0f W/m²", sunSample.irradianceWm2);
            ImGui::Text("Above horizon: %s", sunSample.elevationDeg > 0.0f ? "yes" : "no");
            ImGui::Separator();
            ImGui::TextUnformatted("Occlusion");
            ImGui::Checkbox("Shadow map",  &g_shadow.shadowMapEnabled);
            ImGui::Checkbox("Horizon map", &g_shadow.horizonMapEnabled);
            ImGui::SliderFloat("Bias const", &g_shadow.depthBiasConstant, 0.0f, 8.0f, "%.2f");
            ImGui::SliderFloat("Bias slope", &g_shadow.depthBiasSlope,    0.0f, 8.0f, "%.2f");
            ImGui::Separator();
            ImGui::TextUnformatted("Sun hours / day");
            ImGui::Checkbox("Show colormap", &g_sunHours.enabled);
            ImGui::SetItemTooltip("Replace terrain shading with a colormap of total\n"
                                  "direct-sun hours for the selected date.");

            // Legend strip: only useful when the visualisation is actually on.
            if (g_sunHours.enabled) {
                ImDrawList* draw = ImGui::GetWindowDrawList();
                const float w     = ImGui::CalcItemWidth();
                const float h     = 14.0f;
                const ImVec2 p0   = ImGui::GetCursorScreenPos();
                const int segs    = 64;
                for (int i = 0; i < segs; ++i) {
                    const float t0 = float(i)     / float(segs);
                    const float t1 = float(i + 1) / float(segs);
                    const glm::vec3 c = viridisCpu(t0);
                    const ImU32 col = IM_COL32(
                        int(c.r * 255.0f), int(c.g * 255.0f), int(c.b * 255.0f), 255);
                    draw->AddRectFilled(
                        ImVec2(p0.x + t0 * w, p0.y),
                        ImVec2(p0.x + t1 * w, p0.y + h),
                        col);
                }
                ImGui::Dummy(ImVec2(w, h));
                // Three labels under the gradient: 0, mid, max.
                char left[16], mid[16], right[16];
                std::snprintf(left,  sizeof(left),  "0 h");
                std::snprintf(mid,   sizeof(mid),   "%.1f h", g_sunHours.maxHoursScale * 0.5f);
                std::snprintf(right, sizeof(right), "%.1f h", g_sunHours.maxHoursScale);
                const float midX   = ImGui::CalcTextSize(mid).x;
                const float rightX = ImGui::CalcTextSize(right).x;
                ImGui::TextUnformatted(left);
                ImGui::SameLine(w * 0.5f - midX * 0.5f);
                ImGui::TextUnformatted(mid);
                ImGui::SameLine(w - rightX);
                ImGui::TextUnformatted(right);
            }

            ImGui::SliderFloat("Sample step",  &g_sunHours.stepMinutes,   5.0f, 60.0f, "%.0f min");
            ImGui::SetItemTooltip("How often per day to sample the sun position.\n"
                                  "Smaller = finer time resolution, slightly slower to recompute.");
            ImGui::SliderFloat("Top of scale", &g_sunHours.maxHoursScale, 4.0f, 16.0f, "%.1f h");
            ImGui::SetItemTooltip("Hours-per-day that maps to the brightest yellow.\n"
                                  "Lower = more contrast at shaded points; higher = covers\n"
                                  "longer summer days without clipping.");
            ImGui::Separator();
            ImGui::TextUnformatted("Display");
            ImGui::SliderFloat("Exposure", &g_tone.exposure, 0.25f, 4.0f, "%.2f×");
            ImGui::SetItemTooltip("Linear multiplier before the ACES tonemap.\n"
                                  "Higher = brighter midtones, faster roll-off into white.\n"
                                  "Does not affect the sun-hours colormap.");
            ImGui::Separator();
            ImGui::TextUnformatted("Picked point");
            ImGui::TextDisabled("(right-click on terrain to sample)");
            if (g_pickResult.valid) {
                ImGui::Text("Lat  %9.5f°", g_pickResult.latDeg);
                ImGui::Text("Lon  %9.5f°", g_pickResult.lonDeg);
                ImGui::Text("Elev %5.0f m",  g_pickResult.elevation);
                ImGui::Text("Sun  %5.2f h/day", g_pickResult.sunHours);
            } else {
                ImGui::TextDisabled("(no sample yet)");
            }

            ImGui::Separator();
            ImGui::TextUnformatted("Route");
            if (g_route.vertexCount > 0) {
                ImGui::Text("%u waypoints — %.2f km", g_route.vertexCount, double(g_route.lengthKm));
                ImGui::Checkbox("Show on terrain", &g_route.visible);
                // Compact table — scrolls if longer than the visible area.
                if (ImGui::CollapsingHeader("Waypoints + sun")) {
                    if (ImGui::BeginTable("waypoints", 5,
                            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg
                          | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit,
                            ImVec2(0.0f, 240.0f))) {
                        ImGui::TableSetupScrollFreeze(0, 1);
                        ImGui::TableSetupColumn("#",        ImGuiTableColumnFlags_WidthFixed, 30.0f);
                        ImGui::TableSetupColumn("Lat",      ImGuiTableColumnFlags_WidthFixed, 65.0f);
                        ImGui::TableSetupColumn("Lon",      ImGuiTableColumnFlags_WidthFixed, 65.0f);
                        ImGui::TableSetupColumn("Elev (m)", ImGuiTableColumnFlags_WidthFixed, 55.0f);
                        ImGui::TableSetupColumn("Sun h",    ImGuiTableColumnFlags_WidthFixed, 50.0f);
                        ImGui::TableHeadersRow();
                        for (uint32_t i = 0; i < g_route.vertexCount; ++i) {
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0); ImGui::Text("%u", i);
                            ImGui::TableSetColumnIndex(1); ImGui::Text("%7.4f", g_route.waypoints[i].lat);
                            ImGui::TableSetColumnIndex(2); ImGui::Text("%7.4f", g_route.waypoints[i].lon);
                            ImGui::TableSetColumnIndex(3); ImGui::Text("%4.0f", g_route.enu[i].z - kRouteLiftMetres);
                            ImGui::TableSetColumnIndex(4);
                            if (i < g_route.sunHours.size()) {
                                ImGui::Text("%4.1f", double(g_route.sunHours[i]));
                            } else {
                                ImGui::TextUnformatted("--");
                            }
                        }
                        ImGui::EndTable();
                    }
                }
            } else {
                ImGui::TextDisabled("Drop a .gpx file on the window to load.");
            }
        }
        ImGui::End();

        ImGui::Render();

        // ---- Re-bake sun-hours if any dependent parameter changed ----
        // Cheap (~few ms) so we just dispatch synchronously whenever the user
        // slides a date/location/step control.
        if (lastBakedParams.latDeg    != g_sun.latDeg
         || lastBakedParams.lonDeg    != g_sun.lonDeg
         || lastBakedParams.year      != g_sun.year
         || lastBakedParams.month     != g_sun.month
         || lastBakedParams.day       != g_sun.day
         || lastBakedParams.tzOffsetH != g_sun.tzOffsetH
         || lastBakedParams.stepMin   != g_sunHours.stepMinutes) {
            runSunHoursBake();
        }

        // ---- Camera matrices from orbit state ----
        const glm::vec3 eye    = orbitEye(g_camera);
        const glm::vec3 center = g_camera.target;
        const glm::vec3 upVec  = glm::vec3(0.0f, 0.0f, 1.0f);

        const float aspect = static_cast<float>(sc.extent.width)
                           / static_cast<float>(sc.extent.height);

        const bool renderShadows = g_shadow.shadowMapEnabled && sunSample.directionToSun.z > 0.0f;

        CameraUBO cam{};
        cam.view = glm::lookAt(eye, center, upVec);
        cam.proj = glm::perspective(glm::radians(60.0f), aspect, 100.0f, 400000.0f);
        cam.proj[1][1] *= -1.0f;             // Vulkan NDC y-flip
        cam.viewProj    = cam.proj * cam.view;
        cam.invViewProj = glm::inverse(cam.viewProj);
        cam.lightViewProj = renderShadows
            ? computeLightViewProj(sunSample.directionToSun, mesh.aabbMin, mesh.aabbMax)
            : glm::mat4(1.0f);
        cam.sunDirAndIrradiance = glm::vec4(sunSample.directionToSun,
                                            sunSample.irradianceWm2);
        // .x = shadow map enabled, .y = horizon map enabled (horizon is
        // independent of sun position so it's left on even when the sun is
        // below the horizon — the frag handles night separately).
        cam.occlusionParams = glm::vec4(renderShadows                ? 1.0f : 0.0f,
                                        g_shadow.horizonMapEnabled   ? 1.0f : 0.0f,
                                        0.0f, 0.0f);
        cam.sunHoursParams = glm::vec4(g_sunHours.enabled ? 1.0f : 0.0f,
                                       g_sunHours.maxHoursScale,
                                       0.0f, 0.0f);
        cam.toneParams = glm::vec4(g_tone.exposure, 0.0f, 0.0f, 0.0f);
        cam.terrainAabb = glm::vec4(mesh.aabbMin.x, mesh.aabbMin.y,
                                    mesh.aabbMax.x, mesh.aabbMax.y);
        std::memcpy(cameraUbos[frameIndex].mapped, &cam, sizeof(cam));

        // 1. Wait for the previous use of this frame slot to finish.
        VK_CHECK(vkWaitForFences(gpu.device, 1, &frame.inFlight, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences  (gpu.device, 1, &frame.inFlight));

        // 2. Acquire a swapchain image; the imageAvailable semaphore signals
        //    once the presentation engine releases it.
        uint32_t imageIndex = 0;
        VK_CHECK(vkAcquireNextImageKHR(gpu.device, sc.swapchain, UINT64_MAX,
                                       frame.imageAvailable, VK_NULL_HANDLE, &imageIndex));

        // 3. Record commands: shadow pass (if enabled), then color+depth main
        //    pass with terrain + ImGui, transition swapchain to present.
        VK_CHECK(vkResetCommandBuffer(frame.cmd, 0));
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(frame.cmd, &beginInfo));

        // ---- Shadow pass ----
        // Skip when shadows are disabled OR the sun is below the horizon
        // (nothing useful to occlude). The shadow map keeps its stale
        // contents in SHADER_READ_ONLY layout; the frag's shadowParams.x
        // check forces visibility=1.0 in that case.
        if (renderShadows) {
            transitionImage(frame.cmd, shadowMap.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                                | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            VkRenderingAttachmentInfo shadowAttach{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            shadowAttach.imageView                  = shadowMap.view;
            shadowAttach.imageLayout                = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            shadowAttach.loadOp                     = VK_ATTACHMENT_LOAD_OP_CLEAR;
            shadowAttach.storeOp                    = VK_ATTACHMENT_STORE_OP_STORE;
            shadowAttach.clearValue.depthStencil    = {1.0f, 0};

            VkRenderingInfo shadowRender{VK_STRUCTURE_TYPE_RENDERING_INFO};
            shadowRender.renderArea       = {{0, 0}, {kShadowMapSize, kShadowMapSize}};
            shadowRender.layerCount       = 1;
            shadowRender.pDepthAttachment = &shadowAttach;

            vkCmdBeginRendering(frame.cmd, &shadowRender);

            VkViewport shadowViewport{};
            shadowViewport.width    = static_cast<float>(kShadowMapSize);
            shadowViewport.height   = static_cast<float>(kShadowMapSize);
            shadowViewport.minDepth = 0.0f;
            shadowViewport.maxDepth = 1.0f;
            vkCmdSetViewport(frame.cmd, 0, 1, &shadowViewport);
            VkRect2D shadowScissor{{0, 0}, {kShadowMapSize, kShadowMapSize}};
            vkCmdSetScissor(frame.cmd, 0, 1, &shadowScissor);
            vkCmdSetDepthBias(frame.cmd, g_shadow.depthBiasConstant, 0.0f, g_shadow.depthBiasSlope);

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
            VkDeviceSize sVbOffset = 0;
            vkCmdBindVertexBuffers(frame.cmd, 0, 1, &terrainVB.buffer, &sVbOffset);
            vkCmdBindIndexBuffer  (frame.cmd, terrainIB.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdPushConstants(frame.cmd, shadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(glm::mat4), &cam.lightViewProj);
            vkCmdDrawIndexed(frame.cmd, static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);

            vkCmdEndRendering(frame.cmd);

            transitionImage(frame.cmd, shadowMap.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }

        transitionImage(frame.cmd, sc.images[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        colorAttachment.imageView          = sc.imageViews[imageIndex];
        colorAttachment.imageLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp             = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp            = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color   = kClearColor;

        VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depthAttachment.imageView                  = depth.view;
        depthAttachment.imageLayout                = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp                     = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp                    = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue.depthStencil    = {1.0f, 0};

        VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
        renderingInfo.renderArea           = {{0, 0}, sc.extent};
        renderingInfo.layerCount           = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments    = &colorAttachment;
        renderingInfo.pDepthAttachment     = &depthAttachment;

        vkCmdBeginRendering(frame.cmd, &renderingInfo);

        VkViewport viewport{};
        viewport.x        = 0.0f;
        viewport.y        = 0.0f;
        viewport.width    = static_cast<float>(sc.extent.width);
        viewport.height   = static_cast<float>(sc.extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(frame.cmd, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, sc.extent};
        vkCmdSetScissor(frame.cmd, 0, 1, &scissor);

        // ---- Sky pass ----
        // Fullscreen triangle. No depth write, no vertex inputs. Filled first
        // so the terrain (which writes depth normally) can overdraw it. The
        // existing colour-attachment clear is now functionally dead but cheap.
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline);
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                skyPipelineLayout, 0, 1, &cameraDescSets[frameIndex], 0, nullptr);
        vkCmdDraw(frame.cmd, 3, 1, 0, 0);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, terrainPipeline);
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &cameraDescSets[frameIndex], 0, nullptr);

        VkDeviceSize vbOffset = 0;
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &terrainVB.buffer, &vbOffset);
        vkCmdBindIndexBuffer  (frame.cmd, terrainIB.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed      (frame.cmd, static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);

        // ---- Route line strip ----
        // Drawn after terrain so cast-shadow / occlusion-by-terrain reads
        // correctly via the shared depth buffer. Skipped when no route loaded
        // or the user toggled it off.
        if (g_route.vertexCount > 1 && g_route.visible) {
            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, routePipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    routePipelineLayout, 0, 1,
                                    &cameraDescSets[frameIndex], 0, nullptr);
            // wideLines feature isn't enabled in forfun's device — width must
            // stay at 1.0 or validation complains. Magenta against terrain is
            // visible even at one pixel.
            vkCmdSetLineWidth(frame.cmd, 1.0f);
            VkDeviceSize rOffset = 0;
            vkCmdBindVertexBuffers(frame.cmd, 0, 1, &g_route.vb.buffer, &rOffset);
            vkCmdDraw(frame.cmd, g_route.vertexCount, 1, 0, 0);
        }

        // ImGui draws into the same swapchain attachment; we let it run
        // without depth so panels stay on top regardless of the terrain.
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), frame.cmd);

        vkCmdEndRendering(frame.cmd);

        // ---- Pick: copy depth at cursor into a host-readable buffer ----
        // Recorded into the SAME cmd buffer as the frame that produced the
        // depth values we want. We re-transition back to DEPTH_ATTACHMENT_OPTIMAL
        // so the persistent layout invariant is preserved (next frame's clear
        // assumes it).
        const bool pickThisFrame = g_pickRequest.pending
            && g_pickRequest.pixelX >= 0
            && g_pickRequest.pixelY >= 0
            && static_cast<uint32_t>(g_pickRequest.pixelX) < sc.extent.width
            && static_cast<uint32_t>(g_pickRequest.pixelY) < sc.extent.height;
        if (pickThisFrame) {
            transitionImage(frame.cmd, depth.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_COPY_BIT,
                            VK_ACCESS_2_TRANSFER_READ_BIT);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {g_pickRequest.pixelX, g_pickRequest.pixelY, 0};
            region.imageExtent = {1, 1, 1};
            vkCmdCopyImageToBuffer(frame.cmd, depth.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   pickDepthBuf.buffer, 1, &region);

            transitionImage(frame.cmd, depth.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COPY_BIT,
                            VK_ACCESS_2_TRANSFER_READ_BIT,
                            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        }

        transitionImage(frame.cmd, sc.images[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);

        VK_CHECK(vkEndCommandBuffer(frame.cmd));

        // 4. Submit: wait on imageAvailable, signal renderFinished + fence.
        VkCommandBufferSubmitInfo cbSubmit{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cbSubmit.commandBuffer = frame.cmd;

        VkSemaphoreSubmitInfo waitSem{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        waitSem.semaphore = frame.imageAvailable;
        waitSem.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSemaphoreSubmitInfo signalSem{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signalSem.semaphore = frame.renderFinished;
        signalSem.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

        VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submitInfo.waitSemaphoreInfoCount   = 1;
        submitInfo.pWaitSemaphoreInfos      = &waitSem;
        submitInfo.signalSemaphoreInfoCount = 1;
        submitInfo.pSignalSemaphoreInfos    = &signalSem;
        submitInfo.commandBufferInfoCount   = 1;
        submitInfo.pCommandBufferInfos      = &cbSubmit;

        VK_CHECK(vkQueueSubmit2(gpu.graphicsQueue, 1, &submitInfo, frame.inFlight));

        // 5. Present.
        VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores    = &frame.renderFinished;
        presentInfo.swapchainCount     = 1;
        presentInfo.pSwapchains        = &sc.swapchain;
        presentInfo.pImageIndices      = &imageIndex;
        VK_CHECK(vkQueuePresentKHR(gpu.graphicsQueue, &presentInfo));

        // ---- Pick: complete the readback now that the depth copy has been
        // submitted with the rest of the frame. A queue wait is overkill but
        // simple, and a click is a one-shot event so a brief stall is fine.
        if (pickThisFrame) {
            VK_CHECK(vkQueueWaitIdle(gpu.graphicsQueue));
            float depthValue = 0.0f;
            std::memcpy(&depthValue, pickDepthBuf.mapped, sizeof(float));

            if (depthValue >= 0.999f) {
                // Far plane = clicked on sky; nothing to report.
                g_pickRequest.pending = false;
            } else {
                // Unproject NDC → world. Vulkan: framebuffer Y goes down, NDC Y
                // also goes down post-viewport, so the y-flip is already baked
                // into the proj matrix we used to render.
                const float ndcX = (2.0f * (g_pickRequest.pixelX + 0.5f) / sc.extent.width)  - 1.0f;
                const float ndcY = (2.0f * (g_pickRequest.pixelY + 0.5f) / sc.extent.height) - 1.0f;
                const glm::vec4 clip(ndcX, ndcY, depthValue, 1.0f);
                const glm::vec4 worldH = cam.invViewProj * clip;
                const glm::vec3 worldPos = glm::vec3(worldH) / worldH.w;

                // Map ENU back to lat/lon via the tile's frame.
                const double lon = mesh.frame.centerLon
                                 + double(worldPos.x) / mesh.frame.metresPerDegreeLon;
                const double lat = mesh.frame.centerLat
                                 + double(worldPos.y) / mesh.frame.metresPerDegreeLat;

                // Compute the sun-hours texel for this point and read it back.
                const glm::vec2 extent(mesh.aabbMax.x - mesh.aabbMin.x,
                                       mesh.aabbMax.y - mesh.aabbMin.y);
                const float uvU = (worldPos.x - mesh.aabbMin.x) / extent.x;
                const float uvV = (mesh.aabbMax.y - worldPos.y) / extent.y;  // N→S
                const int texU = std::clamp(int(uvU * float(kSunHoursMapSize)),
                                            0, int(kSunHoursMapSize) - 1);
                const int texV = std::clamp(int(uvV * float(kSunHoursMapSize)),
                                            0, int(kSunHoursMapSize) - 1);

                // Tiny one-shot submit to copy that single texel.
                VkCommandBufferAllocateInfo cbAi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                cbAi.commandPool        = sunHoursPool;
                cbAi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                cbAi.commandBufferCount = 1;
                VkCommandBuffer cmd = VK_NULL_HANDLE;
                VK_CHECK(vkAllocateCommandBuffers(gpu.device, &cbAi, &cmd));

                VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

                transitionImage(cmd, sunHoursImage, VK_IMAGE_ASPECT_COLOR_BIT,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                VK_PIPELINE_STAGE_2_COPY_BIT,
                                VK_ACCESS_2_TRANSFER_READ_BIT);

                VkBufferImageCopy hRegion{};
                hRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                hRegion.imageSubresource.layerCount = 1;
                hRegion.imageOffset = {texU, texV, 0};
                hRegion.imageExtent = {1, 1, 1};
                vkCmdCopyImageToBuffer(cmd, sunHoursImage,
                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       pickHoursBuf.buffer, 1, &hRegion);

                transitionImage(cmd, sunHoursImage, VK_IMAGE_ASPECT_COLOR_BIT,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_PIPELINE_STAGE_2_COPY_BIT,
                                VK_ACCESS_2_TRANSFER_READ_BIT,
                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

                VK_CHECK(vkEndCommandBuffer(cmd));
                VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                sub.commandBufferCount = 1;
                sub.pCommandBuffers    = &cmd;
                VK_CHECK(vkQueueSubmit(gpu.graphicsQueue, 1, &sub, VK_NULL_HANDLE));
                VK_CHECK(vkQueueWaitIdle(gpu.graphicsQueue));
                vkFreeCommandBuffers(gpu.device, sunHoursPool, 1, &cmd);

                float hoursValue = 0.0f;
                std::memcpy(&hoursValue, pickHoursBuf.mapped, sizeof(float));

                g_pickResult.valid     = true;
                g_pickResult.worldPos  = worldPos;
                g_pickResult.latDeg    = lat;
                g_pickResult.lonDeg    = lon;
                g_pickResult.elevation = worldPos.z;
                g_pickResult.sunHours  = hoursValue;
                g_pickRequest.pending  = false;
            }
        }

        frameIndex = (frameIndex + 1) % kFramesInFlight;
    }

    // Make sure the GPU is idle before tearing down resources still in use.
    VK_CHECK(vkDeviceWaitIdle(gpu.device));

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(gpu.device, imguiDescPool, nullptr);

    if (g_route.vb.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(gpu.allocator, g_route.vb.buffer, g_route.vb.allocation);
    }
    vkDestroyPipeline(gpu.device, routePipeline, nullptr);
    vkDestroyPipelineLayout(gpu.device, routePipelineLayout, nullptr);
    vkDestroyPipeline(gpu.device, skyPipeline, nullptr);
    vkDestroyPipelineLayout(gpu.device, skyPipelineLayout, nullptr);
    vkDestroyPipeline(gpu.device, shadowPipeline, nullptr);
    vkDestroyPipelineLayout(gpu.device, shadowPipelineLayout, nullptr);
    vkDestroyPipeline(gpu.device, terrainPipeline, nullptr);
    vkDestroyPipelineLayout(gpu.device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(gpu.device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(gpu.device, cameraSetLayout, nullptr);
    for (auto& b : cameraUbos) vmaDestroyBuffer(gpu.allocator, b.buffer, b.allocation);
    vmaDestroyBuffer(gpu.allocator, terrainIB.buffer, terrainIB.allocation);
    vmaDestroyBuffer(gpu.allocator, terrainVB.buffer, terrainVB.allocation);
    vkDestroyCommandPool(gpu.device, sunHoursPool, nullptr);
    vkDestroyPipeline(gpu.device, sunHoursPipeline, nullptr);
    vkDestroyPipelineLayout(gpu.device, sunHoursPipelineLayout, nullptr);
    vkDestroyDescriptorPool(gpu.device, sunHoursDescPool, nullptr);
    vkDestroyDescriptorSetLayout(gpu.device, sunHoursSetLayout, nullptr);
    vmaDestroyBuffer(gpu.allocator, sunHoursReadback.buffer, sunHoursReadback.allocation);
    vmaDestroyBuffer(gpu.allocator, pickHoursBuf.buffer, pickHoursBuf.allocation);
    vmaDestroyBuffer(gpu.allocator, pickDepthBuf.buffer, pickDepthBuf.allocation);
    vmaDestroyBuffer(gpu.allocator, sunSamplesBuf.buffer, sunSamplesBuf.allocation);
    vkDestroySampler(gpu.device, sunHoursSampler, nullptr);
    vkDestroyImageView(gpu.device, sunHoursStorageView, nullptr);
    vkDestroyImageView(gpu.device, sunHoursSampleView,  nullptr);
    vmaDestroyImage(gpu.allocator, sunHoursImage, sunHoursAlloc);
    vkDestroySampler(gpu.device, horizonSampler, nullptr);
    vkDestroyImageView(gpu.device, horizonStorageView, nullptr);
    vkDestroyImageView(gpu.device, horizonSampleView,  nullptr);
    vmaDestroyImage(gpu.allocator, horizonImage, horizonAlloc);
    vkDestroySampler(gpu.device, heightMapSampler, nullptr);
    vkDestroyImageView(gpu.device, heightMapView, nullptr);
    vmaDestroyImage(gpu.allocator, heightMapImage, heightMapAlloc);
    vkDestroySampler(gpu.device, shadowSampler, nullptr);
    vkDestroyImageView(gpu.device, shadowMap.view, nullptr);
    vmaDestroyImage(gpu.allocator, shadowMap.image, shadowMap.allocation);
    vkDestroyImageView(gpu.device, depth.view, nullptr);
    vmaDestroyImage(gpu.allocator, depth.image, depth.allocation);
    for (auto& f : frames) forfun::destroyFrameContext(gpu, f);
    forfun::destroySwapchain(gpu, sc);
    forfun::destroyDevice(gpu);
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
