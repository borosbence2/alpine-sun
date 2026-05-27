#pragma once

// Aggregates the UI-state structs and their process-wide globals. main.cpp
// reads/writes these directly each frame; the GLFW scroll + drop callbacks
// also poke a couple of them, which is the original reason they're globals.

#include "gpx.h"
#include "vk_helpers.h"   // Buffer

#include <glm/glm.hpp>

#include <string>
#include <vector>

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

struct SunHoursUi {
    bool  enabled       = false;
    float stepMinutes   = 10.0f;   // 144 samples/day at the default
    float maxHoursScale = 14.0f;   // normalises the viridis colormap
};

struct ToneUi {
    float exposure = 1.0f;
};

struct SatUi {
    bool enabled = false;   // off by default — procedural shading is the demo look
};

// Avalanche terrain overlay. Heuristic colour map of slope-angle brackets
// (industry standard ≈ 27–50° is "avalanche terrain"), optionally amplified
// for south-facing high-sun-hours slopes (wet-slide bias). Off by default;
// has a visible disclaimer in the UI that it is NOT a hazard forecast.
struct AvalancheUi {
    bool enabled       = false;
    bool solarLoading  = true;   // checked by default once the overlay is on
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

// Hover sample — recomputed every frame via CPU ray-march against the DEM.
// Cheap (~microseconds) so we don't need to gate it behind a click.
struct HoverResult {
    bool      valid     = false;
    glm::vec3 worldPos  = glm::vec3(0.0f);
    double    latDeg    = 0.0;
    double    lonDeg    = 0.0;
    float     elevation = 0.0f;
    float     sunHours  = 0.0f;
    double    cursorX   = 0.0;   // window pixels — used to place the tooltip
    double    cursorY   = 0.0;
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
    bool   middleDown    = false;
    bool   rightDown     = false;
    double lastX         = 0.0;
    double lastY         = 0.0;
    double scrollPending = 0.0;   // GLFW scroll deltas accumulate here; consumed each frame
};

// Globals so the GLFW scroll + drop callbacks can reach them. We only ever
// have one window in this process, so a singleton is fine.
extern OrbitCamera    g_camera;
extern InputState     g_input;
extern SunUi          g_sun;
extern ShadowUi       g_shadow;
extern SunHoursUi     g_sunHours;
extern ToneUi         g_tone;
extern AvalancheUi    g_avalanche;
extern PickRequest    g_pickRequest;
extern PickResult     g_pickResult;
extern HoverResult    g_hover;
extern PendingGpxLoad g_pendingGpx;
extern Route          g_route;
extern SatUi          g_sat;
