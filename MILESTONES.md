# alpine-sun — milestone plan

A Vulkan-based tool for alpine trip planning: load real DEM terrain, place the
sun via NREL SPA, and visualise sun/shadow on a face (or sun-hours-per-day
across a region). Built on top of [`forfun-graphics`](../forfun-graphics) used
as a library (`forfun_core`).

## Decisions baked in

- **Library API model**: app-owns-loop. `forfun_core` exposes building blocks
  (`Device`, `Swapchain`, `FrameContext`, `Renderer`, `Material`, `IBL`,
  `Camera`); `alpine-sun` writes its own main loop. Chosen for flexibility
  around heavy ImGui, time-scrubbing, and compute-only precompute passes.
- **Phase 1 test region**: Matterhorn / Cervino. Copernicus GLO-30 baseline,
  swissALTI3D (0.5 m) reserved as a later upgrade.

## Phase 0 — Library extraction (`forfun-graphics` → `forfun_core` + demo)

Split into two passes during execution. **0A** (done) is a structural CMake
split with no extraction work — `forfun_core` gets all current `src/*.cpp`
except `main.cpp`, and `main.cpp` stays in `helmet_demo`. **0B** (on-demand)
extracts `Engine`/`Device`/`Swapchain`/`FrameContext` from `helmet_demo`'s
1400-line `main()` into the library, one piece at a time, driven by what
alpine-sun's terrain MVP actually needs.

### Phase 0A — Structural CMake split (DONE)

| ID         | Description                                                                          | Status |
|------------|--------------------------------------------------------------------------------------|--------|
| A0.lib.1   | Split target: `forfun_core` STATIC lib + `helmet_demo` exe                           | DONE   |
| A0.lib.2   | API model: app-owns-loop (decided)                                                   | DONE   |
| A0.lib.6   | Moved `add_shader` into `cmake/ForfunShaders.cmake`                                  | DONE   |
| A0.lib.7a  | alpine-sun skeleton: window opens, links `forfun_core` (no render yet)               | DONE   |

### Phase 0B — Incremental engine extraction (on-demand during Phase 1)

| ID         | Description                                                                          | Status |
|------------|--------------------------------------------------------------------------------------|--------|
| A0.lib.3   | Move public headers under `include/forfun/` for clearer namespacing                  | TODO   |
| A0.lib.4   | Hide internals: VMA out of public types (PIMPL or detail/), vk-bootstrap stays demo-side | TODO |
| A0.lib.5   | CMake install + package config so external apps can use `find_package(forfun_core)`  | TODO   |
| A0.lib.7b.1 | **Device**: instance + surface + physical + logical + queues + VMA (callback-based surface creation) | DONE |
| A0.lib.7b.2 | **Swapchain**: VkSwapchainKHR + images + image views + extent + format (no resize yet) | DONE |
| A0.lib.7b.3 | **FrameContext**: per-frame command pool + cmd buffer + sync (image-available, render-finished, in-flight fence). Demos manage their own ring/array. | DONE |
| A0.lib.7c  | alpine-sun renders the terrain mesh via the extracted engine primitives              | TODO   |

**Phase 0B is functionally complete.** alpine-sun now has Device + Swapchain +
FrameContext via `forfun_core`. A1.terrain.5–7 (terrain shader, camera, render
loop) are unblocked.

## Phase 1 — Terrain MVP (Matterhorn)

| ID            | Description                                                                | Status |
|---------------|----------------------------------------------------------------------------|--------|
| A1.terrain.1  | GeoTIFF loader (libtiff + zlib via CPM; tiled+scanline; uint32_t counts)   | DONE   |
| A1.terrain.2  | Auto-download Copernicus GLO-30 Matterhorn tile (~30 MB from AWS Open Data) | DONE  |
| A1.terrain.3  | Local ENU frame (centred on tile midpoint; metres-per-degree precomputed)  | DONE   |
| A1.terrain.4  | Heightmap → triangle mesh (configurable stride; central-difference normals) | DONE  |
| A1.terrain.5  | Terrain `.mat` (slope/height debug shading; reuses existing PBR plumbing)  | TODO   |
| A1.terrain.6  | Orbit camera (target = summit) + fly mode toggle                           | TODO   |
| A1.terrain.7  | Render Matterhorn, tone-mapped, with sky                                   | TODO   |

A1.terrain.5–7 are blocked on Phase 0B (extracting Engine / Device / Swapchain
/ FrameContext from `helmet_demo`'s `main.cpp` so alpine-sun can drive a
render loop).

## Phase 2 — Sun positioning

| ID         | Description                                                                  | Status |
|------------|------------------------------------------------------------------------------|--------|
| A2.sun.1   | NREL SPA implementation (vendored C, ~600 LOC, public domain)                | TODO   |
| A2.sun.2   | `SunDriver(lat, lon, date, time, tz) → vec3 direction + irradiance`          | TODO   |
| A2.sun.3   | Wire `SunDriver` to the engine's existing directional light                  | TODO   |
| A2.sun.4   | ImGui panel: date picker, time slider, location entry                        | TODO   |
| A2.sun.5   | Verify against published sunrise/sunset for Zermatt on a few dates           | TODO   |

## Phase 3 — Sun occlusion (the core technical work)

| ID         | Description                                                                  | Status |
|------------|------------------------------------------------------------------------------|--------|
| A3.occ.1   | Sun-POV shadow map (reuse existing shadow infra; large extent, ortho)        | TODO   |
| A3.occ.2   | Horizon map data model — RG16F or R32 storage image, N azimuth bins          | TODO   |
| A3.occ.3   | Horizon map compute shader (per-texel scan, configurable radius)             | TODO   |
| A3.occ.4   | Terrain frag shader: O(1) horizon sample → `sunVisibility` factor            | TODO   |
| A3.occ.5   | A/B toggle in ImGui: horizon-map vs shadow-map vs both                       | TODO   |
| A3.occ.6   | **Optional** `VK_KHR_ray_tracing_pipeline` + BVH for sharp near-terrain shadows | TODO |

## Phase 4 — Sun-hours visualisation

| ID          | Description                                                                 | Status |
|-------------|-----------------------------------------------------------------------------|--------|
| A4.heat.1   | Precompute: N timesteps × sun visibility → accumulator texture              | TODO   |
| A4.heat.2   | Color ramp (viridis-style) → decal on terrain                               | TODO   |
| A4.heat.3   | Date/month picker → re-run accumulation                                     | TODO   |
| A4.heat.4   | Face-selection mode: paint a region, get exact sun-hours/day for it         | TODO   |

## Phase 5 — Trip planning features

| ID         | Description                                                                  | Status |
|------------|------------------------------------------------------------------------------|--------|
| A5.use.1   | GPX import → polyline rendered on terrain                                    | TODO   |
| A5.use.2   | Per-waypoint sun query → "this pitch in shade until 10:30" table             | TODO   |
| A5.use.3   | Satellite imagery overlay (Sentinel-2 or Mapbox raster tiles)                | TODO   |
| A5.use.4   | Atmospheric scattering for alpenglow accuracy (Hillaire sky model)           | TODO   |
| A5.use.5   | Save/load trip plans (JSON: location + route + date)                         | TODO   |

## Phase 6 — Deferred

| ID         | Description                                                                  | Status |
|------------|------------------------------------------------------------------------------|--------|
| A6.poly.1  | Quadtree / clipmap LOD for streaming large regions                           | TODO   |
| A6.poly.2  | Snow line + glacier mask overlay (OSM / RGI data)                            | TODO   |
| A6.poly.3  | Cloud shadows from forecast API                                              | TODO   |
| A6.poly.4  | Photogrammetry import (drone scans) for sub-meter wall detail                | TODO   |

## Risks

- **Library extraction (Phase 0) is the riskiest single step.** If `main.cpp`
  in `forfun-graphics` turns out to be tightly coupled to render
  orchestration, A0.lib.3-4 could balloon. Read through before committing.
- **Avoid GDAL** unless full reprojection is needed. libtiff + a small
  projection routine usually suffices.
- **Ray tracing (A3.occ.6) is optional.** Horizon maps deliver ~90% of the
  visual quality at a fraction of the complexity. Treat RT as a stretch goal.
