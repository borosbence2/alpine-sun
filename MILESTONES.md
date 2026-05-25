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

CPU pipeline first (loader → frame → mesh), then GPU plumbing (render loop →
mesh upload → depth → camera), then actual terrain rendering (shader → camera
controls → final scene).

### Phase 1a — CPU pipeline (DONE)

| ID            | Description                                                                | Status |
|---------------|----------------------------------------------------------------------------|--------|
| A1.terrain.1  | GeoTIFF loader (libtiff + zlib via CPM; tiled+scanline; uint32_t counts)   | DONE   |
| A1.terrain.2  | Auto-download Copernicus GLO-30 Matterhorn tile (~30 MB from AWS Open Data) | DONE  |
| A1.terrain.3  | Local ENU frame (centred on tile midpoint; metres-per-degree precomputed)  | DONE   |
| A1.terrain.4  | Heightmap → triangle mesh (configurable stride; central-difference normals) | DONE  |

### Phase 1b — GPU plumbing (in progress)

| ID            | Description                                                                | Status |
|---------------|----------------------------------------------------------------------------|--------|
| A1.render.0   | Render-loop skeleton: acquire → record → submit → present, clear-color only | DONE  |
| A1.render.1   | Depth attachment (transitioned once to DEPTH_OPTIMAL; CLEAR loadOp per frame) | DONE |
| A1.render.2   | Upload terrain VB (6.2 MB) + IB (4.6 MB) via staging (uploadToBuffer)      | DONE   |
| A1.render.3   | Camera UBO (view + proj + viewProj) + descriptor set per frame             | DONE   |

### Phase 1c — Terrain rendering

| ID            | Description                                                                | Status |
|---------------|----------------------------------------------------------------------------|--------|
| A1.terrain.5  | Hand-written GLSL terrain shader (slope/height debug + fixed-sun Lambert) + graphics pipeline + draw call | DONE |
| A1.terrain.6  | Orbit camera: left-drag yaw/pitch, scroll-zoom; default framed on Matterhorn. Fly mode deferred. | DONE |
| A1.terrain.7  | ACES tonemap (Narkowicz fit) in `terrain.frag` + `sky.frag`; sky.frag is a fullscreen-triangle pass that derives view-dirs via `invViewProj`, produces a sun-aware day/night gradient with warm horizon glow when the sun is low. Exposure slider in ImGui. Sun-hours colormap bypasses the tonemap so the legend stays faithful. | DONE |

**Phase 1 functionally complete** — you can load the Matterhorn DEM, mesh it,
render it, and orbit around. A1.terrain.7 is polish; Phase 2 (sun
positioning via NREL SPA) is the next substantive milestone.

## Phase 2 — Sun positioning

| ID         | Description                                                                  | Status |
|------------|------------------------------------------------------------------------------|--------|
| A2.sun.1   | PSA (Blanco-Muriel 2001) vendored at src/psa.{h,cpp} — ~80 LOC, public-domain. Chosen over NREL SPA to avoid the NREL licence; accurate to ~0.01° in 2026, easily upgradable. | DONE |
| A2.sun.2   | `sun::compute(lat, lon, date, local_time, tz) → {directionToSun (ENU), elevation°, azimuth°, irradiance W/m²}` in src/sun_driver.{h,cpp}. | DONE |
| A2.sun.3   | Terrain frag shader reads `sunDirAndIrradiance` from the camera UBO; below-horizon falls off via smoothstep. Hardcoded sun is gone. | DONE |
| A2.sun.4   | ImGui panel "Sun" — lat/lon, year/month/day, local-hour slider, UTC offset, live azimuth/elevation/irradiance readout. | DONE |
| A2.sun.5   | Eyeballed against the ImGui readout — sun rises/sets and tracks plausibly through the day for Zermatt. Good enough for the alpine use case. | DONE |

## Phase 3 — Sun occlusion (the core technical work)

| ID         | Description                                                                  | Status |
|------------|------------------------------------------------------------------------------|--------|
| A3.occ.1   | Sun-POV ortho shadow map at 2048², depth-only pipeline driven by `terrain_depth.vert` (push-constant lightViewProj). Light frustum auto-fits to the terrain AABB rotated into light view. Terrain frag samples with 3×3 PCF. ImGui exposes enable + slope/constant bias. | DONE |
| A3.occ.2   | Horizon-map storage image: `R16_SFLOAT`, 512² × 32 azimuth layers (2D array). Built from the full-resolution DEM, uploaded once as an `R32F` sampled texture. | DONE |
| A3.occ.3   | `horizon_map.comp` raymarches each texel across the heightmap (128 steps × 150m = 19.2 km radius). One-shot dispatch at startup, terrain-only — independent of sun position so it serves all times of day. | DONE |
| A3.occ.4   | Terrain frag samples `sampler2DArray uHorizonMap` (binding 2). World-XY → UV via `terrainAabb` UBO field; sun azimuth → bin float; linear interpolation between adjacent bins; smoothstep comparison vs sun elevation. | DONE |
| A3.occ.5   | ImGui exposes "Shadow map" + "Horizon map" as independent toggles; `min(vShadow, vHorizon)` so either occluder darkens. | DONE |
| A3.occ.6   | **Optional** `VK_KHR_ray_tracing_pipeline` + BVH for sharp near-terrain shadows | TODO |

## Phase 4 — Sun-hours visualisation

| ID          | Description                                                                 | Status |
|-------------|-----------------------------------------------------------------------------|--------|
| A4.heat.1   | `sun_hours.comp` accumulates sun-hours-per-day per output texel. CPU-side fills a UBO with 144 sun directions (10-min steps via PSA); GPU loops them and reads the horizon map for each direction. R32F 512² accumulator. | DONE |
| A4.heat.2   | `viridis()` polynomial colormap in `terrain.frag`; when "Show colormap" is on, the terrain reads as sun-hours with a subtle shape-cue from N.z. | DONE |
| A4.heat.3   | Auto-recompute whenever lat/lon/year/month/day/tz/step changes — synchronous dispatch (~few ms), invisible to user. | DONE   |
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
