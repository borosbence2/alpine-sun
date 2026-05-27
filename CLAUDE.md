# alpine-sun — codebase guide

A Vulkan visualisation of solar irradiance over Alpine terrain. DEM tiles +
GPX routes go in; per-frame sun direction, shadow + horizon occlusion,
sun-hours-per-day overlays, satellite albedo, Hillaire sky, and an aerial
perspective haze come out.

Built on `forfun-graphics` (sibling source tree, consumed via CPM). All
Vulkan helpers, sync types, and `kFramesInFlight` come from `forfun_core`.

## File map

Source lives in `src/`. The split is by *subsystem ownership*, not by
"types vs functions":

| File                          | What it owns |
|-------------------------------|--------------|
| `main.cpp`                    | Entry point, window + swapchain + device, terrain rendering pipeline (terrain/depth/sky/route), per-frame loop, ImGui panels, picking, satellite imagery, route subsystem, hover ray-march. |
| `presets.h/.cpp`              | `TerrainPreset` struct, built-in preset table, `--terrain` CLI parsing (`resolveTerrainPreset`). |
| `app_state.h/.cpp`            | UI-state structs (`SunUi`, `ShadowUi`, `SunHoursUi`, `ToneUi`, `SatUi`, `AvalancheUi`, `Route`, `OrbitCamera`, `InputState`, pick/hover/pending-GPX) and their process-wide globals (`g_camera`, `g_sun`, etc.). |
| `atmosphere.h/.cpp`           | `AtmosphereSystem` — heightmap, horizon map, sun-hours accumulator, transmittance + sky-view LUTs, their pipelines and bakes. `createAtmosphereSystem` / `destroyAtmosphereSystem` / `bakeSunHours` / `bakeSkyView` / `updateRouteSunHoursFromAtmosphere`. |
| `dem_loader.h/.cpp`           | GeoTIFF loading + lat/lon cropping (libtiff). |
| `geo_frame.h/.cpp`            | ENU frame conversions (lat/lon ↔ east/north metres). |
| `terrain_mesh.h/.cpp`         | DEM → triangle mesh (with stride) → AABB. |
| `sun_driver.h/.cpp`           | Wraps `psa.h` to return `directionToSun` + direct-beam irradiance for a (lat, lon, date, time, tz). |
| `psa.h/.cpp`                  | PSA solar-position algorithm (Blanco-Muriel et al.). Pure math, no Vulkan. |
| `gpx.h/.cpp`                  | GPX parser → `std::vector<Waypoint>` (lat/lon/ele). |
| `vendor/stb_image.h`          | Header-only declarations; impl provided transitively by forfun_core. |

Shaders are in `shaders/` and compiled to embedded SPIR-V headers by the
`add_shader` macro from forfun-graphics' CMake.

## Conventions

- **UI globals live in `app_state.h`.** Anything new that the GLFW
  callbacks, ImGui panels, or per-frame loop need to share should join
  the existing `g_*` declarations there. Don't proliferate file-local
  globals in `main.cpp`.
- **Each GPU subsystem owns its own teardown.** `AtmosphereSystem`'s
  destroy function fully releases its images, samplers, descriptor
  pools, pipelines, and command pool. New subsystems should follow the
  same pattern: `createX` returns a value, `destroyX` takes it by
  reference. main.cpp's cleanup section is just a list of these calls.
- **Bake-driver state stays where the bake is triggered.** The
  AtmosphereSystem does not track *when* to re-bake — main.cpp owns
  `lastBakedParams` and `lastSkyBakeSunDir` and decides per frame.
- **No dynamic allocation in the hot path.** Per-frame buffers are
  pre-allocated; route VBs are uploaded on GPX drop.

## Where to add a new feature

- **A new render pass / pipeline** (e.g. a debug overlay): keep its
  pipeline + descriptor objects in `main.cpp` near `terrainPipeline` /
  `skyPipeline` / `routePipeline`. Push its sampled inputs into the
  shared `cameraDescSets` if they're per-frame.
- **A new sun/atmosphere bake or LUT**: extend `AtmosphereSystem` in
  `atmosphere.h`. Add the image + sampler + pipeline handles, init it
  from `createAtmosphereSystem`, tear it down in
  `destroyAtmosphereSystem`, and expose a `bake*` free function if the
  bake is re-runnable.
- **A new UI toggle / panel slider**: add a field to the relevant
  `*Ui` struct in `app_state.h`. If it influences a shader, pipe it
  through the existing `CameraUBO` (which lives in `main.cpp`).
- **A new preset**: append to `kPresets` in `presets.cpp` and add the
  CMake `alpine_sun_download_dem` / `alpine_sun_download_sat` calls in
  `CMakeLists.txt` so its tiles get fetched at configure time.
- **A new keyboard / mouse interaction**: extend `InputState` in
  `app_state.h`, wire the GLFW callback in `main.cpp`'s setup block.

## Build

CMake 3.22+. Configure once:

```
cmake -S . -B build
cmake --build build --config Debug --target alpine_sun
```

The configure step downloads two Copernicus GLO-30 DEM tiles (Matterhorn,
Everest) and two ESRI satellite images via `file(DOWNLOAD ...)` — no API
keys needed. Run with `--terrain <name>` (defaults to `matterhorn`); see
`--help` for the preset list.
