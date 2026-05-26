# alpine-sun

A Vulkan-based tool for alpine trip planning. Loads real DEM terrain, places
the sun via the PSA (Blanco-Muriel 2001) algorithm, and visualises
sun/shadow at any time of day — plus sun-hours per day across the region,
GPX routes draped on the terrain, and optional satellite imagery as the
albedo.

The headline use case: pick a date, set the local hour, and see exactly
when each face of your peak goes into the sun to help planning the summit windows, and help tracking possible avanalches. Right-click any point to read
its sun-hours, latitude, longitude and elevation.

<img width="1280" height="747" alt="alpine-sun-5" src="https://github.com/user-attachments/assets/6809e621-5aaa-4bc9-9e68-ab9fec5e6bcb" />

<img width="1264" height="741" alt="alpine-sun-6" src="https://github.com/user-attachments/assets/5fc1c6dd-a7d8-459e-891c-e96b733a7838" />

<img width="1273" height="749" alt="alpine-sun-4" src="https://github.com/user-attachments/assets/7048e08f-9453-403b-bdaa-dc556c7626e0" />

<img width="1274" height="740" alt="alpine-sun-1" src="https://github.com/user-attachments/assets/f693932e-8ee6-4af2-a465-59a4f7754f4c" />

<img width="1275" height="754" alt="alpine-sun-3" src="https://github.com/user-attachments/assets/1ebd9451-24d9-4142-a278-fdfb07ef0e46" />

<img width="1273" height="745" alt="alpine-sun-2" src="https://github.com/user-attachments/assets/b9410f46-c484-4508-bb2f-9429226ab4f9" />


---

## Building

### Prerequisites

- A C++20 compiler (MSVC 2019+, recent GCC or Clang)
- CMake 3.22+
- Vulkan SDK 1.3+ (validation layers strongly recommended during development)
- A sibling checkout of [`forfun-graphics`](../forfun-graphics) — this project
  uses its `forfun_core` static library for Vulkan boilerplate (device,
  swapchain, frame context, helpers). The expected layout is:

  ```
  parent/
    alpine-sun/        ← this repo
    forfun-graphics/   ← sibling checkout
  ```

  If your layout differs, pass `-DFORFUN_GRAPHICS_DIR=/path/to/forfun-graphics`
  at configure time.

### Configure + build

```bash
cmake -B build
cmake --build build --config Debug
```

The first configure downloads a handful of test assets (~60–70 MB total):

- Two Copernicus GLO-30 DEM tiles (Matterhorn + Everest regions)
- Two ESRI World Imagery JPEGs (one per close-up preset)
- `stb_image.h` from `nothings/stb`

Everything lands under `assets/` and `src/vendor/`, both gitignored.

### Run

```bash
./build/Debug/alpine_sun.exe
```

---

## Controls

| Input             | Action                                                     |
|-------------------|------------------------------------------------------------|
| Hover             | Floating tooltip with lat/lon/elevation/sun-hours          |
| Left-drag         | Orbit the camera around the look-at target                 |
| Middle-drag       | Pan the look-at target in the screen plane                 |
| Scroll wheel      | Zoom (changes orbit distance)                              |
| Right-click       | Pin a sample to the side panel (lat/lon/elev/hours)        |
| Drop a `.gpx`     | Load that route, draped on the terrain                     |
| ESC               | Quit                                                       |

The orbit camera starts framed on the preset's "observer" point — Matterhorn
summit for the Swiss preset, Everest summit for the Nepal preset.

---

## Terrain presets

Pick a region with `--terrain <name>`:

```bash
./alpine_sun --terrain matterhorn       # default — ~9 × 10 km close-up
./alpine_sun --terrain matterhorn-wide  # original 1° × 1° tile (~80 × 110 km)
./alpine_sun --terrain everest          # ~15 × 10 km close-up of the SE side
./alpine_sun --help                     # list all presets
```

Each preset carries:

- A Copernicus GLO-30 sub-region (cropped from a single 1° GeoTIFF)
- An observer lat/lon and timezone offset (drives the initial sun panel)
- A camera framing distance
- Optionally: a bundled GPX route and a satellite imagery JPEG

The two close-ups use **full DEM detail** (mesh stride 1, ~150–180k vertices);
the wide preset uses stride 8 (~200k vertices over a 100× larger area).

---

## What's in the app

### Sun panel

Lives on the left side of the window. Sections from top to bottom:

- **Location**: latitude, longitude (degrees). Defaults to the preset's
  observer point.
- **Date**: year, month, day (drag the integer to scrub).
- **Time**: local hour (0–24, fractional), UTC offset.
- **Readout**: sun azimuth, elevation, irradiance, above-horizon flag.
- **Occlusion**: independent toggles for the sun-POV shadow map and the
  precomputed horizon map, plus shadow bias sliders.
- **Sun hours / day**: enable a viridis colormap of accumulated direct sun
  for the selected date. Includes a legend strip, sample-step slider
  (coarser = faster recompute), and "top of scale" slider.
- **Display**: ACES exposure, satellite-albedo toggle (when imagery is
  bundled for the preset).
- **Picked point**: appears once you right-click anywhere — lat/lon,
  elevation, sun-hours/day for that exact spot. The floating tooltip that
  follows the cursor shows the same values live; the picked panel is for
  when you want a value to stay put.
- **Route**: status + visibility checkbox + collapsible per-waypoint table
  with index, lat, lon, elevation, sun-hours/day.

### Two occlusion techniques running in parallel

- **Sun-POV shadow map**: orthographic depth pass from the sun direction
  over the terrain VB/IB, sampled with 3×3 PCF in the terrain fragment
  shader. Catches sharp near-terrain shadows. Auto-fit per-frame to the
  rotated AABB.
- **Horizon map**: at startup, a compute shader marches each output texel
  along 32 azimuth bins through the DEM, recording the maximum apparent
  elevation per bin. At runtime the terrain fragment shader looks up the
  bin pair surrounding the sun's azimuth and smoothsteps against the sun's
  elevation. O(1) per fragment, catches large-scale ridge occlusion the
  shadow map can't resolve at this scale.

Either or both can be on. They combine with `min(visibilityA, visibilityB)`.

### Sun-hours-per-day visualisation

After every change to date/location/timezone, a compute shader (~few ms)
walks 144 sun-direction samples across the day (default 10-minute step),
queries the horizon map at each, and accumulates the visible hours per
output texel into an R32F image. Enabling the colormap replaces the
procedural shading with a viridis-mapped readout, and the same data is read
back into a 1 MB host buffer so the route's per-waypoint sun-hours table is
just a CPU lookup.

### Atmospheric sky + tonemap

A physically-based sky in the Hillaire 2020 style. Two pre-baked LUTs feed
a fullscreen-triangle sky pass:

- **Transmittance LUT** (256×64): one-shot at startup. Maps `(cos viewZenith,
  altitude)` → RGB transmittance through the atmosphere. Earth-tuned
  constants for Rayleigh, Mie and ozone extinction.
- **Sky-view LUT** (192×108): rebaked whenever the sun's direction drifts
  more than ~0.5°. Single-scattering integration along each direction from
  the observer, with phase functions sampled at the current sun angle.

`sky.frag` reconstructs the view direction from `invViewProj`, looks up the
sky-view LUT with horizon-biased non-linear v sampling, and punches a small
sun disk where the view direction lines up with the sun. The result is
proper Rayleigh blue, warm reddening near the horizon at low sun, and a
deep cool sky at night — without any of the ad-hoc gradient math the
previous procedural sky used.

The terrain `frag` shader also samples the sky-view LUT — in the direction
of each fragment's surface normal — as the **ambient** term. Shaded north
faces pick up cool blue from the zenith at noon; at sunset, faces oriented
toward the sun pick up warm orange. A tiny constant floor keeps shaded
areas readable on moonless nights.

Both sky and terrain pass through the same ACES (Narkowicz fit) tonemap
with a user-controlled exposure.

Not yet implemented: multi-scattering LUT and aerial perspective. The
single-scattering single-bounce result already covers the dominant visual.

### GPX routes

Drag-and-drop a `.gpx` file onto the window. The tiny built-in parser
(`src/gpx.{h,cpp}`) reads `<trkpt>` and `<rtept>` elements. Each waypoint's
lat/lon is converted to the local ENU frame; elevation comes from a
bilinear DEM sample (the parser also reads `<ele>` but we override — keeps
the line glued to *our* terrain surface). The line gets lifted ~15 m so it
floats above the surface without z-fighting.

Two routes come bundled:

- **Matterhorn**: hand-traced approximation of the Hörnli ridge (Hörnli Hut
  → summit). ~1.4 km, 9 waypoints.
- **Everest**: real Strava route of the South-Col line (Camp 2 → Lhotse face
  → South Col → SE ridge → summit). Several hundred waypoints, ~10 km.

### Satellite imagery

Toggle "Satellite albedo" to swap the procedural grass/rock/snow ramp for a
real satellite photo of the region (ESRI World Imagery, fetched at CMake
configure time). Stored as `R8G8B8A8_SRGB` so the sampler does sRGB→linear,
then the same ACES tonemap applies. Same world-XY→UV mapping as the
horizon and sun-hours samplers.

### Picking + hover

Hover any spot on the terrain to see a small four-line tooltip next to the
cursor with lat / lon / elevation / sun-hours. The hover does a **CPU
ray-march** against the DEM (50 m step + binary refine; tens of
microseconds per frame), so no GPU stall.

Right-click works the same way but the result *pins* to the side panel
instead of chasing the cursor. The picking path uses depth-buffer readback
of the rendered frame: depth attachment → 1-texel copy → unproject via
`invViewProj`. Clicking on the sky is silently ignored.

Both paths sample the same CPU-resident sun-hours readback buffer for the
hours/day number, so they're internally consistent.

---

## Architecture

- Single executable, ~3.2k LOC of `main.cpp` plus small files for the GPX
  parser, DEM loader, terrain mesh, geo frame, PSA, and sun driver.
- Built on top of `forfun_core` (from the sibling `forfun-graphics` repo) —
  reuses Device, Swapchain, FrameContext, VMA helpers, transitionImage,
  buffer/image upload, shader compilation pipeline, ImGui static lib.
- One descriptor set with six bindings drives the terrain pipeline:
  camera UBO, shadow map sampler, horizon map sampler2DArray, sun-hours
  sampler2D, satellite sampler2D, sky-view LUT sampler2D.
- Nine shaders: `terrain.vert` + `terrain.frag`, `terrain_depth.vert`
  (shadow pass, push-constant matrix), `sky.vert` + `sky.frag`,
  `route.vert` + `route.frag`, `horizon_map.comp`, `sun_hours.comp`,
  `transmittance.comp`, `sky_view.comp`.
- All compute work runs at startup or only when a depending parameter
  changes — horizon map and transmittance LUT are one-shot, sun-hours
  rebakes when date/location updates, sky-view rebakes when the sun's
  direction drifts more than ~0.5°.

See [`MILESTONES.md`](MILESTONES.md) for a detailed log of every milestone
and the design choices behind it.

---

## Data sources + attribution

- **DEM**: [Copernicus GLO-30](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model)
  via AWS Open Data (`copernicus-dem-30m` S3 bucket). Public domain, no
  authentication.
- **Satellite imagery**: [ESRI World Imagery](https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9)
  via the public Export Image REST endpoint. Free for non-commercial use,
  no API key. Sources vary by region (Maxar, Airbus, USGS NAIP, etc.).
- **Sun algorithm**: PSA from Blanco-Muriel, Alarcón-Padilla, López-Moratalla
  and Lara-Coira, *Solar Energy* Vol. 70, No. 5, 2001. Public-domain
  pseudocode transcribed straight from the paper. Accurate to ~0.01° near
  the present epoch.
- **`stb_image.h`**: [Sean Barrett's stb](https://github.com/nothings/stb),
  public domain.
- **Matterhorn Hörnli ridge GPX**: hand-traced from public maps for demo
  purposes; not climbing-grade accuracy.
- **Everest South-Col route GPX**: Strava route by Kenton Cool, used with
  OpenStreetMap contributor attribution.

---

## Status

Phases 0 through 5 substantively complete — see
[`MILESTONES.md`](MILESTONES.md) for the per-task log. Remaining open items:

- `A3.occ.6` — optional ray-traced sharp shadows via `VK_KHR_ray_tracing_pipeline`.
- `A5.use.5` — save/load scenario JSON.
- Hillaire follow-ups — multi-scattering LUT and aerial-perspective volume.
- Phase 6 polish — quadtree LOD, glacier mask overlay, cloud shadows,
  photogrammetry import.

---

## Caveats

- **Tile-edge crop**: Matterhorn and Everest both sit close to the north
  edge of their respective Copernicus GLO-30 tiles, so the close-up presets
  have only a few km of terrain north of the summit. Multi-tile stitching
  would lift this but adds complexity.
- **DEM resolution**: GLO-30 is 30 m/pixel everywhere. Cropping makes
  features bigger on screen but doesn't add detail. Higher-resolution
  sources (swissALTI3D 0.5 m, HMA-DEM 8 m, SETSM 2 m) exist but require
  authentication, much larger file sizes, or both.
- **Horizon-map pessimism**: the horizon map records the *single highest
  ridge* along each azimuth. It can't tell that the sun-ray actually misses
  a particular ridge by a few metres, so it sometimes over-shades narrow
  valleys. The shadow map has the opposite trade-off (resolution-limited
  at large scales but locally accurate).
- **`wideLines`**: the route's line strip renders at 1 px width — the
  feature isn't enabled in forfun's device. Magenta against terrain is
  visible enough for a demo; could be widened later by enabling the
  feature or rendering as triangle strips.
