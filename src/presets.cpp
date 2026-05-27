#include "presets.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

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
     ALPINE_SUN_SAT_MATTERHORN_PATH,
     "Matterhorn close-up (~9 km × 10 km) — full DEM detail"},
    // The original wide preset: full 1° tile, stride=8 to keep the mesh tractable.
    // No bundled satellite imagery; the area is too large for a single ESRI
    // export to read sharply.
    {"matterhorn-wide", ALPINE_SUN_MATTERHORN_TILE_PATH,
     7.0, 8.0, 45.0, 46.0, 8,
     45.976, 7.658, 2.0f, 30000.0f,
     nullptr,
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
     ALPINE_SUN_SAT_EVEREST_PATH,
     "Mt Everest close-up (~15 km × 10 km) — full DEM detail"},
};

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

} // namespace

const TerrainPreset& resolveTerrainPreset(int argc, char** argv) {
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
            std::exit(EXIT_SUCCESS);
        }
    }
    return findPresetOrDie(presetName);
}
