#pragma once

#include <cstdint>

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
    // Optional satellite-imagery JPEG (CMake-fetched ESRI World Imagery) used
    // as terrain albedo when the user enables "Satellite mode". nullptr means
    // satellite imagery is unavailable for this preset.
    const char* satImagePath;
    const char* description;
};

// Resolves the --terrain argument against the built-in preset list, prints
// --help and exits successfully when asked, or prints the available names and
// exits with failure when given an unknown preset. Always returns a valid
// preset reference (or terminates the process).
const TerrainPreset& resolveTerrainPreset(int argc, char** argv);
