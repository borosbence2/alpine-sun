#pragma once

// DEM (Digital Elevation Model) loader for WGS84 GeoTIFFs.
//
// Targeted at Copernicus GLO-30: float32 elevation in metres above ellipsoid,
// row-major north-to-south (TIFF orientation), WGS84 lat/lon grid with a
// fixed pixel scale defined by ModelPixelScaleTag (33550) and a tie point
// defined by ModelTiepointTag (33922).

#include <cstdint>
#include <string>
#include <vector>

namespace dem {

struct GeoExtent {
    double minLon = 0.0;   // SW corner, WGS84 degrees
    double minLat = 0.0;
    double maxLon = 0.0;   // NE corner
    double maxLat = 0.0;
};

struct Tile {
    std::vector<float> elevation;       // size = width * height, row-major, north-to-south
    uint32_t           width        = 0;
    uint32_t           height       = 0;
    GeoExtent          extent;
    double             pixelSizeLon = 0.0;  // degrees per pixel (longitude)
    double             pixelSizeLat = 0.0;  // degrees per pixel (latitude, positive)
    float              minElevation = 0.0f;
    float              maxElevation = 0.0f;
};

// Loads a WGS84 GeoTIFF DEM into `out`. Returns false on any failure (file
// missing, unsupported pixel format, missing required geo tags); the error is
// printed to stderr. On success, `out` is fully populated.
bool loadGeoTIFF(const std::string& path, Tile& out);

} // namespace dem
