#include "geo_frame.h"
#include "dem_loader.h"

#include <cmath>
#include <numbers>

namespace geo {

EnuFrame frameFromTile(const dem::Tile& tile) {
    EnuFrame f;
    f.centerLon = 0.5 * (tile.extent.minLon + tile.extent.maxLon);
    f.centerLat = 0.5 * (tile.extent.minLat + tile.extent.maxLat);
    // WGS84 mean meridian arc length / 360°.
    f.metresPerDegreeLat = 111319.5;
    f.metresPerDegreeLon = std::cos(f.centerLat * std::numbers::pi / 180.0)
                         * f.metresPerDegreeLat;
    return f;
}

} // namespace geo
