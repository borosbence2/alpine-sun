#pragma once

// Local East-North-Up tangent plane.
//
// Origin at (centerLon, centerLat) at sea level. East is +X, North is +Y,
// Up is +Z, all in metres. Accurate to a few metres across a 1° tile; not
// suitable for cross-zone work (use a real geodetic projection for that).

namespace dem { struct Tile; }

namespace geo {

struct EnuFrame {
    double centerLon          = 0.0;   // WGS84 degrees
    double centerLat          = 0.0;
    double metresPerDegreeLat = 0.0;   // ~111319.5
    double metresPerDegreeLon = 0.0;   // cos(centerLat) * metresPerDegreeLat
};

EnuFrame frameFromTile(const dem::Tile& tile);

inline double lonToEast(const EnuFrame& f, double lon) {
    return (lon - f.centerLon) * f.metresPerDegreeLon;
}
inline double latToNorth(const EnuFrame& f, double lat) {
    return (lat - f.centerLat) * f.metresPerDegreeLat;
}

} // namespace geo
