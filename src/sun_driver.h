#pragma once

// SunDriver — turns a wall-clock observation (location + civil date + time)
// into a sun direction usable directly by the terrain shader.
//
// Coordinate frame: matches the terrain mesh's local ENU (East +X, North +Y,
// Up +Z). `directionToSun` is a unit vector pointing FROM the surface TO the
// sun; below-horizon means its .z component is negative.

#include <glm/glm.hpp>

namespace sun {

struct Sample {
    glm::vec3 directionToSun;   // unit vector in ENU; .z<0 means below horizon
    float     elevationDeg;     // 90° = overhead, 0° = horizon, negative = night
    float     azimuthDeg;       // 0° = N, 90° = E, 180° = S, 270° = W
    float     irradianceWm2;    // clear-sky direct beam at top of atmosphere,
                                // clamped to 0 when sun is below horizon. No
                                // atmospheric extinction yet — that's Phase 5.
};

// Civil date/time: localHour is local clock time (0..24, fractional). tzOffsetH
// is the local standard offset from UTC in hours (e.g. +1 for CET, +2 for CEST,
// -5 for EST). We don't auto-detect DST — caller passes the offset for the
// date they care about.
Sample compute(double latitudeDeg,
               double longitudeDeg,
               int    year,
               int    month,
               int    day,
               double localHour,
               double tzOffsetH);

} // namespace sun
