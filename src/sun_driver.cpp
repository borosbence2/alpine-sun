#include "sun_driver.h"

#include "psa.h"

#include <algorithm>
#include <cmath>

namespace sun {

namespace {

constexpr double kPi = 3.14159265358979323846;
// Solar constant — extraterrestrial irradiance at 1 AU, NREL 2008.
constexpr float kSolarConstant = 1361.0f; // W/m²

inline double deg2rad(double d) { return d * kPi / 180.0; }
inline double rad2deg(double r) { return r * 180.0 / kPi; }

} // namespace

Sample compute(double latitudeDeg,
               double longitudeDeg,
               int    year,
               int    month,
               int    day,
               double localHour,
               double tzOffsetH) {
    const double utcHour = localHour - tzOffsetH;
    // PSA wants UTC hour as a real number in roughly [0, 24); the algorithm
    // tolerates out-of-range values via the Julian-day formula, so we don't
    // need to roll month/day forward when the conversion crosses midnight.

    const PsaAngles a = psaCompute(year, month, day, utcHour,
                                   deg2rad(longitudeDeg),
                                   deg2rad(latitudeDeg));

    const double elevation = kPi * 0.5 - a.zenithRad;   // 0 at horizon, π/2 overhead
    const double cosEl = std::cos(elevation);
    const double sinEl = std::sin(elevation);
    const double cosAz = std::cos(a.azimuthRad);
    const double sinAz = std::sin(a.azimuthRad);

    // ENU: azimuth is measured from North clockwise (PSA convention), so
    //   east  = sin(az) * cos(el)
    //   north = cos(az) * cos(el)
    //   up    = sin(el)
    Sample s{};
    s.directionToSun = glm::vec3(static_cast<float>(sinAz * cosEl),
                                 static_cast<float>(cosAz * cosEl),
                                 static_cast<float>(sinEl));
    s.elevationDeg   = static_cast<float>(rad2deg(elevation));
    s.azimuthDeg     = static_cast<float>(rad2deg(a.azimuthRad));
    s.irradianceWm2  = kSolarConstant * std::max(0.0f, static_cast<float>(sinEl));
    return s;
}

} // namespace sun
