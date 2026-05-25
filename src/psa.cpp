// PSA solar position algorithm. See psa.h for accuracy bounds + citation.
//
// The integer arithmetic in the Julian-day step is intentional: PSA uses
// truncating division on signed ints, which is what reproduces the published
// constants. Don't "fix" it to floating-point.

#include "psa.h"

#include <cmath>

namespace sun {

namespace {

constexpr double kTwoPi = 6.28318530717958647692;
constexpr double kPi    = 3.14159265358979323846;
// Earth equatorial radius / 1 AU. Used to apply parallax correction so the
// computed zenith is at the observer rather than Earth-center.
constexpr double kEarthMeanRadius = 6371.01;     // km
constexpr double kAstronomicalUnit = 149597890.0;// km

} // namespace

PsaAngles psaCompute(int    year,
                     int    month,
                     int    day,
                     double hourUtc,
                     double longitudeRad,
                     double latitudeRad) {
    // --- 1. Julian day at the requested instant, referenced to J2000 (noon
    //        UT on 2000-01-01). The published formula is exact for the
    //        proleptic Gregorian calendar.
    const long aux1 = (month - 14) / 12;
    const long aux2 = (1461L * (year + 4800L + aux1)) / 4
                    + (367L * (month - 2 - 12 * aux1)) / 12
                    - (3L * ((year + 4900L + aux1) / 100L)) / 4
                    + day - 32075L;
    const double julianDate = static_cast<double>(aux2) - 0.5 + hourUtc / 24.0;
    const double elapsedDays = julianDate - 2451545.0;

    // --- 2. Ecliptic coordinates.
    const double omega    = 2.1429       - 0.0010394594  * elapsedDays;
    const double meanLong = 4.8950630    + 0.017202791698 * elapsedDays;
    const double meanAnom = 6.2400600    + 0.0172019699  * elapsedDays;
    const double eclipticLong = meanLong
                              + 0.03341607 * std::sin(meanAnom)
                              + 0.00034894 * std::sin(2.0 * meanAnom)
                              - 0.0001134
                              - 0.0000203 * std::sin(omega);
    const double eclipticObliquity = 0.4090928
                                   - 6.2140e-9 * elapsedDays
                                   + 0.0000396 * std::cos(omega);

    // --- 3. Celestial (equatorial) coordinates.
    const double sinEclLong = std::sin(eclipticLong);
    const double dy = std::cos(eclipticObliquity) * sinEclLong;
    const double dx = std::cos(eclipticLong);
    double rightAscension = std::atan2(dy, dx);
    if (rightAscension < 0.0) rightAscension += kTwoPi;
    const double declination = std::asin(std::sin(eclipticObliquity) * sinEclLong);

    // --- 4. Local coordinates: convert to hour angle via Greenwich mean
    //        sidereal time, then to zenith/azimuth via the observer's
    //        latitude.
    const double gmst = 6.6974243242 + 0.0657098283 * elapsedDays + hourUtc;
    const double lmst = (gmst * 15.0 + longitudeRad * 180.0 / kPi) * kPi / 180.0;
    const double hourAngle = lmst - rightAscension;

    const double cosLat = std::cos(latitudeRad);
    const double sinLat = std::sin(latitudeRad);
    const double cosHa  = std::cos(hourAngle);

    double zenith = std::acos(cosLat * cosHa * std::cos(declination)
                              + std::sin(declination) * sinLat);
    double azimuth = std::atan2(-std::sin(hourAngle),
                                std::tan(declination) * cosLat - sinLat * cosHa);
    if (azimuth < 0.0) azimuth += kTwoPi;

    // Parallax correction: the zenith above is geocentric; shift to the
    // observer's location on Earth's surface. The effect peaks near the
    // horizon (~9 arcseconds) and is negligible overhead, but it's cheap.
    const double parallax = (kEarthMeanRadius / kAstronomicalUnit) * std::sin(zenith);
    zenith += parallax;

    return PsaAngles{zenith, azimuth};
}

} // namespace sun
