#pragma once

// PSA solar position algorithm (Blanco-Muriel, Alarcón-Padilla, López-Moratalla,
// Lara-Coira; Solar Energy Vol. 70, No. 5, 2001). Public-domain pseudocode
// transcribed straight from the paper.
//
// Valid 1999-2015 to ±0.5' (Blanco-Muriel's stated bounds); drifts gradually
// outside that window — accurate to ~0.01° (≈30") near 2026, more than enough
// for alpine sun/shadow work where DEM error dominates. Upgrade path is NREL
// SPA if we ever need sub-arcsecond accuracy.

namespace sun {

// All angles in radians on the way out. Azimuth follows the meteorological
// convention used by PSA: measured from North (=0), increasing clockwise (East
// = π/2). Zenith is angle from local vertical (0 = directly overhead, π/2 =
// horizon). Date+time must be in UTC.
struct PsaAngles {
    double zenithRad;
    double azimuthRad;
};

PsaAngles psaCompute(int    year,        // e.g. 2026
                     int    month,       // 1..12
                     int    day,         // 1..31
                     double hourUtc,     // fractional UTC hour, 0..24
                     double longitudeRad,// East positive
                     double latitudeRad);// North positive

} // namespace sun
