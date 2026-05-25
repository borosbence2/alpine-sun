#pragma once

// Tiny GPX reader. Pulls lat/lon/elevation out of every `<trkpt>` and
// `<rtept>` element in the file and concatenates them into a single
// polyline. Segment boundaries are flattened — good enough for sun-on-route
// visualisation; multi-segment splitting can layer on later.
//
// We hand-roll a scanner rather than vendoring an XML library because GPX is
// regular enough that string searches handle it and adding a dep for one
// feature feels disproportionate.

#include <string>
#include <vector>

namespace gpx {

struct Waypoint {
    double lat;     // WGS84 degrees, +N
    double lon;     // WGS84 degrees, +E
    float  ele;     // metres (0 if missing from the GPX)
    bool   hasEle;  // false → ele wasn't in the file, caller should sample DEM
};

// Reads `path` and fills `out`. Returns false on file-open failure; an
// unparseable or empty file returns true with an empty `out`.
bool loadFile(const std::string& path, std::vector<Waypoint>& out);

} // namespace gpx
