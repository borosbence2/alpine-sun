#include "gpx.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace gpx {

namespace {

// Pulls the value of `name="..."` out of an XML opening tag. Returns NaN if
// the attribute isn't present — caller treats that as a parse failure.
double extractAttribute(const std::string& tag, const char* name) {
    std::string key = std::string(name) + "=\"";
    const size_t k = tag.find(key);
    if (k == std::string::npos) return std::nan("");
    const size_t valStart = k + key.size();
    const size_t valEnd   = tag.find('"', valStart);
    if (valEnd == std::string::npos) return std::nan("");
    return std::atof(tag.substr(valStart, valEnd - valStart).c_str());
}

// Returns the elevation in metres found between `start` and `limit`, or NaN
// if no `<ele>...</ele>` lives in that window.
float extractElevation(const std::string& content, size_t start, size_t limit) {
    const size_t s = content.find("<ele>", start);
    if (s == std::string::npos || s >= limit) return std::nanf("");
    const size_t valStart = s + 5;
    const size_t valEnd   = content.find("</ele>", valStart);
    if (valEnd == std::string::npos || valEnd >= limit) return std::nanf("");
    return static_cast<float>(std::atof(content.substr(valStart, valEnd - valStart).c_str()));
}

} // namespace

bool loadFile(const std::string& path, std::vector<Waypoint>& out) {
    out.clear();
    std::ifstream file(path);
    if (!file) {
        std::fprintf(stderr, "gpx: cannot open %s\n", path.c_str());
        return false;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    const std::string content = ss.str();

    size_t pos = 0;
    while (true) {
        // Find the next track or route point. We treat both identically; GPX
        // distinguishes them but for visualisation they're the same polyline.
        const size_t trkpt = content.find("<trkpt", pos);
        const size_t rtept = content.find("<rtept", pos);
        size_t start = std::min(trkpt, rtept);
        if (start == std::string::npos) break;
        const bool isTrk = (trkpt < rtept);

        const size_t openEnd = content.find('>', start);
        if (openEnd == std::string::npos) break;
        const std::string opener = content.substr(start, openEnd - start);

        const double lat = extractAttribute(opener, "lat");
        const double lon = extractAttribute(opener, "lon");
        if (std::isnan(lat) || std::isnan(lon)) {
            pos = openEnd + 1;
            continue;
        }

        // Element body ends at the matching close tag; lock to it so a stray
        // <ele> belonging to the *next* point can't leak into this one.
        const char* closeTag = isTrk ? "</trkpt>" : "</rtept>";
        const size_t closeEnd = content.find(closeTag, openEnd);
        const size_t limit    = (closeEnd == std::string::npos) ? content.size() : closeEnd;

        Waypoint wp{};
        wp.lat = lat;
        wp.lon = lon;
        const float ele = extractElevation(content, openEnd, limit);
        if (!std::isnan(ele)) {
            wp.ele    = ele;
            wp.hasEle = true;
        }
        out.push_back(wp);

        pos = (closeEnd == std::string::npos) ? (openEnd + 1) : (closeEnd + 1);
    }

    return true;
}

} // namespace gpx
