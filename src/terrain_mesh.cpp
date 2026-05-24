#include "terrain_mesh.h"
#include "dem_loader.h"

#include <algorithm>
#include <cstdio>
#include <limits>

namespace terrain {

namespace {

// Returns the height at sampled-grid coordinate (sx, sy), clamped to the grid
// edge. The sampled grid samples the source DEM every `stride` pixels.
float sampleClamped(const dem::Tile& tile, uint32_t stride,
                    int sx, int sy,
                    uint32_t sampledW, uint32_t sampledH) {
    const int cx = std::clamp(sx, 0, static_cast<int>(sampledW) - 1);
    const int cy = std::clamp(sy, 0, static_cast<int>(sampledH) - 1);
    const uint32_t px = static_cast<uint32_t>(cx) * stride;
    const uint32_t py = static_cast<uint32_t>(cy) * stride;
    return tile.elevation[static_cast<size_t>(py) * tile.width + px];
}

} // namespace

Mesh makeMesh(const dem::Tile& tile, const MeshOptions& opts) {
    Mesh mesh;
    mesh.frame = geo::frameFromTile(tile);

    const uint32_t stride   = std::max(1u, opts.stride);
    const uint32_t sampledW = (tile.width  + stride - 1) / stride;
    const uint32_t sampledH = (tile.height + stride - 1) / stride;

    mesh.vertices.resize(static_cast<size_t>(sampledW) * sampledH);

    // The DEM is row-major north-to-south: row 0 sits at maxLat.
    // Vertex spacing in metres is `pixelSize_degrees * metresPerDegree * stride`.
    const double dxMetres = tile.pixelSizeLon * mesh.frame.metresPerDegreeLon * stride;
    const double dyMetres = tile.pixelSizeLat * mesh.frame.metresPerDegreeLat * stride;

    // Position of sampled grid corner (0,0) in ENU metres.
    const double x0 = geo::lonToEast (mesh.frame, tile.extent.minLon);
    const double y0 = geo::latToNorth(mesh.frame, tile.extent.maxLat);

    glm::vec3 aabbMin( std::numeric_limits<float>::infinity());
    glm::vec3 aabbMax(-std::numeric_limits<float>::infinity());

    for (uint32_t sy = 0; sy < sampledH; ++sy) {
        for (uint32_t sx = 0; sx < sampledW; ++sx) {
            const float z = sampleClamped(tile, stride,
                                          static_cast<int>(sx),
                                          static_cast<int>(sy),
                                          sampledW, sampledH);
            const float x = static_cast<float>(x0 + sx * dxMetres);
            const float y = static_cast<float>(y0 - sy * dyMetres);  // sy grows southward

            Vertex& v = mesh.vertices[static_cast<size_t>(sy) * sampledW + sx];
            v.pos = glm::vec3(x, y, z);
            v.uv  = glm::vec2(static_cast<float>(sx) / static_cast<float>(sampledW - 1),
                              static_cast<float>(sy) / static_cast<float>(sampledH - 1));

            if (opts.computeNormals) {
                const float zL = sampleClamped(tile, stride, sx - 1, sy, sampledW, sampledH);
                const float zR = sampleClamped(tile, stride, sx + 1, sy, sampledW, sampledH);
                const float zD = sampleClamped(tile, stride, sx, sy - 1, sampledW, sampledH);
                const float zU = sampleClamped(tile, stride, sx, sy + 1, sampledW, sampledH);
                const float dzdx = (zR - zL) / static_cast<float>(2.0 * dxMetres);
                const float dzdy = (zD - zU) / static_cast<float>(2.0 * dyMetres);  // y flipped
                v.normal = glm::normalize(glm::vec3(-dzdx, -dzdy, 1.0f));
            } else {
                v.normal = glm::vec3(0.0f, 0.0f, 1.0f);
            }

            aabbMin = glm::min(aabbMin, v.pos);
            aabbMax = glm::max(aabbMax, v.pos);
        }
    }
    mesh.aabbMin = aabbMin;
    mesh.aabbMax = aabbMax;

    // Two triangles per cell, counterclockwise when viewed from +Z.
    //   (sx, sy) -- (sx+1, sy)
    //      |             |
    //   (sx, sy+1) (sx+1, sy+1)
    const uint64_t cellsW = (sampledW > 0) ? (sampledW - 1) : 0;
    const uint64_t cellsH = (sampledH > 0) ? (sampledH - 1) : 0;
    mesh.indices.reserve(cellsW * cellsH * 6);
    for (uint32_t sy = 0; sy < cellsH; ++sy) {
        for (uint32_t sx = 0; sx < cellsW; ++sx) {
            const uint32_t tl =  sy      * sampledW +  sx;
            const uint32_t tr =  sy      * sampledW + (sx + 1);
            const uint32_t bl = (sy + 1) * sampledW +  sx;
            const uint32_t br = (sy + 1) * sampledW + (sx + 1);
            // Triangle 1: tl, bl, tr  (CCW when y goes screen-down and we
            // look down +Z onto the xy plane)
            mesh.indices.push_back(tl);
            mesh.indices.push_back(bl);
            mesh.indices.push_back(tr);
            // Triangle 2: tr, bl, br
            mesh.indices.push_back(tr);
            mesh.indices.push_back(bl);
            mesh.indices.push_back(br);
        }
    }

    return mesh;
}

} // namespace terrain
