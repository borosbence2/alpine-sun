#pragma once

// Heightmap → triangle mesh, in the local ENU frame.
//
// Vertices are forfun_core's Vertex (pos + normal + uv). Index buffer uses
// uint32 since stride=1 over a Copernicus GLO-30 tile exceeds the uint16
// limit (3600 × 3600 = 12.96M vertices).

#include "geo_frame.h"
#include "types.h"     // forfun_core Vertex

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace dem { struct Tile; }

namespace terrain {

struct MeshOptions {
    // Sample every Nth pixel of the source DEM. stride=1 = full resolution
    // (~13M verts for a GLO-30 tile); stride=8 = 1/64 (~200k verts) — good
    // for MVP iteration.
    uint32_t stride         = 1;
    bool     computeNormals = true;
};

struct Mesh {
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;
    glm::vec3             aabbMin{0.0f};
    glm::vec3             aabbMax{0.0f};
    geo::EnuFrame         frame;
};

Mesh makeMesh(const dem::Tile& tile, const MeshOptions& opts = {});

} // namespace terrain
