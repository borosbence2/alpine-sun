// alpine-sun — entry point.
//
// Phase 1 in progress. On startup, loads the Matterhorn DEM tile and prints
// its metadata as a smoke test for the GeoTIFF loader (A1.terrain.1). Opens a
// window and runs an empty event loop; rendering arrives once engine
// primitives are extracted from helmet_demo in Phase 0B.

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "dem_loader.h"
#include "terrain_mesh.h"

#include <cstdio>
#include <cstdlib>

namespace {

void smokeTestPipeline() {
    dem::Tile tile;
    const char* path = ALPINE_SUN_MATTERHORN_TILE_PATH;
    std::printf("dem: loading %s\n", path);
    if (!dem::loadGeoTIFF(path, tile)) {
        std::fprintf(stderr, "dem: load failed\n");
        std::exit(EXIT_FAILURE);
    }
    std::printf("dem: %u x %u, lon [%.4f .. %.4f], lat [%.4f .. %.4f]\n",
                tile.width, tile.height,
                tile.extent.minLon, tile.extent.maxLon,
                tile.extent.minLat, tile.extent.maxLat);
    std::printf("dem: pixel size %.6f° lon, %.6f° lat\n",
                tile.pixelSizeLon, tile.pixelSizeLat);
    std::printf("dem: elevation [%.1f .. %.1f] m\n",
                static_cast<double>(tile.minElevation),
                static_cast<double>(tile.maxElevation));

    // stride=8 keeps the MVP mesh light (~200k verts, ~400k tris) and quick
    // to regenerate while iterating. Drop to stride=1 for full GLO-30 detail.
    terrain::MeshOptions opts;
    opts.stride = 8;
    const terrain::Mesh mesh = terrain::makeMesh(tile, opts);
    std::printf("mesh: %zu verts, %zu tris, stride=%u\n",
                mesh.vertices.size(), mesh.indices.size() / 3, opts.stride);
    std::printf("mesh: AABB min=(%.0f, %.0f, %.0f) max=(%.0f, %.0f, %.0f) m\n",
                mesh.aabbMin.x, mesh.aabbMin.y, mesh.aabbMin.z,
                mesh.aabbMax.x, mesh.aabbMax.y, mesh.aabbMax.z);
    std::printf("mesh: ENU origin lon=%.4f lat=%.4f, m/deg lon=%.1f lat=%.1f\n",
                mesh.frame.centerLon, mesh.frame.centerLat,
                mesh.frame.metresPerDegreeLon, mesh.frame.metresPerDegreeLat);
}

} // namespace

int main() {
    smokeTestPipeline();

    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit failed\n");
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "alpine-sun", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }

    std::printf("alpine-sun: window open. Press ESC to quit.\n");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
