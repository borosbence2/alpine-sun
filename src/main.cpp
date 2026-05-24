// alpine-sun — entry point.
//
// Phase 1 in progress. On startup, loads the Matterhorn DEM tile, generates
// a terrain mesh, then creates a Vulkan device via forfun_core. Opens a
// window and runs an empty event loop. Actual rendering arrives once the
// rest of Phase 0B is extracted (Swapchain, FrameContext).

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "dem_loader.h"
#include "terrain_mesh.h"
#include "device.h"        // forfun::createDevice / destroyDevice
#include "swapchain.h"     // forfun::createSwapchain / destroySwapchain
#include "frame_context.h" // forfun::createFrameContext / destroyFrameContext
#include "vk_helpers.h"    // transitionImage
#include "types.h"         // VK_CHECK, kFramesInFlight

#include <array>

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

    forfun::Device gpu = forfun::createDevice({
        .appName          = "alpine-sun",
        .enableValidation = true,
        .createSurface    = [window](VkInstance inst) {
            VkSurfaceKHR s = VK_NULL_HANDLE;
            VK_CHECK(glfwCreateWindowSurface(inst, window, nullptr, &s));
            return s;
        },
    });
    std::printf("vulkan: device created, graphicsFamily=%u\n", gpu.graphicsFamily);

    forfun::Swapchain sc = forfun::createSwapchain(gpu, {
        .desiredExtent      = {1280, 720},
        .desiredFormat      = VK_FORMAT_B8G8R8A8_SRGB,
        .desiredColorSpace  = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .desiredPresentMode = VK_PRESENT_MODE_FIFO_KHR,
    });
    std::printf("vulkan: swapchain created, %ux%u, %zu images\n",
                sc.extent.width, sc.extent.height, sc.images.size());

    std::array<forfun::FrameContext, kFramesInFlight> frames{};
    for (auto& f : frames) f = forfun::createFrameContext(gpu);
    std::printf("vulkan: %u frame contexts ready (per-frame cmd pool + sync)\n",
                kFramesInFlight);
    std::printf("alpine-sun: window open. Press ESC to quit.\n");

    // Dawn-on-snow tint: pale warm pink. Distinctive enough to confirm the
    // clear actually happened (vs. driver-default black).
    constexpr VkClearColorValue kClearColor = {{0.95f, 0.82f, 0.78f, 1.0f}};

    uint32_t frameIndex = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        forfun::FrameContext& frame = frames[frameIndex];

        // 1. Wait for the previous use of this frame slot to finish.
        VK_CHECK(vkWaitForFences(gpu.device, 1, &frame.inFlight, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences  (gpu.device, 1, &frame.inFlight));

        // 2. Acquire a swapchain image; the imageAvailable semaphore signals
        //    once the presentation engine releases it.
        uint32_t imageIndex = 0;
        VK_CHECK(vkAcquireNextImageKHR(gpu.device, sc.swapchain, UINT64_MAX,
                                       frame.imageAvailable, VK_NULL_HANDLE, &imageIndex));

        // 3. Record commands: transition to color-attachment, clear via
        //    dynamic rendering, transition to present-src.
        VK_CHECK(vkResetCommandBuffer(frame.cmd, 0));
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(frame.cmd, &beginInfo));

        transitionImage(frame.cmd, sc.images[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        colorAttachment.imageView          = sc.imageViews[imageIndex];
        colorAttachment.imageLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp             = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp            = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color   = kClearColor;

        VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
        renderingInfo.renderArea           = {{0, 0}, sc.extent};
        renderingInfo.layerCount           = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments    = &colorAttachment;

        vkCmdBeginRendering(frame.cmd, &renderingInfo);
        // (terrain draws go here in A1.terrain.7)
        vkCmdEndRendering(frame.cmd);

        transitionImage(frame.cmd, sc.images[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);

        VK_CHECK(vkEndCommandBuffer(frame.cmd));

        // 4. Submit: wait on imageAvailable, signal renderFinished + fence.
        VkCommandBufferSubmitInfo cbSubmit{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cbSubmit.commandBuffer = frame.cmd;

        VkSemaphoreSubmitInfo waitSem{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        waitSem.semaphore = frame.imageAvailable;
        waitSem.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSemaphoreSubmitInfo signalSem{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signalSem.semaphore = frame.renderFinished;
        signalSem.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

        VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submitInfo.waitSemaphoreInfoCount   = 1;
        submitInfo.pWaitSemaphoreInfos      = &waitSem;
        submitInfo.signalSemaphoreInfoCount = 1;
        submitInfo.pSignalSemaphoreInfos    = &signalSem;
        submitInfo.commandBufferInfoCount   = 1;
        submitInfo.pCommandBufferInfos      = &cbSubmit;

        VK_CHECK(vkQueueSubmit2(gpu.graphicsQueue, 1, &submitInfo, frame.inFlight));

        // 5. Present.
        VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores    = &frame.renderFinished;
        presentInfo.swapchainCount     = 1;
        presentInfo.pSwapchains        = &sc.swapchain;
        presentInfo.pImageIndices      = &imageIndex;
        VK_CHECK(vkQueuePresentKHR(gpu.graphicsQueue, &presentInfo));

        frameIndex = (frameIndex + 1) % kFramesInFlight;
    }

    // Make sure the GPU is idle before tearing down resources still in use.
    VK_CHECK(vkDeviceWaitIdle(gpu.device));

    for (auto& f : frames) forfun::destroyFrameContext(gpu, f);
    forfun::destroySwapchain(gpu, sc);
    forfun::destroyDevice(gpu);
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
