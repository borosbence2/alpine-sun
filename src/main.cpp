// alpine-sun — entry point.
//
// Phase 1c: actually draws the Matterhorn terrain mesh. Camera is hardcoded
// for now (orbit/fly controls arrive in A1.terrain.6). Slope/height debug
// shading + placeholder fixed-sun Lambert lighting (real sun via SPA in
// Phase 2).

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "dem_loader.h"
#include "terrain_mesh.h"
#include "device.h"        // forfun::createDevice / destroyDevice
#include "swapchain.h"     // forfun::createSwapchain / destroySwapchain
#include "frame_context.h" // forfun::createFrameContext / destroyFrameContext
#include "vk_helpers.h"    // transitionImage, createBufferGPU/HostMapped, etc.
#include "types.h"         // VK_CHECK, kFramesInFlight, Vertex, kDepthFormat

#include <glm/gtc/matrix_transform.hpp>

#include "terrain.vert.h"
#include "terrain.frag.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// std140-ish camera block. All three matrices are written each frame; the
// shader uses viewProj, but view+proj are kept for future passes (shadow,
// horizon precompute) that may want them separately.
struct alignas(16) CameraUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
};

// Orbit camera: eye orbits around `target` on a sphere of radius `distance`.
// yaw is the compass angle from +X (east) measured CCW around +Z; pitch is the
// elevation angle above the horizontal plane through `target`.
struct OrbitCamera {
    glm::vec3 target      = glm::vec3(12000.0f, 53000.0f, 2500.0f);  // near Matterhorn
    float     yaw         = -2.356f;   // ~-135°: camera SW of target
    float     pitch       =  0.393f;   // ~22.5° above
    float     distance    = 30000.0f;
    float     minDistance =   500.0f;
    float     maxDistance = 250000.0f;
};

struct InputState {
    bool   leftDown      = false;
    double lastX         = 0.0;
    double lastY         = 0.0;
    double scrollPending = 0.0;   // GLFW scroll deltas accumulate here; consumed each frame
};

// Globals so the GLFW scroll callback can reach them. We only ever have one
// window in this process, so a singleton is fine.
OrbitCamera g_camera;
InputState  g_input;

void scrollCallback(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset) {
    g_input.scrollPending += yoffset;
}

glm::vec3 orbitEye(const OrbitCamera& c) {
    const float cp = std::cos(c.pitch);
    const float sp = std::sin(c.pitch);
    const float cy = std::cos(c.yaw);
    const float sy = std::sin(c.yaw);
    return c.target + c.distance * glm::vec3(cp * cy, cp * sy, sp);
}

terrain::Mesh loadAndBuildTerrain() {
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
    terrain::Mesh mesh = terrain::makeMesh(tile, opts);
    std::printf("mesh: %zu verts, %zu tris, stride=%u\n",
                mesh.vertices.size(), mesh.indices.size() / 3, opts.stride);
    std::printf("mesh: AABB min=(%.0f, %.0f, %.0f) max=(%.0f, %.0f, %.0f) m\n",
                mesh.aabbMin.x, mesh.aabbMin.y, mesh.aabbMin.z,
                mesh.aabbMax.x, mesh.aabbMax.y, mesh.aabbMax.z);
    std::printf("mesh: ENU origin lon=%.4f lat=%.4f, m/deg lon=%.1f lat=%.1f\n",
                mesh.frame.centerLon, mesh.frame.centerLat,
                mesh.frame.metresPerDegreeLon, mesh.frame.metresPerDegreeLat);
    return mesh;
}

} // namespace

int main() {
    terrain::Mesh mesh = loadAndBuildTerrain();

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

    // ---- Depth attachment ----
    DepthImage depth = createDepthImage(gpu.allocator, gpu.device, sc.extent,
                                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

    // ---- Transient pool for one-time uploads + depth transition ----
    VkCommandPool uploadPool = VK_NULL_HANDLE;
    {
        VkCommandPoolCreateInfo poolCi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolCi.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCi.queueFamilyIndex = gpu.graphicsFamily;
        VK_CHECK(vkCreateCommandPool(gpu.device, &poolCi, nullptr, &uploadPool));
    }

    // Depth image lives in DEPTH_ATTACHMENT_OPTIMAL for its whole lifetime.
    // CLEAR loadOp handles the per-frame reset to far plane, so we don't need
    // to transition it again.
    {
        VkCommandBufferAllocateInfo cbAi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbAi.commandPool        = uploadPool;
        cbAi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbAi.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateCommandBuffers(gpu.device, &cbAi, &cmd));

        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
        transitionImage(cmd, depth.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sub.commandBufferCount = 1;
        sub.pCommandBuffers    = &cmd;
        VK_CHECK(vkQueueSubmit(gpu.graphicsQueue, 1, &sub, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(gpu.graphicsQueue));
        vkFreeCommandBuffers(gpu.device, uploadPool, 1, &cmd);
    }

    // ---- Upload terrain VB + IB ----
    const VkDeviceSize vbBytes = sizeof(Vertex)   * mesh.vertices.size();
    const VkDeviceSize ibBytes = sizeof(uint32_t) * mesh.indices.size();
    Buffer terrainVB = createBufferGPU(gpu.allocator, vbBytes,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Buffer terrainIB = createBufferGPU(gpu.allocator, ibBytes,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT  | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    uploadToBuffer(gpu.device, gpu.graphicsQueue, uploadPool, gpu.allocator,
                   mesh.vertices.data(), vbBytes, terrainVB.buffer);
    uploadToBuffer(gpu.device, gpu.graphicsQueue, uploadPool, gpu.allocator,
                   mesh.indices.data(),  ibBytes, terrainIB.buffer);
    vkDestroyCommandPool(gpu.device, uploadPool, nullptr);
    std::printf("gpu: terrain uploaded — VB %.1f MB, IB %.1f MB\n",
                vbBytes / (1024.0 * 1024.0), ibBytes / (1024.0 * 1024.0));

    // ---- Camera UBO + per-frame descriptor sets ----
    std::array<Buffer, kFramesInFlight> cameraUbos{};
    for (auto& b : cameraUbos) {
        b = createBufferHostMapped(gpu.allocator, sizeof(CameraUBO),
                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    }

    VkDescriptorSetLayoutBinding camBinding{};
    camBinding.binding         = 0;
    camBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    camBinding.descriptorCount = 1;
    camBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo dslCi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCi.bindingCount = 1;
    dslCi.pBindings    = &camBinding;
    VkDescriptorSetLayout cameraSetLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(gpu.device, &dslCi, nullptr, &cameraSetLayout));

    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = kFramesInFlight;
    VkDescriptorPoolCreateInfo dpCi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCi.maxSets       = kFramesInFlight;
    dpCi.poolSizeCount = 1;
    dpCi.pPoolSizes    = &poolSize;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(gpu.device, &dpCi, nullptr, &descPool));

    std::array<VkDescriptorSet, kFramesInFlight> cameraDescSets{};
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = descPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &cameraSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(gpu.device, &ai, &cameraDescSets[i]));

        VkDescriptorBufferInfo bi{};
        bi.buffer = cameraUbos[i].buffer;
        bi.range  = sizeof(CameraUBO);

        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = cameraDescSets[i];
        w.dstBinding      = 0;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo     = &bi;
        vkUpdateDescriptorSets(gpu.device, 1, &w, 0, nullptr);
    }

    // ---- Terrain pipeline ----
    VkShaderModule vertModule = createShaderModule(gpu.device, terrain_vert_spv, sizeof(terrain_vert_spv));
    VkShaderModule fragModule = createShaderModule(gpu.device, terrain_frag_spv, sizeof(terrain_frag_spv));

    VkPipelineLayoutCreateInfo plCi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCi.setLayoutCount = 1;
    plCi.pSetLayouts    = &cameraSetLayout;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(gpu.device, &plCi, nullptr, &pipelineLayout));

    VkPipelineShaderStageCreateInfo stages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
    };
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName  = "main";
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName  = "main";

    VkVertexInputBindingDescription vbBinding{};
    vbBinding.binding   = 0;
    vbBinding.stride    = sizeof(Vertex);
    vbBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vbAttrs[3]{};
    vbAttrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
    vbAttrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
    vbAttrs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv)};

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &vbBinding;
    vi.vertexAttributeDescriptionCount = 3;
    vi.pVertexAttributeDescriptions    = vbAttrs;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_BACK_BIT;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments    = &cba;

    VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynStates;

    VkPipelineRenderingCreateInfo prCi{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    prCi.colorAttachmentCount    = 1;
    prCi.pColorAttachmentFormats = &sc.imageFormat;
    prCi.depthAttachmentFormat   = kDepthFormat;

    VkGraphicsPipelineCreateInfo gpCi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpCi.pNext               = &prCi;
    gpCi.stageCount          = 2;
    gpCi.pStages             = stages;
    gpCi.pVertexInputState   = &vi;
    gpCi.pInputAssemblyState = &ia;
    gpCi.pViewportState      = &vp;
    gpCi.pRasterizationState = &rs;
    gpCi.pMultisampleState   = &ms;
    gpCi.pDepthStencilState  = &ds;
    gpCi.pColorBlendState    = &cb;
    gpCi.pDynamicState       = &dyn;
    gpCi.layout              = pipelineLayout;
    VkPipeline terrainPipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(gpu.device, VK_NULL_HANDLE, 1, &gpCi, nullptr, &terrainPipeline));

    // Shader modules can be destroyed once the pipeline is built.
    vkDestroyShaderModule(gpu.device, fragModule, nullptr);
    vkDestroyShaderModule(gpu.device, vertModule, nullptr);

    std::printf("gpu: terrain pipeline ready\n");

    std::printf("alpine-sun: window open. Left-drag to orbit, scroll to zoom, ESC to quit.\n");

    glfwSetScrollCallback(window, scrollCallback);

    // Dawn-on-snow tint: pale warm pink. Distinctive enough to confirm the
    // clear actually happened (vs. driver-default black).
    constexpr VkClearColorValue kClearColor = {{0.95f, 0.82f, 0.78f, 1.0f}};

    uint32_t frameIndex = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // ---- Input → orbit camera ----
        {
            double mx = 0.0, my = 0.0;
            glfwGetCursorPos(window, &mx, &my);
            const bool leftHeld = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

            if (leftHeld && g_input.leftDown) {
                const double dx = mx - g_input.lastX;
                const double dy = my - g_input.lastY;
                // 0.3° per pixel feels right for this scale.
                constexpr float kSensitivity = 0.005f;
                g_camera.yaw   -= static_cast<float>(dx) * kSensitivity;
                g_camera.pitch += static_cast<float>(dy) * kSensitivity;
                // Clamp pitch just shy of the poles to avoid lookAt singularity.
                constexpr float kPitchLimit = 1.4835f;  // ~85°
                g_camera.pitch = std::clamp(g_camera.pitch, -kPitchLimit, kPitchLimit);
            }
            g_input.leftDown = leftHeld;
            g_input.lastX    = mx;
            g_input.lastY    = my;

            if (g_input.scrollPending != 0.0) {
                // Each scroll tick scales distance by 0.9 (in) or 1/0.9 (out).
                const float zoom = std::pow(0.9f, static_cast<float>(g_input.scrollPending));
                g_camera.distance = std::clamp(g_camera.distance * zoom,
                                               g_camera.minDistance, g_camera.maxDistance);
                g_input.scrollPending = 0.0;
            }
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

        VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depthAttachment.imageView                  = depth.view;
        depthAttachment.imageLayout                = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp                     = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp                    = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue.depthStencil    = {1.0f, 0};

        VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
        renderingInfo.renderArea           = {{0, 0}, sc.extent};
        renderingInfo.layerCount           = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments    = &colorAttachment;
        renderingInfo.pDepthAttachment     = &depthAttachment;

        // ---- Camera matrices from orbit state ----
        const glm::vec3 eye    = orbitEye(g_camera);
        const glm::vec3 center = g_camera.target;
        const glm::vec3 upVec  = glm::vec3(0.0f, 0.0f, 1.0f);

        const float aspect = static_cast<float>(sc.extent.width)
                           / static_cast<float>(sc.extent.height);

        CameraUBO cam{};
        cam.view = glm::lookAt(eye, center, upVec);
        cam.proj = glm::perspective(glm::radians(60.0f), aspect, 100.0f, 400000.0f);
        cam.proj[1][1] *= -1.0f;             // Vulkan NDC y-flip
        cam.viewProj = cam.proj * cam.view;
        std::memcpy(cameraUbos[frameIndex].mapped, &cam, sizeof(cam));

        vkCmdBeginRendering(frame.cmd, &renderingInfo);

        VkViewport viewport{};
        viewport.x        = 0.0f;
        viewport.y        = 0.0f;
        viewport.width    = static_cast<float>(sc.extent.width);
        viewport.height   = static_cast<float>(sc.extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(frame.cmd, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, sc.extent};
        vkCmdSetScissor(frame.cmd, 0, 1, &scissor);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, terrainPipeline);
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &cameraDescSets[frameIndex], 0, nullptr);

        VkDeviceSize vbOffset = 0;
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &terrainVB.buffer, &vbOffset);
        vkCmdBindIndexBuffer  (frame.cmd, terrainIB.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed      (frame.cmd, static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);

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

    vkDestroyPipeline(gpu.device, terrainPipeline, nullptr);
    vkDestroyPipelineLayout(gpu.device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(gpu.device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(gpu.device, cameraSetLayout, nullptr);
    for (auto& b : cameraUbos) vmaDestroyBuffer(gpu.allocator, b.buffer, b.allocation);
    vmaDestroyBuffer(gpu.allocator, terrainIB.buffer, terrainIB.allocation);
    vmaDestroyBuffer(gpu.allocator, terrainVB.buffer, terrainVB.allocation);
    vkDestroyImageView(gpu.device, depth.view, nullptr);
    vmaDestroyImage(gpu.allocator, depth.image, depth.allocation);
    for (auto& f : frames) forfun::destroyFrameContext(gpu, f);
    forfun::destroySwapchain(gpu, sc);
    forfun::destroyDevice(gpu);
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
