#include "atmosphere.h"

#include "app_state.h"
#include "sun_driver.h"
#include "types.h"
#include "vk_helpers.h"

#include "horizon_map.comp.h"
#include "sun_hours.comp.h"
#include "transmittance.comp.h"
#include "sky_view.comp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

// Matches the sun_hours.comp UBO layout in std140. The trailing pad keeps
// the struct size a multiple of 16 bytes so we can sizeof() it directly.
struct alignas(16) SunSamplesUBO {
    glm::vec4 dirs[kMaxSunSamples];
    int       count;
    float     hoursPerStep;
    float     _pad0;
    float     _pad1;
};

// Populates `out` with one calendar day's worth of sun directions, sampled
// every `stepMinutes` from local midnight to midnight. Below-horizon samples
// (.z < 0) are stored as-is — the compute shader filters them out per thread.
// Returns the number of populated entries (clamped to kMaxSunSamples).
int fillSunSamples(SunSamplesUBO& out, const SunUi& sun, float stepMinutes) {
    const float clampedStep = std::max(1.0f, stepMinutes);
    int stepsPerDay = static_cast<int>(std::ceil(24.0f * 60.0f / clampedStep));
    if (stepsPerDay > static_cast<int>(kMaxSunSamples)) stepsPerDay = kMaxSunSamples;
    const float hoursPerStep = 24.0f / static_cast<float>(stepsPerDay);
    out.count        = stepsPerDay;
    out.hoursPerStep = hoursPerStep;
    out._pad0        = 0.0f;
    out._pad1        = 0.0f;
    for (int i = 0; i < stepsPerDay; ++i) {
        const float localHour = (static_cast<float>(i) + 0.5f) * hoursPerStep;
        const sun::Sample s   = sun::compute(sun.latDeg, sun.lonDeg,
                                             sun.year, sun.month, sun.day,
                                             localHour, sun.tzOffsetH);
        out.dirs[i] = glm::vec4(s.directionToSun, 0.0f);
    }
    for (int i = stepsPerDay; i < static_cast<int>(kMaxSunSamples); ++i) {
        out.dirs[i] = glm::vec4(0.0f);
    }
    return stepsPerDay;
}

// Populates `route.sunHours` by looking up each waypoint's world-XY in the
// CPU-side sun-hours grid. Cheap — one nearest-neighbour read per waypoint.
void updateRouteSunHoursFromReadback(Route& route,
                                     const float* sunHoursData,
                                     int sunHoursSize,
                                     const glm::vec3& aabbMin,
                                     const glm::vec3& aabbMax) {
    if (route.vertexCount == 0) {
        route.sunHours.clear();
        return;
    }
    const glm::vec2 extent(aabbMax.x - aabbMin.x, aabbMax.y - aabbMin.y);
    route.sunHours.resize(route.vertexCount);
    for (uint32_t i = 0; i < route.vertexCount; ++i) {
        const glm::vec3& p = route.enu[i];
        const float u = (p.x - aabbMin.x) / extent.x;
        const float v = (aabbMax.y - p.y) / extent.y;       // N→S
        const int x = std::clamp(int(u * sunHoursSize), 0, sunHoursSize - 1);
        const int y = std::clamp(int(v * sunHoursSize), 0, sunHoursSize - 1);
        route.sunHours[i] = sunHoursData[y * sunHoursSize + x];
    }
}

// Submit + wait helper. Sun-hours / sky-view rebakes are infrequent and
// small enough (~ms) that the brief queue stall is invisible.
void submitAndWait(const forfun::Device& gpu, VkCommandBuffer cmd) {
    VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    sub.commandBufferCount = 1;
    sub.pCommandBuffers    = &cmd;
    VK_CHECK(vkQueueSubmit(gpu.graphicsQueue, 1, &sub, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(gpu.graphicsQueue));
}

VkCommandBuffer beginOneShotCmd(const forfun::Device& gpu, VkCommandPool pool) {
    VkCommandBufferAllocateInfo cbAi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbAi.commandPool        = pool;
    cbAi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(gpu.device, &cbAi, &cmd));

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    return cmd;
}

// ---- Heightmap (R32F) ----
void createHeightMap(const forfun::Device& gpu,
                     const dem::Tile&      tile,
                     AtmosphereSystem&     atm) {
    VkImageCreateInfo imageCi{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageCi.imageType     = VK_IMAGE_TYPE_2D;
    imageCi.format        = VK_FORMAT_R32_SFLOAT;
    imageCi.extent        = {tile.width, tile.height, 1};
    imageCi.mipLevels     = 1;
    imageCi.arrayLayers   = 1;
    imageCi.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageCi.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageCi.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCi.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo allocCi{};
    allocCi.usage = VMA_MEMORY_USAGE_AUTO;
    VK_CHECK(vmaCreateImage(gpu.allocator, &imageCi, &allocCi,
                            &atm.heightMapImage, &atm.heightMapAlloc, nullptr));

    VkImageViewCreateInfo viewCi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewCi.image            = atm.heightMapImage;
    viewCi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewCi.format           = VK_FORMAT_R32_SFLOAT;
    viewCi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(gpu.device, &viewCi, nullptr, &atm.heightMapView));

    // Linear filter so the compute shader gets bilinear samples between
    // DEM texels — modestly smoother bake than nearest.
    VkSamplerCreateInfo sCi{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sCi.magFilter    = VK_FILTER_LINEAR;
    sCi.minFilter    = VK_FILTER_LINEAR;
    sCi.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sCi.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sCi.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sCi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(gpu.device, &sCi, nullptr, &atm.heightMapSampler));
}

void uploadHeightMap(const forfun::Device& gpu,
                     const dem::Tile&      tile,
                     AtmosphereSystem&     atm) {
    const VkDeviceSize demBytes =
        static_cast<VkDeviceSize>(tile.width) * tile.height * sizeof(float);

    VkBufferCreateInfo stagingCi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    stagingCi.size  = demBytes;
    stagingCi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo stagingAllocCi{};
    stagingAllocCi.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocCi.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                         | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer          stagingBuf  = VK_NULL_HANDLE;
    VmaAllocation     stagingAlloc = VK_NULL_HANDLE;
    VmaAllocationInfo stagingInfo{};
    VK_CHECK(vmaCreateBuffer(gpu.allocator, &stagingCi, &stagingAllocCi,
                             &stagingBuf, &stagingAlloc, &stagingInfo));
    std::memcpy(stagingInfo.pMappedData, tile.elevation.data(),
                static_cast<size_t>(demBytes));

    VkCommandBuffer cmd = beginOneShotCmd(gpu, atm.bakePool);
    transitionImage(cmd, atm.heightMapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {tile.width, tile.height, 1};
    vkCmdCopyBufferToImage(cmd, stagingBuf, atm.heightMapImage,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    transitionImage(cmd, atm.heightMapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    VK_CHECK(vkEndCommandBuffer(cmd));
    submitAndWait(gpu, cmd);
    vkFreeCommandBuffers(gpu.device, atm.bakePool, 1, &cmd);
    vmaDestroyBuffer(gpu.allocator, stagingBuf, stagingAlloc);
    std::printf("gpu: heightmap uploaded — %u x %u R32F (%.1f MB)\n",
                tile.width, tile.height, demBytes / (1024.0 * 1024.0));
}

// ---- Horizon map (R16F 2D array, layer = azimuth bin) ----
void createHorizonMap(const forfun::Device& gpu, AtmosphereSystem& atm) {
    VkImageCreateInfo imageCi{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageCi.imageType     = VK_IMAGE_TYPE_2D;
    imageCi.format        = VK_FORMAT_R16_SFLOAT;
    imageCi.extent        = {kHorizonMapSize, kHorizonMapSize, 1};
    imageCi.mipLevels     = 1;
    imageCi.arrayLayers   = kHorizonBins;
    imageCi.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageCi.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageCi.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCi.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo allocCi{};
    allocCi.usage = VMA_MEMORY_USAGE_AUTO;
    VK_CHECK(vmaCreateImage(gpu.allocator, &imageCi, &allocCi,
                            &atm.horizonImage, &atm.horizonAlloc, nullptr));

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image            = atm.horizonImage;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vci.format           = VK_FORMAT_R16_SFLOAT;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, kHorizonBins};
    VK_CHECK(vkCreateImageView(gpu.device, &vci, nullptr, &atm.horizonSampleView));
    VK_CHECK(vkCreateImageView(gpu.device, &vci, nullptr, &atm.horizonStorageView));

    // Linear filter for spatial XY, but layer (bin) sampling is always
    // nearest in Vulkan — the frag shader manually interpolates between
    // adjacent bins instead.
    VkSamplerCreateInfo sCi{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sCi.magFilter    = VK_FILTER_LINEAR;
    sCi.minFilter    = VK_FILTER_LINEAR;
    sCi.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sCi.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sCi.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sCi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(gpu.device, &sCi, nullptr, &atm.horizonSampler));
}

void bakeHorizonMap(const forfun::Device& gpu, AtmosphereSystem& atm) {
    VkDescriptorSetLayoutBinding bakeBindings[2]{};
    bakeBindings[0].binding         = 0;
    bakeBindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bakeBindings[0].descriptorCount = 1;
    bakeBindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    bakeBindings[1].binding         = 1;
    bakeBindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bakeBindings[1].descriptorCount = 1;
    bakeBindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslCi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCi.bindingCount = 2;
    dslCi.pBindings    = bakeBindings;
    VkDescriptorSetLayout bakeSetLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(gpu.device, &dslCi, nullptr, &bakeSetLayout));

    VkDescriptorPoolSize bakePoolSizes[2]{};
    bakePoolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bakePoolSizes[0].descriptorCount = 1;
    bakePoolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bakePoolSizes[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpCi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCi.maxSets       = 1;
    dpCi.poolSizeCount = 2;
    dpCi.pPoolSizes    = bakePoolSizes;
    VkDescriptorPool bakeDescPool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(gpu.device, &dpCi, nullptr, &bakeDescPool));

    VkDescriptorSet bakeDescSet = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = bakeDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &bakeSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(gpu.device, &ai, &bakeDescSet));

    VkDescriptorImageInfo heightInfo{};
    heightInfo.sampler     = atm.heightMapSampler;
    heightInfo.imageView   = atm.heightMapView;
    heightInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo horizonStoreInfo{};
    horizonStoreInfo.imageView   = atm.horizonStorageView;
    horizonStoreInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet bakeWrites[2]{};
    bakeWrites[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    bakeWrites[0].dstSet          = bakeDescSet;
    bakeWrites[0].dstBinding      = 0;
    bakeWrites[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bakeWrites[0].descriptorCount = 1;
    bakeWrites[0].pImageInfo      = &heightInfo;
    bakeWrites[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    bakeWrites[1].dstSet          = bakeDescSet;
    bakeWrites[1].dstBinding      = 1;
    bakeWrites[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bakeWrites[1].descriptorCount = 1;
    bakeWrites[1].pImageInfo      = &horizonStoreInfo;
    vkUpdateDescriptorSets(gpu.device, 2, bakeWrites, 0, nullptr);

    VkPushConstantRange bakePush{};
    bakePush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bakePush.size       = sizeof(glm::vec4) * 2;  // aabbMinMax + params

    VkPipelineLayoutCreateInfo bakePlCi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    bakePlCi.setLayoutCount         = 1;
    bakePlCi.pSetLayouts            = &bakeSetLayout;
    bakePlCi.pushConstantRangeCount = 1;
    bakePlCi.pPushConstantRanges    = &bakePush;
    VkPipelineLayout bakePipelineLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(gpu.device, &bakePlCi, nullptr, &bakePipelineLayout));

    VkShaderModule bakeModule = createShaderModule(
        gpu.device, horizon_map_comp_spv, sizeof(horizon_map_comp_spv));

    VkPipelineShaderStageCreateInfo bakeStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    bakeStage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    bakeStage.module = bakeModule;
    bakeStage.pName  = "main";

    VkComputePipelineCreateInfo bakeCpCi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    bakeCpCi.stage  = bakeStage;
    bakeCpCi.layout = bakePipelineLayout;
    VkPipeline bakePipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(gpu.device, VK_NULL_HANDLE, 1, &bakeCpCi, nullptr, &bakePipeline));
    vkDestroyShaderModule(gpu.device, bakeModule, nullptr);

    VkCommandBuffer cmd = beginOneShotCmd(gpu, atm.bakePool);

    transitionImageRange(cmd, atm.horizonImage, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         0, 1, 0, kHorizonBins);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bakePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            bakePipelineLayout, 0, 1, &bakeDescSet, 0, nullptr);

    struct {
        glm::vec4 aabbMinMax;
        glm::vec4 params;
    } push{};
    push.aabbMinMax = glm::vec4(atm.aabbMin.x, atm.aabbMin.y,
                                atm.aabbMax.x, atm.aabbMax.y);
    push.params     = glm::vec4(kHorizonStepDistMeters,
                                static_cast<float>(kHorizonMaxSteps),
                                0.0f, 0.0f);
    vkCmdPushConstants(cmd, bakePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);

    const uint32_t groupsX = (kHorizonMapSize + 7) / 8;
    const uint32_t groupsY = (kHorizonMapSize + 7) / 8;
    vkCmdDispatch(cmd, groupsX, groupsY, kHorizonBins);

    transitionImageRange(cmd, atm.horizonImage, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                         0, 1, 0, kHorizonBins);

    VK_CHECK(vkEndCommandBuffer(cmd));
    submitAndWait(gpu, cmd);
    vkFreeCommandBuffers(gpu.device, atm.bakePool, 1, &cmd);

    vkDestroyPipeline(gpu.device, bakePipeline, nullptr);
    vkDestroyPipelineLayout(gpu.device, bakePipelineLayout, nullptr);
    vkDestroyDescriptorPool(gpu.device, bakeDescPool, nullptr);
    vkDestroyDescriptorSetLayout(gpu.device, bakeSetLayout, nullptr);
    std::printf("gpu: horizon map baked — %u² × %u bins, %.0fm step × %d steps\n",
                kHorizonMapSize, kHorizonBins,
                static_cast<double>(kHorizonStepDistMeters), kHorizonMaxSteps);
}

// ---- Sun-hours accumulator + persistent compute pipeline ----
void createSunHoursResources(const forfun::Device& gpu, AtmosphereSystem& atm) {
    VkImageCreateInfo imageCi{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageCi.imageType     = VK_IMAGE_TYPE_2D;
    imageCi.format        = VK_FORMAT_R32_SFLOAT;
    imageCi.extent        = {kSunHoursMapSize, kSunHoursMapSize, 1};
    imageCi.mipLevels     = 1;
    imageCi.arrayLayers   = 1;
    imageCi.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageCi.tiling        = VK_IMAGE_TILING_OPTIMAL;
    // TRANSFER_SRC so we can copy back to host every bake (waypoint table)
    // and copy single texels for right-click sampling.
    imageCi.usage         = VK_IMAGE_USAGE_STORAGE_BIT
                          | VK_IMAGE_USAGE_SAMPLED_BIT
                          | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageCi.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo allocCi{};
    allocCi.usage = VMA_MEMORY_USAGE_AUTO;
    VK_CHECK(vmaCreateImage(gpu.allocator, &imageCi, &allocCi,
                            &atm.sunHoursImage, &atm.sunHoursAlloc, nullptr));

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image            = atm.sunHoursImage;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vci.format           = VK_FORMAT_R32_SFLOAT;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(gpu.device, &vci, nullptr, &atm.sunHoursStorageView));
    VK_CHECK(vkCreateImageView(gpu.device, &vci, nullptr, &atm.sunHoursSampleView));

    VkSamplerCreateInfo sCi{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sCi.magFilter    = VK_FILTER_LINEAR;
    sCi.minFilter    = VK_FILTER_LINEAR;
    sCi.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sCi.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sCi.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sCi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(gpu.device, &sCi, nullptr, &atm.sunHoursSampler));

    atm.sunSamplesBuf = createBufferHostMapped(
        gpu.allocator, sizeof(SunSamplesUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    const VkDeviceSize kSunHoursReadbackBytes =
        static_cast<VkDeviceSize>(kSunHoursMapSize) * kSunHoursMapSize * sizeof(float);
    atm.sunHoursReadback = createBufferHostMapped(
        gpu.allocator, kSunHoursReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Persistent descriptor-set + pipeline (re-dispatched on date change).
    VkDescriptorSetLayoutBinding bindings[3]{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding         = 2;
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslCi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCi.bindingCount = 3;
    dslCi.pBindings    = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(gpu.device, &dslCi, nullptr, &atm.sunHoursSetLayout));

    VkDescriptorPoolSize poolSizes[3]{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1;
    poolSizes[2].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[2].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpCi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCi.maxSets       = 1;
    dpCi.poolSizeCount = 3;
    dpCi.pPoolSizes    = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(gpu.device, &dpCi, nullptr, &atm.sunHoursDescPool));

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = atm.sunHoursDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &atm.sunHoursSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(gpu.device, &ai, &atm.sunHoursDescSet));

    VkDescriptorImageInfo horizonReadInfo{};
    horizonReadInfo.sampler     = atm.horizonSampler;
    horizonReadInfo.imageView   = atm.horizonSampleView;
    horizonReadInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo sunHoursStoreInfo{};
    sunHoursStoreInfo.imageView   = atm.sunHoursStorageView;
    sunHoursStoreInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorBufferInfo samplesInfo{};
    samplesInfo.buffer = atm.sunSamplesBuf.buffer;
    samplesInfo.range  = sizeof(SunSamplesUBO);

    VkWriteDescriptorSet writes[3]{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = atm.sunHoursDescSet;
    writes[0].dstBinding      = 0;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo      = &horizonReadInfo;
    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = atm.sunHoursDescSet;
    writes[1].dstBinding      = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo      = &sunHoursStoreInfo;
    writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet          = atm.sunHoursDescSet;
    writes[2].dstBinding      = 2;
    writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].descriptorCount = 1;
    writes[2].pBufferInfo     = &samplesInfo;
    vkUpdateDescriptorSets(gpu.device, 3, writes, 0, nullptr);

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size       = sizeof(glm::vec4);  // aabbMinMax

    VkPipelineLayoutCreateInfo plCi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCi.setLayoutCount         = 1;
    plCi.pSetLayouts            = &atm.sunHoursSetLayout;
    plCi.pushConstantRangeCount = 1;
    plCi.pPushConstantRanges    = &pushRange;
    VK_CHECK(vkCreatePipelineLayout(gpu.device, &plCi, nullptr, &atm.sunHoursPipelineLayout));

    VkShaderModule mod = createShaderModule(gpu.device,
        sun_hours_comp_spv, sizeof(sun_hours_comp_spv));
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName  = "main";
    VkComputePipelineCreateInfo cpCi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpCi.stage  = stage;
    cpCi.layout = atm.sunHoursPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(gpu.device, VK_NULL_HANDLE, 1, &cpCi, nullptr, &atm.sunHoursPipeline));
    vkDestroyShaderModule(gpu.device, mod, nullptr);
}

// ---- Transmittance LUT (one-shot bake) ----
void createTransmittanceLut(const forfun::Device& gpu, AtmosphereSystem& atm) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
    ci.extent        = {kTransmittanceLutW, kTransmittanceLutH, 1};
    ci.mipLevels     = 1;
    ci.arrayLayers   = 1;
    ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ci.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo ac{};
    ac.usage = VMA_MEMORY_USAGE_AUTO;
    VK_CHECK(vmaCreateImage(gpu.allocator, &ci, &ac, &atm.transImage, &atm.transAlloc, nullptr));

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image            = atm.transImage;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vci.format           = VK_FORMAT_R16G16B16A16_SFLOAT;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(gpu.device, &vci, nullptr, &atm.transStorageView));
    VK_CHECK(vkCreateImageView(gpu.device, &vci, nullptr, &atm.transSampleView));

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(gpu.device, &sci, nullptr, &atm.transSampler));
}

void bakeTransmittanceLut(const forfun::Device& gpu, AtmosphereSystem& atm) {
    VkDescriptorSetLayoutBinding b{};
    b.binding         = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslCi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCi.bindingCount = 1;
    dslCi.pBindings    = &b;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(gpu.device, &dslCi, nullptr, &dsl));

    VkDescriptorPoolSize ps{};
    ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpCi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCi.maxSets       = 1;
    dpCi.poolSizeCount = 1;
    dpCi.pPoolSizes    = &ps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(gpu.device, &dpCi, nullptr, &pool));

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &dsl;
    VK_CHECK(vkAllocateDescriptorSets(gpu.device, &ai, &set));

    VkDescriptorImageInfo storeInfo{};
    storeInfo.imageView   = atm.transStorageView;
    storeInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet          = set;
    w.dstBinding      = 0;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w.descriptorCount = 1;
    w.pImageInfo      = &storeInfo;
    vkUpdateDescriptorSets(gpu.device, 1, &w, 0, nullptr);

    VkPipelineLayoutCreateInfo plCi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCi.setLayoutCount = 1;
    plCi.pSetLayouts    = &dsl;
    VkPipelineLayout plLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(gpu.device, &plCi, nullptr, &plLayout));

    VkShaderModule mod = createShaderModule(gpu.device,
        transmittance_comp_spv, sizeof(transmittance_comp_spv));
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName  = "main";
    VkComputePipelineCreateInfo cpCi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpCi.stage  = stage;
    cpCi.layout = plLayout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(gpu.device, VK_NULL_HANDLE, 1, &cpCi, nullptr, &pipeline));
    vkDestroyShaderModule(gpu.device, mod, nullptr);

    VkCommandBuffer cmd = beginOneShotCmd(gpu, atm.bakePool);
    transitionImage(cmd, atm.transImage, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            plLayout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(cmd, (kTransmittanceLutW + 7) / 8, (kTransmittanceLutH + 7) / 8, 1);
    transitionImage(cmd, atm.transImage, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                        | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    VK_CHECK(vkEndCommandBuffer(cmd));
    submitAndWait(gpu, cmd);
    vkFreeCommandBuffers(gpu.device, atm.bakePool, 1, &cmd);

    vkDestroyPipeline(gpu.device, pipeline, nullptr);
    vkDestroyPipelineLayout(gpu.device, plLayout, nullptr);
    vkDestroyDescriptorPool(gpu.device, pool, nullptr);
    vkDestroyDescriptorSetLayout(gpu.device, dsl, nullptr);
    std::printf("gpu: transmittance LUT baked — %u x %u RGBA16F\n",
                kTransmittanceLutW, kTransmittanceLutH);
}

// ---- Sky-view LUT image + persistent pipeline ----
void createSkyViewLut(const forfun::Device& gpu, AtmosphereSystem& atm) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
    ci.extent        = {kSkyViewLutW, kSkyViewLutH, 1};
    ci.mipLevels     = 1;
    ci.arrayLayers   = 1;
    ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ci.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo ac{};
    ac.usage = VMA_MEMORY_USAGE_AUTO;
    VK_CHECK(vmaCreateImage(gpu.allocator, &ci, &ac, &atm.skyViewImage, &atm.skyViewAlloc, nullptr));

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image            = atm.skyViewImage;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vci.format           = VK_FORMAT_R16G16B16A16_SFLOAT;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(gpu.device, &vci, nullptr, &atm.skyViewStorageView));
    VK_CHECK(vkCreateImageView(gpu.device, &vci, nullptr, &atm.skyViewSampleView));

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;     // azimuth wraps
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(gpu.device, &sci, nullptr, &atm.skyViewSampler));
}

void createSkyViewPipeline(const forfun::Device& gpu, AtmosphereSystem& atm) {
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslCi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCi.bindingCount = 2;
    dslCi.pBindings    = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(gpu.device, &dslCi, nullptr, &atm.skyViewSetLayout));

    VkDescriptorPoolSize ps[2]{};
    ps[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[0].descriptorCount = 1;
    ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpCi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCi.maxSets       = 1;
    dpCi.poolSizeCount = 2;
    dpCi.pPoolSizes    = ps;
    VK_CHECK(vkCreateDescriptorPool(gpu.device, &dpCi, nullptr, &atm.skyViewDescPool));

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = atm.skyViewDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &atm.skyViewSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(gpu.device, &ai, &atm.skyViewDescSet));

    VkDescriptorImageInfo transInfo{};
    transInfo.sampler     = atm.transSampler;
    transInfo.imageView   = atm.transSampleView;
    transInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo storeInfo{};
    storeInfo.imageView   = atm.skyViewStorageView;
    storeInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet writes[2]{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = atm.skyViewDescSet;
    writes[0].dstBinding      = 0;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo      = &transInfo;
    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = atm.skyViewDescSet;
    writes[1].dstBinding      = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo      = &storeInfo;
    vkUpdateDescriptorSets(gpu.device, 2, writes, 0, nullptr);

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size       = sizeof(glm::vec4) * 2;   // sunDir + observer params

    VkPipelineLayoutCreateInfo plCi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCi.setLayoutCount         = 1;
    plCi.pSetLayouts            = &atm.skyViewSetLayout;
    plCi.pushConstantRangeCount = 1;
    plCi.pPushConstantRanges    = &pcr;
    VK_CHECK(vkCreatePipelineLayout(gpu.device, &plCi, nullptr, &atm.skyViewPipelineLayout));

    VkShaderModule mod = createShaderModule(gpu.device,
        sky_view_comp_spv, sizeof(sky_view_comp_spv));
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName  = "main";
    VkComputePipelineCreateInfo cpCi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpCi.stage  = stage;
    cpCi.layout = atm.skyViewPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(gpu.device, VK_NULL_HANDLE, 1, &cpCi, nullptr, &atm.skyViewPipeline));
    vkDestroyShaderModule(gpu.device, mod, nullptr);
}

} // namespace

AtmosphereSystem createAtmosphereSystem(const forfun::Device& gpu,
                                        const terrain::Mesh&  mesh,
                                        const dem::Tile&      tile) {
    AtmosphereSystem atm;
    atm.aabbMin = mesh.aabbMin;
    atm.aabbMax = mesh.aabbMax;

    // Shared transient pool — survives the program because re-bakes happen
    // whenever the user changes date/location.
    VkCommandPoolCreateInfo poolCi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolCi.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
                            | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCi.queueFamilyIndex = gpu.graphicsFamily;
    VK_CHECK(vkCreateCommandPool(gpu.device, &poolCi, nullptr, &atm.bakePool));

    createHeightMap(gpu, tile, atm);
    uploadHeightMap(gpu, tile, atm);
    createHorizonMap(gpu, atm);
    bakeHorizonMap(gpu, atm);
    createSunHoursResources(gpu, atm);
    createTransmittanceLut(gpu, atm);
    bakeTransmittanceLut(gpu, atm);
    createSkyViewLut(gpu, atm);
    createSkyViewPipeline(gpu, atm);

    return atm;
}

void destroyAtmosphereSystem(const forfun::Device& gpu, AtmosphereSystem& atm) {
    vkDestroyPipeline(gpu.device, atm.skyViewPipeline, nullptr);
    vkDestroyPipelineLayout(gpu.device, atm.skyViewPipelineLayout, nullptr);
    vkDestroyDescriptorPool(gpu.device, atm.skyViewDescPool, nullptr);
    vkDestroyDescriptorSetLayout(gpu.device, atm.skyViewSetLayout, nullptr);
    vkDestroySampler(gpu.device, atm.skyViewSampler, nullptr);
    vkDestroyImageView(gpu.device, atm.skyViewSampleView, nullptr);
    vkDestroyImageView(gpu.device, atm.skyViewStorageView, nullptr);
    vmaDestroyImage(gpu.allocator, atm.skyViewImage, atm.skyViewAlloc);

    vkDestroySampler(gpu.device, atm.transSampler, nullptr);
    vkDestroyImageView(gpu.device, atm.transSampleView, nullptr);
    vkDestroyImageView(gpu.device, atm.transStorageView, nullptr);
    vmaDestroyImage(gpu.allocator, atm.transImage, atm.transAlloc);

    vkDestroyCommandPool(gpu.device, atm.bakePool, nullptr);
    vkDestroyPipeline(gpu.device, atm.sunHoursPipeline, nullptr);
    vkDestroyPipelineLayout(gpu.device, atm.sunHoursPipelineLayout, nullptr);
    vkDestroyDescriptorPool(gpu.device, atm.sunHoursDescPool, nullptr);
    vkDestroyDescriptorSetLayout(gpu.device, atm.sunHoursSetLayout, nullptr);
    vmaDestroyBuffer(gpu.allocator, atm.sunHoursReadback.buffer, atm.sunHoursReadback.allocation);
    vmaDestroyBuffer(gpu.allocator, atm.sunSamplesBuf.buffer, atm.sunSamplesBuf.allocation);
    vkDestroySampler(gpu.device, atm.sunHoursSampler, nullptr);
    vkDestroyImageView(gpu.device, atm.sunHoursStorageView, nullptr);
    vkDestroyImageView(gpu.device, atm.sunHoursSampleView,  nullptr);
    vmaDestroyImage(gpu.allocator, atm.sunHoursImage, atm.sunHoursAlloc);

    vkDestroySampler(gpu.device, atm.horizonSampler, nullptr);
    vkDestroyImageView(gpu.device, atm.horizonStorageView, nullptr);
    vkDestroyImageView(gpu.device, atm.horizonSampleView,  nullptr);
    vmaDestroyImage(gpu.allocator, atm.horizonImage, atm.horizonAlloc);

    vkDestroySampler(gpu.device, atm.heightMapSampler, nullptr);
    vkDestroyImageView(gpu.device, atm.heightMapView, nullptr);
    vmaDestroyImage(gpu.allocator, atm.heightMapImage, atm.heightMapAlloc);
}

int bakeSunHours(const forfun::Device& gpu,
                 AtmosphereSystem&     atm,
                 const SunUi&          sun,
                 const SunHoursUi&     sunHoursUi,
                 Route&                route) {
    SunSamplesUBO* samples = static_cast<SunSamplesUBO*>(atm.sunSamplesBuf.mapped);
    const int n = fillSunSamples(*samples, sun, sunHoursUi.stepMinutes);

    VkCommandBuffer cmd = beginOneShotCmd(gpu, atm.bakePool);

    // Discard previous contents (UNDEFINED) — we overwrite every texel.
    transitionImage(cmd, atm.sunHoursImage, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, atm.sunHoursPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            atm.sunHoursPipelineLayout, 0, 1, &atm.sunHoursDescSet, 0, nullptr);

    glm::vec4 push(atm.aabbMin.x, atm.aabbMin.y, atm.aabbMax.x, atm.aabbMax.y);
    vkCmdPushConstants(cmd, atm.sunHoursPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);

    const uint32_t groupsX = (kSunHoursMapSize + 7) / 8;
    const uint32_t groupsY = (kSunHoursMapSize + 7) / 8;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);

    // Copy the just-baked accumulator to a host-visible buffer so the
    // route waypoint table can look up sun-hours per point without
    // round-tripping the GPU again. We go through TRANSFER_SRC because
    // imageStore writes need to be flushed before COPY can read them.
    transitionImage(cmd, atm.sunHoursImage, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferImageCopy rbRegion{};
    rbRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    rbRegion.imageSubresource.layerCount = 1;
    rbRegion.imageExtent = {kSunHoursMapSize, kSunHoursMapSize, 1};
    vkCmdCopyImageToBuffer(cmd, atm.sunHoursImage,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           atm.sunHoursReadback.buffer, 1, &rbRegion);

    transitionImage(cmd, atm.sunHoursImage, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    VK_CHECK(vkEndCommandBuffer(cmd));
    submitAndWait(gpu, cmd);
    vkFreeCommandBuffers(gpu.device, atm.bakePool, 1, &cmd);

    std::printf("gpu: sun-hours baked — %d samples × %.1f h step\n",
                n, static_cast<double>(samples->hoursPerStep));

    // Pull the freshly-baked accumulator into our route waypoint cache.
    // Safe to read now — the queue wait above guarantees the copy has
    // landed in atm.sunHoursReadback.mapped.
    updateRouteSunHoursFromReadback(
        route,
        static_cast<const float*>(atm.sunHoursReadback.mapped),
        int(kSunHoursMapSize),
        atm.aabbMin, atm.aabbMax);

    return n;
}

void updateRouteSunHoursFromAtmosphere(const AtmosphereSystem& atm, Route& route) {
    updateRouteSunHoursFromReadback(
        route,
        static_cast<const float*>(atm.sunHoursReadback.mapped),
        int(kSunHoursMapSize),
        atm.aabbMin, atm.aabbMax);
}

void bakeSkyView(const forfun::Device& gpu,
                 AtmosphereSystem&     atm,
                 const glm::vec3&      sunDir,
                 float                 observerAltM) {
    VkCommandBuffer cmd = beginOneShotCmd(gpu, atm.bakePool);

    // Discard previous contents — every texel is overwritten.
    transitionImage(cmd, atm.skyViewImage, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, atm.skyViewPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            atm.skyViewPipelineLayout, 0, 1, &atm.skyViewDescSet, 0, nullptr);

    struct {
        glm::vec4 sunDir;
        glm::vec4 observer;
    } push{};
    push.sunDir   = glm::vec4(sunDir, 0.0f);
    push.observer = glm::vec4(observerAltM, 0.0f, 0.0f, 0.0f);
    vkCmdPushConstants(cmd, atm.skyViewPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    vkCmdDispatch(cmd, (kSkyViewLutW + 7) / 8, (kSkyViewLutH + 7) / 8, 1);

    transitionImage(cmd, atm.skyViewImage, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    VK_CHECK(vkEndCommandBuffer(cmd));
    submitAndWait(gpu, cmd);
    vkFreeCommandBuffers(gpu.device, atm.bakePool, 1, &cmd);
}
