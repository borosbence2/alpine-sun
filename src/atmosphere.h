#pragma once

// Bundles the precomputed atmospheric LUTs and the sun-related terrain bakes
// (DEM heightmap, horizon map, sun-hours accumulator, Hillaire transmittance
// + sky-view) into a single subsystem. Each render-side descriptor set in
// main.cpp wires in the *SampleView + *Sampler fields it cares about; the
// bake command pool + compute pipelines live entirely inside this module.

#include "device.h"        // forfun::Device
#include "vk_helpers.h"    // Buffer, VmaAllocation
#include "terrain_mesh.h"  // terrain::Mesh
#include "dem_loader.h"    // dem::Tile

#include <glm/glm.hpp>

#include <cstdint>

struct Route;
struct SunUi;
struct SunHoursUi;

// Horizon map: per-texel apparent-elevation profile of the surrounding ridges.
// 512² × 32 bins × 2 bytes ≈ 16 MB. Baked once at startup.
constexpr uint32_t kHorizonMapSize        = 512;
constexpr uint32_t kHorizonBins           = 32;
constexpr float    kHorizonStepDistMeters = 150.0f;
constexpr int      kHorizonMaxSteps       = 128;

// Sun-hours map: per-texel daylight hours accumulator. Step count drives
// the per-day sample budget; 256 covers 5-min steps over 24 h.
constexpr uint32_t kSunHoursMapSize = 512;
constexpr uint32_t kMaxSunSamples   = 256;

// Hillaire-style atmosphere LUTs.
constexpr uint32_t kTransmittanceLutW = 256;
constexpr uint32_t kTransmittanceLutH = 64;
constexpr uint32_t kSkyViewLutW       = 192;
constexpr uint32_t kSkyViewLutH       = 108;

struct AtmosphereSystem {
    // --- Heightmap (DEM as R32F, input to horizon bake) ---
    VkImage       heightMapImage   = VK_NULL_HANDLE;
    VkImageView   heightMapView    = VK_NULL_HANDLE;
    VmaAllocation heightMapAlloc   = VK_NULL_HANDLE;
    VkSampler     heightMapSampler = VK_NULL_HANDLE;

    // --- Horizon map (R16F 2D array, one-shot bake) ---
    VkImage       horizonImage       = VK_NULL_HANDLE;
    VkImageView   horizonSampleView  = VK_NULL_HANDLE;
    VkImageView   horizonStorageView = VK_NULL_HANDLE;
    VmaAllocation horizonAlloc       = VK_NULL_HANDLE;
    VkSampler     horizonSampler     = VK_NULL_HANDLE;

    // --- Sun-hours accumulator (R32F, re-baked on date/loc/step change) ---
    VkImage       sunHoursImage       = VK_NULL_HANDLE;
    VkImageView   sunHoursStorageView = VK_NULL_HANDLE;
    VkImageView   sunHoursSampleView  = VK_NULL_HANDLE;
    VmaAllocation sunHoursAlloc       = VK_NULL_HANDLE;
    VkSampler     sunHoursSampler     = VK_NULL_HANDLE;
    Buffer        sunSamplesBuf{};       // host-mapped UBO of per-step sun dirs
    Buffer        sunHoursReadback{};    // host-mapped copy of the just-baked map
    VkDescriptorSetLayout sunHoursSetLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      sunHoursDescPool       = VK_NULL_HANDLE;
    VkDescriptorSet       sunHoursDescSet        = VK_NULL_HANDLE;
    VkPipelineLayout      sunHoursPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            sunHoursPipeline       = VK_NULL_HANDLE;

    // --- Transmittance LUT (RGBA16F, one-shot bake) ---
    VkImage       transImage       = VK_NULL_HANDLE;
    VkImageView   transStorageView = VK_NULL_HANDLE;
    VkImageView   transSampleView  = VK_NULL_HANDLE;
    VmaAllocation transAlloc       = VK_NULL_HANDLE;
    VkSampler     transSampler     = VK_NULL_HANDLE;

    // --- Sky-view LUT (RGBA16F, re-baked on sun-direction change) ---
    VkImage       skyViewImage       = VK_NULL_HANDLE;
    VkImageView   skyViewStorageView = VK_NULL_HANDLE;
    VkImageView   skyViewSampleView  = VK_NULL_HANDLE;
    VmaAllocation skyViewAlloc       = VK_NULL_HANDLE;
    VkSampler     skyViewSampler     = VK_NULL_HANDLE;
    VkDescriptorSetLayout skyViewSetLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      skyViewDescPool       = VK_NULL_HANDLE;
    VkDescriptorSet       skyViewDescSet        = VK_NULL_HANDLE;
    VkPipelineLayout      skyViewPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            skyViewPipeline       = VK_NULL_HANDLE;

    // --- Shared command pool for all atmospheric bakes ---
    VkCommandPool bakePool = VK_NULL_HANDLE;

    // Terrain AABB captured at creation time, used as bake push-constant data.
    glm::vec3 aabbMin{0.0f};
    glm::vec3 aabbMax{0.0f};
};

// Uploads the DEM, creates every atmosphere image/pipeline/sampler, and runs
// the one-shot horizon + transmittance bakes. The sun-hours and sky-view
// LUTs are NOT baked here — call bakeSunHours / bakeSkyView at least once
// before any render submission samples them.
AtmosphereSystem createAtmosphereSystem(const forfun::Device& gpu,
                                        const terrain::Mesh&  mesh,
                                        const dem::Tile&      tile);

void destroyAtmosphereSystem(const forfun::Device& gpu, AtmosphereSystem& atm);

// Dispatches the sun-hours compute, blocks until done, then refreshes
// `route.sunHours` from the freshly read-back accumulator. Returns the
// number of sun samples actually used (clamped to kMaxSunSamples).
int bakeSunHours(const forfun::Device& gpu,
                 AtmosphereSystem&     atm,
                 const SunUi&          sun,
                 const SunHoursUi&     sunHoursUi,
                 Route&                route);

// Dispatches the sky-view compute for a given sun direction and observer
// altitude, blocks until done. Caller decides when to re-run (e.g. on sun
// movement past a small dot-product threshold).
void bakeSkyView(const forfun::Device& gpu,
                 AtmosphereSystem&     atm,
                 const glm::vec3&      sunDir,
                 float                 observerAltM);

// Populates `route.sunHours` from the cached sun-hours readback (filled by
// the last bakeSunHours). Used when a fresh GPX is loaded between bakes.
void updateRouteSunHoursFromAtmosphere(const AtmosphereSystem& atm, Route& route);
