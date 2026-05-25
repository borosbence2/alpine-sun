cmake_minimum_required(VERSION 3.22)
project(alpine_sun LANGUAGES CXX)

set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# Our cmake/ dir hosts a FindZLIB shim that routes libtiff's find_package(ZLIB)
# to the CPM-provided zlibstatic target. Must be on the path BEFORE the
# add_subdirectory below so libtiff sees it.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ----------------------------------------------------------------------------
# Consume forfun-graphics (forfun_core lib) as a sibling source tree.
# Override with -DFORFUN_GRAPHICS_DIR=/path/to/forfun-graphics if needed.
# ----------------------------------------------------------------------------
set(FORFUN_GRAPHICS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../forfun-graphics"
    CACHE PATH "Path to the forfun-graphics source tree")

if(NOT EXISTS "${FORFUN_GRAPHICS_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Cannot find forfun-graphics at ${FORFUN_GRAPHICS_DIR}.\n"
        "Pass -DFORFUN_GRAPHICS_DIR=/path/to/forfun-graphics during configure.")
endif()

# EXCLUDE_FROM_ALL so we don't build helmet_demo when building alpine_sun.
add_subdirectory("${FORFUN_GRAPHICS_DIR}"
                 "${CMAKE_CURRENT_BINARY_DIR}/_forfun_graphics"
                 EXCLUDE_FROM_ALL)

# ----------------------------------------------------------------------------
# DEM-loading dependencies
# ----------------------------------------------------------------------------
# CPM was loaded transitively by the forfun-graphics subdir above. We reuse it
# for libtiff. Copernicus GLO-30 tiles are DEFLATE-compressed GeoTIFF, so we
# also pull in zlib (libtiff's only mandatory codec dependency).

CPMAddPackage(
    NAME              zlib
    GITHUB_REPOSITORY madler/zlib
    GIT_TAG           v1.3.1
    OPTIONS
        "ZLIB_BUILD_EXAMPLES OFF"
)

# libtiff's find_package(ZLIB) is intercepted by our FindZLIB shim in cmake/.
CPMAddPackage(
    NAME           libtiff
    GIT_REPOSITORY https://gitlab.com/libtiff/libtiff.git
    GIT_TAG        v4.6.0
    OPTIONS
        "tiff-tools OFF"
        "tiff-tests OFF"
        "tiff-contrib OFF"
        "tiff-docs OFF"
        "tiff-deprecated OFF"
        "tiff-install OFF"
        "zlib ON"
        "jpeg OFF"
        "jbig OFF"
        "lzma OFF"
        "zstd OFF"
        "webp OFF"
        "lerc OFF"
        "libdeflate OFF"
        "BUILD_SHARED_LIBS OFF"
)

# ----------------------------------------------------------------------------
# Test data: Copernicus GLO-30 DEM tiles for the built-in preset regions.
# Each tile is a 1° × 1° WGS84 GeoTIFF, ~30 MB, served by AWS Open Data with
# no authentication. We download every tile any preset references; the app
# picks one at runtime via --terrain and crops to the preset's window.
# ----------------------------------------------------------------------------
set(DEM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets/dem")
file(MAKE_DIRECTORY "${DEM_DIR}")

function(alpine_sun_download_dem name)
    set(_path "${DEM_DIR}/${name}")
    if(NOT EXISTS "${_path}")
        message(STATUS "Downloading Copernicus GLO-30 tile ${name} (~30 MB)...")
        string(REPLACE ".tif" "" _stem "${name}")
        file(DOWNLOAD
            "https://copernicus-dem-30m.s3.amazonaws.com/${_stem}/${name}"
            "${_path}"
            SHOW_PROGRESS
            STATUS _dl_status
        )
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            list(GET _dl_status 1 _dl_msg)
            file(REMOVE "${_path}")
            message(FATAL_ERROR "Failed to download ${name}: ${_dl_msg}")
        endif()
    endif()
endfunction()

# Matterhorn — Pennine Alps. Used by matterhorn-wide and matterhorn presets.
alpine_sun_download_dem("Copernicus_DSM_COG_10_N45_00_E007_00_DEM.tif")
# Everest — Mahalangur Himal. Used by the everest preset.
alpine_sun_download_dem("Copernicus_DSM_COG_10_N27_00_E086_00_DEM.tif")

set(MATTERHORN_TILE_PATH "${DEM_DIR}/Copernicus_DSM_COG_10_N45_00_E007_00_DEM.tif")
set(EVEREST_TILE_PATH    "${DEM_DIR}/Copernicus_DSM_COG_10_N27_00_E086_00_DEM.tif")

# ----------------------------------------------------------------------------
# stb_image header — forfun_core already provides the IMPLEMENTATION symbols
# via its tinygltf bundle, so we only need the declarations to call stbi_load.
# Downloaded once at configure time to keep the repo lean.
# ----------------------------------------------------------------------------
set(STB_IMAGE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/vendor/stb_image.h")
if(NOT EXISTS "${STB_IMAGE_HEADER}")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/vendor")
    message(STATUS "Downloading stb_image.h ...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
        "${STB_IMAGE_HEADER}"
        STATUS _stb_status)
    list(GET _stb_status 0 _stb_code)
    if(NOT _stb_code EQUAL 0)
        list(GET _stb_status 1 _stb_msg)
        file(REMOVE "${STB_IMAGE_HEADER}")
        message(FATAL_ERROR "Failed to download stb_image.h: ${_stb_msg}")
    endif()
endif()

# ----------------------------------------------------------------------------
# Satellite imagery (NASA Worldview Landsat 30 m annual composite).
# Public, no key, returns JPEG cropped to a lat/lon bbox. We fetch one image
# per preset at configure time so the binary can use it as terrain albedo.
# ----------------------------------------------------------------------------
set(SAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets/sat")
file(MAKE_DIRECTORY "${SAT_DIR}")

function(alpine_sun_download_sat name min_lat min_lon max_lat max_lon width height)
    set(_path "${SAT_DIR}/${name}.jpg")
    if(NOT EXISTS "${_path}")
        message(STATUS "Downloading satellite imagery for ${name}...")
        # ESRI World Imagery via the public Export Image REST endpoint.
        # No API key required for low-volume access; imagery is typically
        # high-resolution (sub-30 m in most mountain regions). BBOX order
        # here is xmin,ymin,xmax,ymax = lon,lat,lon,lat with bboxSR=4326.
        set(_url
            "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
            "?bbox=${min_lon},${min_lat},${max_lon},${max_lat}"
            "&bboxSR=4326"
            "&imageSR=4326"
            "&size=${width},${height}"
            "&format=jpg"
            "&f=image")
        string(JOIN "" _url ${_url})
        file(DOWNLOAD "${_url}" "${_path}" SHOW_PROGRESS STATUS _dl_status)
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            list(GET _dl_status 1 _dl_msg)
            file(REMOVE "${_path}")
            message(WARNING "Failed to download sat imagery for ${name}: ${_dl_msg}\n"
                            "Satellite mode will be unavailable for this preset.")
        endif()
    endif()
endfunction()

# Matterhorn close-up: ~9 km × 10 km, aspect ~1.33
alpine_sun_download_sat("matterhorn" 45.91 7.60 46.00 7.72 1024 768)
# Everest close-up:    ~15 km × 10 km, aspect ~1.67
alpine_sun_download_sat("everest"    27.91 86.85 28.00 87.00 1024 614)

set(SAT_MATTERHORN_PATH "${SAT_DIR}/matterhorn.jpg")
set(SAT_EVEREST_PATH    "${SAT_DIR}/everest.jpg")

# ----------------------------------------------------------------------------
# alpine_sun executable
# ----------------------------------------------------------------------------
add_executable(alpine_sun
    src/main.cpp
    src/dem_loader.cpp
    src/geo_frame.cpp
    src/terrain_mesh.cpp
    src/psa.cpp
    src/sun_driver.cpp
    src/gpx.cpp
)

target_link_libraries(alpine_sun PRIVATE
    forfun::core   # alias for forfun_core
    glfw           # exposed by the forfun-graphics subdir via CPM
    tiff           # libtiff static lib target
    imgui          # static lib defined in forfun-graphics' CMakeLists
)

target_include_directories(alpine_sun PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vendor"
)

target_compile_definitions(alpine_sun PRIVATE
    ALPINE_SUN_MATTERHORN_TILE_PATH="${MATTERHORN_TILE_PATH}"
    ALPINE_SUN_EVEREST_TILE_PATH="${EVEREST_TILE_PATH}"
    ALPINE_SUN_ROUTE_MATTERHORN_HORNLI="${CMAKE_CURRENT_SOURCE_DIR}/assets/matterhorn-hornli.gpx"
    ALPINE_SUN_ROUTE_EVEREST_SUMMIT="${CMAKE_CURRENT_SOURCE_DIR}/assets/everest-summit.gpx"
    ALPINE_SUN_SAT_MATTERHORN_PATH="${SAT_MATTERHORN_PATH}"
    ALPINE_SUN_SAT_EVEREST_PATH="${SAT_EVEREST_PATH}"
)

# Shaders — ForfunShaders.cmake's add_shader() is available transitively
# through the forfun-graphics subdir.
add_shader(alpine_sun "${CMAKE_CURRENT_SOURCE_DIR}/shaders/terrain.vert"       terrain_vert_spv)
add_shader(alpine_sun "${CMAKE_CURRENT_SOURCE_DIR}/shaders/terrain.frag"       terrain_frag_spv)
add_shader(alpine_sun "${CMAKE_CURRENT_SOURCE_DIR}/shaders/terrain_depth.vert" terrain_depth_vert_spv)
add_shader(alpine_sun "${CMAKE_CURRENT_SOURCE_DIR}/shaders/horizon_map.comp"   horizon_map_comp_spv)
add_shader(alpine_sun "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sun_hours.comp"     sun_hours_comp_spv)
add_shader(alpine_sun "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sky.vert"           sky_vert_spv)
add_shader(alpine_sun "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sky.frag"           sky_frag_spv)
add_shader(alpine_sun "${CMAKE_CURRENT_SOURCE_DIR}/shaders/route.vert"         route_vert_spv)
add_shader(alpine_sun "${CMAKE_CURRENT_SOURCE_DIR}/shaders/route.frag"         route_frag_spv)
add_shader(alpine_sun "${CMAKE_CURRENT_SOURCE_DIR}/shaders/transmittance.comp" transmittance_comp_spv)
add_shader(alpine_sun "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sky_view.comp"      sky_view_comp_spv)

if(MSVC)
    target_compile_options(alpine_sun PRIVATE /W4 /permissive-)
else()
    target_compile_options(alpine_sun PRIVATE -Wall -Wextra -Wpedantic)
endif()
