cmake_minimum_required(VERSION 3.22)
project(alpine_sun LANGUAGES CXX)

set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

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
# alpine_sun executable
# ----------------------------------------------------------------------------
add_executable(alpine_sun src/main.cpp)

target_link_libraries(alpine_sun PRIVATE
    forfun::core   # alias for forfun_core
    glfw           # exposed by the forfun-graphics subdir via CPM
)

if(MSVC)
    target_compile_options(alpine_sun PRIVATE /W4 /permissive-)
else()
    target_compile_options(alpine_sun PRIVATE -Wall -Wextra -Wpedantic)
endif()
