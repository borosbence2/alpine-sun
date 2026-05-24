# Find-module shim that lets libtiff's find_package(ZLIB) discover the
# CPM-provided zlib (zlibstatic target), so DEFLATE-compressed GeoTIFFs work.
#
# Loaded automatically when this directory is on CMAKE_MODULE_PATH.

if(TARGET zlibstatic)
    if(NOT TARGET ZLIB::ZLIB)
        add_library(ZLIB::ZLIB ALIAS zlibstatic)
    endif()
    set(ZLIB_FOUND          TRUE)
    set(ZLIB_INCLUDE_DIR    "${zlib_SOURCE_DIR}" "${zlib_BINARY_DIR}")
    set(ZLIB_INCLUDE_DIRS   "${ZLIB_INCLUDE_DIR}")
    set(ZLIB_LIBRARIES      zlibstatic)
    set(ZLIB_VERSION_STRING "1.3.1")
    set(ZLIB_VERSION_MAJOR  1)
    set(ZLIB_VERSION_MINOR  3)
    set(ZLIB_VERSION_PATCH  1)
else()
    message(FATAL_ERROR
        "FindZLIB shim invoked but CPM 'zlibstatic' target not found. "
        "Did CPMAddPackage(NAME zlib ...) run before find_package(ZLIB)?")
endif()
