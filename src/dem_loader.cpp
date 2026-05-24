#include "dem_loader.h"

#include <tiffio.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>

namespace dem {

namespace {

// GeoTIFF model tags we read. libtiff doesn't ship definitions for these
// because GeoTIFF is a separate spec; we register them as custom tags so
// TIFFGetField can return their values.
constexpr ttag_t kTagModelPixelScale = 33550;
constexpr ttag_t kTagModelTiepoint   = 33922;

const TIFFFieldInfo kGeoTiffFields[] = {
    {kTagModelPixelScale, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, true, true,
     const_cast<char*>("ModelPixelScaleTag")},
    {kTagModelTiepoint,   -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, true, true,
     const_cast<char*>("ModelTiepointTag")},
};

void registerGeoTiffTags(TIFF* tif) {
    TIFFMergeFieldInfo(tif, kGeoTiffFields,
                       sizeof(kGeoTiffFields) / sizeof(kGeoTiffFields[0]));
}

struct TiffHandle {
    TIFF* tif = nullptr;
    ~TiffHandle() { if (tif) TIFFClose(tif); }
};

// Copies one decoded tile into the row-major output buffer, clipping at the
// image edge (last column/row of tiles is typically partial).
void blitTile(const float* tileBuf,
              uint32_t     tileWidth,
              uint32_t     tileHeight,
              uint32_t     originX,
              uint32_t     originY,
              uint32_t     imageWidth,
              uint32_t     imageHeight,
              float*       dst) {
    const uint32_t copyRows = std::min(tileHeight, imageHeight - originY);
    const uint32_t copyCols = std::min(tileWidth,  imageWidth  - originX);
    for (uint32_t ty = 0; ty < copyRows; ++ty) {
        const float* src    = tileBuf + static_cast<size_t>(ty) * tileWidth;
        float*       dstRow = dst + (static_cast<size_t>(originY + ty) * imageWidth) + originX;
        std::copy_n(src, copyCols, dstRow);
    }
}

bool readScanlineLayout(TIFF* tif, uint32_t width, uint32_t height,
                        std::vector<float>& out) {
    const tsize_t scanlineBytes = TIFFScanlineSize(tif);
    if (scanlineBytes != static_cast<tsize_t>(width * sizeof(float))) {
        std::fprintf(stderr,
                     "dem: unexpected scanline size %lld for width %u\n",
                     static_cast<long long>(scanlineBytes), width);
        return false;
    }
    for (uint32_t row = 0; row < height; ++row) {
        float* dst = out.data() + static_cast<size_t>(row) * width;
        if (TIFFReadScanline(tif, dst, row, 0) < 0) {
            std::fprintf(stderr, "dem: TIFFReadScanline failed at row %u\n", row);
            return false;
        }
    }
    return true;
}

bool readTiledLayout(TIFF* tif, uint32_t width, uint32_t height,
                     std::vector<float>& out) {
    uint32_t tileWidth = 0, tileHeight = 0;
    if (!TIFFGetField(tif, TIFFTAG_TILEWIDTH,  &tileWidth) ||
        !TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight)) {
        std::fprintf(stderr, "dem: tiled image missing TIFFTAG_TILEWIDTH/LENGTH\n");
        return false;
    }
    const tsize_t tileBytes = TIFFTileSize(tif);
    if (tileBytes != static_cast<tsize_t>(tileWidth * tileHeight * sizeof(float))) {
        std::fprintf(stderr,
                     "dem: unexpected tile size %lld for %ux%u float tiles\n",
                     static_cast<long long>(tileBytes), tileWidth, tileHeight);
        return false;
    }
    std::vector<float> tileBuf(static_cast<size_t>(tileWidth) * tileHeight);
    for (uint32_t y = 0; y < height; y += tileHeight) {
        for (uint32_t x = 0; x < width; x += tileWidth) {
            if (TIFFReadTile(tif, tileBuf.data(), x, y, 0, 0) < 0) {
                std::fprintf(stderr, "dem: TIFFReadTile failed at (%u,%u)\n", x, y);
                return false;
            }
            blitTile(tileBuf.data(), tileWidth, tileHeight, x, y, width, height,
                     out.data());
        }
    }
    return true;
}

} // namespace

bool loadGeoTIFF(const std::string& path, Tile& out) {
    // Silence libtiff's noisy warnings about unknown GeoKey directory tags
    // (34735+) that we don't consume.
    TIFFSetWarningHandler(nullptr);

    TiffHandle h;
    h.tif = TIFFOpen(path.c_str(), "r");
    if (!h.tif) {
        std::fprintf(stderr, "dem: TIFFOpen failed for '%s'\n", path.c_str());
        return false;
    }
    registerGeoTiffTags(h.tif);

    uint32_t width = 0, height = 0;
    uint16_t bitsPerSample = 0, sampleFormat = 0, samplesPerPixel = 1;
    uint16_t planarConfig = PLANARCONFIG_CONTIG;

    if (!TIFFGetField(h.tif, TIFFTAG_IMAGEWIDTH,  &width) ||
        !TIFFGetField(h.tif, TIFFTAG_IMAGELENGTH, &height)) {
        std::fprintf(stderr, "dem: missing image dimensions\n");
        return false;
    }
    TIFFGetField(h.tif, TIFFTAG_BITSPERSAMPLE,   &bitsPerSample);
    TIFFGetField(h.tif, TIFFTAG_SAMPLEFORMAT,    &sampleFormat);
    TIFFGetField(h.tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFGetField(h.tif, TIFFTAG_PLANARCONFIG,    &planarConfig);

    if (bitsPerSample != 32 || sampleFormat != SAMPLEFORMAT_IEEEFP ||
        samplesPerPixel != 1) {
        std::fprintf(stderr,
                     "dem: unsupported pixel format "
                     "(bits=%u format=%u samples=%u, expected 32-bit float single-channel)\n",
                     bitsPerSample, sampleFormat, samplesPerPixel);
        return false;
    }

    // ---- Geo metadata ----
    // libtiff writes the count as uint32 for custom variable-count tags, so
    // these MUST be uint32_t — uint16_t corrupts the stack (caught by MSVC
    // /RTC1 in Debug builds).
    double*  pixelScale      = nullptr;
    uint32_t pixelScaleCount = 0;
    if (!TIFFGetField(h.tif, kTagModelPixelScale, &pixelScaleCount, &pixelScale) ||
        pixelScaleCount < 2 || pixelScale == nullptr) {
        std::fprintf(stderr, "dem: missing ModelPixelScaleTag (33550)\n");
        return false;
    }

    double*  tiepoint      = nullptr;
    uint32_t tiepointCount = 0;
    if (!TIFFGetField(h.tif, kTagModelTiepoint, &tiepointCount, &tiepoint) ||
        tiepointCount < 6 || tiepoint == nullptr) {
        std::fprintf(stderr, "dem: missing ModelTiepointTag (33922)\n");
        return false;
    }

    // ModelPixelScaleTag: [ScaleX, ScaleY, ScaleZ]
    //   ScaleX = degrees per pixel (longitude)
    //   ScaleY = degrees per pixel (latitude, positive value)
    // ModelTiepointTag: [I, J, K, X, Y, Z]
    //   maps raster (I, J) -> geo (X, Y) — for Copernicus this is (0, 0) -> (minLon, maxLat)
    out.pixelSizeLon = pixelScale[0];
    out.pixelSizeLat = pixelScale[1];
    const double rasterI = tiepoint[0];
    const double rasterJ = tiepoint[1];
    const double geoX    = tiepoint[3];   // longitude at (I, J)
    const double geoY    = tiepoint[4];   // latitude  at (I, J)

    // Translate (I,J)->(lon,lat) into the (0,0)->(NW corner) form Copernicus uses,
    // then derive the SW/NE bbox.
    const double nwLon = geoX - rasterI * out.pixelSizeLon;
    const double nwLat = geoY + rasterJ * out.pixelSizeLat;  // J grows southward
    out.extent.minLon  = nwLon;
    out.extent.maxLon  = nwLon + width  * out.pixelSizeLon;
    out.extent.maxLat  = nwLat;
    out.extent.minLat  = nwLat - height * out.pixelSizeLat;

    // ---- Pixel data ----
    out.width  = width;
    out.height = height;
    out.elevation.assign(static_cast<size_t>(width) * height, 0.0f);

    const bool tiled = TIFFIsTiled(h.tif);
    const bool ok    = tiled
        ? readTiledLayout   (h.tif, width, height, out.elevation)
        : readScanlineLayout(h.tif, width, height, out.elevation);
    if (!ok) return false;

    // ---- Min / max for downstream colour mapping & camera framing ----
    float lo = std::numeric_limits<float>::infinity();
    float hi = -std::numeric_limits<float>::infinity();
    for (float v : out.elevation) {
        if (v == v) {  // skip NaN
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    }
    out.minElevation = lo;
    out.maxElevation = hi;

    return true;
}

} // namespace dem
