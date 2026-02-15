#pragma once

#include "solum_engine/resources/Constants.h"

#include <array>
#include <cstddef>
#include <cstdint>

// Shared region LOD count used by both streaming and rendering subsystems.
inline constexpr int kRegionLodCount = 6;

struct WorldTuningParameters {
    int viewDistanceChunks = 32;
    int verticalChunkMin = 0;
    int verticalChunkMax = COLUMN_CHUNKS_Z - 1;

    // Region renderer LOD mesh decimation in blocks per cell.
    std::array<int, kRegionLodCount> regionLodSteps{1, 2, 4, 8, 16, 32};
    // Distance thresholds where LOD switches from i to i+1.
    std::array<float, kRegionLodCount - 1> regionLodSwitchDistances{
        REGION_BLOCKS_XY * 1.0f,
        REGION_BLOCKS_XY * 2.0f,
        REGION_BLOCKS_XY * 4.0f,
        REGION_BLOCKS_XY * 8.0f,
        REGION_BLOCKS_XY * 16.0f,
    };

    // Heightmap terrain settings.
    const char* heightmapRelativePath = "heightmap.png";
    int heightmapUpscaleFactor = 4;
    bool heightmapWrap = true;
    int terrainMinHeightBlocks = CHUNK_SIZE * 2;
    int terrainMaxHeightBlocks = CHUNK_SIZE * (COLUMN_CHUNKS_Z - 2);
};

inline constexpr WorldTuningParameters kDefaultWorldTuningParameters{};

struct RendererTuningParameters {
    uint32_t meshletsPerPage = 8192u;
    uint32_t initialMeshletPageCount = 2u;
    uint32_t maxMeshletPages = 8u;
    uint32_t initialMeshletMetadataCapacity = 65536u;
    uint32_t maxDrawMeshletsPerFrame = 65536u;
};

inline constexpr RendererTuningParameters kDefaultRendererTuningParameters{};

struct ApplicationTuningParameters {
    std::size_t frameTimeHistorySize = 100;
    float frameSleepBufferSeconds = 0.0005f;
};

inline constexpr ApplicationTuningParameters kDefaultApplicationTuningParameters{};
