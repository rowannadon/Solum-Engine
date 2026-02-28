#include "solum_engine/render/DebugBoundsManager.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "solum_engine/voxel/World.h"

namespace {
constexpr glm::vec4 kChunkBoundsColor{0.2f, 0.95f, 0.35f, 0.22f};
constexpr glm::vec4 kColumnBoundsColor{1.0f, 0.7f, 0.2f, 0.6f};
constexpr glm::vec4 kRegionBoundsColor{0.2f, 0.8f, 1.0f, 0.95f};
constexpr glm::vec4 kMeshletBoundsColor{1.0f, 0.35f, 0.15f, 0.4f};

void appendWireBox(std::vector<DebugLineVertex>& vertices,
                   const glm::vec3& minCorner,
                   const glm::vec3& maxCorner,
                   const glm::vec4& color) {
    const std::array<glm::vec3, 8> corners{
        glm::vec3{minCorner.x, minCorner.y, minCorner.z},
        glm::vec3{maxCorner.x, minCorner.y, minCorner.z},
        glm::vec3{maxCorner.x, maxCorner.y, minCorner.z},
        glm::vec3{minCorner.x, maxCorner.y, minCorner.z},
        glm::vec3{minCorner.x, minCorner.y, maxCorner.z},
        glm::vec3{maxCorner.x, minCorner.y, maxCorner.z},
        glm::vec3{maxCorner.x, maxCorner.y, maxCorner.z},
        glm::vec3{minCorner.x, maxCorner.y, maxCorner.z},
    };

    constexpr std::array<uint8_t, 24> edgeIndices{
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7
    };

    for (size_t i = 0; i < edgeIndices.size(); i += 2) {
        vertices.push_back(DebugLineVertex{corners[edgeIndices[i]], color});
        vertices.push_back(DebugLineVertex{corners[edgeIndices[i + 1]], color});
    }
}
}  // namespace

void DebugBoundsManager::setWorld(const World* world) {
    world_ = world;
}

void DebugBoundsManager::reset() {
    uploadedWorldRevision_ = 0;
    uploadedMeshRevision_ = 0;
    uploadedLayerMask_ = 0u;
}

void DebugBoundsManager::rebuild(uint32_t layerMask,
                                 BoundsDebugPipeline& boundsPipeline,
                                 const MeshletBufferController& meshletBuffers) const {
    const bool includeChunkBounds = (layerMask & kRenderFlagBoundsChunks) != 0u;
    const bool includeColumnBounds = (layerMask & kRenderFlagBoundsColumns) != 0u;
    const bool includeRegionBounds = (layerMask & kRenderFlagBoundsRegions) != 0u;
    const bool includeMeshletBounds = (layerMask & kRenderFlagBoundsMeshlets) != 0u;
    const bool includeWorldBounds = includeChunkBounds || includeColumnBounds || includeRegionBounds;

    std::vector<ColumnCoord> generatedColumns;
    if (includeWorldBounds && world_) {
        world_->copyGeneratedColumns(generatedColumns);
    }

    std::vector<DebugLineVertex> vertices;
    const size_t chunkBoxCount = includeChunkBounds
        ? (generatedColumns.size() * static_cast<size_t>(cfg::COLUMN_HEIGHT))
        : 0u;
    const size_t columnBoxCount = includeColumnBounds ? generatedColumns.size() : 0u;
    const size_t regionBoxEstimate = includeRegionBounds
        ? std::max<size_t>(1, generatedColumns.size() / static_cast<size_t>(cfg::REGION_VOLUME_COLUMNS))
        : 0u;
    const std::vector<MeshletAabb>& activeMeshletBounds = meshletBuffers.activeMeshletBounds();
    const size_t meshletBoxCount = includeMeshletBounds ? activeMeshletBounds.size() : 0u;
    vertices.reserve((chunkBoxCount + columnBoxCount + regionBoxEstimate + meshletBoxCount) * 24ull);

    std::unordered_set<RegionCoord> visibleRegions;
    visibleRegions.reserve(generatedColumns.size());

    for (const ColumnCoord& columnCoord : generatedColumns) {
        const ChunkCoord baseChunk = column_local_to_chunk(columnCoord, 0);
        const BlockCoord columnOrigin = chunk_to_block_origin(baseChunk);
        const glm::vec3 columnMin{
            static_cast<float>(columnOrigin.v.x),
            static_cast<float>(columnOrigin.v.y),
            static_cast<float>(columnOrigin.v.z)
        };
        if (includeColumnBounds) {
            const glm::vec3 columnMax = columnMin + glm::vec3(
                static_cast<float>(cfg::CHUNK_SIZE),
                static_cast<float>(cfg::CHUNK_SIZE),
                static_cast<float>(cfg::COLUMN_HEIGHT_BLOCKS)
            );
            appendWireBox(vertices, columnMin, columnMax, kColumnBoundsColor);
        }

        visibleRegions.insert(column_to_region(columnCoord));

        if (includeChunkBounds) {
            for (int32_t localZ = 0; localZ < cfg::COLUMN_HEIGHT; ++localZ) {
                const ChunkCoord chunkCoord = column_local_to_chunk(columnCoord, localZ);
                const BlockCoord chunkOrigin = chunk_to_block_origin(chunkCoord);
                const glm::vec3 chunkMin{
                    static_cast<float>(chunkOrigin.v.x),
                    static_cast<float>(chunkOrigin.v.y),
                    static_cast<float>(chunkOrigin.v.z)
                };
                const glm::vec3 chunkMax = chunkMin + glm::vec3(
                    static_cast<float>(cfg::CHUNK_SIZE),
                    static_cast<float>(cfg::CHUNK_SIZE),
                    static_cast<float>(cfg::CHUNK_SIZE)
                );
                appendWireBox(vertices, chunkMin, chunkMax, kChunkBoundsColor);
            }
        }
    }

    if (includeRegionBounds) {
        std::vector<RegionCoord> sortedRegions(visibleRegions.begin(), visibleRegions.end());
        std::sort(sortedRegions.begin(), sortedRegions.end());
        for (const RegionCoord& regionCoord : sortedRegions) {
            const ColumnCoord regionOriginColumn = region_to_column_origin(regionCoord);
            const ChunkCoord regionBaseChunk = column_local_to_chunk(regionOriginColumn, 0);
            const BlockCoord regionOrigin = chunk_to_block_origin(regionBaseChunk);
            const glm::vec3 regionMin{
                static_cast<float>(regionOrigin.v.x),
                static_cast<float>(regionOrigin.v.y),
                static_cast<float>(regionOrigin.v.z)
            };
            const glm::vec3 regionMax = regionMin + glm::vec3(
                static_cast<float>(cfg::REGION_SIZE_BLOCKS),
                static_cast<float>(cfg::REGION_SIZE_BLOCKS),
                static_cast<float>(cfg::COLUMN_HEIGHT_BLOCKS)
            );
            appendWireBox(vertices, regionMin, regionMax, kRegionBoundsColor);
        }
    }

    if (includeMeshletBounds) {
        for (const MeshletAabb& bounds : activeMeshletBounds) {
            appendWireBox(vertices, bounds.minCorner, bounds.maxCorner, kMeshletBoundsColor);
        }
    }

    if (!boundsPipeline.updateVertices(vertices)) {
        std::cerr << "Failed to upload debug bounds vertices." << std::endl;
    }
}

void DebugBoundsManager::update(const FrameUniforms& frameUniforms,
                                BoundsDebugPipeline& boundsPipeline,
                                const MeshletBufferController& meshletBuffers) {
    const bool enabled = (frameUniforms.renderFlags[0] & kRenderFlagBoundsDebug) != 0u;
    boundsPipeline.setEnabled(enabled);
    if (!enabled) {
        return;
    }

    const uint64_t worldRevision = world_ ? world_->generationRevision() : 0u;
    const uint64_t meshRevision = meshletBuffers.uploadedMeshRevision();
    const uint32_t layerMask = frameUniforms.renderFlags[0] & kRenderFlagBoundsLayerMask;
    if (layerMask == 0u) {
        boundsPipeline.updateVertices({});
        uploadedLayerMask_ = 0u;
        uploadedWorldRevision_ = worldRevision;
        uploadedMeshRevision_ = meshRevision;
        return;
    }

    const bool includeWorldBounds = (layerMask & (
        kRenderFlagBoundsChunks |
        kRenderFlagBoundsColumns |
        kRenderFlagBoundsRegions)) != 0u;
    const bool includeMeshletBounds = (layerMask & kRenderFlagBoundsMeshlets) != 0u;

    const bool layersChanged = layerMask != uploadedLayerMask_;
    const bool worldChanged = includeWorldBounds && (worldRevision != uploadedWorldRevision_);
    const bool meshChanged = includeMeshletBounds && (meshRevision != uploadedMeshRevision_);
    if (!layersChanged && !worldChanged && !meshChanged) {
        return;
    }

    rebuild(layerMask, boundsPipeline, meshletBuffers);
    uploadedWorldRevision_ = worldRevision;
    uploadedMeshRevision_ = meshRevision;
    uploadedLayerMask_ = layerMask;
}
