#include "solum_engine/render/MeshletBufferController.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>

using namespace wgpu;

MeshletAabb MeshletBufferController::computeMeshletAabb(const Meshlet& meshlet) {
    const float voxelScale = static_cast<float>(std::max(meshlet.voxelScale, 1u));
    const glm::vec3 meshletOrigin = glm::vec3(meshlet.origin);

    if (meshlet.hasCustomBounds) {
        return MeshletAabb{
            meshletOrigin + (meshlet.localBoundsMin * voxelScale),
            meshletOrigin + (meshlet.localBoundsMax * voxelScale)
        };
    }

    if (meshlet.quadCount == 0u) {
        return MeshletAabb{meshletOrigin, meshletOrigin};
    }

    static const std::array<std::array<glm::vec3, 4>, 6> kFaceCornerOffsets{{
        {{glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 0.0f}, glm::vec3{1.0f, 0.0f, 1.0f}, glm::vec3{1.0f, 1.0f, 1.0f}}},
        {{glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 1.0f}}},
        {{glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 1.0f}, glm::vec3{1.0f, 1.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 1.0f}}},
        {{glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec3{1.0f, 0.0f, 1.0f}}},
        {{glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec3{1.0f, 0.0f, 1.0f}, glm::vec3{0.0f, 1.0f, 1.0f}, glm::vec3{1.0f, 1.0f, 1.0f}}},
        {{glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 0.0f}}},
    }};

    const uint32_t safeFaceDirection = std::min(meshlet.faceDirection, 5u);

    bool firstVertex = true;
    glm::vec3 minCorner{0.0f};
    glm::vec3 maxCorner{0.0f};

    for (uint32_t quadIndex = 0; quadIndex < meshlet.quadCount; ++quadIndex) {
        const glm::uvec3 local = unpackMeshletLocalOffset(meshlet.packedQuadLocalOffsets[quadIndex]);
        const glm::vec3 quadBase = meshletOrigin + (glm::vec3(local) * voxelScale);
        for (const glm::vec3& cornerOffset : kFaceCornerOffsets[safeFaceDirection]) {
            const glm::vec3 vertex = quadBase + (cornerOffset * voxelScale);
            if (firstVertex) {
                minCorner = vertex;
                maxCorner = vertex;
                firstVertex = false;
                continue;
            }
            minCorner = glm::min(minCorner, vertex);
            maxCorner = glm::max(maxCorner, vertex);
        }
    }

    return MeshletAabb{minCorner, maxCorner};
}

MeshletAabbGPU MeshletBufferController::toGpuAabb(const MeshletAabb& aabb) {
    return MeshletAabbGPU{
        glm::vec4(aabb.minCorner, 0.0f),
        glm::vec4(aabb.maxCorner, 0.0f)
    };
}

MeshletBufferController::PackedTileLodData MeshletBufferController::packTileLodMeshlets(const std::vector<Meshlet>& meshlets) {
    PackedTileLodData packed;

    uint32_t totalMeshletCount = 0u;
    uint32_t totalQuadWordCount = 0u;
    for (const Meshlet& meshlet : meshlets) {
        if (meshlet.quadCount == 0u) {
            continue;
        }
        ++totalMeshletCount;
        totalQuadWordCount += meshlet.quadCount * MESHLET_QUAD_DATA_WORD_STRIDE;
    }

    packed.metadata.reserve(totalMeshletCount);
    packed.quadData.reserve(totalQuadWordCount);
    packed.aabbGpu.reserve(totalMeshletCount);
    packed.bounds.reserve(totalMeshletCount);

    for (const Meshlet& meshlet : meshlets) {
        if (meshlet.quadCount == 0u) {
            continue;
        }

        MeshletMetadataGPU metadata{};
        metadata.originX = meshlet.origin.x;
        metadata.originY = meshlet.origin.y;
        metadata.originZ = meshlet.origin.z;
        metadata.quadCount = meshlet.quadCount;
        metadata.faceDirection = meshlet.faceDirection;
        metadata.dataOffset = static_cast<uint32_t>(packed.quadData.size());
        metadata.voxelScale = std::max(meshlet.voxelScale, 1u);
        packed.metadata.push_back(metadata);

        const MeshletAabb bounds = computeMeshletAabb(meshlet);
        packed.bounds.push_back(bounds);
        packed.aabbGpu.push_back(toGpuAabb(bounds));

        for (uint32_t i = 0; i < meshlet.quadCount; ++i) {
            packed.quadData.push_back(packMeshletQuadData(
                meshlet.packedQuadLocalOffsets[i],
                meshlet.quadMaterialIds[i]
            ));
            packed.quadData.push_back(packMeshletQuadAuxData(
                meshlet.quadAoData[i],
                meshlet.quadModelQuadIndices[i],
                meshlet.quadUsesVoxelAo[i] != 0u
            ));
        }
    }

    return packed;
}

bool MeshletBufferController::initialize(BufferManager* bufferManager) {
    bufferManager_ = bufferManager;
    allocations_.clear();
    tileSelection_.clear();
    activeRanges_.clear();
    activeMeshletBounds_.clear();
    activeSelectionMeshletCount_ = 0u;
    uploadedMeshRevision_ = 0u;

    if (bufferManager_ == nullptr) {
        return false;
    }

    bool recreated = false;
    if (!ensureBuffers(kInitialMeshletCapacity, kInitialQuadWordCapacity, kInitialRangeCapacity, &recreated)) {
        return false;
    }

    return buildActiveRanges();
}

bool MeshletBufferController::allocateRange(std::vector<FreeRange>& freeList,
                                            uint32_t length,
                                            uint32_t& outOffset) {
    if (length == 0u) {
        outOffset = 0u;
        return true;
    }

    for (auto it = freeList.begin(); it != freeList.end(); ++it) {
        if (it->length < length) {
            continue;
        }
        outOffset = it->offset;
        it->offset += length;
        it->length -= length;
        if (it->length == 0u) {
            freeList.erase(it);
        }
        return true;
    }
    return false;
}

void MeshletBufferController::freeRange(std::vector<FreeRange>& freeList,
                                        uint32_t offset,
                                        uint32_t length) {
    if (length == 0u) {
        return;
    }

    freeList.push_back(FreeRange{offset, length});
    std::sort(freeList.begin(), freeList.end(), [](const FreeRange& a, const FreeRange& b) {
        return a.offset < b.offset;
    });

    std::vector<FreeRange> merged;
    merged.reserve(freeList.size());
    for (const FreeRange& range : freeList) {
        if (merged.empty()) {
            merged.push_back(range);
            continue;
        }

        FreeRange& back = merged.back();
        if (back.offset + back.length >= range.offset) {
            const uint32_t mergedEnd = std::max(back.offset + back.length, range.offset + range.length);
            back.length = mergedEnd - back.offset;
            continue;
        }

        merged.push_back(range);
    }

    freeList = std::move(merged);
}

bool MeshletBufferController::recreateBuffers(uint32_t meshletCapacity,
                                              uint32_t quadWordCapacity,
                                              uint32_t rangeCapacity) {
    if (bufferManager_ == nullptr || meshletCapacity == 0u || quadWordCapacity == 0u || rangeCapacity == 0u) {
        return false;
    }

    BufferDescriptor metadataDesc = Default;
    metadataDesc.label = StringView("meshlet metadata buffer");
    metadataDesc.size = static_cast<uint64_t>(meshletCapacity) * sizeof(MeshletMetadataGPU);
    metadataDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    metadataDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(kMeshMetadataBufferName, metadataDesc)) {
        return false;
    }

    BufferDescriptor meshDataDesc = Default;
    meshDataDesc.label = StringView("meshlet data buffer");
    meshDataDesc.size = static_cast<uint64_t>(quadWordCapacity) * sizeof(uint32_t);
    meshDataDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    meshDataDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(kMeshDataBufferName, meshDataDesc)) {
        return false;
    }

    BufferDescriptor aabbDesc = Default;
    aabbDesc.label = StringView("meshlet aabb buffer");
    aabbDesc.size = static_cast<uint64_t>(meshletCapacity) * sizeof(MeshletAabbGPU);
    aabbDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    aabbDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(kMeshAabbBufferName, aabbDesc)) {
        return false;
    }

    BufferDescriptor visibleIndicesDesc = Default;
    visibleIndicesDesc.label = StringView("visible meshlet indices buffer");
    visibleIndicesDesc.size = static_cast<uint64_t>(meshletCapacity) * sizeof(uint32_t);
    visibleIndicesDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    visibleIndicesDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(kVisibleMeshletIndexBufferName, visibleIndicesDesc)) {
        return false;
    }

    BufferDescriptor activeRangesDesc = Default;
    activeRangesDesc.label = StringView("active meshlet ranges buffer");
    activeRangesDesc.size = static_cast<uint64_t>(rangeCapacity) * sizeof(ActiveMeshletRangeGPU);
    activeRangesDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    activeRangesDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(kActiveMeshletRangeBufferName, activeRangesDesc)) {
        return false;
    }

    BufferDescriptor rangeParamsDesc = Default;
    rangeParamsDesc.label = StringView("active meshlet ranges params buffer");
    rangeParamsDesc.size = sizeof(RangeParamsGPU);
    rangeParamsDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
    rangeParamsDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(kActiveMeshletRangeParamsBufferName, rangeParamsDesc)) {
        return false;
    }

    meshletCapacity_ = meshletCapacity;
    quadWordCapacity_ = quadWordCapacity;
    rangeCapacity_ = rangeCapacity;
    meshletFreeRanges_ = {FreeRange{0u, meshletCapacity_}};
    quadFreeRanges_ = {FreeRange{0u, quadWordCapacity_}};
    return true;
}

bool MeshletBufferController::writeAllocation(const AllocationRecord& record) {
    if (bufferManager_ == nullptr) {
        return false;
    }

    if (!record.packed.metadata.empty()) {
        std::vector<MeshletMetadataGPU> metadata = record.packed.metadata;
        for (MeshletMetadataGPU& entry : metadata) {
            entry.dataOffset += record.quadOffset;
        }

        bufferManager_->writeBuffer(
            kMeshMetadataBufferName,
            static_cast<uint64_t>(record.meshletOffset) * sizeof(MeshletMetadataGPU),
            metadata.data(),
            metadata.size() * sizeof(MeshletMetadataGPU)
        );

        bufferManager_->writeBuffer(
            kMeshAabbBufferName,
            static_cast<uint64_t>(record.meshletOffset) * sizeof(MeshletAabbGPU),
            record.packed.aabbGpu.data(),
            record.packed.aabbGpu.size() * sizeof(MeshletAabbGPU)
        );
    }

    if (!record.packed.quadData.empty()) {
        bufferManager_->writeBuffer(
            kMeshDataBufferName,
            static_cast<uint64_t>(record.quadOffset) * sizeof(uint32_t),
            record.packed.quadData.data(),
            record.packed.quadData.size() * sizeof(uint32_t)
        );
    }

    return true;
}

void MeshletBufferController::releaseAllocation(const MeshTileLodKey& key) {
    const auto it = allocations_.find(key);
    if (it == allocations_.end()) {
        return;
    }

    freeRange(meshletFreeRanges_, it->second.meshletOffset, static_cast<uint32_t>(it->second.packed.metadata.size()));
    freeRange(quadFreeRanges_, it->second.quadOffset, static_cast<uint32_t>(it->second.packed.quadData.size()));
    allocations_.erase(it);
}

bool MeshletBufferController::repackExistingAllocations() {
    std::vector<AllocationRecord> records;
    records.reserve(allocations_.size());
    for (const auto& [_, record] : allocations_) {
        records.push_back(record);
    }
    std::sort(records.begin(), records.end(), [](const AllocationRecord& a, const AllocationRecord& b) {
        return a.key < b.key;
    });

    allocations_.clear();
    meshletFreeRanges_ = {FreeRange{0u, meshletCapacity_}};
    quadFreeRanges_ = {FreeRange{0u, quadWordCapacity_}};

    for (AllocationRecord& record : records) {
        if (!allocateRange(meshletFreeRanges_, static_cast<uint32_t>(record.packed.metadata.size()), record.meshletOffset)) {
            return false;
        }
        if (!allocateRange(quadFreeRanges_, static_cast<uint32_t>(record.packed.quadData.size()), record.quadOffset)) {
            return false;
        }
        if (!writeAllocation(record)) {
            return false;
        }
        allocations_.emplace(record.key, std::move(record));
    }

    return true;
}

bool MeshletBufferController::ensureBuffers(uint32_t requiredMeshlets,
                                            uint32_t requiredQuadWords,
                                            uint32_t requiredRanges,
                                            bool* recreated) {
    if (recreated != nullptr) {
        *recreated = false;
    }

    if (bufferManager_ == nullptr) {
        return false;
    }

    const bool needsRecreate =
        meshletCapacity_ < requiredMeshlets ||
        quadWordCapacity_ < requiredQuadWords ||
        rangeCapacity_ < requiredRanges ||
        !bufferManager_->getBuffer(kMeshDataBufferName) ||
        !bufferManager_->getBuffer(kMeshMetadataBufferName) ||
        !bufferManager_->getBuffer(kMeshAabbBufferName) ||
        !bufferManager_->getBuffer(kVisibleMeshletIndexBufferName) ||
        !bufferManager_->getBuffer(kActiveMeshletRangeBufferName) ||
        !bufferManager_->getBuffer(kActiveMeshletRangeParamsBufferName);

    if (!needsRecreate) {
        return true;
    }

    uint32_t nextMeshletCapacity = std::max(meshletCapacity_, kInitialMeshletCapacity);
    while (nextMeshletCapacity < requiredMeshlets) {
        nextMeshletCapacity = std::max(requiredMeshlets, nextMeshletCapacity * 2u);
    }

    uint32_t nextQuadWordCapacity = std::max(quadWordCapacity_, kInitialQuadWordCapacity);
    while (nextQuadWordCapacity < requiredQuadWords) {
        nextQuadWordCapacity = std::max(requiredQuadWords, nextQuadWordCapacity * 2u);
    }

    uint32_t nextRangeCapacity = std::max(rangeCapacity_, kInitialRangeCapacity);
    while (nextRangeCapacity < requiredRanges) {
        nextRangeCapacity = std::max(requiredRanges, nextRangeCapacity * 2u);
    }

    if (!recreateBuffers(nextMeshletCapacity, nextQuadWordCapacity, nextRangeCapacity)) {
        return false;
    }

    if (!repackExistingAllocations()) {
        return false;
    }

    if (recreated != nullptr) {
        *recreated = true;
    }
    return true;
}

MeshletBufferController::ApplyResult MeshletBufferController::applyDelta(const MeshStreamingDelta& delta) {
    ApplyResult result{};

    if (bufferManager_ == nullptr) {
        return result;
    }

    for (const MeshTileLodKey& key : delta.removals) {
        releaseAllocation(key);
        result.deltaApplied = true;
    }

    for (const MeshTileLodUpload& upsert : delta.upserts) {
        if (upsert.meshlets.empty()) {
            releaseAllocation(upsert.key);
            result.deltaApplied = true;
            continue;
        }

        AllocationRecord record{};
        record.key = upsert.key;
        record.packed = packTileLodMeshlets(upsert.meshlets);

        if (record.packed.metadata.empty()) {
            releaseAllocation(upsert.key);
            result.deltaApplied = true;
            continue;
        }

        const bool hasPrevious = allocations_.find(upsert.key) != allocations_.end();

        uint32_t meshletOffset = 0u;
        uint32_t quadOffset = 0u;
        const uint32_t requiredMeshletWords = static_cast<uint32_t>(record.packed.metadata.size());
        const uint32_t requiredQuadWords = static_cast<uint32_t>(record.packed.quadData.size());
        const bool meshletAllocated = allocateRange(meshletFreeRanges_, requiredMeshletWords, meshletOffset);
        const bool quadAllocated = meshletAllocated && allocateRange(quadFreeRanges_, requiredQuadWords, quadOffset);
        if (!meshletAllocated || !quadAllocated) {
            if (meshletAllocated) {
                freeRange(meshletFreeRanges_, meshletOffset, requiredMeshletWords);
            }
            if (quadAllocated) {
                freeRange(quadFreeRanges_, quadOffset, requiredQuadWords);
            }

            bool recreated = false;
            const uint32_t requiredMeshlets = meshletCapacity_ + static_cast<uint32_t>(record.packed.metadata.size()) + 64u;
            const uint32_t requiredQuadWords = quadWordCapacity_ + static_cast<uint32_t>(record.packed.quadData.size()) + 1024u;
            const uint32_t requiredRanges = std::max<uint32_t>(
                rangeCapacity_,
                static_cast<uint32_t>(tileSelection_.size() + 64u)
            );
            if (!ensureBuffers(requiredMeshlets, requiredQuadWords, requiredRanges, &recreated)) {
                continue;
            }
            result.buffersRecreated = result.buffersRecreated || recreated;

            if (!allocateRange(meshletFreeRanges_, requiredMeshletWords, meshletOffset) ||
                !allocateRange(quadFreeRanges_, requiredQuadWords, quadOffset)) {
                continue;
            }
        }

        record.meshletOffset = meshletOffset;
        record.quadOffset = quadOffset;
        if (!writeAllocation(record)) {
            freeRange(meshletFreeRanges_, meshletOffset, static_cast<uint32_t>(record.packed.metadata.size()));
            freeRange(quadFreeRanges_, quadOffset, static_cast<uint32_t>(record.packed.quadData.size()));
            continue;
        }

        if (hasPrevious) {
            const auto currentOldIt = allocations_.find(upsert.key);
            if (currentOldIt != allocations_.end()) {
                freeRange(meshletFreeRanges_, currentOldIt->second.meshletOffset, static_cast<uint32_t>(currentOldIt->second.packed.metadata.size()));
                freeRange(quadFreeRanges_, currentOldIt->second.quadOffset, static_cast<uint32_t>(currentOldIt->second.packed.quadData.size()));
                allocations_.erase(currentOldIt);
            }
        }

        allocations_[upsert.key] = std::move(record);
        result.deltaApplied = true;
    }

    if (!delta.selectionSnapshot.empty()) {
        tileSelection_.clear();
        for (const MeshTileSelectionEntry& entry : delta.selectionSnapshot) {
            if (entry.selectedLod >= 0) {
                tileSelection_[entry.tile] = entry.selectedLod;
            }
        }
        result.deltaApplied = true;
    }

    bool recreatedForRanges = false;
    if (!ensureBuffers(meshletCapacity_, quadWordCapacity_, std::max<uint32_t>(1u, static_cast<uint32_t>(tileSelection_.size())), &recreatedForRanges)) {
        return result;
    }
    result.buffersRecreated = result.buffersRecreated || recreatedForRanges;

    buildActiveRanges();

    if (delta.revision > uploadedMeshRevision_) {
        uploadedMeshRevision_ = delta.revision;
    }

    return result;
}

bool MeshletBufferController::buildActiveRanges() {
    if (bufferManager_ == nullptr) {
        return false;
    }

    activeRanges_.clear();
    activeMeshletBounds_.clear();
    activeSelectionMeshletCount_ = 0u;

    std::vector<std::pair<MeshTileCoord, int8_t>> orderedSelection;
    orderedSelection.reserve(tileSelection_.size());
    for (const auto& [tile, lod] : tileSelection_) {
        orderedSelection.emplace_back(tile, lod);
    }
    std::sort(orderedSelection.begin(), orderedSelection.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    for (const auto& [tile, lod] : orderedSelection) {
        if (lod < 0) {
            continue;
        }

        const MeshTileLodKey key{tile, static_cast<uint8_t>(lod)};
        const auto it = allocations_.find(key);
        if (it == allocations_.end()) {
            continue;
        }

        const uint32_t count = static_cast<uint32_t>(it->second.packed.metadata.size());
        if (count == 0u) {
            continue;
        }

        activeSelectionMeshletCount_ += count;
        activeRanges_.push_back(ActiveMeshletRangeGPU{
            it->second.meshletOffset,
            count,
            activeSelectionMeshletCount_,
            0u
        });

        activeMeshletBounds_.insert(
            activeMeshletBounds_.end(),
            it->second.packed.bounds.begin(),
            it->second.packed.bounds.end()
        );
    }

    bool recreated = false;
    if (!ensureBuffers(meshletCapacity_, quadWordCapacity_, std::max<uint32_t>(1u, static_cast<uint32_t>(activeRanges_.size())), &recreated)) {
        return false;
    }

    if (!activeRanges_.empty()) {
        bufferManager_->writeBuffer(
            kActiveMeshletRangeBufferName,
            0u,
            activeRanges_.data(),
            activeRanges_.size() * sizeof(ActiveMeshletRangeGPU)
        );
    }

    const RangeParamsGPU rangeParams{
        static_cast<uint32_t>(activeRanges_.size()),
        activeSelectionMeshletCount_,
        0u,
        0u
    };
    bufferManager_->writeBuffer(
        kActiveMeshletRangeParamsBufferName,
        0u,
        &rangeParams,
        sizeof(rangeParams)
    );

    return true;
}

bool MeshletBufferController::hasMeshletManager() const noexcept {
    return bufferManager_ != nullptr;
}

const char* MeshletBufferController::activeMeshDataBufferName() const noexcept {
    return kMeshDataBufferName;
}

const char* MeshletBufferController::activeMeshMetadataBufferName() const noexcept {
    return kMeshMetadataBufferName;
}

const char* MeshletBufferController::activeMeshAabbBufferName() const noexcept {
    return kMeshAabbBufferName;
}

const char* MeshletBufferController::activeVisibleMeshletIndexBufferName() const noexcept {
    return kVisibleMeshletIndexBufferName;
}

const char* MeshletBufferController::activeMeshletRangeBufferName() const noexcept {
    return kActiveMeshletRangeBufferName;
}

const char* MeshletBufferController::activeMeshletRangeParamsBufferName() const noexcept {
    return kActiveMeshletRangeParamsBufferName;
}

uint32_t MeshletBufferController::meshletCount() const noexcept {
    return activeSelectionMeshletCount_;
}

uint32_t MeshletBufferController::activeSelectionMeshletCount() const noexcept {
    return activeSelectionMeshletCount_;
}

uint32_t MeshletBufferController::activeRangeCount() const noexcept {
    return static_cast<uint32_t>(activeRanges_.size());
}

uint32_t MeshletBufferController::verticesPerMeshlet() const noexcept {
    return MESHLET_VERTEX_CAPACITY;
}

uint64_t MeshletBufferController::uploadedMeshRevision() const noexcept {
    return uploadedMeshRevision_;
}

const std::vector<MeshletAabb>& MeshletBufferController::activeMeshletBounds() const noexcept {
    return activeMeshletBounds_;
}

MeshletBufferController::ActiveBindings MeshletBufferController::activeBindings() const noexcept {
    return ActiveBindings{
        activeMeshDataBufferName(),
        activeMeshMetadataBufferName(),
        activeMeshAabbBufferName(),
        activeVisibleMeshletIndexBufferName(),
        activeMeshletRangeBufferName(),
        activeMeshletRangeParamsBufferName(),
        meshletCount(),
        activeRangeCount(),
        verticesPerMeshlet()
    };
}
