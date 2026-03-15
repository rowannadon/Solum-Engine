#include "solum_engine/render/MeshletBufferController.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>

using namespace wgpu;

MeshletBufferController::MeshletBufferController()
    : MeshletBufferController(Config{}) {}

MeshletBufferController::MeshletBufferController(Config config)
    : namePrefix_(std::move(config.namePrefix)),
      geometryVariant_(config.geometryVariant),
      meshDataBufferName_(prefixedName(namePrefix_, "meshlet_data_buffer")),
      meshMetadataBufferName_(prefixedName(namePrefix_, "meshlet_metadata_buffer")),
      meshAabbBufferName_(prefixedName(namePrefix_, "meshlet_aabb_buffer")),
      visibleMeshletIndexBufferName_(prefixedName(namePrefix_, "visible_meshlet_indices_buffer")),
      activeMeshletRangeBufferName_(prefixedName(namePrefix_, "active_meshlet_ranges_buffer")),
      activeMeshletRangeParamsBufferName_(prefixedName(namePrefix_, "active_meshlet_ranges_params_buffer")) {}

std::string MeshletBufferController::prefixedName(const std::string& prefix, const char* baseName) {
    if (prefix.empty()) {
        return std::string(baseName);
    }
    return prefix + "_" + baseName;
}

bool MeshletBufferController::initialize(BufferManager* bufferManager) {
    bufferManager_ = bufferManager;
    allocations_.clear();
    tileSelection_.clear();
    activeRanges_.clear();
    activeSelectionKeys_.clear();
    activeMeshletBoundsCache_.clear();
    activeMeshletBoundsDirty_ = true;
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
    if (!bufferManager_->createBuffer(meshMetadataBufferName_, metadataDesc)) {
        return false;
    }

    BufferDescriptor meshDataDesc = Default;
    meshDataDesc.label = StringView("meshlet data buffer");
    meshDataDesc.size = static_cast<uint64_t>(quadWordCapacity) * sizeof(uint32_t);
    meshDataDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    meshDataDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(meshDataBufferName_, meshDataDesc)) {
        return false;
    }

    BufferDescriptor aabbDesc = Default;
    aabbDesc.label = StringView("meshlet aabb buffer");
    aabbDesc.size = static_cast<uint64_t>(meshletCapacity) * sizeof(MeshletAabbGPU);
    aabbDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    aabbDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(meshAabbBufferName_, aabbDesc)) {
        return false;
    }

    BufferDescriptor visibleIndicesDesc = Default;
    visibleIndicesDesc.label = StringView("visible meshlet indices buffer");
    visibleIndicesDesc.size = static_cast<uint64_t>(meshletCapacity) * sizeof(uint32_t);
    visibleIndicesDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    visibleIndicesDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(visibleMeshletIndexBufferName_, visibleIndicesDesc)) {
        return false;
    }

    BufferDescriptor activeRangesDesc = Default;
    activeRangesDesc.label = StringView("active meshlet ranges buffer");
    activeRangesDesc.size = static_cast<uint64_t>(rangeCapacity) * sizeof(ActiveMeshletRangeGPU);
    activeRangesDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    activeRangesDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(activeMeshletRangeBufferName_, activeRangesDesc)) {
        return false;
    }

    BufferDescriptor rangeParamsDesc = Default;
    rangeParamsDesc.label = StringView("active meshlet ranges params buffer");
    rangeParamsDesc.size = sizeof(RangeParamsGPU);
    rangeParamsDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
    rangeParamsDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(activeMeshletRangeParamsBufferName_, rangeParamsDesc)) {
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

    if (!record.packed) {
        return true;
    }
    const PackedMeshletData& packed = *record.packed;

    if (!packed.metadata.empty()) {
        std::vector<MeshletMetadataGPU> metadata = packed.metadata;
        for (MeshletMetadataGPU& entry : metadata) {
            entry.dataOffset += record.quadOffset;
        }

        bufferManager_->writeBuffer(
            meshMetadataBufferName_,
            static_cast<uint64_t>(record.meshletOffset) * sizeof(MeshletMetadataGPU),
            metadata.data(),
            metadata.size() * sizeof(MeshletMetadataGPU)
        );

        bufferManager_->writeBuffer(
            meshAabbBufferName_,
            static_cast<uint64_t>(record.meshletOffset) * sizeof(MeshletAabbGPU),
            packed.aabbGpu.data(),
            packed.aabbGpu.size() * sizeof(MeshletAabbGPU)
        );
    }

    if (!packed.quadData.empty()) {
        bufferManager_->writeBuffer(
            meshDataBufferName_,
            static_cast<uint64_t>(record.quadOffset) * sizeof(uint32_t),
            packed.quadData.data(),
            packed.quadData.size() * sizeof(uint32_t)
        );
    }

    return true;
}

void MeshletBufferController::releaseAllocation(const MeshTileLodKey& key) {
    const auto it = allocations_.find(key);
    if (it == allocations_.end()) {
        return;
    }

    freeRange(meshletFreeRanges_, it->second.meshletOffset, it->second.reservedMeshletCount);
    freeRange(quadFreeRanges_, it->second.quadOffset, it->second.reservedQuadWordCount);
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
        const uint32_t meshletCount = record.packed
            ? static_cast<uint32_t>(record.packed->metadata.size())
            : 0u;
        const uint32_t quadWordCount = record.packed
            ? static_cast<uint32_t>(record.packed->quadData.size())
            : 0u;
        if (!allocateRange(meshletFreeRanges_, meshletCount, record.meshletOffset)) {
            return false;
        }
        if (!allocateRange(quadFreeRanges_, quadWordCount, record.quadOffset)) {
            freeRange(meshletFreeRanges_, record.meshletOffset, meshletCount);
            return false;
        }
        record.reservedMeshletCount = meshletCount;
        record.reservedQuadWordCount = quadWordCount;
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
        !bufferManager_->getBuffer(meshDataBufferName_) ||
        !bufferManager_->getBuffer(meshMetadataBufferName_) ||
        !bufferManager_->getBuffer(meshAabbBufferName_) ||
        !bufferManager_->getBuffer(visibleMeshletIndexBufferName_) ||
        !bufferManager_->getBuffer(activeMeshletRangeBufferName_) ||
        !bufferManager_->getBuffer(activeMeshletRangeParamsBufferName_);

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

    auto selectionUsesKey = [this](const MeshTileLodKey& key) {
        const auto selectionIt = tileSelection_.find(key.tile);
        return selectionIt != tileSelection_.end() && selectionIt->second == static_cast<int8_t>(key.lod);
    };
    bool activeRangesDirty = false;

    for (const MeshTileLodKey& key : delta.removals) {
        activeRangesDirty = activeRangesDirty || selectionUsesKey(key);
        releaseAllocation(key);
        result.deltaApplied = true;
    }

    for (const MeshTileLodUpload& upsert : delta.upserts) {
        const std::shared_ptr<const PackedMeshletData>& packed = selectPackedForVariant(upsert);
        if (!packed || packed->metadata.empty()) {
            activeRangesDirty = activeRangesDirty || selectionUsesKey(upsert.key);
            releaseAllocation(upsert.key);
            result.deltaApplied = true;
            continue;
        }

        AllocationRecord record{};
        record.key = upsert.key;
        record.packed = packed;

        const auto previousIt = allocations_.find(upsert.key);
        const bool hasPrevious = previousIt != allocations_.end();

        uint32_t meshletOffset = 0u;
        uint32_t quadOffset = 0u;
        const uint32_t requiredMeshletCount = static_cast<uint32_t>(record.packed->metadata.size());
        const uint32_t requiredQuadWords = static_cast<uint32_t>(record.packed->quadData.size());
        bool reusedPreviousReservation = false;

        if (hasPrevious &&
            previousIt->second.reservedMeshletCount >= requiredMeshletCount &&
            previousIt->second.reservedQuadWordCount >= requiredQuadWords) {
            meshletOffset = previousIt->second.meshletOffset;
            quadOffset = previousIt->second.quadOffset;
            record.reservedMeshletCount = previousIt->second.reservedMeshletCount;
            record.reservedQuadWordCount = previousIt->second.reservedQuadWordCount;
            reusedPreviousReservation = true;
        } else {
            const bool meshletAllocated = allocateRange(meshletFreeRanges_, requiredMeshletCount, meshletOffset);
            const bool quadAllocated =
                meshletAllocated && allocateRange(quadFreeRanges_, requiredQuadWords, quadOffset);
            if (!meshletAllocated || !quadAllocated) {
                if (meshletAllocated) {
                    freeRange(meshletFreeRanges_, meshletOffset, requiredMeshletCount);
                }
                if (quadAllocated) {
                    freeRange(quadFreeRanges_, quadOffset, requiredQuadWords);
                }

                bool recreated = false;
                const uint32_t requiredMeshlets = meshletCapacity_ + requiredMeshletCount + 64u;
                const uint32_t requiredQuadWordsTotal = quadWordCapacity_ + requiredQuadWords + 1024u;
                const uint32_t requiredRanges = std::max<uint32_t>(
                    rangeCapacity_,
                    static_cast<uint32_t>(tileSelection_.size() + 64u)
                );
                if (!ensureBuffers(requiredMeshlets, requiredQuadWordsTotal, requiredRanges, &recreated)) {
                    continue;
                }
                result.buffersRecreated = result.buffersRecreated || recreated;
                activeRangesDirty = activeRangesDirty || recreated;

                const bool meshletAllocatedAfterRecreate =
                    allocateRange(meshletFreeRanges_, requiredMeshletCount, meshletOffset);
                const bool quadAllocatedAfterRecreate =
                    meshletAllocatedAfterRecreate &&
                    allocateRange(quadFreeRanges_, requiredQuadWords, quadOffset);
                if (!meshletAllocatedAfterRecreate || !quadAllocatedAfterRecreate) {
                    if (meshletAllocatedAfterRecreate) {
                        freeRange(meshletFreeRanges_, meshletOffset, requiredMeshletCount);
                    }
                    continue;
                }
            }
            record.reservedMeshletCount = requiredMeshletCount;
            record.reservedQuadWordCount = requiredQuadWords;
        }

        record.meshletOffset = meshletOffset;
        record.quadOffset = quadOffset;
        if (!writeAllocation(record)) {
            if (!reusedPreviousReservation) {
                freeRange(meshletFreeRanges_, meshletOffset, record.reservedMeshletCount);
                freeRange(quadFreeRanges_, quadOffset, record.reservedQuadWordCount);
            }
            continue;
        }

        if (hasPrevious && !reusedPreviousReservation) {
            const auto currentOldIt = allocations_.find(upsert.key);
            if (currentOldIt != allocations_.end()) {
                freeRange(meshletFreeRanges_, currentOldIt->second.meshletOffset, currentOldIt->second.reservedMeshletCount);
                freeRange(quadFreeRanges_, currentOldIt->second.quadOffset, currentOldIt->second.reservedQuadWordCount);
                allocations_.erase(currentOldIt);
            }
        }

        allocations_[upsert.key] = std::move(record);
        result.deltaApplied = true;
        activeRangesDirty = activeRangesDirty || selectionUsesKey(upsert.key);

        // Stop uploading if per-frame byte budget is exceeded to avoid
        // overwhelming the Metal command queue on macOS (IOFence deadlock).
        // Skipped upserts will reappear in the next streaming delta.
        if (bufferManager_->isOverBudget()) {
            break;
        }
    }

    if (!delta.selectionChanges.empty()) {
        for (const MeshTileSelectionEntry& entry : delta.selectionChanges) {
            if (entry.selectedLod >= 0) {
                tileSelection_[entry.tile] = entry.selectedLod;
            } else {
                tileSelection_.erase(entry.tile);
            }
        }
        result.deltaApplied = true;
        activeRangesDirty = true;
    }

    if (activeRangesDirty) {
        bool recreatedForRanges = false;
        if (!ensureBuffers(
                meshletCapacity_,
                quadWordCapacity_,
                std::max<uint32_t>(1u, static_cast<uint32_t>(tileSelection_.size())),
                &recreatedForRanges)) {
            return result;
        }
        result.buffersRecreated = result.buffersRecreated || recreatedForRanges;
        if (!buildActiveRanges()) {
            return result;
        }
    }

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
    activeSelectionKeys_.clear();
    activeMeshletBoundsDirty_ = true;
    activeSelectionMeshletCount_ = 0u;

    std::vector<std::pair<MeshTileSliceCoord, int8_t>> orderedSelection;
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
        if (!it->second.packed) {
            continue;
        }

        const uint32_t count = static_cast<uint32_t>(it->second.packed->metadata.size());
        if (count == 0u) {
            continue;
        }

        // Cap total meshlets per controller to prevent a single command buffer
        // from monopolizing the GPU on macOS (no preemption like WDDM).
        if (activeSelectionMeshletCount_ + count > maxActiveMeshlets_) {
            break;
        }

        activeSelectionMeshletCount_ += count;
        activeSelectionKeys_.push_back(key);
        activeRanges_.push_back(ActiveMeshletRangeGPU{
            it->second.meshletOffset,
            count,
            activeSelectionMeshletCount_,
            0u
        });
    }

    bool recreated = false;
    if (!ensureBuffers(meshletCapacity_, quadWordCapacity_, std::max<uint32_t>(1u, static_cast<uint32_t>(activeRanges_.size())), &recreated)) {
        return false;
    }

    if (!activeRanges_.empty()) {
        bufferManager_->writeBuffer(
            activeMeshletRangeBufferName_,
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
        activeMeshletRangeParamsBufferName_,
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
    return meshDataBufferName_.c_str();
}

const char* MeshletBufferController::activeMeshMetadataBufferName() const noexcept {
    return meshMetadataBufferName_.c_str();
}

const char* MeshletBufferController::activeMeshAabbBufferName() const noexcept {
    return meshAabbBufferName_.c_str();
}

const char* MeshletBufferController::activeVisibleMeshletIndexBufferName() const noexcept {
    return visibleMeshletIndexBufferName_.c_str();
}

const char* MeshletBufferController::activeMeshletRangeBufferName() const noexcept {
    return activeMeshletRangeBufferName_.c_str();
}

const char* MeshletBufferController::activeMeshletRangeParamsBufferName() const noexcept {
    return activeMeshletRangeParamsBufferName_.c_str();
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
    if (!activeMeshletBoundsDirty_) {
        return activeMeshletBoundsCache_;
    }

    activeMeshletBoundsCache_.clear();
    activeMeshletBoundsCache_.reserve(activeSelectionMeshletCount_);
    for (const MeshTileLodKey& key : activeSelectionKeys_) {
        const auto allocationIt = allocations_.find(key);
        if (allocationIt == allocations_.end() || !allocationIt->second.packed) {
            continue;
        }
        activeMeshletBoundsCache_.insert(
            activeMeshletBoundsCache_.end(),
            allocationIt->second.packed->bounds.begin(),
            allocationIt->second.packed->bounds.end()
        );
    }
    activeMeshletBoundsDirty_ = false;
    return activeMeshletBoundsCache_;
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

const std::shared_ptr<const PackedMeshletData>& MeshletBufferController::selectPackedForVariant(
    const MeshTileLodUpload& upload
) const {
    if (geometryVariant_ == MeshletGeometryVariant::DoubleSided) {
        return upload.doubleSidedPacked;
    }
    return upload.culledPacked;
}
