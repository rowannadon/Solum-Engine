#include "solum_engine/render/MeshletBufferController.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

bool MeshletBufferController::initialize(BufferManager* bufferManager) {
    bufferManager_ = bufferManager;

    tileSlots_.clear();
    freeSlotIndices_.clear();
    coordToSlot_.clear();
    activeSlotIndices_.clear();
    slotToActivePos_.clear();
    activeRanges_.clear();
    activeMeshletBoundsCache_.clear();
    activeMeshletBoundsDirty_ = true;
    activeRangesDirty_ = true;
    totalActiveMeshlets_ = 0u;
    uploadedMeshRevision_ = 0u;

    if (bufferManager_ == nullptr) {
        return false;
    }

    // Pre-allocate tile slots.
    tileSlots_.resize(kInitialTileSlotCapacity);
    freeSlotIndices_.reserve(kInitialTileSlotCapacity);
    for (uint32_t i = kInitialTileSlotCapacity; i > 0u; --i) {
        freeSlotIndices_.push_back(i - 1u);
    }

    bool recreated = false;
    if (!ensureBuffers(kInitialMeshletCapacity, kInitialQuadWordCapacity, kInitialRangeCapacity, &recreated)) {
        return false;
    }

    return buildActiveRanges();
}

// ---------------------------------------------------------------------------
// Tile Slot Management
// ---------------------------------------------------------------------------

void MeshletBufferController::ensureTileSlotCapacity() {
    if (!freeSlotIndices_.empty()) {
        return;
    }
    const uint32_t oldSize = static_cast<uint32_t>(tileSlots_.size());
    const uint32_t newSize = std::max(oldSize * 2u, 64u);
    tileSlots_.resize(newSize);
    freeSlotIndices_.reserve(newSize - oldSize);
    for (uint32_t i = newSize; i > oldSize; --i) {
        freeSlotIndices_.push_back(i - 1u);
    }
}

uint32_t MeshletBufferController::acquireTileSlot(const MeshTileSliceCoord& coord) {
    auto it = coordToSlot_.find(coord);
    if (it != coordToSlot_.end()) {
        return it->second;
    }

    ensureTileSlotCapacity();

    const uint32_t slotIndex = freeSlotIndices_.back();
    freeSlotIndices_.pop_back();

    TileSlot& slot = tileSlots_[slotIndex];
    slot = TileSlot{};
    slot.coord = coord;
    coordToSlot_[coord] = slotIndex;
    return slotIndex;
}

void MeshletBufferController::releaseTileSlot(uint32_t slotIndex) {
    TileSlot& slot = tileSlots_[slotIndex];

    // Hide from active list if present.
    hideTileSlot(slotIndex);

    // Free all LOD allocations.
    for (uint32_t lod = 0u; lod < kMaxLods; ++lod) {
        LodAllocation& alloc = slot.lods[lod];
        if (!alloc.resident) continue;
        meshletAllocator_.free(alloc.meshletOffset, alloc.reservedMeshletCount);
        quadDataAllocator_.free(alloc.quadDataOffset, alloc.reservedQuadWordCount);
        alloc = LodAllocation{};
    }

    coordToSlot_.erase(slot.coord);
    slot = TileSlot{};
    freeSlotIndices_.push_back(slotIndex);
}

// ---------------------------------------------------------------------------
// Active List Management (O(1) operations)
// ---------------------------------------------------------------------------

void MeshletBufferController::showTileSlot(uint32_t slotIndex) {
    TileSlot& slot = tileSlots_[slotIndex];
    if (slot.inActiveList || slot.activeLod < 0) return;

    const auto& lod = slot.lods[static_cast<uint8_t>(slot.activeLod)];
    if (!lod.resident || lod.meshletCount == 0u) return;

    const uint32_t pos = static_cast<uint32_t>(activeSlotIndices_.size());
    activeSlotIndices_.push_back(slotIndex);
    slotToActivePos_[slotIndex] = pos;
    slot.inActiveList = true;

    totalActiveMeshlets_ += lod.meshletCount;
    activeRangesDirty_ = true;
    activeMeshletBoundsDirty_ = true;
}

void MeshletBufferController::hideTileSlot(uint32_t slotIndex) {
    TileSlot& slot = tileSlots_[slotIndex];
    if (!slot.inActiveList) return;

    if (slot.activeLod >= 0) {
        const auto& lod = slot.lods[static_cast<uint8_t>(slot.activeLod)];
        if (lod.resident) {
            totalActiveMeshlets_ -= lod.meshletCount;
        }
    }

    // O(1) swap-and-pop removal.
    const auto posIt = slotToActivePos_.find(slotIndex);
    if (posIt != slotToActivePos_.end()) {
        const uint32_t pos = posIt->second;
        if (pos < activeSlotIndices_.size() - 1u) {
            const uint32_t lastSlot = activeSlotIndices_.back();
            activeSlotIndices_[pos] = lastSlot;
            slotToActivePos_[lastSlot] = pos;
        }
        activeSlotIndices_.pop_back();
        slotToActivePos_.erase(posIt);
    }

    slot.inActiveList = false;
    activeRangesDirty_ = true;
    activeMeshletBoundsDirty_ = true;
}

void MeshletBufferController::switchTileLod(uint32_t slotIndex, int8_t newLod) {
    TileSlot& slot = tileSlots_[slotIndex];
    if (slot.activeLod == newLod) return;

    const bool wasActive = slot.inActiveList;

    // Remove contribution of old LOD.
    if (wasActive && slot.activeLod >= 0) {
        const auto& oldLod = slot.lods[static_cast<uint8_t>(slot.activeLod)];
        if (oldLod.resident) {
            totalActiveMeshlets_ -= oldLod.meshletCount;
        }
    }

    slot.activeLod = newLod;

    // If the new LOD is valid and resident, update contribution.
    if (wasActive && newLod >= 0) {
        const auto& newLodAlloc = slot.lods[static_cast<uint8_t>(newLod)];
        if (newLodAlloc.resident && newLodAlloc.meshletCount > 0u) {
            totalActiveMeshlets_ += newLodAlloc.meshletCount;
        } else {
            // New LOD not resident: remove from active list.
            hideTileSlot(slotIndex);
            return;
        }
    } else if (wasActive && newLod < 0) {
        // Deselected: remove from active list.
        hideTileSlot(slotIndex);
        return;
    }

    activeRangesDirty_ = true;
    activeMeshletBoundsDirty_ = true;
}

// ---------------------------------------------------------------------------
// GPU Buffer Management
// ---------------------------------------------------------------------------

bool MeshletBufferController::recreateBuffers(uint32_t meshletCapacity,
                                              uint32_t quadWordCapacity,
                                              uint32_t rangeCapacity) {
    if (bufferManager_ == nullptr || meshletCapacity == 0u ||
        quadWordCapacity == 0u || rangeCapacity == 0u) {
        return false;
    }

    BufferDescriptor metadataDesc = Default;
    metadataDesc.label = StringView("meshlet metadata buffer");
    metadataDesc.size = static_cast<uint64_t>(meshletCapacity) * sizeof(MeshletMetadataGPU);
    metadataDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    metadataDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(meshMetadataBufferName_, metadataDesc)) return false;

    BufferDescriptor meshDataDesc = Default;
    meshDataDesc.label = StringView("meshlet data buffer");
    meshDataDesc.size = static_cast<uint64_t>(quadWordCapacity) * sizeof(uint32_t);
    meshDataDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    meshDataDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(meshDataBufferName_, meshDataDesc)) return false;

    BufferDescriptor aabbDesc = Default;
    aabbDesc.label = StringView("meshlet aabb buffer");
    aabbDesc.size = static_cast<uint64_t>(meshletCapacity) * sizeof(MeshletAabbGPU);
    aabbDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    aabbDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(meshAabbBufferName_, aabbDesc)) return false;

    BufferDescriptor visibleIndicesDesc = Default;
    visibleIndicesDesc.label = StringView("visible meshlet indices buffer");
    visibleIndicesDesc.size = static_cast<uint64_t>(meshletCapacity) * sizeof(uint32_t);
    visibleIndicesDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    visibleIndicesDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(visibleMeshletIndexBufferName_, visibleIndicesDesc)) return false;

    BufferDescriptor activeRangesDesc = Default;
    activeRangesDesc.label = StringView("active meshlet ranges buffer");
    activeRangesDesc.size = static_cast<uint64_t>(rangeCapacity) * sizeof(ActiveMeshletRangeGPU);
    activeRangesDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    activeRangesDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(activeMeshletRangeBufferName_, activeRangesDesc)) return false;

    BufferDescriptor rangeParamsDesc = Default;
    rangeParamsDesc.label = StringView("active meshlet ranges params buffer");
    rangeParamsDesc.size = sizeof(RangeParamsGPU);
    rangeParamsDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
    rangeParamsDesc.mappedAtCreation = false;
    if (!bufferManager_->createBuffer(activeMeshletRangeParamsBufferName_, rangeParamsDesc)) return false;

    meshletCapacity_ = meshletCapacity;
    quadWordCapacity_ = quadWordCapacity;
    rangeCapacity_ = rangeCapacity;

    meshletAllocator_.reset(meshletCapacity_);
    quadDataAllocator_.reset(quadWordCapacity_);

    return true;
}

bool MeshletBufferController::repackExistingAllocations() {
    // Collect all resident allocations across all tile slots.
    struct RepackEntry {
        uint32_t slotIndex;
        uint8_t lod;
    };
    std::vector<RepackEntry> entries;
    for (uint32_t si = 0u; si < static_cast<uint32_t>(tileSlots_.size()); ++si) {
        const TileSlot& slot = tileSlots_[si];
        // Skip slots that aren't in use (not in coordToSlot_).
        if (coordToSlot_.find(slot.coord) == coordToSlot_.end() ||
            coordToSlot_.at(slot.coord) != si) {
            continue;
        }
        for (uint32_t l = 0u; l < kMaxLods; ++l) {
            if (slot.lods[l].resident) {
                entries.push_back({si, static_cast<uint8_t>(l)});
            }
        }
    }

    // Reset allocators.
    meshletAllocator_.reset(meshletCapacity_);
    quadDataAllocator_.reset(quadWordCapacity_);

    // Re-allocate and re-upload each entry.
    for (const RepackEntry& entry : entries) {
        LodAllocation& alloc = tileSlots_[entry.slotIndex].lods[entry.lod];
        if (!alloc.packed) {
            alloc.resident = false;
            continue;
        }

        const uint32_t meshletCount = static_cast<uint32_t>(alloc.packed->metadata.size());
        const uint32_t quadWordCount = static_cast<uint32_t>(alloc.packed->quadData.size());

        uint32_t newMeshletOffset = 0u;
        uint32_t newQuadOffset = 0u;
        if (!meshletAllocator_.allocate(meshletCount, newMeshletOffset)) return false;
        if (!quadDataAllocator_.allocate(quadWordCount, newQuadOffset)) {
            meshletAllocator_.free(newMeshletOffset, meshletCount);
            return false;
        }

        alloc.meshletOffset = newMeshletOffset;
        alloc.meshletCount = meshletCount;
        alloc.reservedMeshletCount = meshletCount;
        alloc.quadDataOffset = newQuadOffset;
        alloc.quadDataWordCount = quadWordCount;
        alloc.reservedQuadWordCount = quadWordCount;

        if (!writeLodAllocation(newMeshletOffset, newQuadOffset, *alloc.packed)) {
            return false;
        }
    }

    return true;
}

bool MeshletBufferController::ensureBuffers(uint32_t requiredMeshlets,
                                            uint32_t requiredQuadWords,
                                            uint32_t requiredRanges,
                                            bool* recreated) {
    if (recreated != nullptr) *recreated = false;
    if (bufferManager_ == nullptr) return false;

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

    if (!needsRecreate) return true;

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

    if (recreated != nullptr) *recreated = true;
    return true;
}

// ---------------------------------------------------------------------------
// GPU Data Writes
// ---------------------------------------------------------------------------

bool MeshletBufferController::writeLodAllocation(uint32_t meshletOffset,
                                                 uint32_t quadDataOffset,
                                                 const PackedMeshletData& packed) {
    if (bufferManager_ == nullptr) return false;

    if (!packed.metadata.empty()) {
        // Patch dataOffset in metadata to reflect the quad data position.
        std::vector<MeshletMetadataGPU> metadata = packed.metadata;
        for (MeshletMetadataGPU& entry : metadata) {
            entry.dataOffset += quadDataOffset;
        }

        bufferManager_->writeBuffer(
            meshMetadataBufferName_,
            static_cast<uint64_t>(meshletOffset) * sizeof(MeshletMetadataGPU),
            metadata.data(),
            metadata.size() * sizeof(MeshletMetadataGPU));

        bufferManager_->writeBuffer(
            meshAabbBufferName_,
            static_cast<uint64_t>(meshletOffset) * sizeof(MeshletAabbGPU),
            packed.aabbGpu.data(),
            packed.aabbGpu.size() * sizeof(MeshletAabbGPU));
    }

    if (!packed.quadData.empty()) {
        bufferManager_->writeBuffer(
            meshDataBufferName_,
            static_cast<uint64_t>(quadDataOffset) * sizeof(uint32_t),
            packed.quadData.data(),
            packed.quadData.size() * sizeof(uint32_t));
    }

    return true;
}

// ---------------------------------------------------------------------------
// applyDelta -- processes streaming upserts, removals, and selection changes
// ---------------------------------------------------------------------------

MeshletBufferController::ApplyResult MeshletBufferController::applyDelta(const MeshStreamingDelta& delta) {
    ApplyResult result{};
    if (bufferManager_ == nullptr) return result;

    // --- Removals ---
    for (const MeshTileLodKey& key : delta.removals) {
        auto coordIt = coordToSlot_.find(key.tile);
        if (coordIt == coordToSlot_.end()) continue;

        const uint32_t slotIndex = coordIt->second;
        TileSlot& slot = tileSlots_[slotIndex];
        if (key.lod >= kMaxLods) continue;

        LodAllocation& alloc = slot.lods[key.lod];
        if (alloc.resident) {
            // If this was the active LOD, hide before freeing.
            if (slot.activeLod == static_cast<int8_t>(key.lod)) {
                hideTileSlot(slotIndex);
                slot.activeLod = -1;
            }
            meshletAllocator_.free(alloc.meshletOffset, alloc.reservedMeshletCount);
            quadDataAllocator_.free(alloc.quadDataOffset, alloc.reservedQuadWordCount);
            alloc = LodAllocation{};
        }

        // Check if this slot has any remaining LODs. If not, release the slot.
        bool anyResident = false;
        for (uint32_t l = 0u; l < kMaxLods; ++l) {
            if (slot.lods[l].resident) { anyResident = true; break; }
        }
        if (!anyResident && slot.activeLod < 0) {
            releaseTileSlot(slotIndex);
        }

        result.deltaApplied = true;
    }

    // --- Upserts ---
    for (const MeshTileLodUpload& upsert : delta.upserts) {
        const std::shared_ptr<const PackedMeshletData>& packed = selectPackedForVariant(upsert);
        if (!packed || packed->metadata.empty()) {
            // Empty mesh: treat as removal of this LOD.
            auto coordIt = coordToSlot_.find(upsert.key.tile);
            if (coordIt != coordToSlot_.end()) {
                const uint32_t si = coordIt->second;
                TileSlot& slot = tileSlots_[si];
                if (upsert.key.lod < kMaxLods) {
                    LodAllocation& alloc = slot.lods[upsert.key.lod];
                    if (alloc.resident) {
                        if (slot.activeLod == static_cast<int8_t>(upsert.key.lod)) {
                            hideTileSlot(si);
                        }
                        meshletAllocator_.free(alloc.meshletOffset, alloc.reservedMeshletCount);
                        quadDataAllocator_.free(alloc.quadDataOffset, alloc.reservedQuadWordCount);
                        alloc = LodAllocation{};
                        activeRangesDirty_ = true;
                    }
                }
            }
            result.deltaApplied = true;
            continue;
        }

        if (upsert.key.lod >= kMaxLods) continue;

        const uint32_t slotIndex = acquireTileSlot(upsert.key.tile);
        TileSlot& slot = tileSlots_[slotIndex];
        LodAllocation& alloc = slot.lods[upsert.key.lod];

        const uint32_t requiredMeshletCount = static_cast<uint32_t>(packed->metadata.size());
        const uint32_t requiredQuadWords = static_cast<uint32_t>(packed->quadData.size());

        // Check if existing allocation can be reused in-place.
        const bool canReuse = alloc.resident &&
                              alloc.reservedMeshletCount >= requiredMeshletCount &&
                              alloc.reservedQuadWordCount >= requiredQuadWords;

        uint32_t meshletOffset = 0u;
        uint32_t quadOffset = 0u;

        if (canReuse) {
            meshletOffset = alloc.meshletOffset;
            quadOffset = alloc.quadDataOffset;
        } else {
            // Free old allocation if it exists.
            if (alloc.resident) {
                meshletAllocator_.free(alloc.meshletOffset, alloc.reservedMeshletCount);
                quadDataAllocator_.free(alloc.quadDataOffset, alloc.reservedQuadWordCount);
            }

            // Allocate new space.
            bool meshletOk = meshletAllocator_.allocate(requiredMeshletCount, meshletOffset);
            bool quadOk = meshletOk && quadDataAllocator_.allocate(requiredQuadWords, quadOffset);

            if (!meshletOk || !quadOk) {
                if (meshletOk) meshletAllocator_.free(meshletOffset, requiredMeshletCount);
                if (quadOk) quadDataAllocator_.free(quadOffset, requiredQuadWords);

                // Need bigger buffers.
                bool recreated = false;
                const uint32_t neededMeshlets = meshletCapacity_ + requiredMeshletCount + 64u;
                const uint32_t neededQuadWords = quadWordCapacity_ + requiredQuadWords + 1024u;
                const uint32_t neededRanges = std::max<uint32_t>(
                    rangeCapacity_,
                    static_cast<uint32_t>(activeSlotIndices_.size() + 64u));

                if (!ensureBuffers(neededMeshlets, neededQuadWords, neededRanges, &recreated)) {
                    alloc.resident = false;
                    continue;
                }
                result.buffersRecreated = result.buffersRecreated || recreated;
                if (recreated) activeRangesDirty_ = true;

                // Retry allocation after repack.
                meshletOk = meshletAllocator_.allocate(requiredMeshletCount, meshletOffset);
                quadOk = meshletOk && quadDataAllocator_.allocate(requiredQuadWords, quadOffset);
                if (!meshletOk || !quadOk) {
                    if (meshletOk) meshletAllocator_.free(meshletOffset, requiredMeshletCount);
                    alloc.resident = false;
                    continue;
                }
            }
        }

        // Write meshlet data to GPU.
        if (!writeLodAllocation(meshletOffset, quadOffset, *packed)) {
            if (!canReuse) {
                meshletAllocator_.free(meshletOffset, requiredMeshletCount);
                quadDataAllocator_.free(quadOffset, requiredQuadWords);
            }
            continue;
        }

        // Update allocation record.
        alloc.meshletOffset = meshletOffset;
        alloc.meshletCount = requiredMeshletCount;
        alloc.reservedMeshletCount = canReuse ? alloc.reservedMeshletCount : requiredMeshletCount;
        alloc.quadDataOffset = quadOffset;
        alloc.quadDataWordCount = requiredQuadWords;
        alloc.reservedQuadWordCount = canReuse ? alloc.reservedQuadWordCount : requiredQuadWords;
        alloc.packed = packed;
        alloc.resident = true;

        result.deltaApplied = true;

        // If this is the active LOD for this slot, mark ranges dirty.
        if (slot.activeLod == static_cast<int8_t>(upsert.key.lod)) {
            activeRangesDirty_ = true;
            activeMeshletBoundsDirty_ = true;
            // Ensure the slot is shown if it should be.
            if (!slot.inActiveList) {
                showTileSlot(slotIndex);
            }
        }

        if (bufferManager_->isOverBudget()) {
            break;
        }
    }

    // --- Selection changes ---
    if (!delta.selectionChanges.empty()) {
        for (const MeshTileSelectionEntry& entry : delta.selectionChanges) {
            auto coordIt = coordToSlot_.find(entry.tile);
            if (coordIt == coordToSlot_.end()) {
                if (entry.selectedLod < 0) continue;
                // No slot yet for this coord. Create one with the selection
                // but it won't be shown until data arrives.
                const uint32_t si = acquireTileSlot(entry.tile);
                tileSlots_[si].activeLod = entry.selectedLod;
                continue;
            }

            const uint32_t slotIndex = coordIt->second;

            if (entry.selectedLod < 0) {
                // Deselect this tile.
                hideTileSlot(slotIndex);
                tileSlots_[slotIndex].activeLod = -1;
            } else {
                switchTileLod(slotIndex, entry.selectedLod);

                // Try to show if the new LOD is resident.
                TileSlot& slot = tileSlots_[slotIndex];
                if (!slot.inActiveList && slot.activeLod >= 0) {
                    const auto& lodAlloc = slot.lods[static_cast<uint8_t>(slot.activeLod)];
                    if (lodAlloc.resident && lodAlloc.meshletCount > 0u) {
                        showTileSlot(slotIndex);
                    }
                }
            }
        }
        result.deltaApplied = true;
    }

    // --- Rebuild active ranges if anything changed ---
    if (activeRangesDirty_) {
        bool recreatedForRanges = false;
        if (!ensureBuffers(
                meshletCapacity_,
                quadWordCapacity_,
                std::max<uint32_t>(1u, static_cast<uint32_t>(activeSlotIndices_.size())),
                &recreatedForRanges)) {
            return result;
        }
        result.buffersRecreated = result.buffersRecreated || recreatedForRanges;
        buildActiveRanges();
    }

    if (delta.revision > uploadedMeshRevision_) {
        uploadedMeshRevision_ = delta.revision;
    }

    return result;
}

// ---------------------------------------------------------------------------
// buildActiveRanges -- linear scan over activeSlotIndices_, no hash lookups
// ---------------------------------------------------------------------------

bool MeshletBufferController::buildActiveRanges() {
    if (bufferManager_ == nullptr) return false;

    activeRanges_.clear();
    activeRanges_.reserve(activeSlotIndices_.size());
    activeMeshletBoundsDirty_ = true;

    uint32_t prefixSum = 0u;
    uint32_t computedTotal = 0u;

    for (const uint32_t slotIdx : activeSlotIndices_) {
        const TileSlot& slot = tileSlots_[slotIdx];
        if (slot.activeLod < 0) continue;

        const LodAllocation& lod = slot.lods[static_cast<uint8_t>(slot.activeLod)];
        if (!lod.resident || lod.meshletCount == 0u) continue;

        // Cap total meshlets per controller (macOS limit).
        if (computedTotal + lod.meshletCount > maxActiveMeshlets_) break;

        computedTotal += lod.meshletCount;
        prefixSum += lod.meshletCount;
        activeRanges_.push_back(ActiveMeshletRangeGPU{
            lod.meshletOffset,
            lod.meshletCount,
            prefixSum,
            0u});
    }

    totalActiveMeshlets_ = computedTotal;

    // Ensure range buffer is large enough.
    bool recreated = false;
    if (!ensureBuffers(meshletCapacity_, quadWordCapacity_,
                       std::max<uint32_t>(1u, static_cast<uint32_t>(activeRanges_.size())),
                       &recreated)) {
        return false;
    }

    if (!activeRanges_.empty()) {
        bufferManager_->writeBuffer(
            activeMeshletRangeBufferName_,
            0u,
            activeRanges_.data(),
            activeRanges_.size() * sizeof(ActiveMeshletRangeGPU));
    }

    const RangeParamsGPU rangeParams{
        static_cast<uint32_t>(activeRanges_.size()),
        computedTotal,
        0u, 0u};
    bufferManager_->writeBuffer(
        activeMeshletRangeParamsBufferName_,
        0u,
        &rangeParams,
        sizeof(rangeParams));

    activeRangesDirty_ = false;
    return true;
}

// ---------------------------------------------------------------------------
// Accessors (unchanged public API)
// ---------------------------------------------------------------------------

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
    return totalActiveMeshlets_;
}

uint32_t MeshletBufferController::activeSelectionMeshletCount() const noexcept {
    return totalActiveMeshlets_;
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
    activeMeshletBoundsCache_.reserve(totalActiveMeshlets_);

    for (const uint32_t slotIdx : activeSlotIndices_) {
        const TileSlot& slot = tileSlots_[slotIdx];
        if (slot.activeLod < 0) continue;

        const LodAllocation& lod = slot.lods[static_cast<uint8_t>(slot.activeLod)];
        if (!lod.resident || !lod.packed) continue;

        activeMeshletBoundsCache_.insert(
            activeMeshletBoundsCache_.end(),
            lod.packed->bounds.begin(),
            lod.packed->bounds.end());
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
        verticesPerMeshlet()};
}

const std::shared_ptr<const PackedMeshletData>& MeshletBufferController::selectPackedForVariant(
    const MeshTileLodUpload& upload) const {
    if (geometryVariant_ == MeshletGeometryVariant::DoubleSided) {
        return upload.doubleSidedPacked;
    }
    return upload.culledPacked;
}
