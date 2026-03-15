#include "solum_engine/render/MeshletBufferController.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>

#include <webgpu/webgpu.hpp>

using namespace wgpu;

namespace {
constexpr uint32_t kInvalidIndex = ResidentTileLodHandle::kInvalidSlot;

glm::vec4 toVec4(const glm::vec3& value) {
    return glm::vec4(value, 0.0f);
}

const char* variantLabel(MeshletGeometryVariant variant) {
    switch (variant) {
    case MeshletGeometryVariant::DoubleSided:
        return "double-sided";
    case MeshletGeometryVariant::Culled:
    default:
        return "culled";
    }
}

uint32_t floorLog2(uint32_t value) {
    uint32_t result = 0u;
    while (value > 1u) {
        value >>= 1u;
        ++result;
    }
    return result;
}

uint32_t ceilLog2(uint32_t value) {
    if (value <= 1u) {
        return 0u;
    }
    return floorLog2(value - 1u) + 1u;
}
}  // namespace

void MeshletBufferController::PageAllocator::reset(uint32_t capacityUnits, uint32_t pageSizeUnits) {
    pageSize = std::max(pageSizeUnits, 1u);
    pageCount = (capacityUnits + pageSize - 1u) / pageSize;
    maxOrder = (pageCount > 0u) ? floorLog2(pageCount) : 0u;
    freeLists.assign(maxOrder + 1u, {});
    freeBlockOrders.clear();

    uint32_t nextStartPage = 0u;
    uint32_t remainingPages = pageCount;
    while (remainingPages > 0u) {
        uint32_t order = floorLog2(remainingPages);
        while (order > 0u && (nextStartPage & ((1u << order) - 1u)) != 0u) {
            --order;
        }

        addFreeBlock(nextStartPage, order);
        const uint32_t blockPages = 1u << order;
        nextStartPage += blockPages;
        remainingPages -= blockPages;
    }
}

uint32_t MeshletBufferController::PageAllocator::pagesForUnits(uint32_t unitCount) const noexcept {
    if (unitCount == 0u) {
        return 0u;
    }
    return (unitCount + pageSize - 1u) / pageSize;
}

bool MeshletBufferController::PageAllocator::allocate(uint32_t unitCount, PageRun& outRun) {
    const uint32_t requiredPages = pagesForUnits(unitCount);
    if (requiredPages == 0u) {
        outRun = {};
        return true;
    }
    if (requiredPages > pageCount || freeLists.empty()) {
        return false;
    }

    const uint32_t targetOrder = ceilLog2(requiredPages);
    for (uint32_t order = targetOrder; order < freeLists.size(); ++order) {
        if (freeLists[order].empty()) {
            continue;
        }

        const uint32_t startPage = *freeLists[order].begin();
        removeFreeBlock(startPage, order);

        uint32_t splitOrder = order;
        while (splitOrder > targetOrder) {
            --splitOrder;
            addFreeBlock(startPage + (1u << splitOrder), splitOrder);
        }

        outRun.startPage = startPage;
        outRun.pageCount = 1u << targetOrder;
        return true;
    }

    return false;
}

void MeshletBufferController::PageAllocator::release(const PageRun& run) {
    if (!run.valid()) {
        return;
    }

    uint32_t startPage = run.startPage;
    uint32_t order = floorLog2(std::max(run.pageCount, 1u));
    while (order < freeLists.size()) {
        const uint32_t blockPages = 1u << order;
        const uint32_t buddyStart = startPage ^ blockPages;
        const auto buddyIt = freeBlockOrders.find(buddyStart);
        if (buddyIt == freeBlockOrders.end() || buddyIt->second != order) {
            break;
        }

        removeFreeBlock(buddyStart, order);
        startPage = std::min(startPage, buddyStart);
        ++order;
    }

    addFreeBlock(startPage, std::min<uint32_t>(order, maxOrder));
}

uint32_t MeshletBufferController::PageAllocator::unitOffset(const PageRun& run) const noexcept {
    return run.startPage * pageSize;
}

void MeshletBufferController::PageAllocator::addFreeBlock(uint32_t startPage, uint32_t order) {
    if (order >= freeLists.size()) {
        return;
    }
    freeLists[order].insert(startPage);
    freeBlockOrders[startPage] = order;
}

void MeshletBufferController::PageAllocator::removeFreeBlock(uint32_t startPage, uint32_t order) {
    if (order >= freeLists.size()) {
        return;
    }
    freeLists[order].erase(startPage);
    freeBlockOrders.erase(startPage);
}

MeshletBufferController::MeshletBufferController()
    : MeshletBufferController(Config{}) {}

MeshletBufferController::MeshletBufferController(Config config)
    : config_(std::move(config)),
      namePrefix_(config_.namePrefix),
      geometryVariant_(config_.geometryVariant),
      meshDataBufferName_(prefixedName(namePrefix_, "meshlet_data_buffer")),
      meshMetadataBufferName_(prefixedName(namePrefix_, "meshlet_metadata_buffer")),
      meshAabbBufferName_(prefixedName(namePrefix_, "meshlet_aabb_buffer")),
      visibleMeshletIndexBufferName_(prefixedName(namePrefix_, "visible_meshlet_indices_buffer")),
      residentTileLodBufferName_(prefixedName(namePrefix_, "resident_tile_lods_buffer")),
      tileSlotBufferName_(prefixedName(namePrefix_, "tile_slots_buffer")),
      visibleTileIdBufferName_(prefixedName(namePrefix_, "visible_tile_ids_buffer")),
      tileSceneParamsBufferName_(prefixedName(namePrefix_, "tile_scene_params_buffer")) {}

std::string MeshletBufferController::prefixedName(const std::string& prefix, const char* baseName) {
    if (prefix.empty()) {
        return std::string(baseName);
    }
    return prefix + "_" + baseName;
}

bool MeshletBufferController::initialize(BufferManager* bufferManager) {
    bufferManager_ = bufferManager;
    residentRecords_.clear();
    freeResidentSlots_.clear();
    residentSlotByKey_.clear();
    tileSlots_.clear();
    tileSlotsGpu_.clear();
    visibleTileIds_.clear();
    tileSlotIndexByCoord_.clear();
    dirtyResidentSlots_.clear();
    residentDirtyMask_.clear();
    dirtyTileSlots_.clear();
    tileDirtyMask_.clear();
    dirtyVisibleIndices_.clear();
    visibleIndexDirtyMask_.clear();
    warnedTileSlotCapacity_ = false;
    warnedResidentSlotCapacity_ = false;
    warnedArenaCapacity_ = false;
    activeSelectionMeshletCount_ = 0u;
    uploadedMeshRevision_ = 0u;
    activeMeshletBoundsCache_.clear();
    activeMeshletBoundsDirty_ = true;
    sceneParamsDirty_ = true;

    if (bufferManager_ == nullptr) {
        return false;
    }

    meshletPages_.reset(config_.meshletCapacity, config_.meshletPageSize);
    quadPages_.reset(config_.quadWordCapacity, config_.quadWordPageSize);

    residentRecords_.reserve(config_.residentTileCapacity);
    residentDirtyMask_.assign(config_.residentTileCapacity, 0u);
    tileSlots_.reserve(config_.tileSlotCapacity);
    tileSlotsGpu_.reserve(config_.tileSlotCapacity);
    tileDirtyMask_.assign(config_.tileSlotCapacity, 0u);
    visibleTileIds_.reserve(config_.tileSlotCapacity);
    visibleIndexDirtyMask_.assign(config_.tileSlotCapacity, 0u);

    return createBuffers() && flushDirtyState();
}

bool MeshletBufferController::createBuffers() {
    if (bufferManager_ == nullptr) {
        return false;
    }

    BufferDescriptor metadataDesc = Default;
    metadataDesc.label = StringView("meshlet metadata buffer");
    metadataDesc.size = static_cast<uint64_t>(config_.meshletCapacity) * sizeof(MeshletMetadataGPU);
    metadataDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    if (!bufferManager_->createBuffer(meshMetadataBufferName_, metadataDesc)) {
        return false;
    }

    BufferDescriptor meshDataDesc = Default;
    meshDataDesc.label = StringView("meshlet data buffer");
    meshDataDesc.size = static_cast<uint64_t>(config_.quadWordCapacity) * sizeof(uint32_t);
    meshDataDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    if (!bufferManager_->createBuffer(meshDataBufferName_, meshDataDesc)) {
        return false;
    }

    BufferDescriptor aabbDesc = Default;
    aabbDesc.label = StringView("meshlet aabb buffer");
    aabbDesc.size = static_cast<uint64_t>(config_.meshletCapacity) * sizeof(MeshletAabbGPU);
    aabbDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    if (!bufferManager_->createBuffer(meshAabbBufferName_, aabbDesc)) {
        return false;
    }

    BufferDescriptor visibleIndicesDesc = Default;
    visibleIndicesDesc.label = StringView("visible meshlet indices buffer");
    visibleIndicesDesc.size = static_cast<uint64_t>(config_.meshletCapacity) * sizeof(uint32_t);
    visibleIndicesDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    if (!bufferManager_->createBuffer(visibleMeshletIndexBufferName_, visibleIndicesDesc)) {
        return false;
    }

    BufferDescriptor residentDesc = Default;
    residentDesc.label = StringView("resident tile lod buffer");
    residentDesc.size = static_cast<uint64_t>(config_.residentTileCapacity) * sizeof(ResidentTileLodGPU);
    residentDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    if (!bufferManager_->createBuffer(residentTileLodBufferName_, residentDesc)) {
        return false;
    }

    BufferDescriptor tileSlotDesc = Default;
    tileSlotDesc.label = StringView("tile slots buffer");
    tileSlotDesc.size = static_cast<uint64_t>(config_.tileSlotCapacity) * sizeof(TileSlotGPU);
    tileSlotDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    if (!bufferManager_->createBuffer(tileSlotBufferName_, tileSlotDesc)) {
        return false;
    }

    BufferDescriptor visibleTileDesc = Default;
    visibleTileDesc.label = StringView("visible tile ids buffer");
    visibleTileDesc.size = static_cast<uint64_t>(config_.tileSlotCapacity) * sizeof(VisibleTileId);
    visibleTileDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    if (!bufferManager_->createBuffer(visibleTileIdBufferName_, visibleTileDesc)) {
        return false;
    }

    BufferDescriptor sceneParamsDesc = Default;
    sceneParamsDesc.label = StringView("tile scene params buffer");
    sceneParamsDesc.size = sizeof(TileSceneParamsGPU);
    sceneParamsDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform | BufferUsage::Storage;
    return bufferManager_->createBuffer(tileSceneParamsBufferName_, sceneParamsDesc) != nullptr;
}

bool MeshletBufferController::writeAllocation(uint32_t meshletStart,
                                              uint32_t quadWordStart,
                                              const PackedMeshletData& packed) const {
    if (bufferManager_ == nullptr) {
        return false;
    }

    if (!packed.metadata.empty()) {
        std::vector<MeshletMetadataGPU> metadata = packed.metadata;
        for (MeshletMetadataGPU& entry : metadata) {
            entry.dataOffset += quadWordStart;
        }

        bufferManager_->writeBuffer(
            meshMetadataBufferName_,
            static_cast<uint64_t>(meshletStart) * sizeof(MeshletMetadataGPU),
            metadata.data(),
            metadata.size() * sizeof(MeshletMetadataGPU)
        );
        bufferManager_->writeBuffer(
            meshAabbBufferName_,
            static_cast<uint64_t>(meshletStart) * sizeof(MeshletAabbGPU),
            packed.aabbGpu.data(),
            packed.aabbGpu.size() * sizeof(MeshletAabbGPU)
        );
    }

    if (!packed.quadData.empty()) {
        bufferManager_->writeBuffer(
            meshDataBufferName_,
            static_cast<uint64_t>(quadWordStart) * sizeof(uint32_t),
            packed.quadData.data(),
            packed.quadData.size() * sizeof(uint32_t)
        );
    }

    return true;
}

MeshletBufferController::TileAabb MeshletBufferController::computeTileAabb(const PackedMeshletData& packed) const {
    TileAabb bounds{};
    if (packed.bounds.empty()) {
        return bounds;
    }

    bounds.minCorner = packed.bounds.front().minCorner;
    bounds.maxCorner = packed.bounds.front().maxCorner;
    for (const MeshletAabb& meshletBounds : packed.bounds) {
        bounds.minCorner = glm::min(bounds.minCorner, meshletBounds.minCorner);
        bounds.maxCorner = glm::max(bounds.maxCorner, meshletBounds.maxCorner);
    }
    return bounds;
}

const std::shared_ptr<const PackedMeshletData>& MeshletBufferController::selectPackedForVariant(
    const MeshTileLodUpload& upload
) const {
    if (geometryVariant_ == MeshletGeometryVariant::DoubleSided) {
        return upload.doubleSidedPacked;
    }
    return upload.culledPacked;
}

ResidentTileLodHandle MeshletBufferController::handleForSlot(uint32_t slot) const noexcept {
    if (slot >= residentRecords_.size()) {
        return {};
    }
    return ResidentTileLodHandle{slot, residentRecords_[slot].generation};
}

bool MeshletBufferController::isHandleValid(ResidentTileLodHandle handle) const noexcept {
    return residentForHandle(handle) != nullptr;
}

ResidentTileLodHandle MeshletBufferController::chooseTileResidentHandle(const TileSlotState& tileSlot) const noexcept {
    auto validHandleForLod = [this, &tileSlot](int32_t lod) -> ResidentTileLodHandle {
        if (lod < 0 || static_cast<uint32_t>(lod) >= kMaxResidentLods) {
            return {};
        }

        const ResidentTileLodHandle handle = tileSlot.lodHandles[static_cast<uint32_t>(lod)];
        return isHandleValid(handle) ? handle : ResidentTileLodHandle{};
    };

    if (ResidentTileLodHandle desiredHandle = validHandleForLod(tileSlot.selectedLod); desiredHandle.valid()) {
        return desiredHandle;
    }

    if (isHandleValid(tileSlot.selectedResident)) {
        return tileSlot.selectedResident;
    }

    if (tileSlot.selectedLod >= 0) {
        for (uint32_t offset = 1u; offset < kMaxResidentLods; ++offset) {
            if (ResidentTileLodHandle lowerHandle =
                    validHandleForLod(static_cast<int32_t>(tileSlot.selectedLod) - static_cast<int32_t>(offset));
                lowerHandle.valid()) {
                return lowerHandle;
            }
            if (ResidentTileLodHandle upperHandle =
                    validHandleForLod(static_cast<int32_t>(tileSlot.selectedLod) + static_cast<int32_t>(offset));
                upperHandle.valid()) {
                return upperHandle;
            }
        }
    }

    for (const ResidentTileLodHandle& handle : tileSlot.lodHandles) {
        if (isHandleValid(handle)) {
            return handle;
        }
    }

    return {};
}

MeshletBufferController::ResidentRecord* MeshletBufferController::residentForHandle(ResidentTileLodHandle handle) noexcept {
    if (!handle.valid() || handle.slot >= residentRecords_.size()) {
        return nullptr;
    }
    ResidentRecord& record = residentRecords_[handle.slot];
    if (!record.active || record.generation != handle.generation) {
        return nullptr;
    }
    return &record;
}

const MeshletBufferController::ResidentRecord* MeshletBufferController::residentForHandle(ResidentTileLodHandle handle) const noexcept {
    if (!handle.valid() || handle.slot >= residentRecords_.size()) {
        return nullptr;
    }
    const ResidentRecord& record = residentRecords_[handle.slot];
    if (!record.active || record.generation != handle.generation) {
        return nullptr;
    }
    return &record;
}

uint32_t MeshletBufferController::ensureTileSlot(const MeshTileSliceCoord& tile) {
    const auto existing = tileSlotIndexByCoord_.find(tile);
    if (existing != tileSlotIndexByCoord_.end()) {
        return existing->second;
    }

    if (tileSlots_.size() >= config_.tileSlotCapacity) {
        if (!warnedTileSlotCapacity_) {
            warnedTileSlotCapacity_ = true;
            std::cerr << "MeshletBufferController(" << variantLabel(geometryVariant_)
                      << "): tile slot capacity exhausted at " << config_.tileSlotCapacity
                      << " slots; additional tiles will not be tracked until capacity is increased."
                      << std::endl;
        }
        return kInvalidIndex;
    }

    const uint32_t slotIndex = static_cast<uint32_t>(tileSlots_.size());
    tileSlots_.push_back(TileSlotState{});
    tileSlots_.back().tile = tile;
    tileSlots_.back().allocated = true;
    tileSlotsGpu_.push_back(TileSlotGPU{});
    tileSlotIndexByCoord_[tile] = slotIndex;
    markTileDirty(slotIndex);
    sceneParamsDirty_ = true;
    return slotIndex;
}

int32_t MeshletBufferController::allocateResidentSlot() {
    if (!freeResidentSlots_.empty()) {
        const uint32_t slot = freeResidentSlots_.back();
        freeResidentSlots_.pop_back();
        return static_cast<int32_t>(slot);
    }
    if (residentRecords_.size() >= config_.residentTileCapacity) {
        if (!warnedResidentSlotCapacity_) {
            warnedResidentSlotCapacity_ = true;
            std::cerr << "MeshletBufferController(" << variantLabel(geometryVariant_)
                      << "): resident TileLod slot capacity exhausted at " << config_.residentTileCapacity
                      << " entries; eviction pressure is now coming entirely from the slot table."
                      << std::endl;
        }
        return -1;
    }
    residentRecords_.push_back(ResidentRecord{});
    return static_cast<int32_t>(residentRecords_.size() - 1u);
}

void MeshletBufferController::markResidentDirty(uint32_t residentSlot) {
    if (residentSlot >= residentDirtyMask_.size() || residentDirtyMask_[residentSlot] != 0u) {
        return;
    }
    residentDirtyMask_[residentSlot] = 1u;
    dirtyResidentSlots_.push_back(residentSlot);
}

void MeshletBufferController::markTileDirty(uint32_t tileSlot) {
    if (tileSlot >= tileDirtyMask_.size() || tileDirtyMask_[tileSlot] != 0u) {
        return;
    }
    tileDirtyMask_[tileSlot] = 1u;
    dirtyTileSlots_.push_back(tileSlot);
}

void MeshletBufferController::markVisibleIndexDirty(uint32_t visibleIndex) {
    if (visibleIndex >= visibleIndexDirtyMask_.size() || visibleIndexDirtyMask_[visibleIndex] != 0u) {
        return;
    }
    visibleIndexDirtyMask_[visibleIndex] = 1u;
    dirtyVisibleIndices_.push_back(visibleIndex);
}

uint32_t MeshletBufferController::visibleMeshletContribution(uint32_t tileSlotIndex) const noexcept {
    if (tileSlotIndex >= tileSlots_.size()) {
        return 0u;
    }
    const TileSlotState& tileSlot = tileSlots_[tileSlotIndex];
    if (!tileSlot.visible) {
        return 0u;
    }
    const ResidentRecord* resident = residentForHandle(tileSlot.selectedResident);
    return resident ? resident->gpu.meshletCount : 0u;
}

void MeshletBufferController::updateVisibleMeshletContribution(uint32_t tileSlotIndex, int64_t delta) noexcept {
    if (delta == 0) {
        return;
    }
    if (delta < 0) {
        const uint64_t amount = static_cast<uint64_t>(-delta);
        activeSelectionMeshletCount_ = (amount >= activeSelectionMeshletCount_)
            ? 0u
            : static_cast<uint32_t>(activeSelectionMeshletCount_ - amount);
        return;
    }
    activeSelectionMeshletCount_ += static_cast<uint32_t>(delta);
}

void MeshletBufferController::updateTileResidentFlags(uint32_t tileSlotIndex) {
    if (tileSlotIndex >= tileSlots_.size()) {
        return;
    }
    TileSlotGPU& gpu = tileSlotsGpu_[tileSlotIndex];
    gpu.flags &= ~kTileSlotFlagAnyResident;
    for (const ResidentTileLodHandle& handle : tileSlots_[tileSlotIndex].lodHandles) {
        if (isHandleValid(handle)) {
            gpu.flags |= kTileSlotFlagAnyResident;
            break;
        }
    }
}

void MeshletBufferController::updateTileAabb(uint32_t tileSlotIndex) {
    if (tileSlotIndex >= tileSlots_.size()) {
        return;
    }

    TileSlotGPU& gpu = tileSlotsGpu_[tileSlotIndex];
    const ResidentRecord* selectedResident = residentForHandle(tileSlots_[tileSlotIndex].selectedResident);
    if (selectedResident != nullptr) {
        gpu.minCorner = toVec4(selectedResident->tileAabb.minCorner);
        gpu.maxCorner = toVec4(selectedResident->tileAabb.maxCorner);
        return;
    }

    for (const ResidentTileLodHandle& handle : tileSlots_[tileSlotIndex].lodHandles) {
        const ResidentRecord* resident = residentForHandle(handle);
        if (resident == nullptr) {
            continue;
        }
        gpu.minCorner = toVec4(resident->tileAabb.minCorner);
        gpu.maxCorner = toVec4(resident->tileAabb.maxCorner);
        return;
    }

    gpu.minCorner = glm::vec4(0.0f);
    gpu.maxCorner = glm::vec4(0.0f);
}

void MeshletBufferController::setTileVisible(const MeshTileSliceCoord& tile, bool visible) {
    const uint32_t tileSlotIndex = ensureTileSlot(tile);
    if (tileSlotIndex == kInvalidIndex) {
        return;
    }

    TileSlotState& tileSlot = tileSlots_[tileSlotIndex];
    if (tileSlot.visible == visible) {
        return;
    }

    updateVisibleMeshletContribution(tileSlotIndex, -static_cast<int64_t>(visibleMeshletContribution(tileSlotIndex)));
    tileSlot.visible = visible;
    tileSlotsGpu_[tileSlotIndex].visible = visible ? 1u : 0u;

    if (visible) {
        tileSlot.visibleListIndex = static_cast<uint32_t>(visibleTileIds_.size());
        visibleTileIds_.push_back(tileSlotIndex);
        markVisibleIndexDirty(tileSlot.visibleListIndex);
    } else if (tileSlot.visibleListIndex != kInvalidIndex && tileSlot.visibleListIndex < visibleTileIds_.size()) {
        const uint32_t removedIndex = tileSlot.visibleListIndex;
        const uint32_t swappedTileSlot = visibleTileIds_.back();
        visibleTileIds_[removedIndex] = swappedTileSlot;
        visibleTileIds_.pop_back();
        markVisibleIndexDirty(removedIndex);
        if (removedIndex < visibleTileIds_.size()) {
            tileSlots_[swappedTileSlot].visibleListIndex = removedIndex;
        }
        tileSlot.visibleListIndex = kInvalidIndex;
    }

    updateVisibleMeshletContribution(tileSlotIndex, static_cast<int64_t>(visibleMeshletContribution(tileSlotIndex)));
    markTileDirty(tileSlotIndex);
    sceneParamsDirty_ = true;
    activeMeshletBoundsDirty_ = true;
}

void MeshletBufferController::updateTileSelectedResident(uint32_t tileSlotIndex) {
    if (tileSlotIndex >= tileSlots_.size()) {
        return;
    }

    TileSlotState& tileSlot = tileSlots_[tileSlotIndex];
    const ResidentTileLodHandle nextHandle = chooseTileResidentHandle(tileSlot);

    tileSlot.selectedResident = nextHandle;
    TileSlotGPU& gpu = tileSlotsGpu_[tileSlotIndex];
    if (const ResidentRecord* resident = residentForHandle(nextHandle)) {
        gpu.selectedResidentSlot = nextHandle.slot;
        gpu.flags |= kTileSlotFlagSelectedResidentValid;
        gpu.minCorner = toVec4(resident->tileAabb.minCorner);
        gpu.maxCorner = toVec4(resident->tileAabb.maxCorner);
    } else {
        gpu.selectedResidentSlot = kInvalidIndex;
        gpu.flags &= ~kTileSlotFlagSelectedResidentValid;
        updateTileAabb(tileSlotIndex);
    }
}

void MeshletBufferController::setTileSelectedLod(const MeshTileSliceCoord& tile, int8_t selectedLod) {
    const uint32_t tileSlotIndex = ensureTileSlot(tile);
    if (tileSlotIndex == kInvalidIndex) {
        return;
    }

    TileSlotState& tileSlot = tileSlots_[tileSlotIndex];
    if (tileSlot.selectedLod == selectedLod) {
        setTileVisible(tile, selectedLod >= 0);
        return;
    }

    updateVisibleMeshletContribution(tileSlotIndex, -static_cast<int64_t>(visibleMeshletContribution(tileSlotIndex)));
    tileSlot.selectedLod = selectedLod;
    updateTileSelectedResident(tileSlotIndex);
    updateVisibleMeshletContribution(tileSlotIndex, static_cast<int64_t>(visibleMeshletContribution(tileSlotIndex)));
    markTileDirty(tileSlotIndex);
    sceneParamsDirty_ = true;
    activeMeshletBoundsDirty_ = true;

    setTileVisible(tile, selectedLod >= 0);
}

void MeshletBufferController::setResidentLod(const MeshTileSliceCoord& tile,
                                             uint8_t lod,
                                             ResidentTileLodHandle handle) {
    if (lod >= kMaxResidentLods) {
        return;
    }
    const uint32_t tileSlotIndex = ensureTileSlot(tile);
    if (tileSlotIndex == kInvalidIndex) {
        return;
    }

    updateVisibleMeshletContribution(tileSlotIndex, -static_cast<int64_t>(visibleMeshletContribution(tileSlotIndex)));
    tileSlots_[tileSlotIndex].lodHandles[lod] = handle;
    updateTileResidentFlags(tileSlotIndex);
    if (tileSlots_[tileSlotIndex].selectedLod == static_cast<int8_t>(lod)) {
        updateTileSelectedResident(tileSlotIndex);
    } else {
        updateTileAabb(tileSlotIndex);
    }
    updateVisibleMeshletContribution(tileSlotIndex, static_cast<int64_t>(visibleMeshletContribution(tileSlotIndex)));
    markTileDirty(tileSlotIndex);
    sceneParamsDirty_ = true;
    activeMeshletBoundsDirty_ = true;
}

void MeshletBufferController::clearResidentLod(const MeshTileSliceCoord& tile,
                                               uint8_t lod,
                                               ResidentTileLodHandle handle) {
    if (lod >= kMaxResidentLods) {
        return;
    }
    const auto tileIt = tileSlotIndexByCoord_.find(tile);
    if (tileIt == tileSlotIndexByCoord_.end()) {
        return;
    }

    const uint32_t tileSlotIndex = tileIt->second;
    TileSlotState& tileSlot = tileSlots_[tileSlotIndex];
    if (tileSlot.lodHandles[lod].slot != handle.slot || tileSlot.lodHandles[lod].generation != handle.generation) {
        return;
    }

    updateVisibleMeshletContribution(tileSlotIndex, -static_cast<int64_t>(visibleMeshletContribution(tileSlotIndex)));
    tileSlot.lodHandles[lod] = {};
    updateTileResidentFlags(tileSlotIndex);
    if (tileSlot.selectedLod == static_cast<int8_t>(lod)) {
        updateTileSelectedResident(tileSlotIndex);
    } else {
        updateTileAabb(tileSlotIndex);
    }
    updateVisibleMeshletContribution(tileSlotIndex, static_cast<int64_t>(visibleMeshletContribution(tileSlotIndex)));
    markTileDirty(tileSlotIndex);
    sceneParamsDirty_ = true;
    activeMeshletBoundsDirty_ = true;
}

int32_t MeshletBufferController::chooseEvictionCandidate(const MeshTileLodKey* protectedKey) const {
    auto pickCandidate = [this, protectedKey](bool allowVisibleTiles) -> int32_t {
        int32_t bestIndex = -1;
        uint64_t bestScore = 0u;

        for (uint32_t index = 0u; index < residentRecords_.size(); ++index) {
            const ResidentRecord& record = residentRecords_[index];
            if (!record.active) {
                continue;
            }
            if (protectedKey != nullptr && record.key == *protectedKey) {
                continue;
            }

            const auto tileSlotIt = tileSlotIndexByCoord_.find(record.key.tile);
            const TileSlotState* tileSlot = (tileSlotIt != tileSlotIndexByCoord_.end())
                ? &tileSlots_[tileSlotIt->second]
                : nullptr;
            const bool visible = tileSlot != nullptr && tileSlot->visible;
            if (!allowVisibleTiles && visible) {
                continue;
            }

            const bool selectedResident = tileSlot != nullptr &&
                tileSlot->selectedResident.slot == index &&
                tileSlot->selectedResident.generation == record.generation;
            if (selectedResident) {
                continue;
            }

            const bool requestedLod = tileSlot != nullptr && tileSlot->selectedLod >= 0 &&
                static_cast<uint32_t>(tileSlot->selectedLod) == static_cast<uint32_t>(record.key.lod);

            uint64_t score = 0u;
            score |= visible ? 0u : (1ull << 63);
            score |= requestedLod ? 0u : (1ull << 62);
            score |= static_cast<uint64_t>(record.key.lod) << 48;
            score |= (std::numeric_limits<uint32_t>::max() - record.lastTouchedRevision);

            if (bestIndex < 0 || score > bestScore) {
                bestIndex = static_cast<int32_t>(index);
                bestScore = score;
            }
        }

        return bestIndex;
    };

    if (const int32_t invisibleCandidate = pickCandidate(false); invisibleCandidate >= 0) {
        return invisibleCandidate;
    }

    return pickCandidate(true);
}

bool MeshletBufferController::evictForAllocation(uint32_t requiredMeshlets,
                                                 uint32_t requiredQuadWords,
                                                 const MeshTileLodKey* protectedKey) {
    PageRun testMeshletRun{};
    PageRun testQuadRun{};
    while (!meshletPages_.allocate(requiredMeshlets, testMeshletRun) ||
           !quadPages_.allocate(requiredQuadWords, testQuadRun)) {
        if (testMeshletRun.valid()) {
            meshletPages_.release(testMeshletRun);
            testMeshletRun = {};
        }
        if (testQuadRun.valid()) {
            quadPages_.release(testQuadRun);
            testQuadRun = {};
        }

        const int32_t evictionCandidate = chooseEvictionCandidate(protectedKey);
        if (evictionCandidate < 0) {
            return false;
        }
        const MeshTileLodKey key = residentRecords_[static_cast<size_t>(evictionCandidate)].key;
        if (!removeResidentTileLod(key)) {
            return false;
        }
    }

    if (testMeshletRun.valid()) {
        meshletPages_.release(testMeshletRun);
    }
    if (testQuadRun.valid()) {
        quadPages_.release(testQuadRun);
    }
    return true;
}

bool MeshletBufferController::removeResidentTileLod(const MeshTileLodKey& key) {
    const auto residentIt = residentSlotByKey_.find(key);
    if (residentIt == residentSlotByKey_.end()) {
        return false;
    }

    const uint32_t residentSlot = residentIt->second;
    if (residentSlot >= residentRecords_.size()) {
        residentSlotByKey_.erase(residentIt);
        return false;
    }

    ResidentRecord& record = residentRecords_[residentSlot];
    const ResidentTileLodHandle oldHandle{residentSlot, record.generation};
    clearResidentLod(key.tile, key.lod, oldHandle);

    meshletPages_.release(record.meshletPages);
    quadPages_.release(record.quadPages);

    record.active = false;
    record.packed.reset();
    record.meshletPages = {};
    record.quadPages = {};
    ++record.generation;
    record.gpu = ResidentTileLodGPU{};

    residentSlotByKey_.erase(residentIt);
    freeResidentSlots_.push_back(residentSlot);
    markResidentDirty(residentSlot);
    sceneParamsDirty_ = true;
    return true;
}

bool MeshletBufferController::upsertResidentTileLod(const MeshTileLodUpload& upload) {
    const std::shared_ptr<const PackedMeshletData>& packed = selectPackedForVariant(upload);
    if (!packed || packed->metadata.empty()) {
        return removeResidentTileLod(upload.key);
    }

    const uint32_t requiredMeshlets = static_cast<uint32_t>(packed->metadata.size());
    const uint32_t requiredQuadWords = static_cast<uint32_t>(packed->quadData.size());
    if (requiredMeshlets > config_.meshletCapacity || requiredQuadWords > config_.quadWordCapacity) {
        if (!warnedArenaCapacity_) {
            warnedArenaCapacity_ = true;
            std::cerr << "MeshletBufferController(" << variantLabel(geometryVariant_)
                      << "): upload for tile lod exceeds fixed arena capacity (meshlets="
                      << requiredMeshlets << "/" << config_.meshletCapacity
                      << ", quadWords=" << requiredQuadWords << "/" << config_.quadWordCapacity << ")."
                      << std::endl;
        }
        return false;
    }

    ResidentRecord* record = nullptr;
    uint32_t residentSlot = kInvalidIndex;
    bool createdNewSlot = false;

    const auto existingIt = residentSlotByKey_.find(upload.key);
    if (existingIt != residentSlotByKey_.end()) {
        residentSlot = existingIt->second;
        record = &residentRecords_[residentSlot];
    } else {
        int32_t slot = allocateResidentSlot();
        while (slot < 0) {
            const int32_t evictionCandidate = chooseEvictionCandidate(&upload.key);
            if (evictionCandidate < 0) {
                return false;
            }
            const MeshTileLodKey key = residentRecords_[static_cast<size_t>(evictionCandidate)].key;
            if (!removeResidentTileLod(key)) {
                return false;
            }
            slot = allocateResidentSlot();
        }
        residentSlot = static_cast<uint32_t>(slot);
        record = &residentRecords_[residentSlot];
        record->generation = std::max(record->generation, 1u);
        createdNewSlot = true;
    }

    if (record == nullptr) {
        return false;
    }

    const uint32_t existingMeshletUnits = record->meshletPages.pageCount * meshletPages_.pageSize;
    const uint32_t existingQuadUnits = record->quadPages.pageCount * quadPages_.pageSize;

    if (record->active &&
        existingMeshletUnits >= requiredMeshlets &&
        existingQuadUnits >= requiredQuadWords) {
        // Reuse the existing page run.
    } else {
        if (record->active) {
            meshletPages_.release(record->meshletPages);
            quadPages_.release(record->quadPages);
            record->meshletPages = {};
            record->quadPages = {};
        }
        if (!evictForAllocation(requiredMeshlets, requiredQuadWords, &upload.key)) {
            if (createdNewSlot) {
                freeResidentSlots_.push_back(residentSlot);
            }
            return false;
        }

        PageRun meshletRun{};
        PageRun quadRun{};
        if (!meshletPages_.allocate(requiredMeshlets, meshletRun) ||
            !quadPages_.allocate(requiredQuadWords, quadRun)) {
            if (meshletRun.valid()) {
                meshletPages_.release(meshletRun);
            }
            if (quadRun.valid()) {
                quadPages_.release(quadRun);
            }
            if (createdNewSlot) {
                freeResidentSlots_.push_back(residentSlot);
            }
            return false;
        }
        record->meshletPages = meshletRun;
        record->quadPages = quadRun;
    }

    const uint32_t meshletStart = meshletPages_.unitOffset(record->meshletPages);
    const uint32_t quadStart = quadPages_.unitOffset(record->quadPages);
    if (!writeAllocation(meshletStart, quadStart, *packed)) {
        if (createdNewSlot) {
            meshletPages_.release(record->meshletPages);
            quadPages_.release(record->quadPages);
            record->meshletPages = {};
            record->quadPages = {};
            freeResidentSlots_.push_back(residentSlot);
        }
        return false;
    }

    record->key = upload.key;
    record->packed = packed;
    record->tileAabb = computeTileAabb(*packed);
    record->gpu.meshletStart = meshletStart;
    record->gpu.meshletCount = requiredMeshlets;
    record->gpu.quadWordStart = quadStart;
    record->gpu.flags = kResidentFlagActive;
    record->gpu.minCorner = toVec4(record->tileAabb.minCorner);
    record->gpu.maxCorner = toVec4(record->tileAabb.maxCorner);
    record->lastTouchedRevision = static_cast<uint32_t>(std::min<uint64_t>(upload.revision, std::numeric_limits<uint32_t>::max()));
    record->active = true;
    if (createdNewSlot) {
        residentSlotByKey_[upload.key] = residentSlot;
    }

    setResidentLod(upload.key.tile, upload.key.lod, ResidentTileLodHandle{residentSlot, record->generation});
    markResidentDirty(residentSlot);
    sceneParamsDirty_ = true;
    return true;
}

MeshletBufferController::ApplyResult MeshletBufferController::applyDelta(const MeshStreamingDelta& delta) {
    ApplyResult result{};

    if (bufferManager_ == nullptr) {
        return result;
    }

    for (const MeshTileLodKey& key : delta.removals) {
        result.deltaApplied = removeResidentTileLod(key) || result.deltaApplied;
    }

    for (const MeshTileLodUpload& upsert : delta.upserts) {
        result.deltaApplied = upsertResidentTileLod(upsert) || result.deltaApplied;
    }

    for (const MeshTileSelectionEntry& selection : delta.selectionChanges) {
        setTileSelectedLod(selection.tile, selection.selectedLod);
        result.deltaApplied = true;
    }

    if (result.deltaApplied) {
        flushDirtyState();
    }

    if (delta.revision > uploadedMeshRevision_) {
        uploadedMeshRevision_ = delta.revision;
    }
    return result;
}

bool MeshletBufferController::flushDirtyState() {
    if (bufferManager_ == nullptr) {
        return false;
    }

    for (uint32_t residentSlot : dirtyResidentSlots_) {
        if (residentSlot >= residentDirtyMask_.size()) {
            continue;
        }
        residentDirtyMask_[residentSlot] = 0u;
        const ResidentTileLodGPU gpu = (residentSlot < residentRecords_.size())
            ? residentRecords_[residentSlot].gpu
            : ResidentTileLodGPU{};
        bufferManager_->writeBuffer(
            residentTileLodBufferName_,
            static_cast<uint64_t>(residentSlot) * sizeof(ResidentTileLodGPU),
            &gpu,
            sizeof(ResidentTileLodGPU)
        );
    }
    dirtyResidentSlots_.clear();

    for (uint32_t tileSlot : dirtyTileSlots_) {
        if (tileSlot >= tileDirtyMask_.size() || tileSlot >= tileSlotsGpu_.size()) {
            continue;
        }
        tileDirtyMask_[tileSlot] = 0u;
        bufferManager_->writeBuffer(
            tileSlotBufferName_,
            static_cast<uint64_t>(tileSlot) * sizeof(TileSlotGPU),
            &tileSlotsGpu_[tileSlot],
            sizeof(TileSlotGPU)
        );
    }
    dirtyTileSlots_.clear();

    for (uint32_t visibleIndex : dirtyVisibleIndices_) {
        if (visibleIndex >= visibleIndexDirtyMask_.size()) {
            continue;
        }
        visibleIndexDirtyMask_[visibleIndex] = 0u;
        const VisibleTileId value = (visibleIndex < visibleTileIds_.size()) ? visibleTileIds_[visibleIndex] : 0u;
        bufferManager_->writeBuffer(
            visibleTileIdBufferName_,
            static_cast<uint64_t>(visibleIndex) * sizeof(VisibleTileId),
            &value,
            sizeof(VisibleTileId)
        );
    }
    dirtyVisibleIndices_.clear();

    if (sceneParamsDirty_) {
        const TileSceneParamsGPU params{
            static_cast<uint32_t>(visibleTileIds_.size()),
            static_cast<uint32_t>(tileSlots_.size()),
            static_cast<uint32_t>(residentRecords_.size()),
            activeSelectionMeshletCount_
        };
        bufferManager_->writeBuffer(tileSceneParamsBufferName_, 0u, &params, sizeof(params));
        sceneParamsDirty_ = false;
    }

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
    return residentTileLodBufferName_.c_str();
}

const char* MeshletBufferController::activeMeshletRangeParamsBufferName() const noexcept {
    return tileSceneParamsBufferName_.c_str();
}

const char* MeshletBufferController::residentTileLodBufferName() const noexcept {
    return residentTileLodBufferName_.c_str();
}

const char* MeshletBufferController::tileSlotBufferName() const noexcept {
    return tileSlotBufferName_.c_str();
}

const char* MeshletBufferController::visibleTileIdBufferName() const noexcept {
    return visibleTileIdBufferName_.c_str();
}

const char* MeshletBufferController::tileSceneParamsBufferName() const noexcept {
    return tileSceneParamsBufferName_.c_str();
}

uint32_t MeshletBufferController::meshletCount() const noexcept {
    return activeSelectionMeshletCount_;
}

uint32_t MeshletBufferController::activeSelectionMeshletCount() const noexcept {
    return activeSelectionMeshletCount_;
}

uint32_t MeshletBufferController::activeRangeCount() const noexcept {
    return static_cast<uint32_t>(visibleTileIds_.size());
}

uint32_t MeshletBufferController::verticesPerMeshlet() const noexcept {
    return MESHLET_VERTEX_CAPACITY;
}

uint32_t MeshletBufferController::visibleTileCount() const noexcept {
    return static_cast<uint32_t>(visibleTileIds_.size());
}

uint32_t MeshletBufferController::residentTileCount() const noexcept {
    return static_cast<uint32_t>(residentSlotByKey_.size());
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
    for (VisibleTileId visibleTileId : visibleTileIds_) {
        if (visibleTileId >= tileSlots_.size()) {
            continue;
        }
        const ResidentRecord* resident = residentForHandle(tileSlots_[visibleTileId].selectedResident);
        if (resident == nullptr || !resident->packed) {
            continue;
        }
        activeMeshletBoundsCache_.insert(
            activeMeshletBoundsCache_.end(),
            resident->packed->bounds.begin(),
            resident->packed->bounds.end()
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
        meshletCount(),
        activeRangeCount(),
        verticesPerMeshlet()
    };
}
