#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <shared_mutex>
#include <utility>

namespace {
bool hasRenderablePackedData(const std::shared_ptr<const PackedMeshletData>& packed) {
    return packed != nullptr && !packed->metadata.empty();
}
}  // namespace

std::vector<MeshTileLodUpload> MeshManager::consumePendingTileLodUploads(std::size_t maxCount) {
    std::vector<MeshTileLodUpload> uploads;
    uploads.reserve(maxCount);

    std::unique_lock<std::shared_mutex> lock(meshMutex_);
    auto consumeFromQueue = [&](std::deque<MeshTileLodKey>& queue,
                                std::unordered_set<MeshTileLodKey>& set,
                                bool highPriority) {
        while (!queue.empty() && uploads.size() < maxCount) {
            const MeshTileLodKey key = queue.front();
            queue.pop_front();
            if (set.erase(key) == 0u) {
                continue;
            }
            if (highPriority) {
                pendingUploadSet_.erase(key);
            } else if (pendingPriorityUploadSet_.find(key) != pendingPriorityUploadSet_.end()) {
                continue;
            }

            const auto tileIt = meshTiles_.find(key.tile.tile);
            if (tileIt == meshTiles_.end()) {
                continue;
            }
            const auto lodIt = tileIt->second.lodStates.find(key.lod);
            if (lodIt == tileIt->second.lodStates.end()) {
                continue;
            }
            const auto sliceIt = lodIt->second.find(key.tile.z);
            if (sliceIt == lodIt->second.end() || !sliceIt->second.resident) {
                continue;
            }

            MeshTileLodUpload upload{};
            upload.key = key;
            upload.culledPacked = sliceIt->second.culledPacked;
            upload.doubleSidedPacked = sliceIt->second.doubleSidedPacked;
            upload.revision = sliceIt->second.revision;
            sliceIt->second.uploadQueued = false;
            uploads.push_back(std::move(upload));
        }
    };

    consumeFromQueue(pendingPriorityUploadOrder_, pendingPriorityUploadSet_, true);
    consumeFromQueue(pendingUploadOrder_, pendingUploadSet_, false);

    return uploads;
}

std::vector<MeshTileLodKey> MeshManager::consumePendingTileLodRemovals(std::size_t maxCount) {
    std::vector<MeshTileLodKey> removals;
    removals.reserve(maxCount);

    std::unique_lock<std::shared_mutex> lock(meshMutex_);
    while (!pendingRemovalOrder_.empty() && removals.size() < maxCount) {
        const MeshTileLodKey key = pendingRemovalOrder_.front();
        pendingRemovalOrder_.pop_front();
        pendingRemovalSet_.erase(key);
        removals.push_back(key);
    }

    return removals;
}

bool MeshManager::consumeSelectionSnapshot(uint64_t& outRevision,
                                           std::vector<MeshTileSelectionEntry>& outSelection) {
    std::unique_lock<std::shared_mutex> lock(meshMutex_);
    if (!selectionSnapshotDirty_) {
        return false;
    }

    outSelection.clear();
    outSelection.reserve(meshTiles_.size() * static_cast<std::size_t>(meshTileSliceCount_));
    for (const auto& [tileCoord, tileState] : meshTiles_) {
        if (tileState.selectedLod < 0) {
            continue;
        }
        for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
            outSelection.push_back(MeshTileSelectionEntry{
                MeshTileSliceCoord{tileCoord, zSlice},
                tileState.selectedLod
            });
        }
    }
    std::sort(outSelection.begin(), outSelection.end(), [](const MeshTileSelectionEntry& a, const MeshTileSelectionEntry& b) {
        return a.tile < b.tile;
    });

    ++selectionRevision_;
    outRevision = selectionRevision_;
    selectionSnapshotDirty_ = false;
    return true;
}

void MeshManager::queueTileLodUploadLocked(const MeshTileLodKey& key, bool highPriority) {
    if (highPriority) {
        if (pendingPriorityUploadSet_.insert(key).second) {
            pendingPriorityUploadOrder_.push_back(key);
        }
        return;
    }

    if (pendingPriorityUploadSet_.find(key) != pendingPriorityUploadSet_.end()) {
        return;
    }
    if (pendingUploadSet_.insert(key).second) {
        pendingUploadOrder_.push_back(key);
    }
}

void MeshManager::queueTileLodRemovalLocked(const MeshTileLodKey& key) {
    pendingUploadSet_.erase(key);
    pendingPriorityUploadSet_.erase(key);
    if (pendingRemovalSet_.insert(key).second) {
        pendingRemovalOrder_.push_back(key);
    }
}

int8_t MeshManager::chooseRenderableLodForTileLocked(const MeshTileState& state) const {
    auto hasResidentMesh = [this, &state](int32_t lod) {
        if (lod < 0) {
            return false;
        }
        const auto lodIt = state.lodStates.find(static_cast<uint8_t>(lod));
        if (lodIt == state.lodStates.end()) {
            return false;
        }

        for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
            const auto sliceIt = lodIt->second.find(zSlice);
            if (sliceIt == lodIt->second.end() || !sliceIt->second.resident) {
                return false;
            }
        }
        return true;
    };
    auto hasRenderableMesh = [this, &state, &hasResidentMesh](int32_t lod) {
        if (lod < 0) {
            return false;
        }
        if (!hasResidentMesh(lod)) {
            return false;
        }

        const auto lodIt = state.lodStates.find(static_cast<uint8_t>(lod));
        if (lodIt == state.lodStates.end()) {
            return false;
        }

        for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
            const auto sliceIt = lodIt->second.find(zSlice);
            if (sliceIt == lodIt->second.end()) {
                continue;
            }
            if (hasRenderablePackedData(sliceIt->second.culledPacked) ||
                hasRenderablePackedData(sliceIt->second.doubleSidedPacked)) {
                return true;
            }
        }

        return false;
    };

    if (state.desiredLod >= 0) {
        // Prefer desired LOD if available, then coarser fallbacks first.
        for (int32_t lod = state.desiredLod; lod < config_.lodLevelCount; ++lod) {
            if (hasRenderableMesh(lod)) {
                return static_cast<int8_t>(lod);
            }
        }
        for (int32_t lod = state.desiredLod - 1; lod >= 0; --lod) {
            if (hasRenderableMesh(lod)) {
                return static_cast<int8_t>(lod);
            }
        }
        // If every resident option is empty, keep the prior resident fallback behavior.
        for (int32_t lod = state.desiredLod; lod < config_.lodLevelCount; ++lod) {
            if (hasResidentMesh(lod)) {
                return static_cast<int8_t>(lod);
            }
        }
        for (int32_t lod = state.desiredLod - 1; lod >= 0; --lod) {
            if (hasResidentMesh(lod)) {
                return static_cast<int8_t>(lod);
            }
        }
        return -1;
    }

    // No desired LOD yet: prefer coarsest available.
    for (int32_t lod = config_.lodLevelCount - 1; lod >= 0; --lod) {
        if (hasRenderableMesh(lod)) {
            return static_cast<int8_t>(lod);
        }
    }
    for (int32_t lod = config_.lodLevelCount - 1; lod >= 0; --lod) {
        if (hasResidentMesh(lod)) {
            return static_cast<int8_t>(lod);
        }
    }
    return -1;
}

bool MeshManager::refreshSelectedLodLocked(MeshTileState& state) const {
    const int8_t selected = chooseRenderableLodForTileLocked(state);
    if (selected == state.selectedLod) {
        return false;
    }

    state.selectedLod = selected;
    return true;
}

bool MeshManager::hasPendingJobs() const {
    std::shared_lock<std::shared_mutex> lock(meshMutex_);
    return !pendingTileLodJobs_.empty() ||
           !pendingPriorityTileLodJobs_.empty() ||
           !deferredRemeshTileLods_.empty() ||
           !pendingPriorityUploadOrder_.empty() ||
           !pendingUploadOrder_.empty() ||
           !pendingRemovalOrder_.empty() ||
           selectionSnapshotDirty_;
}
