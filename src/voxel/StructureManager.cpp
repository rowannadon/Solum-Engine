#include "solum_engine/voxel/StructureManager.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "ogt_vox/ogt_vox.h"

#include "solum_engine/resources/Coords.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/Column.h"

namespace {

constexpr uint32_t kPointThresholdScale = 0xFFFFFFu;

uint64_t splitmix64(uint64_t value) {
    uint64_t x = value + 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

uint64_t packCellKey(int32_t x, int32_t y) {
    const uint32_t ux = static_cast<uint32_t>(x);
    const uint32_t uy = static_cast<uint32_t>(y);
    return (static_cast<uint64_t>(ux) << 32) | static_cast<uint64_t>(uy);
}

bool tieBreakCellOrder(int32_t ax, int32_t ay, int32_t bx, int32_t by) {
    if (ax != bx) {
        return ax < bx;
    }
    return ay < by;
}

glm::ivec3 transformVoxel(const glm::ivec3& local, const ogt_vox_transform& transform) {
    const glm::vec3 ax(transform.m00, transform.m10, transform.m20);
    const glm::vec3 ay(transform.m01, transform.m11, transform.m21);
    const glm::vec3 az(transform.m02, transform.m12, transform.m22);
    const glm::vec3 tr(transform.m30, transform.m31, transform.m32);

    const glm::vec3 world = (ax * static_cast<float>(local.x)) +
                            (ay * static_cast<float>(local.y)) +
                            (az * static_cast<float>(local.z)) +
                            tr;
    return glm::ivec3(
        static_cast<int32_t>(std::lround(world.x)),
        static_cast<int32_t>(std::lround(world.y)),
        static_cast<int32_t>(std::lround(world.z))
    );
}

bool mapColorToMaterial(const ogt_vox_rgba& color,
                        const std::vector<StructureManager::ColorMaterialMapping>& mappings,
                        BlockMaterial& outMaterial) {
    if (mappings.empty()) {
        return false;
    }

    for (const StructureManager::ColorMaterialMapping& mapping : mappings) {
        if (mapping.r == color.r &&
            mapping.g == color.g &&
            mapping.b == color.b &&
            mapping.a == color.a) {
            outMaterial = mapping.material;
            return outMaterial.unpack().id != 0;
        }
    }

    int32_t bestDistance = std::numeric_limits<int32_t>::max();
    std::size_t bestIndex = 0;
    for (std::size_t i = 0; i < mappings.size(); ++i) {
        const int32_t dr = static_cast<int32_t>(color.r) - static_cast<int32_t>(mappings[i].r);
        const int32_t dg = static_cast<int32_t>(color.g) - static_cast<int32_t>(mappings[i].g);
        const int32_t db = static_cast<int32_t>(color.b) - static_cast<int32_t>(mappings[i].b);
        const int32_t distance = (dr * dr) + (dg * dg) + (db * db);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = i;
        }
    }

    outMaterial = mappings[bestIndex].material;
    return outMaterial.unpack().id != 0;
}

}  // namespace

StructureManager::StructureManager()
    : StructureManager(SamplerConfig{}) {}

StructureManager::StructureManager(const SamplerConfig& samplerConfig)
    : samplerConfig_(samplerConfig) {
    cellSize_ = std::max(1, samplerConfig_.cellSize);
    minDistance_ = std::max(1, samplerConfig_.minDistance);
    minDistanceSq_ = static_cast<int64_t>(minDistance_) * static_cast<int64_t>(minDistance_);
    seed_ = static_cast<uint64_t>(samplerConfig_.seed);
    const float occupancy = std::clamp(samplerConfig_.cellOccupancy, 0.0f, 1.0f);
    occupancyThreshold_ = static_cast<uint32_t>(std::lround(occupancy * static_cast<float>(kPointThresholdScale)));

    const double ratio = static_cast<double>(minDistance_) / static_cast<double>(cellSize_);
    neighborRangeCells_ = std::max(1, static_cast<int32_t>(std::ceil(ratio)));
}

void StructureManager::clear() {
    structures_.clear();
    totalSelectionWeight_ = 0;
    maxHorizontalReach_ = 0;
}

bool StructureManager::addStructure(const StructureDefinition& definition) {
    LoadedStructure structure;
    if (!loadVoxStructure(definition, structure)) {
        return false;
    }

    if (structure.voxels.empty()) {
        std::cerr << "StructureManager: structure '" << definition.name
                  << "' has no mapped solid voxels after color filtering." << std::endl;
        return false;
    }

    structures_.push_back(std::move(structure));
    const LoadedStructure& inserted = structures_.back();
    totalSelectionWeight_ += static_cast<uint64_t>(std::max<uint32_t>(inserted.selectionWeight, 1u));
    maxHorizontalReach_ = std::max(maxHorizontalReach_, inserted.horizontalReach);
    std::cout << "StructureManager: loaded '" << inserted.name << "' voxels=" << inserted.voxels.size()
              << " horizontalReach=" << inserted.horizontalReach << std::endl;
    return true;
}

bool StructureManager::hasStructures() const noexcept {
    return !structures_.empty();
}

int32_t StructureManager::maxHorizontalReach() const noexcept {
    return maxHorizontalReach_;
}

void StructureManager::collectPointsForBounds(const glm::ivec2& minInclusive,
                                              const glm::ivec2& maxExclusive,
                                              std::vector<PlacementPoint>& outPoints) const {
    outPoints.clear();
    if (structures_.empty()) {
        return;
    }
    if (maxExclusive.x <= minInclusive.x || maxExclusive.y <= minInclusive.y) {
        return;
    }

    const int32_t maxXInclusive = maxExclusive.x - 1;
    const int32_t maxYInclusive = maxExclusive.y - 1;
    const int32_t minCellX = floor_div(minInclusive.x, cellSize_);
    const int32_t minCellY = floor_div(minInclusive.y, cellSize_);
    const int32_t maxCellX = floor_div(maxXInclusive, cellSize_);
    const int32_t maxCellY = floor_div(maxYInclusive, cellSize_);

    const int32_t cellSpanX = (maxCellX - minCellX) + 1;
    const int32_t cellSpanY = (maxCellY - minCellY) + 1;
    if (cellSpanX <= 0 || cellSpanY <= 0) {
        return;
    }

    const int64_t estimatedCount = static_cast<int64_t>(cellSpanX) * static_cast<int64_t>(cellSpanY);
    if (estimatedCount > 0 && estimatedCount < static_cast<int64_t>(std::numeric_limits<size_t>::max())) {
        outPoints.reserve(static_cast<size_t>(estimatedCount));
    }

    for (int32_t cellY = minCellY; cellY <= maxCellY; ++cellY) {
        for (int32_t cellX = minCellX; cellX <= maxCellX; ++cellX) {
            const CandidatePoint candidate = makeCandidateForCell(cellX, cellY);
            if (!candidate.active) {
                continue;
            }
            if (candidate.worldX < minInclusive.x || candidate.worldX >= maxExclusive.x ||
                candidate.worldY < minInclusive.y || candidate.worldY >= maxExclusive.y) {
                continue;
            }
            if (!isCandidateAccepted(candidate)) {
                continue;
            }

            outPoints.push_back(PlacementPoint{
                glm::ivec2{candidate.worldX, candidate.worldY},
                candidate.key
            });
        }
    }

    std::sort(outPoints.begin(), outPoints.end(), [](const PlacementPoint& a, const PlacementPoint& b) {
        if (a.worldXY.y != b.worldXY.y) {
            return a.worldXY.y < b.worldXY.y;
        }
        if (a.worldXY.x != b.worldXY.x) {
            return a.worldXY.x < b.worldXY.x;
        }
        return a.key < b.key;
    });
}

void StructureManager::placeStructureForPoint(const PlacementPoint& point,
                                              const glm::ivec3& anchorWorld,
                                              const glm::ivec3& clipMinInclusive,
                                              const glm::ivec3& clipMaxExclusive,
                                              Column& column) const {
    if (structures_.empty()) {
        return;
    }
    if (clipMaxExclusive.x <= clipMinInclusive.x ||
        clipMaxExclusive.y <= clipMinInclusive.y ||
        clipMaxExclusive.z <= clipMinInclusive.z) {
        return;
    }

    const std::size_t structureIndex = pickStructureIndex(point.key);
    if (structureIndex >= structures_.size()) {
        return;
    }

    const LoadedStructure& structure = structures_[structureIndex];
    for (const LoadedVoxel& voxel : structure.voxels) {
        const glm::ivec3 offset = voxel.local - structure.generationOrigin;
        const glm::ivec3 worldPos = anchorWorld + offset;

        if (worldPos.x < clipMinInclusive.x || worldPos.x >= clipMaxExclusive.x ||
            worldPos.y < clipMinInclusive.y || worldPos.y >= clipMaxExclusive.y ||
            worldPos.z < clipMinInclusive.z || worldPos.z >= clipMaxExclusive.z) {
            continue;
        }
        if (worldPos.z < 0 || worldPos.z >= cfg::COLUMN_HEIGHT_BLOCKS) {
            continue;
        }

        const int32_t localX = worldPos.x - clipMinInclusive.x;
        const int32_t localY = worldPos.y - clipMinInclusive.y;
        if (localX < 0 || localX >= cfg::CHUNK_SIZE || localY < 0 || localY >= cfg::CHUNK_SIZE) {
            continue;
        }

        column.setBlock(
            static_cast<uint8_t>(localX),
            static_cast<uint8_t>(localY),
            static_cast<uint16_t>(worldPos.z),
            voxel.material
        );
    }
}

StructureManager::CandidatePoint StructureManager::makeCandidateForCell(int32_t cellX, int32_t cellY) const {
    CandidatePoint candidate;
    candidate.cellX = cellX;
    candidate.cellY = cellY;

    const uint64_t hash = splitmix64(packCellKey(cellX, cellY) ^ seed_);
    const uint32_t occupancyValue = static_cast<uint32_t>((hash >> 32) & kPointThresholdScale);
    if (occupancyValue > occupancyThreshold_) {
        candidate.active = false;
        return candidate;
    }

    const uint32_t jitterXBits = static_cast<uint32_t>((hash >> 8) & 0xFFFFu);
    const uint32_t jitterYBits = static_cast<uint32_t>((hash >> 24) & 0xFFFFu);
    const int32_t jitterX = static_cast<int32_t>(jitterXBits % static_cast<uint32_t>(cellSize_));
    const int32_t jitterY = static_cast<int32_t>(jitterYBits % static_cast<uint32_t>(cellSize_));

    candidate.worldX = (cellX * cellSize_) + jitterX;
    candidate.worldY = (cellY * cellSize_) + jitterY;
    candidate.priority = splitmix64(hash ^ 0xA0B1C2D3E4F56789ull);
    candidate.key = splitmix64(hash ^ 0xBADC0FFEE0DDF00Dull);
    candidate.active = true;
    return candidate;
}

bool StructureManager::isCandidateAccepted(const CandidatePoint& candidate) const {
    if (!candidate.active) {
        return false;
    }

    for (int32_t dy = -neighborRangeCells_; dy <= neighborRangeCells_; ++dy) {
        for (int32_t dx = -neighborRangeCells_; dx <= neighborRangeCells_; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }

            const CandidatePoint neighbor = makeCandidateForCell(candidate.cellX + dx, candidate.cellY + dy);
            if (!neighbor.active) {
                continue;
            }

            const int64_t diffX = static_cast<int64_t>(candidate.worldX) - static_cast<int64_t>(neighbor.worldX);
            const int64_t diffY = static_cast<int64_t>(candidate.worldY) - static_cast<int64_t>(neighbor.worldY);
            const int64_t distanceSq = (diffX * diffX) + (diffY * diffY);
            if (distanceSq >= minDistanceSq_) {
                continue;
            }

            if (neighbor.priority < candidate.priority) {
                return false;
            }
            if (neighbor.priority == candidate.priority &&
                tieBreakCellOrder(neighbor.cellX, neighbor.cellY, candidate.cellX, candidate.cellY)) {
                return false;
            }
        }
    }

    return true;
}

std::size_t StructureManager::pickStructureIndex(uint64_t pointKey) const {
    if (structures_.empty()) {
        return std::numeric_limits<std::size_t>::max();
    }
    if (structures_.size() == 1 || totalSelectionWeight_ == 0) {
        return 0;
    }

    const uint64_t selector = splitmix64(pointKey ^ seed_) % totalSelectionWeight_;
    uint64_t running = 0;
    for (std::size_t i = 0; i < structures_.size(); ++i) {
        running += static_cast<uint64_t>(std::max<uint32_t>(structures_[i].selectionWeight, 1u));
        if (selector < running) {
            return i;
        }
    }

    return structures_.size() - 1;
}

bool StructureManager::loadVoxStructure(const StructureDefinition& definition, LoadedStructure& outStructure) const {
    outStructure = LoadedStructure{};
    outStructure.name = definition.name;
    outStructure.generationOrigin = definition.generationOrigin;
    outStructure.selectionWeight = std::max<uint32_t>(definition.selectionWeight, 1u);

    if (definition.voxFilePath.empty()) {
        std::cerr << "StructureManager: skipping unnamed path for structure '" << definition.name << "'." << std::endl;
        return false;
    }
    if (definition.colorMappings.empty()) {
        std::cerr << "StructureManager: structure '" << definition.name
                  << "' has no color mappings; nothing to load." << std::endl;
        return false;
    }

    std::ifstream file(definition.voxFilePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "StructureManager: failed to open .vox file '" << definition.voxFilePath
                  << "' for structure '" << definition.name << "'." << std::endl;
        return false;
    }

    const std::streampos endPos = file.tellg();
    if (endPos <= 0) {
        std::cerr << "StructureManager: .vox file '" << definition.voxFilePath
                  << "' is empty or unreadable." << std::endl;
        return false;
    }
    if (endPos > static_cast<std::streampos>(std::numeric_limits<uint32_t>::max())) {
        std::cerr << "StructureManager: .vox file '" << definition.voxFilePath
                  << "' is too large for ogt_vox." << std::endl;
        return false;
    }

    const std::streamsize fileSize = static_cast<std::streamsize>(endPos);
    std::vector<uint8_t> buffer(static_cast<size_t>(fileSize), 0u);
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), fileSize)) {
        std::cerr << "StructureManager: failed to read .vox file '" << definition.voxFilePath << "'." << std::endl;
        return false;
    }

    const ogt_vox_scene* scene = ogt_vox_read_scene(buffer.data(), static_cast<uint32_t>(buffer.size()));
    if (scene == nullptr) {
        std::cerr << "StructureManager: ogt_vox failed to parse '" << definition.voxFilePath << "'." << std::endl;
        return false;
    }

    bool loaded = false;
    do {
        if (scene->num_models == 0 || scene->models == nullptr) {
            std::cerr << "StructureManager: .vox file '" << definition.voxFilePath
                      << "' has no readable models." << std::endl;
            break;
        }

        struct TempVoxel {
            glm::ivec3 world{0, 0, 0};
            BlockMaterial material{};
        };

        std::vector<TempVoxel> tempVoxels;
        glm::ivec3 minCorner{
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max()
        };

        auto addModelInstance = [&](const ogt_vox_model* model, const ogt_vox_transform& transform) {
            if (model == nullptr || model->voxel_data == nullptr ||
                model->size_x == 0 || model->size_y == 0 || model->size_z == 0) {
                return;
            }

            for (uint32_t z = 0; z < model->size_z; ++z) {
                for (uint32_t y = 0; y < model->size_y; ++y) {
                    for (uint32_t x = 0; x < model->size_x; ++x) {
                        const uint64_t voxelIndex64 =
                            static_cast<uint64_t>(x) +
                            static_cast<uint64_t>(y) * static_cast<uint64_t>(model->size_x) +
                            static_cast<uint64_t>(z) * static_cast<uint64_t>(model->size_x) * static_cast<uint64_t>(model->size_y);
                        const uint8_t colorIndex = model->voxel_data[static_cast<size_t>(voxelIndex64)];
                        if (colorIndex == 0) {
                            continue;
                        }

                        const ogt_vox_rgba color = scene->palette.color[colorIndex];
                        if (color.a == 0) {
                            continue;
                        }

                        BlockMaterial mappedMaterial{};
                        if (!mapColorToMaterial(color, definition.colorMappings, mappedMaterial)) {
                            continue;
                        }

                        const glm::ivec3 worldPos = transformVoxel(glm::ivec3{
                            static_cast<int32_t>(x),
                            static_cast<int32_t>(y),
                            static_cast<int32_t>(z)
                        }, transform);

                        minCorner = glm::min(minCorner, worldPos);
                        tempVoxels.push_back(TempVoxel{worldPos, mappedMaterial});
                    }
                }
            }
        };

        if (scene->num_instances > 0 && scene->instances != nullptr) {
            for (uint32_t i = 0; i < scene->num_instances; ++i) {
                const ogt_vox_instance& instance = scene->instances[i];
                bool hidden = instance.hidden;
                if (!hidden && scene->layers != nullptr && instance.layer_index < scene->num_layers) {
                    hidden = scene->layers[instance.layer_index].hidden;
                }
                if (!hidden && scene->groups != nullptr && instance.group_index < scene->num_groups) {
                    hidden = scene->groups[instance.group_index].hidden;
                }
                if (hidden || instance.model_index >= scene->num_models) {
                    continue;
                }

                addModelInstance(scene->models[instance.model_index], instance.transform);
            }
        } else {
            const ogt_vox_transform identity = ogt_vox_transform_get_identity();
            for (uint32_t modelIndex = 0; modelIndex < scene->num_models; ++modelIndex) {
                addModelInstance(scene->models[modelIndex], identity);
            }
        }

        if (tempVoxels.empty()) {
            std::cerr << "StructureManager: structure '" << definition.name
                      << "' produced no mapped voxels. Check color mappings for '"
                      << definition.voxFilePath << "'." << std::endl;
            break;
        }

        const glm::ivec3 originPoint = minCorner + definition.generationOrigin;
        outStructure.voxels.reserve(tempVoxels.size());
        int32_t localReach = 0;
        for (const TempVoxel& voxel : tempVoxels) {
            const glm::ivec3 local = voxel.world - minCorner;
            const glm::ivec3 offset = voxel.world - originPoint;
            localReach = std::max(localReach, std::abs(offset.x));
            localReach = std::max(localReach, std::abs(offset.y));
            outStructure.voxels.push_back(LoadedVoxel{local, voxel.material});
        }

        outStructure.horizontalReach = localReach;
        loaded = true;
    } while (false);

    ogt_vox_destroy_scene(scene);
    return loaded;
}
