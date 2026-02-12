#include "solum_engine/voxel/ChunkMesher.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/BlockMaterial.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace {
	constexpr int kChunkSize = CHUNK_SIZE;
	constexpr int kChunkSizePadded = CHUNK_SIZE_P;
	constexpr int kChunkPlaneArea = kChunkSize * kChunkSize;
	constexpr int kPaddedPlaneArea = kChunkSizePadded * kChunkSizePadded;
	constexpr int kPaddedBlockCount = kChunkSizePadded * kChunkSizePadded * kChunkSizePadded;

	inline bool IsSolid(const BlockMaterial& block) {
		return ((block.data >> 16u) & 0xFFFFu) != 0u;
	}

	inline uint8_t EncodeNormalComponent(int component) {
		if (component > 0) return 255;
		if (component < 0) return 0;
		return 127;
	}

	const std::array<std::array<glm::ivec3, 4>, 6> kFaceVertexOffsets = { {
		// +X
		{ glm::ivec3(1, 0, 0), glm::ivec3(1, 0, 1), glm::ivec3(1, 1, 0), glm::ivec3(1, 1, 1) },
		// -X
		{ glm::ivec3(0, 0, 0), glm::ivec3(0, 1, 0), glm::ivec3(0, 0, 1), glm::ivec3(0, 1, 1) },
		// +Y
		{ glm::ivec3(0, 1, 0), glm::ivec3(1, 1, 0), glm::ivec3(0, 1, 1), glm::ivec3(1, 1, 1) },
		// -Y
		{ glm::ivec3(0, 0, 0), glm::ivec3(0, 0, 1), glm::ivec3(1, 0, 0), glm::ivec3(1, 0, 1) },
		// +Z
		{ glm::ivec3(0, 0, 1), glm::ivec3(0, 1, 1), glm::ivec3(1, 0, 1), glm::ivec3(1, 1, 1) },
		// -Z
		{ glm::ivec3(0, 0, 0), glm::ivec3(1, 0, 0), glm::ivec3(0, 1, 0), glm::ivec3(1, 1, 0) },
	} };

	const std::array<glm::ivec3, 6> kFaceNormals = { {
		glm::ivec3(1, 0, 0),
		glm::ivec3(-1, 0, 0),
		glm::ivec3(0, 1, 0),
		glm::ivec3(0, -1, 0),
		glm::ivec3(0, 0, 1),
		glm::ivec3(0, 0, -1),
	} };

	constexpr std::array<std::array<uint8_t, 6>, 6> kFaceIndexOrder = { {
		{ 0, 1, 2, 1, 3, 2 }, // +X
		{ 0, 2, 1, 1, 2, 3 }, // -X
		{ 0, 2, 1, 1, 2, 3 }, // +Y
		{ 0, 1, 2, 1, 3, 2 }, // -Y
		{ 0, 1, 2, 1, 3, 2 }, // +Z
		{ 0, 2, 1, 1, 3, 2 }, // -Z
	} };
}

std::pair<std::vector<VertexAttributes>, std::vector<uint16_t>> ChunkMesher::mesh(Chunk& chunk, const std::vector<Chunk*>& neighbors) {
	std::array<BlockMaterial, kPaddedBlockCount> paddedBlockData;
	paddedBlockData.fill(BlockMaterial{ 0 });
	vertices.clear();
	indices.clear();

	BlockCoord chunkOrigin = chunk_to_block_origin(chunk.pos);

	auto paddedIndex = [&](int x, int y, int z) {
		return (x * kPaddedPlaneArea) + (y * kChunkSizePadded) + z;
		};

	const BlockMaterial* chunkData = chunk.data.data();
	auto paddedPtr = paddedBlockData.data();

	for (int x = 0; x < kChunkSize; ++x) {
		const BlockMaterial* srcPlane = chunkData + x * kChunkPlaneArea;
		for (int y = 0; y < kChunkSize; ++y) {
			BlockMaterial* dstRow = paddedPtr + paddedIndex(x + 1, y + 1, 1);
			const BlockMaterial* srcRow = srcPlane + y * kChunkSize;
			std::memcpy(dstRow, srcRow, kChunkSize * sizeof(BlockMaterial));
		}
	}

	std::array<Chunk*, 6> neighborChunks{};
	neighborChunks.fill(nullptr);
	const size_t neighborCount = std::min(neighbors.size(), neighborChunks.size());
	for (size_t i = 0; i < neighborCount; ++i) {
		neighborChunks[i] = neighbors[i];
	}

	auto copyXPlane = [&](int destX, int srcX, const BlockMaterial* src) {
		const BlockMaterial* srcPlane = src + srcX * kChunkPlaneArea;
		for (int y = 0; y < kChunkSize; ++y) {
			BlockMaterial* dstRow = paddedPtr + paddedIndex(destX, y + 1, 1);
			const BlockMaterial* srcRow = srcPlane + y * kChunkSize;
			std::memcpy(dstRow, srcRow, kChunkSize * sizeof(BlockMaterial));
		}
		};

	auto copyYPlane = [&](int destY, int srcY, const BlockMaterial* src) {
		for (int x = 0; x < kChunkSize; ++x) {
			const BlockMaterial* srcRow = src + x * kChunkPlaneArea + srcY * kChunkSize;
			BlockMaterial* dstRow = paddedPtr + paddedIndex(x + 1, destY, 1);
			std::memcpy(dstRow, srcRow, kChunkSize * sizeof(BlockMaterial));
		}
		};

	auto copyZPlane = [&](int destZ, int srcZ, const BlockMaterial* src) {
		for (int x = 0; x < kChunkSize; ++x) {
			const BlockMaterial* srcPlane = src + x * kChunkPlaneArea;
			for (int y = 0; y < kChunkSize; ++y) {
				paddedPtr[paddedIndex(x + 1, y + 1, destZ)] = srcPlane[y * kChunkSize + srcZ];
			}
		}
		};

	for (int dir = 0; dir < 6; ++dir) {
		Chunk* neighbor = neighborChunks[dir];
		if (!neighbor) {
			continue;
		}

		const BlockMaterial* neighborData = neighbor->data.data();
		switch (static_cast<Direction>(dir)) {
		case Direction::PlusX:
			copyXPlane(kChunkSize + 1, 0, neighborData);
			break;
		case Direction::MinusX:
			copyXPlane(0, kChunkSize - 1, neighborData);
			break;
		case Direction::PlusY:
			copyYPlane(kChunkSize + 1, 0, neighborData);
			break;
		case Direction::MinusY:
			copyYPlane(0, kChunkSize - 1, neighborData);
			break;
		case Direction::PlusZ:
			copyZPlane(kChunkSize + 1, 0, neighborData);
			break;
		case Direction::MinusZ:
			copyZPlane(0, kChunkSize - 1, neighborData);
			break;
		}
	}

	vertices.reserve(kChunkSize * kChunkSize * 12);
	indices.reserve(kChunkSize * kChunkSize * 12);

	constexpr uint16_t uvMax = std::numeric_limits<uint16_t>::max();

	for (int x = 0; x < kChunkSize; ++x) {
		for (int y = 0; y < kChunkSize; ++y) {
			for (int z = 0; z < kChunkSize; ++z) {
				const BlockMaterial block = chunkData[x * kChunkPlaneArea + y * kChunkSize + z];
				if (!IsSolid(block)) {
					continue;
				}

				const int paddedX = x + 1;
				const int paddedY = y + 1;
				const int paddedZ = z + 1;
				const uint16_t materialId = static_cast<uint16_t>((block.data >> 16u) & 0xFFFFu);

				for (int dir = 0; dir < 6; ++dir) {
					const glm::ivec3 offset = directionOffsets[dir];
					const int neighborX = paddedX + offset.x;
					const int neighborY = paddedY + offset.y;
					const int neighborZ = paddedZ + offset.z;
					const BlockMaterial neighborBlock = paddedBlockData[paddedIndex(neighborX, neighborY, neighborZ)];
					if (IsSolid(neighborBlock)) {
						continue;
					}

					if (vertices.size() > std::numeric_limits<uint16_t>::max() - 4) {
						continue;
					}

					uint16_t baseIndex = static_cast<uint16_t>(vertices.size());
					for (const glm::ivec3& vertexOffset : kFaceVertexOffsets[dir]) {
						VertexAttributes vertex{};

						vertex.x = static_cast<uint16_t>(x + vertexOffset.x + chunkOrigin.x());
						vertex.y = static_cast<uint16_t>(y + vertexOffset.y + chunkOrigin.y());
						vertex.z = static_cast<uint16_t>(z + vertexOffset.z + chunkOrigin.z());

						switch (static_cast<Direction>(dir)) {
						case Direction::PlusX:
						case Direction::MinusX:
							vertex.u = vertexOffset.z ? uvMax : 0;
							vertex.v = vertexOffset.y ? uvMax : 0;
							break;
						case Direction::PlusY:
						case Direction::MinusY:
							vertex.u = vertexOffset.x ? uvMax : 0;
							vertex.v = vertexOffset.z ? uvMax : 0;
							break;
						case Direction::PlusZ:
						case Direction::MinusZ:
							vertex.u = vertexOffset.x ? uvMax : 0;
							vertex.v = vertexOffset.y ? uvMax : 0;
							break;
						}

						vertex.material = materialId;
						vertex.n_x = EncodeNormalComponent(kFaceNormals[dir].x);
						vertex.n_y = EncodeNormalComponent(kFaceNormals[dir].y);
						vertex.n_z = EncodeNormalComponent(kFaceNormals[dir].z);

						vertices.push_back(vertex);
					}

					for (uint8_t idx : kFaceIndexOrder[dir]) {
						indices.push_back(static_cast<uint16_t>(baseIndex + idx));
					}
				}
			}
		}
	}

	return { vertices, indices };
}
