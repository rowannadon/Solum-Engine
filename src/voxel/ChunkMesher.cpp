#include "solum_engine/voxel/ChunkMesher.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/BlockMaterial.h"

#include <algorithm>
#include <array>
#include <climits>
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

	inline uint16_t ClampU16(int32_t value) {
		return static_cast<uint16_t>(std::clamp(value, 0, 65535));
	}

	inline uint8_t DecodeNormalSign(uint8_t encoded) {
		if (encoded > 170u) {
			return 2u; // +1
		}
		if (encoded < 85u) {
			return 0u; // -1
		}
		return 1u; // 0
	}

	inline uint8_t EncodeNormalAxisIndex(const VertexAttributes& vertex) {
		const uint8_t sx = DecodeNormalSign(vertex.n_x);
		const uint8_t sy = DecodeNormalSign(vertex.n_y);
		const uint8_t sz = DecodeNormalSign(vertex.n_z);

		if (sx == 2u) return 0u; // +X
		if (sx == 0u) return 1u; // -X
		if (sy == 2u) return 2u; // +Y
		if (sy == 0u) return 3u; // -Y
		if (sz == 2u) return 4u; // +Z
		return 5u;               // -Z
	}

	inline PackedVertexAttributes PackVertex(const VertexAttributes& vertex, const glm::ivec3& origin) {
		const uint16_t relX = ClampU16(vertex.x - origin.x);
		const uint16_t relY = ClampU16(vertex.y - origin.y);
		const uint16_t relZ = ClampU16(vertex.z - origin.z);
		const uint32_t uBit = (vertex.u > 32767u) ? 1u : 0u;
		const uint32_t vBit = (vertex.v > 32767u) ? 1u : 0u;
		const uint32_t normalBits = static_cast<uint32_t>(EncodeNormalAxisIndex(vertex)) & 0x7u;
		const uint32_t lodBits = static_cast<uint32_t>(vertex.lodLevel) & 0xFFu;

		PackedVertexAttributes packed{};
		packed.xy = static_cast<uint32_t>(relX) | (static_cast<uint32_t>(relY) << 16u);
		packed.zMaterial = static_cast<uint32_t>(relZ) | (static_cast<uint32_t>(vertex.material) << 16u);
		packed.packedFlags = uBit | (vBit << 1u) | (normalBits << 2u) | (lodBits << 5u);
		return packed;
	}

	MeshData BuildMeshletsFromTriangles(const std::vector<VertexAttributes>& vertices, const std::vector<uint32_t>& indices) {
		MeshData meshData;
		if (vertices.empty() || indices.empty()) {
			return meshData;
		}

		const std::size_t quadCount = indices.size() / 6ull;
		if (quadCount == 0) {
			return meshData;
		}

		const std::size_t meshletCount = (quadCount + MeshData::kMeshletQuadCapacity - 1ull) / MeshData::kMeshletQuadCapacity;
		meshData.meshlets.reserve(meshletCount);
		meshData.packedVertices.reserve(meshletCount * MeshData::kMeshletVertexCapacity);
		meshData.packedIndices.reserve(meshletCount * MeshData::kMeshletIndexCapacity);

		for (std::size_t quadBase = 0; quadBase < quadCount; quadBase += MeshData::kMeshletQuadCapacity) {
			const uint32_t quadsInMeshlet = static_cast<uint32_t>(std::min<std::size_t>(MeshData::kMeshletQuadCapacity, quadCount - quadBase));
			const std::size_t globalVertexBase = quadBase * 4ull;
			if (globalVertexBase >= vertices.size()) {
				break;
			}

			const std::size_t requestedVertexCount = static_cast<std::size_t>(quadsInMeshlet) * 4ull;
			const std::size_t availableVertexCount = std::min<std::size_t>(requestedVertexCount, vertices.size() - globalVertexBase);
			if (availableVertexCount == 0) {
				continue;
			}

			glm::ivec3 origin{INT_MAX, INT_MAX, INT_MAX};
			for (std::size_t i = 0; i < availableVertexCount; ++i) {
				const VertexAttributes& v = vertices[globalVertexBase + i];
				origin.x = std::min(origin.x, v.x);
				origin.y = std::min(origin.y, v.y);
				origin.z = std::min(origin.z, v.z);
			}

			MeshletInfo info{};
			info.origin = origin;
			info.lodLevel = static_cast<uint32_t>(vertices[globalVertexBase].lodLevel);
			info.quadCount = quadsInMeshlet;
			info.vertexOffset = static_cast<uint32_t>(meshData.packedVertices.size());
			info.indexOffset = static_cast<uint32_t>(meshData.packedIndices.size());

			for (std::size_t i = 0; i < availableVertexCount; ++i) {
				meshData.packedVertices.push_back(PackVertex(vertices[globalVertexBase + i], origin));
			}
			while ((meshData.packedVertices.size() - info.vertexOffset) < MeshData::kMeshletVertexCapacity) {
				meshData.packedVertices.push_back(PackedVertexAttributes{});
			}

			const std::size_t globalIndexBase = quadBase * 6ull;
			const std::size_t indexCount = static_cast<std::size_t>(quadsInMeshlet) * 6ull;
			for (std::size_t i = 0; i < indexCount; ++i) {
				uint32_t localIndex = 0u;
				const std::size_t globalIndexPos = globalIndexBase + i;
				if (globalIndexPos < indices.size()) {
					const uint32_t globalIndex = indices[globalIndexPos];
					if (globalIndex >= globalVertexBase && globalIndex < (globalVertexBase + availableVertexCount)) {
						localIndex = static_cast<uint32_t>(globalIndex - globalVertexBase);
					}
				}
				meshData.packedIndices.push_back(localIndex);
			}
			while ((meshData.packedIndices.size() - info.indexOffset) < MeshData::kMeshletIndexCapacity) {
				meshData.packedIndices.push_back(0u);
			}

			meshData.meshlets.push_back(info);
		}

		return meshData;
	}
}

MeshData ChunkMesher::mesh(Chunk& chunk, const std::vector<Chunk*>& neighbors) {
	std::array<BlockMaterial, kPaddedBlockCount> paddedBlockData;
	paddedBlockData.fill(BlockMaterial{ 0 });
	vertices.clear();
	indices.clear();

	BlockCoord chunkOrigin = chunk_to_block_origin(chunk.coord());

	auto paddedIndex = [&](int x, int y, int z) {
		return (x * kPaddedPlaneArea) + (y * kChunkSizePadded) + z;
		};

	const BlockMaterial* chunkData = chunk.getBlockData();
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

		const BlockMaterial* neighborData = neighbor->getBlockData();
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

					const uint32_t baseIndex = static_cast<uint32_t>(vertices.size());
					for (const glm::ivec3& vertexOffset : kFaceVertexOffsets[dir]) {
						VertexAttributes vertex{};

						vertex.x = x + vertexOffset.x + chunkOrigin.x();
						vertex.y = y + vertexOffset.y + chunkOrigin.y();
						vertex.z = z + vertexOffset.z + chunkOrigin.z();

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
						indices.push_back(baseIndex + static_cast<uint32_t>(idx));
					}
				}
			}
		}
	}

	return BuildMeshletsFromTriangles(vertices, indices);
}
