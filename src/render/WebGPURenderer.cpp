#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <exception>
#include <unordered_set>

#include "solum_engine/render/WebGPURenderer.h"

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_wgpu.h>

namespace {
constexpr glm::vec4 kChunkBoundsColor{0.2f, 0.95f, 0.35f, 0.22f};
constexpr glm::vec4 kColumnBoundsColor{1.0f, 0.7f, 0.2f, 0.6f};
constexpr glm::vec4 kRegionBoundsColor{0.2f, 0.8f, 1.0f, 0.95f};

struct PreparedMeshUploadData {
	std::vector<MeshletMetadataGPU> metadata;
	std::vector<uint16_t> quadData;
	uint32_t totalMeshletCount = 0;
	uint32_t totalQuadCount = 0;
	uint32_t requiredMeshletCapacity = 64u;
	uint32_t requiredQuadCapacity = 64u * MESHLET_QUAD_CAPACITY;
};

PreparedMeshUploadData prepareMeshUploadData(const std::vector<Meshlet>& meshlets) {
	PreparedMeshUploadData prepared;

	for (const Meshlet& meshlet : meshlets) {
		if (meshlet.quadCount == 0) {
			continue;
		}
		++prepared.totalMeshletCount;
		prepared.totalQuadCount += meshlet.quadCount;
	}

	prepared.metadata.reserve(prepared.totalMeshletCount);
	prepared.quadData.reserve(prepared.totalQuadCount);

	for (const Meshlet& meshlet : meshlets) {
		if (meshlet.quadCount == 0) {
			continue;
		}

		MeshletMetadataGPU metadata{};
		metadata.originX = meshlet.origin.x;
		metadata.originY = meshlet.origin.y;
		metadata.originZ = meshlet.origin.z;
		metadata.quadCount = meshlet.quadCount;
		metadata.faceDirection = meshlet.faceDirection;
		metadata.dataOffset = static_cast<uint32_t>(prepared.quadData.size());
		metadata.voxelScale = std::max(meshlet.voxelScale, 1u);
		prepared.metadata.push_back(metadata);

		prepared.quadData.insert(
			prepared.quadData.end(),
			meshlet.packedQuadLocalOffsets.begin(),
			meshlet.packedQuadLocalOffsets.begin() + static_cast<std::ptrdiff_t>(meshlet.quadCount)
		);
	}

	prepared.requiredMeshletCapacity = std::max(prepared.totalMeshletCount + 16u, 64u);
	prepared.requiredQuadCapacity = std::max(
		prepared.totalQuadCount + 1024u,
		prepared.requiredMeshletCapacity * MESHLET_QUAD_CAPACITY
	);

	return prepared;
}

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

WebGPURenderer::~WebGPURenderer() {
	stopStreamingThread();
}

bool WebGPURenderer::initialize() {
	RenderConfig config;

	context = std::make_unique<WebGPUContext>();
	if (!context->initialize(config)) {
		return false;
	}

	pipelineManager = std::make_unique<PipelineManager>(context->getDevice(), context->getSurfaceFormat());
	bufferManager = std::make_unique<BufferManager>(context->getDevice(), context->getQueue());
	textureManager = std::make_unique<TextureManager>(context->getDevice(), context->getQueue());
	meshletManager = std::make_unique<MeshletManager>();

	services_.emplace(*bufferManager, *textureManager, *pipelineManager, *context);

	{
		BufferDescriptor desc = Default;
		desc.label = StringView("uniform buffer");
		desc.size = sizeof(FrameUniforms);
		desc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
		desc.mappedAtCreation = false;
		Buffer ubo = bufferManager->createBuffer("uniform_buffer", desc);
		if (!ubo) return false;
	}

	World::Config worldConfig;
	worldConfig.columnLoadRadius = 256;
	worldConfig.jobConfig.worker_threads = 2;

	MeshManager::Config meshConfig;
	meshConfig.lodChunkRadii = {16, 48, 96, 144};
	meshConfig.jobConfig.worker_threads = worldConfig.jobConfig.worker_threads;
	const int32_t clampedWorldRadius = std::max(1, worldConfig.columnLoadRadius);
	for (int32_t& lodRadius : meshConfig.lodChunkRadii) {
		lodRadius = std::min(lodRadius, clampedWorldRadius);
	}
	std::sort(meshConfig.lodChunkRadii.begin(), meshConfig.lodChunkRadii.end());
	meshConfig.lodChunkRadii.erase(
		std::unique(meshConfig.lodChunkRadii.begin(), meshConfig.lodChunkRadii.end()),
		meshConfig.lodChunkRadii.end()
	);
	if (meshConfig.lodChunkRadii.empty()) {
		meshConfig.lodChunkRadii.push_back(clampedWorldRadius);
	}

	world_ = std::make_unique<World>(worldConfig);
	voxelMeshManager_ = std::make_unique<MeshManager>(*world_, meshConfig);
	uploadColumnRadius_ = std::min(
		clampedWorldRadius,
		std::max(1, meshConfig.lodChunkRadii.back() + 1)
	);

	const glm::vec3 initialPlayerPosition(0.0f, 0.0f, 0.0f);
	const BlockCoord initialBlock{
		static_cast<int32_t>(std::floor(initialPlayerPosition.x)),
		static_cast<int32_t>(std::floor(initialPlayerPosition.y)),
		static_cast<int32_t>(std::floor(initialPlayerPosition.z))
	};
	const ColumnCoord initialColumn = chunk_to_column(block_to_chunk(initialBlock));
	// Initialize meshlet buffers empty; world/mesh scheduling occurs on the streaming thread.
	if (!uploadMeshlets(PendingMeshUpload{})) {
		std::cerr << "Failed to initialize meshlet buffers." << std::endl;
		return false;
	}
	uploadedMeshRevision_ = 0;
	uploadedCenterColumn_ = initialColumn;
	hasUploadedCenterColumn_ = true;
	lastMeshUploadTimeSeconds_ = -1.0;

	voxelPipeline_.emplace(*services_);
	voxelPipeline_->setDrawConfig(meshletManager->getVerticesPerMeshlet(), meshletManager->getMeshletCount());
	if (!voxelPipeline_->build()) {
		std::cerr << "Failed to create voxel pipeline and resources." << std::endl;
		return false;
	}
	boundsDebugPipeline_.emplace(*services_);
	if (!boundsDebugPipeline_->build()) {
		std::cerr << "Failed to create bounds debug pipeline and resources." << std::endl;
		return false;
	}
	uploadedDebugBoundsRevision_ = 0;
	uploadedDebugBoundsLayerMask_ = 0u;

	try {
		startStreamingThread(initialPlayerPosition, initialColumn);
	} catch (const std::exception& ex) {
		std::cerr << "Failed to start streaming thread: " << ex.what() << std::endl;
		return false;
	}

	return true;
}

void WebGPURenderer::createRenderingTextures() {
	if (!voxelPipeline_.has_value()) {
		return;
	}
	if (!voxelPipeline_->createResources()) {
		std::cerr << "Failed to recreate voxel rendering resources." << std::endl;
		return;
	}
	if (!voxelPipeline_->createBindGroup()) {
		std::cerr << "Failed to recreate voxel bind group." << std::endl;
	}
}

void WebGPURenderer::removeRenderingTextures() {
	if (!voxelPipeline_.has_value()) {
		return;
	}
	voxelPipeline_->removeResources();
}

bool WebGPURenderer::resizeSurfaceAndAttachments() {
	int width = 0;
	int height = 0;
	glfwGetFramebufferSize(context->getWindow(), &width, &height);
	if (width <= 0 || height <= 0) {
		return false;
	}

	removeRenderingTextures();
	context->unconfigureSurface();
	if (!context->configureSurface()) {
		return false;
	}
	createRenderingTextures();
	resizePending = false;
	return true;
}

void WebGPURenderer::requestResize() {
	resizePending = true;
}

WebGPUContext* WebGPURenderer::getContext() {
	return context.get();
}

PipelineManager* WebGPURenderer::getPipelineManager() {
	return pipelineManager.get();
}

TextureManager* WebGPURenderer::getTextureManager() {
	return textureManager.get();
}

BufferManager* WebGPURenderer::getBufferManager() {
	return bufferManager.get();
}

bool WebGPURenderer::uploadMeshlets(PendingMeshUpload&& upload) {
	const uint32_t requiredMeshletCapacity = std::max(
		upload.requiredMeshletCapacity,
		std::max(upload.totalMeshletCount + 16u, 64u)
	);
	const uint32_t requiredQuadCapacity = std::max(
		upload.requiredQuadCapacity,
		std::max(
			upload.totalQuadCount + 1024u,
			requiredMeshletCapacity * MESHLET_QUAD_CAPACITY
		)
	);

	const bool requiresRecreate =
		!meshletManager ||
		meshletCapacity_ < requiredMeshletCapacity ||
		quadCapacity_ < requiredQuadCapacity;

	if (requiresRecreate) {
		bufferManager->deleteBuffer(MeshletManager::kMeshMetadataBufferName);
		bufferManager->deleteBuffer(MeshletManager::kMeshDataBufferName);

		auto replacement = std::make_unique<MeshletManager>();
		if (!replacement->initialize(bufferManager.get(), requiredMeshletCapacity, requiredQuadCapacity)) {
			std::cerr << "Failed to create meshlet buffers." << std::endl;
			return false;
		}

		meshletManager = std::move(replacement);
		meshletCapacity_ = requiredMeshletCapacity;
		quadCapacity_ = requiredQuadCapacity;

		if (voxelPipeline_.has_value() && !voxelPipeline_->createBindGroup()) {
			std::cerr << "Failed to recreate voxel bind group after meshlet buffer resize." << std::endl;
			return false;
		}
	}

	meshletManager->adoptPreparedData(std::move(upload.metadata), std::move(upload.quadData));
	if (!meshletManager->upload()) {
		std::cerr << "Failed to upload meshlet buffers." << std::endl;
		return false;
	}

	if (voxelPipeline_.has_value()) {
		voxelPipeline_->setDrawConfig(
			meshletManager->getVerticesPerMeshlet(),
			meshletManager->getMeshletCount()
		);
	}

	return true;
}

glm::vec3 WebGPURenderer::extractCameraPosition(const FrameUniforms& frameUniforms) const {
	return glm::vec3(frameUniforms.inverseViewMatrix[3]);
}

ColumnCoord WebGPURenderer::extractCameraColumn(const FrameUniforms& frameUniforms) const {
	const glm::vec3 cameraPosition = extractCameraPosition(frameUniforms);
	const BlockCoord cameraBlock{
		static_cast<int32_t>(std::floor(cameraPosition.x)),
		static_cast<int32_t>(std::floor(cameraPosition.y)),
		static_cast<int32_t>(std::floor(cameraPosition.z))
	};
	return chunk_to_column(block_to_chunk(cameraBlock));
}

int32_t WebGPURenderer::cameraColumnChebyshevDistance(const ColumnCoord& a, const ColumnCoord& b) const {
	const int32_t dx = std::abs(a.v.x - b.v.x);
	const int32_t dy = std::abs(a.v.y - b.v.y);
	return std::max(dx, dy);
}

void WebGPURenderer::startStreamingThread(const glm::vec3& initialCameraPosition, const ColumnCoord& initialCenterColumn) {
	stopStreamingThread();

	{
		std::lock_guard<std::mutex> lock(streamingMutex_);
		streamingStopRequested_ = false;
		hasLatestStreamingCamera_ = true;
		latestStreamingCamera_ = initialCameraPosition;
		pendingMeshUpload_.reset();
		streamerLastPreparedRevision_ = uploadedMeshRevision_;
		streamerLastPreparedCenter_ = initialCenterColumn;
		streamerHasLastPreparedCenter_ = true;
		streamerLastSnapshotTime_.reset();
	}

	streamingThread_ = std::thread([this] {
		streamingThreadMain();
	});
}

void WebGPURenderer::stopStreamingThread() {
	{
		std::lock_guard<std::mutex> lock(streamingMutex_);
		streamingStopRequested_ = true;
		hasLatestStreamingCamera_ = false;
	}
	streamingCv_.notify_all();

	if (streamingThread_.joinable()) {
		streamingThread_.join();
	}

	{
		std::lock_guard<std::mutex> lock(streamingMutex_);
		streamingStopRequested_ = false;
		pendingMeshUpload_.reset();
		streamerLastSnapshotTime_.reset();
	}
}

void WebGPURenderer::streamingThreadMain() {
	glm::vec3 cameraPosition{0.0f, 0.0f, 0.0f};
	bool hasCameraPosition = false;

	while (true) {
		{
			std::unique_lock<std::mutex> lock(streamingMutex_);
			streamingCv_.wait_for(lock, std::chrono::milliseconds(16), [this] {
				return streamingStopRequested_ || hasLatestStreamingCamera_;
			});
			if (streamingStopRequested_) {
				return;
			}
			if (hasLatestStreamingCamera_) {
				cameraPosition = latestStreamingCamera_;
				hasLatestStreamingCamera_ = false;
				hasCameraPosition = true;
			}
		}

		if (!hasCameraPosition || !world_ || !voxelMeshManager_) {
			continue;
		}

		world_->updatePlayerPosition(cameraPosition);
		voxelMeshManager_->updatePlayerPosition(cameraPosition);

		const BlockCoord cameraBlock{
			static_cast<int32_t>(std::floor(cameraPosition.x)),
			static_cast<int32_t>(std::floor(cameraPosition.y)),
			static_cast<int32_t>(std::floor(cameraPosition.z))
		};
		const ColumnCoord centerColumn = chunk_to_column(block_to_chunk(cameraBlock));
		const bool centerChanged = !streamerHasLastPreparedCenter_ || !(centerColumn == streamerLastPreparedCenter_);
		const int32_t centerShift = streamerHasLastPreparedCenter_
			? cameraColumnChebyshevDistance(centerColumn, streamerLastPreparedCenter_)
			: 0;
		const int32_t centerUploadStrideChunks = std::max(2, uploadColumnRadius_ / 8);

		const uint64_t currentRevision = voxelMeshManager_->meshRevision();
		if (currentRevision == streamerLastPreparedRevision_ &&
			(!centerChanged || centerShift < centerUploadStrideChunks)) {
			continue;
		}

		const bool pendingJobs = world_->hasPendingJobs() || voxelMeshManager_->hasPendingJobs();
		const double minSnapshotIntervalSeconds =
			(uploadColumnRadius_ >= 8) ? 0.35 :
			(uploadColumnRadius_ >= 4) ? 0.25 :
			0.15;
		const auto now = std::chrono::steady_clock::now();
		const bool intervalElapsed =
			!streamerLastSnapshotTime_.has_value() ||
			std::chrono::duration<double>(now - *streamerLastSnapshotTime_).count() >= minSnapshotIntervalSeconds;
		const bool forceForCenterChange = centerChanged && centerShift >= centerUploadStrideChunks;

		if (pendingJobs && !intervalElapsed && !forceForCenterChange) {
			continue;
		}

		std::vector<Meshlet> meshlets = voxelMeshManager_->copyMeshletsAround(centerColumn, uploadColumnRadius_);
		PreparedMeshUploadData prepared = prepareMeshUploadData(meshlets);

		{
			std::lock_guard<std::mutex> lock(streamingMutex_);
			if (streamingStopRequested_) {
				return;
			}
			pendingMeshUpload_ = PendingMeshUpload{
				std::move(prepared.metadata),
				std::move(prepared.quadData),
				prepared.totalMeshletCount,
				prepared.totalQuadCount,
				prepared.requiredMeshletCapacity,
				prepared.requiredQuadCapacity,
				currentRevision,
				centerColumn
			};
		}

		streamerLastPreparedRevision_ = currentRevision;
		streamerLastPreparedCenter_ = centerColumn;
		streamerHasLastPreparedCenter_ = true;
		streamerLastSnapshotTime_ = now;
	}
}

void WebGPURenderer::updateWorldStreaming(const FrameUniforms& frameUniforms) {
	if (!world_ || !voxelMeshManager_) {
		return;
	}

	const glm::vec3 cameraPosition = extractCameraPosition(frameUniforms);

	{
		std::lock_guard<std::mutex> lock(streamingMutex_);
		hasLatestStreamingCamera_ = true;
		latestStreamingCamera_ = cameraPosition;
	}
	streamingCv_.notify_one();

	std::optional<PendingMeshUpload> pendingUpload;
	{
		std::lock_guard<std::mutex> lock(streamingMutex_);
		if (pendingMeshUpload_.has_value()) {
			pendingUpload = std::move(pendingMeshUpload_);
			pendingMeshUpload_.reset();
		}
	}

	if (!pendingUpload.has_value()) {
		return;
	}

	if (uploadMeshlets(std::move(*pendingUpload))) {
		uploadedMeshRevision_ = pendingUpload->meshRevision;
		uploadedCenterColumn_ = pendingUpload->centerColumn;
		hasUploadedCenterColumn_ = true;
		lastMeshUploadTimeSeconds_ = glfwGetTime();
	}
}

void WebGPURenderer::rebuildDebugBounds(uint32_t layerMask) {
	if (!world_ || !boundsDebugPipeline_.has_value()) {
		return;
	}

	std::vector<ColumnCoord> generatedColumns;
	world_->copyGeneratedColumns(generatedColumns);

	std::vector<DebugLineVertex> vertices;
	const size_t chunkBoxCount = generatedColumns.size() * static_cast<size_t>(cfg::COLUMN_HEIGHT);
	const size_t columnBoxCount = generatedColumns.size();
	const size_t regionBoxEstimate = std::max<size_t>(1, generatedColumns.size() / static_cast<size_t>(cfg::REGION_VOLUME_COLUMNS));
	vertices.reserve((chunkBoxCount + columnBoxCount + regionBoxEstimate) * 24ull);

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
		if ((layerMask & kRenderFlagBoundsColumns) != 0u) {
			const glm::vec3 columnMax = columnMin + glm::vec3(
				static_cast<float>(cfg::CHUNK_SIZE),
				static_cast<float>(cfg::CHUNK_SIZE),
				static_cast<float>(cfg::COLUMN_HEIGHT_BLOCKS)
			);
			appendWireBox(vertices, columnMin, columnMax, kColumnBoundsColor);
		}

		visibleRegions.insert(column_to_region(columnCoord));

		if ((layerMask & kRenderFlagBoundsChunks) != 0u) {
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

	if ((layerMask & kRenderFlagBoundsRegions) != 0u) {
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

	if (!boundsDebugPipeline_->updateVertices(vertices)) {
		std::cerr << "Failed to upload debug bounds vertices." << std::endl;
	}
}

void WebGPURenderer::updateDebugBounds(const FrameUniforms& frameUniforms) {
	if (!boundsDebugPipeline_.has_value() || !world_) {
		return;
	}

	const bool enabled = (frameUniforms.renderFlags[0] & kRenderFlagBoundsDebug) != 0u;
	boundsDebugPipeline_->setEnabled(enabled);
	if (!enabled) {
		return;
	}

	const uint64_t worldRevision = world_->generationRevision();
	const uint32_t layerMask = frameUniforms.renderFlags[0] & kRenderFlagBoundsLayerMask;
	if (layerMask == 0u) {
		boundsDebugPipeline_->updateVertices({});
		uploadedDebugBoundsLayerMask_ = 0u;
		uploadedDebugBoundsRevision_ = worldRevision;
		return;
	}

	const bool layersChanged = layerMask != uploadedDebugBoundsLayerMask_;
	if (!layersChanged && worldRevision == uploadedDebugBoundsRevision_) {
		return;
	}

	rebuildDebugBounds(layerMask);
	uploadedDebugBoundsRevision_ = worldRevision;
	uploadedDebugBoundsLayerMask_ = layerMask;
}

void WebGPURenderer::renderFrame(FrameUniforms& uniforms) {
	int fbWidth = 0;
	int fbHeight = 0;
	glfwGetFramebufferSize(context->getWindow(), &fbWidth, &fbHeight);
	if (fbWidth > 0 && fbHeight > 0 &&
		(fbWidth != context->width || fbHeight != context->height)) {
		requestResize();
	}

	if (resizePending) {
		if (!resizeSurfaceAndAttachments()) {
			return;
		}
	}

	updateWorldStreaming(uniforms);
	updateDebugBounds(uniforms);

	auto [surfaceTexture, targetView] = GetNextSurfaceViewData();
	(void)surfaceTexture;
	if (!targetView) return;

	CommandEncoderDescriptor encoderDesc = Default;
	encoderDesc.label = StringView("Frame command encoder");
	CommandEncoder encoder = context->getDevice().createCommandEncoder(encoderDesc);

	// GEOMETRY RENDER PASS
	{
		voxelPipeline_->render(targetView, encoder, [&](RenderPassEncoder& pass) {
			if (boundsDebugPipeline_.has_value()) {
				boundsDebugPipeline_->draw(pass);
			}
			ImGui::Render();
			ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);
		});
	}

	// End frame timing
	//benchmarkManager->endFrame("frame_timer", encoder);

	CommandBufferDescriptor cmdBufferDescriptor = Default;
	cmdBufferDescriptor.label = StringView("Frame command buffer");
	CommandBuffer command = encoder.finish(cmdBufferDescriptor);
	encoder.release();

	context->getQueue().submit(1, &command);
	/*context->getQueue().onSubmittedWorkDone(wgpu::CallbackMode::AllowProcessEvents,
		[&](wgpu::QueueWorkDoneStatus status) {
			if (status == wgpu::QueueWorkDoneStatus::Success) {
				benchmarkManager->processFrameTime("frame_timing");
			}
		});*/

	command.release();

#ifdef WEBGPU_BACKEND_DAWN
	context->getDevice().tick();
#endif

	// Now process timing (this will print frame time by default)
	//benchmarkManager->processFrameTime("frame_timer");

	targetView.release();
	context->getSurface().present();

#ifdef WEBGPU_BACKEND_DAWN
	context->getDevice().tick();
#endif
}

GLFWwindow* WebGPURenderer::getWindow() {
	return context->getWindow();
}

std::pair<SurfaceTexture, TextureView> WebGPURenderer::GetNextSurfaceViewData() {
	SurfaceTexture surfaceTexture;
	context->getSurface().getCurrentTexture(&surfaceTexture);
	Texture texture = surfaceTexture.texture;

	if (surfaceTexture.status == SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
		requestResize();
		if (texture) {
			texture.release();
		}
		return { surfaceTexture, nullptr };
	}

	if (surfaceTexture.status != SurfaceGetCurrentTextureStatus::SuccessOptimal) {
		if (surfaceTexture.status == SurfaceGetCurrentTextureStatus::Outdated ||
			surfaceTexture.status == SurfaceGetCurrentTextureStatus::Lost) {
			requestResize();
		}
		if (texture) {
			texture.release();
		}
		return { surfaceTexture, nullptr };
	}

	TextureViewDescriptor viewDescriptor;
	viewDescriptor.nextInChain = nullptr;
	viewDescriptor.label = StringView("Surface texture view");
	viewDescriptor.format = texture.getFormat();
	viewDescriptor.dimension = TextureViewDimension::_2D;
	viewDescriptor.baseMipLevel = 0;
	viewDescriptor.mipLevelCount = 1;
	viewDescriptor.baseArrayLayer = 0;
	viewDescriptor.arrayLayerCount = 1;
	viewDescriptor.aspect = TextureAspect::All;
	TextureView targetView = texture.createView(viewDescriptor);

#ifndef WEBGPU_BACKEND_WGPU
	// We no longer need the texture, only its view
	// (NB: with wgpu-native, surface textures must be release after the call to wgpuSurfacePresent)
	texture.release();
#endif // WEBGPU_BACKEND_WGPU

	return { surfaceTexture, targetView };
}

void WebGPURenderer::terminate() {
	stopStreamingThread();
	if (boundsDebugPipeline_.has_value()) {
		boundsDebugPipeline_->removeResources();
		boundsDebugPipeline_.reset();
	}
	if (voxelPipeline_.has_value()) {
		voxelPipeline_->removeResources();
		voxelPipeline_.reset();
	}
	voxelMeshManager_.reset();
	world_.reset();
	textureManager->terminate();
	pipelineManager->terminate();
	bufferManager->terminate();
}
