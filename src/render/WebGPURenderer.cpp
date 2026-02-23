#include <algorithm>
#include <cmath>
#include <iostream>
#include <exception>

#include "solum_engine/render/WebGPURenderer.h"

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_wgpu.h>

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
	worldConfig.columnLoadRadius = 128;
	worldConfig.jobConfig.worker_threads = 4;

	MeshManager::Config meshConfig;
	meshConfig.lodChunkRadii = {4, 16, 32, 64};
	meshConfig.jobConfig.worker_threads = worldConfig.jobConfig.worker_threads;
	const int32_t maxLodSpanChunks = 1 << (static_cast<int32_t>(meshConfig.lodChunkRadii.size()) - 1);
	const int32_t requiredGenerationRadius = meshConfig.lodChunkRadii.back() + (maxLodSpanChunks - 1);
	worldConfig.columnLoadRadius = std::max(worldConfig.columnLoadRadius, requiredGenerationRadius);

	world_ = std::make_unique<World>(worldConfig);
	voxelMeshManager_ = std::make_unique<MeshManager>(*world_, meshConfig);
	uploadColumnRadius_ = std::max(1, meshConfig.lodChunkRadii.back() + 1);

	const glm::vec3 initialPlayerPosition(0.0f, 0.0f, 0.0f);
	world_->updatePlayerPosition(initialPlayerPosition);
	voxelMeshManager_->updatePlayerPosition(initialPlayerPosition);

	const BlockCoord initialBlock{
		static_cast<int32_t>(std::floor(initialPlayerPosition.x)),
		static_cast<int32_t>(std::floor(initialPlayerPosition.y)),
		static_cast<int32_t>(std::floor(initialPlayerPosition.z))
	};
	const ColumnCoord initialColumn = chunk_to_column(block_to_chunk(initialBlock));
	// Initialize meshlet buffers immediately (possibly empty), then stream in meshlets
	// incrementally as jobs complete.
	if (!uploadMeshlets(voxelMeshManager_->copyMeshletsAround(initialColumn, uploadColumnRadius_))) {
		std::cerr << "Failed to initialize meshlet buffers." << std::endl;
		return false;
	}
	uploadedMeshRevision_ = voxelMeshManager_->meshRevision();
	uploadedCenterColumn_ = initialColumn;
	hasUploadedCenterColumn_ = true;
	lastMeshUploadTimeSeconds_ = -1.0;

	voxelPipeline_.emplace(*services_);
	voxelPipeline_->setDrawConfig(meshletManager->getVerticesPerMeshlet(), meshletManager->getMeshletCount());
	if (!voxelPipeline_->build()) {
		std::cerr << "Failed to create voxel pipeline and resources." << std::endl;
		return false;
	}

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

bool WebGPURenderer::uploadMeshlets(const std::vector<Meshlet>& meshlets) {
	uint32_t totalMeshletCount = static_cast<uint32_t>(meshlets.size());
	uint32_t totalQuadCount = 0;
	for (const Meshlet& meshlet : meshlets) {
		totalQuadCount += meshlet.quadCount;
	}

	const uint32_t requiredMeshletCapacity = std::max(totalMeshletCount + 16u, 64u);
	const uint32_t requiredQuadCapacity = std::max(
		totalQuadCount + 1024u,
		requiredMeshletCapacity * MESHLET_QUAD_CAPACITY
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

	meshletManager->clear();
	meshletManager->registerMeshletGroup(meshlets);
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

		{
			std::lock_guard<std::mutex> lock(streamingMutex_);
			if (streamingStopRequested_) {
				return;
			}
			pendingMeshUpload_ = PendingMeshUpload{
				std::move(meshlets),
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

	if (uploadMeshlets(pendingUpload->meshlets)) {
		uploadedMeshRevision_ = pendingUpload->meshRevision;
		uploadedCenterColumn_ = pendingUpload->centerColumn;
		hasUploadedCenterColumn_ = true;
		lastMeshUploadTimeSeconds_ = glfwGetTime();
	} else {
		std::lock_guard<std::mutex> lock(streamingMutex_);
		if (!pendingMeshUpload_.has_value()) {
			pendingMeshUpload_ = std::move(pendingUpload);
		}
	}
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

	auto [surfaceTexture, targetView] = GetNextSurfaceViewData();
	(void)surfaceTexture;
	if (!targetView) return;

	CommandEncoderDescriptor encoderDesc = Default;
	encoderDesc.label = StringView("Frame command encoder");
	CommandEncoder encoder = context->getDevice().createCommandEncoder(encoderDesc);

	// GEOMETRY RENDER PASS
	{
		voxelPipeline_->render(targetView, encoder, [&](RenderPassEncoder& pass) {
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
	voxelMeshManager_.reset();
	world_.reset();
	textureManager->terminate();
	pipelineManager->terminate();
	bufferManager->terminate();
}
