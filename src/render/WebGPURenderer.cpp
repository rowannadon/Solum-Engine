#include <algorithm>
#include <cmath>
#include <iostream>

#include "solum_engine/render/WebGPURenderer.h"

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_wgpu.h>

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
	worldConfig.columnLoadRadius = 16;
	worldConfig.jobConfig.worker_threads = 4;
	world_ = std::make_unique<World>(worldConfig);

	MeshManager::Config meshConfig;
	meshConfig.columnMeshRadius = std::max(1, worldConfig.columnLoadRadius);
	meshConfig.jobConfig.worker_threads = worldConfig.jobConfig.worker_threads;
	voxelMeshManager_ = std::make_unique<MeshManager>(*world_, meshConfig);
	uploadColumnRadius_ = std::max(1, meshConfig.columnMeshRadius + 1);

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

void WebGPURenderer::updateWorldStreaming(const FrameUniforms& frameUniforms) {
	if (!world_ || !voxelMeshManager_) {
		return;
	}

	const glm::vec3 cameraPosition = extractCameraPosition(frameUniforms);
	world_->updatePlayerPosition(cameraPosition);
	voxelMeshManager_->updatePlayerPosition(cameraPosition);
	const ColumnCoord centerColumn = extractCameraColumn(frameUniforms);
	const bool centerChanged = !hasUploadedCenterColumn_ || !(centerColumn == uploadedCenterColumn_);
	const int32_t centerShift = hasUploadedCenterColumn_
		? cameraColumnChebyshevDistance(centerColumn, uploadedCenterColumn_)
		: 0;

	const uint64_t currentRevision = voxelMeshManager_->meshRevision();
	if (!centerChanged && currentRevision == uploadedMeshRevision_) {
		return;
	}

	const bool pendingJobs = world_->hasPendingJobs() || voxelMeshManager_->hasPendingJobs();
	const double now = glfwGetTime();
	const double minUploadIntervalSeconds =
		(uploadColumnRadius_ >= 8) ? 0.35 :
		(uploadColumnRadius_ >= 4) ? 0.25 :
		0.15;
	const bool intervalElapsed =
		lastMeshUploadTimeSeconds_ < 0.0 || (now - lastMeshUploadTimeSeconds_) >= minUploadIntervalSeconds;
	const bool forceForLargeCenterShift = centerShift >= 2;

	// Avoid full-buffer upload every single column-step while generation is in flight.
	// We still force upload when the camera moved far enough to avoid appearing frozen.
	if (pendingJobs && !intervalElapsed && !forceForLargeCenterShift) {
		return;
	}

	const std::vector<Meshlet> meshlets = voxelMeshManager_->copyMeshletsAround(centerColumn, uploadColumnRadius_);
	if (uploadMeshlets(meshlets)) {
		uploadedMeshRevision_ = currentRevision;
		uploadedCenterColumn_ = centerColumn;
		hasUploadedCenterColumn_ = true;
		lastMeshUploadTimeSeconds_ = now;
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
	voxelMeshManager_.reset();
	world_.reset();
	textureManager->terminate();
	pipelineManager->terminate();
	bufferManager->terminate();
}
