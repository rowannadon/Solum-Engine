#include <algorithm>
#include <random>
#include <thread>
#include <chrono>
#include <iostream>
#include "solum_engine/render/WebGPURenderer.h"

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

	RegionManager rm;
	RegionCoord coord = RegionCoord(glm::ivec2(0));
	rm.addRegion(coord);

	Region* region = rm.getRegion(coord);

	TerrainGenerator gen;
	
	ChunkMesher mesher;

	std::vector<const Chunk*> neighbors(6, nullptr);

	std::vector<Meshlet> totalMeshlets;
	for (int x = 0; x < 8; x++) {
		for (int y = 0; y < 8; y++) {
			Column& col = region->getColumn(x, y);
			gen.generateColumn(glm::ivec3(x * cfg::CHUNK_SIZE, y * cfg::CHUNK_SIZE, 0), col);

			for (int chunk_z = 0; chunk_z < cfg::COLUMN_HEIGHT; chunk_z++) {
				const ChunkCoord chunkCoord = column_local_to_chunk(ColumnCoord(x, y), chunk_z);
				std::vector<Meshlet> chunkMeshlets = mesher.mesh(col.getChunk(static_cast<uint8_t>(chunk_z)), chunkCoord, neighbors);
				totalMeshlets.insert(totalMeshlets.end(), chunkMeshlets.begin(), chunkMeshlets.end());
			}
		}
	}
	

	uint32_t totalMeshletCount = static_cast<uint32_t>(totalMeshlets.size());
	uint32_t totalQuadCount = 0;
	for (const Meshlet& meshlet : totalMeshlets) {
		totalQuadCount += meshlet.quadCount;
	}

	const uint32_t meshletCapacity = std::max(totalMeshletCount + 16u, 64u);
	const uint32_t quadCapacity = std::max(totalQuadCount + 1024u, meshletCapacity * MESHLET_QUAD_CAPACITY);

	if (!meshletManager->initialize(bufferManager.get(), meshletCapacity, quadCapacity)) {
		std::cerr << "Failed to create meshlet buffers." << std::endl;
		return false;
	}

	meshletManager->clear();
	meshletManager->registerMeshletGroup(totalMeshlets);
	if (!meshletManager->upload()) {
		std::cerr << "Failed to upload meshlet buffers." << std::endl;
		return false;
	}

	voxelPipeline_.emplace(*services_);
	voxelPipeline_->setDrawConfig(meshletManager->getVerticesPerMeshlet(), meshletManager->getMeshletCount());
	if (!voxelPipeline_->build()) {
		std::cerr << "Failed to create voxel pipeline and resources." << std::endl;
		return false;
	}

	return true;
}

void WebGPURenderer::createRenderingTextures() {
	if (!voxelPipeline_->createResources()) {
		std::cerr << "Failed to recreate voxel rendering resources." << std::endl;
		return;
	}
	if (!voxelPipeline_->createBindGroup()) {
		std::cerr << "Failed to recreate voxel bind group." << std::endl;
	}
}

void WebGPURenderer::removeRenderingTextures() {
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

	auto [surfaceTexture, targetView] = GetNextSurfaceViewData();
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
	textureManager->terminate();
	pipelineManager->terminate();
	bufferManager->terminate();
}
