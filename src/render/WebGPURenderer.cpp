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

	BufferManager* buf = bufferManager.get();
	TextureManager* tex = textureManager.get();
	PipelineManager* pip = pipelineManager.get();

	{
		BufferDescriptor desc{};
		desc.label = StringView("uniform buffer");
		desc.size = sizeof(FrameUniforms);
		desc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
		desc.mappedAtCreation = false;
		Buffer ubo = bufferManager->createBuffer("uniform_buffer", desc);
		if (!ubo) return false;
	}

	UnpackedBlockMaterial mat;
	mat.id = 1;


	UnpackedBlockMaterial air;
	air.id = 0;

	for (int x = 0; x < CHUNK_SIZE; x++) {
		for (int y = 0; y < CHUNK_SIZE; y++) {
			for (int z = 0; z < CHUNK_SIZE; z++) {
				if (rand() > 31000) {
					chunk.setBlock(ivec3(x, y, z), mat);
				}
				else {
					chunk.setBlock(ivec3(x, y, z), air);
				}
			}
		}
	}

	std::vector<Chunk*> neighbors;
	neighbors.push_back(nullptr);
	neighbors.push_back(nullptr);
	neighbors.push_back(nullptr);
	neighbors.push_back(nullptr);
	neighbors.push_back(nullptr);
	neighbors.push_back(nullptr);

	auto [vertexData, indexData] = mesher.mesh(chunk, neighbors);

	vertexCount = static_cast<uint32_t>(vertexData.size());

	{
		BufferDescriptor desc{};
		desc.label = StringView("vertex buffer");
		desc.size = sizeof(VertexAttributes) * vertexCount;
		desc.usage = BufferUsage::CopyDst | BufferUsage::Vertex;
		desc.mappedAtCreation = false;
		Buffer vbo = bufferManager->createBuffer("vertex_buffer", desc);
		if (!vbo) return false;
	}

	indexCount = static_cast<uint32_t>(indexData.size());

	{
		BufferDescriptor desc{};
		desc.label = StringView("index buffer");
		desc.size = sizeof(uint16_t) * indexCount;
		desc.usage = BufferUsage::CopyDst | BufferUsage::Index;
		desc.mappedAtCreation = false;
		Buffer ibo = bufferManager->createBuffer("index_buffer", desc);
		if (!ibo) return false;
	}

	bufferManager->writeBuffer("vertex_buffer", 0, vertexData.data(), vertexCount * sizeof(VertexAttributes));
	bufferManager->writeBuffer("index_buffer", 0, indexData.data(), indexCount * sizeof(uint16_t));

	voxelPipeline.init(bufferManager.get(), textureManager.get(), pipelineManager.get(), context.get());
	voxelPipeline.createResources();
	voxelPipeline.createPipeline();
	voxelPipeline.createBindGroup();

	return true;
}

void WebGPURenderer::createRenderingTextures() {
	voxelPipeline.createResources();
	voxelPipeline.createBindGroup();
}

void WebGPURenderer::removeRenderingTextures() {
	voxelPipeline.removeResources();
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
	auto [surfaceTexture, targetView] = GetNextSurfaceViewData();
	if (!targetView) return;

	CommandEncoderDescriptor encoderDesc = Default;
	encoderDesc.label = StringView("Frame command encoder");
	CommandEncoder encoder = context->getDevice().createCommandEncoder(encoderDesc);

	// GEOMETRY RENDER PASS
	{
		voxelPipeline.render(targetView, encoder);
	}

	// GUI RENDER PASS
	{
		RenderPassDescriptor renderPassDesc = {};
		RenderPassColorAttachment renderPassColorAttachment = {};
		renderPassColorAttachment.view = targetView;
		//renderPassColorAttachment.resolveTarget = targetView;
		renderPassColorAttachment.loadOp = LoadOp::Load;  // Keep existing terrain rendering
		renderPassColorAttachment.storeOp = StoreOp::Store;
		renderPassColorAttachment.clearValue = Color{ 0.0, 0.0, 0.0, 1.0 };
#ifndef WEBGPU_BACKEND_WGPU
		renderPassColorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

		renderPassDesc.colorAttachmentCount = 1;
		renderPassDesc.colorAttachments = &renderPassColorAttachment;

		renderPassDesc.depthStencilAttachment = nullptr;
		renderPassDesc.timestampWrites = nullptr;

		RenderPassEncoder imguiRenderPass = encoder.beginRenderPass(renderPassDesc);

		// Render ImGUI
		ImGui::Render();
		ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), imguiRenderPass);

		imguiRenderPass.end();
		imguiRenderPass.release();
	}

	// End frame timing
	//benchmarkManager->endFrame("frame_timer", encoder);

	CommandBufferDescriptor cmdBufferDescriptor = {};
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

	if (surfaceTexture.status != SurfaceGetCurrentTextureStatus::SuccessOptimal) {
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


