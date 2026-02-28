#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include "solum_engine/render/WebGPURenderer.h"
#include "solum_engine/voxel/World.h"

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_wgpu.h>

using namespace wgpu;

namespace {
constexpr glm::vec4 kChunkBoundsColor{0.2f, 0.95f, 0.35f, 0.22f};
constexpr glm::vec4 kColumnBoundsColor{1.0f, 0.7f, 0.2f, 0.6f};
constexpr glm::vec4 kRegionBoundsColor{0.2f, 0.8f, 1.0f, 0.95f};
constexpr glm::vec4 kMeshletBoundsColor{1.0f, 0.35f, 0.15f, 0.4f};

std::string loadTextFile(const char* path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		return {};
	}
	std::ostringstream ss;
	ss << file.rdbuf();
	return ss.str();
}

uint32_t computeMipCount(uint32_t width, uint32_t height) {
	uint32_t mipCount = 1u;
	uint32_t w = std::max(width, 1u);
	uint32_t h = std::max(height, 1u);
	while (w > 1u || h > 1u) {
		w = std::max(1u, w / 2u);
		h = std::max(1u, h / 2u);
		++mipCount;
	}
	return mipCount;
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

bool WebGPURenderer::initialize() {
	RenderConfig config;

	context = std::make_unique<WebGPUContext>();
	if (!context->initialize(config)) {
		return false;
	}

	pipelineManager = std::make_unique<PipelineManager>(context->getDevice(), context->getSurfaceFormat());
	bufferManager = std::make_unique<BufferManager>(context->getDevice(), context->getQueue());
	textureManager = std::make_unique<TextureManager>(context->getDevice(), context->getQueue());
	materialManager = std::make_unique<MaterialManager>();

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

	if (!materialManager->initialize(*bufferManager, *textureManager)) {
		std::cerr << "Failed to initialize material manager resources." << std::endl;
		return false;
	}

	if (!meshletBuffers_.initialize(bufferManager.get())) {
		std::cerr << "Failed to initialize meshlet buffers." << std::endl;
		return false;
	}

	voxelPipeline_.emplace(*services_);
	voxelPipeline_->setDrawConfig(meshletBuffers_.verticesPerMeshlet(), meshletBuffers_.meshletCount());
	if (!voxelPipeline_->build()) {
		std::cerr << "Failed to create voxel pipeline and resources." << std::endl;
		return false;
	}
	if (!initializeMeshletOcclusionResources()) {
		std::cerr << "Failed to initialize meshlet occlusion resources." << std::endl;
		return false;
	}
	if (!initializeMeshletCullingResources()) {
		std::cerr << "Failed to initialize meshlet culling resources." << std::endl;
		return false;
	}
	voxelPipeline_->setIndirectDrawBuffer(kMeshletCullIndirectArgsBufferName, 0u);
	boundsDebugPipeline_.emplace(*services_);
	if (!boundsDebugPipeline_->build()) {
		std::cerr << "Failed to create bounds debug pipeline and resources." << std::endl;
		return false;
	}
	uploadedDebugBoundsRevision_ = 0;
	uploadedDebugBoundsMeshRevision_ = 0;
	uploadedDebugBoundsLayerMask_ = 0u;

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
	const bool bindOk = meshletBuffers_.hasMeshletManager()
		? voxelPipeline_->createBindGroupForMeshBuffers(
			meshletBuffers_.activeMeshDataBufferName(),
			meshletBuffers_.activeMeshMetadataBufferName(),
			meshletBuffers_.activeVisibleMeshletIndexBufferName())
		: voxelPipeline_->createBindGroup();
	if (!bindOk) {
		std::cerr << "Failed to recreate voxel bind group." << std::endl;
	}
	if (!createOcclusionDepthResources()) {
		std::cerr << "Failed to recreate meshlet occlusion depth resources." << std::endl;
	}
	if (meshletDepthPrepassPipeline_ && !refreshMeshletOcclusionBindGroup()) {
		std::cerr << "Failed to refresh meshlet depth prepass bind group." << std::endl;
	}
	if (meshletCullPipeline_) {
		updateMeshletCullParams(meshletBuffers_.effectiveMeshletCountForPasses());
	}
	if (meshletCullPipeline_ && !refreshMeshletCullingBindGroup()) {
		std::cerr << "Failed to refresh meshlet culling bind group." << std::endl;
	}
}

void WebGPURenderer::removeRenderingTextures() {
	if (!voxelPipeline_.has_value()) {
		return;
	}
	voxelPipeline_->removeResources();
	removeOcclusionDepthResources();
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

bool WebGPURenderer::createOcclusionDepthResources() {
	if (!textureManager || !context) {
		return false;
	}

	textureManager->removeTextureView(kOcclusionHiZViewName);
	textureManager->removeTexture(kOcclusionHiZTextureName);
	textureManager->removeTextureView(kOcclusionDepthViewName);
	textureManager->removeTexture(kOcclusionDepthTextureName);

	const uint32_t width = std::max(1, context->width / static_cast<int>(kOcclusionDepthDownsample));
	const uint32_t height = std::max(1, context->height / static_cast<int>(kOcclusionDepthDownsample));
	occlusionDepthWidth_ = width;
	occlusionDepthHeight_ = height;
	occlusionHiZMipCount_ = computeMipCount(width, height);

	TextureDescriptor depthDesc = Default;
	depthDesc.label = StringView("meshlet occlusion depth texture");
	depthDesc.dimension = TextureDimension::_2D;
	depthDesc.format = TextureFormat::Depth32Float;
	depthDesc.mipLevelCount = 1;
	depthDesc.sampleCount = 1;
	depthDesc.size = {width, height, 1};
	depthDesc.usage = TextureUsage::RenderAttachment | TextureUsage::TextureBinding;
	depthDesc.viewFormatCount = 0;
	depthDesc.viewFormats = nullptr;
	if (!textureManager->createTexture(kOcclusionDepthTextureName, depthDesc)) {
		return false;
	}

	TextureViewDescriptor depthViewDesc = Default;
	depthViewDesc.aspect = TextureAspect::DepthOnly;
	depthViewDesc.baseArrayLayer = 0;
	depthViewDesc.arrayLayerCount = 1;
	depthViewDesc.baseMipLevel = 0;
	depthViewDesc.mipLevelCount = 1;
	depthViewDesc.dimension = TextureViewDimension::_2D;
	depthViewDesc.format = TextureFormat::Depth32Float;
	if (!textureManager->createTextureView(
		kOcclusionDepthTextureName,
		kOcclusionDepthViewName,
		depthViewDesc
	)) {
		return false;
	}

	TextureDescriptor hizDesc = Default;
	hizDesc.label = StringView("meshlet occlusion hiz texture");
	hizDesc.dimension = TextureDimension::_2D;
	hizDesc.format = TextureFormat::R32Float;
	hizDesc.mipLevelCount = occlusionHiZMipCount_;
	hizDesc.sampleCount = 1;
	hizDesc.size = {width, height, 1};
	hizDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
	hizDesc.viewFormatCount = 0;
	hizDesc.viewFormats = nullptr;
	if (!textureManager->createTexture(kOcclusionHiZTextureName, hizDesc)) {
		return false;
	}

	TextureViewDescriptor hizViewDesc = Default;
	hizViewDesc.aspect = TextureAspect::All;
	hizViewDesc.baseArrayLayer = 0;
	hizViewDesc.arrayLayerCount = 1;
	hizViewDesc.baseMipLevel = 0;
	hizViewDesc.mipLevelCount = occlusionHiZMipCount_;
	hizViewDesc.dimension = TextureViewDimension::_2D;
	hizViewDesc.format = TextureFormat::R32Float;
	return textureManager->createTextureView(
		kOcclusionHiZTextureName,
		kOcclusionHiZViewName,
		hizViewDesc
	) != nullptr;
}

void WebGPURenderer::removeOcclusionDepthResources() {
	if (!textureManager) {
		return;
	}
	textureManager->removeTextureView(kOcclusionHiZViewName);
	textureManager->removeTexture(kOcclusionHiZTextureName);
	textureManager->removeTextureView(kOcclusionDepthViewName);
	textureManager->removeTexture(kOcclusionDepthTextureName);
	occlusionHiZMipCount_ = 1u;
	occlusionDepthWidth_ = 1u;
	occlusionDepthHeight_ = 1u;
}

bool WebGPURenderer::initializeMeshletOcclusionResources() {
	if (!context || !bufferManager) {
		return false;
	}
	if (!createOcclusionDepthResources()) {
		return false;
	}

	std::vector<BindGroupLayoutEntry> prepassLayoutEntries(3, Default);
	prepassLayoutEntries[0].binding = 0;
	prepassLayoutEntries[0].visibility = ShaderStage::Vertex;
	prepassLayoutEntries[0].buffer.type = BufferBindingType::Uniform;
	prepassLayoutEntries[0].buffer.minBindingSize = sizeof(FrameUniforms);

	prepassLayoutEntries[1].binding = 1;
	prepassLayoutEntries[1].visibility = ShaderStage::Vertex;
	prepassLayoutEntries[1].buffer.type = BufferBindingType::ReadOnlyStorage;

	prepassLayoutEntries[2].binding = 2;
	prepassLayoutEntries[2].visibility = ShaderStage::Vertex;
	prepassLayoutEntries[2].buffer.type = BufferBindingType::ReadOnlyStorage;

	BindGroupLayoutDescriptor prepassLayoutDesc = Default;
	prepassLayoutDesc.label = StringView("meshlet depth prepass bgl");
	prepassLayoutDesc.entryCount = static_cast<uint32_t>(prepassLayoutEntries.size());
	prepassLayoutDesc.entries = prepassLayoutEntries.data();
	meshletDepthPrepassBindGroupLayout_ = context->getDevice().createBindGroupLayout(prepassLayoutDesc);
	if (!meshletDepthPrepassBindGroupLayout_) {
		return false;
	}

	const std::string prepassShaderSource = loadTextFile(SHADER_DIR "/meshlet_depth_prepass.wgsl");
	if (prepassShaderSource.empty()) {
		return false;
	}

	ShaderModuleWGSLDescriptor shaderCodeDesc{};
	shaderCodeDesc.chain.sType = SType::ShaderSourceWGSL;
	shaderCodeDesc.code = StringView(prepassShaderSource);

	ShaderModuleDescriptor shaderDesc = Default;
#ifdef WEBGPU_BACKEND_WGPU
	shaderDesc.hintCount = 0;
	shaderDesc.hints = nullptr;
#endif
	shaderDesc.nextInChain = &shaderCodeDesc.chain;
	ShaderModule prepassShader = context->getDevice().createShaderModule(shaderDesc);
	if (!prepassShader) {
		return false;
	}

	PipelineLayoutDescriptor pipelineLayoutDesc = Default;
	pipelineLayoutDesc.bindGroupLayoutCount = 1;
	pipelineLayoutDesc.bindGroupLayouts = reinterpret_cast<WGPUBindGroupLayout*>(&meshletDepthPrepassBindGroupLayout_);
	PipelineLayout pipelineLayout = context->getDevice().createPipelineLayout(pipelineLayoutDesc);
	if (!pipelineLayout) {
		prepassShader.release();
		return false;
	}

	RenderPipelineDescriptor pipelineDesc = Default;
	pipelineDesc.label = StringView("meshlet depth prepass pipeline");
	pipelineDesc.layout = pipelineLayout;
	pipelineDesc.vertex.module = prepassShader;
	pipelineDesc.vertex.entryPoint = StringView("vs_main");
	pipelineDesc.vertex.constantCount = 0;
	pipelineDesc.vertex.constants = nullptr;
	pipelineDesc.vertex.bufferCount = 0;
	pipelineDesc.vertex.buffers = nullptr;

	pipelineDesc.primitive.topology = PrimitiveTopology::TriangleList;
	pipelineDesc.primitive.stripIndexFormat = IndexFormat::Undefined;
	pipelineDesc.primitive.frontFace = FrontFace::CCW;
	pipelineDesc.primitive.cullMode = CullMode::Back;

	pipelineDesc.multisample.count = 1;
	pipelineDesc.multisample.mask = ~0u;
	pipelineDesc.multisample.alphaToCoverageEnabled = false;

	DepthStencilState depthState = Default;
	depthState.depthWriteEnabled = OptionalBool::True;
	depthState.depthCompare = CompareFunction::Less;
	depthState.format = TextureFormat::Depth32Float;
	depthState.stencilReadMask = 0;
	depthState.stencilWriteMask = 0;
	pipelineDesc.depthStencil = &depthState;
	pipelineDesc.fragment = nullptr;

	meshletDepthPrepassPipeline_ = context->getDevice().createRenderPipeline(pipelineDesc);
	pipelineLayout.release();
	prepassShader.release();
	if (!meshletDepthPrepassPipeline_) {
		return false;
	}

	std::vector<BindGroupLayoutEntry> hizSeedLayoutEntries(2, Default);
	hizSeedLayoutEntries[0].binding = 0;
	hizSeedLayoutEntries[0].visibility = ShaderStage::Compute;
	hizSeedLayoutEntries[0].texture.sampleType = TextureSampleType::Depth;
	hizSeedLayoutEntries[0].texture.viewDimension = TextureViewDimension::_2D;
	hizSeedLayoutEntries[1].binding = 1;
	hizSeedLayoutEntries[1].visibility = ShaderStage::Compute;
	hizSeedLayoutEntries[1].storageTexture.access = StorageTextureAccess::WriteOnly;
	hizSeedLayoutEntries[1].storageTexture.format = TextureFormat::R32Float;
	hizSeedLayoutEntries[1].storageTexture.viewDimension = TextureViewDimension::_2D;

	BindGroupLayoutDescriptor hizSeedLayoutDesc = Default;
	hizSeedLayoutDesc.label = StringView("meshlet hiz seed bgl");
	hizSeedLayoutDesc.entryCount = static_cast<uint32_t>(hizSeedLayoutEntries.size());
	hizSeedLayoutDesc.entries = hizSeedLayoutEntries.data();
	meshletHiZSeedBindGroupLayout_ = context->getDevice().createBindGroupLayout(hizSeedLayoutDesc);
	if (!meshletHiZSeedBindGroupLayout_) {
		return false;
	}

	std::vector<BindGroupLayoutEntry> hizDownsampleLayoutEntries(2, Default);
	hizDownsampleLayoutEntries[0].binding = 0;
	hizDownsampleLayoutEntries[0].visibility = ShaderStage::Compute;
	hizDownsampleLayoutEntries[0].texture.sampleType = TextureSampleType::UnfilterableFloat;
	hizDownsampleLayoutEntries[0].texture.viewDimension = TextureViewDimension::_2D;
	hizDownsampleLayoutEntries[1].binding = 1;
	hizDownsampleLayoutEntries[1].visibility = ShaderStage::Compute;
	hizDownsampleLayoutEntries[1].storageTexture.access = StorageTextureAccess::WriteOnly;
	hizDownsampleLayoutEntries[1].storageTexture.format = TextureFormat::R32Float;
	hizDownsampleLayoutEntries[1].storageTexture.viewDimension = TextureViewDimension::_2D;

	BindGroupLayoutDescriptor hizDownsampleLayoutDesc = Default;
	hizDownsampleLayoutDesc.label = StringView("meshlet hiz downsample bgl");
	hizDownsampleLayoutDesc.entryCount = static_cast<uint32_t>(hizDownsampleLayoutEntries.size());
	hizDownsampleLayoutDesc.entries = hizDownsampleLayoutEntries.data();
	meshletHiZDownsampleBindGroupLayout_ = context->getDevice().createBindGroupLayout(hizDownsampleLayoutDesc);
	if (!meshletHiZDownsampleBindGroupLayout_) {
		return false;
	}

	const std::string hizSeedSource = loadTextFile(SHADER_DIR "/meshlet_hiz_seed.wgsl");
	const std::string hizDownsampleSource = loadTextFile(SHADER_DIR "/meshlet_hiz_downsample.wgsl");
	if (hizSeedSource.empty() || hizDownsampleSource.empty()) {
		return false;
	}

	auto createComputePipelineFromSource = [this](const std::string& source,
	                                             BindGroupLayout layoutBgl,
	                                             const char* label) -> ComputePipeline {
		ShaderModuleWGSLDescriptor codeDesc{};
		codeDesc.chain.sType = SType::ShaderSourceWGSL;
		codeDesc.code = StringView(source);

		ShaderModuleDescriptor moduleDesc = Default;
#ifdef WEBGPU_BACKEND_WGPU
		moduleDesc.hintCount = 0;
		moduleDesc.hints = nullptr;
#endif
		moduleDesc.nextInChain = &codeDesc.chain;
		ShaderModule module = context->getDevice().createShaderModule(moduleDesc);
		if (!module) {
			return nullptr;
		}

		PipelineLayoutDescriptor layoutDesc = Default;
		layoutDesc.bindGroupLayoutCount = 1;
		layoutDesc.bindGroupLayouts = reinterpret_cast<WGPUBindGroupLayout*>(&layoutBgl);
		PipelineLayout pipelineLayout = context->getDevice().createPipelineLayout(layoutDesc);
		if (!pipelineLayout) {
			module.release();
			return nullptr;
		}

		ComputePipelineDescriptor computeDesc = Default;
		computeDesc.label = StringView(label);
		computeDesc.layout = pipelineLayout;
		computeDesc.compute.module = module;
		computeDesc.compute.entryPoint = StringView("cs_main");
		ComputePipeline pipeline = context->getDevice().createComputePipeline(computeDesc);

		pipelineLayout.release();
		module.release();
		return pipeline;
	};

	meshletHiZSeedPipeline_ = createComputePipelineFromSource(
		hizSeedSource,
		meshletHiZSeedBindGroupLayout_,
		"meshlet hiz seed pipeline"
	);
	if (!meshletHiZSeedPipeline_) {
		return false;
	}

	meshletHiZDownsamplePipeline_ = createComputePipelineFromSource(
		hizDownsampleSource,
		meshletHiZDownsampleBindGroupLayout_,
		"meshlet hiz downsample pipeline"
	);
	if (!meshletHiZDownsamplePipeline_) {
		return false;
	}

	return refreshMeshletOcclusionBindGroup();
}

bool WebGPURenderer::refreshMeshletOcclusionBindGroup() {
	if (!bufferManager || !meshletBuffers_.hasMeshletManager() || !meshletDepthPrepassBindGroupLayout_) {
		return false;
	}

	Buffer uniformBuffer = bufferManager->getBuffer("uniform_buffer");
	Buffer meshDataBuffer = bufferManager->getBuffer(meshletBuffers_.activeMeshDataBufferName());
	Buffer metadataBuffer = bufferManager->getBuffer(meshletBuffers_.activeMeshMetadataBufferName());
	if (!uniformBuffer || !meshDataBuffer || !metadataBuffer) {
		return false;
	}

	std::vector<BindGroupEntry> entries(3, Default);
	entries[0].binding = 0;
	entries[0].buffer = uniformBuffer;
	entries[0].offset = 0;
	entries[0].size = sizeof(FrameUniforms);

	entries[1].binding = 1;
	entries[1].buffer = meshDataBuffer;
	entries[1].offset = 0;
	entries[1].size = meshDataBuffer.getSize();

	entries[2].binding = 2;
	entries[2].buffer = metadataBuffer;
	entries[2].offset = 0;
	entries[2].size = metadataBuffer.getSize();

	if (meshletDepthPrepassBindGroup_) {
		meshletDepthPrepassBindGroup_.release();
		meshletDepthPrepassBindGroup_ = nullptr;
	}

	BindGroupDescriptor bgDesc = Default;
	bgDesc.label = StringView("meshlet depth prepass bg");
	bgDesc.layout = meshletDepthPrepassBindGroupLayout_;
	bgDesc.entryCount = static_cast<uint32_t>(entries.size());
	bgDesc.entries = entries.data();
	meshletDepthPrepassBindGroup_ = context->getDevice().createBindGroup(bgDesc);
	return meshletDepthPrepassBindGroup_ != nullptr;
}

void WebGPURenderer::encodeMeshletOcclusionDepthPass(CommandEncoder encoder) {
	if (!meshletDepthPrepassPipeline_ || !meshletDepthPrepassBindGroup_ ||
		!meshletBuffers_.hasMeshletManager() || !textureManager) {
		return;
	}

	const uint32_t meshletCount = meshletBuffers_.effectiveMeshletCountForPasses();
	if (meshletCount == 0u) {
		return;
	}

	TextureView occlusionDepthView = textureManager->getTextureView(kOcclusionDepthViewName);
	if (!occlusionDepthView) {
		return;
	}

	RenderPassDepthStencilAttachment depthAttachment = Default;
	depthAttachment.view = occlusionDepthView;
	depthAttachment.depthClearValue = 1.0f;
	depthAttachment.depthLoadOp = LoadOp::Clear;
	depthAttachment.depthStoreOp = StoreOp::Store;
	depthAttachment.depthReadOnly = false;
	depthAttachment.stencilClearValue = 0;
	depthAttachment.stencilLoadOp = LoadOp::Undefined;
	depthAttachment.stencilStoreOp = StoreOp::Undefined;
	depthAttachment.stencilReadOnly = true;

	RenderPassDescriptor passDesc = Default;
	passDesc.colorAttachmentCount = 0;
	passDesc.colorAttachments = nullptr;
	passDesc.depthStencilAttachment = &depthAttachment;
	passDesc.timestampWrites = nullptr;

	RenderPassEncoder pass = encoder.beginRenderPass(passDesc);
	pass.setPipeline(meshletDepthPrepassPipeline_);
	pass.setBindGroup(0, meshletDepthPrepassBindGroup_, 0, nullptr);
	pass.draw(MESHLET_VERTEX_CAPACITY, meshletCount, 0, 0);
	pass.end();
	pass.release();
}

void WebGPURenderer::encodeMeshletOcclusionHierarchyPass(CommandEncoder encoder) {
	if (!context || !textureManager || !meshletHiZSeedPipeline_ || !meshletHiZDownsamplePipeline_ ||
		!meshletHiZSeedBindGroupLayout_ || !meshletHiZDownsampleBindGroupLayout_) {
		return;
	}

	TextureView depthView = textureManager->getTextureView(kOcclusionDepthViewName);
	Texture hizTexture = textureManager->getTexture(kOcclusionHiZTextureName);
	if (!depthView || !hizTexture) {
		return;
	}

	const uint32_t mipCount = std::max(occlusionHiZMipCount_, 1u);
	const Device device = context->getDevice();

	auto createHizMipView = [hizTexture](uint32_t mipLevel) -> TextureView {
		TextureViewDescriptor viewDesc = Default;
		viewDesc.aspect = TextureAspect::All;
		viewDesc.baseArrayLayer = 0;
		viewDesc.arrayLayerCount = 1;
		viewDesc.baseMipLevel = mipLevel;
		viewDesc.mipLevelCount = 1;
		viewDesc.dimension = TextureViewDimension::_2D;
		viewDesc.format = TextureFormat::R32Float;
		return hizTexture.createView(viewDesc);
	};

	ComputePassDescriptor passDesc = Default;
	ComputePassEncoder pass = encoder.beginComputePass(passDesc);

	{
		TextureView dstMip0View = createHizMipView(0u);
		if (dstMip0View) {
			std::array<BindGroupEntry, 2> entries{};
			entries[0] = Default;
			entries[0].binding = 0;
			entries[0].textureView = depthView;
			entries[1] = Default;
			entries[1].binding = 1;
			entries[1].textureView = dstMip0View;

			BindGroupDescriptor bgDesc = Default;
			bgDesc.label = StringView("meshlet hiz seed bg");
			bgDesc.layout = meshletHiZSeedBindGroupLayout_;
			bgDesc.entryCount = static_cast<uint32_t>(entries.size());
			bgDesc.entries = entries.data();
			BindGroup bg = device.createBindGroup(bgDesc);

			if (bg) {
				pass.setPipeline(meshletHiZSeedPipeline_);
				pass.setBindGroup(0, bg, 0, nullptr);
				const uint32_t gx = (std::max(occlusionDepthWidth_, 1u) + kOcclusionHiZWorkgroupSize - 1u) / kOcclusionHiZWorkgroupSize;
				const uint32_t gy = (std::max(occlusionDepthHeight_, 1u) + kOcclusionHiZWorkgroupSize - 1u) / kOcclusionHiZWorkgroupSize;
				pass.dispatchWorkgroups(gx, gy, 1u);
				bg.release();
			}

			dstMip0View.release();
		}
	}

	for (uint32_t mip = 1u; mip < mipCount; ++mip) {
		TextureView srcView = createHizMipView(mip - 1u);
		TextureView dstView = createHizMipView(mip);
		if (!srcView || !dstView) {
			if (srcView) {
				srcView.release();
			}
			if (dstView) {
				dstView.release();
			}
			continue;
		}

		std::array<BindGroupEntry, 2> entries{};
		entries[0] = Default;
		entries[0].binding = 0;
		entries[0].textureView = srcView;
		entries[1] = Default;
		entries[1].binding = 1;
		entries[1].textureView = dstView;

		BindGroupDescriptor bgDesc = Default;
		bgDesc.label = StringView("meshlet hiz downsample bg");
		bgDesc.layout = meshletHiZDownsampleBindGroupLayout_;
		bgDesc.entryCount = static_cast<uint32_t>(entries.size());
		bgDesc.entries = entries.data();
		BindGroup bg = device.createBindGroup(bgDesc);

		if (bg) {
			const uint32_t mipWidth = std::max(1u, occlusionDepthWidth_ >> mip);
			const uint32_t mipHeight = std::max(1u, occlusionDepthHeight_ >> mip);
			const uint32_t gx = (mipWidth + kOcclusionHiZWorkgroupSize - 1u) / kOcclusionHiZWorkgroupSize;
			const uint32_t gy = (mipHeight + kOcclusionHiZWorkgroupSize - 1u) / kOcclusionHiZWorkgroupSize;
			pass.setPipeline(meshletHiZDownsamplePipeline_);
			pass.setBindGroup(0, bg, 0, nullptr);
			pass.dispatchWorkgroups(gx, gy, 1u);
			bg.release();
		}

		srcView.release();
		dstView.release();
	}

	pass.end();
	pass.release();
}

bool WebGPURenderer::initializeMeshletCullingResources() {
	if (!context || !bufferManager) {
		return false;
	}

	{
		BufferDescriptor paramsDesc = Default;
		paramsDesc.label = StringView("meshlet cull params buffer");
		paramsDesc.size = 16u;
		paramsDesc.usage = BufferUsage::Uniform | BufferUsage::CopyDst;
		paramsDesc.mappedAtCreation = false;
		if (!bufferManager->createBuffer(kMeshletCullParamsBufferName, paramsDesc)) {
			return false;
		}
	}

	{
		BufferDescriptor indirectDesc = Default;
		indirectDesc.label = StringView("meshlet cull indirect args buffer");
		indirectDesc.size = sizeof(uint32_t) * 4u;
		indirectDesc.usage = BufferUsage::Storage | BufferUsage::Indirect | BufferUsage::CopyDst;
		indirectDesc.mappedAtCreation = false;
		if (!bufferManager->createBuffer(kMeshletCullIndirectArgsBufferName, indirectDesc)) {
			return false;
		}
	}

	{
		BufferDescriptor resetDesc = Default;
		resetDesc.label = StringView("meshlet cull indirect reset buffer");
		resetDesc.size = sizeof(uint32_t) * 4u;
		resetDesc.usage = BufferUsage::CopySrc | BufferUsage::CopyDst;
		resetDesc.mappedAtCreation = false;
		if (!bufferManager->createBuffer(kMeshletCullIndirectResetBufferName, resetDesc)) {
			return false;
		}

		const uint32_t drawArgsReset[4] = {MESHLET_VERTEX_CAPACITY, 0u, 0u, 0u};
		bufferManager->writeBuffer(
			kMeshletCullIndirectResetBufferName,
			0u,
			drawArgsReset,
			sizeof(drawArgsReset)
		);
	}

	std::vector<BindGroupLayoutEntry> cullLayoutEntries(6, Default);
	cullLayoutEntries[0].binding = 0;
	cullLayoutEntries[0].visibility = ShaderStage::Compute;
	cullLayoutEntries[0].buffer.type = BufferBindingType::Uniform;
	cullLayoutEntries[0].buffer.minBindingSize = sizeof(FrameUniforms);

	cullLayoutEntries[1].binding = 1;
	cullLayoutEntries[1].visibility = ShaderStage::Compute;
	cullLayoutEntries[1].buffer.type = BufferBindingType::ReadOnlyStorage;

	cullLayoutEntries[2].binding = 2;
	cullLayoutEntries[2].visibility = ShaderStage::Compute;
	cullLayoutEntries[2].buffer.type = BufferBindingType::Storage;

	cullLayoutEntries[3].binding = 3;
	cullLayoutEntries[3].visibility = ShaderStage::Compute;
	cullLayoutEntries[3].buffer.type = BufferBindingType::Storage;

	cullLayoutEntries[4].binding = 4;
	cullLayoutEntries[4].visibility = ShaderStage::Compute;
	cullLayoutEntries[4].buffer.type = BufferBindingType::Uniform;
	cullLayoutEntries[4].buffer.minBindingSize = 16u;

	cullLayoutEntries[5].binding = 5;
	cullLayoutEntries[5].visibility = ShaderStage::Compute;
	cullLayoutEntries[5].texture.sampleType = TextureSampleType::UnfilterableFloat;
	cullLayoutEntries[5].texture.viewDimension = TextureViewDimension::_2D;

	BindGroupLayoutDescriptor bglDesc = Default;
	bglDesc.label = StringView("meshlet cull bgl");
	bglDesc.entryCount = static_cast<uint32_t>(cullLayoutEntries.size());
	bglDesc.entries = cullLayoutEntries.data();
	meshletCullBindGroupLayout_ = context->getDevice().createBindGroupLayout(bglDesc);
	if (!meshletCullBindGroupLayout_) {
		return false;
	}

	const std::string cullShaderSource = loadTextFile(SHADER_DIR "/meshlet_cull.wgsl");
	if (cullShaderSource.empty()) {
		return false;
	}

	ShaderModuleWGSLDescriptor shaderCodeDesc{};
	shaderCodeDesc.chain.sType = SType::ShaderSourceWGSL;
	shaderCodeDesc.code = StringView(cullShaderSource);

	ShaderModuleDescriptor shaderDesc = Default;
#ifdef WEBGPU_BACKEND_WGPU
	shaderDesc.hintCount = 0;
	shaderDesc.hints = nullptr;
#endif
	shaderDesc.nextInChain = &shaderCodeDesc.chain;
	ShaderModule cullShader = context->getDevice().createShaderModule(shaderDesc);
	if (!cullShader) {
		return false;
	}

	PipelineLayoutDescriptor layoutDesc = Default;
	layoutDesc.bindGroupLayoutCount = 1;
	layoutDesc.bindGroupLayouts = reinterpret_cast<WGPUBindGroupLayout*>(&meshletCullBindGroupLayout_);
	PipelineLayout layout = context->getDevice().createPipelineLayout(layoutDesc);
	if (!layout) {
		cullShader.release();
		return false;
	}

	ComputePipelineDescriptor pipelineDesc = Default;
	pipelineDesc.label = StringView("meshlet cull pipeline");
	pipelineDesc.layout = layout;
	pipelineDesc.compute.module = cullShader;
	pipelineDesc.compute.entryPoint = StringView("cs_main");
	meshletCullPipeline_ = context->getDevice().createComputePipeline(pipelineDesc);

	layout.release();
	cullShader.release();

	if (!meshletCullPipeline_) {
		return false;
	}

	updateMeshletCullParams(meshletBuffers_.meshletCount());
	return refreshMeshletCullingBindGroup();
}

void WebGPURenderer::updateMeshletCullParams(uint32_t meshletCount) {
	if (!bufferManager) {
		return;
	}
	const uint32_t params[4] = {meshletCount, std::max(occlusionHiZMipCount_, 1u), 0u, 0u};
	bufferManager->writeBuffer(kMeshletCullParamsBufferName, 0u, params, sizeof(params));
}

bool WebGPURenderer::refreshMeshletCullingBindGroup() {
	if (!bufferManager || !meshletBuffers_.hasMeshletManager() || !meshletCullBindGroupLayout_ || !textureManager) {
		return false;
	}

	Buffer uniformBuffer = bufferManager->getBuffer("uniform_buffer");
	Buffer meshletAabbBuffer = bufferManager->getBuffer(meshletBuffers_.activeMeshAabbBufferName());
	Buffer visibleIndicesBuffer = bufferManager->getBuffer(meshletBuffers_.activeVisibleMeshletIndexBufferName());
	Buffer drawArgsBuffer = bufferManager->getBuffer(kMeshletCullIndirectArgsBufferName);
	Buffer cullParamsBuffer = bufferManager->getBuffer(kMeshletCullParamsBufferName);
	TextureView occlusionHiZView = textureManager->getTextureView(kOcclusionHiZViewName);
	if (!uniformBuffer || !meshletAabbBuffer || !visibleIndicesBuffer || !drawArgsBuffer || !cullParamsBuffer || !occlusionHiZView) {
		return false;
	}

	std::vector<BindGroupEntry> entries(6, Default);
	entries[0].binding = 0;
	entries[0].buffer = uniformBuffer;
	entries[0].offset = 0;
	entries[0].size = sizeof(FrameUniforms);

	entries[1].binding = 1;
	entries[1].buffer = meshletAabbBuffer;
	entries[1].offset = 0;
	entries[1].size = meshletAabbBuffer.getSize();

	entries[2].binding = 2;
	entries[2].buffer = visibleIndicesBuffer;
	entries[2].offset = 0;
	entries[2].size = visibleIndicesBuffer.getSize();

	entries[3].binding = 3;
	entries[3].buffer = drawArgsBuffer;
	entries[3].offset = 0;
	entries[3].size = drawArgsBuffer.getSize();

	entries[4].binding = 4;
	entries[4].buffer = cullParamsBuffer;
	entries[4].offset = 0;
	entries[4].size = 16u;

	entries[5].binding = 5;
	entries[5].textureView = occlusionHiZView;

	if (meshletCullBindGroup_) {
		meshletCullBindGroup_.release();
		meshletCullBindGroup_ = nullptr;
	}

	BindGroupDescriptor bgDesc = Default;
	bgDesc.label = StringView("meshlet cull bg");
	bgDesc.layout = meshletCullBindGroupLayout_;
	bgDesc.entryCount = static_cast<uint32_t>(entries.size());
	bgDesc.entries = entries.data();
	meshletCullBindGroup_ = context->getDevice().createBindGroup(bgDesc);
	return meshletCullBindGroup_ != nullptr;
}

void WebGPURenderer::encodeMeshletCullingPass(CommandEncoder encoder) {
	if (!meshletCullPipeline_ || !meshletCullBindGroup_ || !meshletBuffers_.hasMeshletManager()) {
		return;
	}

	Buffer resetBuffer = bufferManager->getBuffer(kMeshletCullIndirectResetBufferName);
	Buffer indirectArgsBuffer = bufferManager->getBuffer(kMeshletCullIndirectArgsBufferName);
	if (!resetBuffer || !indirectArgsBuffer) {
		return;
	}
	encoder.copyBufferToBuffer(
		resetBuffer,
		0u,
		indirectArgsBuffer,
		0u,
		sizeof(uint32_t) * 4u
	);

	const uint32_t meshletCount = meshletBuffers_.effectiveMeshletCountForPasses();
	if (meshletCount == 0u) {
		return;
	}

	ComputePassDescriptor passDesc = Default;
	ComputePassEncoder pass = encoder.beginComputePass(passDesc);
	pass.setPipeline(meshletCullPipeline_);
	pass.setBindGroup(0, meshletCullBindGroup_, 0, nullptr);
	const uint32_t workgroupCount = (meshletCount + kMeshletCullWorkgroupSize - 1u) / kMeshletCullWorkgroupSize;
	pass.dispatchWorkgroups(workgroupCount, 1u, 1u);
	pass.end();
	pass.release();
}

void WebGPURenderer::recordTimingNs(TimingStage stage, uint64_t ns) noexcept {
	const std::size_t stageIndex = static_cast<std::size_t>(stage);
	TimingAccumulator& accumulator = timingAccumulators_[stageIndex];
	accumulator.totalNs.fetch_add(ns, std::memory_order_relaxed);
	accumulator.callCount.fetch_add(1, std::memory_order_relaxed);

	uint64_t observedMax = accumulator.maxNs.load(std::memory_order_relaxed);
	while (ns > observedMax &&
	       !accumulator.maxNs.compare_exchange_weak(
	           observedMax,
	           ns,
	           std::memory_order_relaxed,
	           std::memory_order_relaxed)) {
	}
}

WebGPURenderer::TimingRawTotals WebGPURenderer::captureTimingRawTotals() const {
	TimingRawTotals totals;
	for (std::size_t i = 0; i < static_cast<std::size_t>(TimingStage::Count); ++i) {
		const TimingAccumulator& accumulator = timingAccumulators_[i];
		totals.totalNs[i] = accumulator.totalNs.load(std::memory_order_relaxed);
		totals.callCount[i] = accumulator.callCount.load(std::memory_order_relaxed);
		totals.maxNs[i] = accumulator.maxNs.load(std::memory_order_relaxed);
	}

	totals.mainUploadsApplied = mainUploadsApplied_.load(std::memory_order_relaxed);
	return totals;
}

TimingStageSnapshot WebGPURenderer::makeStageSnapshot(const TimingRawTotals& current,
                                                      const TimingRawTotals& previous,
                                                      TimingStage stage,
                                                      double sampleWindowSeconds) {
	const std::size_t i = static_cast<std::size_t>(stage);
	const uint64_t deltaNs = current.totalNs[i] - previous.totalNs[i];
	const uint64_t deltaCalls = current.callCount[i] - previous.callCount[i];
	const double deltaMs = static_cast<double>(deltaNs) / 1'000'000.0;
	const double window = std::max(sampleWindowSeconds, 1e-6);

	TimingStageSnapshot snapshot;
	snapshot.averageMs = (deltaCalls > 0) ? (deltaMs / static_cast<double>(deltaCalls)) : 0.0;
	snapshot.peakMs = static_cast<double>(current.maxNs[i]) / 1'000'000.0;
	snapshot.totalMsPerSecond = deltaMs / window;
	snapshot.callsPerSecond = static_cast<double>(deltaCalls) / window;
	snapshot.totalCalls = current.callCount[i];
	return snapshot;
}

RuntimeTimingSnapshot WebGPURenderer::getRuntimeTimingSnapshot() {
	RuntimeTimingSnapshot snapshot;
	const TimingRawTotals currentTotals = captureTimingRawTotals();
	const auto now = std::chrono::steady_clock::now();

	{
		std::lock_guard<std::mutex> lock(timingSnapshotMutex_);
		if (!lastTimingSampleTime_.has_value()) {
			lastTimingSampleTime_ = now;
			lastTimingRawTotals_ = currentTotals;
		} else {
			const double sampleWindowSeconds = std::chrono::duration<double>(now - *lastTimingSampleTime_).count();
			snapshot.sampleWindowSeconds = sampleWindowSeconds;
			snapshot.mainUploadMeshlets = makeStageSnapshot(
				currentTotals,
				lastTimingRawTotals_,
				TimingStage::MainUploadMeshlets,
				sampleWindowSeconds
			);
			snapshot.mainUpdateDebugBounds = makeStageSnapshot(
				currentTotals,
				lastTimingRawTotals_,
				TimingStage::MainUpdateDebugBounds,
				sampleWindowSeconds
			);
			snapshot.mainRenderFrameCpu = makeStageSnapshot(
				currentTotals,
				lastTimingRawTotals_,
				TimingStage::MainRenderFrameCpu,
				sampleWindowSeconds
			);
			snapshot.mainAcquireSurface = makeStageSnapshot(
				currentTotals,
				lastTimingRawTotals_,
				TimingStage::MainAcquireSurface,
				sampleWindowSeconds
			);
			snapshot.mainEncodeCommands = makeStageSnapshot(
				currentTotals,
				lastTimingRawTotals_,
				TimingStage::MainEncodeCommands,
				sampleWindowSeconds
			);
			snapshot.mainQueueSubmit = makeStageSnapshot(
				currentTotals,
				lastTimingRawTotals_,
				TimingStage::MainQueueSubmit,
				sampleWindowSeconds
			);
			snapshot.mainPresent = makeStageSnapshot(
				currentTotals,
				lastTimingRawTotals_,
				TimingStage::MainPresent,
				sampleWindowSeconds
			);
			snapshot.mainDeviceTick = makeStageSnapshot(
				currentTotals,
				lastTimingRawTotals_,
				TimingStage::MainDeviceTick,
				sampleWindowSeconds
			);
			snapshot.mainUploadsApplied =
				currentTotals.mainUploadsApplied - lastTimingRawTotals_.mainUploadsApplied;

			lastTimingSampleTime_ = now;
			lastTimingRawTotals_ = currentTotals;
		}
	}

	snapshot.pendingUploadQueued =
		meshletBuffers_.hasPendingOrActiveUpload();
	return snapshot;
}

void WebGPURenderer::setDebugWorld(const World* world) {
	debugWorld_ = world;
}

void WebGPURenderer::queueMeshUpload(StreamingMeshUpload&& upload) {
	meshletBuffers_.queueUpload(std::move(upload));
}

bool WebGPURenderer::isMeshUploadInProgress() const noexcept {
	return meshletBuffers_.isUploadInProgress();
}

uint64_t WebGPURenderer::uploadedMeshRevision() const noexcept {
	return meshletBuffers_.uploadedMeshRevision();
}

void WebGPURenderer::processPendingMeshUploads() {
	if (!meshletBuffers_.hasPendingOrActiveUpload()) {
		return;
	}

	const auto uploadStart = std::chrono::steady_clock::now();
	auto finalizeUploadTiming = [this, &uploadStart]() {
		recordTimingNs(
			TimingStage::MainUploadMeshlets,
			static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::steady_clock::now() - uploadStart
			).count())
		);
	};

	const MeshletBufferController::ProcessResult result = meshletBuffers_.processPendingUpload();

	if (result.buffersRecreated) {
		if (voxelPipeline_.has_value() &&
			!voxelPipeline_->createBindGroupForMeshBuffers(
				meshletBuffers_.activeMeshDataBufferName(),
				meshletBuffers_.activeMeshMetadataBufferName(),
				meshletBuffers_.activeVisibleMeshletIndexBufferName())) {
			std::cerr << "Failed to bind resized meshlet buffers." << std::endl;
			finalizeUploadTiming();
			return;
		}
	}

	if (result.uploadApplied) {
		if (voxelPipeline_.has_value()) {
			if (!voxelPipeline_->createBindGroupForMeshBuffers(
					meshletBuffers_.activeMeshDataBufferName(),
					meshletBuffers_.activeMeshMetadataBufferName(),
					meshletBuffers_.activeVisibleMeshletIndexBufferName())) {
				std::cerr << "Failed to bind active meshlet buffers." << std::endl;
				finalizeUploadTiming();
				return;
			}
			voxelPipeline_->setDrawConfig(
				meshletBuffers_.verticesPerMeshlet(),
				meshletBuffers_.meshletCount()
			);
		}
		mainUploadsApplied_.fetch_add(1, std::memory_order_relaxed);
	}

	if (result.buffersRecreated || result.uploadApplied) {
		if (meshletDepthPrepassBindGroupLayout_ && meshletDepthPrepassPipeline_) {
			if (!refreshMeshletOcclusionBindGroup()) {
				std::cerr << "Failed to bind meshlet depth prepass resources after upload." << std::endl;
				finalizeUploadTiming();
				return;
			}
		}

		if (meshletCullBindGroupLayout_ && meshletCullPipeline_) {
			const uint32_t meshletCount = result.uploadApplied
				? meshletBuffers_.meshletCount()
				: meshletBuffers_.effectiveMeshletCountForPasses();
			updateMeshletCullParams(meshletCount);
			if (!refreshMeshletCullingBindGroup()) {
				std::cerr << "Failed to bind meshlet culling resources after upload." << std::endl;
				finalizeUploadTiming();
				return;
			}
		}
	}

	finalizeUploadTiming();
}

void WebGPURenderer::rebuildDebugBounds(uint32_t layerMask) {
	if (!boundsDebugPipeline_.has_value()) {
		return;
	}

	const bool includeChunkBounds = (layerMask & kRenderFlagBoundsChunks) != 0u;
	const bool includeColumnBounds = (layerMask & kRenderFlagBoundsColumns) != 0u;
	const bool includeRegionBounds = (layerMask & kRenderFlagBoundsRegions) != 0u;
	const bool includeMeshletBounds = (layerMask & kRenderFlagBoundsMeshlets) != 0u;
	const bool includeWorldBounds = includeChunkBounds || includeColumnBounds || includeRegionBounds;

	std::vector<ColumnCoord> generatedColumns;
	if (includeWorldBounds && debugWorld_) {
		debugWorld_->copyGeneratedColumns(generatedColumns);
	}

	std::vector<DebugLineVertex> vertices;
	const size_t chunkBoxCount = includeChunkBounds ? (generatedColumns.size() * static_cast<size_t>(cfg::COLUMN_HEIGHT)) : 0u;
	const size_t columnBoxCount = includeColumnBounds ? generatedColumns.size() : 0u;
	const size_t regionBoxEstimate = includeRegionBounds
		? std::max<size_t>(1, generatedColumns.size() / static_cast<size_t>(cfg::REGION_VOLUME_COLUMNS))
		: 0u;
	const std::vector<MeshletAabb>& activeMeshletBounds = meshletBuffers_.activeMeshletBounds();
	const size_t meshletBoxCount = includeMeshletBounds ? activeMeshletBounds.size() : 0u;
	vertices.reserve((chunkBoxCount + columnBoxCount + regionBoxEstimate + meshletBoxCount) * 24ull);

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
		if (includeColumnBounds) {
			const glm::vec3 columnMax = columnMin + glm::vec3(
				static_cast<float>(cfg::CHUNK_SIZE),
				static_cast<float>(cfg::CHUNK_SIZE),
				static_cast<float>(cfg::COLUMN_HEIGHT_BLOCKS)
			);
			appendWireBox(vertices, columnMin, columnMax, kColumnBoundsColor);
		}

		visibleRegions.insert(column_to_region(columnCoord));

		if (includeChunkBounds) {
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

	if (includeRegionBounds) {
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

	if (includeMeshletBounds) {
		for (const MeshletAabb& bounds : activeMeshletBounds) {
			appendWireBox(vertices, bounds.minCorner, bounds.maxCorner, kMeshletBoundsColor);
		}
	}

	if (!boundsDebugPipeline_->updateVertices(vertices)) {
		std::cerr << "Failed to upload debug bounds vertices." << std::endl;
	}
}

void WebGPURenderer::updateDebugBounds(const FrameUniforms& frameUniforms) {
	if (!boundsDebugPipeline_.has_value()) {
		return;
	}

	const bool enabled = (frameUniforms.renderFlags[0] & kRenderFlagBoundsDebug) != 0u;
	boundsDebugPipeline_->setEnabled(enabled);
	if (!enabled) {
		return;
	}

	const uint64_t worldRevision = debugWorld_ ? debugWorld_->generationRevision() : 0u;
	const uint64_t meshRevision = meshletBuffers_.uploadedMeshRevision();
	const uint32_t layerMask = frameUniforms.renderFlags[0] & kRenderFlagBoundsLayerMask;
	if (layerMask == 0u) {
		boundsDebugPipeline_->updateVertices({});
		uploadedDebugBoundsLayerMask_ = 0u;
		uploadedDebugBoundsRevision_ = worldRevision;
		uploadedDebugBoundsMeshRevision_ = meshRevision;
		return;
	}

	const bool includeWorldBounds = (layerMask & (
		kRenderFlagBoundsChunks |
		kRenderFlagBoundsColumns |
		kRenderFlagBoundsRegions)) != 0u;
	const bool includeMeshletBounds = (layerMask & kRenderFlagBoundsMeshlets) != 0u;

	const bool layersChanged = layerMask != uploadedDebugBoundsLayerMask_;
	const bool worldChanged = includeWorldBounds && (worldRevision != uploadedDebugBoundsRevision_);
	const bool meshChanged = includeMeshletBounds && (meshRevision != uploadedDebugBoundsMeshRevision_);
	if (!layersChanged && !worldChanged && !meshChanged) {
		return;
	}

	rebuildDebugBounds(layerMask);
	uploadedDebugBoundsRevision_ = worldRevision;
	uploadedDebugBoundsMeshRevision_ = meshRevision;
	uploadedDebugBoundsLayerMask_ = layerMask;
}

void WebGPURenderer::renderFrame(FrameUniforms& uniforms) {
	const auto frameCpuStart = std::chrono::steady_clock::now();
	auto finalizeFrameTiming = [this, &frameCpuStart]() {
		recordTimingNs(
			TimingStage::MainRenderFrameCpu,
			static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::steady_clock::now() - frameCpuStart
			).count())
		);
	};

	int fbWidth = 0;
	int fbHeight = 0;
	glfwGetFramebufferSize(context->getWindow(), &fbWidth, &fbHeight);
	if (fbWidth > 0 && fbHeight > 0 &&
		(fbWidth != context->width || fbHeight != context->height)) {
		requestResize();
	}

	if (resizePending) {
		if (!resizeSurfaceAndAttachments()) {
			finalizeFrameTiming();
			return;
		}
	}

	processPendingMeshUploads();

	const auto debugUpdateStart = std::chrono::steady_clock::now();
	updateDebugBounds(uniforms);
	recordTimingNs(
		TimingStage::MainUpdateDebugBounds,
		static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now() - debugUpdateStart
		).count())
	);

	const auto acquireStart = std::chrono::steady_clock::now();
	auto [surfaceTexture, targetView] = GetNextSurfaceViewData();
	recordTimingNs(
		TimingStage::MainAcquireSurface,
		static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now() - acquireStart
		).count())
	);
	(void)surfaceTexture;
	if (!targetView) {
		finalizeFrameTiming();
		return;
	}

	const auto encodeStart = std::chrono::steady_clock::now();
	CommandEncoderDescriptor encoderDesc = Default;
	encoderDesc.label = StringView("Frame command encoder");
	CommandEncoder encoder = context->getDevice().createCommandEncoder(encoderDesc);
	if (uniforms.occlusionParams[0] >= 0.5f) {
		encodeMeshletOcclusionDepthPass(encoder);
		encodeMeshletOcclusionHierarchyPass(encoder);
	}
	encodeMeshletCullingPass(encoder);

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
	recordTimingNs(
		TimingStage::MainEncodeCommands,
		static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now() - encodeStart
		).count())
	);

	const auto submitStart = std::chrono::steady_clock::now();
	context->getQueue().submit(1, &command);
	recordTimingNs(
		TimingStage::MainQueueSubmit,
		static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now() - submitStart
		).count())
	);
	/*context->getQueue().onSubmittedWorkDone(wgpu::CallbackMode::AllowProcessEvents,
		[&](wgpu::QueueWorkDoneStatus status) {
			if (status == wgpu::QueueWorkDoneStatus::Success) {
				benchmarkManager->processFrameTime("frame_timing");
			}
		});*/

	command.release();

#ifdef WEBGPU_BACKEND_DAWN
	{
		const auto tickStart = std::chrono::steady_clock::now();
	context->getDevice().tick();
		recordTimingNs(
			TimingStage::MainDeviceTick,
			static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::steady_clock::now() - tickStart
			).count())
		);
	}
#endif

	// Now process timing (this will print frame time by default)
	//benchmarkManager->processFrameTime("frame_timer");

	targetView.release();
	const auto presentStart = std::chrono::steady_clock::now();
	context->getSurface().present();
	recordTimingNs(
		TimingStage::MainPresent,
		static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now() - presentStart
		).count())
	);

#ifdef WEBGPU_BACKEND_DAWN
	{
		const auto tickStart = std::chrono::steady_clock::now();
	context->getDevice().tick();
		recordTimingNs(
			TimingStage::MainDeviceTick,
			static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::steady_clock::now() - tickStart
			).count())
		);
	}
#endif

	finalizeFrameTiming();
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

	TextureViewDescriptor viewDescriptor = Default;
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
	removeOcclusionDepthResources();
	if (meshletDepthPrepassBindGroup_) {
		meshletDepthPrepassBindGroup_.release();
		meshletDepthPrepassBindGroup_ = nullptr;
	}
	if (meshletDepthPrepassPipeline_) {
		meshletDepthPrepassPipeline_.release();
		meshletDepthPrepassPipeline_ = nullptr;
	}
	if (meshletDepthPrepassBindGroupLayout_) {
		meshletDepthPrepassBindGroupLayout_.release();
		meshletDepthPrepassBindGroupLayout_ = nullptr;
	}
	if (meshletHiZSeedPipeline_) {
		meshletHiZSeedPipeline_.release();
		meshletHiZSeedPipeline_ = nullptr;
	}
	if (meshletHiZDownsamplePipeline_) {
		meshletHiZDownsamplePipeline_.release();
		meshletHiZDownsamplePipeline_ = nullptr;
	}
	if (meshletHiZSeedBindGroupLayout_) {
		meshletHiZSeedBindGroupLayout_.release();
		meshletHiZSeedBindGroupLayout_ = nullptr;
	}
	if (meshletHiZDownsampleBindGroupLayout_) {
		meshletHiZDownsampleBindGroupLayout_.release();
		meshletHiZDownsampleBindGroupLayout_ = nullptr;
	}
	if (meshletCullBindGroup_) {
		meshletCullBindGroup_.release();
		meshletCullBindGroup_ = nullptr;
	}
	if (meshletCullPipeline_) {
		meshletCullPipeline_.release();
		meshletCullPipeline_ = nullptr;
	}
	if (meshletCullBindGroupLayout_) {
		meshletCullBindGroupLayout_.release();
		meshletCullBindGroupLayout_ = nullptr;
	}
	if (boundsDebugPipeline_.has_value()) {
		boundsDebugPipeline_->removeResources();
		boundsDebugPipeline_.reset();
	}
	if (voxelPipeline_.has_value()) {
		voxelPipeline_->removeResources();
		voxelPipeline_.reset();
	}
	meshletBuffers_.resetPendingUploads();
	debugWorld_ = nullptr;
	if (materialManager) {
		materialManager->terminate(*bufferManager, *textureManager);
		materialManager.reset();
	}
	textureManager->terminate();
	pipelineManager->terminate();
	bufferManager->terminate();
}
