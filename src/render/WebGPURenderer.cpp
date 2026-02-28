#include "solum_engine/render/WebGPURenderer.h"

#include <chrono>
#include <iostream>

#include <imgui/backends/imgui_impl_wgpu.h>
#include <imgui/imgui.h>

using namespace wgpu;

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
        if (!ubo) {
            return false;
        }
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

    meshletOcclusionPipeline_.emplace(*services_);
    if (!meshletOcclusionPipeline_->build(meshletBuffers_)) {
        std::cerr << "Failed to initialize meshlet occlusion resources." << std::endl;
        return false;
    }

    meshletCullingPipeline_.emplace(*services_);
    if (!meshletCullingPipeline_->build(
            meshletBuffers_,
            meshletOcclusionPipeline_->hizMipCount(),
            MeshletOcclusionPipeline::kOcclusionHiZViewName)) {
        std::cerr << "Failed to initialize meshlet culling resources." << std::endl;
        return false;
    }

    voxelPipeline_->setIndirectDrawBuffer(MeshletCullingPipeline::kIndirectArgsBufferName, 0u);

    boundsDebugPipeline_.emplace(*services_);
    if (!boundsDebugPipeline_->build()) {
        std::cerr << "Failed to create bounds debug pipeline and resources." << std::endl;
        return false;
    }

    debugBoundsManager_.reset();
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

    if (meshletOcclusionPipeline_.has_value() &&
        !meshletOcclusionPipeline_->recreateResources(meshletBuffers_)) {
        std::cerr << "Failed to recreate meshlet occlusion depth resources." << std::endl;
    }

    if (meshletCullingPipeline_.has_value()) {
        const uint32_t hizMipCount = meshletOcclusionPipeline_.has_value()
            ? meshletOcclusionPipeline_->hizMipCount()
            : 1u;
        meshletCullingPipeline_->updateCullParams(
            meshletBuffers_.effectiveMeshletCountForPasses(),
            hizMipCount
        );

        const char* hizViewName = meshletOcclusionPipeline_.has_value()
            ? MeshletOcclusionPipeline::kOcclusionHiZViewName
            : nullptr;
        if (!meshletCullingPipeline_->refreshBindGroup(meshletBuffers_, hizViewName)) {
            std::cerr << "Failed to refresh meshlet culling bind group." << std::endl;
        }
    }
}

void WebGPURenderer::removeRenderingTextures() {
    if (voxelPipeline_.has_value()) {
        voxelPipeline_->removeResources();
    }
    if (meshletOcclusionPipeline_.has_value()) {
        meshletOcclusionPipeline_->removeResources();
    }
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

RuntimeTimingSnapshot WebGPURenderer::getRuntimeTimingSnapshot() {
    return timingTracker_.snapshot(meshletBuffers_.hasPendingOrActiveUpload());
}

void WebGPURenderer::setDebugWorld(const World* world) {
    debugBoundsManager_.setWorld(world);
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
        timingTracker_.record(
            MainTimingStage::UploadMeshlets,
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
        timingTracker_.incrementMainUploadsApplied();
    }

    if (result.buffersRecreated || result.uploadApplied) {
        if (meshletOcclusionPipeline_.has_value() &&
            !meshletOcclusionPipeline_->refreshMeshBindGroup(meshletBuffers_)) {
            std::cerr << "Failed to bind meshlet depth prepass resources after upload." << std::endl;
            finalizeUploadTiming();
            return;
        }

        if (meshletCullingPipeline_.has_value()) {
            const uint32_t meshletCount = result.uploadApplied
                ? meshletBuffers_.meshletCount()
                : meshletBuffers_.effectiveMeshletCountForPasses();
            const uint32_t hizMipCount = meshletOcclusionPipeline_.has_value()
                ? meshletOcclusionPipeline_->hizMipCount()
                : 1u;

            meshletCullingPipeline_->updateCullParams(meshletCount, hizMipCount);
            if (!meshletCullingPipeline_->refreshBindGroup(
                    meshletBuffers_,
                    MeshletOcclusionPipeline::kOcclusionHiZViewName)) {
                std::cerr << "Failed to bind meshlet culling resources after upload." << std::endl;
                finalizeUploadTiming();
                return;
            }
        }
    }

    finalizeUploadTiming();
}

void WebGPURenderer::renderFrame(FrameUniforms& uniforms) {
    const auto frameCpuStart = std::chrono::steady_clock::now();
    auto finalizeFrameTiming = [this, &frameCpuStart]() {
        timingTracker_.record(
            MainTimingStage::RenderFrameCpu,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - frameCpuStart
            ).count())
        );
    };

    while (framesInFlight_.load(std::memory_order_acquire) >= kMaxFramesInFlight) {
        context->instance.processEvents();
    }

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
    if (boundsDebugPipeline_.has_value()) {
        debugBoundsManager_.update(uniforms, *boundsDebugPipeline_, meshletBuffers_);
    }
    timingTracker_.record(
        MainTimingStage::UpdateDebugBounds,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - debugUpdateStart
        ).count())
    );

    const auto acquireStart = std::chrono::steady_clock::now();
    auto [surfaceTexture, targetView] = GetNextSurfaceViewData();
    timingTracker_.record(
        MainTimingStage::AcquireSurface,
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

    if (uniforms.occlusionParams[0] >= 0.5f && meshletOcclusionPipeline_.has_value()) {
        meshletOcclusionPipeline_->encodeDepthPrepass(encoder, meshletBuffers_);
        meshletOcclusionPipeline_->encodeHierarchyPass(encoder);
    }

    if (meshletCullingPipeline_.has_value()) {
        meshletCullingPipeline_->encode(encoder, meshletBuffers_);
    }

    if (voxelPipeline_.has_value()) {
        voxelPipeline_->render(targetView, encoder, [&](RenderPassEncoder& pass) {
            if (boundsDebugPipeline_.has_value()) {
                boundsDebugPipeline_->draw(pass);
            }
            ImGui::Render();
            ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);
        });
    }

    CommandBufferDescriptor cmdBufferDescriptor = Default;
    cmdBufferDescriptor.label = StringView("Frame command buffer");
    CommandBuffer command = encoder.finish(cmdBufferDescriptor);
    encoder.release();
    timingTracker_.record(
        MainTimingStage::EncodeCommands,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - encodeStart
        ).count())
    );

    const auto submitStart = std::chrono::steady_clock::now();
    context->getQueue().submit(1, &command);
    timingTracker_.record(
        MainTimingStage::QueueSubmit,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - submitStart
        ).count())
    );

    framesInFlight_.fetch_add(1, std::memory_order_release);
    context->getQueue().onSubmittedWorkDone(
        wgpu::CallbackMode::AllowProcessEvents,
        [this](wgpu::QueueWorkDoneStatus) {
            framesInFlight_.fetch_sub(1, std::memory_order_release);
        }
    );

    command.release();

#ifdef WEBGPU_BACKEND_DAWN
    {
        const auto tickStart = std::chrono::steady_clock::now();
        context->getDevice().tick();
        timingTracker_.record(
            MainTimingStage::DeviceTick,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - tickStart
            ).count())
        );
    }
#endif

    targetView.release();
    const auto presentStart = std::chrono::steady_clock::now();
    context->getSurface().present();
    timingTracker_.record(
        MainTimingStage::Present,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - presentStart
        ).count())
    );

#ifdef WEBGPU_BACKEND_DAWN
    {
        const auto tickStart = std::chrono::steady_clock::now();
        context->getDevice().tick();
        timingTracker_.record(
            MainTimingStage::DeviceTick,
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
        return {surfaceTexture, nullptr};
    }

    if (surfaceTexture.status != SurfaceGetCurrentTextureStatus::SuccessOptimal) {
        if (surfaceTexture.status == SurfaceGetCurrentTextureStatus::Outdated ||
            surfaceTexture.status == SurfaceGetCurrentTextureStatus::Lost) {
            requestResize();
        }
        if (texture) {
            texture.release();
        }
        return {surfaceTexture, nullptr};
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
    texture.release();
#endif

    return {surfaceTexture, targetView};
}

void WebGPURenderer::terminate() {
    if (boundsDebugPipeline_.has_value()) {
        boundsDebugPipeline_->removeResources();
        boundsDebugPipeline_.reset();
    }

    if (meshletCullingPipeline_.has_value()) {
        meshletCullingPipeline_->removeResources();
        meshletCullingPipeline_.reset();
    }

    if (meshletOcclusionPipeline_.has_value()) {
        meshletOcclusionPipeline_->removeResources();
        meshletOcclusionPipeline_.reset();
    }

    if (voxelPipeline_.has_value()) {
        voxelPipeline_->removeResources();
        voxelPipeline_.reset();
    }

    meshletBuffers_.resetPendingUploads();
    debugBoundsManager_.setWorld(nullptr);
    debugBoundsManager_.reset();

    if (materialManager && bufferManager && textureManager) {
        materialManager->terminate(*bufferManager, *textureManager);
    }
    materialManager.reset();

    if (textureManager) {
        textureManager->terminate();
    }
    if (pipelineManager) {
        pipelineManager->terminate();
    }
    if (bufferManager) {
        bufferManager->terminate();
    }
}
