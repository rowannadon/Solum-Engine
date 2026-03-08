#include "solum_engine/render/WebGPURenderer.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

#include <imgui/backends/imgui_impl_wgpu.h>
#include <imgui/imgui.h>

using namespace wgpu;

bool WebGPURenderer::initialize() {
    return initialize(Config{});
}

bool WebGPURenderer::initialize(const Config& config) {
    (void)config;
    RenderConfig renderConfig;

    context = std::make_unique<WebGPUContext>();
    if (!context->initialize(renderConfig)) {
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

    if (!culledMeshletBuffers_.initialize(bufferManager.get())) {
        std::cerr << "Failed to initialize culled meshlet buffers." << std::endl;
        return false;
    }
    if (!doubleSidedMeshletBuffers_.initialize(bufferManager.get())) {
        std::cerr << "Failed to initialize double-sided meshlet buffers." << std::endl;
        return false;
    }

    culledVoxelPipeline_.emplace(*services_, VoxelPipeline::Config{
        "culled",
        CullMode::Back,
        true
    });
    culledVoxelPipeline_->setDrawConfig(
        culledMeshletBuffers_.verticesPerMeshlet(),
        culledMeshletBuffers_.meshletCount()
    );
    if (!culledVoxelPipeline_->build()) {
        std::cerr << "Failed to create culled voxel pipeline and resources." << std::endl;
        return false;
    }

    doubleSidedVoxelPipeline_.emplace(*services_, VoxelPipeline::Config{
        "double_sided",
        CullMode::None,
        false
    });
    doubleSidedVoxelPipeline_->setDrawConfig(
        doubleSidedMeshletBuffers_.verticesPerMeshlet(),
        doubleSidedMeshletBuffers_.meshletCount()
    );
    if (!doubleSidedVoxelPipeline_->build()) {
        std::cerr << "Failed to create double-sided voxel pipeline resources." << std::endl;
        return false;
    }

    meshletOcclusionPipeline_.emplace(*services_);
    if (!meshletOcclusionPipeline_->build(culledMeshletBuffers_)) {
        std::cerr << "Failed to initialize meshlet occlusion resources." << std::endl;
        return false;
    }

    culledMeshletCullingPipeline_.emplace(*services_, MeshletCullingPipeline::Config{"culled"});
    if (!culledMeshletCullingPipeline_->build(
            culledMeshletBuffers_,
            meshletOcclusionPipeline_->hizMipCount(),
            MeshletOcclusionPipeline::kOcclusionHiZViewName)) {
        std::cerr << "Failed to initialize culled meshlet culling resources." << std::endl;
        return false;
    }

    doubleSidedMeshletCullingPipeline_.emplace(*services_, MeshletCullingPipeline::Config{"double_sided"});
    if (!doubleSidedMeshletCullingPipeline_->build(
            doubleSidedMeshletBuffers_,
            meshletOcclusionPipeline_->hizMipCount(),
            MeshletOcclusionPipeline::kOcclusionHiZViewName)) {
        std::cerr << "Failed to initialize double-sided meshlet culling resources." << std::endl;
        return false;
    }

    culledVoxelPipeline_->setIndirectDrawBuffer(culledMeshletCullingPipeline_->indirectArgsBufferName(), 0u);
    doubleSidedVoxelPipeline_->setIndirectDrawBuffer(doubleSidedMeshletCullingPipeline_->indirectArgsBufferName(), 0u);

    if (!refreshMeshBindings(false, true)) {
        std::cerr << "Failed to refresh mesh bindings during renderer initialization." << std::endl;
        return false;
    }

    boundsDebugPipeline_.emplace(*services_);
    if (!boundsDebugPipeline_->build()) {
        std::cerr << "Failed to create bounds debug pipeline and resources." << std::endl;
        return false;
    }

    debugBoundsManager_.reset();
    return true;
}

bool WebGPURenderer::refreshMeshBindings(bool uploadApplied, bool rebuildDrawConfig) {
    (void)uploadApplied;
    if (!culledVoxelPipeline_.has_value() || !doubleSidedVoxelPipeline_.has_value()) {
        return false;
    }

    const MeshletBufferController::ActiveBindings culledBindings = culledMeshletBuffers_.activeBindings();
    if (!culledVoxelPipeline_->createBindGroupForMeshBuffers(
            culledBindings.meshDataBufferName,
            culledBindings.meshMetadataBufferName,
            culledBindings.visibleMeshletIndexBufferName)) {
        return false;
    }

    const MeshletBufferController::ActiveBindings doubleSidedBindings = doubleSidedMeshletBuffers_.activeBindings();
    if (!doubleSidedVoxelPipeline_->createBindGroupForMeshBuffers(
            doubleSidedBindings.meshDataBufferName,
            doubleSidedBindings.meshMetadataBufferName,
            doubleSidedBindings.visibleMeshletIndexBufferName)) {
        return false;
    }

    if (rebuildDrawConfig) {
        culledVoxelPipeline_->setDrawConfig(culledBindings.verticesPerMeshlet, culledBindings.meshletCount);
        doubleSidedVoxelPipeline_->setDrawConfig(
            doubleSidedBindings.verticesPerMeshlet,
            doubleSidedBindings.meshletCount
        );
    }

    if (meshletOcclusionPipeline_.has_value() &&
        !meshletOcclusionPipeline_->refreshMeshBindGroup(culledMeshletBuffers_)) {
        return false;
    }

    if (culledMeshletCullingPipeline_.has_value()) {
        const uint32_t hizMipCount = meshletOcclusionPipeline_.has_value()
            ? meshletOcclusionPipeline_->hizMipCount()
            : 1u;
        culledMeshletCullingPipeline_->updateCullParams(
            culledBindings.meshletCount,
            hizMipCount,
            culledBindings.activeRangeCount
        );

        const char* hizViewName = meshletOcclusionPipeline_.has_value()
            ? MeshletOcclusionPipeline::kOcclusionHiZViewName
            : nullptr;
        if (!culledMeshletCullingPipeline_->refreshBindGroup(culledMeshletBuffers_, hizViewName)) {
            return false;
        }
    }

    if (doubleSidedMeshletCullingPipeline_.has_value()) {
        const uint32_t hizMipCount = meshletOcclusionPipeline_.has_value()
            ? meshletOcclusionPipeline_->hizMipCount()
            : 1u;
        doubleSidedMeshletCullingPipeline_->updateCullParams(
            doubleSidedBindings.meshletCount,
            hizMipCount,
            doubleSidedBindings.activeRangeCount
        );

        const char* hizViewName = meshletOcclusionPipeline_.has_value()
            ? MeshletOcclusionPipeline::kOcclusionHiZViewName
            : nullptr;
        if (!doubleSidedMeshletCullingPipeline_->refreshBindGroup(doubleSidedMeshletBuffers_, hizViewName)) {
            return false;
        }
    }

    return true;
}

void WebGPURenderer::createRenderingTextures() {
    if (!culledVoxelPipeline_.has_value()) {
        return;
    }

    if (!culledVoxelPipeline_->createResources()) {
        std::cerr << "Failed to recreate voxel rendering resources." << std::endl;
        return;
    }

    if (meshletOcclusionPipeline_.has_value() &&
        !meshletOcclusionPipeline_->recreateResources(culledMeshletBuffers_)) {
        std::cerr << "Failed to recreate meshlet occlusion depth resources." << std::endl;
    }

    if (!refreshMeshBindings(false, true)) {
        std::cerr << "Failed to refresh mesh bindings after resize/resource recreation." << std::endl;
    }
}

void WebGPURenderer::removeRenderingTextures() {
    if (doubleSidedVoxelPipeline_.has_value()) {
        doubleSidedVoxelPipeline_->removeResources();
    }
    if (culledVoxelPipeline_.has_value()) {
        culledVoxelPipeline_->removeResources();
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

BufferManager* WebGPURenderer::getBufferManager() {
    return bufferManager.get();
}

std::shared_ptr<const BlockModelLibrary> WebGPURenderer::getBlockModelLibrary() const {
    if (!materialManager) {
        return {};
    }
    return materialManager->blockModelLibrary();
}

RuntimeTimingSnapshot WebGPURenderer::getRuntimeTimingSnapshot() {
    return timingTracker_.snapshot(pendingMeshDelta_.has_value());
}

void WebGPURenderer::setDebugWorld(const World* world) {
    debugBoundsManager_.setWorld(world);
}

void WebGPURenderer::queueMeshDelta(MeshStreamingDelta&& delta) {
    pendingMeshDelta_ = std::move(delta);
}

uint64_t WebGPURenderer::uploadedMeshRevision() const noexcept {
    return std::max(
        culledMeshletBuffers_.uploadedMeshRevision(),
        doubleSidedMeshletBuffers_.uploadedMeshRevision()
    );
}

void WebGPURenderer::processPendingMeshUploads() {
    if (!pendingMeshDelta_.has_value()) {
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

    bufferManager->resetFrameBudget();

    const MeshletBufferController::ApplyResult culledResult =
        culledMeshletBuffers_.applyDelta(*pendingMeshDelta_);
    const MeshletBufferController::ApplyResult doubleSidedResult =
        doubleSidedMeshletBuffers_.applyDelta(*pendingMeshDelta_);
    pendingMeshDelta_.reset();

    const bool buffersRecreated = culledResult.buffersRecreated || doubleSidedResult.buffersRecreated;
    const bool deltaApplied = culledResult.deltaApplied || doubleSidedResult.deltaApplied;
    if (buffersRecreated || deltaApplied) {
        if (!refreshMeshBindings(deltaApplied, true)) {
            std::cerr << "Failed to refresh mesh pipeline resources after upload." << std::endl;
            finalizeUploadTiming();
            return;
        }
    }

    if (deltaApplied) {
        timingTracker_.incrementMainUploadsApplied();
    }

    finalizeUploadTiming();
}

bool WebGPURenderer::checkGpuStall(const char* stage,
                                   std::chrono::steady_clock::time_point start,
                                   std::chrono::milliseconds threshold) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    if (elapsed > threshold) {
        std::cerr << "GPU stall detected in " << stage << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
                  << "ms (threshold " << threshold.count() << "ms). Requesting shutdown."
                  << std::endl;
        gpuStallDetected_ = true;
        glfwSetWindowShouldClose(context->getWindow(), GLFW_TRUE);
        return true;
    }
    return false;
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

    if (gpuStallDetected_ || context->isDeviceLost()) {
        finalizeFrameTiming();
        return;
    }

    // Wait for GPU to catch up, but with a timeout to avoid hanging the system.
    {
        const auto waitStart = std::chrono::steady_clock::now();
        while (framesInFlight_.load(std::memory_order_acquire) >= kMaxFramesInFlight) {
            context->instance.processEvents();
            if (checkGpuStall("framesInFlight wait", waitStart, std::chrono::milliseconds(2000))) {
                finalizeFrameTiming();
                return;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
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
        debugBoundsManager_.update(uniforms, *boundsDebugPipeline_, culledMeshletBuffers_);
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
        meshletOcclusionPipeline_->encodeDepthPrepass(encoder, culledMeshletBuffers_);
        meshletOcclusionPipeline_->encodeHierarchyPass(encoder);
    }

    if (culledMeshletCullingPipeline_.has_value()) {
        culledMeshletCullingPipeline_->encode(encoder, culledMeshletBuffers_);
    }
    if (doubleSidedMeshletCullingPipeline_.has_value()) {
        doubleSidedMeshletCullingPipeline_->encode(encoder, doubleSidedMeshletBuffers_);
    }

    const bool hasDoubleSidedMeshlets = doubleSidedMeshletBuffers_.activeSelectionMeshletCount() > 0u;
    bool renderedBasePass = false;
    if (culledVoxelPipeline_.has_value()) {
        renderedBasePass = true;
        if (!hasDoubleSidedMeshlets || !doubleSidedVoxelPipeline_.has_value()) {
            culledVoxelPipeline_->render(
                targetView,
                encoder,
                VoxelPipeline::RenderOptions{true, true},
                [&](RenderPassEncoder& pass) {
                    if (boundsDebugPipeline_.has_value()) {
                        boundsDebugPipeline_->draw(pass);
                    }
                    ImGui::Render();
                    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);
                }
            );
        } else {
            culledVoxelPipeline_->render(targetView, encoder, VoxelPipeline::RenderOptions{true, true});
        }
    }

    if (hasDoubleSidedMeshlets && doubleSidedVoxelPipeline_.has_value()) {
        doubleSidedVoxelPipeline_->render(
            targetView,
            encoder,
            VoxelPipeline::RenderOptions{!renderedBasePass, !renderedBasePass},
            [&](RenderPassEncoder& pass) {
                if (boundsDebugPipeline_.has_value()) {
                    boundsDebugPipeline_->draw(pass);
                }
                ImGui::Render();
                ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);
            }
        );
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
        const auto tickElapsed = std::chrono::steady_clock::now() - tickStart;
        timingTracker_.record(
            MainTimingStage::DeviceTick,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(tickElapsed).count())
        );
        if (checkGpuStall("device.tick (pre-present)", tickStart, std::chrono::milliseconds(1500))) {
            targetView.release();
            finalizeFrameTiming();
            return;
        }
    }
#endif

    targetView.release();
    const auto presentStart = std::chrono::steady_clock::now();
    context->getSurface().present();
    const auto presentElapsed = std::chrono::steady_clock::now() - presentStart;
    timingTracker_.record(
        MainTimingStage::Present,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(presentElapsed).count())
    );
    if (checkGpuStall("present", presentStart, std::chrono::milliseconds(1500))) {
        finalizeFrameTiming();
        return;
    }

#ifdef WEBGPU_BACKEND_DAWN
    {
        const auto tickStart = std::chrono::steady_clock::now();
        context->getDevice().tick();
        const auto tickElapsed = std::chrono::steady_clock::now() - tickStart;
        timingTracker_.record(
            MainTimingStage::DeviceTick,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(tickElapsed).count())
        );
        if (checkGpuStall("device.tick (post-present)", tickStart, std::chrono::milliseconds(1500))) {
            finalizeFrameTiming();
            return;
        }
    }
#endif

    // Track slow frames for upload throttling.
    const auto totalFrameTime = std::chrono::steady_clock::now() - frameCpuStart;
    if (totalFrameTime > std::chrono::milliseconds(200)) {
        ++consecutiveSlowFrames_;
        if (consecutiveSlowFrames_ >= kMaxConsecutiveSlowFrames) {
            std::cerr << "GPU overload: " << kMaxConsecutiveSlowFrames
                      << " consecutive slow frames. Requesting shutdown." << std::endl;
            gpuStallDetected_ = true;
            glfwSetWindowShouldClose(context->getWindow(), GLFW_TRUE);
        }
    } else {
        consecutiveSlowFrames_ = 0;
    }

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

    if (doubleSidedMeshletCullingPipeline_.has_value()) {
        doubleSidedMeshletCullingPipeline_->removeResources();
        doubleSidedMeshletCullingPipeline_.reset();
    }

    if (culledMeshletCullingPipeline_.has_value()) {
        culledMeshletCullingPipeline_->removeResources();
        culledMeshletCullingPipeline_.reset();
    }

    if (meshletOcclusionPipeline_.has_value()) {
        meshletOcclusionPipeline_->removeResources();
        meshletOcclusionPipeline_.reset();
    }

    if (doubleSidedVoxelPipeline_.has_value()) {
        doubleSidedVoxelPipeline_->removeResources();
        doubleSidedVoxelPipeline_.reset();
    }
    if (culledVoxelPipeline_.has_value()) {
        culledVoxelPipeline_->removeResources();
        culledVoxelPipeline_.reset();
    }

    pendingMeshDelta_.reset();
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
