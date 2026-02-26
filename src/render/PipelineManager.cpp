#include "solum_engine/render/PipelineManager.h"
#include <functional>
#include <unordered_set>
#include <string>
#include <iostream>

using namespace wgpu;

RenderPipeline PipelineManager::createRenderPipeline(const std::string& pipelineName, PipelineConfig& config) {
    std::cout << "Creating shader module..." << std::endl;
    ShaderModule shaderModule = loadShaderModule(config.shaderPath, device);
    std::cout << "Shader module: " << shaderModule << std::endl;
    if (shaderModule == nullptr) {
        std::cout << "Failed to load shader: " << config.shaderPath << std::endl;
        return nullptr;
    }

    RenderPipelineDescriptor pipelineDesc = Default;
    pipelineDesc.label = StringView(pipelineName);

    // Handle vertex buffer configuration
    VertexBufferLayout vertexBufferLayout = Default;
    if (config.useVertexBuffers && !config.vertexAttributes.empty()) {
        vertexBufferLayout.attributeCount = static_cast<uint32_t>(config.vertexAttributes.size());
        vertexBufferLayout.attributes = config.vertexAttributes.data();
        vertexBufferLayout.arrayStride = config.vertexBufferStride > 0
            ? config.vertexBufferStride
            : sizeof(VertexAttributes);
        vertexBufferLayout.stepMode = VertexStepMode::Vertex;

        pipelineDesc.vertex.bufferCount = 1;
        pipelineDesc.vertex.buffers = &vertexBufferLayout;
    }
    else {
        // No vertex buffers - for procedural geometry
        pipelineDesc.vertex.bufferCount = 0;
        pipelineDesc.vertex.buffers = nullptr;
    }

    pipelineDesc.vertex.module = shaderModule;
    pipelineDesc.vertex.entryPoint = StringView(config.vertexShaderName);
    pipelineDesc.vertex.constantCount = 0;
    pipelineDesc.vertex.constants = nullptr;

    // Primitive state
    pipelineDesc.primitive.topology = config.topology;
    pipelineDesc.primitive.stripIndexFormat = IndexFormat::Undefined;
    pipelineDesc.primitive.frontFace = FrontFace::CCW;
    pipelineDesc.primitive.cullMode = config.cullMode;

    // Multisample state
    pipelineDesc.multisample.count = config.sampleCount;
    pipelineDesc.multisample.mask = ~0u;
    pipelineDesc.multisample.alphaToCoverageEnabled = config.alphaToCoverageEnabled;

    // Fragment state
    FragmentState fragmentState = Default;
    pipelineDesc.fragment = &fragmentState;
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = StringView(config.fragmentShaderName);
    fragmentState.constantCount = 0;
    fragmentState.constants = nullptr;

    BlendState blendState = Default;
    ColorTargetState colorTarget = Default;

    if (config.useColorTarget) {
        // Blend state
        if (config.useCustomBlending) {
            blendState = config.blendState;
            colorTarget.blend = &blendState;
        }
        else {
            colorTarget.blend = nullptr;
        }

        // Color target state
        colorTarget.format = config.useCustomColorFormat ? config.colorFormat : surfaceFormat;
        colorTarget.writeMask = ColorWriteMask::All;

        fragmentState.targetCount = 1;
        fragmentState.targets = &colorTarget;
    }
    else {
        fragmentState.targetCount = 0;
        fragmentState.targets = nullptr;
    }

    // Depth stencil state - declare outside to keep in scope
    DepthStencilState depthStencilState = Default;

    // Only configure and set depth stencil if useDepthStencil is true
    if (config.useDepthStencil) {
        depthStencilState.depthCompare = config.depthCompare;
        depthStencilState.depthWriteEnabled = config.depthWriteEnabled ? OptionalBool::True : OptionalBool::False;
        depthStencilState.format = config.depthFormat;
        depthStencilState.stencilReadMask = 0;
        depthStencilState.stencilWriteMask = 0;
        pipelineDesc.depthStencil = &depthStencilState;
    }
    else {
        pipelineDesc.depthStencil = nullptr;
    }

    // Pipeline layout
    PipelineLayoutDescriptor layoutDesc = Default;
    layoutDesc.bindGroupLayoutCount = (uint32_t)config.bindGroupLayouts.size();
    layoutDesc.bindGroupLayouts = reinterpret_cast<WGPUBindGroupLayout*>(config.bindGroupLayouts.data());
    PipelineLayout layout = device.createPipelineLayout(layoutDesc);

    pipelineDesc.layout = layout;

    RenderPipeline pipeline = device.createRenderPipeline(pipelineDesc);
    std::cout << "Render pipeline: " << pipeline << std::endl;

    auto existingPipeline = pipelines.find(pipelineName);
    if (existingPipeline != pipelines.end() && existingPipeline->second) {
        existingPipeline->second.release();
        pipelines.erase(existingPipeline);
    }

    pipelines[pipelineName] = pipeline;

    // Clean up
    shaderModule.release();
    layout.release();

    return pipeline;
}

BindGroupLayout PipelineManager::createBindGroupLayout(const std::string& bindGroupLayoutName,
                                                       const std::vector<BindGroupLayoutEntry>& entries) {
    auto existing = bindGroupLayouts.find(bindGroupLayoutName);
    if (existing != bindGroupLayouts.end() && existing->second) {
        existing->second.release();
        bindGroupLayouts.erase(existing);
    }

    BindGroupLayoutDescriptor chunkDataBindGroupLayoutDesc = Default;
    chunkDataBindGroupLayoutDesc.entryCount = (uint32_t)entries.size();
    chunkDataBindGroupLayoutDesc.entries = entries.data();

    BindGroupLayout layout = device.createBindGroupLayout(chunkDataBindGroupLayoutDesc);
    bindGroupLayouts[bindGroupLayoutName] = layout;
    return layout;
}

void PipelineManager::deleteBindGroup(const std::string& bindGroupName) {
    BindGroup group = getBindGroup(bindGroupName);
    if (group) {
        group.release();
        bindGroups.erase(bindGroupName);
    }
}

BindGroup PipelineManager::createBindGroup(const std::string& bindGroupName,
                                           const std::string& bindGroupLayoutName,
                                           const std::vector<BindGroupEntry>& bindings) {
    auto layoutIt = bindGroupLayouts.find(bindGroupLayoutName);
    if (layoutIt == bindGroupLayouts.end()) {
        return nullptr;
    }
    BindGroupLayout layout = layoutIt->second;
    if (!layout) {
		return nullptr;
    }

    BindGroupDescriptor bindGroupDesc = Default;
    bindGroupDesc.label = StringView(bindGroupName);
    bindGroupDesc.layout = layout;
    bindGroupDesc.entryCount = (uint32_t)bindings.size();
    bindGroupDesc.entries = bindings.data();

    auto existing = bindGroups.find(bindGroupName);
    if (existing != bindGroups.end() && existing->second) {
        existing->second.release();
        bindGroups.erase(existing);
    }

    BindGroup bindGroup = device.createBindGroup(bindGroupDesc);
    bindGroups[bindGroupName] = bindGroup;
    return bindGroup;
}

RenderPipeline PipelineManager::getPipeline(const std::string& pipelineName) const {
    auto pipeline = pipelines.find(pipelineName);
    if (pipeline != pipelines.end()) {
        return pipeline->second;
    }
    return nullptr;
}

BindGroupLayout PipelineManager::getBindGroupLayout(const std::string& bindGroupLayoutName) const {
    auto layout = bindGroupLayouts.find(bindGroupLayoutName);
    if (layout != bindGroupLayouts.end()) {
        return layout->second;
    }
    return nullptr;
}

BindGroup PipelineManager::getBindGroup(const std::string& bindGroupName) const {
    auto bindGroup = bindGroups.find(bindGroupName);
    if (bindGroup != bindGroups.end()) {
        return bindGroup->second;
    }
    return nullptr;
}

void PipelineManager::terminate() {
    for (auto& pair : pipelines) {
        if (pair.second) {
            pair.second.release();
        }
    }

    for (auto& pair : bindGroupLayouts) {
        if (pair.second) {
            pair.second.release();
        }
    }

    for (auto& pair : bindGroups) {
        if (pair.second) {
            pair.second.release();
        }
    }

    pipelines.clear();
    bindGroupLayouts.clear();
    bindGroups.clear();
}

ShaderModule PipelineManager::loadShaderModule(const std::filesystem::path& path, Device device) {
    // This will hold the fully expanded WGSL source
    std::string shaderSource;

    // Tracks files currently on the include stack (cycle detection)
    std::unordered_set<std::string> includeStack;

    // Recursive loader for a single file
    std::function<bool(const std::filesystem::path&)> loadFile;
    loadFile = [&](const std::filesystem::path& currentPath) -> bool {
        // Normalize key for cycle detection
        std::string key;
        try {
            key = std::filesystem::absolute(currentPath).string();
        } catch (...) {
            key = currentPath.string();
        }

        if (includeStack.find(key) != includeStack.end()) {
            std::cerr << "Detected cyclic shader include: " << currentPath << std::endl;
            return false;
        }

        includeStack.insert(key);

        std::ifstream file(currentPath);
        if (!file.is_open()) {
            std::cerr << "Failed to open shader file: " << currentPath << std::endl;
            includeStack.erase(key);
            return false;
        }

        const std::filesystem::path baseDir = currentPath.parent_path();
        std::string line;

        // Optionally add a comment so you can see file boundaries in errors
        shaderSource += "// ---- begin include: " + currentPath.string() + " ----\n";

        while (std::getline(file, line)) {
            // Trim leading whitespace
            std::string trimmed = line;
            auto firstNonWs = trimmed.find_first_not_of(" \t\r\n");
            if (firstNonWs != std::string::npos) {
                trimmed.erase(0, firstNonWs);
            } else {
                // Line is all whitespace
                shaderSource += line + "\n";
                continue;
            }

            // Check for: // #include "something"
            if (trimmed.rfind("// #include", 0) == 0) {
                auto firstQuote = trimmed.find('"');
                auto lastQuote  = trimmed.find('"', firstQuote + 1);

                if (firstQuote != std::string::npos &&
                    lastQuote  != std::string::npos &&
                    lastQuote > firstQuote + 1) {

                    std::string includePathStr = trimmed.substr(firstQuote + 1,
                                                                lastQuote - firstQuote - 1);

                    std::filesystem::path includePath(includePathStr);

                    // Relative paths are resolved against the including file's directory
                    if (!includePath.is_absolute()) {
                        includePath = baseDir / includePath;
                    }

                    if (!loadFile(includePath)) {
                        std::cerr << "Failed to load included shader file: "
                                  << includePath << std::endl;
                        includeStack.erase(key);
                        return false;
                    }

                    // After including, we can add a newline to separate logically
                    shaderSource += "\n";
                    continue; // Do not keep the original include line
                }
                // If parsing failed, fall through and treat it as a normal comment line
            }

            // Normal line (not an include directive) — just append as-is
            shaderSource += line + "\n";
        }

        shaderSource += "// ---- end include: " + currentPath.string() + " ----\n\n";

        includeStack.erase(key);
        return true;
    };

    // Load the root shader file (and all of its includes)
    if (!loadFile(path)) {
        return nullptr;
    }

    // Now create the actual shader module from the expanded source
    ShaderModuleWGSLDescriptor shaderCodeDesc{};
    shaderCodeDesc.chain.next  = nullptr;
    shaderCodeDesc.chain.sType = SType::ShaderSourceWGSL;
    shaderCodeDesc.code        = StringView(shaderSource);

    ShaderModuleDescriptor shaderDesc = Default;
#ifdef WEBGPU_BACKEND_WGPU
    shaderDesc.hintCount = 0;
    shaderDesc.hints     = nullptr;
#endif
    shaderDesc.nextInChain = &shaderCodeDesc.chain;

    return device.createShaderModule(shaderDesc);
}
