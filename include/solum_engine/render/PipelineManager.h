#ifndef PIPELINE_MANAGER
#define PIPELINE_MANAGER

#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include <filesystem>
#include <fstream> 
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>

#include "solum_engine/render/VertexAttributes.h"

struct PipelineConfig {
    std::string shaderPath;
    std::string fragmentShaderName = "fs_main";
    std::string vertexShaderName = "vs_main";
    std::vector<wgpu::VertexAttribute> vertexAttributes;
    std::vector<wgpu::BindGroupLayout> bindGroupLayouts;
    wgpu::TextureFormat colorFormat = wgpu::TextureFormat::BGRA8Unorm;
    wgpu::TextureFormat depthFormat = wgpu::TextureFormat::Depth24Plus;
    uint32_t sampleCount = 4;
    wgpu::PrimitiveTopology topology = wgpu::PrimitiveTopology::TriangleList;
    wgpu::CullMode cullMode = wgpu::CullMode::Back;
    bool depthWriteEnabled = true;
    wgpu::CompareFunction depthCompare = wgpu::CompareFunction::Less;
    bool useVertexBuffers = true;  // New flag to control vertex buffer usage
    uint64_t vertexBufferStride = sizeof(VertexAttributes);
    bool useColorTarget = true;
    bool useCustomBlending = false;
    wgpu::BlendState blendState;
    bool alphaToCoverageEnabled = false;
    bool useCustomColorFormat = false;
    bool useDepthStencil = true;  // Add this flag
};

class PipelineManager {
    std::unordered_map<std::string, wgpu::RenderPipeline> pipelines;
    std::unordered_map<std::string, wgpu::BindGroupLayout> bindGroupLayouts;
    std::unordered_map<std::string, wgpu::BindGroup> bindGroups;
    wgpu::Device device;
    wgpu::TextureFormat surfaceFormat;

public:
    PipelineManager(wgpu::Device d, wgpu::TextureFormat sf) : device(d), surfaceFormat(sf) {}

    wgpu::RenderPipeline createRenderPipeline(const std::string& pipelineName, PipelineConfig& config);

    wgpu::BindGroupLayout createBindGroupLayout(const std::string& bindGroupLayoutName,
                                                const std::vector<wgpu::BindGroupLayoutEntry>& entries);
    wgpu::BindGroup createBindGroup(const std::string& bindGroupName,
                                    const std::string& bindGroupLayoutName,
                                    const std::vector<wgpu::BindGroupEntry>& bindings);
    wgpu::RenderPipeline getPipeline(const std::string& pipelineName) const;
    wgpu::BindGroupLayout getBindGroupLayout(const std::string& bindGroupLayoutName) const;
    wgpu::BindGroup getBindGroup(const std::string& bindGroupName) const;
    void deleteBindGroup(const std::string& bindGroupName);

    void terminate();
private:
    wgpu::ShaderModule loadShaderModule(const std::filesystem::path& path, wgpu::Device device);
};

#endif
