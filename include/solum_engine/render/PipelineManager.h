#ifndef PIPELINE_MANAGER
#define PIPELINE_MANAGER

#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include <filesystem>
#include <fstream> 
#include <vector>

using namespace wgpu;

struct PipelineConfig {
    std::string shaderPath;
    std::string fragmentShaderName = "fs_main";
    std::string vertexShaderName = "vs_main";
    std::vector<VertexAttribute> vertexAttributes;
    std::vector<BindGroupLayout> bindGroupLayouts;
    TextureFormat colorFormat = TextureFormat::BGRA8Unorm;
    TextureFormat depthFormat = TextureFormat::Depth24Plus;
    uint32_t sampleCount = 4;
    PrimitiveTopology topology = PrimitiveTopology::TriangleList;
    CullMode cullMode = CullMode::Back;
    bool depthWriteEnabled = true;
    CompareFunction depthCompare = CompareFunction::Less;
    bool useVertexBuffers = true;  // New flag to control vertex buffer usage
    bool useColorTarget = true;
    bool useCustomBlending = false;
    BlendState blendState;
    bool alphaToCoverageEnabled = false;
    bool useCustomColorFormat = false;
    bool useDepthStencil = true;  // Add this flag
};

struct ComputePipelineConfig {
    std::string shaderPath;
    std::string computeShaderName = "compute_main";
    std::vector<BindGroupLayout> bindGroupLayouts;
};

class PipelineManager {
    std::unordered_map<std::string, RenderPipeline> pipelines;
    std::unordered_map<std::string, ComputePipeline> computePipelines;
    std::unordered_map<std::string, BindGroupLayout> bindGroupLayouts;
    std::unordered_map<std::string, BindGroup> bindGroups;
    Device device;
    TextureFormat surfaceFormat;

public:
    PipelineManager(Device d, TextureFormat sf) : device(d), surfaceFormat(sf) {}

    RenderPipeline createRenderPipeline(const std::string pipelineName, PipelineConfig & config);
    ComputePipeline createComputePipeline(const std::string pipelineName, ComputePipelineConfig& config);

    BindGroupLayout createBindGroupLayout(const std::string bindGroupLayoutName, const std::vector<BindGroupLayoutEntry>& entries);
    BindGroup createBindGroup(const std::string bindGroupName, const std::string bindGroupLayoutName, const std::vector<BindGroupEntry>& bindings);
    RenderPipeline getPipeline(std::string pipelineName);
    ComputePipeline getComputePipeline(std::string pipelineName);
    BindGroupLayout getBindGroupLayout(std::string bindGroupLayoutName);
    BindGroup getBindGroup(std::string bindGroupName);
    void deleteBindGroup(std::string bindGroupName);

    void terminate();
private:
    ShaderModule loadShaderModule(const std::filesystem::path& path, Device device);
};

#endif