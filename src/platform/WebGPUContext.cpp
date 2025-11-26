#include "solum_engine/platform/WebGPUContext.h"

bool WebGPUContext::initialize(const RenderConfig& config) {
    // Create instance descriptor
    InstanceDescriptor desc = {};
    desc.nextInChain = nullptr;

    // Make sure the uncaptured error callback is called as soon as an error
    // occurs rather than at the next call to "wgpuDeviceTick".
    DawnTogglesDescriptor toggles = Default;
    toggles.chain.next = nullptr;
    toggles.chain.sType = SType::DawnTogglesDescriptor;

    std::vector<const char*> enabledToggles = {
    "allow_unsafe_apis",
    "enable_immediate_error_handling",
    };

    toggles.disabledToggleCount = 0;
    toggles.enabledToggleCount = enabledToggles.size();
    toggles.enabledToggles = enabledToggles.data();
    desc.nextInChain = &toggles.chain;

    // Create the webgpu instance
    instance = wgpuCreateInstance(&desc);

    // We can check whether there is actually an instance created
    if (!instance) {
        std::cerr << "Could not initialize WebGPU!" << std::endl;
        return false;
    }

    // initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Could not initialize GLFW!" << std::endl;
        return false;
    }

    // create the window
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window = glfwCreateWindow(config.width, config.height, config.title, nullptr, nullptr);

    if (!window) {
        std::cerr << "Could not open window!" << std::endl;
        glfwTerminate();
        return 1;
    }

    surface = glfwGetWGPUSurface(instance, window);

    //surface = glfwCreateWindowWGPUSurface(instance, window);

    if (!surface) {
        std::cerr << "Could not create surface" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return false;
    }


    std::cout << "Requesting adapter..." << std::endl;

    RequestAdapterOptions adapterOpts = {};
    adapterOpts.nextInChain = nullptr;
    adapterOpts.compatibleSurface = surface;
    adapterOpts.powerPreference = WGPUPowerPreference_HighPerformance;
    adapterOpts.backendType = WGPUBackendType_Vulkan;

    RequestAdapterCallbackInfo callbackInfo = {};
    callbackInfo.nextInChain = nullptr;
    callbackInfo.mode = CallbackMode::WaitAnyOnly;
    callbackInfo.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView msg, void* userdata1, void* userdata2) {
        std::string message(static_cast<const char*>(msg.data), msg.length);
        
        if (status == WGPURequestAdapterStatus_Success) {
            *static_cast<WGPUAdapter*>(userdata1) = adapter;
        }
        else {
            std::cerr << "Could not get WebGPU adapter: " << message << std::endl;
            *static_cast<WGPUAdapter*>(userdata1) = nullptr;
        }
        };
    callbackInfo.userdata1 = &adapter;

    adapter = instance.requestAdapter(adapterOpts);

    std::cout << "Got adapter: " << adapter << std::endl;

    instance.release();

    Limits supportedLimits;
    adapter.getLimits(&supportedLimits);

    std::cout << "Requesting device..." << std::endl;

    DeviceDescriptor deviceDesc = {};
    Limits requiredLimits = GetRequiredLimits(adapter);
    deviceDesc.nextInChain = nullptr;
    deviceDesc.label = StringView("The Device"); // anything works here, that's your call
    deviceDesc.requiredLimits = &requiredLimits;
    deviceDesc.defaultQueue.nextInChain = nullptr;
    deviceDesc.defaultQueue.label = StringView("Main Queue");
    std::vector<FeatureName> requiredFeatures = {
        FeatureName::IndirectFirstInstance,
        FeatureName::MultiDrawIndirect,
        FeatureName::Unorm16TextureFormats
    };
    deviceDesc.requiredFeatures = (const WGPUFeatureName*)requiredFeatures.data();
    deviceDesc.requiredFeatureCount = (uint32_t)requiredFeatures.size();

    DeviceLostCallbackInfo deviceLostCallbackInfo = {};
    deviceLostCallbackInfo.nextInChain = nullptr;
    deviceLostCallbackInfo.mode = CallbackMode::AllowSpontaneous;
    deviceLostCallbackInfo.callback = [](const WGPUDevice *device, WGPUDeviceLostReason reason, WGPUStringView msg, void* userdata1, void* userdata2) {
        std::string message(static_cast<const char*>(msg.data), msg.length);
        std::cout << "Device lost: reason " << reason;
        if (message.length() > 0) std::cout << " (" << message << ")";
        std::cout << std::endl;
        };

    // A function that is invoked whenever the device stops being available.
    deviceDesc.deviceLostCallbackInfo = deviceLostCallbackInfo;

    UncapturedErrorCallbackInfo errorCallbackInfo = {};
    errorCallbackInfo.nextInChain = nullptr;
    errorCallbackInfo.callback = [](const WGPUDevice *device, WGPUErrorType type, WGPUStringView msg, void* userdata1, void* userdata2) {
        std::string message(static_cast<const char*>(msg.data), msg.length);
        std::cout << "Uncaptured device error: type " << type;
        if (message.length() > 0) std::cout << " (" << message << ")";
        std::cout << std::endl;
        };

    deviceDesc.uncapturedErrorCallbackInfo = errorCallbackInfo;

    device = adapter.requestDevice(deviceDesc);
    std::cout << "Got device: " << device << std::endl;

    if (!device.hasFeature(FeatureName::TimestampQuery)) {
        std::cout << "Timestamp queries are not supported!" << std::endl;
    }

    queue = device.getQueue();

    configureSurface();

    return true;
}

void WebGPUContext::terminate() {
    unconfigureSurface();
    queue.release();
    device.release();
    surface.release();
    adapter.release();

    glfwDestroyWindow(window);
    glfwTerminate();
}

bool WebGPUContext::configureSurface() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    SurfaceConfiguration config = {};
    config.nextInChain = nullptr;
    config.width = static_cast<uint32_t>(width);
    config.height = static_cast<uint32_t>(height);

    SurfaceCapabilities capabilities = {};
    surface.getCapabilities(adapter, &capabilities);

    if (capabilities.formatCount > 0) {
        surfaceFormat = capabilities.formats[0];
    }
    else {
        std::cerr << "No surface formats available!" << std::endl;
        return false;
    }

    config.format = surfaceFormat;

    //std::cout << "Surface format: " << magic_enum::enum_name<WGPUTextureFormat>(surfaceFormat) << std::endl;

    // And we do not need any particular view format:
    config.viewFormatCount = 0;
    config.viewFormats = nullptr;
    config.usage = TextureUsage::RenderAttachment;
    config.device = device;
    config.presentMode = PresentMode::Fifo;
    config.alphaMode = CompositeAlphaMode::Auto;

    surface.configure(config);

    return true;
}

void WebGPUContext::unconfigureSurface() {
    surface.unconfigure();
}

Limits WebGPUContext::GetRequiredLimits(Adapter adapter) const {
    // Get adapter supported limits, in case we need them
    Limits deviceLimits;
    adapter.getLimits(&deviceLimits);

    // Subtlety
    const_cast<WebGPUContext*>(this)->uniformStride = ceilToNextMultiple(
        (uint32_t)sizeof(FrameUniforms),
        (uint32_t)deviceLimits.minUniformBufferOffsetAlignment
    );

    // Don't forget to = Default
    Limits requiredLimits = Default;

    // We use at most 1 vertex attribute for now
    requiredLimits.maxVertexAttributes = 1;
    // We should also tell that we use 1 vertex buffers
    requiredLimits.maxVertexBuffers = 1;
    // Maximum size of a buffer is 6 vertices of 2 float each
    requiredLimits.maxBufferSize = 36000 * 32768 * sizeof(int);

    requiredLimits.maxStorageBufferBindingSize = 4294967295;
    // Maximum stride between 2 consecutive vertices in the vertex buffer
    requiredLimits.maxVertexBufferArrayStride = sizeof(int);

    // These two limits are different because they are "minimum" limits,
    // they are the only ones we may forward from the adapter's supported
    // limits.
    requiredLimits.minUniformBufferOffsetAlignment = deviceLimits.minUniformBufferOffsetAlignment;
    requiredLimits.minStorageBufferOffsetAlignment = deviceLimits.minStorageBufferOffsetAlignment;

    // There is a maximum of 3 float forwarded from vertex to fragment shader
    requiredLimits.maxInterStageShaderVariables = 21;

    // We use at most 1 bind group for now
    requiredLimits.maxBindGroups = 4;
    requiredLimits.maxBindGroupsPlusVertexBuffers = 8;
    // We use at most 1 uniform buffer per stage
    requiredLimits.maxUniformBuffersPerShaderStage = 1;
    // Add the possibility to sample a texture in a shader
    requiredLimits.maxSampledTexturesPerShaderStage = 1;
    // Uniform structs have a size of maximum 16 float (more than what we need)
    requiredLimits.maxUniformBufferBindingSize = 16 * 4 * sizeof(float);
    // Extra limit requirement
    requiredLimits.maxDynamicUniformBuffersPerPipelineLayout = 1;

    requiredLimits.maxComputeInvocationsPerWorkgroup = 1024;

    requiredLimits.maxSamplersPerShaderStage = 1;

    // For the depth buffer, we enable textures (up to the size of the window):
    requiredLimits.maxTextureDimension1D = 2048;
    requiredLimits.maxTextureDimension2D = 16384;
    requiredLimits.maxTextureDimension3D = 2048;
    requiredLimits.maxTextureArrayLayers = 1;


    return requiredLimits;
}

uint32_t WebGPUContext::ceilToNextMultiple(uint32_t value, uint32_t step) const {
    uint32_t divide_and_ceil = value / step + (value % step == 0 ? 0 : 1);
    return step * divide_and_ceil;
}

