#include "solum_engine/platform/WebGPUContext.h"
#include <cstdlib>
#include <cstring>
#include <optional>

namespace {
const char* presentModeName(PresentMode mode) {
    switch (mode) {
    case PresentMode::Immediate:
        return "Immediate";
    case PresentMode::Mailbox:
        return "Mailbox";
    case PresentMode::Fifo:
        return "Fifo";
    case PresentMode::FifoRelaxed:
        return "FifoRelaxed";
    default:
        return "Unknown";
    }
}

PresentMode choosePreferredPresentMode(const SurfaceCapabilities& capabilities) {
    auto supports = [&capabilities](PresentMode mode) {
        for (size_t i = 0; i < capabilities.presentModeCount; ++i) {
            if (capabilities.presentModes[i] == mode) {
                return true;
            }
        }
        return false;
    };

    auto parseEnvPresentMode = []() -> std::optional<PresentMode> {
        const char* modeValue = std::getenv("SOL_PRESENT_MODE");
        if (modeValue == nullptr) {
            return std::nullopt;
        }
        if (std::strcmp(modeValue, "immediate") == 0) {
            return PresentMode::Immediate;
        }
        if (std::strcmp(modeValue, "mailbox") == 0) {
            return PresentMode::Mailbox;
        }
        if (std::strcmp(modeValue, "fifo") == 0) {
            return PresentMode::Fifo;
        }
        if (std::strcmp(modeValue, "fifo_relaxed") == 0) {
            return PresentMode::FifoRelaxed;
        }
        return std::nullopt;
    };

    if (const std::optional<PresentMode> envMode = parseEnvPresentMode(); envMode.has_value()) {
        if (supports(*envMode)) {
            return *envMode;
        }
        std::cout << "Requested present mode via SOL_PRESENT_MODE is unavailable: "
                  << presentModeName(*envMode) << std::endl;
    }

#if defined(_WIN32) || defined(__APPLE__)
return PresentMode::Fifo;
    if (supports(PresentMode::Immediate)) {
        return PresentMode::Immediate;
    }
    if (supports(PresentMode::Mailbox)) {
        return PresentMode::Mailbox;
    }
    if (supports(PresentMode::FifoRelaxed)) {
        return PresentMode::FifoRelaxed;
    }
    if (supports(PresentMode::Fifo)) {
        return PresentMode::Fifo;
    }
#else
    if (supports(PresentMode::Mailbox)) {
        return PresentMode::Mailbox;
    }
    if (supports(PresentMode::Immediate)) {
        return PresentMode::Immediate;
    }
    if (supports(PresentMode::FifoRelaxed)) {
        return PresentMode::FifoRelaxed;
    }
    if (supports(PresentMode::Fifo)) {
        return PresentMode::Fifo;
    }
#endif

    return (capabilities.presentModeCount > 0) ? capabilities.presentModes[0] : PresentMode::Fifo;
}
}  // namespace

bool WebGPUContext::initialize(const RenderConfig& config) {
    // Create instance descriptor
    InstanceDescriptor desc = Default;
    desc.nextInChain = nullptr;

    // Make sure the uncaptured error callback is called as soon as an error
    // occurs rather than at the next call to "wgpuDeviceTick".
    DawnTogglesDescriptor toggles = Default;
    toggles.chain.next = nullptr;
    toggles.chain.sType = SType::DawnTogglesDescriptor;

    std::vector<const char*> enabledToggles = {
    //"allow_unsafe_apis",
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
        return false;
    }

    surface = glfwGetWGPUSurface(instance, window);

    if (!surface) {
        std::cerr << "Could not create surface" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return false;
    }


    std::cout << "Requesting adapter..." << std::endl;

    RequestAdapterOptions adapterOpts = Default;
    adapterOpts.nextInChain = nullptr;
    adapterOpts.compatibleSurface = surface;
    adapterOpts.powerPreference = WGPUPowerPreference_HighPerformance;

    #ifdef _WIN32
        // Let Dawn pick the native best backend on Windows (typically D3D12).
        // Forcing Vulkan can increase swapchain acquire stalls on some drivers.
        adapterOpts.backendType = WGPUBackendType_Undefined;
    #elif __linux__
        adapterOpts.backendType = WGPUBackendType_Vulkan;
    #elif __APPLE__
        adapterOpts.backendType = WGPUBackendType_Metal;
    #else
        std::cout << "Unknown or unsupported platform." << std::endl;
        return false;
    #endif

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

    Limits supportedLimits;
    adapter.getLimits(&supportedLimits);

    std::cout << "Requesting device..." << std::endl;

    DeviceDescriptor deviceDesc = Default;
    Limits requiredLimits = GetRequiredLimits(adapter);
    deviceDesc.nextInChain = nullptr;
    deviceDesc.label = StringView("The Device"); // anything works here, that's your call
    deviceDesc.requiredLimits = &requiredLimits;
    deviceDesc.defaultQueue.nextInChain = nullptr;
    deviceDesc.defaultQueue.label = StringView("Main Queue");
    std::vector<FeatureName> requiredFeatures = {
        FeatureName::IndirectFirstInstance,
        //FeatureName::MultiDrawIndirect,
        FeatureName::Unorm16TextureFormats
    };
    deviceDesc.requiredFeatures = (const WGPUFeatureName*)requiredFeatures.data();
    deviceDesc.requiredFeatureCount = (uint32_t)requiredFeatures.size();

    DeviceLostCallbackInfo deviceLostCallbackInfo = Default;
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

    UncapturedErrorCallbackInfo errorCallbackInfo = Default;
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

    if (!configureSurface()) {
        std::cerr << "Could not configure surface during initialization." << std::endl;
        return false;
    }

    return true;
}

void WebGPUContext::terminate() {
    unconfigureSurface();
    queue.release();
    device.release();
    surface.release();
    adapter.release();
    instance.release();

    glfwDestroyWindow(window);
    glfwTerminate();
}

bool WebGPUContext::configureSurface() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    if (width <= 0 || height <= 0) {
        std::cerr << "configureSurface skipped invalid size " << width << "x" << height << std::endl;
        return false;
    }
    this->width = width;
    this->height = height;

    SurfaceConfiguration config = Default;
    config.nextInChain = nullptr;
    config.width = static_cast<uint32_t>(this->width);
    config.height = static_cast<uint32_t>(this->height);

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
    config.presentMode = choosePreferredPresentMode(capabilities);
    config.alphaMode = CompositeAlphaMode::Auto;

    surface.configure(config);
    std::cout << "Configured surface " << this->width << "x" << this->height
              << " present mode: " << presentModeName(config.presentMode) << std::endl;

    return true;
}

void WebGPUContext::unconfigureSurface() {
    surface.unconfigure();
}

Limits WebGPUContext::GetRequiredLimits(Adapter adapter) const {
    // Get adapter supported limits, in case we need them
    Limits deviceLimits;
    adapter.getLimits(&deviceLimits);

    // Request full adapter-supported limits to avoid accidentally constraining
    // secondary pipelines (e.g. ImGui) below what they require.
    Limits requiredLimits = deviceLimits;

    return requiredLimits;
}
