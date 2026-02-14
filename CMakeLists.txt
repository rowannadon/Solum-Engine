# My CMakeLists.txt 

cmake_minimum_required(VERSION 3.1...3.25)
project(
    SolumEngine # name of the project, which will also be the name of the visual studio solution if you use it
    VERSION 0.1.0 # any version number
    LANGUAGES CXX C # programming languages used by the project
)

set(FASTNOISE2_NOISETOOL OFF CACHE BOOL "Build Noise Tool" FORCE) #example if don't need the graph tool
set(WEBGPU_BUILD_FROM_SOURCE ON CACHE BOOL "Build WebGPU from source")

add_subdirectory(external/glfw)
add_subdirectory(external/webgpu)
add_subdirectory(external/glfw3webgpu)
add_subdirectory(external/FastNoise2)
add_subdirectory(external/imgui)

# Header-only includes
set(THIRD_PARTY_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glm
    ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/external/lodepng
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ogt_vox
    ${CMAKE_CURRENT_SOURCE_DIR}/external/nlohmann_json
)

file(GLOB_RECURSE SOLUM_ENGINE_SRC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
list(APPEND SOLUM_ENGINE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/lodepng/lodepng.cpp
)

add_executable(solum_engine ${SOLUM_ENGINE_SRC})

# We add an option to enable different settings when developing the app than
# when distributing it.
option(DEV_MODE "Set up development helper settings" ON)

# Add the 'webgpu' target as a dependency of our App
target_include_directories(solum_engine
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external
        ${THIRD_PARTY_INCLUDE_DIRS}
)

target_link_libraries(solum_engine PRIVATE glfw webgpu glfw3webgpu FastNoise imgui)

if(DEV_MODE)
    # In dev mode, we load resources from the source tree, so that when we
    # dynamically edit resources (like shaders), these are correctly
    # versionned.
    target_compile_definitions(solum_engine PRIVATE
        RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
        SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    )
else()
    # In release mode, we just load resources relatively to wherever the
    # executable is launched from, so that the binary is portable
    target_compile_definitions(solum_engine PRIVATE
        RESOURCE_DIR="./assets"
        SHADER_DIR="./shaders"
    )
endif()


set_target_properties(solum_engine PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    COMPILE_WARNING_AS_ERROR OFF
)

if (MSVC)
    target_compile_options(solum_engine PRIVATE /wd4244)
else()
    target_compile_options(solum_engine PRIVATE -Wall -Wextra -pedantic)
endif()

if (XCODE)
    set_target_properties(solum_engine PROPERTIES
        XCODE_GENERATE_SCHEME ON
        XCODE_SCHEME_ENABLE_GPU_FRAME_CAPTURE_MODE "Metal"
    )
endif()

target_copy_webgpu_binaries(solum_engine)
