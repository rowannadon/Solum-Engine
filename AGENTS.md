This is an in progress project for a flexible, high performance voxel game engine built using WebGPU (Dawn) and C++. The engine is designed to render simple, minecraft style graphics and blocks.

Testing: Testing will be performed manually, do NOT attempt any build or verification steps after making changes.

Directory structure:
/include/solum_engine - header files (directory structure mirrors src)

/shaders - WGSL shaders

/src/core - main application code
/src/platform - platform specific initialization for WebGPU
/src/render - render related code
/src/render/pipelines - render pipelines
/src/ui - ui related code (ImGui)
/src/voxel - voxel world related code