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

Architecture:
The storage of the world data in memory is organized hierarchically into Chunks (16x16x16 blocks). These are palette compressed. The uncompressed size of a single voxel is 4 bytes and this is the BlockMaterial type. Each chunk stores a mip chain of downscaled data which is used for LOD generation.

Chunks are organized into 32 tall Columns, representing a single column that is the height of the world. 

Columns are organized into Regions, which hold 32x32 Columns.

The engine uses a SSOT principle for the voxel data, the World class manages the generation of the world and stores it in the underlying data format. It provides an interface for getting block data, and this is used by anything that accesses the data such as meshing. Separating meshing from the underlying data simplifies the generation of LOD meshes.


The MeshManager is responsible for reading sections of the world using the World interface and generating meshes, it operates on a 4x4 column MeshTile unit. This class is also responsible for generating LOD meshes. The rendering occurs by breaking up each mesh into fixed size 128 quad meshlets, which are then drawn using drawInstanced.