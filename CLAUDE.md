# Solum Engine

A C++20 WebGPU voxel renderer with GPU-driven meshlet culling, streaming LOD, and procedural terrain generation.

## Build

CMake project. Dependencies: GLFW3, WebGPU, glfw3webgpu, FastNoise2, ImGui, GLM, stb, lodepng, nlohmann_json, ogt_vox.

- Dev mode: loads shaders/assets from source tree (live shader editing)
- Release mode: loads relative to executable
- macOS: Xcode GPU frame capture enabled

## Architecture Overview

### Threading Model

Two threads: **main** (render) and **streaming** (world/mesh generation).

```
Main Thread                     Streaming Thread
──────────────────────────────  ──────────────────────────────
Application loop                World (column generation jobs)
└─ WebGPURenderer.renderFrame() MeshManager (tile LOD meshing)
   ├─ processPendingMeshUploads  └─ StreamingMeshUpload queue
   ├─ Hi-Z depth prepass              ↑
   ├─ Hi-Z pyramid build         consumed by main thread
   ├─ Meshlet cull (compute)
   └─ Voxel render (indirect)
```

### Voxel World Hierarchy

Four coordinate spaces (type-safe tagged types prevent mixing):

```
Region (32×32 columns, 512×512 blocks)
  └─ Column (1 column wide, 32 chunks tall)
     └─ Chunk (16³ voxels, palette-compressed, 5-level mipmaps)
        └─ BlockMaterial (32-bit packed: material ID, water level, direction, rotation)
```

- **Z-up coordinate system**: X/Y horizontal, Z vertical
- **World**: procedural terrain via FastNoise2, distance-prioritized streaming
- **Chunk**: palette compression with dynamic bit packing; mip levels 16→8→4→2→1

### Meshlet System (Core Rendering Primitive)

Chunks are meshed into **Meshlets** — face-direction-aligned quad groups (max 128 quads each).

- `Meshlet` (CPU): origin, faceDirection, quadCount, voxelScale, packed quad data
- `MeshletMetadataGPU` (32B): origin XYZ, quad count, face direction, data offset, voxel scale
- `MeshletAabbGPU` (32B): min/max corners (vec4f aligned)

**MeshletBufferController**: double-buffered GPU upload, streamed in 1 MB/frame chunks.

### Mesh LOD

**MeshManager** organizes meshes into hierarchical tiles:

- Screen-space error (SSE) drives LOD selection
- Hysteresis prevents thrashing at LOD boundaries
- Distance-based priority scheduling
- Tiles store multiple LOD variants; `TileLodCoord` / `TileLodCellCoord` address them

**ChunkMesher**: greedy quad merging for all 6 face directions, outputs Meshlet arrays.

### GPU Render Pipeline (per frame)

| Pass | Shader | Purpose |
|------|--------|---------|
| Meshlet depth prepass | `meshlet_depth_prepass.wgsl` | Render AABB occluders to depth |
| Hi-Z seed | `meshlet_hiz_seed.wgsl` | 2× downsample to start pyramid |
| Hi-Z downsample | `meshlet_hiz_downsample.wgsl` | Build full mip pyramid (8×8 workgroups) |
| Meshlet cull | `meshlet_cull.wgsl` | Frustum + Hi-Z occlusion, writes indirect args (128-thread workgroups) |
| Main voxel render | `voxel.wgsl` | Indirect draw, quad expansion, texture atlas, AO |
| Debug bounds | `debug_bounds.wgsl` | Optional AABB visualization |

`uniforms.wgsl` is a shared include (frame matrices, occlusion params).

### Key Managers

| Class | Responsibility |
|-------|---------------|
| `WebGPURenderer` | Central render orchestrator; owns all GPU resources and passes |
| `PipelineManager` | Loads WGSL shaders, creates/caches render & compute pipelines |
| `BufferManager` | Named GPU buffer registry |
| `TextureManager` | GPU textures, views, samplers |
| `MaterialManager` | Material ID → texture array index mappings |
| `MeshletBufferController` | Double-buffered meshlet GPU uploads |

### Performance Infrastructure

- **Runtime timing**: atomic per-stage counters (totalNs, callCount, maxNs); thread-safe snapshots
- Tracked stages: Upload, UpdateBounds, EncodeCmds, Submit, Present, StreamWait, WorldUpdate, MeshUpdate, etc.
- Skip reasons logged: NoCamera, Unchanged, Throttled

## Key Files

```
include/solum_engine/
  core/             Application, FirstPersonCamera
  platform/         WebGPUContext
  render/           WebGPURenderer, MeshletBufferController
  render/pipelines/ VoxelPipeline, BoundsDebugPipeline, occlusion pipelines
  voxel/            World, Region, Column, Chunk, BlockMaterial
                    MeshManager, ChunkMesher, VoxelStreamingSystem
  resources/        Constants, coordinate type tags
  ui/               GuiManager (ImGui)
src/                Implementation (.cpp) files
shaders/            WGSL shaders (voxel, meshlet_*, debug_bounds, uniforms)
```

## Design Patterns

- **GPU-driven rendering**: compute shader culling writes indirect draw args; no CPU readback
- **Double-buffered meshlets**: two GPU buffer sets swap to avoid CPU/GPU sync stalls
- **Streaming budget**: 1 MB/frame GPU upload cap prevents frame hitches
- **Type-safe coordinates**: tagged `GridCoord` types with explicit converters and hash specializations
- **Named resource registry**: managers key resources by string name for flexible pipeline composition
