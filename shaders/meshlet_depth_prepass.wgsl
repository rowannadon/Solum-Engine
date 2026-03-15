// #include "uniforms.wgsl"

struct TileSlot {
    selectedResidentSlot: u32,
    visible: u32,
    flags: u32,
    pad0: u32,
    minCorner: vec4f,
    maxCorner: vec4f,
};

struct TileSceneParams {
    visibleTileCount: u32,
    tileSlotCount: u32,
    residentTileCount: u32,
    totalVisibleMeshlets: u32,
};

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read> visibleTileIds: array<u32>;
@group(0) @binding(2) var<storage, read> tileSlots: array<TileSlot>;
@group(0) @binding(3) var<uniform> sceneParams: TileSceneParams;

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
};

fn corner_position(minCorner: vec3f, maxCorner: vec3f, index: u32) -> vec3f {
    switch index {
        case 0u: { return vec3f(minCorner.x, minCorner.y, minCorner.z); }
        case 1u: { return vec3f(maxCorner.x, minCorner.y, minCorner.z); }
        case 2u: { return vec3f(maxCorner.x, maxCorner.y, minCorner.z); }
        case 3u: { return vec3f(minCorner.x, maxCorner.y, minCorner.z); }
        case 4u: { return vec3f(minCorner.x, minCorner.y, maxCorner.z); }
        case 5u: { return vec3f(maxCorner.x, minCorner.y, maxCorner.z); }
        case 6u: { return vec3f(maxCorner.x, maxCorner.y, maxCorner.z); }
        default: { return vec3f(minCorner.x, maxCorner.y, maxCorner.z); }
    }
}

fn triangle_corner_index(vertexIndex: u32) -> u32 {
    let triangleVertex = vertexIndex % 3u;
    let triangleIndex = vertexIndex / 3u;

    switch triangleIndex {
        case 0u: {
            if (triangleVertex == 0u) { return 0u; }
            if (triangleVertex == 1u) { return 1u; }
            return 2u;
        }
        case 1u: {
            if (triangleVertex == 0u) { return 0u; }
            if (triangleVertex == 1u) { return 2u; }
            return 3u;
        }
        case 2u: {
            if (triangleVertex == 0u) { return 4u; }
            if (triangleVertex == 1u) { return 6u; }
            return 5u;
        }
        case 3u: {
            if (triangleVertex == 0u) { return 4u; }
            if (triangleVertex == 1u) { return 7u; }
            return 6u;
        }
        case 4u: {
            if (triangleVertex == 0u) { return 0u; }
            if (triangleVertex == 1u) { return 4u; }
            return 5u;
        }
        case 5u: {
            if (triangleVertex == 0u) { return 0u; }
            if (triangleVertex == 1u) { return 5u; }
            return 1u;
        }
        case 6u: {
            if (triangleVertex == 0u) { return 1u; }
            if (triangleVertex == 1u) { return 5u; }
            return 6u;
        }
        case 7u: {
            if (triangleVertex == 0u) { return 1u; }
            if (triangleVertex == 1u) { return 6u; }
            return 2u;
        }
        case 8u: {
            if (triangleVertex == 0u) { return 2u; }
            if (triangleVertex == 1u) { return 6u; }
            return 7u;
        }
        case 9u: {
            if (triangleVertex == 0u) { return 2u; }
            if (triangleVertex == 1u) { return 7u; }
            return 3u;
        }
        case 10u: {
            if (triangleVertex == 0u) { return 3u; }
            if (triangleVertex == 1u) { return 7u; }
            return 4u;
        }
        default: {
            if (triangleVertex == 0u) { return 3u; }
            if (triangleVertex == 1u) { return 4u; }
            return 0u;
        }
    }
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    if (in.instance_idx >= sceneParams.visibleTileCount) {
        out.position = vec4f(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let tileSlotIndex = visibleTileIds[in.instance_idx];
    if (tileSlotIndex >= sceneParams.tileSlotCount) {
        out.position = vec4f(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let tileSlot = tileSlots[tileSlotIndex];
    if (tileSlot.visible == 0u || (tileSlot.flags & 0x2u) == 0u) {
        out.position = vec4f(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let corner = corner_position(tileSlot.minCorner.xyz, tileSlot.maxCorner.xyz, triangle_corner_index(in.vertex_idx));
    out.position = world_to_cull_clip_position(local_to_world_position(corner));
    return out;
}
