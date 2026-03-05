// #include "uniforms.wgsl"
// #include "meshlet_shared.wgsl"

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read> meshletDataWords: array<u32>;
@group(0) @binding(2) var<storage, read> meshletMetadata: array<MeshletMetadata>;

struct ActiveMeshletRange {
    meshletOffset: u32,
    meshletCount: u32,
    prefixEnd: u32,
    pad: u32,
};

struct ActiveRangeParams {
    rangeCount: u32,
    totalActiveMeshlets: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(3) var<storage, read> activeRanges: array<ActiveMeshletRange>;
@group(0) @binding(4) var<uniform> activeRangeParams: ActiveRangeParams;

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
};

fn resolve_meshlet_index(activeIndex: u32) -> u32 {
    if (activeRangeParams.rangeCount == 0u) {
        return 0xffffffffu;
    }

    var left: u32 = 0u;
    var right: u32 = activeRangeParams.rangeCount;
    while (left < right) {
        let mid = (left + right) / 2u;
        let prefix = activeRanges[mid].prefixEnd;
        if (activeIndex < prefix) {
            right = mid;
        } else {
            left = mid + 1u;
        }
    }

    if (left >= activeRangeParams.rangeCount) {
        return 0xffffffffu;
    }

    let range = activeRanges[left];
    let rangeStart = range.prefixEnd - range.meshletCount;
    if (activeIndex < rangeStart || activeIndex >= range.prefixEnd) {
        return 0xffffffffu;
    }

    return range.meshletOffset + (activeIndex - rangeStart);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let meshletIndex = resolve_meshlet_index(in.instance_idx);
    if (meshletIndex == 0xffffffffu) {
        out.position = vec4f(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let meshlet = meshletMetadata[meshletIndex];
    let quadIdx = in.vertex_idx / 6u;
    let triangleVertex = in.vertex_idx % 6u;

    if (quadIdx >= meshlet.quadCount) {
        out.position = vec4f(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let sample = sample_meshlet_quad_vertex(meshlet, quadIdx, triangleVertex);
    let worldSpacePosition = local_to_world_position(sample.worldPosition);
    out.position = world_to_cull_clip_position(worldSpacePosition);
    return out;
}
