// #include "uniforms.wgsl"
// #include "meshlet_shared.wgsl"

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read> meshletDataWords: array<u32>;
@group(0) @binding(2) var<storage, read> meshletMetadata: array<MeshletMetadata>;

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let meshlet = meshletMetadata[in.instance_idx];
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
