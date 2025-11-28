// #include "uniforms.wgsl"

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32,
    @location(0) x: u32,
    @location(1) y: u32,
    @location(2) z: u32,
    @location(3) u: u32,
    @location(4) v: u32,
    @location(5) material: u32,
    @location(6) n_x: u32,
    @location(7) n_y: u32,
    @location(8) n_z: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) uv: vec2f,
};

struct FragmentInput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) uv: vec2f,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let position = vec3f(f32(in.x), f32(in.y), f32(in.z));

    let world_position = frameUniforms.modelMatrix * vec4f(position, 1.0);
    let view_position = frameUniforms.viewMatrix * world_position;
    out.position = frameUniforms.projectionMatrix * view_position;

    return out;
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4f {
    return vec4f(0.0, 0.4, 1.0, 1.0);
}