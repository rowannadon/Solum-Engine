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

fn decodeNormalComponent(value: u32) -> f32 {
    return (f32(value) / 255.0) * 2.0 - 1.0;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let position = vec3f(f32(in.x), f32(in.y), f32(in.z));

    let world_position = frameUniforms.modelMatrix * vec4f(position, 1.0);
    let view_position = frameUniforms.viewMatrix * world_position;
    out.position = frameUniforms.projectionMatrix * view_position;

    out.normal = vec3f(
        decodeNormalComponent(in.n_x),
        decodeNormalComponent(in.n_y),
        decodeNormalComponent(in.n_z)
    );

    out.uv = vec2f(f32(in.u) / 65535.0, f32(in.v) / 65535.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let lightDir = normalize(vec3f(0.45, 0.8, 0.35));
    let ndotl = max(dot(normalize(in.normal), lightDir), 0.0);
    let ambient = 0.3;
    let shade = ambient + (1.0 - ambient) * ndotl;
    let baseColor = vec3f(0.0, 0.4, 1.0);
    return vec4f(baseColor * shade, 1.0);
}
