// #include "uniforms.wgsl"

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;

struct VSInput {
    @location(0) position: vec3f,
    @location(1) color: vec4f,
};

struct VSOutput {
    @builtin(position) clipPosition: vec4f,
    @location(0) color: vec4f,
};

@vertex
fn vs_main(input: VSInput) -> VSOutput {
    var output: VSOutput;
    let worldPosition = frameUniforms.modelMatrix * vec4f(input.position, 1.0);
    output.clipPosition = frameUniforms.projectionMatrix * frameUniforms.viewMatrix * worldPosition;
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VSOutput) -> @location(0) vec4f {
    return input.color;
}
