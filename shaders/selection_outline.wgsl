// #include "uniforms.wgsl"

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;

struct VSInput {
    @location(0) lineStartAndAlong: vec4f,
    @location(1) lineEndAndSide: vec4f,
    @location(2) color: vec4f,
};

struct VSOutput {
    @builtin(position) clipPosition: vec4f,
    @location(0) color: vec4f,
};

@vertex
fn vs_main(input: VSInput) -> VSOutput {
    var output: VSOutput;

    let lineStart = input.lineStartAndAlong.xyz;
    let lineEnd = input.lineEndAndSide.xyz;
    let along = input.lineStartAndAlong.w;
    let sidePixels = input.lineEndAndSide.w;

    let clipStart = world_to_clip_position(local_to_world_position(lineStart));
    let clipEnd = world_to_clip_position(local_to_world_position(lineEnd));
    let baseClip = mix(clipStart, clipEnd, along);

    let safeStartW = max(abs(clipStart.w), 0.0001);
    let safeEndW = max(abs(clipEnd.w), 0.0001);
    let startNdc = clipStart.xy / safeStartW;
    let endNdc = clipEnd.xy / safeEndW;

    var screenDirPixels = (endNdc - startNdc) * (frameUniforms.viewportParams.xy * 0.5);
    if (dot(screenDirPixels, screenDirPixels) < 0.0001) {
        screenDirPixels = vec2f(1.0, 0.0);
    }

    let perpPixels = normalize(vec2f(-screenDirPixels.y, screenDirPixels.x));
    let offsetNdc = perpPixels * sidePixels * frameUniforms.viewportParams.zw;

    output.clipPosition = vec4f(
        baseClip.xy + offsetNdc * baseClip.w,
        baseClip.z,
        baseClip.w
    );
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VSOutput) -> @location(0) vec4f {
    return input.color;
}
