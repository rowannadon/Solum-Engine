@group(0) @binding(0) var srcMip: texture_2d<f32>;
@group(0) @binding(1) var dstMip: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let dstDims = textureDimensions(dstMip);
    if (gid.x >= dstDims.x || gid.y >= dstDims.y) {
        return;
    }

    let srcDims = textureDimensions(srcMip, 0u);
    let base = vec2i(gid.xy * 2u);
    let maxX = i32(max(srcDims.x, 1u) - 1u);
    let maxY = i32(max(srcDims.y, 1u) - 1u);

    let p00 = vec2i(clamp(base.x + 0, 0, maxX), clamp(base.y + 0, 0, maxY));
    let p10 = vec2i(clamp(base.x + 1, 0, maxX), clamp(base.y + 0, 0, maxY));
    let p01 = vec2i(clamp(base.x + 0, 0, maxX), clamp(base.y + 1, 0, maxY));
    let p11 = vec2i(clamp(base.x + 1, 0, maxX), clamp(base.y + 1, 0, maxY));

    let d00 = textureLoad(srcMip, p00, 0u).x;
    let d10 = textureLoad(srcMip, p10, 0u).x;
    let d01 = textureLoad(srcMip, p01, 0u).x;
    let d11 = textureLoad(srcMip, p11, 0u).x;

    let maxDepth = max(max(d00, d10), max(d01, d11));
    textureStore(dstMip, vec2i(gid.xy), vec4f(maxDepth, 0.0, 0.0, 1.0));
}
