@group(0) @binding(0) var depthTex: texture_depth_2d;
@group(0) @binding(1) var hizMip0: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(hizMip0);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let depth = textureLoad(depthTex, vec2i(gid.xy), 0);
    textureStore(hizMip0, vec2i(gid.xy), vec4f(depth, 0.0, 0.0, 1.0));
}
