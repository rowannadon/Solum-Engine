// #include "uniforms.wgsl"

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;

struct MeshletMetadata {
    originAndLod: vec4i,
    baseAndCounts: vec4u,
};

struct PackedVertex {
    xy: u32,
    zMaterial: u32,
    packedFlags: u32,
};

@group(0) @binding(1) var<storage, read> meshletMetadata: array<MeshletMetadata>;
@group(0) @binding(2) var<storage, read> meshletVertices: array<PackedVertex>;
@group(0) @binding(3) var<storage, read> meshletIndices: array<u32>;

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) uv: vec2f,
    @location(2) color: vec3f,
    @location(3) world_pos: vec3f,
    @location(4) @interpolate(flat) lod_level: u32,
    @location(5) @interpolate(flat) meshlet_id: u32,
};

fn hash_u32(x: u32) -> u32 {
    var h = x;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}

fn hash_to_color(id: u32) -> vec3f {
    let h1 = hash_u32(id);
    let h2 = hash_u32(id ^ 0x9e3779b9u);
    let h3 = hash_u32(id ^ 0x85ebca6bu);

    return vec3f(
        f32(h1 & 0xffu) / 255.0,
        f32(h2 & 0xffu) / 255.0,
        f32(h3 & 0xffu) / 255.0
    );
}

fn hash_i2_to_color(ix: i32, iy: i32) -> vec3f {
    let ux = bitcast<u32>(ix);
    let uy = bitcast<u32>(iy);
    let packed = (ux * 0x9e3779b9u) ^ (uy * 0x85ebca6bu);
    return hash_to_color(packed);
}

fn lod_debug_color(lod_level: u32) -> vec3f {
    switch lod_level {
        case 0u: { return vec3f(0.95, 0.15, 0.15); }
        case 1u: { return vec3f(0.95, 0.55, 0.15); }
        case 2u: { return vec3f(0.95, 0.9, 0.15); }
        case 3u: { return vec3f(0.2, 0.85, 0.25); }
        case 4u: { return vec3f(0.2, 0.45, 0.95); }
        default: { return vec3f(0.8, 0.2, 0.85); }
    }
}

fn decodeNormal(index: u32) -> vec3f {
    switch index {
        case 0u: { return vec3f(1.0, 0.0, 0.0); }
        case 1u: { return vec3f(-1.0, 0.0, 0.0); }
        case 2u: { return vec3f(0.0, 1.0, 0.0); }
        case 3u: { return vec3f(0.0, -1.0, 0.0); }
        case 4u: { return vec3f(0.0, 0.0, 1.0); }
        default: { return vec3f(0.0, 0.0, -1.0); }
    }
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let meshletInfo = meshletMetadata[in.instance_idx];
    let localIndex = meshletIndices[meshletInfo.baseAndCounts.y + in.vertex_idx];
    let packed = meshletVertices[meshletInfo.baseAndCounts.x + localIndex];

    let relX = i32(packed.xy & 0xffffu);
    let relY = i32((packed.xy >> 16u) & 0xffffu);
    let relZ = i32(packed.zMaterial & 0xffffu);
    let material = (packed.zMaterial >> 16u) & 0xffffu;

    let flags = packed.packedFlags;
    let uFlag = flags & 0x1u;
    let vFlag = (flags >> 1u) & 0x1u;
    let normalIdx = (flags >> 2u) & 0x7u;

    let position = vec3f(
        f32(meshletInfo.originAndLod.x + relX),
        f32(meshletInfo.originAndLod.y + relY),
        f32(meshletInfo.originAndLod.z + relZ)
    );

    let world_position = frameUniforms.modelMatrix * vec4f(position, 1.0);
    let view_position = frameUniforms.viewMatrix * world_position;
    out.position = frameUniforms.projectionMatrix * view_position;

    out.normal = decodeNormal(normalIdx);
    out.uv = vec2f(f32(uFlag), f32(vFlag));
    out.color = hash_to_color(material);
    out.world_pos = position;
    out.lod_level = u32(meshletInfo.originAndLod.w);
    out.meshlet_id = in.instance_idx;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let debug_flags = frameUniforms.debugParams.x;

    var debugColor = vec3f(0.0, 0.0, 0.0);
    var debugContribs: f32 = 0.0;

    if ((debug_flags & 0x1u) != 0u) {
        let regionX = i32(floor(in.world_pos.x / 512.0));
        let regionY = i32(floor(in.world_pos.y / 512.0));
        debugColor += hash_i2_to_color(regionX, regionY);
        debugContribs += 1.0;
    }

    if ((debug_flags & 0x2u) != 0u) {
        debugColor += lod_debug_color(in.lod_level);
        debugContribs += 1.0;
    }

    if ((debug_flags & 0x4u) != 0u) {
        let chunkX = i32(floor(in.world_pos.x / 32.0));
        let chunkY = i32(floor(in.world_pos.y / 32.0));
        debugColor += hash_i2_to_color(chunkX, chunkY);
        debugContribs += 1.0;
    }

    if ((debug_flags & 0x8u) != 0u) {
        debugColor += hash_to_color(in.meshlet_id);
        debugContribs += 1.0;
    }

    let baseColor = select(in.color, debugColor / max(debugContribs, 1.0), debugContribs > 0.0);

    let lightDir = normalize(vec3f(0.45, 1.0, 0.35));
    let ndotl = dot(normalize(in.normal), lightDir);
    let ambient = 0.4;
    let shade = ambient + (1.0 - ambient) * ndotl;

    let linearColor = baseColor * shade;

    return vec4f(linearColor, 1.0);
}
