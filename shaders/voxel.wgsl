// #include "uniforms.wgsl"
// #include "meshlet_shared.wgsl"

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read> meshletDataWords: array<u32>;
@group(0) @binding(2) var<storage, read> meshletMetadata: array<MeshletMetadata>;
@group(0) @binding(3) var<storage, read> materialToTexture: array<u32, 65536>;
@group(0) @binding(4) var<storage, read> visibleMeshletIndices: array<u32>;
@group(0) @binding(5) var materialTextures: texture_2d_array<f32>;
@group(0) @binding(6) var<storage, read> modelQuads: array<ModelQuad>;
@group(0) @binding(7) var materialSampler: sampler;

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) worldPosition: vec3f,
    @location(1) texCoord: vec2f,
    @location(2) @interpolate(flat) materialId: u32,
    @location(3) debugColor: vec3f,
    @location(4) ao: f32,
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

fn face_uv(face: u32, cornerOffset: vec3f) -> vec2f {
    if (face == 0u || face == 1u) {
        return vec2f(cornerOffset.y, cornerOffset.z);
    }
    if (face == 2u || face == 3u) {
        return vec2f(cornerOffset.x, cornerOffset.z);
    }
    return vec2f(cornerOffset.x, cornerOffset.y);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let meshletIndex = visibleMeshletIndices[in.instance_idx];
    let meshlet = meshletMetadata[meshletIndex];
    let quadIdx = in.vertex_idx / 6u;
    let triangleVertex = in.vertex_idx % 6u;

    if (quadIdx >= meshlet.quadCount) {
        out.position = vec4f(2.0, 2.0, 2.0, 1.0);
        out.worldPosition = vec3f(0.0, 0.0, 0.0);
        out.texCoord = vec2f(0.0, 0.0);
        out.materialId = 0u;
        out.debugColor = vec3f(0.0, 0.0, 0.0);
        out.ao = 1.0;
        return out;
    }

    let sample = sample_meshlet_quad_vertex(meshlet, quadIdx, triangleVertex);

    let worldSpacePosition = local_to_world_position(sample.worldPosition);
    out.position = world_to_clip_position(worldSpacePosition);

    out.worldPosition = worldSpacePosition.xyz;
    if (sample.useVoxelAo) {
        out.texCoord = face_uv(meshlet.faceDirection, sample.cornerOffset);
    } else {
        out.texCoord = sample.texCoord;
    }
    out.materialId = decode_material_id(sample.quadData);
    if (sample.useVoxelAo) {
        out.ao = f32(decode_vertex_ao(sample.quadAoData, sample.corner)) / 3.0;
    } else {
        out.ao = 1.0;
    }

    let meshletColorSeed = (bitcast<u32>(meshlet.originX) * 73856093u) ^
        (bitcast<u32>(meshlet.originY) * 19349663u) ^
        (bitcast<u32>(meshlet.originZ) * 83492791u) ^
        (meshlet.faceDirection * 2654435761u);
    out.debugColor = hash_to_color(meshletColorSeed);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let dx = dpdx(in.worldPosition);
    let dy = dpdy(in.worldPosition);
    let normal = normalize(cross(dx, dy));

    let lightDir = normalize(vec3f(1.0, 0.5, 1.0));
    let ndotl = abs(dot(normal, lightDir));
    let ambient = 0.3;
    let aoShade = mix(0.25, 1.0, clamp(in.ao, 0.0, 1.0));
    let shade = (ambient * aoShade) + (1.0 - ambient) * ndotl;

    let meshletDebugEnabled = (frameUniforms.renderFlags.x & 0x1u) != 0u;
    var baseColor = vec3f(0.5, 0.5, 0.5);
    if (meshletDebugEnabled) {
        baseColor = in.debugColor;
    } else {
        let safeMaterialId = min(in.materialId, 65535u);
        let textureLayer = materialToTexture[safeMaterialId];
        baseColor = textureSample(materialTextures, materialSampler, in.texCoord, i32(textureLayer)).rgb;
    }

    let linearColor = baseColor * shade;
    return vec4f(linearColor, 1.0);
}
