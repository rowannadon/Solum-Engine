// #include "uniforms.wgsl"

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read> meshletDataWords: array<u32>;

struct MeshletMetadata {
    originX: i32,
    originY: i32,
    originZ: i32,
    quadCount: u32,
    faceDirection: u32,
    dataOffset: u32,
    voxelScale: u32,
    pad1: u32,
};

@group(0) @binding(2) var<storage, read> meshletMetadata: array<MeshletMetadata>;
@group(0) @binding(3) var<storage, read> materialToTexture: array<u32, 65536>;
@group(0) @binding(4) var materialTextures: texture_2d_array<f32>;
@group(0) @binding(5) var materialSampler: sampler;

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

fn fetch_quad_data(quadOffset: u32) -> u32 {
    return meshletDataWords[quadOffset];
}

fn decode_local_offset(packed: u32) -> vec3u {
    let offset = packed & 0xffffu;
    return vec3u(
        offset & 0x1fu,
        (offset >> 5u) & 0x1fu,
        (offset >> 10u) & 0x1fu
    );
}

fn decode_material_id(packed: u32) -> u32 {
    return (packed >> 16u) & 0xffffu;
}

fn decode_flip(packedAoData: u32) -> bool {
    return ((packedAoData >> 8u) & 0x1u) != 0u;
}

fn decode_vertex_ao(packedAoData: u32, corner: u32) -> u32 {
    let shift = corner * 2u;
    return (packedAoData >> shift) & 0x3u;
}

fn corner_from_triangle_vertex(triangleVertex: u32, flipped: bool) -> u32 {
    if (!flipped) {
        // Unflipped: [0,1,2] and [2,1,3].
        switch triangleVertex {
            case 0u: { return 0u; }
            case 1u: { return 1u; }
            case 2u: { return 2u; }
            case 3u: { return 2u; }
            case 4u: { return 1u; }
            default: { return 3u; }
        }
    }

    // Flipped: [0,1,3] and [0,3,2].
    switch triangleVertex {
        case 0u: { return 0u; }
        case 1u: { return 1u; }
        case 2u: { return 3u; }
        case 3u: { return 0u; }
        case 4u: { return 3u; }
        default: { return 2u; }
    }
}

fn face_corner_offset(face: u32, corner: u32) -> vec3f {
    switch face {
        case 0u: {
            switch corner {
                case 0u: { return vec3f(1.0, 0.0, 0.0); }
                case 1u: { return vec3f(1.0, 1.0, 0.0); }
                case 2u: { return vec3f(1.0, 0.0, 1.0); }
                default: { return vec3f(1.0, 1.0, 1.0); }
            }
        }
        case 1u: {
            switch corner {
                case 0u: { return vec3f(0.0, 0.0, 0.0); }
                case 1u: { return vec3f(0.0, 0.0, 1.0); }
                case 2u: { return vec3f(0.0, 1.0, 0.0); }
                default: { return vec3f(0.0, 1.0, 1.0); }
            }
        }
        case 2u: {
            switch corner {
                case 0u: { return vec3f(0.0, 1.0, 0.0); }
                case 1u: { return vec3f(0.0, 1.0, 1.0); }
                case 2u: { return vec3f(1.0, 1.0, 0.0); }
                default: { return vec3f(1.0, 1.0, 1.0); }
            }
        }
        case 3u: {
            switch corner {
                case 0u: { return vec3f(0.0, 0.0, 0.0); }
                case 1u: { return vec3f(1.0, 0.0, 0.0); }
                case 2u: { return vec3f(0.0, 0.0, 1.0); }
                default: { return vec3f(1.0, 0.0, 1.0); }
            }
        }
        case 4u: {
            switch corner {
                case 0u: { return vec3f(0.0, 0.0, 1.0); }
                case 1u: { return vec3f(1.0, 0.0, 1.0); }
                case 2u: { return vec3f(0.0, 1.0, 1.0); }
                default: { return vec3f(1.0, 1.0, 1.0); }
            }
        }
        default: {
            switch corner {
                case 0u: { return vec3f(0.0, 0.0, 0.0); }
                case 1u: { return vec3f(0.0, 1.0, 0.0); }
                case 2u: { return vec3f(1.0, 0.0, 0.0); }
                default: { return vec3f(1.0, 1.0, 0.0); }
            }
        }
    }
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

    let meshlet = meshletMetadata[in.instance_idx];
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

    let quadDataOffset = meshlet.dataOffset + (quadIdx * 2u);
    let quadData = fetch_quad_data(quadDataOffset);
    let quadAoData = fetch_quad_data(quadDataOffset + 1u);
    let blockLocal = decode_local_offset(quadData);
    let corner = corner_from_triangle_vertex(triangleVertex, decode_flip(quadAoData));
    let cornerOffset = face_corner_offset(meshlet.faceDirection, corner);
    let voxelScale = f32(max(meshlet.voxelScale, 1u));

    let meshletOrigin = vec3f(f32(meshlet.originX), f32(meshlet.originY), f32(meshlet.originZ));
    let worldPosition =
        meshletOrigin +
        (vec3f(f32(blockLocal.x), f32(blockLocal.y), f32(blockLocal.z)) + cornerOffset) * voxelScale;

    let worldSpacePosition = frameUniforms.modelMatrix * vec4f(worldPosition, 1.0);
    let viewPosition = frameUniforms.viewMatrix * worldSpacePosition;
    out.position = frameUniforms.projectionMatrix * viewPosition;

    out.worldPosition = worldSpacePosition.xyz;
    out.texCoord = face_uv(meshlet.faceDirection, cornerOffset);
    out.materialId = decode_material_id(quadData);
    out.ao = f32(decode_vertex_ao(quadAoData, corner)) / 3.0;

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
