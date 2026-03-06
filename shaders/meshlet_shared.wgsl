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

struct ModelQuad {
    vertexPositions: array<vec4f, 4>,
    uvs: array<vec2f, 4>,
    aoValues: array<f32, 4>,
    normal: vec4f,
};

struct MeshletQuadVertexSample {
    worldPosition: vec3f,
    texCoord: vec2f,
    cornerOffset: vec3f,
    quadData: u32,
    quadAoData: u32,
    corner: u32,
    useVoxelAo: bool,
};

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

fn decode_model_quad_index(packedAuxData: u32) -> u32 {
    return (packedAuxData >> 9u) & 0x3fffffu;
}

fn decode_use_voxel_ao(packedAuxData: u32) -> bool {
    return ((packedAuxData >> 31u) & 0x1u) != 0u;
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

fn sample_meshlet_quad_vertex(
    meshlet: MeshletMetadata,
    quadIdx: u32,
    triangleVertex: u32
) -> MeshletQuadVertexSample {
    let quadDataOffset = meshlet.dataOffset + (quadIdx * 2u);
    let quadData = fetch_quad_data(quadDataOffset);
    let quadAoData = fetch_quad_data(quadDataOffset + 1u);
    let modelQuadIndex = decode_model_quad_index(quadAoData);
    let blockLocal = decode_local_offset(quadData);
    let corner = corner_from_triangle_vertex(triangleVertex, decode_flip(quadAoData));
    let voxelScale = f32(max(meshlet.voxelScale, 1u));
    let meshletOrigin = vec3f(f32(meshlet.originX), f32(meshlet.originY), f32(meshlet.originZ));
    let blockBase = vec3f(f32(blockLocal.x), f32(blockLocal.y), f32(blockLocal.z));
    let useVoxelAo = decode_use_voxel_ao(quadAoData);
    var cornerOffset = vec3f(0.0, 0.0, 0.0);
    var texCoord = vec2f(0.0, 0.0);
    var worldPosition = vec3f(0.0, 0.0, 0.0);

    if (useVoxelAo) {
        cornerOffset = face_corner_offset(meshlet.faceDirection, corner);
        worldPosition = meshletOrigin + (blockBase + cornerOffset) * voxelScale;
    } else {
        let modelQuad = modelQuads[modelQuadIndex];
        cornerOffset = modelQuad.vertexPositions[corner].xyz;
        texCoord = modelQuad.uvs[corner];
        worldPosition = meshletOrigin + (blockBase + cornerOffset) * voxelScale;
    }

    var sample: MeshletQuadVertexSample;
    sample.worldPosition = worldPosition;
    sample.texCoord = texCoord;
    sample.cornerOffset = cornerOffset;
    sample.quadData = quadData;
    sample.quadAoData = quadAoData;
    sample.corner = corner;
    sample.useVoxelAo = useVoxelAo;
    return sample;
}
