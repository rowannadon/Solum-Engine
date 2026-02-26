struct FrameUniforms {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    modelMatrix: mat4x4f,
    inverseProjectionMatrix: mat4x4f,
    inverseViewMatrix: mat4x4f,
    renderFlags: vec4u,
    occlusionParams: vec4f,
};

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

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read> meshletDataWords: array<u32>;
@group(0) @binding(2) var<storage, read> meshletMetadata: array<MeshletMetadata>;

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
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

fn decode_flip(packedAoData: u32) -> bool {
    return ((packedAoData >> 8u) & 0x1u) != 0u;
}

fn corner_from_triangle_vertex(triangleVertex: u32, flipped: bool) -> u32 {
    if (!flipped) {
        switch triangleVertex {
            case 0u: { return 0u; }
            case 1u: { return 1u; }
            case 2u: { return 2u; }
            case 3u: { return 2u; }
            case 4u: { return 1u; }
            default: { return 3u; }
        }
    }

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

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let meshlet = meshletMetadata[in.instance_idx];
    let quadIdx = in.vertex_idx / 6u;
    let triangleVertex = in.vertex_idx % 6u;

    if (quadIdx >= meshlet.quadCount) {
        out.position = vec4f(2.0, 2.0, 2.0, 1.0);
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
    return out;
}
