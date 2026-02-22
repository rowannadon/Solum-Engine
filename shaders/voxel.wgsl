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

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) worldPosition: vec3f,
    @location(1) color: vec3f,
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

fn fetch_packed_quad(quadOffset: u32) -> u32 {
    let word = meshletDataWords[quadOffset >> 1u];
    if ((quadOffset & 1u) == 0u) {
        return word & 0xffffu;
    }
    return (word >> 16u) & 0xffffu;
}

fn decode_local_offset(packed: u32) -> vec3u {
    return vec3u(
        packed & 0x1fu,
        (packed >> 5u) & 0x1fu,
        (packed >> 10u) & 0x1fu
    );
}

fn corner_from_triangle_vertex(face: u32, triangleVertex: u32) -> u32 {
    switch face {
        case 0u: {
            switch triangleVertex {
                case 0u: { return 0u; }
                case 1u: { return 1u; }
                case 2u: { return 2u; }
                case 3u: { return 1u; }
                case 4u: { return 3u; }
                default: { return 2u; }
            }
        }
        case 1u: {
            switch triangleVertex {
                case 0u: { return 0u; }
                case 1u: { return 2u; }
                case 2u: { return 1u; }
                case 3u: { return 1u; }
                case 4u: { return 2u; }
                default: { return 3u; }
            }
        }
        case 2u: {
            switch triangleVertex {
                case 0u: { return 0u; }
                case 1u: { return 2u; }
                case 2u: { return 1u; }
                case 3u: { return 1u; }
                case 4u: { return 2u; }
                default: { return 3u; }
            }
        }
        case 3u: {
            switch triangleVertex {
                case 0u: { return 0u; }
                case 1u: { return 1u; }
                case 2u: { return 2u; }
                case 3u: { return 1u; }
                case 4u: { return 3u; }
                default: { return 2u; }
            }
        }
        case 4u: {
            switch triangleVertex {
                case 0u: { return 0u; }
                case 1u: { return 1u; }
                case 2u: { return 2u; }
                case 3u: { return 1u; }
                case 4u: { return 3u; }
                default: { return 2u; }
            }
        }
        default: {
            switch triangleVertex {
                case 0u: { return 0u; }
                case 1u: { return 2u; }
                case 2u: { return 1u; }
                case 3u: { return 1u; }
                case 4u: { return 3u; }
                default: { return 2u; }
            }
        }
    }
}

fn face_corner_offset(face: u32, corner: u32) -> vec3f {
    switch face {
        case 0u: {
            switch corner {
                case 0u: { return vec3f(1.0, 0.0, 0.0); }
                case 1u: { return vec3f(1.0, 0.0, 1.0); }
                case 2u: { return vec3f(1.0, 1.0, 0.0); }
                default: { return vec3f(1.0, 1.0, 1.0); }
            }
        }
        case 1u: {
            switch corner {
                case 0u: { return vec3f(0.0, 0.0, 0.0); }
                case 1u: { return vec3f(0.0, 1.0, 0.0); }
                case 2u: { return vec3f(0.0, 0.0, 1.0); }
                default: { return vec3f(0.0, 1.0, 1.0); }
            }
        }
        case 2u: {
            switch corner {
                case 0u: { return vec3f(0.0, 1.0, 0.0); }
                case 1u: { return vec3f(1.0, 1.0, 0.0); }
                case 2u: { return vec3f(0.0, 1.0, 1.0); }
                default: { return vec3f(1.0, 1.0, 1.0); }
            }
        }
        case 3u: {
            switch corner {
                case 0u: { return vec3f(0.0, 0.0, 0.0); }
                case 1u: { return vec3f(0.0, 0.0, 1.0); }
                case 2u: { return vec3f(1.0, 0.0, 0.0); }
                default: { return vec3f(1.0, 0.0, 1.0); }
            }
        }
        case 4u: {
            switch corner {
                case 0u: { return vec3f(0.0, 0.0, 1.0); }
                case 1u: { return vec3f(0.0, 1.0, 1.0); }
                case 2u: { return vec3f(1.0, 0.0, 1.0); }
                default: { return vec3f(1.0, 1.0, 1.0); }
            }
        }
        default: {
            switch corner {
                case 0u: { return vec3f(0.0, 0.0, 0.0); }
                case 1u: { return vec3f(1.0, 0.0, 0.0); }
                case 2u: { return vec3f(0.0, 1.0, 0.0); }
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
        out.worldPosition = vec3f(0.0, 0.0, 0.0);
        out.color = vec3f(0.0, 0.0, 0.0);
        return out;
    }

    let packedOffset = fetch_packed_quad(meshlet.dataOffset + quadIdx);
    let blockLocal = decode_local_offset(packedOffset);
    let corner = corner_from_triangle_vertex(meshlet.faceDirection, triangleVertex);
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

    let worldBlockX = meshlet.originX + i32(blockLocal.x) * i32(max(meshlet.voxelScale, 1u));
    let worldBlockY = meshlet.originY + i32(blockLocal.y) * i32(max(meshlet.voxelScale, 1u));
    let worldBlockZ = meshlet.originZ + i32(blockLocal.z) * i32(max(meshlet.voxelScale, 1u));

    let colorSeed = (bitcast<u32>(worldBlockX) * 73856093u) ^
        (bitcast<u32>(worldBlockY) * 19349663u) ^
        (bitcast<u32>(worldBlockZ) * 83492791u);
    let meshletDebugEnabled = (frameUniforms.renderFlags.x & 0x1u) != 0u;
    let meshletColorSeed = (bitcast<u32>(meshlet.originX) * 73856093u) ^
        (bitcast<u32>(meshlet.originY) * 19349663u) ^
        (bitcast<u32>(meshlet.originZ) * 83492791u) ^
        (meshlet.faceDirection * 2654435761u);
    if (meshletDebugEnabled) {
        out.color = hash_to_color(meshletColorSeed);
    } else {
        out.color = hash_to_color(colorSeed);
    }

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let dx = dpdx(in.worldPosition);
    let dy = dpdy(in.worldPosition);
    let normal = normalize(cross(dx, dy));

    let lightDir = normalize(vec3f(0.45, 0.6, 0.35));
    let ndotl = abs(dot(normal, lightDir));
    let ambient = 0.3;
    let shade = ambient + (1.0 - ambient) * ndotl;

    let linearColor = in.color * shade;
    return vec4f(linearColor, 1.0);
}
