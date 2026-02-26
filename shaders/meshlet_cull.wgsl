struct FrameUniforms {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    modelMatrix: mat4x4f,
    inverseProjectionMatrix: mat4x4f,
    inverseViewMatrix: mat4x4f,
    renderFlags: vec4u,
};

struct MeshletAabb {
    minCorner: vec4f,
    maxCorner: vec4f,
};

struct CullParams {
    meshletCount: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read> meshletAabbs: array<MeshletAabb>;
@group(0) @binding(2) var<storage, read_write> visibleMeshletIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> drawArgsWords: array<atomic<u32>, 4>;
@group(0) @binding(4) var<uniform> cullParams: CullParams;

const kCullEpsilon: f32 = 0.0001;
var<workgroup> clipFromLocalWg: mat4x4f;

fn corner_position(minCorner: vec3f, maxCorner: vec3f, index: u32) -> vec3f {
    switch index {
        case 0u: { return vec3f(minCorner.x, minCorner.y, minCorner.z); }
        case 1u: { return vec3f(maxCorner.x, minCorner.y, minCorner.z); }
        case 2u: { return vec3f(maxCorner.x, maxCorner.y, minCorner.z); }
        case 3u: { return vec3f(minCorner.x, maxCorner.y, minCorner.z); }
        case 4u: { return vec3f(minCorner.x, minCorner.y, maxCorner.z); }
        case 5u: { return vec3f(maxCorner.x, minCorner.y, maxCorner.z); }
        case 6u: { return vec3f(maxCorner.x, maxCorner.y, maxCorner.z); }
        default: { return vec3f(minCorner.x, maxCorner.y, maxCorner.z); }
    }
}

fn is_visible(aabb: MeshletAabb, clipFromLocal: mat4x4f) -> bool {
    let minCorner = aabb.minCorner.xyz;
    let maxCorner = aabb.maxCorner.xyz;

    var allOutsideLeft = true;
    var allOutsideRight = true;
    var allOutsideBottom = true;
    var allOutsideTop = true;
    var allOutsideNear = true;
    var allOutsideFar = true;

    for (var cornerIndex: u32 = 0u; cornerIndex < 8u; cornerIndex = cornerIndex + 1u) {
        let corner = corner_position(minCorner, maxCorner, cornerIndex);
        let clip = clipFromLocal * vec4f(corner, 1.0);

        allOutsideLeft = allOutsideLeft && ((clip.x + clip.w) < -kCullEpsilon);
        allOutsideRight = allOutsideRight && ((clip.w - clip.x) < -kCullEpsilon);
        allOutsideBottom = allOutsideBottom && ((clip.y + clip.w) < -kCullEpsilon);
        allOutsideTop = allOutsideTop && ((clip.w - clip.y) < -kCullEpsilon);
        allOutsideNear = allOutsideNear && (clip.z < -kCullEpsilon);
        allOutsideFar = allOutsideFar && ((clip.w - clip.z) < -kCullEpsilon);

        if (!allOutsideLeft &&
            !allOutsideRight &&
            !allOutsideBottom &&
            !allOutsideTop &&
            !allOutsideNear &&
            !allOutsideFar) {
            return true;
        }
    }

    return !(allOutsideLeft ||
             allOutsideRight ||
             allOutsideBottom ||
             allOutsideTop ||
             allOutsideNear ||
             allOutsideFar);
}

@compute @workgroup_size(128, 1, 1)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_index) localIndex: u32
) {
    if (localIndex == 0u) {
        clipFromLocalWg = frameUniforms.projectionMatrix * frameUniforms.viewMatrix * frameUniforms.modelMatrix;
    }
    workgroupBarrier();

    let meshletIndex = gid.x;
    if (meshletIndex >= cullParams.meshletCount) {
        return;
    }

    if (!is_visible(meshletAabbs[meshletIndex], clipFromLocalWg)) {
        return;
    }

    let visibleIndex = atomicAdd(&drawArgsWords[1], 1u);
    visibleMeshletIndices[visibleIndex] = meshletIndex;
}
