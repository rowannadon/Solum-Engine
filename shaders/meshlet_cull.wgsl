// #include "uniforms.wgsl"

struct MeshletAabb {
    minCorner: vec4f,
    maxCorner: vec4f,
};

struct ActiveMeshletRange {
    meshletOffset: u32,
    meshletCount: u32,
    prefixEnd: u32,
    pad: u32,
};

struct CullParams {
    totalMeshlets: u32,
    hizMipCount: u32,
    activeTileCount: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read> meshletAabbs: array<MeshletAabb>;
@group(0) @binding(2) var<storage, read_write> visibleMeshletIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> drawArgsWords: array<atomic<u32>, 4>;
@group(0) @binding(4) var<uniform> cullParams: CullParams;
@group(0) @binding(5) var occlusionHiZTex: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> activeRanges: array<ActiveMeshletRange>;

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

fn uv_to_texel(uv: vec2f, dims: vec2u) -> vec2i {
    let maxX = f32(max(dims.x, 1u) - 1u);
    let maxY = f32(max(dims.y, 1u) - 1u);
    let x = i32(clamp(uv.x * f32(dims.x), 0.0, maxX));
    let y = i32(clamp(uv.y * f32(dims.y), 0.0, maxY));
    return vec2i(x, y);
}

fn is_occluded(aabb: MeshletAabb, clipFromLocal: mat4x4f) -> bool {
    if (frameUniforms.occlusionParams.x < 0.5) {
        return false;
    }

    let dims = textureDimensions(occlusionHiZTex, 0u);
    if (dims.x == 0u || dims.y == 0u) {
        return false;
    }

    let minCorner = aabb.minCorner.xyz;
    let maxCorner = aabb.maxCorner.xyz;
    let cameraPosition = frameUniforms.inverseCullingViewMatrix[3].xyz;
    let nearSkipDistance = max(frameUniforms.occlusionParams.z, 0.0);
    let nearSkipDistanceSq = nearSkipDistance * nearSkipDistance;
    let closestPoint = clamp(cameraPosition, minCorner, maxCorner);
    let toCamera = closestPoint - cameraPosition;
    if (dot(toCamera, toCamera) <= nearSkipDistanceSq) {
        return false;
    }

    var minUv = vec2f(1.0, 1.0);
    var maxUv = vec2f(0.0, 0.0);
    var nearestDepth = 1.0;
    var hasProjectedCorner = false;

    for (var cornerIndex: u32 = 0u; cornerIndex < 8u; cornerIndex = cornerIndex + 1u) {
        let corner = corner_position(minCorner, maxCorner, cornerIndex);
        let clip = clipFromLocal * vec4f(corner, 1.0);
        if (clip.w <= kCullEpsilon) {
            continue;
        }

        let ndc = clip.xyz / clip.w;
        let uv = vec2f(ndc.x * 0.5 + 0.5, (-ndc.y) * 0.5 + 0.5);
        minUv = min(minUv, uv);
        maxUv = max(maxUv, uv);
        let depthZo = ndc.z;
        let depthNo = ndc.z * 0.5 + 0.5;
        nearestDepth = min(nearestDepth, clamp(min(depthZo, depthNo), 0.0, 1.0));
        hasProjectedCorner = true;
    }

    if (!hasProjectedCorner) {
        return false;
    }

    let uvMin = clamp(minUv, vec2f(0.0, 0.0), vec2f(1.0, 1.0));
    let uvMax = clamp(maxUv, vec2f(0.0, 0.0), vec2f(1.0, 1.0));
    let uvSpanPx = (uvMax - uvMin) * vec2f(f32(dims.x), f32(dims.y));
    let minSpanPx = max(frameUniforms.occlusionParams.w, 0.0);

    if (uvSpanPx.x <= minSpanPx || uvSpanPx.y <= minSpanPx) {
        return false;
    }

    let maxSpanPx = max(uvSpanPx.x, uvSpanPx.y);
    let mipCount = max(cullParams.hizMipCount, 1u);
    let desiredMip = i32(ceil(log2(max(maxSpanPx, 1.0))));
    let mip = u32(clamp(desiredMip, 0, i32(mipCount) - 1));
    let mipDims = textureDimensions(occlusionHiZTex, mip);
    if (mipDims.x == 0u || mipDims.y == 0u) {
        return false;
    }

    let minTexel = uv_to_texel(uvMin, mipDims);
    let maxTexel = uv_to_texel(uvMax, mipDims);
    let x0 = min(minTexel.x, maxTexel.x);
    let x1 = max(minTexel.x, maxTexel.x);
    let y0 = min(minTexel.y, maxTexel.y);
    let y1 = max(minTexel.y, maxTexel.y);

    var hizMaxDepth = 0.0;
    for (var y = y0; y <= y1; y = y + 1) {
        for (var x = x0; x <= x1; x = x + 1) {
            hizMaxDepth = max(hizMaxDepth, textureLoad(occlusionHiZTex, vec2i(x, y), mip).x);
        }
    }

    let depthBias = max(frameUniforms.occlusionParams.y, 0.0);
    let maybeVisible = nearestDepth <= (hizMaxDepth + depthBias);
    return !maybeVisible;
}

// Tile-dispatch model: one workgroup per active tile.
// Each workgroup reads its tile's meshlet range from activeRanges and iterates
// over the meshlets in a strided pattern across threads. This eliminates the
// O(log n) binary search per thread from the old per-meshlet dispatch.
@compute @workgroup_size(128, 1, 1)
fn cs_main(
    @builtin(workgroup_id) wgId: vec3u,
    @builtin(local_invocation_index) lid: u32
) {
    let tileIndex = wgId.x;
    if (tileIndex >= cullParams.activeTileCount) {
        return;
    }

    if (lid == 0u) {
        clipFromLocalWg = cull_clip_from_local_matrix();
    }
    workgroupBarrier();

    let tile = activeRanges[tileIndex];
    let meshletOffset = tile.meshletOffset;
    let meshletCount = tile.meshletCount;

    for (var i = lid; i < meshletCount; i += 128u) {
        let meshletIndex = meshletOffset + i;

        if (!is_visible(meshletAabbs[meshletIndex], clipFromLocalWg)) {
            continue;
        }
        if (is_occluded(meshletAabbs[meshletIndex], clipFromLocalWg)) {
            continue;
        }

        let visibleIndex = atomicAdd(&drawArgsWords[1], 1u);
        visibleMeshletIndices[visibleIndex] = meshletIndex;
    }
}
