// #include "uniforms.wgsl"
// #include "meshlet_shared.wgsl"

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read> meshletDataWords: array<u32>;
@group(0) @binding(2) var<storage, read> meshletMetadata: array<MeshletMetadata>;
struct MaterialMetadata {
    textureLayer: u32,
    flags: u32,
    randomOffsetAmount: f32,
    pad0: f32,
};

@group(0) @binding(3) var<storage, read> materialMetadata: array<MaterialMetadata, 65536>;
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
    @location(5) @interpolate(flat) useVoxelAo: u32,
    @location(6) @interpolate(flat) blockCoord: vec3i,
    @location(7) @interpolate(flat) packedLight: u32,
    @location(8) @interpolate(flat) isSeamMeshlet: u32,
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

fn luminance(color: vec3f) -> f32 {
    return dot(color, vec3f(0.2126, 0.7152, 0.0722));
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

fn hash_block_coord(blockCoord: vec3i) -> u32 {
    let x = bitcast<u32>(blockCoord.x);
    let y = bitcast<u32>(blockCoord.y);
    let z = bitcast<u32>(blockCoord.z);
    let seed = (x * 73856093u) ^ (y * 19349663u) ^ (z * 83492791u);
    return hash_u32(seed);
}

fn rotate_uv_local(localUv: vec2f, rotation: u32) -> vec2f {
    let rot = rotation & 0x3u;
    if (rot == 1u) {
        return vec2f(localUv.y, 1.0 - localUv.x);
    }
    if (rot == 2u) {
        return vec2f(1.0 - localUv.x, 1.0 - localUv.y);
    }
    if (rot == 3u) {
        return vec2f(1.0 - localUv.y, localUv.x);
    }
    return localUv;
}

fn hash_to_signed_unit(seed: u32) -> f32 {
    let h = hash_u32(seed);
    let normalized = f32(h & 0xffffu) / 65535.0;
    return (normalized * 2.0) - 1.0;
}

fn random_offset_for_block(blockCoord: vec3i, axisMask: u32, amount: f32) -> vec3f {
    if (axisMask == 0u || amount <= 0.0) {
        return vec3f(0.0, 0.0, 0.0);
    }

    let baseHash = hash_block_coord(blockCoord);
    var offset = vec3f(0.0, 0.0, 0.0);
    if ((axisMask & 0x1u) != 0u) {
        offset.x = hash_to_signed_unit(baseHash ^ 0x68bc21ebu) * amount;
    }
    if ((axisMask & 0x2u) != 0u) {
        offset.y = hash_to_signed_unit(baseHash ^ 0x02e5be93u) * amount;
    }
    if ((axisMask & 0x4u) != 0u) {
        offset.z = hash_to_signed_unit(baseHash ^ 0x967a889bu) * amount;
    }
    return offset;
}

fn rotate_x(v: vec3f, angle: f32) -> vec3f {
    let c = cos(angle);
    let s = sin(angle);
    return vec3f(v.x, (v.y * c) - (v.z * s), (v.y * s) + (v.z * c));
}

fn rotate_y(v: vec3f, angle: f32) -> vec3f {
    let c = cos(angle);
    let s = sin(angle);
    return vec3f((v.x * c) + (v.z * s), v.y, (-v.x * s) + (v.z * c));
}

fn rotate_z(v: vec3f, angle: f32) -> vec3f {
    let c = cos(angle);
    let s = sin(angle);
    return vec3f((v.x * c) - (v.y * s), (v.x * s) + (v.y * c), v.z);
}

fn hash_to_angle(seed: u32) -> f32 {
    let h = hash_u32(seed);
    return (f32(h) / 4294967295.0) * 6.28318530718;
}

fn random_model_rotation_for_block(
    blockCoord: vec3i,
    enabled: bool,
    axisMask: u32,
    blockSize: f32,
    localPosition: vec3f
) -> vec3f {
    if (!enabled || axisMask == 0u) {
        return localPosition;
    }

    let center = vec3f(blockCoord) + vec3f(0.5 * blockSize);
    var rotated = localPosition - center;
    let baseHash = hash_block_coord(blockCoord);
    if ((axisMask & 0x1u) != 0u) {
        rotated = rotate_x(rotated, hash_to_angle(baseHash ^ 0x51633e2du));
    }
    if ((axisMask & 0x2u) != 0u) {
        rotated = rotate_y(rotated, hash_to_angle(baseHash ^ 0x68bc21ebu));
    }
    if ((axisMask & 0x4u) != 0u) {
        rotated = rotate_z(rotated, hash_to_angle(baseHash ^ 0x02e5be93u));
    }
    return rotated + center;
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
        out.useVoxelAo = 0u;
        out.blockCoord = vec3i(0, 0, 0);
        out.packedLight = 0u;
        out.isSeamMeshlet = 0u;
        return out;
    }

    let sample = sample_meshlet_quad_vertex(meshlet, quadIdx, triangleVertex);
    let isSeamMeshlet = (meshlet.flags & 0x1u) != 0u;

    let decodedMaterialId = decode_material_id(sample.quadData);
    let safeMaterialId = min(decodedMaterialId, 65535u);
    let material = materialMetadata[safeMaterialId];
    let flags = material.flags;
    let offsetDirectionMask = (flags >> 1u) & 0x7u;
    let modelRotationEnabled = ((flags & 0x10u) != 0u) && !isSeamMeshlet;
    let modelRotationDirectionMask = (flags >> 5u) & 0x7u;
    let offsetAmount = select(clamp(material.randomOffsetAmount, 0.0, 1.0), 0.0, isSeamMeshlet);
    let offset = random_offset_for_block(sample.blockCoord, offsetDirectionMask, offsetAmount);
    var localPosition = sample.worldPosition + offset;
    localPosition = random_model_rotation_for_block(
        sample.blockCoord,
        modelRotationEnabled,
        modelRotationDirectionMask,
        sample.blockSize,
        localPosition
    );

    let worldSpacePosition = local_to_world_position(localPosition);
    out.position = world_to_clip_position(worldSpacePosition);

    out.worldPosition = worldSpacePosition.xyz;
    if (sample.useVoxelAo) {
        out.texCoord = face_uv(meshlet.faceDirection, sample.cornerOffset);
    } else {
        out.texCoord = sample.texCoord;
    }
    out.materialId = decodedMaterialId;
    if (sample.useVoxelAo) {
        out.ao = f32(decode_vertex_ao(sample.quadAoData, sample.corner)) / 3.0;
        out.useVoxelAo = 1u;
    } else {
        out.ao = 1.0;
        out.useVoxelAo = 0u;
    }
    out.blockCoord = sample.blockCoord;
    out.packedLight = sample.packedLightData;
    out.isSeamMeshlet = select(0u, 1u, isSeamMeshlet);

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
    let dayPhase = fract(frameUniforms.timeParams.x);
    let sunHeight = sin(dayPhase * 6.28318530718);
    let daylight = smoothstep(-0.18, 0.12, sunHeight);
    let lightDir = normalize(vec3f(1.0, 0.5, 1.0));
    let ndotl = abs(dot(normal, lightDir));
    let aoShade = mix(0.25, 1.0, clamp(in.ao, 0.0, 1.0));
    let ambient = 0.3;
    let shade = (ambient * aoShade) + (1.0 - ambient) * ndotl;

    let meshletDebugEnabled = (frameUniforms.renderFlags.x & 0x1u) != 0u;
    var baseColor = vec3f(0.5, 0.5, 0.5);
    if (meshletDebugEnabled) {
        baseColor = in.debugColor;
    } else {
        let safeMaterialId = min(in.materialId, 65535u);
        let material = materialMetadata[safeMaterialId];
        let flags = material.flags;
        let textureLayer = material.textureLayer;
        var sampleUv = in.texCoord;
        if ((flags & 0x1u) != 0u && in.useVoxelAo != 0u && in.isSeamMeshlet == 0u) {
            let rotation = hash_block_coord(in.blockCoord) & 0x3u;
            let tileUv = floor(sampleUv);
            let localUv = fract(sampleUv);
            sampleUv = tileUv + rotate_uv_local(localUv, rotation);
        }
        let albedo = textureSample(materialTextures, materialSampler, sampleUv, i32(textureLayer));
        if (albedo.a == 0.0) {
            discard;
        }
        baseColor = albedo.rgb;
    }

    let frontPackedLight = in.packedLight & 0xffu;
    let backPackedLight = (in.packedLight >> 8u) & 0xffu;
    let isDoubleSidedMaterial = (materialMetadata[min(in.materialId, 65535u)].flags & 0x100u) != 0u;
    let skyFront = decode_sky_light(frontPackedLight);
    let blockFront = decode_block_light(frontPackedLight);
    let skyBack = decode_sky_light(backPackedLight);
    let blockBack = decode_block_light(backPackedLight);
    let chosenSky = select(skyFront, max(skyFront, skyBack), isDoubleSidedMaterial);
    let chosenBlock = select(blockFront, max(blockFront, blockBack), isDoubleSidedMaterial);
    let skyLight = f32(chosenSky) / 15.0;
    let blockLight = f32(chosenBlock) / 15.0;
    let skyLightColor = mix(vec3f(0.55, 0.6, 0.72), vec3f(1.0, 0.98, 0.95), daylight);
    let blockLightColor = vec3f(1.0, 0.92, 0.62);
    let skyContribution = skyLight * mix(0.45, 1.0, daylight);
    let skyRadiance = skyLightColor * skyContribution;
    let skyLuma = luminance(skyRadiance);
    let blockStrength = pow(blockLight, 0.85);
    let blockVisibility = 1.0 - smoothstep(0.08, 0.72, skyLuma);
    let blockExposure = mix(1.35, 0.3, smoothstep(0.0, 1.0, skyLuma));
    let blockRadiance = blockLightColor * blockStrength * blockExposure * blockVisibility;
    let combinedRadiance = max(vec3f(0.0, 0.0, 0.0), skyRadiance + blockRadiance);
    let totalLightColor = vec3f(1.0, 1.0, 1.0) - exp(-combinedRadiance * 1.65);
    let shaded = max(0.45, shade);

    let linearColor = baseColor * shaded * totalLightColor;
    return vec4f(linearColor, 1.0);
}
