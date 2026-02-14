// #include "uniforms.wgsl"

@group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;

struct VertexInput {
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32,
    @location(0) x: i32,
    @location(1) y: i32,
    @location(2) z: i32,
    @location(3) u: u32,
    @location(4) v: u32,
    @location(5) material: u32,
    @location(6) n_x: u32,
    @location(7) n_y: u32,
    @location(8) n_z: u32,
    @location(9) lod_level: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) uv: vec2f,
    @location(2) color: vec3f,
    @location(3) world_pos: vec3f,
    @location(4) @interpolate(flat) lod_level: u32,
};

fn decodeNormalComponent(value: u32) -> f32 {
    return (f32(value) / 255.0) * 2.0 - 1.0;
}

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
        case 0u: { return vec3f(0.95, 0.15, 0.15); } // red
        case 1u: { return vec3f(0.95, 0.55, 0.15); } // orange
        case 2u: { return vec3f(0.95, 0.9, 0.15); } // yellow
        case 3u: { return vec3f(0.2, 0.85, 0.25); } // green
        case 4u: { return vec3f(0.2, 0.45, 0.95); } // blue
        default: { return vec3f(0.8, 0.2, 0.85); }
    }
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let position = vec3f(f32(in.x), f32(in.y), f32(in.z));

    let world_position = frameUniforms.modelMatrix * vec4f(position, 1.0);
    let view_position = frameUniforms.viewMatrix * world_position;
    out.position = frameUniforms.projectionMatrix * view_position;

    out.normal = vec3f(
        decodeNormalComponent(in.n_x),
        decodeNormalComponent(in.n_y),
        decodeNormalComponent(in.n_z)
    );

    out.uv = vec2f(f32(in.u) / 65535.0, f32(in.v) / 65535.0);

    out.color = hash_to_color(in.material);
    out.world_pos = position;
    out.lod_level = in.lod_level;

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

    let baseColor = select(in.color, debugColor / max(debugContribs, 1.0), debugContribs > 0.0);

    let lightDir = normalize(vec3f(0.45, 1.0, 0.35));
    let ndotl = dot(normalize(in.normal), lightDir);
    let ambient = 0.4;
    let shade = ambient + (1.0 - ambient) * ndotl;

    let linearColor = baseColor * shade;

    return vec4f(linearColor, 1.0);
}
