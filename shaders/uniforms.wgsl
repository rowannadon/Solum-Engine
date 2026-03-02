struct FrameUniforms {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    modelMatrix: mat4x4f,

    inverseProjectionMatrix: mat4x4f,
    inverseViewMatrix: mat4x4f,
    cullingViewMatrix: mat4x4f,
    inverseCullingViewMatrix: mat4x4f,

    renderFlags: vec4u,
    occlusionParams: vec4f,
};

fn local_to_world_position(localPosition: vec3f) -> vec4f {
    return frameUniforms.modelMatrix * vec4f(localPosition, 1.0);
}

fn world_to_clip_position(worldPosition: vec4f) -> vec4f {
    return frameUniforms.projectionMatrix * frameUniforms.viewMatrix * worldPosition;
}

fn world_to_cull_clip_position(worldPosition: vec4f) -> vec4f {
    return frameUniforms.projectionMatrix * frameUniforms.cullingViewMatrix * worldPosition;
}

fn clip_from_local_matrix() -> mat4x4f {
    return frameUniforms.projectionMatrix * frameUniforms.viewMatrix * frameUniforms.modelMatrix;
}

fn cull_clip_from_local_matrix() -> mat4x4f {
    return frameUniforms.projectionMatrix * frameUniforms.cullingViewMatrix * frameUniforms.modelMatrix;
}
