struct FrameUniforms {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    modelMatrix: mat4x4f,

    inverseProjectionMatrix: mat4x4f,
    inverseViewMatrix: mat4x4f,

    renderFlags: vec4u,
};
