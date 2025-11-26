struct FrameUniforms {
    projectionMatrix: mat4x4f,
    infiniteProjectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    modelMatrix: mat4x4f,

    inverseProjectionMatrix: mat4x4f,
    inverseViewMatrix: mat4x4f,

    lightViewMatrix: mat4x4f,
    lightProjectionMatrix: mat4x4f,

    lightDirection: vec3f,
    transparent: u32,

    highlightedVoxelPos: vec3i,
    time: f32,

    cameraWorldPos: vec3f,
    padding2: f32,

    lightPosition: vec3f,
    padding1: u32,

    screenSize: vec2f,
	padding3: f32,
	padding4: f32,

    cameraOffset: vec3f,
	padding5: f32,
};