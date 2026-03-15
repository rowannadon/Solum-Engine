struct ResidentTileLod {
    meshletStart: u32,
    meshletCount: u32,
    quadWordStart: u32,
    flags: u32,
    minCorner: vec4f,
    maxCorner: vec4f,
};

struct TileSlot {
    selectedResidentSlot: u32,
    visible: u32,
    flags: u32,
    pad0: u32,
    minCorner: vec4f,
    maxCorner: vec4f,
};

struct TileSceneParams {
    visibleTileCount: u32,
    tileSlotCount: u32,
    residentTileCount: u32,
    totalVisibleMeshlets: u32,
};

@group(0) @binding(0) var<storage, read> visibleTileIds: array<u32>;
@group(0) @binding(1) var<storage, read> tileSlots: array<TileSlot>;
@group(0) @binding(2) var<storage, read> residentTileLods: array<ResidentTileLod>;
@group(0) @binding(3) var<storage, read_write> visibleMeshletIndices: array<u32>;
@group(0) @binding(4) var<storage, read_write> drawArgsWords: array<atomic<u32>, 4>;
@group(0) @binding(5) var<uniform> sceneParams: TileSceneParams;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let visibleTileIndex = gid.x;
    if (visibleTileIndex >= sceneParams.visibleTileCount) {
        return;
    }

    let tileSlotIndex = visibleTileIds[visibleTileIndex];
    if (tileSlotIndex >= sceneParams.tileSlotCount) {
        return;
    }

    let tileSlot = tileSlots[tileSlotIndex];
    if (tileSlot.visible == 0u || (tileSlot.flags & 0x1u) == 0u) {
        return;
    }

    let residentSlot = tileSlot.selectedResidentSlot;
    if (residentSlot >= sceneParams.residentTileCount) {
        return;
    }

    let resident = residentTileLods[residentSlot];
    if ((resident.flags & 0x1u) == 0u || resident.meshletCount == 0u) {
        return;
    }

    let firstVisibleIndex = atomicAdd(&drawArgsWords[1], resident.meshletCount);
    for (var i: u32 = 0u; i < resident.meshletCount; i = i + 1u) {
        visibleMeshletIndices[firstVisibleIndex + i] = resident.meshletStart + i;
    }
}
