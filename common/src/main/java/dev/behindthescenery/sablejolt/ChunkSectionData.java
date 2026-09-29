package dev.behindthescenery.sablejolt;

/**
 * A 16x16x16 voxel chunk section. Blocks are stored as packed ints in xzy order
 * (x fastest changing): {@code (colliderId << 16) | (fluidLevel << 8) | voxelStateId},
 * matching the wire format used by the Sable pipeline.
 */
public final class ChunkSectionData {
    public static final int SIZE = 16;
    public static final int BLOCKS = SIZE * SIZE * SIZE;

    private final int[] blocks = new int[BLOCKS];

    public static int index(final int x, final int y, final int z) {
        return x + (z << 4) + (y << 8);
    }

    public int get(final int x, final int y, final int z) {
        return this.blocks[index(x, y, z)];
    }

    public void set(final int x, final int y, final int z, final int packed) {
        this.blocks[index(x, y, z)] = packed;
    }

    public int[] array() {
        return this.blocks;
    }

    public int colliderId(final int x, final int y, final int z) {
        return this.blocks[index(x, y, z)] >>> 16;
    }

    public int voxelState(final int x, final int y, final int z) {
        return this.blocks[index(x, y, z)] & 0xFF;
    }

    public int fluidLevel(final int x, final int y, final int z) {
        return fluidLevelOf(this.blocks[index(x, y, z)]);
    }

    public static int colliderIdOf(final int packed) {
        return packed >>> 16;
    }

    public static int voxelStateOf(final int packed) {
        return packed & 0xFF;
    }

    /** Fluid fill amount as uploaded from the pipeline (0 = no fluid, 1..8 = Minecraft fluid level). */
    public static int fluidLevelOf(final int packed) {
        return (packed >>> 8) & 0xF;
    }

    public static long packSectionPos(final int x, final int y, final int z) {
        long l = 0L;
        l |= (x & 0x3FFFFFL) << 42;
        l |= y & 0xFFFFFL;
        l |= (z & 0x3FFFFFL) << 20;
        return l;
    }
}
