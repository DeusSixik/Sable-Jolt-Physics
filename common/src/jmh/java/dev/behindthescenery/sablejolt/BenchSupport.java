package dev.behindthescenery.sablejolt;

import dev.behindthescenery.sablejolt.collider.JoltVoxelColliderData;

import java.lang.reflect.Method;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Shared scaffolding for the scene benchmarks. Headless: the stub
 * {@code SableLoaderPlatformImpl} on the run classpath points the native dir
 * at a temp folder, so {@link JoltNative#ensureInitialized()} works without a
 * game instance. Everything here uses only APIs present both before and after
 * the optimization pass, so the same harness compiles against either tree
 * (version-specific paths go through reflection with graceful fallback).
 */
public final class BenchSupport {
    private static final AtomicBoolean BOOT = new AtomicBoolean();

    private BenchSupport() {
    }

    public static void boot() {
        if (BOOT.compareAndSet(false, true)) {
            JoltNative.ensureInitialized();
        }
    }

    public static JoltPhysicsScene newScene() {
        boot();
        return new JoltPhysicsScene(0.0, -20.0, 0.0, 0.05);
    }

    /**
     * Registers one full-cube solid entry and one fluid entry in the scene
     * registry. Returns {@code [solidId, fluidId]} as collider ids.
     */
    public static int[] registerEntries(final JoltPhysicsScene scene) {
        final JoltVoxelColliderData.Registry reg = scene.colliderRegistry;
        final JoltVoxelColliderData solid =
                reg.create(1.0, 1.0, 0.0, false, null, null);
        solid.addBox(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        final JoltVoxelColliderData fluid =
                reg.create(1.0, 1.0, 0.0, true, null, null);
        fluid.addBox(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        return new int[]{reg.indexOf(solid) + 1, reg.indexOf(fluid) + 1};
    }

    /**
     * One 16x16x16 section: lower half solid cubes, upper half fluid.
     * Works both as body geometry (solids) and as fluid marking.
     */
    public static int[] mixedSection(final int solidId, final int fluidId) {
        final int[] data = new int[ChunkSectionData.BLOCKS];
        final int solidPacked = (solidId << 16) | 1;
        final int fluidPacked = (fluidId << 16) | (9 << 8) | 1;
        for (int y = 0; y < 16; y++) {
            final int packed = y < 8 ? solidPacked : fluidPacked;
            final int rowBase = y << 8;
            for (int z = 0; z < 16; z++) {
                final int cellBase = rowBase + (z << 4);
                for (int x = 0; x < 16; x++) {
                    data[cellBase + x] = packed;
                }
            }
        }
        return data;
    }

    /** Fully solid 16x16x16 section. */
    public static int[] solidSection(final int solidId) {
        final int[] data = new int[ChunkSectionData.BLOCKS];
        final int packed = (solidId << 16) | 1;
        java.util.Arrays.fill(data, packed);
        return data;
    }

    /** Reflective lookup that returns null (instead of throwing) when absent. */
    public static Method method(final Class<?> owner, final String name,
                                final Class<?>... params) {
        try {
            final Method m = owner.getDeclaredMethod(name, params);
            m.setAccessible(true);
            return m;
        } catch (final NoSuchMethodException e) {
            return null;
        }
    }
}
