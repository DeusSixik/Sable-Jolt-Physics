package dev.behindthescenery.sableJoltPhysics;

import dev.behindthescenery.sablejolt.JoltNative;

public final class SableJoltPhysics {
    public static final String MOD_ID = "sable_jolt_physics";

    public static void init() {
        JoltNative.ensureInitialized();
    }
}
