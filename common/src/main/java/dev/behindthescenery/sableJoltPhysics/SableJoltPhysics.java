package dev.behindthescenery.sableJoltPhysics;

import dev.behindthescenery.sablejolt.SableJoltConfig;

public final class SableJoltPhysics {
    public static final String MOD_ID = "sable_jolt_physics";

    public static void init() {
        // Creates the config file on first launch. The natives are loaded lazily
        // when the first physics world is created, so clients that only join
        // servers never extract or load them.
        SableJoltConfig.get();
    }
}
