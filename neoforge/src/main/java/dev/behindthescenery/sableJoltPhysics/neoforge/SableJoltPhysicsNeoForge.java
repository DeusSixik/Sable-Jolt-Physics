package dev.behindthescenery.sableJoltPhysics.neoforge;

import dev.behindthescenery.sableJoltPhysics.SableJoltPhysics;
import net.neoforged.fml.common.Mod;

@Mod(SableJoltPhysics.MOD_ID)
public final class SableJoltPhysicsNeoForge {
    public SableJoltPhysicsNeoForge() {
        // Run our common setup.
        SableJoltPhysics.init();
    }
}
