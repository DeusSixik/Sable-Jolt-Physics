package dev.behindthescenery.sableJoltPhysics.mixin;

import dev.behindthescenery.sablejolt.JoltPhysicsPipeline;
import dev.ryanhcode.sable.api.physics.PhysicsPipeline;
import dev.ryanhcode.sable.physics.impl.rapier.RapierPhysicsPipelineProvider;
import net.minecraft.server.level.ServerLevel;
import org.jetbrains.annotations.NotNull;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Overwrite;

/**
 * Redirects the original sable_rapier pipeline provider to the Jolt implementation,
 * so that Sable always ends up using {@link JoltPhysicsPipeline} even when the
 * rapier provider wins pipeline selection.
 */
@Mixin(value = RapierPhysicsPipelineProvider.class, remap = false)
public class RapierPhysicsPipelineProviderMixin {

    /**
     * @author BehindTheScenery
     * @reason Replace the Rapier physics pipeline with the Jolt physics pipeline
     */
    @Overwrite(remap = false)
    public @NotNull PhysicsPipeline createPipeline(@NotNull final ServerLevel level) {
        return new JoltPhysicsPipeline(level);
    }
}
