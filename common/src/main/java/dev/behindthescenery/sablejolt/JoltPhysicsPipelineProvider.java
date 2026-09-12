package dev.behindthescenery.sablejolt;

import dev.ryanhcode.sable.api.physics.PhysicsPipeline;
import dev.ryanhcode.sable.api.physics.PhysicsPipelineProvider;
import net.minecraft.server.level.ServerLevel;
import org.jetbrains.annotations.NotNull;

/**
 * Registered via ServiceLoader with a priority above the rapier provider (default 1000),
 * so Sable picks this provider directly; the mixin into
 * {@link dev.ryanhcode.sable.physics.impl.rapier.RapierPhysicsPipelineProvider} acts
 * as a fallback that redirects any rapier selection to Jolt as well.
 */
@PhysicsPipelineProvider.LoadPriority(2000)
public final class JoltPhysicsPipelineProvider implements PhysicsPipelineProvider {

    @Override
    public @NotNull PhysicsPipeline createPipeline(@NotNull final ServerLevel level) {
        return new JoltPhysicsPipeline(level);
    }
}

