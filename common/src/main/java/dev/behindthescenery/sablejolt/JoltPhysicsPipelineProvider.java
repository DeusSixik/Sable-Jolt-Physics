package dev.behindthescenery.sablejolt;

import dev.ryanhcode.sable.Sable;
import dev.ryanhcode.sable.api.physics.PhysicsPipeline;
import dev.ryanhcode.sable.api.physics.PhysicsPipelineProvider;
import dev.ryanhcode.sable.physics.impl.rapier.RapierPhysicsPipeline;
import net.minecraft.server.level.ServerLevel;
import org.jetbrains.annotations.NotNull;

/**
 * Registered via ServiceLoader with a priority above the rapier provider (default 1000),
 * so Sable picks this provider directly; the mixin into
 * {@link dev.ryanhcode.sable.physics.impl.rapier.RapierPhysicsPipelineProvider} acts
 * as a fallback that redirects any rapier selection to Jolt as well.
 * <p>
 * Both paths go through {@link #useJolt()}: the Rapier pipeline is returned when the
 * config selects it or when the Jolt natives cannot be loaded on this system. Sable
 * persists sub-levels in an engine-independent format, so either engine can load
 * worlds saved by the other.
 */
@PhysicsPipelineProvider.LoadPriority(2000)
public final class JoltPhysicsPipelineProvider implements PhysicsPipelineProvider {

    private static volatile boolean announced;

    @Override
    public @NotNull PhysicsPipeline createPipeline(@NotNull final ServerLevel level) {
        if (useJolt()) {
            return new JoltPhysicsPipeline(level);
        }
        return new RapierPhysicsPipeline(level);
    }

    /**
     * Whether new pipelines should use Jolt: enabled in the config and the
     * natives load on this system.
     */
    public static boolean useJolt() {
        final SableJoltConfig config = SableJoltConfig.get();
        final boolean jolt = config.backend == SableJoltConfig.Backend.JOLT && JoltNative.tryInitialize();
        if (!announced) {
            announced = true;
            if (jolt) {
                Sable.LOGGER.info("[SableJolt] Using the Jolt physics backend (rapier_compatible_physics={})",
                        config.rapierCompatiblePhysics);
            } else if (config.backend == SableJoltConfig.Backend.RAPIER) {
                Sable.LOGGER.info("[SableJolt] Using Sable's Rapier physics backend (backend=rapier in config)");
            } else {
                Sable.LOGGER.warn("[SableJolt] Falling back to Sable's Rapier physics backend");
            }
        }
        return jolt;
    }
}
