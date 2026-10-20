package dev.behindthescenery.sableJoltPhysics.mixin;

import dev.behindthescenery.sablejolt.JoltPhysicsPipeline;
import dev.behindthescenery.sablejolt.JoltPhysicsPipelineProvider;
import dev.ryanhcode.sable.api.physics.PhysicsPipeline;
import dev.ryanhcode.sable.physics.impl.rapier.RapierPhysicsPipelineProvider;
import net.minecraft.server.level.ServerLevel;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Redirects the original sable_rapier pipeline provider to the Jolt implementation,
 * so that Sable ends up using {@link JoltPhysicsPipeline} even when the rapier
 * provider wins pipeline selection. When Jolt is disabled in the config or cannot
 * load on this system, the original Rapier pipeline is left untouched.
 */
@Mixin(value = RapierPhysicsPipelineProvider.class, remap = false)
public class RapierPhysicsPipelineProviderMixin {

    @Inject(method = "createPipeline", at = @At("HEAD"), cancellable = true, remap = false, require = 0)
    private void sablejolt$createJoltPipeline(final ServerLevel level, final CallbackInfoReturnable<PhysicsPipeline> cir) {
        if (JoltPhysicsPipelineProvider.useJolt()) {
            cir.setReturnValue(new JoltPhysicsPipeline(level));
        }
    }
}
