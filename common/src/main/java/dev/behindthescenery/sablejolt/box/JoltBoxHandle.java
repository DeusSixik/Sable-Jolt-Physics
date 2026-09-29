package dev.behindthescenery.sablejolt.box;

import dev.behindthescenery.sablejolt.JoltPhysicsScene;
import dev.ryanhcode.sable.api.physics.object.box.BoxHandle;
import dev.ryanhcode.sable.companion.math.Pose3d;
import dev.ryanhcode.sable.companion.math.Pose3dc;
import org.jetbrains.annotations.ApiStatus;
import org.joml.Quaterniondc;
import org.joml.Vector3dc;

@ApiStatus.Internal
public final class JoltBoxHandle implements BoxHandle {
    private final JoltPhysicsScene scene;
    private final int id;
    private final JoltPhysicsScene.PoseCache cache = new JoltPhysicsScene.PoseCache();

    public static JoltBoxHandle create(final JoltPhysicsScene scene, final Pose3dc pose, final Vector3dc halfExtents, final double mass) {
        final Vector3dc pos = pose.position();
        final Quaterniondc rot = pose.orientation();

        final int id = scene.nextRuntimeId();
        scene.createBox(id, mass, halfExtents.x(), halfExtents.y(), halfExtents.z(), pos, rot);
        return new JoltBoxHandle(scene, id);
    }

    private JoltBoxHandle(final JoltPhysicsScene scene, final int id) {
        this.scene = scene;
        this.id = id;
    }

    /**
     * Queries the pose of the box from the physics engine
     */
    @Override
    public void readPose(final Pose3d dest) {
        this.scene.getPose(this.id, this.cache);

        dest.position().set(this.cache.p1, this.cache.p2, this.cache.p3);
        dest.orientation().set(this.cache.p4, this.cache.p5, this.cache.p6, this.cache.p7);
    }

    /**
     * Removes the box from the physics pipeline
     */
    @Override
    public void remove() {
        this.scene.removeBox(this.id);
    }

    /**
     * Wakes up the box
     */
    @Override
    public void wakeUp() {
        this.scene.wakeUpObject(this.id);
    }

    /**
     * @return the runtime ID of the box
     */
    @Override
    public int getRuntimeId() {
        return this.id;
    }
}
