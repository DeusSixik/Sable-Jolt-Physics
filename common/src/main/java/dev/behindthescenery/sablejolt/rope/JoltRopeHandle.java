package dev.behindthescenery.sablejolt.rope;

import dev.behindthescenery.sablejolt.JoltPhysicsScene;
import dev.ryanhcode.sable.api.physics.object.rope.RopeHandle;
import dev.ryanhcode.sable.sublevel.ServerSubLevel;
import org.jetbrains.annotations.ApiStatus;
import org.joml.Vector3d;
import org.joml.Vector3dc;

import java.util.List;

@ApiStatus.Internal
public record JoltRopeHandle(JoltPhysicsScene scene, long handle) implements RopeHandle {

    public static JoltRopeHandle create(final JoltPhysicsScene scene, final double pointRadius, final List<Vector3d> points) {
        final double[] coordinates = new double[points.size() * 3];

        for (int i = 0; i < points.size(); i++) {
            final Vector3d point = points.get(i);
            coordinates[i * 3] = point.x;
            coordinates[i * 3 + 1] = point.y;
            coordinates[i * 3 + 2] = point.z;
        }

        final long handle = scene.createRope(pointRadius, points.get(0).distance(points.get(1)), coordinates, points.size());
        return new JoltRopeHandle(scene, handle);
    }

    /**
     * Queries the points of the rope from the physics engine
     */
    @Override
    public void readPose(final List<Vector3d> dest) {
        final double[] coordinates = this.scene.queryRope(this.handle);
        for (int i = 0; i < coordinates.length; i += 3) {
            dest.get(i / 3).set(coordinates[i], coordinates[i + 1], coordinates[i + 2]);
        }
    }

    /**
     * Removes the rope from the physics pipeline
     */
    @Override
    public void remove() {
        this.scene.removeRope(this.handle);
    }

    /**
     * Sets the extension constraint length of the first segment
     */
    @Override
    public void setFirstSegmentLength(final double length) {
        this.scene.setRopeFirstSegmentLength(this.handle, length);
    }

    /**
     * Removes the point at the beginning of the rope
     */
    @Override
    public void removeFirstPoint() {
        this.scene.removeRopePointAtStart(this.handle);
    }

    /**
     * Adds a point to the beginning of the rope
     */
    @Override
    public void addPoint(final Vector3dc position) {
        this.scene.addRopePointAtStart(this.handle, position.x(), position.y(), position.z());
    }

    /**
     * Sets an attachment
     */
    @Override
    public void setAttachment(final AttachmentPoint attachmentPoint, final Vector3dc location, final ServerSubLevel subLevel) {
        this.scene.setRopeAttachment(this.handle, subLevel == null ? -1 : subLevel.getRuntimeId(), location.x(), location.y(), location.z(), attachmentPoint == AttachmentPoint.END);
    }

    /**
     * Wakes up the rope
     */
    @Override
    public void wakeUp() {
        this.scene.wakeUpRope(this.handle);
    }
}
