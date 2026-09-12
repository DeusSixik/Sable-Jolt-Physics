package dev.behindthescenery.sablejolt.constraint;

import dev.behindthescenery.sablejolt.JoltPhysicsScene;
import dev.ryanhcode.sable.api.physics.constraint.ConstraintJointAxis;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.Nullable;
import org.joml.Vector3d;

/**
 * Base class of all Jolt constraint handles.
 */
@ApiStatus.Internal
public abstract class JoltConstraintHandle {
    protected final JoltPhysicsScene scene;
    protected long handle = -1;

    protected JoltConstraintHandle(final JoltPhysicsScene scene) {
        this.scene = scene;
    }

    protected void attach(final long handle) {
        this.handle = handle;
    }

    public void setContactsEnabled(final boolean enabled) {
        final JoltPhysicsScene.JointRecord record = this.record();
        if (record != null) {
            record.contactsEnabled = enabled;
        }
    }

    public void getJointImpulses(final Vector3d linearImpulseDest, final Vector3d angularImpulseDest) {
        // jolt-jni does not expose accumulated constraint lambdas; report zero impulses
        linearImpulseDest.set(0.0, 0.0, 0.0);
        angularImpulseDest.set(0.0, 0.0, 0.0);
    }

    public void setMotor(final ConstraintJointAxis axis, final double target, final double stiffness, final double damping, final boolean hasForceLimit, final double maxForce) {
        this.setMotorImpl(axis.ordinal(), target, stiffness, damping, hasForceLimit, maxForce);
    }

    protected abstract void setMotorImpl(int axisOrdinal, double target, double stiffness, double damping, boolean hasForceLimit, double maxForce);

    public void remove() {
        if (this.handle != -1) {
            this.scene.removeJoint(this.handle);
            this.handle = -1;
        }
    }

    public boolean isValid() {
        return this.handle != -1 && this.scene.joint(this.handle) != null && this.scene.joint(this.handle).constraint != null;
    }

    protected void assertValid() {
        if (!this.isValid()) {
            throw new RuntimeException("Attempted to mutate an invalid constraint");
        }
    }

    @Nullable
    protected JoltPhysicsScene.JointRecord record() {
        return this.handle == -1 ? null : this.scene.joint(this.handle);
    }
}
