package dev.behindthescenery.sablejolt.constraint.free;

import com.github.stephengold.joltjni.Body;
import com.github.stephengold.joltjni.MotorSettings;
import com.github.stephengold.joltjni.Quat;
import com.github.stephengold.joltjni.RVec3;
import com.github.stephengold.joltjni.SixDofConstraint;
import com.github.stephengold.joltjni.SixDofConstraintSettings;
import com.github.stephengold.joltjni.Vec3;
import com.github.stephengold.joltjni.enumerate.EAxis;
import com.github.stephengold.joltjni.enumerate.EConstraintSpace;
import com.github.stephengold.joltjni.enumerate.EMotorState;
import dev.behindthescenery.sablejolt.JoltDebugLogging;
import dev.behindthescenery.sablejolt.JoltPhysicsScene;
import dev.behindthescenery.sablejolt.constraint.JoltConstraintHandle;
import dev.behindthescenery.sablejolt.constraint.SixDofMotors;
import dev.behindthescenery.sablejolt.constraint.StaffDebug;
import dev.behindthescenery.sablejolt.constraint.fixed.JoltFixedConstraintHandle;
import dev.ryanhcode.sable.api.physics.PhysicsPipelineBody;
import dev.ryanhcode.sable.api.physics.constraint.FreeConstraintConfiguration;
import dev.ryanhcode.sable.api.physics.constraint.FreeConstraintHandle;
import dev.ryanhcode.sable.sublevel.ServerSubLevel;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.Nullable;
import org.joml.Quaterniond;
import org.joml.Vector3d;

@ApiStatus.Internal
public class JoltFreeConstraintHandle extends JoltConstraintHandle implements FreeConstraintHandle {

    private final SixDofMotors.MotorParams[] motors = new SixDofMotors.MotorParams[SixDofMotors.AXIS_COUNT];
    private com.github.stephengold.joltjni.SixDofConstraintSettings settings;
    private org.joml.Quaterniond frameQuat = new org.joml.Quaterniond();

    /**
     * The world-space target of the world-side anchor in the scene (plot) frame.
     * Only used when {@link #renderFrameServo} is false (the anchor geometry
     * carries the target, motor targets are zero — the iron handle pattern).
     */
    private final Vector3d targetWorld = new Vector3d();

    /**
     * True when the caller drives the motors with per-tick joint-space targets
     * expressed in the render frame (the physics staff pattern: pos1 is the
     * ZERO sentinel). The servo then measures the body anchor through the
     * sub-level's logical pose instead of the constraint geometry.
     */
    private boolean renderFrameServo;

    @Nullable
    private ServerSubLevel subLevel;

    public static JoltFreeConstraintHandle create(final JoltPhysicsScene scene, @Nullable final PhysicsPipelineBody bodyA, @Nullable final PhysicsPipelineBody bodyB, final FreeConstraintConfiguration config, final boolean renderFrameServo) {
        final JoltPhysicsScene.SableBody sbA = bodyA == null ? null : scene.body(bodyA.getRuntimeId());
        final JoltPhysicsScene.SableBody sbB = bodyB == null ? null : scene.body(bodyB.getRuntimeId());
        if (sbA == null && sbB == null) {
            return new JoltFreeConstraintHandle(scene, -1);
        }

        final int joltA = sbA != null ? sbA.joltId : scene.groundBodyId();
        final int joltB = sbB != null ? sbB.joltId : scene.groundBodyId();

        final SixDofConstraintSettings settings = new SixDofConstraintSettings();
        settings.setSpace(EConstraintSpace.LocalToBodyCom);
        // Velocity motors measure their own error externally (in double precision),
        // so the raw anchor positions only matter for the joint frame orientation,
        // not for the servo: keep them small to stay away from float precision loss.
        settings.setPosition1(RVec3.sZero());

        // Sable's contract: pos2 arrives in the plot-global frame while the Jolt
        // scene (and this body) live in the render frame next to the player.
        // Project it with the sub-level's logical pose (the same transform Sable's
        // projectOutOfSubLevel applies), then express it relative to the LIVE body
        // center of mass so the linear servo force produces realistic torque
        // around the grabbed block.
        final ServerSubLevel ssl = bodyB instanceof final ServerSubLevel s ? s
                : bodyA instanceof final ServerSubLevel s2 ? s2 : null;
        final org.joml.Vector3dc p2World = ssl != null
                ? ssl.logicalPose().transformPosition(config.pos2(), new org.joml.Vector3d())
                : config.pos2();
        double p2x = p2World.x();
        double p2y = p2World.y();
        double p2z = p2World.z();
        if (sbB != null) {
            final RVec3 com = sbB.body.getCenterOfMassPosition();
            final Quat inv = sbB.body.getRotation().conjugated();
            final Vec3 off = JoltFixedConstraintHandle.rotate(inv.getX(), inv.getY(), inv.getZ(), inv.getW(),
                    (float) (p2x - com.xx()), (float) (p2y - com.yy()), (float) (p2z - com.zz()));
            p2x = off.getX();
            p2y = off.getY();
            p2z = off.getZ();
        }
        settings.setPosition2(new RVec3((float) p2x, (float) p2y, (float) p2z));

        final var q = config.orientation();
        settings.setAxisX1(JoltFixedConstraintHandle.rotate((float) q.x(), (float) q.y(), (float) q.z(), (float) q.w(), Vec3.sAxisX()));
        settings.setAxisY1(JoltFixedConstraintHandle.rotate((float) q.x(), (float) q.y(), (float) q.z(), (float) q.w(), Vec3.sAxisY()));

        for (final EAxis axis : EAxis.values()) {
            if (axis == EAxis.Num) {
                continue;
            }
            settings.makeFreeAxis(axis);
        }

        final SixDofConstraint constraint = (SixDofConstraint) scene.createConstraint(settings, joltA, joltB);

        final var record = new JoltPhysicsScene.JointRecord();
        record.joltIdA = joltA;
        record.joltIdB = joltB;
        record.constraint = constraint;
        record.contactsEnabled = true;
        record.fixedContacts = true;
        record.genericSettings = settings;
        final long id = scene.registerJoint(record);

        if (JoltDebugLogging.STAFF) {
            StaffDebug.constraintCreated(sbB != null ? sbB.runtimeId : -1,
                    config.pos2().x(), config.pos2().y(), config.pos2().z(),
                    q.x(), q.y(), q.z(), q.w());
        }

        final JoltFreeConstraintHandle handle = new JoltFreeConstraintHandle(scene, id);
        handle.settings = settings;
        handle.frameQuat.set(q.x(), q.y(), q.z(), q.w());
        handle.renderFrameServo = renderFrameServo;
        if (!renderFrameServo) {
            handle.targetWorld.set(config.pos1().x(), config.pos1().y(), config.pos1().z());
        }
        handle.subLevel = ssl;
        return handle;
    }

    private JoltFreeConstraintHandle(final JoltPhysicsScene scene, final long handle) {
        super(scene);
        this.attach(handle);
    }

    @Override
    public void remove() {
        // The angular servo can leave significant residual spin (it drives the
        // body towards up to MAX_ANGULAR_SPEED while following the grab). On
        // release that reads as the object wildly spinning for several turns.
        // Damp it here — both for the per-tick constraint recreation and for the
        // final release.
        try {
            final JoltPhysicsScene.JointRecord record = this.record();
            if (record != null && record.constraint instanceof final SixDofConstraint constraint) {
                final Body body = constraint.getBody2();
                if (body != null && body.isDynamic()) {
                    // Fully stop the residual spin: the servo re-drives omega within
                    // the same tick during the per-tick constraint recreation, so
                    // only the final release ever keeps the zeroed value.
                    body.setAngularVelocity(0.0f, 0.0f, 0.0f);
                }
            }
        } catch (final Throwable ignored) {
            // never block removal on diagnostics
        }
        super.remove();
    }

    @Override
    protected void setMotorImpl(final int axisOrdinal, final double target, final double stiffness, final double damping, final boolean hasForceLimit, final double maxForce) {
        final JoltPhysicsScene.JointRecord record = this.record();
        if (record == null || !(record.constraint instanceof final SixDofConstraint constraint)) {
            return;
        }

        this.motors[axisOrdinal] = new SixDofMotors.MotorParams(target, stiffness, damping, hasForceLimit, maxForce);
        SixDofMotors.applyMotorFree(constraint, this.settings, axisOrdinal, this.motors, this.scene,
                record.joltIdA, record.joltIdB, this.frameQuat, this.targetWorld, this.renderFrameServo, this.subLevel);
    }
}
