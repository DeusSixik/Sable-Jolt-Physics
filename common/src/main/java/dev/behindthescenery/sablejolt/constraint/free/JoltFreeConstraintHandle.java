package dev.behindthescenery.sablejolt.constraint.free;

import com.github.stephengold.joltjni.Quat;
import com.github.stephengold.joltjni.RVec3;
import com.github.stephengold.joltjni.SixDofConstraint;
import com.github.stephengold.joltjni.SixDofConstraintSettings;
import com.github.stephengold.joltjni.Vec3;
import com.github.stephengold.joltjni.enumerate.EAxis;
import com.github.stephengold.joltjni.enumerate.EConstraintSpace;
import dev.behindthescenery.sablejolt.JoltDebugLogging;
import dev.behindthescenery.sablejolt.JoltPhysicsScene;
import dev.behindthescenery.sablejolt.constraint.JoltConstraintHandle;
import dev.behindthescenery.sablejolt.constraint.SixDofMotors;
import dev.behindthescenery.sablejolt.constraint.StaffDebug;
import dev.behindthescenery.sablejolt.constraint.fixed.JoltFixedConstraintHandle;
import dev.ryanhcode.sable.api.physics.PhysicsPipelineBody;
import dev.ryanhcode.sable.api.physics.constraint.FreeConstraintConfiguration;
import dev.ryanhcode.sable.api.physics.constraint.FreeConstraintHandle;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.Nullable;

@ApiStatus.Internal
public class JoltFreeConstraintHandle extends JoltConstraintHandle implements FreeConstraintHandle {

    private final SixDofMotors.MotorParams[] motors = new SixDofMotors.MotorParams[SixDofMotors.AXIS_COUNT];
    private com.github.stephengold.joltjni.SixDofConstraintSettings settings;
    private org.joml.Quaterniond frameQuat = new org.joml.Quaterniond();

    public static JoltFreeConstraintHandle create(final JoltPhysicsScene scene, @Nullable final PhysicsPipelineBody bodyA, @Nullable final PhysicsPipelineBody bodyB, final FreeConstraintConfiguration config) {
        final JoltPhysicsScene.SableBody sbA = bodyA == null ? null : scene.body(bodyA.getRuntimeId());
        final JoltPhysicsScene.SableBody sbB = bodyB == null ? null : scene.body(bodyB.getRuntimeId());
        if (sbA == null && sbB == null) {
            return new JoltFreeConstraintHandle(scene, -1);
        }

        final int joltA = sbA != null ? sbA.joltId : scene.groundBodyId();
        final int joltB = sbB != null ? sbB.joltId : scene.groundBodyId();

        final double comAx = sbA != null ? sbA.centerOfMass.x : 0.0;
        final double comAy = sbA != null ? sbA.centerOfMass.y : 0.0;
        final double comAz = sbA != null ? sbA.centerOfMass.z : 0.0;
        final double comBx = sbB != null ? sbB.centerOfMass.x : 0.0;
        final double comBy = sbB != null ? sbB.centerOfMass.y : 0.0;
        final double comBz = sbB != null ? sbB.centerOfMass.z : 0.0;

        final SixDofConstraintSettings settings = new SixDofConstraintSettings();
        settings.setSpace(EConstraintSpace.LocalToBodyCom);
        settings.setPosition1(scene.toLocalAnchor(sbA, config.pos1().x(), config.pos1().y(), config.pos1().z()));
        // Center the body-side constraint frame at the body's center of mass so the
        // angular servo rotates the object around its own axis (in place) instead of
        // swinging it around the grab point.
        settings.setPosition2(RVec3.sZero());

        final var q = config.orientation();
        final Quat quat = new Quat((float) q.x(), (float) q.y(), (float) q.z(), (float) q.w());
        settings.setAxisX1(JoltFixedConstraintHandle.rotate(quat, Vec3.sAxisX()));
        settings.setAxisY1(JoltFixedConstraintHandle.rotate(quat, Vec3.sAxisY()));

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
        return handle;
    }

    private JoltFreeConstraintHandle(final JoltPhysicsScene scene, final long handle) {
        super(scene);
        this.attach(handle);
    }

    @Override
    protected void setMotorImpl(final int axisOrdinal, final double target, final double stiffness, final double damping, final boolean hasForceLimit, final double maxForce) {
        final JoltPhysicsScene.JointRecord record = this.record();
        if (record == null || !(record.constraint instanceof final SixDofConstraint constraint)) {
            return;
        }

        this.motors[axisOrdinal] = new SixDofMotors.MotorParams(target, stiffness, damping, hasForceLimit, maxForce);
        SixDofMotors.applyMotor(constraint, this.settings, axisOrdinal, this.motors, this.scene, record.joltIdA, record.joltIdB, this.frameQuat);
    }
}
