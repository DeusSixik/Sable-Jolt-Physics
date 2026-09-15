package dev.behindthescenery.sablejolt.constraint.fixed;

import com.github.stephengold.joltjni.FixedConstraintSettings;
import com.github.stephengold.joltjni.RVec3;
import com.github.stephengold.joltjni.Vec3;
import com.github.stephengold.joltjni.enumerate.EConstraintSpace;
import dev.behindthescenery.sablejolt.JoltPhysicsScene;
import dev.behindthescenery.sablejolt.constraint.JoltConstraintHandle;
import dev.ryanhcode.sable.api.physics.PhysicsPipelineBody;
import dev.ryanhcode.sable.api.physics.constraint.FixedConstraintConfiguration;
import dev.ryanhcode.sable.api.physics.constraint.FixedConstraintHandle;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.Nullable;

@ApiStatus.Internal
public class JoltFixedConstraintHandle extends JoltConstraintHandle implements FixedConstraintHandle {

    public static JoltFixedConstraintHandle create(final JoltPhysicsScene scene, @Nullable final PhysicsPipelineBody bodyA, @Nullable final PhysicsPipelineBody bodyB, final FixedConstraintConfiguration config) {
        final JoltPhysicsScene.SableBody sbA = bodyA == null ? null : scene.body(bodyA.getRuntimeId());
        final JoltPhysicsScene.SableBody sbB = bodyB == null ? null : scene.body(bodyB.getRuntimeId());
        if (sbA == null && sbB == null) {
            return new JoltFixedConstraintHandle(scene, -1);
        }

        final int joltA = sbA != null ? sbA.joltId : scene.groundBodyId();
        final int joltB = sbB != null ? sbB.joltId : scene.groundBodyId();

        final FixedConstraintSettings settings = new FixedConstraintSettings();
        settings.setSpace(EConstraintSpace.LocalToBodyCom);
        settings.setPoint1(scene.toLocalAnchor(sbA, config.pos1().x(), config.pos1().y(), config.pos1().z()));
        settings.setPoint2(scene.toLocalAnchor(sbB, config.pos2().x(), config.pos2().y(), config.pos2().z()));

        // frame 1 is rotated by the configured orientation, frame 2 stays identity (mirrors the rapier joint)
        final var q = config.orientation();
        settings.setAxisX1(rotate((float) q.x(), (float) q.y(), (float) q.z(), (float) q.w(), Vec3.sAxisX()));
        settings.setAxisY1(rotate((float) q.x(), (float) q.y(), (float) q.z(), (float) q.w(), Vec3.sAxisY()));

        final var constraint = scene.createConstraint(settings, joltA, joltB);

        final var record = new JoltPhysicsScene.JointRecord();
        record.joltIdA = joltA;
        record.joltIdB = joltB;
        record.constraint = constraint;
        record.contactsEnabled = false;
        record.fixedContacts = true;
        final long id = scene.registerJoint(record);

        return new JoltFixedConstraintHandle(scene, id);
    }

    private JoltFixedConstraintHandle(final JoltPhysicsScene scene, final long handle) {
        super(scene);
        this.attach(handle);
    }

    public static Vec3 rotate(final float qx, final float qy, final float qz, final float qw,
                              final float vx, final float vy, final float vz) {
        final float tx = 2.0f * (qy * vz - qz * vy);
        final float ty = 2.0f * (qz * vx - qx * vz);
        final float tz = 2.0f * (qx * vy - qy * vx);
        return new Vec3(
                vx + qw * tx + (qy * tz - qz * ty),
                vy + qw * ty + (qz * tx - qx * tz),
                vz + qw * tz + (qx * ty - qy * tx));
    }

    public static Vec3 rotate(final float qx, final float qy, final float qz, final float qw,
                              final Vec3 v) {
        // NOTE: parameter order is (quaternion, vector) here; keep the mapping aligned
        return rotate(qx, qy, qz, qw, v.getX(), v.getY(), v.getZ());
    }

    @Override
    protected void setMotorImpl(final int axisOrdinal, final double target, final double stiffness, final double damping, final boolean hasForceLimit, final double maxForce) {
        // fixed constraints have no motors
    }
}
