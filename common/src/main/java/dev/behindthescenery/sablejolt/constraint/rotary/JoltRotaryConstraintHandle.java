package dev.behindthescenery.sablejolt.constraint.rotary;

import com.github.stephengold.joltjni.HingeConstraint;
import com.github.stephengold.joltjni.HingeConstraintSettings;
import com.github.stephengold.joltjni.MotorSettings;
import com.github.stephengold.joltjni.SpringSettings;
import com.github.stephengold.joltjni.Vec3;
import com.github.stephengold.joltjni.RVec3;
import com.github.stephengold.joltjni.enumerate.EMotorState;
import com.github.stephengold.joltjni.enumerate.ESpringMode;
import dev.behindthescenery.sablejolt.JoltPhysicsScene;
import dev.behindthescenery.sablejolt.constraint.JoltConstraintHandle;
import dev.ryanhcode.sable.api.physics.PhysicsPipelineBody;
import dev.ryanhcode.sable.api.physics.constraint.RotaryConstraintConfiguration;
import dev.ryanhcode.sable.api.physics.constraint.RotaryConstraintHandle;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.Nullable;

@ApiStatus.Internal
public class JoltRotaryConstraintHandle extends JoltConstraintHandle implements RotaryConstraintHandle {

    public static JoltRotaryConstraintHandle create(final JoltPhysicsScene scene, @Nullable final PhysicsPipelineBody bodyA, @Nullable final PhysicsPipelineBody bodyB, final RotaryConstraintConfiguration config) {
        final JoltPhysicsScene.SableBody sbA = bodyA == null ? null : scene.body(bodyA.getRuntimeId());
        final JoltPhysicsScene.SableBody sbB = bodyB == null ? null : scene.body(bodyB.getRuntimeId());
        if (sbA == null && sbB == null) {
            return new JoltRotaryConstraintHandle(scene, -1);
        }

        final int joltA = sbA != null ? sbA.joltId : scene.groundBodyId();
        final int joltB = sbB != null ? sbB.joltId : scene.groundBodyId();

        final double comAx = sbA != null ? sbA.centerOfMass.x : 0.0;
        final double comAy = sbA != null ? sbA.centerOfMass.y : 0.0;
        final double comAz = sbA != null ? sbA.centerOfMass.z : 0.0;
        final double comBx = sbB != null ? sbB.centerOfMass.x : 0.0;
        final double comBy = sbB != null ? sbB.centerOfMass.y : 0.0;
        final double comBz = sbB != null ? sbB.centerOfMass.z : 0.0;

        final HingeConstraintSettings settings = new HingeConstraintSettings();
        settings.setSpace(com.github.stephengold.joltjni.enumerate.EConstraintSpace.LocalToBodyCom);
        settings.setPoint1(scene.toLocalAnchor(sbA, config.pos1().x(), config.pos1().y(), config.pos1().z()));
        settings.setPoint2(scene.toLocalAnchor(sbB, config.pos2().x(), config.pos2().y(), config.pos2().z()));

        final double nx1 = config.normal1().x(), ny1 = config.normal1().y(), nz1 = config.normal1().z();
        final double nx2 = config.normal2().x(), ny2 = config.normal2().y(), nz2 = config.normal2().z();
        settings.setHingeAxis1(normalize(new Vec3((float) nx1, (float) ny1, (float) nz1)));
        settings.setHingeAxis2(normalize(new Vec3((float) nx2, (float) ny2, (float) nz2)));
        settings.setNormalAxis1(perpendicular(settings.getHingeAxis1()));
        settings.setNormalAxis2(perpendicular(settings.getHingeAxis2()));

        final HingeConstraint constraint = (HingeConstraint) scene.createConstraint(settings, joltA, joltB);

        final var record = new JoltPhysicsScene.JointRecord();
        record.joltIdA = joltA;
        record.joltIdB = joltB;
        record.constraint = constraint;
        record.contactsEnabled = true;
        record.fixedContacts = false;
        final long id = scene.registerJoint(record);

        final JoltRotaryConstraintHandle handle = new JoltRotaryConstraintHandle(scene, id);
        return handle;
    }

    private JoltRotaryConstraintHandle(final JoltPhysicsScene scene, final long handle) {
        super(scene);
        this.attach(handle);
    }

    private static Vec3 normalize(final Vec3 v) {
        final float length = v.length();
        if (length < 1.0e-6f) {
            return new Vec3(0.0f, 1.0f, 0.0f);
        }
        return new Vec3(v.getX() / length, v.getY() / length, v.getZ() / length);
    }

    private static Vec3 perpendicular(final Vec3 axis) {
        final float ax = Math.abs(axis.getX()), ay = Math.abs(axis.getY()), az = Math.abs(axis.getZ());
        final Vec3 other = ax < 0.9f ? new Vec3(1.0f, 0.0f, 0.0f) : new Vec3(0.0f, 0.0f, 1.0f);
        return normalize(new Vec3(
                axis.getY() * other.getZ() - axis.getZ() * other.getY(),
                axis.getZ() * other.getX() - axis.getX() * other.getZ(),
                axis.getX() * other.getY() - axis.getY() * other.getX()));
    }

    @Override
    protected void setMotorImpl(final int axisOrdinal, final double target, final double stiffness, final double damping, final boolean hasForceLimit, final double maxForce) {
        final JoltPhysicsScene.JointRecord record = this.record();
        if (record == null || !(record.constraint instanceof final HingeConstraint hinge)) {
            return;
        }
        hinge.setMotorState(EMotorState.Position);
        final MotorSettings motor = hinge.getMotorSettings();
        final SpringSettings spring = motor.getSpringSettings();
        spring.setMode(ESpringMode.StiffnessAndDamping);
        spring.setStiffness((float) stiffness);
        spring.setDamping((float) damping);
        if (hasForceLimit) {
            motor.setTorqueLimit((float) maxForce);
        }
        hinge.setTargetAngle((float) target);
    }
}
