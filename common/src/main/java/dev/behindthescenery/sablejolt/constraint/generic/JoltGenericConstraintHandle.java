package dev.behindthescenery.sablejolt.constraint.generic;

import com.github.stephengold.joltjni.MotorSettings;
import com.github.stephengold.joltjni.Quat;
import com.github.stephengold.joltjni.RVec3;
import com.github.stephengold.joltjni.SixDofConstraint;
import com.github.stephengold.joltjni.SixDofConstraintSettings;
import com.github.stephengold.joltjni.SpringSettings;
import com.github.stephengold.joltjni.Vec3;
import com.github.stephengold.joltjni.enumerate.EAxis;
import com.github.stephengold.joltjni.enumerate.EConstraintSpace;
import com.github.stephengold.joltjni.enumerate.EMotorState;
import com.github.stephengold.joltjni.enumerate.ESpringMode;
import dev.behindthescenery.sablejolt.JoltPhysicsScene;
import dev.behindthescenery.sablejolt.constraint.JoltConstraintHandle;
import dev.behindthescenery.sablejolt.constraint.SixDofMotors;
import dev.behindthescenery.sablejolt.constraint.fixed.JoltFixedConstraintHandle;
import dev.ryanhcode.sable.api.physics.PhysicsPipelineBody;
import dev.ryanhcode.sable.api.physics.constraint.ConstraintJointAxis;
import dev.ryanhcode.sable.api.physics.constraint.GenericConstraintConfiguration;
import dev.ryanhcode.sable.api.physics.constraint.GenericConstraintHandle;
import net.minecraft.server.level.ServerLevel;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.joml.Quaterniond;
import org.joml.Quaterniondc;
import org.joml.Vector3dc;

@ApiStatus.Internal
public class JoltGenericConstraintHandle extends JoltConstraintHandle implements GenericConstraintHandle {

    private static final int FRAME_SIDE_FIRST = 0;
    private static final int FRAME_SIDE_SECOND = 1;

    /**
     * Creates a jolt generic constraint handle
     */
    @Contract("_, _, _, _ -> new")
    public static @NotNull JoltGenericConstraintHandle create(final JoltPhysicsScene scene, @Nullable final PhysicsPipelineBody bodyA, @Nullable final PhysicsPipelineBody bodyB, final GenericConstraintConfiguration config) {
        final JoltPhysicsScene.SableBody sbA = bodyA == null ? null : scene.body(bodyA.getRuntimeId());
        final JoltPhysicsScene.SableBody sbB = bodyB == null ? null : scene.body(bodyB.getRuntimeId());
        if (sbA == null && sbB == null) {
            return new JoltGenericConstraintHandle(scene, -1);
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
        settings.setPosition2(scene.toLocalAnchor(sbB, config.pos2().x(), config.pos2().y(), config.pos2().z()));

        final Quaterniondc rotA = config.orientation1();
        final Quaterniondc rotB = config.orientation2();
        final Quat quatA = new Quat((float) rotA.x(), (float) rotA.y(), (float) rotA.z(), (float) rotA.w());
        final Quat quatB = new Quat((float) rotB.x(), (float) rotB.y(), (float) rotB.z(), (float) rotB.w());
        settings.setAxisX1(JoltFixedConstraintHandle.rotate(quatA, Vec3.sAxisX()));
        settings.setAxisY1(JoltFixedConstraintHandle.rotate(quatA, Vec3.sAxisY()));
        settings.setAxisX2(JoltFixedConstraintHandle.rotate(quatB, Vec3.sAxisX()));
        settings.setAxisY2(JoltFixedConstraintHandle.rotate(quatB, Vec3.sAxisY()));

        for (final EAxis axis : EAxis.values()) {
            if (axis == EAxis.Num) {
                continue;
            }
            settings.makeFreeAxis(axis);
        }
        for (final ConstraintJointAxis locked : config.lockedAxes()) {
            settings.makeFixedAxis(EAxis.values()[locked.ordinal()]);
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

        final JoltGenericConstraintHandle handle = new JoltGenericConstraintHandle(scene, id);
        handle.frameA = config.pos1();
        handle.frameB = config.pos2();
        handle.rotA = new Quaterniond(rotA);
        handle.rotB = new Quaterniond(rotB);
        handle.sbA = sbA;
        handle.sbB = sbB;
        handle.joltA = joltA;
        handle.joltB = joltB;
        return handle;
    }

    private Vector3dc frameA;
    private Vector3dc frameB;
    private Quaterniond rotA;
    private Quaterniond rotB;
    private JoltPhysicsScene.SableBody sbA;
    private JoltPhysicsScene.SableBody sbB;
    private int joltA;
    private int joltB;
    private final SixDofMotors.MotorParams[] motors = new SixDofMotors.MotorParams[SixDofMotors.AXIS_COUNT];
    private com.github.stephengold.joltjni.SixDofConstraintSettings ownSettings;

    private JoltGenericConstraintHandle(final JoltPhysicsScene scene, final long handle) {
        super(scene);
        this.attach(handle);
    }

    @Override
    public void setFrame1(final Vector3dc localPosition, final Quaterniondc localOrientation) {
        this.assertValid();
        this.frameA = localPosition;
        this.rotA = new Quaterniond(localOrientation);
        this.rebuild();
    }

    @Override
    public void setFrame2(final Vector3dc localPosition, final Quaterniondc localOrientation) {
        this.assertValid();
        this.frameB = localPosition;
        this.rotB = new Quaterniond(localOrientation);
        this.rebuild();
    }

    /**
     * Adds / sets an axis limit on this constraint
     */
    @Override
    public void setLimit(final ConstraintJointAxis axis, final double min, final double max) {
        this.assertValid();
        if (!(this.record().constraint instanceof final SixDofConstraint constraint)) {
            return;
        }
        this.record().limits[axis.ordinal()][0] = min;
        this.record().limits[axis.ordinal()][1] = max;
        this.applyLimits(constraint, axis.ordinal(), min, max);
    }

    private static void applyLimits(final SixDofConstraint constraint, final int axisOrdinal, final double min, final double max) {
        final EAxis axis = EAxis.values()[axisOrdinal];
        if (axis.ordinal() < 3) {
            final Vec3 minV = constraint.getTranslationLimitsMin();
            final Vec3 maxV = constraint.getTranslationLimitsMax();
            final float[] minArr = {minV.getX(), minV.getY(), minV.getZ()};
            final float[] maxArr = {maxV.getX(), maxV.getY(), maxV.getZ()};
            minArr[axisOrdinal] = (float) min;
            maxArr[axisOrdinal] = (float) max;
            constraint.setTranslationLimits(new Vec3(minArr[0], minArr[1], minArr[2]), new Vec3(maxArr[0], maxArr[1], maxArr[2]));
        } else {
            final int i = axisOrdinal - 3;
            final Vec3 minV = constraint.getRotationLimitsMin();
            final Vec3 maxV = constraint.getRotationLimitsMax();
            final float[] minArr = {minV.getX(), minV.getY(), minV.getZ()};
            final float[] maxArr = {maxV.getX(), maxV.getY(), maxV.getZ()};
            minArr[i] = (float) min;
            maxArr[i] = (float) max;
            constraint.setRotationLimits(new Vec3(minArr[0], minArr[1], minArr[2]), new Vec3(maxArr[0], maxArr[1], maxArr[2]));
        }
    }

    /**
     * Locks the given constraint axes on this constraint
     */
    @Override
    public void lockAxes(final ConstraintJointAxis @NotNull... axes) {
        this.assertValid();
        if (!(this.record().constraint instanceof final SixDofConstraint constraint)) {
            return;
        }

        for (final ConstraintJointAxis axis : axes) {
            this.record().locked[axis.ordinal()] = true;
            // a locked axis is a limited axis with min == max == 0
            this.applyLimits(constraint, axis.ordinal(), 0.0, 0.0);
        }
    }

    /**
     * Rebuilds the constraint with the current frames (Jolt does not support frame mutation at runtime).
     */
    private void rebuild() {
        final JoltPhysicsScene.JointRecord record = this.record();
        if (record == null) {
            return;
        }

        final double comAx = this.sbA != null ? this.sbA.centerOfMass.x : 0.0;
        final double comAy = this.sbA != null ? this.sbA.centerOfMass.y : 0.0;
        final double comAz = this.sbA != null ? this.sbA.centerOfMass.z : 0.0;
        final double comBx = this.sbB != null ? this.sbB.centerOfMass.x : 0.0;
        final double comBy = this.sbB != null ? this.sbB.centerOfMass.y : 0.0;
        final double comBz = this.sbB != null ? this.sbB.centerOfMass.z : 0.0;

        final SixDofConstraintSettings settings = new SixDofConstraintSettings();
        settings.setSpace(EConstraintSpace.LocalToBodyCom);
        settings.setPosition1(this.scene.toLocalAnchor(this.sbA, this.frameA.x(), this.frameA.y(), this.frameA.z()));
        settings.setPosition2(this.scene.toLocalAnchor(this.sbB, this.frameB.x(), this.frameB.y(), this.frameB.z()));

        final Quat quatA = new Quat((float) this.rotA.x, (float) this.rotA.y, (float) this.rotA.z, (float) this.rotA.w);
        final Quat quatB = new Quat((float) this.rotB.x, (float) this.rotB.y, (float) this.rotB.z, (float) this.rotB.w);
        settings.setAxisX1(JoltFixedConstraintHandle.rotate(quatA, Vec3.sAxisX()));
        settings.setAxisY1(JoltFixedConstraintHandle.rotate(quatA, Vec3.sAxisY()));
        settings.setAxisX2(JoltFixedConstraintHandle.rotate(quatB, Vec3.sAxisX()));
        settings.setAxisY2(JoltFixedConstraintHandle.rotate(quatB, Vec3.sAxisY()));

        for (int i = 0; i < 6; i++) {
            if (record.locked[i] || (record.limits[i][0] == 0.0 && record.limits[i][1] == 0.0 && isLockedDefault(i))) {
                settings.makeFixedAxis(EAxis.values()[i]);
            } else {
                settings.makeFreeAxis(EAxis.values()[i]);
            }
        }

        this.scene.removeConstraint(record.constraint);
        final SixDofConstraint constraint = (SixDofConstraint) this.scene.createConstraint(settings, this.joltA, this.joltB);
        record.constraint = constraint;
        record.genericSettings = settings;
        this.ownSettings = settings;

        for (int i = 0; i < 6; i++) {
            if (record.limits[i][1] > record.limits[i][0] && !record.locked[i]) {
                applyLimits(constraint, i, record.limits[i][0], record.limits[i][1]);
            } else if (record.locked[i]) {
                applyLimits(constraint, i, 0.0, 0.0);
            }
        }

        for (int i = 0; i < 6; i++) {
            if (this.motors[i] != null) {
                SixDofMotors.applyMotor(constraint, this.ownSettings, i, this.motors, this.scene, record.joltIdA, record.joltIdB, this.rotA);
            }
        }
    }

    private static boolean isLockedDefault(final int axis) {
        return false;
    }

    @Override
    protected void setMotorImpl(final int axisOrdinal, final double target, final double stiffness, final double damping, final boolean hasForceLimit, final double maxForce) {
        final JoltPhysicsScene.JointRecord record = this.record();
        if (record == null || !(record.constraint instanceof final SixDofConstraint constraint)) {
            return;
        }

        this.motors[axisOrdinal] = new SixDofMotors.MotorParams(target, stiffness, damping, hasForceLimit, maxForce);
        SixDofMotors.applyMotor(constraint, this.ownSettings, axisOrdinal, this.motors, this.scene, record.joltIdA, record.joltIdB, this.rotA);
    }
}
