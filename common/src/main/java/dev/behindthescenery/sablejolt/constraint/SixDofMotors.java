package dev.behindthescenery.sablejolt.constraint;

import com.github.stephengold.joltjni.Body;
import com.github.stephengold.joltjni.BodyInterface;
import com.github.stephengold.joltjni.MotorSettings;
import com.github.stephengold.joltjni.Quat;
import com.github.stephengold.joltjni.RVec3;
import com.github.stephengold.joltjni.SixDofConstraint;
import com.github.stephengold.joltjni.Vec3;
import com.github.stephengold.joltjni.enumerate.EAxis;
import com.github.stephengold.joltjni.enumerate.EMotorState;
import dev.behindthescenery.sablejolt.JoltDebugLogging;
import dev.behindthescenery.sablejolt.JoltPhysicsScene;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.Nullable;
import org.joml.Quaterniond;
import org.joml.Vector3d;

/**
 * Shared helpers for applying Sable constraint motors to Jolt {@link SixDofConstraint}s.
 * <p>
 * Jolt's {@code setTargetPositionCs(Vec3)} / {@code setTargetOrientationCs(Quat)} set the
 * motor target for all axes of their group at once, while the Sable API sets one axis at
 * a time; the per-axis state is therefore accumulated and re-applied in full.
 * <p>
 * Sable motors are driven with potentially enormous position errors (sub-levels live
 * millions of blocks away in the plot grid and are dragged towards the player region).
 * Position springs explode over such errors, so the motors are driven in
 * {@code Velocity} mode: target velocity = error * gain, capped. This is an
 * unconditionally stable servo that behaves like the rapier motor the staff expects.
 */
@ApiStatus.Internal
public final class SixDofMotors {

    /**
     * The stored motor parameters of a single axis.
     */
    public record MotorParams(double target, double stiffness, double damping, boolean hasForceLimit, double maxForce) {
    }

    public static final int AXIS_COUNT = 6;

    /**
     * Servo caps. The velocity command is applied instantly by the solver, so the
     * caps must be low enough to prevent bang-bang oscillation around the target:
     * linear drag stays at walking-to-sprint speed, rotation at ~460 deg/s with the
     * total (follow + correction) clamped a bit higher.
     */
    private static final double MAX_LINEAR_SPEED = 30.0;
    private static final double MAX_ANGULAR_SPEED = 6.0;
    private static final double MAX_ANGULAR_TOTAL_SPEED = 8.0;
    private static final double MAX_LINEAR_GAIN = 8.0;

    /**
     * Sable physics tick rate.
     */
    private static final double DT = 0.05;

    private SixDofMotors() {
    }

    /**
     * Applies the motor of one axis and re-applies the shared target state of all axes.
     *
     * @param state per-axis motor state, entries may be null (axis never motorized)
     * @param jointFrameQuatLocal the joint frame1 rotation in body 1's local space
     *        (targets are expressed in this frame); may be null for world-aligned frames
     */
    public static void applyMotor(final SixDofConstraint constraint, final com.github.stephengold.joltjni.SixDofConstraintSettings settings,
                                  final int axisOrdinal, @Nullable final MotorParams[] state,
                                  final JoltPhysicsScene scene, final int joltIdA, final int joltIdB,
                                  @Nullable final Quaterniond jointFrameQuatLocal) {
        final MotorParams params = state[axisOrdinal];
        if (params == null) {
            return;
        }

        final EAxis axis = EAxis.values()[axisOrdinal];
        constraint.setMotorState(axis, EMotorState.Velocity);

        final MotorSettings motor = constraint.getMotorSettings(axis);
        if (params.hasForceLimit()) {
            motor.setForceLimit((float) params.maxForce());
            motor.setTorqueLimit((float) params.maxForce());
        }

        final double gain = servoGain(params.stiffness(), effectiveMass(constraint));

        final Vector3d current = currentLinearPositionCs(constraint, settings, jointFrameQuatLocal);
        double vx = (targetOf(state, 0) - current.x) * gain;
        double vy = (targetOf(state, 1) - current.y) * gain;
        double vz = (targetOf(state, 2) - current.z) * gain;
        final double speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (speed > MAX_LINEAR_SPEED) {
            final double scale = MAX_LINEAR_SPEED / speed;
            vx *= scale;
            vy *= scale;
            vz *= scale;
        }
        if (!isFinite(vx) || !isFinite(vy) || !isFinite(vz)) {
            vx = 0.0;
            vy = 0.0;
            vz = 0.0;
        }
        constraint.setTargetVelocityCs(new Vec3((float) vx, (float) vy, (float) vz));

        // ANGULAR: a velocity servo expressed in WORLD space (stable across the
        // per-tick constraint recreation):
        //   omega = follow (rotation speed of the target frame) + correction (pull
        //   towards the target orientation). The follow term never reverses direction,
        //   so continuous rotation does not flip backwards at the 180-degree wrap.
        double wx = 0.0;
        double wy = 0.0;
        double wz = 0.0;
        if (axisOrdinal >= 3) {
            final Body b1 = constraint.getBody1();
            final Body b2 = constraint.getBody2();
            final Quat r1 = b1.getRotation();
            final Quat r2 = b2.getRotation();

            final Quat jointLocal = jointFrameQuatLocal == null ? Quat.sIdentity()
                    : new Quat((float) jointFrameQuatLocal.x, (float) jointFrameQuatLocal.y,
                            (float) jointFrameQuatLocal.z, (float) jointFrameQuatLocal.w);
            // world rotation of joint frame 1
            final Quat q1w = mulQuat(r1, jointLocal);
            final Quaterniond q1wJ = new Quaterniond(q1w.getX(), q1w.getY(), q1w.getZ(), q1w.getW());

            // desired world rotation of body 2 = frame1 world rotation * target (constraint space)
            final Quaterniond desiredCs = new Quaterniond().rotationXYZ(targetOf(state, 3), targetOf(state, 4), targetOf(state, 5));
            final Quaterniond desiredW = q1wJ.mul(desiredCs, new Quaterniond());

            final Quaterniond bodyNow = new Quaterniond(r2.getX(), r2.getY(), r2.getZ(), r2.getW());

            // follow term: track the target frame's rotation speed (unwrapped by
            // per-tick shortest-arc deltas, which are always small)
            Vector3d omegaW = scene == null ? null : scene.getCachedAngularOmega(joltIdB);
            if (omegaW == null || axisOrdinal == 3) {
                omegaW = new Vector3d();
                final Quaterniond prevDesired = scene == null ? null : scene.getAngularFollowPrev(joltIdB);
                if (prevDesired != null) {
                    final Quaterniond delta = shortestArc(desiredW.mul(prevDesired.conjugate(new Quaterniond()), new Quaterniond()));
                    final double ang = 2.0 * Math.acos(Math.max(-1.0, Math.min(1.0, delta.w)));
                    if (ang > 1.0e-5) {
                        final double s = Math.sqrt(Math.max(0.0, 1.0 - delta.w * delta.w));
                        final double inv = 1.0 / s;
                        final double w = Math.min(ang / DT, MAX_ANGULAR_SPEED);
                        omegaW.set(delta.x * inv * w, delta.y * inv * w, delta.z * inv * w);
                    }
                }

                // correction term: pull the remaining orientation error to zero
                final Quaterniond errW = shortestArc(desiredW.mul(bodyNow.conjugate(new Quaterniond()), new Quaterniond()));
                final double errAngle = 2.0 * Math.acos(Math.max(-1.0, Math.min(1.0, errW.w)));
                if (errAngle > 1.0e-5) {
                    final double s = Math.sqrt(Math.max(0.0, 1.0 - errW.w * errW.w));
                    final double inv = 1.0 / s;
                    final double w = Math.min(errAngle * gain, MAX_ANGULAR_SPEED);
                    omegaW.x += errW.x * inv * w;
                    omegaW.y += errW.y * inv * w;
                    omegaW.z += errW.z * inv * w;
                }

                final double mag = Math.sqrt(omegaW.x * omegaW.x + omegaW.y * omegaW.y + omegaW.z * omegaW.z);
                if (mag > MAX_ANGULAR_TOTAL_SPEED) {
                    final double scale = MAX_ANGULAR_TOTAL_SPEED / mag;
                    omegaW.x *= scale;
                    omegaW.y *= scale;
                    omegaW.z *= scale;
                }
                if (!isFinite(omegaW.x) || !isFinite(omegaW.y) || !isFinite(omegaW.z)) {
                    omegaW.set(0.0, 0.0, 0.0);
                }
                if (scene != null) {
                    scene.setAngularFollowPrev(joltIdB, desiredW);
                    scene.setCachedAngularOmega(joltIdB, omegaW);
                }
            }

            // world -> constraint space (frame 1 rotation)
            final Quat q1wInv = q1w.conjugated();
            final Vec3 wCs = rotate(new Vec3((float) omegaW.x, (float) omegaW.y, (float) omegaW.z), q1wInv);
            wx = wCs.getX();
            wy = wCs.getY();
            wz = wCs.getZ();
            constraint.setTargetAngularVelocityCs(new Vec3((float) wx, (float) wy, (float) wz));
        }

        // Jolt motors never wake attached bodies; the gun expects immediate response
        scene.getBodyInterface().activateBody(joltIdA);
        scene.getBodyInterface().activateBody(joltIdB);

        if (JoltDebugLogging.STAFF) {
            final var b2 = constraint.getBody2();
            final RVec3 com2 = b2.getCenterOfMassPosition();
            StaffDebug.motor((int) b2.getUserData(), com2.xx(), com2.yy(), com2.zz(),
                    current, axisOrdinal, params.target(),
                    new Vector3d(vx, vy, vz),
                    new Vector3d(wx, wy, wz),
                    0.0, gain, effectiveMass(constraint),
                    params.stiffness(), params.damping());
        }
    }

    private static Quaterniond shortestArc(final Quaterniond q) {
        final Quaterniond out = new Quaterniond(q);
        if (out.w < 0.0) {
            out.x = -out.x;
            out.y = -out.y;
            out.z = -out.z;
            out.w = -out.w;
        }
        return out;
    }

    private static Quat mulQuat(final Quat a, final Quat b) {
        final float ax = a.getX(), ay = a.getY(), az = a.getZ(), aw = a.getW();
        final float bx = b.getX(), by = b.getY(), bz = b.getZ(), bw = b.getW();
        return new Quat(
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz);
    }

    private static double targetOf(final MotorParams[] state, final int axis) {
        final MotorParams p = state[axis];
        return p == null ? 0.0 : p.target();
    }

    private static boolean isFinite(final double v) {
        return !Double.isNaN(v) && !Double.isInfinite(v);
    }

    /**
     * Servo gain (per second): omega = sqrt(K / m), clamped.
     */
    private static double servoGain(final double stiffness, final float effectiveMass) {
        final double k = Math.max(stiffness, 1.0e-6);
        return Math.min(MAX_LINEAR_GAIN, Math.sqrt(k / effectiveMass));
    }

    /**
     * Computes the current linear position of body 2 relative to body 1, expressed in
     * the JOINT frame of body 1 (the same frame the Sable motor targets use: the joint
     * frame rotation includes the constraint orientation, not just the body rotation).
     */
    private static Vector3d currentLinearPositionCs(final SixDofConstraint constraint, final com.github.stephengold.joltjni.SixDofConstraintSettings settings,
                                                    @Nullable final Quaterniond jointFrameQuatLocal) {
        try {
            final Body b1 = constraint.getBody1();
            final Body b2 = constraint.getBody2();

            final RVec3 com1 = b1.getCenterOfMassPosition();
            final RVec3 com2 = b2.getCenterOfMassPosition();
            final Quat r1 = b1.getRotation();
            final Quat r2 = b2.getRotation();

            final var p1 = settings.getPosition1();
            final var p2 = settings.getPosition2();

            final Vec3 a1 = rotate(new Vec3((float) p1.xx(), (float) p1.yy(), (float) p1.zz()), r1);
            final Vec3 a2 = rotate(new Vec3((float) p2.xx(), (float) p2.yy(), (float) p2.zz()), r2);

            // world-space delta between anchors
            final double dx = (com2.xx() + a2.getX()) - (com1.xx() + a1.getX());
            final double dy = (com2.yy() + a2.getY()) - (com1.yy() + a1.getY());
            final double dz = (com2.zz() + a2.getZ()) - (com1.zz() + a1.getZ());

            // into the joint frame: world delta rotated by the inverse of
            // (body1 rotation * joint frame local rotation)
            final Quat inv;
            if (jointFrameQuatLocal != null) {
                final float qx = r1.getX(), qy = r1.getY(), qz = r1.getZ(), qw = r1.getW();
                final float jx = (float) jointFrameQuatLocal.x, jy = (float) jointFrameQuatLocal.y;
                final float jz = (float) jointFrameQuatLocal.z, jw = (float) jointFrameQuatLocal.w;
                // combined = r1 * jointFrameQuatLocal
                final float cx = qw * jx + qx * jw + qy * jz - qz * jy;
                final float cy = qw * jy - qx * jz + qy * jw + qz * jx;
                final float cz = qw * jz + qx * jy - qy * jx + qz * jw;
                final float cw = qw * jw - qx * jx - qy * jy - qz * jz;
                inv = new Quat(-cx, -cy, -cz, cw);
            } else {
                inv = r1.conjugated();
            }

            final Vec3 local = rotate(new Vec3((float) dx, (float) dy, (float) dz), inv);
            return new Vector3d(local.getX(), local.getY(), local.getZ());
        } catch (final Throwable t) {
            return new Vector3d();
        }
    }

    private static Vec3 rotate(final Vec3 v, final Quat q) {
        final float qx = q.getX(), qy = q.getY(), qz = q.getZ(), qw = q.getW();
        final float vx = v.getX(), vy = v.getY(), vz = v.getZ();
        final float tx = 2.0f * (qy * vz - qz * vy);
        final float ty = 2.0f * (qz * vx - qx * vz);
        final float tz = 2.0f * (qx * vy - qy * vx);
        return new Vec3(
                vx + qw * tx + (qy * tz - qz * ty),
                vy + qw * ty + (qz * tx - qx * tz),
                vz + qw * tz + (qx * ty - qy * tx));
    }

    /**
     * The reduced mass of the two constraint bodies (static bodies count as infinite mass).
     */
    private static float effectiveMass(final SixDofConstraint constraint) {
        float invMass = 0.0f;
        try {
            final var b1 = constraint.getBody1();
            if (b1.isDynamic() && b1.getMotionProperties() != null) {
                invMass += b1.getMotionProperties().getInverseMass();
            }
            final var b2 = constraint.getBody2();
            if (b2.isDynamic() && b2.getMotionProperties() != null) {
                invMass += b2.getMotionProperties().getInverseMass();
            }
        } catch (final Throwable ignored) {
            return 1.0f;
        }
        return invMass > 1.0e-9f ? 1.0f / invMass : 1.0f;
    }
}
