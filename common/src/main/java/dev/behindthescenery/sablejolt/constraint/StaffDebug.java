package dev.behindthescenery.sablejolt.constraint;

import dev.ryanhcode.sable.Sable;
import org.jetbrains.annotations.ApiStatus;
import org.joml.Vector3d;

/**
 * Rate-limited debug dump for physics staff drag sessions.
 * Enabled with {@code -Dsablejolt.staff=true}.
 */
@ApiStatus.Internal
public final class StaffDebug {
    private static final boolean ENABLED = Boolean.getBoolean("sablejolt.staff");
    private static long lastFlush;
    private static int creations;
    private static int motorCalls;

    private static int bodyId = -1;
    private static final Vector3d bodyPos = new Vector3d();
    private static final Vector3d current = new Vector3d();
    private static final Vector3d target = new Vector3d();
    private static final Vector3d vel = new Vector3d();
    private static final Vector3d wTarget = new Vector3d();
    private static double errAngle;
    private static double gain;
    private static double mass = -1;
    private static double lastTarget;
    private static double lastK;
    private static double lastD;
    private static int lastAxis = -1;

    private StaffDebug() {
    }

    public static void constraintCreated(final int bodyRuntimeId, final double px, final double py, final double pz,
                                         final double qx, final double qy, final double qz, final double qw) {
        if (!ENABLED) {
            return;
        }
        creations++;
        Sable.LOGGER.info("[SableJolt:staff] constraint created: body={} anchor2=({}) orientation=({})",
                bodyRuntimeId, f3(px, py, pz), f3(qx, qy, qz, qw));
    }

    public static void constraintRemoved() {
        if (!ENABLED) {
            return;
        }
        removed++;
    }

    public static void motor(final int bodyRuntimeId, final double bx, final double by, final double bz,
                             final Vector3d currentCs, final int axis, final double axisTarget,
                             final Vector3d velocity, final Vector3d angularVelocity, final double errAngleRad,
                             final double servoGain, final double effMass,
                             final double stiffness, final double axisDamping) {
        if (!ENABLED) {
            return;
        }

        bodyId = bodyRuntimeId;
        bodyPos.set(bx, by, bz);
        current.set(currentCs);
        if (axis < 3) {
            // accumulate the shared linear target; only complete after LINEAR_Z
            target.setComponent(axis, axisTarget);
        }
        vel.set(velocity);
        wTarget.set(angularVelocity);
        errAngle = errAngleRad;
        gain = servoGain;
        mass = effMass;
        lastAxis = axis;
        lastTarget = axisTarget;
        lastK = stiffness;
        lastD = axisDamping;
        motorCalls++;

        flush();
    }

    private static void flush() {
        final long now = System.currentTimeMillis();
        if (now - lastFlush < 1000) {
            return;
        }
        lastFlush = now;

        Sable.LOGGER.info("[SableJolt:staff] === dump: created={} removed={} motorCalls={} ===", creations, removed, motorCalls);
        creations = 0;
        removed = 0;
        motorCalls = 0;

        if (bodyId < 0) {
            return;
        }

        Sable.LOGGER.info("[SableJolt:staff] body={} worldPos={} curCs={} target={}",
                bodyId, f3(bodyPos), f3(current), f3(target));
        Sable.LOGGER.info("[SableJolt:staff] err={} servoVel={} gain={} mass={}",
                f3(target.x - current.x, target.y - current.y, target.z - current.z), f3(vel),
                s(gain), s(mass));
        Sable.LOGGER.info("[SableJolt:staff] rotErrAngle={}rad angVel={} lastMotor: axis={} target={} K={} D={}",
                s(errAngle), f3(wTarget), axisName(lastAxis), s(lastTarget), s(lastK), s(lastD));
    }

    private static String axisName(final int axis) {
        return switch (axis) {
            case 0 -> "LINEAR_X";
            case 1 -> "LINEAR_Y";
            case 2 -> "LINEAR_Z";
            case 3 -> "ANGULAR_X";
            case 4 -> "ANGULAR_Y";
            case 5 -> "ANGULAR_Z";
            default -> "?" + axis;
        };
    }

    private static String s(final double v) {
        return String.format("%.3f", v);
    }

    private static String f3(final Vector3d vec) {
        return f3(vec.x, vec.y, vec.z);
    }

    private static String f3(final double x, final double y, final double z) {
        return "(" + s(x) + ", " + s(y) + ", " + s(z) + ")";
    }

    private static String f3(final double x, final double y, final double z, final double w) {
        return "(" + s(x) + ", " + s(y) + ", " + s(z) + ", " + s(w) + ")";
    }

    private static int removed;
}
