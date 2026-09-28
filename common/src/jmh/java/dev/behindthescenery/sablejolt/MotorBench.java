package dev.behindthescenery.sablejolt;

import com.github.stephengold.joltjni.SixDofConstraint;
import com.github.stephengold.joltjni.SixDofConstraintSettings;
import com.github.stephengold.joltjni.TwoBodyConstraint;
import com.github.stephengold.joltjni.enumerate.EConstraintSpace;
import dev.behindthescenery.sablejolt.constraint.SixDofMotors;
import org.joml.Quaterniond;
import org.joml.Vector3d;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.concurrent.TimeUnit;

/**
 * Per-tick 6-axis servo cost of one constraint: mirrors the handle loops
 * (all six {@code setMotorImpl} calls in order). {@link #motor} covers the
 * generic path, {@link #motorFree} the creative-grip path.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@State(Scope.Benchmark)
public class MotorBench {
    private JoltPhysicsScene scene;
    private SixDofConstraint constraint;
    private SixDofConstraintSettings settings;
    private SixDofMotors.MotorParams[] motors;
    private int joltA;
    private int joltB;
    private final Vector3d targetWorld = new Vector3d(5.0, 64.0, 0.0);

    @Setup(org.openjdk.jmh.annotations.Level.Trial)
    public void setup() {
        this.scene = BenchSupport.newScene();
        this.scene.createBox(10, 10.0, 0.5, 0.5, 0.5,
                new Vector3d(0.0, 64.0, 0.0), new Quaterniond());
        this.scene.createBox(11, 10.0, 0.5, 0.5, 0.5,
                new Vector3d(2.0, 64.0, 0.0), new Quaterniond());
        final JoltPhysicsScene.SableBody sbA = this.scene.body(10);
        final JoltPhysicsScene.SableBody sbB = this.scene.body(11);
        this.joltA = sbA.joltId;
        this.joltB = sbB.joltId;
        this.settings = new SixDofConstraintSettings();
        this.settings.setSpace(EConstraintSpace.LocalToBodyCom);
        this.settings.setPosition1(this.scene.toLocalAnchor(sbA, 0.0, 64.0, 0.0));
        this.settings.setPosition2(this.scene.toLocalAnchor(sbB, 2.0, 64.0, 0.0));
        final TwoBodyConstraint created =
                this.scene.createConstraint(this.settings, this.joltA, this.joltB);
        this.constraint = (SixDofConstraint) created;
        this.motors = new SixDofMotors.MotorParams[SixDofMotors.AXIS_COUNT];
        for (int i = 0; i < this.motors.length; i++) {
            this.motors[i] = new SixDofMotors.MotorParams(
                    i < 3 ? 0.5 : 0.1, 50.0, 5.0, true, 1000.0);
        }
        this.scene.wakeUpObject(10);
        this.scene.wakeUpObject(11);
    }

    @Setup(org.openjdk.jmh.annotations.Level.Invocation)
    public void wake() {
        this.scene.wakeUpObject(10);
        this.scene.wakeUpObject(11);
    }

    @Benchmark
    public void motor() {
        for (int i = 0; i < 6; i++) {
            SixDofMotors.applyMotor(this.constraint, this.settings, i,
                    this.motors, this.scene, this.joltA, this.joltB, null);
        }
    }

    @Benchmark
    public void motorFree() {
        for (int i = 0; i < 6; i++) {
            SixDofMotors.applyMotorFree(this.constraint, this.settings, i,
                    this.motors, this.scene, this.joltA, this.joltB,
                    null, this.targetWorld, false, null);
        }
    }
}
