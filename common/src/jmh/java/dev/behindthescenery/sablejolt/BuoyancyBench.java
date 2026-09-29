package dev.behindthescenery.sablejolt;

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

import java.lang.reflect.Method;
import java.util.concurrent.TimeUnit;

/**
 * Buoyancy in isolation (no solver): 6 bodies x ~4k children parked inside
 * the fluid band. Old path: one {@code addForce} JNI call per submerged
 * child plus one for the centroid float force. New path: two JNI calls per
 * body total (accumulated force + torque).
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@State(Scope.Benchmark)
public class BuoyancyBench {
    private static final int BODIES = 6;

    private JoltPhysicsScene scene;
    private Method computeBuoyancy;

    @Setup(org.openjdk.jmh.annotations.Level.Trial)
    public void setup() throws Exception {
        this.scene = BenchSupport.newScene();
        final int[] ids = BenchSupport.registerEntries(this.scene);
        final Quaterniond rot = new Quaterniond();
        for (int b = 0; b < BODIES; b++) {
            final int baseX = b * 32;
            final double cx = baseX + 16.0;
            final double cy = 71.5;
            final double cz = 7.5;
            this.scene.createSubLevel(b, new Vector3d(cx, cy, cz), rot);
            this.scene.setLocalBounds(b, baseX, 60, 0, baseX + 31, 79, 15);
            this.scene.setCenterOfMass(b, cx, cy, cz);
            final int[] data = BenchSupport.mixedSection(ids[0], ids[1]);
            this.scene.addChunk(b * 2, 4, 0, data, true, -1);
            this.scene.addChunk(b * 2 + 1, 4, 0, data, true, -1);
        }
        this.scene.step(0.05);
        final JoltPhysicsScene.SableBody probe = this.scene.body(0);
        if (probe.children.isEmpty()) {
            throw new IllegalStateException("body has no children");
        }
        this.computeBuoyancy = BenchSupport.method(
                JoltPhysicsScene.class, "computeBuoyancy");
        if (this.computeBuoyancy == null) {
            throw new IllegalStateException("computeBuoyancy not found");
        }
        // Sanity: buoyancy must actually find submerged volume in this scene,
        // otherwise the benchmark measures the early-out path.
        this.park();
        this.computeBuoyancy.invoke(this.scene);
    }

    private void park() {
        for (int b = 0; b < BODIES; b++) {
            final JoltPhysicsScene.SableBody sb = this.scene.body(b);
            this.scene.getBodyInterface().setLinearAndAngularVelocity(
                    sb.joltId, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            this.scene.teleportObject(b, sb.centerOfMass.x,
                    sb.centerOfMass.y + 8.0, sb.centerOfMass.z,
                    0.0, 0.0, 0.0, 1.0);
            this.scene.wakeUpObject(b);
        }
    }

    @Setup(org.openjdk.jmh.annotations.Level.Invocation)
    public void resetBodies() {
        this.park();
    }

    @Benchmark
    public void buoyancy() throws Exception {
        this.computeBuoyancy.invoke(this.scene);
    }
}
