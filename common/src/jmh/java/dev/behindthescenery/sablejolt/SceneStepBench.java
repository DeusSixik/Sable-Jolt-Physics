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

import java.util.concurrent.TimeUnit;

/**
 * Steady-state simulation tick: 6 sub-level bodies x 2 mixed sections each,
 * gravity on, bodies parked so their solid blocks sit inside the fluid band
 * (physics COM one half-section above the rebuild COM). Exercises the full
 * buoyancy path (per-child drag + centroid float force), the batch snapshot,
 * the solver update and the post-step passes — the per-tick hot path.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@State(Scope.Benchmark)
public class SceneStepBench {
    private static final int BODIES = 6;

    private JoltPhysicsScene scene;
    private final double[] homeX = new double[BODIES];
    private final double[] homeY = new double[BODIES];
    private final double[] homeZ = new double[BODIES];
    private final int[] joltIds = new int[BODIES];

    @Setup(org.openjdk.jmh.annotations.Level.Trial)
    public void setup() {
        this.scene = BenchSupport.newScene();
        final int[] ids = BenchSupport.registerEntries(this.scene);
        final int solidId = ids[0];
        final int fluidId = ids[1];
        final Quaterniond rot = new Quaterniond();
        for (int b = 0; b < BODIES; b++) {
            final int baseX = b * 32;
            final double cx = baseX + 16.0;
            final double cy = 71.5;
            final double cz = 7.5;
            this.scene.createSubLevel(b, new Vector3d(cx, cy, cz), rot);
            this.scene.setLocalBounds(b, baseX, 60, 0, baseX + 31, 79, 15);
            this.scene.setCenterOfMass(b, cx, cy, cz);
            final int[] data = BenchSupport.mixedSection(solidId, fluidId);
            this.scene.addChunk(b * 2, 4, 0, data, true, -1);
            this.scene.addChunk(b * 2 + 1, 4, 0, data, true, -1);
            this.homeX[b] = cx;
            // Park the physics COM 8 blocks above the rebuild COM: solid
            // children (local y in [-7.5, 0.5]) land at world y in [72.5, 80],
            // i.e. inside the fluid half of the sections, so every child
            // contributes drag + submerged volume every tick.
            this.homeY[b] = cy + 8.0;
            this.homeZ[b] = cz;
            this.reset(b);
        }
        // Build all shapes once; measured steps are steady-state.
        this.scene.step(0.05);
        for (int b = 0; b < BODIES; b++) {
            this.reset(b);
        }
    }

    private void reset(final int b) {
        final JoltPhysicsScene.SableBody sb = this.scene.body(b);
        this.joltIds[b] = sb.joltId;
        this.scene.getBodyInterface().setLinearAndAngularVelocity(
                sb.joltId, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        this.scene.teleportObject(b, this.homeX[b], this.homeY[b], this.homeZ[b],
                0.0, 0.0, 0.0, 1.0);
        this.scene.wakeUpObject(b);
    }

    @Setup(org.openjdk.jmh.annotations.Level.Invocation)
    public void resetBodies() {
        // Park every body in the identical physical state so each measured
        // step does the same work (setup cost is excluded from measurement).
        for (int b = 0; b < BODIES; b++) {
            this.reset(b);
        }
    }

    @Benchmark
    public void step() {
        this.scene.step(0.05);
    }
}
