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
 * Full compound rebuild of one body (4 solid sections, ~16k leaves):
 * section scan, leaf emission, sub-shape id mapping and the native
 * {@code setShape}. The dedup/throttle guards are reset before every
 * invocation (in setup, excluded from measurement) so each call rebuilds.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 4, time = 1)
@Fork(1)
@State(Scope.Benchmark)
public class RebuildBench {
    private JoltPhysicsScene scene;
    private JoltPhysicsScene.SableBody body;
    private Method rebuildShape;

    @Setup(org.openjdk.jmh.annotations.Level.Trial)
    public void setup() throws Exception {
        this.scene = BenchSupport.newScene();
        final int[] ids = BenchSupport.registerEntries(this.scene);
        final double cx = 32.0;
        final double cy = 71.5;
        final double cz = 7.5;
        this.scene.createSubLevel(1, new Vector3d(cx, cy, cz), new Quaterniond());
        this.scene.setLocalBounds(1, 0, 64, 0, 63, 79, 15);
        this.scene.setCenterOfMass(1, cx, cy, cz);
        final int[] data = BenchSupport.solidSection(ids[0]);
        for (int sx = 0; sx < 4; sx++) {
            this.scene.addChunk(sx, 4, 0, data, true, -1);
        }
        this.scene.step(0.05);
        this.body = this.scene.body(1);
        if (this.body.children.isEmpty()) {
            throw new IllegalStateException("body has no children");
        }
        this.rebuildShape = BenchSupport.method(JoltPhysicsScene.class,
                "rebuildShape", JoltPhysicsScene.SableBody.class);
        if (this.rebuildShape == null) {
            throw new IllegalStateException("rebuildShape not found");
        }
        // Touch the auxiliary types once so the measured calls are steady.
        this.rebuildShape.invoke(this.scene, this.body);
    }

    @Setup(org.openjdk.jmh.annotations.Level.Invocation)
    public void arm() {
        this.body.rebuiltDataVersion = -1L;
        this.body.lastShapeRebuildNanos = 0L;
    }

    @Benchmark
    public int rebuild() throws Exception {
        this.rebuildShape.invoke(this.scene, this.body);
        return this.body.children.size();
    }
}
