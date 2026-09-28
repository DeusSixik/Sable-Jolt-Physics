package dev.behindthescenery.sablejolt;

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
import org.openjdk.jmh.infra.Blackhole;

import java.lang.reflect.Method;
import java.util.concurrent.TimeUnit;

/**
 * Shared terrain-slab shape cache, steady-state hits: terrain repeats a
 * handful of slab sizes, so every lookup after warmup is a hit. The old
 * version allocated a key object per call.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@State(Scope.Benchmark)
public class SharedBoxBench {
    private static final float[][] SIZES = {
            {0.5f, 0.5f, 0.5f},
            {1.5f, 0.5f, 0.5f},
            {0.5f, 0.5f, 2.5f},
            {8.0f, 0.5f, 8.0f},
    };

    private JoltPhysicsScene scene;
    private Method sharedBoxShape;
    private int cursor;

    @Setup(org.openjdk.jmh.annotations.Level.Trial)
    public void setup() throws Exception {
        this.scene = BenchSupport.newScene();
        this.sharedBoxShape = BenchSupport.method(JoltPhysicsScene.class,
                "sharedBoxShape", float.class, float.class, float.class);
        if (this.sharedBoxShape == null) {
            throw new IllegalStateException("sharedBoxShape not found");
        }
        for (final float[] s : SIZES) {
            this.sharedBoxShape.invoke(this.scene, s[0], s[1], s[2]);
        }
    }

    @Benchmark
    public void hit(final Blackhole bh) throws Exception {
        this.cursor = (this.cursor + 1) & (SIZES.length - 1);
        final float[] s = SIZES[this.cursor];
        bh.consume(this.sharedBoxShape.invoke(this.scene, s[0], s[1], s[2]));
    }
}
