package dev.behindthescenery.sablejolt;

import dev.behindthescenery.sablejolt.collider.JoltVoxelColliderData;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;

/**
 * Collider registry reads under contention: rebuilds, buoyancy and the
 * contact listener hammer {@code Registry.get} from several threads while
 * the old version held a monitor on every call.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@Threads(4)
@State(Scope.Benchmark)
public class RegistryBench {
    private static final int ENTRIES = 256;

    private JoltVoxelColliderData.Registry registry;

    /** Per-thread cursor: threads must contend on the registry, not on harness state. */
    @State(Scope.Thread)
    public static class Cursor {
        int idx;
    }

    @Setup(org.openjdk.jmh.annotations.Level.Trial)
    public void setup() {
        this.registry = new JoltVoxelColliderData.Registry();
        for (int i = 0; i < ENTRIES; i++) {
            final JoltVoxelColliderData data = this.registry.create(
                    1.0, 1.0, 0.0, (i & 1) == 0, null, null);
            data.addBox(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        }
    }

    @Benchmark
    public void get(final Cursor cursor, final Blackhole bh) {
        cursor.idx = (cursor.idx + 1) & (ENTRIES - 1);
        bh.consume(this.registry.get(cursor.idx));
    }
}
