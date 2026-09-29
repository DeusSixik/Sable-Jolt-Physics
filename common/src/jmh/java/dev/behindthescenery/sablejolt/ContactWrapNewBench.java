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
 * New contact-listener entry path: cached {@code Body} wrappers keyed by
 * native address. Uses reflection so the same harness compiles against the
 * pre-optimization tree (where the method is absent and the benchmark
 * degrades to a no-op baseline instead of failing the build).
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@State(Scope.Benchmark)
public class ContactWrapNewBench {
    private static final long[] ADDRESSES = {
            0x11111111L, 0x22222222L, 0x33333333L, 0x44444444L,
            0x55555555L, 0x66666666L, 0x77777777L, 0x88888888L,
    };

    private Method wrapContactBody;
    private int cursor;

    @Setup(org.openjdk.jmh.annotations.Level.Trial)
    public void setup() throws Exception {
        BenchSupport.boot();
        this.wrapContactBody = BenchSupport.method(
                JoltPhysicsScene.class, "wrapContactBody", long.class);
        if (this.wrapContactBody != null) {
            for (final long va : ADDRESSES) {
                this.wrapContactBody.invoke(null, va);
            }
        }
    }

    @Benchmark
    public void wrap(final Blackhole bh) throws Exception {
        if (this.wrapContactBody == null) {
            bh.consume(-1);
            return;
        }
        this.cursor = (this.cursor + 1) & (ADDRESSES.length - 1);
        bh.consume(this.wrapContactBody.invoke(null, ADDRESSES[this.cursor]));
    }
}
