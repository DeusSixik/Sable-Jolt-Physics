package dev.behindthescenery.sablejolt;

import com.github.stephengold.joltjni.Body;
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

import java.util.concurrent.TimeUnit;

/**
 * Old contact-listener entry path: one fresh {@code Body} wrapper (plus its
 * two internal atomics) per body per contact. Compiles against both trees;
 * the replacement is measured by {@link ContactWrapNewBench}.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@State(Scope.Benchmark)
public class ContactWrapOldBench {
    private static final long[] ADDRESSES = {
            0x11111111L, 0x22222222L, 0x33333333L, 0x44444444L,
            0x55555555L, 0x66666666L, 0x77777777L, 0x88888888L,
    };

    private int cursor;

    @Setup(org.openjdk.jmh.annotations.Level.Trial)
    public void setup() {
        BenchSupport.boot();
    }

    @Benchmark
    public void wrap(final Blackhole bh) {
        this.cursor = (this.cursor + 1) & (ADDRESSES.length - 1);
        bh.consume(new Body(ADDRESSES[this.cursor]));
    }
}
