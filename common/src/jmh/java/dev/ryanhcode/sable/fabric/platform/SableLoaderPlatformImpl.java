package dev.ryanhcode.sable.fabric.platform;

import dev.ryanhcode.sable.platform.SableLoaderPlatform;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * <b>HEADLESS JMH SCAFFOLDING — NEVER PUT THIS ON A GAME CLASSPATH.</b>
 * <p>
 * This class intentionally shadows the real Fabric platform implementation
 * (same fully-qualified name, compiled into the {@code jmh} source set only)
 * so {@code JoltNative} can extract its natives outside a running game: the
 * real implementation calls into the Fabric Loader runtime, which does not
 * exist in benchmarks. The JMH task puts {@code src/jmh} output first on its
 * classpath, so this stub wins there — and only there.
 * <p>
 * If this class ever ended up earlier than the real one on a mod runtime
 * classpath, the game directory resolution would silently redirect to a temp
 * folder. It lives in {@code src/jmh} precisely so no jar/remap task picks
 * it up.
 */
public class SableLoaderPlatformImpl implements SableLoaderPlatform {
    public SableLoaderPlatformImpl() {
    }

    @Override
    public String getModVersion(final String modId) {
        return "0.0.0-bench";
    }

    @Override
    public Path getGameDirectory() {
        final String override = System.getProperty("bench.gamedir");
        if (override != null) {
            return Paths.get(override);
        }
        return Paths.get(System.getProperty("java.io.tmpdir"), "sable-jmh");
    }
}
