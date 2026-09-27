package dev.behindthescenery.sablejolt;

import com.github.stephengold.joltjni.Jolt;
import com.github.stephengold.joltjni.JoltPhysicsObject;
import dev.ryanhcode.sable.Sable;
import net.minecraft.CrashReport;
import net.minecraft.CrashReportCategory;
import net.minecraft.ReportedException;
import net.minecraft.Util;
import net.minecraft.Util.OS;
import org.jetbrains.annotations.ApiStatus;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;

/**
 * Loads the jolt-jni native library and bootstraps the Jolt Physics engine.
 * Native binaries are expected on the classpath in the standard jolt-jni layout:
 * {@code <os>/<arch>/com/github/stephengold/<libname>}, as packaged by the
 * classified {@code jolt-jni-<Platform>} artifacts.
 */
@ApiStatus.Internal
public final class JoltNative {
    private static final String LIB_NAME = "joltjni";
    private static final Path NATIVE_DIR = resolveNativeDir();

    private static boolean initialized = false;

    private JoltNative() {
    }

    private static Path resolveNativeDir() {
        final Path gameDir = dev.ryanhcode.sable.platform.SableLoaderPlatform.INSTANCE.getGameDirectory();
        if (gameDir != null) {
            return gameDir.resolve(".sable").resolve("jolt-natives").normalize();
        }
        return Paths.get(System.getProperty("user.home", System.getProperty("user.dir")), ".sable", "jolt-natives");
    }

    private static String getResourcePath() {
        final String arch = System.getProperty("os.arch").contains("arm") || System.getProperty("os.arch").startsWith("aarch64")
                ? "aarch64" : "x86-64";

        final OS os = Util.getPlatform();
        final String prefix;
        final String suffix;
        if (os == OS.WINDOWS) {
            prefix = "windows";
            suffix = LIB_NAME + ".dll";
        } else if (os == OS.OSX) {
            prefix = "osx";
            suffix = "lib" + LIB_NAME + ".dylib";
        } else {
            prefix = "linux";
            suffix = "lib" + LIB_NAME + ".so";
        }
        return prefix + "/" + arch + "/com/github/stephengold/" + suffix;
    }

    /**
     * Ensures the native library is loaded and the engine is bootstrapped.
     */
    public static synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }

        try {
            final String resourcePath = getResourcePath();
            final String fileName = resourcePath.substring(resourcePath.lastIndexOf('/') + 1);

            final byte[] bytes;
            try (final InputStream is = JoltNative.class.getResourceAsStream("/" + resourcePath)) {
                if (is == null) {
                    throw new FileNotFoundException(resourcePath + " (is the classified jolt-jni natives artifact on the classpath?)");
                }
                bytes = is.readAllBytes();
            }
            // Version the target directory with a content hash of the library:
            // two jars in the mods folder must never fight over one file, and a
            // System.load of the SAME path already loaded earlier in this JVM
            // process would silently return the OLD library (Windows reuses the
            // module by path) — the actual cause of "missing native symbol"
            // errors after a jolt-jni upgrade.
            final Path dir = NATIVE_DIR.resolve(ShortHash.of(bytes)).normalize();
            Files.createDirectories(dir);
            final Path target = dir.resolve(fileName);
            // Skip the write when the file from a previous run is still there
            // (a running game process keeps the DLL locked on Windows).
            if (!Files.exists(target) || Files.size(target) != bytes.length) {
                Files.write(target, bytes);
            }
            System.load(target.toAbsolutePath().toString());

            JoltPhysicsObject.startCleaner();
            Jolt.registerDefaultAllocator();
            Jolt.installDefaultAssertCallback();
            Jolt.installDefaultTraceCallback();
            if (!Jolt.newFactory()) {
                throw new IllegalStateException("Failed to create Jolt factory");
            }
            Jolt.registerTypes();

            initialized = true;
            Sable.LOGGER.info("Jolt Physics initialized ({}, {})", Jolt.versionString(), Jolt.buildType());
        } catch (final Throwable t) {
            Sable.LOGGER.error("Sable Jolt failed to load the Jolt natives. Please report with system details and logs to {}", Sable.ISSUE_TRACKER_URL, t);
            final CrashReport crashReport = CrashReport.forThrowable(t, "Sable linking with Jolt natives");
            final CrashReportCategory category = crashReport.addCategory("Natives");
            category.setDetail("Resource", getResourcePath());
            category.setDetail("Native Directory", NATIVE_DIR.toAbsolutePath().toString());
            throw new ReportedException(crashReport);
        }
    }

    /**
     * First 8 uppercase hex chars of the SHA-256 of a byte array.
     */
    private static final class ShortHash {
        static String of(final byte[] bytes) {
            final MessageDigest digest;
            try {
                digest = MessageDigest.getInstance("SHA-256");
            } catch (final NoSuchAlgorithmException e) {
                throw new IllegalStateException(e);
            }
            final StringBuilder hex = new StringBuilder();
            for (final byte b : digest.digest(bytes)) {
                hex.append(Character.forDigit((b >> 4) & 0xF, 16)).append(Character.forDigit(b & 0xF, 16));
            }
            return hex.substring(0, 8).toUpperCase(Locale.ROOT);
        }
    }
}
