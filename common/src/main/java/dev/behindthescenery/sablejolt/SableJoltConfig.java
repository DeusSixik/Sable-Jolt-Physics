package dev.behindthescenery.sablejolt;

import dev.ryanhcode.sable.Sable;
import dev.ryanhcode.sable.platform.SableLoaderPlatform;
import org.jetbrains.annotations.ApiStatus;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.Properties;

/**
 * Loader-independent configuration, stored as {@code config/sable_jolt_physics.properties}.
 * <p>
 * Read once, the first time a physics pipeline is created; edits apply the next
 * time a world is loaded. Unknown or malformed values fall back to defaults, and
 * keys missing from an older file are added on load.
 */
@ApiStatus.Internal
public final class SableJoltConfig {
    public enum Backend {
        JOLT,
        RAPIER
    }

    private static final String FILE_NAME = "sable_jolt_physics.properties";

    private static volatile SableJoltConfig instance;

    /**
     * Physics engine used for Sable sub-levels.
     */
    public final Backend backend;

    /**
     * Reproduce the Rapier backend's behavior where it affects builds (buoyancy,
     * friction, sleeping, motor response, no speed limits on large bodies).
     */
    public final boolean rapierCompatiblePhysics;

    public final int maxBodies;
    public final int maxBodyPairs;
    public final int maxContactConstraints;

    /**
     * Native solver threads (0 = automatic).
     */
    public final int physicsThreads;

    private SableJoltConfig(final Properties props) {
        this.backend = parseBackend(props.getProperty("backend"));
        this.rapierCompatiblePhysics = parseBoolean(props.getProperty("rapier_compatible_physics"), true);
        this.maxBodies = parseInt(props.getProperty("max_bodies"), 262_144, 16_384, 4_000_000);
        this.maxBodyPairs = parseInt(props.getProperty("max_body_pairs"), 262_144, 16_384, 4_000_000);
        this.maxContactConstraints = parseInt(props.getProperty("max_contact_constraints"), 65_536, 8_192, 2_000_000);
        this.physicsThreads = parseInt(props.getProperty("physics_threads"), 0, 0, 256);
    }

    public static SableJoltConfig get() {
        SableJoltConfig config = instance;
        if (config == null) {
            synchronized (SableJoltConfig.class) {
                config = instance;
                if (config == null) {
                    config = load();
                    instance = config;
                }
            }
        }
        return config;
    }

    private static Path configFile() {
        Path gameDir = null;
        try {
            gameDir = SableLoaderPlatform.INSTANCE.getGameDirectory();
        } catch (final Throwable ignored) {
            // fall through to the working directory
        }
        final Path base = gameDir != null ? gameDir : Paths.get(".");
        return base.resolve("config").resolve(FILE_NAME).normalize();
    }

    private static SableJoltConfig load() {
        final Path file = configFile();
        final Properties props = new Properties();
        boolean existed = false;
        if (Files.isRegularFile(file)) {
            existed = true;
            try (final Reader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                props.load(reader);
            } catch (final IOException | IllegalArgumentException e) {
                Sable.LOGGER.error("[SableJolt] Failed to read {}, using defaults", file, e);
            }
        }

        final SableJoltConfig config = new SableJoltConfig(props);

        // Rewrite when the file is missing or lacks keys added by newer versions,
        // keeping the user's values.
        if (!existed || !props.stringPropertyNames().containsAll(KEYS)) {
            try {
                Files.createDirectories(file.getParent());
                Files.writeString(file, config.render(), StandardCharsets.UTF_8);
            } catch (final IOException e) {
                Sable.LOGGER.warn("[SableJolt] Could not write {}", file, e);
            }
        }
        return config;
    }

    private static final java.util.List<String> KEYS = java.util.List.of(
            "backend", "rapier_compatible_physics", "max_bodies", "max_body_pairs",
            "max_contact_constraints", "physics_threads");

    private String render() {
        return """
                # Sable Jolt Physics configuration.
                # Changes apply the next time a world is loaded.

                # Physics engine used for Sable sub-levels: jolt or rapier.
                # Sable saves sub-levels (blocks, position, rotation, velocity) in an engine-independent
                # format, so you can switch in both directions without losing contraptions. Before removing
                # this addon, set this to rapier and load the world once to verify everything.
                backend=%s

                # Make the Jolt engine reproduce Sable's Rapier behavior where it matters for builds:
                # buoyancy, contact friction, sleeping, motor response and no speed limit on large bodies.
                # Recommended for worlds with builds made before this addon was installed.
                rapier_compatible_physics=%s

                # Capacity of the physics world of each dimension. Terrain near sub-levels uses one body per
                # 16x16x16 section. The log warns when a limit is reached; raise the reported value then.
                # Higher values use more memory per dimension.
                max_bodies=%d
                max_body_pairs=%d
                max_contact_constraints=%d

                # Native physics worker threads shared by all dimensions (0 = automatic).
                physics_threads=%d
                """.formatted(
                this.backend.name().toLowerCase(Locale.ROOT),
                this.rapierCompatiblePhysics,
                this.maxBodies,
                this.maxBodyPairs,
                this.maxContactConstraints,
                this.physicsThreads);
    }

    private static Backend parseBackend(final String value) {
        if (value == null || value.isBlank()) {
            return Backend.JOLT;
        }
        try {
            return Backend.valueOf(value.trim().toUpperCase(Locale.ROOT));
        } catch (final IllegalArgumentException e) {
            Sable.LOGGER.warn("[SableJolt] Unknown backend '{}', using jolt", value);
            return Backend.JOLT;
        }
    }

    private static boolean parseBoolean(final String value, final boolean def) {
        if (value == null || value.isBlank()) {
            return def;
        }
        final String v = value.trim().toLowerCase(Locale.ROOT);
        if (v.equals("true") || v.equals("yes") || v.equals("1")) {
            return true;
        }
        if (v.equals("false") || v.equals("no") || v.equals("0")) {
            return false;
        }
        return def;
    }

    private static int parseInt(final String value, final int def, final int min, final int max) {
        if (value == null || value.isBlank()) {
            return def;
        }
        try {
            return Math.max(min, Math.min(max, Integer.parseInt(value.trim())));
        } catch (final NumberFormatException e) {
            return def;
        }
    }
}
