package dev.behindthescenery.sablejolt;

/**
 * Simple switches for verbose diagnostics. Enabled via system properties
 * {@code -Dsablejolt.debug=true} (shape rebuilds) and {@code -Dsablejolt.staff=true}
 * (physics staff drag sessions).
 */
public final class JoltDebugLogging {
    public static final boolean VERBOSE = Boolean.getBoolean("sablejolt.debug");
    public static final boolean STAFF = Boolean.getBoolean("sablejolt.staff");

    private JoltDebugLogging() {
    }
}
