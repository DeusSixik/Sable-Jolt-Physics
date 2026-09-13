package dev.behindthescenery.sablejolt;

/**
 * Simple switches for verbose diagnostics. Enabled via system properties
 * {@code -Dsablejolt.debug=true} (shape rebuilds), {@code -Dsablejolt.staff=true}
 * (physics staff drag sessions) and {@code -Dsablejolt.handle=true} (handle
 * grabbing constraint diagnostics).
 */
public final class JoltDebugLogging {
    public static final boolean VERBOSE = Boolean.getBoolean("sablejolt.debug");
    public static final boolean STAFF = Boolean.getBoolean("sablejolt.staff");
    public static final boolean HANDLE = Boolean.getBoolean("sablejolt.handle");

    private JoltDebugLogging() {
    }
}
