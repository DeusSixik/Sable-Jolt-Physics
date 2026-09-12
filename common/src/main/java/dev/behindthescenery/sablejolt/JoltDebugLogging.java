package dev.behindthescenery.sablejolt;

/**
 * Simple switches for verbose diagnostics. Enabled via system property
 * {@code -Dsablejolt.debug=true}.
 */
public final class JoltDebugLogging {
    public static final boolean VERBOSE = Boolean.getBoolean("sablejolt.debug");

    private JoltDebugLogging() {
    }
}
