package dev.behindthescenery.sablejolt;

import com.github.stephengold.joltjni.*;
import com.github.stephengold.joltjni.enumerate.EActivation;
import com.github.stephengold.joltjni.enumerate.EMotionQuality;
import com.github.stephengold.joltjni.enumerate.EMotionType;
import com.github.stephengold.joltjni.enumerate.EOverrideMassProperties;
import com.github.stephengold.joltjni.enumerate.EPhysicsUpdateError;
import com.github.stephengold.joltjni.enumerate.ESpringMode;
import com.github.stephengold.joltjni.readonly.ConstBodyIdArray;
import com.github.stephengold.joltjni.readonly.ConstShape;
import dev.behindthescenery.sablejolt.collider.JoltVoxelColliderBakery;
import dev.behindthescenery.sablejolt.collider.JoltVoxelColliderData;
import dev.ryanhcode.sable.Sable;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.longs.LongArrayList;
import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import it.unimi.dsi.fastutil.objects.ReferenceOpenHashSet;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.Nullable;
import org.joml.*;

import java.lang.Math;
import java.lang.Runtime;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.IntConsumer;

/**
 * A Jolt physics scene: the direct equivalent of the native rapier {@code PhysicsScene}.
 * <p>
 * Every sub-level is a dynamic body whose compound shape is built from voxel
 * chunk data, one static compound per 16³ section under a small mutable parent
 * (sections are rebuilt individually when their blocks change); the global
 * world is stored as one static body per chunk section. Kinematic contraptions
 * mounted on a sub-level are part of the sub-level's compound, as in Rapier.
 */
@ApiStatus.Internal
public final class JoltPhysicsScene {
    public static final int LAYER_MOVING = 0;
    public static final int LAYER_STATIC = 1;
    public static final int LAYER_ROPE = 2;
    /**
     * Moving bodies whose contacts are forwarded to the Java contact listener:
     * bodies with special blocks or mounted contraptions, dragged bodies and
     * bodies fast enough for an impact report. Collides exactly like
     * {@link #LAYER_MOVING}; the split only saves Java upcalls, which cost
     * ~80% of a step when thousands of bodies move.
     */
    public static final int LAYER_MOVING_REPORT = 3;
    /**
     * Terrain sections with special blocks (friction, restitution, callbacks).
     */
    public static final int LAYER_STATIC_SPECIAL = 4;
    private static final int NUM_OBJECT_LAYERS = 5;

    /**
     * Below this speed bound (m/s) a body cannot take part in an impact
     * report: reports need a relative normal speed of 1.25 m/s (the
     * pipeline's 25 * minMass force threshold), i.e. 0.625 m/s per body.
     */
    private static final float REPORT_SPEED = 0.55f;

    /**
     * Bitmask of all DOFs, used when overriding mass properties.
     */
    private static final int ALLOWED_DOFS_ALL = 0x3F;

    /**
     * Density of fluids in per-block mass units for the Jolt buoyancy model.
     * Buoyancy is gravity-proportional (F = density * submergedVolume * |g|),
     * so a body floats when its average block mass is below this value and
     * sinks when it is above.
     */
    private static final double FLUID_DENSITY = 2.0;

    /**
     * Upward force per submerged block volume of the Rapier buoyancy model
     * (buoyancy.rs), independent of gravity.
     */
    private static final double RAPIER_FLOAT_FORCE = 10.5;

    /**
     * Fluid drag coefficient (both buoyancy models).
     */
    private static final double FLUID_DRAG = 1.7;

    /**
     * Friction of the Rapier colliders (lib.rs / contraptions.rs / boxes.rs /
     * rope.rs; the level collider uses Rapier's default 0.5).
     */
    private static final float FRICTION_SUB_LEVEL = 0.525f;
    private static final float FRICTION_CONTRAPTION = 0.45f;
    private static final float FRICTION_BOX = 0.45f;
    private static final float FRICTION_ROPE = 0.15f;
    private static final float FRICTION_TERRAIN_RAPIER = 0.5f;

    /**
     * Speed limits of sub-level bodies in Rapier-compatible mode (Rapier has
     * none; Jolt's defaults would cap fast propellers at 450 rpm).
     */
    private static final float RAPIER_MAX_LINEAR_VELOCITY = 2_000.0f;
    private static final float RAPIER_MAX_ANGULAR_VELOCITY = 400.0f;

    private static final int CPU_COUNT = Runtime.getRuntime().availableProcessors();

    // region Shared native resources

    /**
     * One native job system for every dimension: per-scene pools multiplied
     * native threads by the dimension count and were never shut down.
     */
    private static JobSystem sharedJobSystem;
    private static ExecutorService sharedWorkers;
    private static int sharedWorkerCount = 1;

    private static synchronized JobSystem sharedJobSystem(final int configuredThreads) {
        if (sharedJobSystem == null) {
            final int threads = configuredThreads > 0 ? configuredThreads : Math.max(1, CPU_COUNT - 1);
            // Extra barriers so dimensions ticked in parallel by other mods never starve.
            sharedJobSystem = new JobSystemThreadPool(Jolt.cMaxPhysicsJobs, Jolt.cMaxPhysicsBarriers * 2, threads);
        }
        return sharedJobSystem;
    }

    private static synchronized ExecutorService sharedWorkers() {
        if (sharedWorkers == null) {
            sharedWorkerCount = Math.max(1, CPU_COUNT - 2);
            final AtomicInteger counter = new AtomicInteger();
            sharedWorkers = Executors.newFixedThreadPool(sharedWorkerCount, r -> {
                final Thread t = new Thread(r, "sable-jolt-worker-" + counter.incrementAndGet());
                t.setDaemon(true);
                return t;
            });
        }
        return sharedWorkers;
    }

    // endregion

    private final SableJoltConfig config;

    /**
     * Rapier-compatible behavior (see {@link SableJoltConfig#rapierCompatiblePhysics}).
     */
    private final boolean rapierMode;

    private final PhysicsSystem system;
    private final BodyInterface bi;

    /**
     * Shared BoxShapes keyed by half extents. Terrain slabs repeat the same few
     * sizes across thousands of sections and rope points repeat one size per
     * rope; Jolt shapes are immutable and reference-counted, so one native
     * instance per size replaces one instance per slab (native memory saver).
     */
    private final Object2ObjectOpenHashMap<BoxKey, BoxShape> sharedBoxShapes = new Object2ObjectOpenHashMap<>();

    private static final class BoxKey {
        private float hx;
        private float hy;
        private float hz;

        private BoxKey() {
            this(0, 0, 0);
        }

        private BoxKey(final float hx, final float hy, final float hz) {
            this.hx = hx;
            this.hy = hy;
            this.hz = hz;
        }

        private BoxKey set(final float hx, final float hy, final float hz) {
            this.hx = hx;
            this.hy = hy;
            this.hz = hz;
            return this;
        }

        @Override
        public boolean equals(final Object obj) {
            if (obj == this) return true;
            if (!(obj instanceof final BoxKey that)) return false;
            return this.hx == that.hx &&
                    this.hy == that.hy &&
                    this.hz == that.hz;
        }

        @Override
        public int hashCode() {
            int result = Float.floatToRawIntBits(this.hx);
            result = 31 * result + Float.floatToRawIntBits(this.hy);
            result = 31 * result + Float.floatToRawIntBits(this.hz);
            return result;
        }

        @Override
        public String toString() {
            return "BoxKey[hx=" + this.hx + ", hy=" + this.hy + ", hz=" + this.hz + ']';
        }
    }

    private static final ThreadLocal<BoxKey> BOX_KEY = ThreadLocal.withInitial(BoxKey::new);

    private BoxShape sharedBoxShape(final float hx, final float hy, final float hz) {
        // Fast path: probe with a reusable ThreadLocal key (no allocation on hit;
        // hits dominate — terrain slabs repeat a handful of sizes). The probe is
        // never stored: on miss a dedicated immutable copy becomes the map key,
        // so later set() calls cannot corrupt the stored hash.
        final BoxKey probe = BOX_KEY.get().set(hx, hy, hz);
        final BoxShape existing = this.sharedBoxShapes.get(probe);
        if (existing != null) {
            return existing;
        }
        final BoxShape created = newBoxShape(hx, hy, hz, 0.02f);
        final BoxShape raced = this.sharedBoxShapes.putIfAbsent(new BoxKey(hx, hy, hz), created);
        return raced != null ? raced : created;
    }

    /**
     * Creates a box shape whose convex radius never exceeds its smallest half
     * extent (Jolt only asserts this in debug builds; a larger radius turns
     * the support function inside out and produces garbage contacts).
     */
    public static BoxShape newBoxShape(final float hx, final float hy, final float hz, final float convexRadius) {
        final float ex = Math.max(hx, 1.0e-4f);
        final float ey = Math.max(hy, 1.0e-4f);
        final float ez = Math.max(hz, 1.0e-4f);
        final float radius = Math.min(convexRadius, 0.5f * Math.min(ex, Math.min(ey, ez)));
        return new BoxShape(ex, ey, ez, radius);
    }

    /**
     * Reused {@code MassProperties}/{@code Mat44} for {@link #setMassProperties}
     * (server thread only): avoids a native allocation pair per body per tick.
     */
    private final MassProperties massPropsScratch = new MassProperties();
    private final Mat44 inertiaScratch = new Mat44(
            0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f,
            0f, 0f, 0f, 1f);

    /**
     * Lock-free batch reads for the snapshot (nothing else touches bodies
     * between steps).
     */
    private final BatchBodyInterface batchNoLock;
    private BodyIdArray snapshotIds;
    private long snapshotGen = 1;

    /**
     * Bodies currently dragged by the creative grip (physics staff): gravity is
     * switched off so weight can neither pull nor tip the body over, while the
     * body stays dynamic — contacts still resolve, so it cannot be pushed
     * through the terrain.
     */
    private final IntOpenHashSet creativeDragBodies = new IntOpenHashSet();
    /**
     * Dragged bodies whose handle was removed; gravity is restored at the next
     * step start. Deferred because the staff recreates its constraint every
     * tick — an immediate restore would flap every tick.
     */
    private final IntOpenHashSet creativeRestorePending = new IntOpenHashSet();

    /**
     * Starts (or confirms) the creative drag of a body. Idempotent per grab.
     */
    public void beginCreativeDrag(final int joltId) {
        this.creativeRestorePending.remove(joltId);
        if (this.creativeDragBodies.add(joltId)) {
            final SableBody sb = this.bodiesByJoltId.get(joltId);
            if (sb != null && sb.body != null) {
                final var motion = sb.body.getMotionProperties();
                if (motion != null) {
                    motion.setGravityFactor(0.0f);
                }
                // The push-out needs every contact of the dragged body.
                this.setBodyLayer(sb, LAYER_MOVING_REPORT);
            }
        }
    }

    /**
     * Marks the end of a creative drag; gravity is restored in
     * {@link #flushCreativeRestore} at the next step start.
     */
    public void endCreativeDrag(final int joltId) {
        if (this.creativeDragBodies.contains(joltId)) {
            this.creativeRestorePending.add(joltId);
        }
    }

    public boolean isCreativeDrag(final int joltId) {
        return this.creativeDragBodies.contains(joltId);
    }

    private void flushCreativeRestore() {
        if (this.creativeRestorePending.isEmpty()) {
            return;
        }
        for (final int joltId : this.creativeRestorePending) {
            this.creativeDragBodies.remove(joltId);
            try {
                final SableBody sb = this.bodiesByJoltId.get(joltId);
                if (sb != null && sb.body != null) {
                    final var motion = sb.body.getMotionProperties();
                    if (motion != null) {
                        motion.setGravityFactor(1.0f);
                    }
                }
            } catch (final Throwable ignored) {
                // body may be gone already
            }
        }
        this.creativeRestorePending.clear();
    }

    private DoubleBuffer snapshotPos;
    private FloatBuffer snapshotRot;
    private FloatBuffer snapshotLin;
    private FloatBuffer snapshotAng;
    private ByteBuffer snapshotActive;

    public BodyInterface getBodyInterface() {
        return this.bi;
    }

    private final TempAllocator tempAllocator;
    private final JobSystem jobSystem;
    private final JobSystem singleThreadJobSystem;
    private static final int MULTITHREAD_BODY_THRESHOLD = 24;

    /**
     * A body with at least this many compound leaves makes the native narrow
     * phase dominate the step; such scenes use the multithreaded job system
     * regardless of the body count.
     */
    private static final int HEAVY_BODY_CHILDREN = 2048;

    /**
     * Jolt buoyancy iterates the body's blocks per substep; above this count the
     * loop is subsampled (with volume compensation) so a 30k-block body costs
     * the same as a 4k-block one.
     */
    private static final int BUOYANCY_MAX_CHILDREN = 4096;

    /**
     * Strong reference: the native physics system only holds a raw pointer.
     */
    private final SceneContactListener contactListener;
    /**
     * Native filter deciding which contact pairs reach {@link #contactListener}.
     */
    private final ObjectLayerPairFilterTable listenerFilter;

    public final JoltVoxelColliderData.Registry colliderRegistry = new JoltVoxelColliderData.Registry();

    /**
     * Set by the pipeline. Kept for API compatibility; box (re)building happens
     * on the server thread in the pipeline, never during rebuilds.
     */
    private volatile JoltVoxelColliderBakery colliderBakery;

    public void attachColliderBakery(final JoltVoxelColliderBakery bakery) {
        this.colliderBakery = bakery;
    }

    private final Int2ObjectOpenHashMap<SableBody> bodies = new Int2ObjectOpenHashMap<>();
    private final Int2ObjectOpenHashMap<SableBody> bodiesByJoltId = new Int2ObjectOpenHashMap<>();
    /**
     * Bodies with a native body, dense; drives the batch snapshot.
     */
    private final ObjectArrayList<SableBody> bodyList = new ObjectArrayList<>();
    private final ObjectArrayList<SableBody> contraptionList = new ObjectArrayList<>();
    private final ObjectArrayList<SableBody> pendingCreation = new ObjectArrayList<>();
    private final Long2ObjectOpenHashMap<GlobalChunk> globalChunks = new Long2ObjectOpenHashMap<>();
    private final Int2ObjectOpenHashMap<GlobalChunk> globalChunksByBodyId = new Int2ObjectOpenHashMap<>();
    private final Long2ObjectOpenHashMap<ChunkSectionData> allChunks = new Long2ObjectOpenHashMap<>();
    /**
     * Plot section key → runtime id of the sub-level owning it (-1 when absent;
     * runtime id 0 is valid).
     */
    private final Long2IntOpenHashMap sectionOwners = new Long2IntOpenHashMap();

    {
        this.sectionOwners.defaultReturnValue(-1);
    }
    private final Long2ObjectOpenHashMap<JointRecord> joints = new Long2ObjectOpenHashMap<>();
    private final Long2ObjectOpenHashMap<RopeStrand> ropes = new Long2ObjectOpenHashMap<>();

    private final Int2ObjectOpenHashMap<Quaterniond> angularFollowPrev = new Int2ObjectOpenHashMap<>();
    private final Int2ObjectOpenHashMap<Vector3d> angularCachedOmega = new Int2ObjectOpenHashMap<>();

    public Quaterniond getAngularFollowPrev(final int joltId) {
        return this.angularFollowPrev.get(joltId);
    }

    public void setAngularFollowPrev(final int joltId, final Quaterniond q) {
        this.angularFollowPrev.computeIfAbsent(joltId, k -> new Quaterniond()).set(q);
    }

    public Vector3d getCachedAngularOmega(final int joltId) {
        return this.angularCachedOmega.get(joltId);
    }

    public void setCachedAngularOmega(final int joltId, final Vector3d omega) {
        this.angularCachedOmega.computeIfAbsent(joltId, k -> new Vector3d()).set(omega);
    }

    public void clearAngularServo(final int joltId) {
        this.angularFollowPrev.remove(joltId);
        this.angularCachedOmega.remove(joltId);
    }

    private final AtomicInteger nextRuntimeId = new AtomicInteger();
    private final AtomicLong nextJointId = new AtomicLong(1);
    private final AtomicLong nextRopeId = new AtomicLong(1);

    private final ObjectArrayList<double[]> reportedCollisions = new ObjectArrayList<>();

    /**
     * Guards reportedCollisions: Jolt fires contact callbacks from worker threads
     * when the solver runs multithreaded.
     */
    private final Object reportedLock = new Object();

    /**
     * Bodies and global chunks whose shapes must be rebuilt; block edits mark them
     * dirty and the actual rebuild runs once per simulation step.
     */
    private final LinkedHashSet<SableBody> dirtyBodies = new LinkedHashSet<>();
    private final LinkedHashSet<GlobalChunk> dirtyGlobalChunks = new LinkedHashSet<>();
    /**
     * Global chunks whose body could not be created (body limit reached); retried.
     */
    private final ReferenceOpenHashSet<GlobalChunk> pendingTerrainBodies = new ReferenceOpenHashSet<>();

    private int groundBodyId;
    private boolean disposed;

    final double gravityX;
    final double gravityY;
    final double gravityZ;
    final double universalDrag;

    /**
     * Velocity/position solver steps requested by Sable's config; the heavy-body
     * reduction restores these.
     */
    private int configuredVelocitySteps = 10;
    private int configuredPositionSteps = 4;

    JoltPhysicsScene(final double gravityX, final double gravityY, final double gravityZ, final double universalDrag) {
        JoltNative.ensureInitialized();

        this.config = SableJoltConfig.get();
        this.rapierMode = this.config.rapierCompatiblePhysics;

        this.gravityX = gravityX;
        this.gravityY = gravityY;
        this.gravityZ = gravityZ;
        this.universalDrag = universalDrag;

        final int numObjLayers = NUM_OBJECT_LAYERS;
        final int numBpLayers = 2;

        final ObjectLayerPairFilterTable ovoFilter = new ObjectLayerPairFilterTable(numObjLayers);
        for (final int moving : new int[]{LAYER_MOVING, LAYER_MOVING_REPORT}) {
            ovoFilter.enableCollision(moving, LAYER_MOVING);
            ovoFilter.enableCollision(moving, LAYER_MOVING_REPORT);
            ovoFilter.enableCollision(moving, LAYER_STATIC);
            ovoFilter.enableCollision(moving, LAYER_STATIC_SPECIAL);
            ovoFilter.enableCollision(moving, LAYER_ROPE);
        }
        ovoFilter.enableCollision(LAYER_STATIC, LAYER_ROPE);
        ovoFilter.enableCollision(LAYER_STATIC_SPECIAL, LAYER_ROPE);

        final BroadPhaseLayerInterfaceTable layerMap = new BroadPhaseLayerInterfaceTable(numObjLayers, numBpLayers);
        layerMap.mapObjectToBroadPhaseLayer(LAYER_MOVING, 0);
        layerMap.mapObjectToBroadPhaseLayer(LAYER_MOVING_REPORT, 0);
        layerMap.mapObjectToBroadPhaseLayer(LAYER_ROPE, 0);
        layerMap.mapObjectToBroadPhaseLayer(LAYER_STATIC, 1);
        layerMap.mapObjectToBroadPhaseLayer(LAYER_STATIC_SPECIAL, 1);

        final ObjectVsBroadPhaseLayerFilterTable ovbFilter = createBroadPhaseFilter(layerMap, numBpLayers, ovoFilter, numObjLayers);

        // Contacts reach Java only when one side is a report body or special
        // terrain touches a moving body; plain resting and slow contacts are
        // handled entirely in native code.
        this.listenerFilter = new ObjectLayerPairFilterTable(numObjLayers);
        for (final int layer : new int[]{LAYER_MOVING, LAYER_MOVING_REPORT, LAYER_STATIC, LAYER_STATIC_SPECIAL, LAYER_ROPE}) {
            this.listenerFilter.enableCollision(LAYER_MOVING_REPORT, layer);
        }
        this.listenerFilter.enableCollision(LAYER_STATIC_SPECIAL, LAYER_MOVING);
        this.listenerFilter.enableCollision(LAYER_STATIC_SPECIAL, LAYER_ROPE);

        this.system = new PhysicsSystem();
        // The system keeps references to the layer tables; capacities come from
        // the config (terrain uses one body per uploaded section).
        this.system.init(this.config.maxBodies, 0, this.config.maxBodyPairs, this.config.maxContactConstraints,
                layerMap, ovbFilter, ovoFilter);
        this.system.setGravity((float) gravityX, (float) gravityY, (float) gravityZ);

        // Contact tuning for voxel-compound sliding. The original Rapier build
        // lets a block platform glide over flat ground; Jolt's stiffer solver
        // snags on the internal seams between adjacent ground boxes. Softer
        // margins + gentler position correction + aggressive internal edge
        // removal reproduce the glide.
        final PhysicsSettings physics = this.system.getPhysicsSettings();
        physics.setPenetrationSlop(0.03f);
        physics.setSpeculativeContactDistance(0.06f);
        physics.setNumVelocitySteps(this.configuredVelocitySteps);
        physics.setNumPositionSteps(this.configuredPositionSteps);
        physics.setBaumgarte(0.12f);
        // default 1e-8 m² is too strict for adjacent box seams; 1e-4 m² (1 cm)
        // treats near-coplanar edges as internal and removes their spikes.
        physics.setInternalEdgeRemovalVertexToleranceSq(1.0e-4f);
        // Big resting structures form huge contact islands; the splitter parallelizes
        // and bounds their solve instead of one monolithic island iteration.
        physics.setUseLargeIslandSplitter(true);
        physics.setUseManifoldReduction(true);
        if (this.rapierMode) {
            // Rapier sub-levels sleep below 0.15 m/s and 0.15 rad/s after 2 s.
            physics.setPointVelocitySleepThreshold(0.15f);
            physics.setTimeBeforeSleep(2.0f);
        }
        this.system.setPhysicsSettings(physics);

        this.bi = this.system.getBodyInterface();
        this.batchNoLock = this.system.getBodyInterfaceNoLock();
        // Fixed block for the common case; large islands fall back to malloc
        // instead of aborting the process when the block runs out.
        this.tempAllocator = new TempAllocatorImplWithMallocFallback(8 * 1024 * 1024);
        this.jobSystem = sharedJobSystem(this.config.physicsThreads);
        this.singleThreadJobSystem = new JobSystemSingleThreaded(Jolt.cMaxPhysicsJobs);

        this.contactListener = new SceneContactListener();
        this.contactListener.setObjectLayerPairFilter(this.listenerFilter);
        // No body / broad-phase filters (a Skip mode would drop every callback).
        this.contactListener.setBodyFilterMode(com.github.stephengold.joltjni.enumerate.EFilterMode.Either);
        this.contactListener.setBroadPhaseLayerFilterMode(com.github.stephengold.joltjni.enumerate.EFilterMode.Either);
        this.contactListener.setEnableAdded(true);
        this.contactListener.setEnablePersisted(true);
        this.contactListener.setEnableRemoved(false);
        this.contactListener.setEnableValidate(false);
        this.system.setContactListener(this.contactListener);

        this.createGround();
    }

    public boolean isRapierCompatible() {
        return this.rapierMode;
    }

    /**
     * Builds the object-vs-broad-phase filter and verifies it against the pair
     * filter. jolt-jni 6.1.1 reads the two layer counts of this table in the
     * opposite order of its Java parameters, which leaves every object layer
     * above 1 colliding with nothing (rope points fell through everything).
     * Whichever order produces the correct table is used.
     */
    private static ObjectVsBroadPhaseLayerFilterTable createBroadPhaseFilter(final BroadPhaseLayerInterfaceTable layerMap, final int numBpLayers,
                                                                             final ObjectLayerPairFilterTable pairFilter, final int numObjLayers) {
        final ObjectVsBroadPhaseLayerFilterTable swapped = new ObjectVsBroadPhaseLayerFilterTable(layerMap, numObjLayers, pairFilter, numBpLayers);
        if (matches(swapped, layerMap, numBpLayers, pairFilter, numObjLayers)) {
            return swapped;
        }
        final ObjectVsBroadPhaseLayerFilterTable documented = new ObjectVsBroadPhaseLayerFilterTable(layerMap, numBpLayers, pairFilter, numObjLayers);
        if (matches(documented, layerMap, numBpLayers, pairFilter, numObjLayers)) {
            return documented;
        }
        Sable.LOGGER.error("[SableJolt] Could not build a consistent broad-phase layer filter; some layers may not collide");
        return swapped;
    }

    private static boolean matches(final ObjectVsBroadPhaseLayerFilterTable table, final BroadPhaseLayerInterfaceTable layerMap, final int numBpLayers,
                                   final ObjectLayerPairFilterTable pairFilter, final int numObjLayers) {
        for (int layer = 0; layer < numObjLayers; layer++) {
            for (int bp = 0; bp < numBpLayers; bp++) {
                boolean expected = false;
                for (int other = 0; other < numObjLayers && !expected; other++) {
                    expected = layerMap.getBroadPhaseLayer(other) == bp && pairFilter.shouldCollide(layer, other);
                }
                if (table.shouldCollide(layer, bp) != expected) {
                    return false;
                }
            }
        }
        return true;
    }

    private Body groundBody;

    private void createGround() {
        // The ground body MUST sit at the world origin: constraint anchors on the
        // world are measured from its center of mass, and a shifted origin would
        // add a phantom offset to every motor target.
        final BodyCreationSettings bcs = new BodyCreationSettings(
                newBoxShape(0.01f, 0.01f, 0.01f, 0.001f),
                0f, 0f, 0f, 0f, 0f, 0f, 1f, EMotionType.Static, LAYER_STATIC);
        final Body body = this.bi.createBody(bcs);
        body.setUserData(-1L);
        this.bi.addBody(body, EActivation.DontActivate);
        this.groundBodyId = body.getId();
        this.groundBody = body;
    }

    public int groundBodyId() {
        return this.groundBodyId;
    }

    //region Body records

    /**
     * One collision leaf: a box of the block at (bx, by, bz). Coordinates are in
     * the owner's block space (plot or contraption-local), world space for terrain.
     */
    public static final class Child {
        public final int bx;
        public final int by;
        public final int bz;
        public final int colliderId;
        /**
         * The body the block belongs to; null for terrain.
         */
        @Nullable
        public final SableBody owner;

        public Child(final int bx, final int by, final int bz, final int colliderId, @Nullable final SableBody owner) {
            this.bx = bx;
            this.by = by;
            this.bz = bz;
            this.colliderId = colliderId;
            this.owner = owner;
        }
    }

    private static final BodySection[] NO_SECTIONS = new BodySection[0];
    private static final SableBody[] NO_BODIES = new SableBody[0];
    private static final Child[] NO_CHILDREN = new Child[0];
    private static final int[] NO_INTS = new int[0];
    private static final float[] NO_FLOATS = new float[0];

    /**
     * The built shape of one 16³ section of a body. The shape's local origin is
     * placed at (ox, oy, oz) in the body's block space, so the section never has
     * to be rebuilt when the body's center of mass moves.
     */
    static final class BodySection {
        final long key;
        final ConstShape shape;
        /**
         * Keeps a created static compound alive while cached.
         */
        @Nullable
        final ShapeRefC ref;
        final double ox;
        final double oy;
        final double oz;
        final Child[] children;
        /**
         * Sub-shape ID bits of the section compound; 0 for a single leaf.
         */
        final int leafBits;
        /**
         * Unique solid blocks (x, y, z triples) and their buoyancy volume factor.
         */
        final int[] blocks;
        final float[] volumes;
        /**
         * Holds blocks with special contact behavior (friction, restitution, callbacks).
         */
        final boolean special;

        BodySection(final long key, final ConstShape shape, @Nullable final ShapeRefC ref,
                    final double ox, final double oy, final double oz,
                    final Child[] children, final int leafBits, final int[] blocks, final float[] volumes, final boolean special) {
            this.key = key;
            this.shape = shape;
            this.ref = ref;
            this.ox = ox;
            this.oy = oy;
            this.oz = oz;
            this.children = children;
            this.leafBits = leafBits;
            this.blocks = blocks;
            this.volumes = volumes;
            this.special = special;
        }

        @Nullable
        Child leaf(final int relativeId) {
            if (this.leafBits == 0) {
                return this.children.length == 1 ? this.children[0] : null;
            }
            final int j = relativeId & ((1 << this.leafBits) - 1);
            return j < this.children.length ? this.children[j] : null;
        }
    }

    /**
     * Number of sub-shape ID bits a compound with {@code n} children uses
     * (Jolt's {@code CompoundShape::GetSubShapeIDBits}).
     */
    static int subShapeBits(final int n) {
        return n <= 1 ? 0 : 32 - Integer.numberOfLeadingZeros(n - 1);
    }

    /**
     * Bookkeeping for a single rigid body managed by this scene.
     */
    public static final class SableBody {
        public enum Kind {SUB_LEVEL, BOX, CONTRAPTION}

        public final Kind kind;
        public final int runtimeId;
        /**
         * Native body id; 0 while the body has no native body (body limit
         * reached, or a contraption embedded in its mount's compound).
         */
        public int joltId;

        /**
         * The Java wrapper of the native body, kept alive for the body's lifetime.
         * {@code new Body(long)} takes a virtual address, not a body ID, so the
         * wrapper from {@code createBody} must be retained instead of reconstructed.
         */
        public Body body;
        int listIndex = -1;
        /**
         * Current object layer of the native body.
         */
        int layer = LAYER_MOVING;

        /**
         * The assembled parent compound (sections, then embedded contraptions).
         */
        public MutableCompoundShape shape;
        /**
         * Every collision leaf, flattened in section order.
         */
        public final ObjectArrayList<Child> children = new ObjectArrayList<>();

        /**
         * Own chunk store of kinematic contraptions (they have no plot java-side).
         */
        public final Long2ObjectOpenHashMap<ChunkSectionData> chunks = new Long2ObjectOpenHashMap<>();

        public int minX;
        public int minY;
        public int minZ;
        public int maxX;
        public int maxY;
        public int maxZ;
        public boolean hasBounds;

        public final Vector3d centerOfMass = new Vector3d();

        // Section cache (see rebuildShape)
        final Long2ObjectOpenHashMap<BodySection> sections = new Long2ObjectOpenHashMap<>();
        /**
         * Plot sections this sub-level owns in the shared chunk store.
         */
        final LongOpenHashSet ownedSections = new LongOpenHashSet();
        final LongOpenHashSet dirtySections = new LongOpenHashSet();
        boolean boundsDirty = true;
        boolean parentDirty = true;
        BodySection[] assembled = NO_SECTIONS;
        SableBody[] embedded = NO_BODIES;
        int parentBits;
        int leafCount;
        /**
         * Contacts must always reach the listener: special blocks or mounted contraptions.
         */
        boolean layerSpecial;
        /**
         * Upper bound of the distance from the center of mass to any block.
         */
        float boundRadius = 1.0f;
        double shapeComX;
        double shapeComY;
        double shapeComZ;

        // Unique solid blocks for buoyancy (x, y, z triples).
        int[] buoyancyBlocks = NO_INTS;
        float[] buoyancyVolumes = NO_FLOATS;
        int buoyancyCount;

        // Per-tick buoyancy of the Rapier model, applied every substep.
        boolean hasTickBuoyancy;
        double tickFx;
        double tickFy;
        double tickFz;
        double tickTx;
        double tickTy;
        double tickTz;

        // Kinematic contraptions
        public int mountId = -1;
        /**
         * Contraption pose: block-space position of its center of mass in the
         * mount (world space when unmounted) and orientation relative to it.
         */
        public final Vector3d relPos = new Vector3d();
        public final Quaterniond relRot = new Quaterniond(1.0, 0.0, 0.0, 0.0);
        /**
         * Contraption velocities in its own frame (Rapier "fake velocities").
         */
        public final Vector3d linVel = new Vector3d();
        public final Vector3d angVel = new Vector3d();
        /**
         * Contraptions embedded in this sub-level's compound.
         */
        final ObjectArrayList<SableBody> mountedContraptions = new ObjectArrayList<>(0);
        /**
         * Whether this contraption is currently part of its mount's compound.
         */
        boolean embeddedInMount;
        /**
         * Activate the body at the next compound assembly (mounted contraptions changed).
         */
        boolean wakeOnAssemble;

        /**
         * Joints and rope attachments using this body (skips the scan on COM shifts).
         */
        int attachmentCount;

        // Last applied mass properties (re-applied if the native body is created late).
        double appliedMass = Double.NaN;
        final Matrix3d appliedInertia = new Matrix3d();

        // Batch-read body state (see refreshBodySnapshot). Served while
        // snapValid (sleeping bodies keep theirs); otherwise direct JNI reads
        // are used instead. Also holds the last known good pose.
        boolean snapValid;
        boolean snapQueued;
        long snapGen;
        double snapX;
        double snapY;
        double snapZ;
        double snapComX;
        double snapComY;
        double snapComZ;
        float snapQx;
        float snapQy;
        float snapQz;
        float snapQw = 1.0f;
        float snapVx;
        float snapVy;
        float snapVz;
        float snapAx;
        float snapAy;
        float snapAz;
        /**
         * Awake as of the last snapshot; set eagerly when this scene activates
         * the body between steps.
         */
        boolean snapActive = true;

        // Deepest contact penetration this step (plus its outward normal) for
        // creatively dragged bodies — used by the end-of-step push-out.
        // Reset in flushDirty, accumulated in the contact listener.
        double dragPenDepth;
        double dragPenNx;
        double dragPenNy;
        double dragPenNz;

        // Kept for the benchmarks; the section cache replaced the full-rebuild
        // dedup and throttle.
        long rebuiltDataVersion = -1;
        long lastShapeRebuildNanos;

        public SableBody(final Kind kind, final int runtimeId) {
            this.kind = kind;
            this.runtimeId = runtimeId;
        }

        public boolean contains(final int x, final int y, final int z) {
            return this.hasBounds
                    && x >= this.minX && x <= this.maxX
                    && y >= this.minY && y <= this.maxY
                    && z >= this.minZ && z <= this.maxZ;
        }

        /**
         * Resolves a contact sub-shape ID to its leaf. The ID is decoded
         * arithmetically: the parent compound contributes the low
         * {@code parentBits} (child index), the section compound the next
         * {@code leafBits} (leaf index); embedded contraptions recurse.
         */
        @Nullable
        public Child childAt(final int subShapeId) {
            final BodySection[] secs = this.assembled;
            final SableBody[] emb = this.embedded;
            final int n = secs.length + emb.length;
            if (n == 0) {
                return null;
            }
            final int pb = this.parentBits;
            final int index = pb == 0 ? 0 : subShapeId & ((1 << pb) - 1);
            final int rest = subShapeId >>> pb;
            if (index < secs.length) {
                return secs[index].leaf(rest);
            }
            final int k = index - secs.length;
            return k < emb.length ? emb[k].childAt(rest) : null;
        }
    }

    public static final class GlobalChunk {
        final int cx;
        final int cy;
        final int cz;
        ChunkSectionData data;
        int joltId;

        /**
         * The Java wrapper of the native body; must stay referenced for the body's
         * whole lifetime.
         */
        Body body;

        /**
         * Section shape, local to the section's minimum corner (the static body
         * sits there; world-absolute float offsets lose precision far from spawn).
         */
        ConstShape shape;

        Child[] children = NO_CHILDREN;
        int leafBits;
        /**
         * Holds blocks with special contact behavior (friction, restitution, callbacks).
         */
        boolean special;
        boolean layerSpecial;

        GlobalChunk(final int cx, final int cy, final int cz, final ChunkSectionData data) {
            this.cx = cx;
            this.cy = cy;
            this.cz = cz;
            this.data = data;
        }

        @Nullable
        Child childAt(final int subShapeId) {
            final Child[] c = this.children;
            if (c.length == 0) {
                return null;
            }
            if (this.leafBits == 0) {
                return c[0];
            }
            final int j = subShapeId & ((1 << this.leafBits) - 1);
            return j < c.length ? c[j] : null;
        }
    }

    public static final class JointRecord {
        public long id;
        public Constraint constraint;
        public int joltIdA = -1;
        public int joltIdB = -1;
        public boolean contactsEnabled = true;
        public boolean fixedContacts;
        public TwoBodyConstraintSettings genericSettings;
        public final double[][] limits = new double[6][2];
        public final boolean[] locked = new boolean[6];
    }

    public static final class RopeAttachment {
        int mountJoltId;
        double x;
        double y;
        double z;
        public Constraint constraint;
    }

    public static final class RopeStrand {
        final IntArrayList points = new IntArrayList();
        final ObjectArrayList<DistanceConstraint> joints = new ObjectArrayList<>();
        double pointRadius;
        double firstJointLength;
        @Nullable RopeAttachment start;
        @Nullable RopeAttachment end;
    }

    //endregion

    //region Body management

    private SableBody createBody(final SableBody.Kind kind, final int runtimeId, final Vector3dc pos, final Quaterniondc rot, final int layer) {
        final SableBody previous = this.bodies.get(runtimeId);
        if (previous != null) {
            // Defensive: a second add for the same id would orphan the first body.
            this.destroyBody(previous);
        }

        final SableBody sb = new SableBody(kind, runtimeId);
        sb.layer = layer;
        sb.snapX = sb.snapComX = pos.x();
        sb.snapY = sb.snapComY = pos.y();
        sb.snapZ = sb.snapComZ = pos.z();
        sb.snapQx = (float) rot.x();
        sb.snapQy = (float) rot.y();
        sb.snapQz = (float) rot.z();
        sb.snapQw = (float) rot.w();
        this.bodies.put(runtimeId, sb);

        if (!this.createNativeBody(sb)) {
            this.pendingCreation.add(sb);
        }
        return sb;
    }

    /**
     * Creates the native body of a record at its last known pose. Returns false
     * (and keeps the record frozen at that pose) when the body limit is reached.
     */
    private boolean createNativeBody(final SableBody sb) {
        // Report layer until the first snapshot classifies the body: a body
        // loaded with velocity can hit something in its very first step.
        if (sb.layer == LAYER_MOVING) {
            sb.layer = LAYER_MOVING_REPORT;
        }
        final MutableCompoundShape shape = new MutableCompoundShape();
        final BodyCreationSettings bcs = new BodyCreationSettings(
                shape,
                sb.snapX, sb.snapY, sb.snapZ,
                sb.snapQx, sb.snapQy, sb.snapQz, sb.snapQw,
                sb.kind == SableBody.Kind.CONTRAPTION ? EMotionType.Kinematic : EMotionType.Dynamic,
                sb.layer);
        bcs.setLinearDamping((float) this.universalDrag);
        bcs.setAngularDamping((float) this.universalDrag);
        bcs.setMotionQuality(EMotionQuality.LinearCast);
        bcs.setFriction(sb.kind == SableBody.Kind.CONTRAPTION ? FRICTION_CONTRAPTION : FRICTION_SUB_LEVEL);
        bcs.setRestitution(0.0f);
        bcs.setApplyGyroscopicForce(true);
        bcs.setOverrideMassProperties(EOverrideMassProperties.CalculateMassAndInertia);
        if (this.rapierMode && sb.kind == SableBody.Kind.SUB_LEVEL) {
            bcs.setMaxLinearVelocity(RAPIER_MAX_LINEAR_VELOCITY);
            bcs.setMaxAngularVelocity(RAPIER_MAX_ANGULAR_VELOCITY);
        }

        final Body body;
        try {
            body = this.bi.createBody(bcs);
        } catch (final IllegalStateException e) {
            this.warnCapacity("sub-level " + sb.runtimeId + " is frozen until bodies free up");
            return false;
        }
        body.setUserData(sb.runtimeId);
        // Rapier creates sub-levels awake; a sleeping new body would hang in the
        // air until something woke it.
        this.bi.addBody(body, sb.kind == SableBody.Kind.CONTRAPTION ? EActivation.DontActivate : EActivation.Activate);

        sb.joltId = body.getId();
        sb.body = body;
        sb.shape = shape;
        sb.listIndex = this.bodyList.size();
        sb.snapValid = true;
        sb.snapActive = sb.kind != SableBody.Kind.CONTRAPTION;
        this.bodyList.add(sb);
        this.bodiesByJoltId.put(sb.joltId, sb);
        sb.boundsDirty = true;
        sb.parentDirty = true;
        this.markDirty(sb);
        if (!Double.isNaN(sb.appliedMass)) {
            this.applyMassProperties(sb, sb.appliedMass, sb.appliedInertia);
        }
        return true;
    }

    private void retryPendingCreation() {
        if (this.pendingCreation.isEmpty()) {
            return;
        }
        for (int i = this.pendingCreation.size() - 1; i >= 0; i--) {
            final SableBody sb = this.pendingCreation.get(i);
            if (this.bodies.get(sb.runtimeId) != sb) {
                this.pendingCreation.remove(i);
                continue;
            }
            if (this.system.getNumBodies() >= this.config.maxBodies - 1) {
                return;
            }
            if (this.createNativeBody(sb)) {
                this.pendingCreation.remove(i);
                if (sb.kind == SableBody.Kind.SUB_LEVEL) {
                    this.attachWaitingContraptions(sb);
                }
            }
        }
    }

    private void removeFromBodyList(final SableBody sb) {
        final int index = sb.listIndex;
        if (index < 0) {
            return;
        }
        final int last = this.bodyList.size() - 1;
        final SableBody moved = this.bodyList.get(last);
        this.bodyList.set(index, moved);
        moved.listIndex = index;
        this.bodyList.remove(last);
        sb.listIndex = -1;
    }

    private void destroyBody(final SableBody sb) {
        if (this.bodies.get(sb.runtimeId) == sb) {
            this.bodies.remove(sb.runtimeId);
        }
        this.pendingCreation.remove(sb);
        synchronized (this.dirtyBodies) {
            this.dirtyBodies.remove(sb);
        }

        if (sb.kind == SableBody.Kind.CONTRAPTION) {
            this.contraptionList.remove(sb);
            this.detachFromMount(sb);
        } else if (sb.kind == SableBody.Kind.SUB_LEVEL) {
            // Mounted contraptions stay registered and re-attach if the
            // sub-level comes back (recovery re-adds it with the same id).
            for (final SableBody contraption : sb.mountedContraptions) {
                contraption.embeddedInMount = false;
            }
            sb.mountedContraptions.clear();
            this.dropOwnedSections(sb);
        }

        if (sb.body != null) {
            this.creativeDragBodies.remove(sb.joltId);
            this.creativeRestorePending.remove(sb.joltId);
            this.bodiesByJoltId.remove(sb.joltId);
            this.removeFromBodyList(sb);
            // Native constraints and rope attachments hold raw references to their
            // bodies: destroying a body with live attachments leaves dangling
            // native references behind, and the next system.update() dies with an
            // access violation inside joltjni.dll. Detach everything first (Sable
            // re-creates the attachments it still needs, e.g. on the remove and
            // re-add recovery path).
            this.detachBodyAttachments(sb.joltId);
            this.bi.removeBody(sb.joltId);
            this.bi.destroyBody(sb.joltId);
            sb.body = null;
            sb.joltId = 0;
        }
    }

    /**
     * Drops the plot sections of a removed sub-level. Sable reuses plot slots, so
     * stale sections would otherwise turn into ghost blocks inside the next
     * sub-level of that slot (and leak 16 KB each). Block edits re-create a
     * missing plot section from scratch, see {@link #changeBlocks}.
     */
    private void dropOwnedSections(final SableBody sb) {
        for (final long key : sb.ownedSections) {
            if (this.sectionOwners.get(key) == sb.runtimeId) {
                this.sectionOwners.remove(key);
                this.allChunks.remove(key);
            }
        }
        sb.ownedSections.clear();
    }

    /**
     * Records the sub-level owning a plot section (kept per body too, so
     * removing a sub-level never scans every section).
     */
    private void setSectionOwner(final long key, final int owner) {
        final int previous = this.sectionOwners.put(key, owner);
        if (previous == owner) {
            return;
        }
        if (previous != -1) {
            final SableBody old = this.bodies.get(previous);
            if (old != null) {
                old.ownedSections.remove(key);
            }
        }
        final SableBody sb = this.bodies.get(owner);
        if (sb != null) {
            sb.ownedSections.add(key);
        }
    }

    private int clearSectionOwner(final long key) {
        final int previous = this.sectionOwners.remove(key);
        if (previous != -1) {
            final SableBody old = this.bodies.get(previous);
            if (old != null) {
                old.ownedSections.remove(key);
            }
        }
        return previous;
    }

    /**
     * Removes every joint and rope attachment mounted on the given body, and
     * drops its servo caches. Called before the native body is destroyed so no
     * native object keeps referencing the freed body.
     */
    private void detachBodyAttachments(final int joltId) {
        if (!this.joints.isEmpty()) {
            // Collect first: removeJoint mutates the map under iteration.
            LongArrayList deadJoints = null;
            for (final var entry : this.joints.long2ObjectEntrySet()) {
                final JointRecord rec = entry.getValue();
                if (rec.joltIdA != joltId && rec.joltIdB != joltId) {
                    continue;
                }
                if (deadJoints == null) {
                    deadJoints = new LongArrayList();
                }
                deadJoints.add(entry.getLongKey());
            }
            if (deadJoints != null) {
                for (int i = 0; i < deadJoints.size(); i++) {
                    this.removeJoint(deadJoints.getLong(i));
                }
            }
        }
        if (!this.ropes.isEmpty()) {
            for (final RopeStrand strand : this.ropes.values()) {
                if (strand.start != null && strand.start.mountJoltId == joltId) {
                    this.removeRopeAttachment(strand.start);
                    strand.start = null;
                }
                if (strand.end != null && strand.end.mountJoltId == joltId) {
                    this.removeRopeAttachment(strand.end);
                    strand.end = null;
                }
            }
        }
        this.angularFollowPrev.remove(joltId);
        this.angularCachedOmega.remove(joltId);
    }

    public void createSubLevel(final int id, final Vector3dc pos, final Quaterniondc rot) {
        if (this.disposed) {
            return;
        }
        final SableBody sb = this.createBody(SableBody.Kind.SUB_LEVEL, id, pos, rot, LAYER_MOVING);
        if (sb.body != null) {
            this.attachWaitingContraptions(sb);
        }
    }

    public void removeSubLevel(final int id) {
        final SableBody sb = this.bodies.get(id);
        if (sb != null) {
            this.destroyBody(sb);
        }
    }

    public void createBox(final int id, final double mass, final double hx, final double hy, final double hz, final Vector3dc pos, final Quaterniondc rot) {
        if (this.disposed) {
            return;
        }
        final SableBody previous = this.bodies.get(id);
        if (previous != null) {
            this.destroyBody(previous);
        }
        final BodyCreationSettings bcs = new BodyCreationSettings(
                newBoxShape((float) hx, (float) hy, (float) hz, 0.025f),
                pos.x(), pos.y(), pos.z(),
                (float) rot.x(), (float) rot.y(), (float) rot.z(), (float) rot.w(),
                EMotionType.Dynamic, LAYER_MOVING_REPORT);
        bcs.setMotionQuality(EMotionQuality.LinearCast);
        bcs.setFriction(FRICTION_BOX);
        bcs.setApplyGyroscopicForce(true);
        final MassProperties mp = new MassProperties();
        mp.setMass((float) Math.max(mass, 1.0e-3));
        bcs.setOverrideMassProperties(EOverrideMassProperties.CalculateInertia);
        bcs.setMassPropertiesOverride(mp);

        final Body body;
        try {
            body = this.bi.createBody(bcs);
        } catch (final IllegalStateException e) {
            this.warnCapacity("physics box " + id + " was not created");
            return;
        }
        body.setUserData(id);
        this.bi.addBody(body, EActivation.Activate);

        final SableBody sb = new SableBody(SableBody.Kind.BOX, id);
        sb.joltId = body.getId();
        sb.layer = LAYER_MOVING_REPORT;
        sb.body = body;
        sb.snapX = sb.snapComX = pos.x();
        sb.snapY = sb.snapComY = pos.y();
        sb.snapZ = sb.snapComZ = pos.z();
        sb.snapQx = (float) rot.x();
        sb.snapQy = (float) rot.y();
        sb.snapQz = (float) rot.z();
        sb.snapQw = (float) rot.w();
        sb.boundsDirty = false;
        sb.parentDirty = false;
        sb.listIndex = this.bodyList.size();
        sb.snapValid = true;
        sb.snapActive = true;
        this.bodyList.add(sb);
        this.bodies.put(id, sb);
        this.bodiesByJoltId.put(sb.joltId, sb);
    }

    public void removeBox(final int id) {
        this.removeSubLevel(id);
    }

    public SableBody body(final int runtimeId) {
        return this.bodies.get(runtimeId);
    }

    @Nullable
    public SableBody bodyByJoltId(final int joltId) {
        return this.bodiesByJoltId.get(joltId);
    }

    //endregion

    //region Pose / mass / teleport

    /**
     * Reusable destination for {@link #getPose} writes; avoids per-read allocations.
     */
    public static class PoseCache {
        public double p1;
        public double p2;
        public double p3;
        public double p4;
        public double p5;
        public double p6;
        public double p7;
    }

    public void getPose(final int id, final PoseCache store) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        if (sb.body == null || (sb.snapValid)) {
            // Fresh snapshot, or the last known pose of a frozen record.
            store.p1 = sb.snapX;
            store.p2 = sb.snapY;
            store.p3 = sb.snapZ;
            store.p4 = sb.snapQx;
            store.p5 = sb.snapQy;
            store.p6 = sb.snapQz;
            store.p7 = sb.snapQw;
            return;
        }
        final Body body = sb.body;
        final RVec3 pos = this.tmpCom.get();
        final Quat rot = this.tmpQuat.get();
        body.getPositionAndRotation(pos, rot);
        store.p1 = pos.xx();
        store.p2 = pos.yy();
        store.p3 = pos.zz();
        store.p4 = rot.getX();
        store.p5 = rot.getY();
        store.p6 = rot.getZ();
        store.p7 = rot.getW();
    }

    /**
     * Updates the center of mass of a body. The body itself was already moved
     * by Sable (teleport by the rotated shift); the compound children follow at
     * the next step without rebuilding any section, and joints anchored on the
     * body keep their block-space anchor (Rapier re-anchors every tick).
     */
    public void setCenterOfMass(final int id, final double x, final double y, final double z) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        final double dx = x - sb.centerOfMass.x;
        final double dy = y - sb.centerOfMass.y;
        final double dz = z - sb.centerOfMass.z;
        if (dx == 0.0 && dy == 0.0 && dz == 0.0) {
            return;
        }
        sb.centerOfMass.set(x, y, z);
        sb.parentDirty = true;
        this.markDirty(sb);
        if (sb.attachmentCount > 0 && sb.body != null) {
            this.notifyComShift(sb.joltId, (float) dx, (float) dy, (float) dz);
        }
        if (sb.kind == SableBody.Kind.CONTRAPTION) {
            this.markMountDirty(sb);
        }
    }

    public void setLocalBounds(final int id, final int minX, final int minY, final int minZ, final int maxX, final int maxY, final int maxZ) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        if (sb.hasBounds && sb.minX == minX && sb.minY == minY && sb.minZ == minZ
                && sb.maxX == maxX && sb.maxY == maxY && sb.maxZ == maxZ) {
            return;
        }
        sb.minX = minX;
        sb.minY = minY;
        sb.minZ = minZ;
        sb.maxX = maxX;
        sb.maxY = maxY;
        sb.maxZ = maxZ;
        sb.hasBounds = true;
        sb.boundsDirty = true;
        this.markDirty(sb);
    }

    /**
     * Overrides the mass and inertia tensor of a body. The center of mass is
     * set separately via {@link #setCenterOfMass}.
     */
    public void setMassProperties(final int id, final double mass, final Matrix3dc inertiaTensor) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        // Zero or non-finite input would make Jolt compute an infinite inverse
        // mass and poison the body with NaN on the next step.
        if (!(mass > 0.0) || !Double.isFinite(mass) || !isFinite(inertiaTensor)) {
            return;
        }
        sb.appliedMass = mass;
        sb.appliedInertia.set(inertiaTensor);
        if (sb.body != null) {
            this.applyMassProperties(sb, mass, inertiaTensor);
        }
    }

    private static boolean isFinite(final Matrix3dc m) {
        return Double.isFinite(m.m00()) && Double.isFinite(m.m01()) && Double.isFinite(m.m02())
                && Double.isFinite(m.m10()) && Double.isFinite(m.m11()) && Double.isFinite(m.m12())
                && Double.isFinite(m.m20()) && Double.isFinite(m.m21()) && Double.isFinite(m.m22());
    }

    private void applyMassProperties(final SableBody sb, final double mass, final Matrix3dc inertiaTensor) {
        final Body body = sb.body;
        if (body == null || !body.isDynamic()) {
            return;
        }
        final MotionProperties motion = body.getMotionProperties();
        if (motion == null) {
            return;
        }

        // Reused across calls: mass stats arrive every tick per body, and the old
        // code allocated a MassProperties + Mat44 + varargs float[] per call.
        // Memory layout preserved from the old varargs sequence.
        final MassProperties mp = this.massPropsScratch;
        mp.setMass((float) mass);
        final Mat44 inertia = this.inertiaScratch;
        inertia.setElement(0, 0, (float) inertiaTensor.m00());
        inertia.setElement(0, 1, (float) inertiaTensor.m10());
        inertia.setElement(0, 2, (float) inertiaTensor.m20());
        inertia.setElement(0, 3, 0.0f);
        inertia.setElement(1, 0, (float) inertiaTensor.m01());
        inertia.setElement(1, 1, (float) inertiaTensor.m11());
        inertia.setElement(1, 2, (float) inertiaTensor.m21());
        inertia.setElement(1, 3, 0.0f);
        inertia.setElement(2, 0, (float) inertiaTensor.m02());
        inertia.setElement(2, 1, (float) inertiaTensor.m12());
        inertia.setElement(2, 2, (float) inertiaTensor.m22());
        inertia.setElement(2, 3, 0.0f);
        inertia.setElement(3, 0, 0.0f);
        inertia.setElement(3, 1, 0.0f);
        inertia.setElement(3, 2, 0.0f);
        inertia.setElement(3, 3, 1.0f);
        mp.setInertia(inertia);
        motion.setMassProperties(ALLOWED_DOFS_ALL, mp);
    }

    public void teleportObject(final int id, final double x, final double y, final double z, final double i, final double j, final double k, final double r) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        if (!(Double.isFinite(x) && Double.isFinite(y) && Double.isFinite(z)
                && Double.isFinite(i) && Double.isFinite(j) && Double.isFinite(k) && Double.isFinite(r))) {
            return;
        }
        // The snapshot keeps serving the new pose (velocities are untouched by
        // a teleport); it is also the restore point of the poison watchdog.
        sb.snapX = sb.snapComX = x;
        sb.snapY = sb.snapComY = y;
        sb.snapZ = sb.snapComZ = z;
        sb.snapQx = (float) i;
        sb.snapQy = (float) j;
        sb.snapQz = (float) k;
        sb.snapQw = (float) r;
        if (sb.body != null) {
            this.bi.setPositionAndRotation(sb.joltId, x, y, z, (float) i, (float) j, (float) k, (float) r, EActivation.Activate);
            sb.snapActive = true;
        }
    }

    public void wakeUpObject(final int id) {
        final SableBody sb = this.bodies.get(id);
        if (sb != null && sb.body != null) {
            this.bi.activateBody(sb.joltId);
            sb.snapActive = true;
        }
    }

    public void addLinearAngularVelocities(final int id, final double lx, final double ly, final double lz,
                                           final double ax, final double ay, final double az, final boolean wakeUp) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null || sb.body == null) {
            return;
        }
        if (!(Double.isFinite(lx) && Double.isFinite(ly) && Double.isFinite(lz)
                && Double.isFinite(ax) && Double.isFinite(ay) && Double.isFinite(az))) {
            return;
        }
        final Body body = sb.body;
        if (!wakeUp && !body.isActive()) {
            return;
        }
        final boolean fresh = sb.snapValid;
        float bvx, bvy, bvz, bax, bay, baz;
        if (fresh) {
            bvx = sb.snapVx;
            bvy = sb.snapVy;
            bvz = sb.snapVz;
            bax = sb.snapAx;
            bay = sb.snapAy;
            baz = sb.snapAz;
        } else {
            final Vec3 v = this.tmpVec.get();
            body.getLinearVelocity(v);
            bvx = v.getX();
            bvy = v.getY();
            bvz = v.getZ();
            body.getAngularVelocity(v);
            bax = v.getX();
            bay = v.getY();
            baz = v.getZ();
        }
        bvx += (float) lx;
        bvy += (float) ly;
        bvz += (float) lz;
        bax += (float) ax;
        bay += (float) ay;
        baz += (float) az;
        this.bi.setLinearAndAngularVelocity(sb.joltId, bvx, bvy, bvz, bax, bay, baz);
        if (fresh) {
            sb.snapVx = bvx;
            sb.snapVy = bvy;
            sb.snapVz = bvz;
            sb.snapAx = bax;
            sb.snapAy = bay;
            sb.snapAz = baz;
        }
        if (wakeUp) {
            this.bi.activateBody(sb.joltId);
            sb.snapActive = true;
        }
    }

    public void getLinearVelocity(final int id, final PoseCache store) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        if (sb.body == null) {
            store.p1 = 0.0;
            store.p2 = 0.0;
            store.p3 = 0.0;
            return;
        }
        if (sb.snapValid) {
            store.p1 = sb.snapVx;
            store.p2 = sb.snapVy;
            store.p3 = sb.snapVz;
            return;
        }
        final Vec3 v = this.tmpVec.get();
        sb.body.getLinearVelocity(v);
        store.p1 = v.getX();
        store.p2 = v.getY();
        store.p3 = v.getZ();
    }

    public void getAngularVelocity(final int id, final PoseCache store) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        if (sb.body == null) {
            store.p1 = 0.0;
            store.p2 = 0.0;
            store.p3 = 0.0;
            return;
        }
        if (sb.snapValid) {
            store.p1 = sb.snapAx;
            store.p2 = sb.snapAy;
            store.p3 = sb.snapAz;
            return;
        }
        final Vec3 v = this.tmpVec.get();
        sb.body.getAngularVelocity(v);
        store.p1 = v.getX();
        store.p2 = v.getY();
        store.p3 = v.getZ();
    }

    /**
     * Current rotation of a body: the snapshot when fresh (impulses arrive
     * between steps, so it is exact), a native read otherwise.
     */
    private Quat currentRotation(final SableBody sb) {
        final Quat rot = this.tmpQuat.get();
        if (sb.snapValid) {
            rot.set(sb.snapQx, sb.snapQy, sb.snapQz, sb.snapQw);
        } else {
            sb.body.getRotation(rot);
        }
        return rot;
    }

    /**
     * Applies an impulse at a local offset (relative to the center of mass), replicating the rapier semantics.
     */
    public void applyForce(final int id, final double x, final double y, final double z,
                           final double fx, final double fy, final double fz, final boolean wakeUp) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null || sb.body == null) {
            return;
        }
        if (!(Double.isFinite(x) && Double.isFinite(y) && Double.isFinite(z)
                && Double.isFinite(fx) && Double.isFinite(fy) && Double.isFinite(fz))) {
            return;
        }
        final Body body = sb.body;
        if (!wakeUp && !body.isActive()) {
            return;
        }

        final Quat rot = this.currentRotation(sb);
        final float qx = rot.getX();
        final float qy = rot.getY();
        final float qz = rot.getZ();
        final float qw = rot.getW();

        Vector3f r = this.rotateInto((float) fx, (float) fy, (float) fz, qx, qy, qz, qw);
        final float impX = r.x, impY = r.y, impZ = r.z;
        r = this.rotateInto((float) x, (float) y, (float) z, qx, qy, qz, qw);
        final float offX = r.x, offY = r.y, offZ = r.z;
        final RVec3 com = this.tmpCom.get();
        body.getCenterOfMassPosition(com);

        // Jolt does not wake bodies on AddImpulse; activation is required separately
        final Vec3 imp = this.tmpVec.get();
        imp.set(impX, impY, impZ);
        final RVec3 point = this.tmpPoint.get();
        point.set(com.xx() + offX, com.yy() + offY, com.zz() + offZ);
        body.addImpulse(imp, point);
        this.invalidateSnapshot(sb); // impulse changed the velocity
        if (wakeUp) {
            this.bi.activateBody(sb.joltId);
            sb.snapActive = true;
        }
    }

    public void applyForceAndTorque(final int id, final double fx, final double fy, final double fz,
                                    final double tx, final double ty, final double tz, final boolean wakeUp) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null || sb.body == null) {
            return;
        }
        if (!(Double.isFinite(fx) && Double.isFinite(fy) && Double.isFinite(fz)
                && Double.isFinite(tx) && Double.isFinite(ty) && Double.isFinite(tz))) {
            return;
        }
        final Body body = sb.body;
        if (!wakeUp && !body.isActive()) {
            return;
        }

        final Quat rot = this.currentRotation(sb);
        final float rotX = rot.getX();
        final float rotY = rot.getY();
        final float rotZ = rot.getZ();
        final float rotW = rot.getW();

        Vector3f r = this.rotateInto((float) fx, (float) fy, (float) fz, rotX, rotY, rotZ, rotW);
        body.addImpulse(r.x, r.y, r.z);
        r = this.rotateInto((float) tx, (float) ty, (float) tz, rotX, rotY, rotZ, rotW);
        body.addAngularImpulse(r.x, r.y, r.z);
        this.invalidateSnapshot(sb); // impulse changed the velocity
        if (wakeUp) {
            this.bi.activateBody(sb.joltId);
            sb.snapActive = true;
        }
    }

    /**
     * Scratch buffer for {@link #rotateInto}; ThreadLocal because callers run
     * both on the server thread and on buoyancy/Jolt job threads.
     */
    private final ThreadLocal<Vector3f> rotateOut = ThreadLocal.withInitial(Vector3f::new);
    private final ThreadLocal<Quat> tmpQuat = ThreadLocal.withInitial(Quat::new);
    private final ThreadLocal<RVec3> tmpCom = ThreadLocal.withInitial(RVec3::new);
    private final ThreadLocal<RVec3> tmpPoint = ThreadLocal.withInitial(RVec3::new);
    private final ThreadLocal<Vec3> tmpVec = ThreadLocal.withInitial(Vec3::new);

    /**
     * Rotates a vector by a quaternion, writing the result into a per-thread
     * scratch buffer (allocation-free). The result is valid only until the next
     * call on the same thread — copy the components out before rotating again.
     */
    private Vector3f rotateInto(final float vx, final float vy, final float vz,
                                final float qx, final float qy, final float qz, final float qw) {
        // t = 2 * cross(q.xyz, v)
        final float tx = 2.0f * (qy * vz - qz * vy);
        final float ty = 2.0f * (qz * vx - qx * vz);
        final float tz = 2.0f * (qx * vy - qy * vx);
        // v + w*t + cross(q.xyz, t)
        final Vector3f out = this.rotateOut.get();
        out.x = vx + qw * tx + (qy * tz - qz * ty);
        out.y = vy + qw * ty + (qz * tx - qx * tz);
        out.z = vz + qw * tz + (qx * ty - qy * tx);
        return out;
    }

    /**
     * Converts a constraint anchor into the body-local frame required by Jolt's
     * {@code EConstraintSpace.LocalToBodyCom}. Sable passes anchors in the
     * body's block space (the plot of a sub-level), which is the body's local
     * frame, so only the center of mass is subtracted — exactly Rapier's
     * {@code local_anchor = pos - center_of_mass}. For the static ground the
     * world point is used as-is.
     */
    public RVec3 toLocalAnchor(@Nullable final SableBody sb, final double x, final double y, final double z) {
        if (sb == null) {
            return new RVec3(x, y, z);
        }
        return new RVec3(x - sb.centerOfMass.x, y - sb.centerOfMass.y, z - sb.centerOfMass.z);
    }

    //endregion

    //region Voxel geometry

    private static boolean isSolidBlock(final int packed, @Nullable final JoltVoxelColliderData entry) {
        final int colliderId = ChunkSectionData.colliderIdOf(packed);
        final int voxelState = ChunkSectionData.voxelStateOf(packed);
        // voxel states: 0=empty, 1=face, 2=edge, 3=corner, 4=interior
        if (voxelState < 1 || voxelState > 3 || colliderId <= 0 || entry == null) {
            return false;
        }
        // fluids never produce collision; they only contribute buoyancy and drag
        if (entry.isFluid) {
            return false;
        }
        return entry.hasBoxes();
    }

    /**
     * Brings a body's compound up to date: rebuilds dirty sections, then
     * reassembles the small parent compound when sections, embedded
     * contraptions or the center of mass changed. A center of mass shift
     * therefore costs one cheap reassembly instead of a full rebuild, and block
     * edits only rebuild the touched sections — no throttling needed.
     */
    private void rebuildShape(final SableBody sb) {
        this.rebuildShape(sb, true);
    }

    /**
     * @param parallelSections build many dirty sections on the worker pool; must
     *                         be false when already running on that pool (a
     *                         worker waiting for queued sub-tasks deadlocks it)
     */
    private void rebuildShape(final SableBody sb, final boolean parallelSections) {
        if (sb.kind == SableBody.Kind.BOX) {
            return;
        }
        // Benchmarks reset this to force a full rebuild.
        if (sb.rebuiltDataVersion == -1L) {
            sb.boundsDirty = true;
            sb.rebuiltDataVersion = 0L;
        }

        boolean sectionsChanged = false;
        if (sb.boundsDirty) {
            sb.boundsDirty = false;
            this.collectSectionWindow(sb);
            sectionsChanged = true;
        }

        if (!sb.dirtySections.isEmpty()) {
            final long[] keys = sb.dirtySections.toLongArray();
            sb.dirtySections.clear();
            final BodySection[] built = new BodySection[keys.length];
            final IntConsumer build = i -> built[i] = this.buildBodySection(sb, keys[i]);
            if (parallelSections && keys.length >= 16) {
                runParallelIndices(keys.length, build);
            } else {
                for (int i = 0; i < keys.length; i++) {
                    build.accept(i);
                }
            }
            for (int i = 0; i < keys.length; i++) {
                if (built[i] == null) {
                    sb.sections.remove(keys[i]);
                } else {
                    sb.sections.put(keys[i], built[i]);
                }
            }
            sectionsChanged = true;
        }

        if (sectionsChanged) {
            this.indexSections(sb);
        }

        if (sectionsChanged || sb.parentDirty
                || sb.shapeComX != sb.centerOfMass.x || sb.shapeComY != sb.centerOfMass.y || sb.shapeComZ != sb.centerOfMass.z) {
            sb.parentDirty = false;
            this.assembleParent(sb);
            sb.lastShapeRebuildNanos = System.nanoTime();
        }
    }

    /**
     * Marks every section the body covers for rebuild and drops sections outside.
     * Sub-levels take their sections from the shared chunk store, filtered by
     * their bounds (like the rapier octree rebuild from main_level_chunks);
     * contraptions own a dedicated store.
     */
    private void collectSectionWindow(final SableBody sb) {
        if (sb.kind == SableBody.Kind.CONTRAPTION) {
            sb.sections.keySet().removeIf((long key) -> !sb.chunks.containsKey(key));
            sb.dirtySections.addAll(sb.chunks.keySet());
            return;
        }
        if (!sb.hasBounds) {
            sb.sections.clear();
            sb.dirtySections.clear();
            return;
        }
        final int cMinX = sb.minX >> 4, cMinY = sb.minY >> 4, cMinZ = sb.minZ >> 4;
        final int cMaxX = sb.maxX >> 4, cMaxY = sb.maxY >> 4, cMaxZ = sb.maxZ >> 4;
        sb.sections.keySet().removeIf((long key) -> {
            final int cx = unpackChunkX(key), cy = unpackChunkY(key), cz = unpackChunkZ(key);
            return cx < cMinX || cx > cMaxX || cy < cMinY || cy > cMaxY || cz < cMinZ || cz > cMaxZ;
        });
        sb.dirtySections.removeIf((long key) -> {
            final int cx = unpackChunkX(key), cy = unpackChunkY(key), cz = unpackChunkZ(key);
            return cx < cMinX || cx > cMaxX || cy < cMinY || cy > cMaxY || cz < cMinZ || cz > cMaxZ;
        });
        final long window = (long) (cMaxX - cMinX + 1) * (cMaxY - cMinY + 1) * (cMaxZ - cMinZ + 1);
        if (window <= this.allChunks.size()) {
            for (int cx = cMinX; cx <= cMaxX; cx++) {
                for (int cy = cMinY; cy <= cMaxY; cy++) {
                    for (int cz = cMinZ; cz <= cMaxZ; cz++) {
                        final long key = ChunkSectionData.packSectionPos(cx, cy, cz);
                        if (this.allChunks.containsKey(key)) {
                            sb.dirtySections.add(key);
                        }
                    }
                }
            }
        } else {
            for (final long key : this.allChunks.keySet()) {
                final int cx = unpackChunkX(key), cy = unpackChunkY(key), cz = unpackChunkZ(key);
                if (cx >= cMinX && cx <= cMaxX && cy >= cMinY && cy <= cMaxY && cz >= cMinZ && cz <= cMaxZ) {
                    sb.dirtySections.add(key);
                }
            }
        }
    }

    /**
     * Refreshes the derived arrays after the section set changed: parent order
     * (sorted by key for determinism), flattened leaves and buoyancy blocks.
     */
    private void indexSections(final SableBody sb) {
        final BodySection[] ordered = sb.sections.values().toArray(NO_SECTIONS);
        Arrays.sort(ordered, (a, b) -> Long.compare(a.key, b.key));
        sb.assembled = ordered;

        int leaves = 0;
        int blocks = 0;
        for (final BodySection s : ordered) {
            leaves += s.children.length;
            blocks += s.volumes.length;
        }
        sb.children.clear();
        sb.children.ensureCapacity(leaves);
        final int[] buoyancyBlocks = blocks == 0 ? NO_INTS : new int[blocks * 3];
        final float[] buoyancyVolumes = blocks == 0 ? NO_FLOATS : new float[blocks];
        int b = 0;
        for (final BodySection s : ordered) {
            sb.children.addElements(sb.children.size(), s.children);
            System.arraycopy(s.blocks, 0, buoyancyBlocks, b * 3, s.blocks.length);
            System.arraycopy(s.volumes, 0, buoyancyVolumes, b, s.volumes.length);
            b += s.volumes.length;
        }
        sb.leafCount = leaves;
        if (sb.hasBounds) {
            final double dx = sb.maxX - sb.minX + 1.0, dy = sb.maxY - sb.minY + 1.0, dz = sb.maxZ - sb.minZ + 1.0;
            sb.boundRadius = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
        sb.buoyancyBlocks = buoyancyBlocks;
        sb.buoyancyVolumes = buoyancyVolumes;
        sb.buoyancyCount = blocks;

        // CCD (LinearCast) sweeps the ENTIRE compound against the world every step;
        // for huge bodies that costs seconds per tick while they fall. Huge bodies
        // move slowly in practice, so switch them to discrete motion.
        if (sb.body != null && sb.kind == SableBody.Kind.SUB_LEVEL && leaves >= HEAVY_BODY_CHILDREN
                && this.bi.getMotionQuality(sb.joltId) != EMotionQuality.Discrete) {
            this.bi.setMotionQuality(sb.joltId, EMotionQuality.Discrete);
        }
    }

    /**
     * Shared identity rotation for compound {@code addShape} calls. The Jolt
     * wrappers read the components immediately and never retain the reference,
     * so one instance serves all threads; {@code Quat.sIdentity()} would
     * allocate a new object per leaf (thousands per rebuild).
     */
    private static final Quat IDENTITY_QUAT = new Quat(0.0f, 0.0f, 0.0f, 1.0f);

    /** Scratch offset for parent inserts (one use at a time per thread). */
    private final ThreadLocal<Vec3> sectionVecScratch = ThreadLocal.withInitial(Vec3::new);
    private final ThreadLocal<Quat> sectionQuatScratch = ThreadLocal.withInitial(Quat::new);

    /**
     * Builds the parent compound from the cached section shapes (offset by the
     * current center of mass) plus the shapes of embedded contraptions, and
     * installs it on the native body.
     */
    private void assembleParent(final SableBody sb) {
        final double comX = sb.centerOfMass.x;
        final double comY = sb.centerOfMass.y;
        final double comZ = sb.centerOfMass.z;
        final MutableCompoundShape parent = new MutableCompoundShape();
        final Vec3 offset = this.sectionVecScratch.get();
        final BodySection[] secs = sb.assembled;
        for (final BodySection s : secs) {
            offset.set((float) (s.ox - comX), (float) (s.oy - comY), (float) (s.oz - comZ));
            parent.addShape(offset, IDENTITY_QUAT, s.shape);
        }

        SableBody[] embedded = NO_BODIES;
        if (sb.kind == SableBody.Kind.SUB_LEVEL && !sb.mountedContraptions.isEmpty()) {
            final ObjectArrayList<SableBody> list = new ObjectArrayList<>(sb.mountedContraptions.size());
            final Quat rotation = this.sectionQuatScratch.get();
            for (final SableBody c : sb.mountedContraptions) {
                if (c.shape == null || c.assembled.length == 0) {
                    continue;
                }
                offset.set((float) (c.relPos.x - comX), (float) (c.relPos.y - comY), (float) (c.relPos.z - comZ));
                rotation.set((float) c.relRot.x, (float) c.relRot.y, (float) c.relRot.z, (float) c.relRot.w);
                parent.addShape(offset, rotation, c.shape);
                list.add(c);
            }
            embedded = list.toArray(NO_BODIES);
        }

        sb.shape = parent;
        sb.embedded = embedded;
        boolean special = embedded.length > 0;
        for (final BodySection s : secs) {
            special |= s.special;
        }
        sb.layerSpecial = special;
        if (special && sb.body != null && sb.layer == LAYER_MOVING) {
            this.setBodyLayer(sb, LAYER_MOVING_REPORT);
        }
        sb.parentBits = subShapeBits(secs.length + embedded.length);
        sb.shapeComX = comX;
        sb.shapeComY = comY;
        sb.shapeComZ = comZ;
        if (sb.body != null) {
            // Mounted contraptions changing wake the body, as Rapier does when a
            // body's colliders change; block edits are woken by Sable itself.
            this.bi.setShape(sb.joltId, parent, false, sb.wakeOnAssemble ? EActivation.Activate : EActivation.DontActivate);
            if (sb.wakeOnAssemble) {
                sb.snapActive = true;
            }
            sb.wakeOnAssemble = false;
        }

        if (JoltDebugLogging.STAFF) {
            Sable.LOGGER.info("[SableJolt] rebuild body {}: kind={} sections={} leaves={} embedded={} bounds=({},{},{})..({},{},{}) com=({},{},{})",
                    sb.runtimeId, sb.kind, secs.length, sb.leafCount, embedded.length,
                    sb.minX, sb.minY, sb.minZ, sb.maxX, sb.maxY, sb.maxZ, comX, comY, comZ);
        }
    }

    /**
     * Reusable per-thread leaf accumulator for {@link #buildBodySection}:
     * flat primitive arrays instead of one {@code Vec3} + list node per leaf.
     * A 16³ section holds at most a few thousand leaves; the arrays grow once
     * per thread and are reused by every later section.
     */
    private static final class SectionScratch {
        float[] ox = new float[256];
        float[] oy = new float[256];
        float[] oz = new float[256];
        ConstShape[] shapes = new ConstShape[256];
        Child[] children = new Child[256];
        int count;
        int[] blocks = new int[768];
        float[] volumes = new float[256];
        int blockCount;

        void ensure(final int needed) {
            if (needed <= this.ox.length) {
                return;
            }
            int cap = this.ox.length;
            while (cap < needed) {
                cap <<= 1;
            }
            this.ox = Arrays.copyOf(this.ox, cap);
            this.oy = Arrays.copyOf(this.oy, cap);
            this.oz = Arrays.copyOf(this.oz, cap);
            this.shapes = Arrays.copyOf(this.shapes, cap);
            this.children = Arrays.copyOf(this.children, cap);
        }

        void ensureBlocks(final int needed) {
            if (needed <= this.volumes.length) {
                return;
            }
            int cap = this.volumes.length;
            while (cap < needed) {
                cap <<= 1;
            }
            this.blocks = Arrays.copyOf(this.blocks, cap * 3);
            this.volumes = Arrays.copyOf(this.volumes, cap);
        }

        void releaseRefs() {
            // Do not pin native shapes between rebuilds.
            Arrays.fill(this.shapes, 0, this.count, null);
            Arrays.fill(this.children, 0, this.count, null);
            this.count = 0;
            this.blockCount = 0;
        }
    }

    private static final ThreadLocal<SectionScratch> SECTION_SCRATCH =
            ThreadLocal.withInitial(SectionScratch::new);

    /**
     * Builds one section of a body from its chunk data, or returns null when
     * the section holds no collision leaf. Safe to call from worker threads:
     * reads only the chunk stores and the collider registry.
     */
    @Nullable
    private BodySection buildBodySection(final SableBody sb, final long key) {
        final boolean contraption = sb.kind == SableBody.Kind.CONTRAPTION;
        final ChunkSectionData data = contraption ? sb.chunks.get(key) : this.allChunks.get(key);
        if (data == null) {
            return null;
        }
        final int cx = unpackChunkX(key);
        final int cy = unpackChunkY(key);
        final int cz = unpackChunkZ(key);
        final int blockMinX = cx << 4;
        final int blockMinY = cy << 4;
        final int blockMinZ = cz << 4;

        final SectionScratch scratch = SECTION_SCRATCH.get();
        scratch.count = 0;
        scratch.blockCount = 0;
        boolean special = false;

        final boolean checkBounds = !contraption;
        final boolean hasBounds = sb.hasBounds;
        final int bMinX = sb.minX, bMinY = sb.minY, bMinZ = sb.minZ;
        final int bMaxX = sb.maxX, bMaxY = sb.maxY, bMaxZ = sb.maxZ;

        String rejectLog = null;
        final int[] blocks = data.array();

        try {
            // NOTE: children stay 1:1 with compound leaf sub-shapes — the ID
            // decoding relies on the leaf order matching this traversal.
            for (int by = 0; by < 16; by++) {
                final int worldY = blockMinY + by;
                final int rowBase = (by << 8);
                for (int bz = 0; bz < 16; bz++) {
                    final int worldZ = blockMinZ + bz;
                    final int cellBase = rowBase + (bz << 4);
                    for (int bx = 0; bx < 16; bx++) {
                        final int packed = blocks[cellBase + bx];
                        if (packed == 0) {
                            continue;
                        }
                        final int colliderId = ChunkSectionData.colliderIdOf(packed);
                        // Single registry lookup per block; reused for boxes + shapes.
                        final JoltVoxelColliderData entry = colliderId == 0 ? null : this.colliderRegistry.get(colliderId - 1);
                        final boolean solid = isSolidBlock(packed, entry);
                        final int worldX = blockMinX + bx;

                        if (JoltDebugLogging.STAFF && colliderId > 0 && (!solid || (checkBounds && !(hasBounds
                                && worldX >= bMinX && worldX <= bMaxX
                                && worldY >= bMinY && worldY <= bMaxY
                                && worldZ >= bMinZ && worldZ <= bMaxZ)))) {
                            final String reason = !solid
                                    ? "state=" + ChunkSectionData.voxelStateOf(packed) + (entry == null ? " entry=null" : " boxes=" + entry.boxes.size())
                                    : "outOfBounds";
                            rejectLog = appendReject(rejectLog, worldX, worldY, worldZ, reason);
                        }

                        if (!solid) {
                            continue;
                        }
                        if (checkBounds && !(hasBounds
                                && worldX >= bMinX && worldX <= bMaxX
                                && worldY >= bMinY && worldY <= bMaxY
                                && worldZ >= bMinZ && worldZ <= bMaxZ)) {
                            continue;
                        }

                        final int blockIndex = scratch.blockCount;
                        scratch.ensureBlocks(blockIndex + 1);
                        scratch.blocks[blockIndex * 3] = worldX;
                        scratch.blocks[blockIndex * 3 + 1] = worldY;
                        scratch.blocks[blockIndex * 3 + 2] = worldZ;
                        scratch.volumes[blockIndex] = entry.volume;
                        special |= entry.needsSpecialContacts();
                        scratch.blockCount = blockIndex + 1;

                        final ObjectArrayList<JoltVoxelColliderData.VoxelBox> secBoxes = entry.boxes;
                        final int elementsSize = secBoxes.size();
                        final Object[] elements = secBoxes.elements();

                        for (int i = 0; i < elementsSize; i++) {
                            final JoltVoxelColliderData.VoxelBox box =
                                    (JoltVoxelColliderData.VoxelBox) elements[i];
                            final int idx = scratch.count;
                            scratch.ensure(idx + 1);
                            // Offsets relative to the section corner: small floats,
                            // exact even at plot coordinates in the millions.
                            scratch.ox[idx] = bx + (box.minX + box.maxX) * 0.5f;
                            scratch.oy[idx] = by + (box.minY + box.maxY) * 0.5f;
                            scratch.oz[idx] = bz + (box.minZ + box.maxZ) * 0.5f;
                            scratch.shapes[idx] = entry.shape(i);
                            scratch.children[idx] = new Child(worldX, worldY, worldZ, colliderId, sb);
                            scratch.count = idx + 1;
                        }
                    }
                }
            }

            if (JoltDebugLogging.STAFF && rejectLog != null) {
                Sable.LOGGER.info("[SableJolt] rebuild body {}: rejected: {}", sb.runtimeId, rejectLog);
            }
            final int leafCount = scratch.count;
            if (leafCount == 0) {
                return null;
            }

            final Child[] children = Arrays.copyOf(scratch.children, leafCount);
            final int[] uniqueBlocks = Arrays.copyOf(scratch.blocks, scratch.blockCount * 3);
            final float[] volumes = Arrays.copyOf(scratch.volumes, scratch.blockCount);

            if (leafCount == 1) {
                // A single leaf goes into the parent directly: Jolt's compound
                // create() would return the leaf itself and drop its offset.
                return new BodySection(key, scratch.shapes[0], null,
                        blockMinX + (double) scratch.ox[0], blockMinY + (double) scratch.oy[0], blockMinZ + (double) scratch.oz[0],
                        children, 0, uniqueBlocks, volumes, special);
            }

            final StaticCompoundShapeSettings settings = new StaticCompoundShapeSettings();
            for (int i = 0; i < leafCount; i++) {
                settings.addShape(scratch.ox[i], scratch.oy[i], scratch.oz[i], scratch.shapes[i]);
            }
            final ShapeResult result = settings.create();
            if (result.hasError()) {
                Sable.LOGGER.error("[SableJolt] body {}: section ({},{},{}) shape failed: {}",
                        sb.runtimeId, cx, cy, cz, result.getError());
                return null;
            }
            final ShapeRefC ref = result.get();
            // The compound keeps the settings' frame (leaves relative to the
            // section corner); a parent places that frame's origin, so the
            // section origin is the corner itself.
            return new BodySection(key, ref.getPtr(), ref,
                    blockMinX, blockMinY, blockMinZ,
                    children, subShapeBits(leafCount), uniqueBlocks, volumes, special);
        } catch (final Throwable t) {
            Sable.LOGGER.error("[SableJolt] body {}: failed to build section ({},{},{})", sb.runtimeId, cx, cy, cz, t);
            return null;
        } finally {
            scratch.releaseRefs();
        }
    }

    private static String appendReject(final String log, final int x, final int y, final int z, final String reason) {
        final String entry = "block(" + x + "," + y + "," + z + ") " + reason;
        return log == null ? entry : log + "; " + entry;
    }

    //endregion

    //region Chunks

    /**
     * Stores a chunk section. Global sections get (or refresh) their static
     * terrain body; plot sections mark the owning sub-level's section dirty.
     *
     * @param ownerId runtime id of the sub-level owning a plot section (-1 if unknown)
     */
    public void addChunk(final int x, final int y, final int z, final int[] data, final boolean global, final int ownerId) {
        if (this.disposed) {
            return;
        }
        final long key = ChunkSectionData.packSectionPos(x, y, z);
        final ChunkSectionData existing = this.allChunks.get(key);
        if (global && existing != null && Arrays.equals(existing.array(), data)) {
            // Re-upload of identical terrain (e.g. Sable adopting a section the
            // pipeline uploaded ahead of it): keep the body as is.
            final GlobalChunk chunk = this.globalChunks.get(key);
            if (chunk != null) {
                return;
            }
        }

        final ChunkSectionData section = new ChunkSectionData();
        System.arraycopy(data, 0, section.array(), 0, ChunkSectionData.BLOCKS);
        this.allChunks.put(key, section);

        if (global) {
            this.clearSectionOwner(key);
            GlobalChunk chunk = this.globalChunks.get(key);
            if (chunk == null) {
                chunk = new GlobalChunk(x, y, z, section);
                this.globalChunks.put(key, chunk);
            } else {
                // Reuse the body: only its shape changes.
                chunk.data = section;
            }
            this.dirtyGlobalChunks.remove(chunk);
            this.refreshGlobalChunkBody(chunk);
            if (JoltDebugLogging.STAFF) {
                Sable.LOGGER.info("[SableJolt] addChunk GLOBAL at ({},{},{}) children={} totalGlobal={}",
                        x, y, z, chunk.children.length, this.globalChunks.size());
            }
            return;
        }

        this.removeGlobalChunk(key);
        if (ownerId != -1) {
            this.setSectionOwner(key, ownerId);
            final SableBody sb = this.bodies.get(ownerId);
            if (sb != null && sb.kind == SableBody.Kind.SUB_LEVEL) {
                sb.dirtySections.add(key);
                this.markDirty(sb);
            }
            return;
        }

        // Owner unknown: rebuild every sub-level whose bounds cover the section,
        // mirroring the rapier shared chunk store behavior.
        for (final SableBody sb : this.bodyList) {
            if (sb.kind != SableBody.Kind.SUB_LEVEL || !sb.hasBounds) {
                continue;
            }
            final boolean intersects = (x << 4) <= sb.maxX && ((x << 4) + 15) >= sb.minX
                    && (y << 4) <= sb.maxY && ((y << 4) + 15) >= sb.minY
                    && (z << 4) <= sb.maxZ && ((z << 4) + 15) >= sb.minZ;
            if (intersects) {
                sb.dirtySections.add(key);
                this.markDirty(sb);
            }
        }
    }

    /**
     * Marks a body's shape for rebuild; the rebuild itself is deferred to the next
     * simulation step so bursts of block edits cost a single rebuild.
     */
    private void markDirty(final SableBody sb) {
        synchronized (this.dirtyBodies) {
            this.dirtyBodies.add(sb);
        }
    }

    private void markMountDirty(final SableBody contraption) {
        if (!contraption.embeddedInMount) {
            return;
        }
        final SableBody mount = this.bodies.get(contraption.mountId);
        if (mount != null) {
            mount.parentDirty = true;
            mount.wakeOnAssemble = true;
            this.markDirty(mount);
        }
    }

    private long flushCounter;

    private void flushDirty() {
        this.flushCreativeRestore();
        if (!this.creativeDragBodies.isEmpty()) {
            for (final int joltId : this.creativeDragBodies) {
                final SableBody sb = this.bodyByJoltId(joltId);
                if (sb != null) {
                    sb.dragPenDepth = 0.0;
                }
            }
        }
        this.flushCounter++;
        if ((this.flushCounter & 31) == 0) {
            this.retryPendingCreation();
            this.retryPendingTerrain();
        }
        if (!this.dirtyBodies.isEmpty()) {
            final ObjectArrayList<SableBody> list;
            synchronized (this.dirtyBodies) {
                list = new ObjectArrayList<>(this.dirtyBodies);
                this.dirtyBodies.clear();
            }
            // Contraptions first: mounts embed their freshly assembled shapes.
            final ObjectArrayList<SableBody> contraptions = new ObjectArrayList<>();
            final ObjectArrayList<SableBody> others = new ObjectArrayList<>(list.size());
            for (final SableBody sb : list) {
                if (this.bodies.get(sb.runtimeId) != sb) {
                    continue; // removed since it was marked
                }
                (sb.kind == SableBody.Kind.CONTRAPTION ? contraptions : others).add(sb);
            }
            if (!contraptions.isEmpty()) {
                this.rebuildAll(contraptions);
                for (final SableBody c : contraptions) {
                    if (c.embeddedInMount) {
                        final SableBody mount = this.bodies.get(c.mountId);
                        if (mount != null && !others.contains(mount)) {
                            mount.parentDirty = true;
                            mount.wakeOnAssemble = true;
                            others.add(mount);
                        }
                    }
                }
            }
            this.rebuildAll(others);
        }
        if (!this.dirtyGlobalChunks.isEmpty()) {
            for (final GlobalChunk chunk : this.dirtyGlobalChunks) {
                try {
                    this.refreshGlobalChunkBody(chunk);
                } catch (final Throwable t) {
                    Sable.LOGGER.error("[SableJolt] failed to refresh global chunk ({},{},{})", chunk.cx, chunk.cy, chunk.cz, t);
                }
            }
            this.dirtyGlobalChunks.clear();
        }
    }

    private void rebuildAll(final ObjectArrayList<SableBody> list) {
        if (list.size() >= 8) {
            // One body per task; sections stay sequential inside (no nested pool use).
            this.runParallel(list, sb -> {
                try {
                    this.rebuildShape(sb, false);
                } catch (final Throwable t) {
                    Sable.LOGGER.error("[SableJolt] failed to rebuild body {}", sb.runtimeId, t);
                }
            });
        } else {
            for (final SableBody sb : list) {
                try {
                    this.rebuildShape(sb, true);
                } catch (final Throwable t) {
                    Sable.LOGGER.error("[SableJolt] failed to rebuild body {}", sb.runtimeId, t);
                }
            }
        }
    }

    public boolean hasChunk(final int sectionX, final int sectionY, final int sectionZ) {
        return this.allChunks.containsKey(ChunkSectionData.packSectionPos(sectionX, sectionY, sectionZ));
    }

    public void removeChunk(final int x, final int y, final int z, final boolean global) {
        final long key = ChunkSectionData.packSectionPos(x, y, z);
        this.allChunks.remove(key);

        if (global) {
            this.removeGlobalChunk(key);
            return;
        }
        final int owner = this.clearSectionOwner(key);
        final SableBody sb = owner == -1 ? null : this.bodies.get(owner);
        if (sb != null && sb.kind == SableBody.Kind.SUB_LEVEL) {
            sb.dirtySections.add(key);
            this.markDirty(sb);
        }
    }

    private void removeGlobalChunk(final long key) {
        final GlobalChunk chunk = this.globalChunks.remove(key);
        if (chunk == null) {
            return;
        }
        this.fluidSections.remove(key);
        this.destroyGlobalChunkBody(chunk);
        // a removed chunk must not be touched by a pending flush
        this.dirtyGlobalChunks.remove(chunk);
        this.pendingTerrainBodies.remove(chunk);
    }

    private void destroyGlobalChunkBody(final GlobalChunk chunk) {
        if (chunk.joltId != 0) {
            this.globalChunksByBodyId.remove(chunk.joltId);
            this.bi.removeBody(chunk.joltId);
            this.bi.destroyBody(chunk.joltId);
            chunk.joltId = 0;
            chunk.body = null;
        }
    }

    public void changeBlock(final int x, final int y, final int z, final int newState) {
        this.changeBlocks(new int[]{x}, new int[]{y}, new int[]{z}, new int[]{newState}, null, 1);
    }

    /**
     * Applies block edits. Each edit goes to the owner of its section: the
     * sub-level owning the plot section (only that section is rebuilt), or the
     * terrain body. Edits in sections no sub-level needs are ignored, like the
     * rapier pipeline does.
     *
     * @param owners per edit, the runtime id of the sub-level whose plot holds
     *               the block, or -1 for the world; null to use the recorded owners
     */
    public void changeBlocks(final int[] wx, final int[] wy, final int[] wz,
                             final int[] packed, @Nullable final int[] owners, final int count) {
        if (this.disposed) {
            return;
        }
        for (int i = 0; i < count; i++) {
            final long key = ChunkSectionData.packSectionPos(wx[i] >> 4, wy[i] >> 4, wz[i] >> 4);
            int owner = owners != null ? owners[i] : this.sectionOwners.getOrDefault(key, -1);
            ChunkSectionData data = this.allChunks.get(key);
            SableBody sb = owner == -1 ? null : this.bodies.get(owner);
            if (sb != null && sb.kind != SableBody.Kind.SUB_LEVEL) {
                sb = null;
            }
            if (data == null) {
                if (sb == null) {
                    continue;
                }
                // A plot section without data starts out as air: re-create it so
                // blocks placed into a reused plot slot always collide.
                data = new ChunkSectionData();
                this.allChunks.put(key, data);
            }
            data.set(wx[i] & 15, wy[i] & 15, wz[i] & 15, packed[i]);

            if (sb != null) {
                this.setSectionOwner(key, owner);
                sb.dirtySections.add(key);
                this.markDirty(sb);
                continue;
            }
            final GlobalChunk chunk = this.globalChunks.get(key);
            if (chunk != null) {
                this.dirtyGlobalChunks.add(chunk);
            }
        }
    }

    /**
     * Legacy overload without owners: uses the recorded section owners.
     */
    public void changeBlocks(final int[] wx, final int[] wy, final int[] wz,
                             final int[] packed, final int count) {
        this.changeBlocks(wx, wy, wz, packed, null, count);
    }

    /**
     * Scratch classification arrays for {@link #buildGlobalChunkShape} (server
     * thread only — chunk bodies are refreshed sequentially): reuses 20 KB per
     * refresh instead of allocating it.
     */
    private final int[] globalCellScratch = new int[ChunkSectionData.BLOCKS];
    private final boolean[] globalUsedScratch = new boolean[ChunkSectionData.BLOCKS];

    /**
     * Rebuilds the static chunk body's shape from the section data. Full cubes
     * of the same block are greedily merged into slabs: a platform sliding over
     * per-block ground boxes sinks a few mm (contact slop) and its leading edge
     * then hits the vertical seam wall of the next box, which reads as
     * "tripping over a stone". Merged slabs remove the seams within a chunk.
     * Blocks with collision callbacks keep individual boxes so callbacks see
     * the exact block. The shape is local to the section corner.
     */
    private void buildGlobalChunkShape(final GlobalChunk chunk) {
        final Vec3 mutableVec3 = new Vec3();
        final StaticCompoundShapeSettings settings = new StaticCompoundShapeSettings();
        final ObjectArrayList<Child> children = new ObjectArrayList<>();
        final ObjectArrayList<ConstShape> leafShapes = new ObjectArrayList<>();
        final FloatArrayCollector offsets = new FloatArrayCollector();

        final int baseX = chunk.cx << 4;
        final int baseY = chunk.cy << 4;
        final int baseZ = chunk.cz << 4;
        final int[] data = chunk.data.array();

        boolean anyFluid = false;
        boolean special = false;

        // Pass 1: classify cells (positive id = mergeable full cube, negative =
        // individual boxes).
        final int[] cellCollider = this.globalCellScratch;
        Arrays.fill(cellCollider, 0);
        for (int idx = 0; idx < ChunkSectionData.BLOCKS; idx++) {
            final int packed = data[idx];
            if (packed == 0) {
                continue;
            }
            final int colliderId = ChunkSectionData.colliderIdOf(packed);
            final JoltVoxelColliderData entry = colliderId == 0 ? null : this.colliderRegistry.get(colliderId - 1);
            if (entry == null) {
                continue;
            }
            // fluids are marked for buoyancy even though they never collide
            if (entry.isFluid) {
                anyFluid = true;
            }
            if (!isSolidBlock(packed, entry)) {
                continue;
            }
            final ObjectArrayList<JoltVoxelColliderData.VoxelBox> boxes = entry.boxes;
            final boolean mergeable = boxes.size() == 1 && boxes.get(0).fullCube() && entry.contactEvents == null;
            cellCollider[idx] = mergeable ? colliderId : -colliderId;
            special |= entry.needsSpecialContacts();
        }

        // Pass 2a: greedy rectangles per Y layer over mergeable cells.
        final boolean[] used = this.globalUsedScratch;
        Arrays.fill(used, false);
        for (int by = 0; by < 16; by++) {
            final int layerBase = by << 8;
            for (int bz = 0; bz < 16; bz++) {
                final int rowBase = layerBase + (bz << 4);
                for (int bx = 0; bx < 16; bx++) {
                    final int idx = rowBase + bx;
                    final int id = cellCollider[idx];
                    if (id <= 0 || used[idx]) {
                        continue;
                    }
                    // extend along X
                    int x1 = bx;
                    while (x1 + 1 < 16 && !used[rowBase + x1 + 1] && cellCollider[rowBase + x1 + 1] == id) {
                        x1++;
                    }
                    // extend along Z while the whole X span matches
                    int z1 = bz;
                    zLoop:
                    while (z1 + 1 < 16) {
                        final int nextRow = layerBase + ((z1 + 1) << 4);
                        for (int x = bx; x <= x1; x++) {
                            final int nIdx = nextRow + x;
                            if (used[nIdx] || cellCollider[nIdx] != id) {
                                break zLoop;
                            }
                        }
                        z1++;
                    }
                    for (int z = bz; z <= z1; z++) {
                        final int r = layerBase + (z << 4);
                        for (int x = bx; x <= x1; x++) {
                            used[r + x] = true;
                        }
                    }

                    // The slab needs its OWN box shape sized to the merge — reusing
                    // the unit box would leave collision only in its center cell.
                    // Shapes are shared by size across the whole world.
                    final float sx = (x1 - bx + 1) * 0.5f;
                    final float sz = (z1 - bz + 1) * 0.5f;
                    leafShapes.add(this.sharedBoxShape(sx, 0.5f, sz));
                    offsets.add(bx + sx, by + 0.5f, bz + sz);
                    children.add(new Child(baseX + bx, baseY + by, baseZ + bz, id, null));
                }
            }
        }

        // Pass 2b: other colliders keep their individual boxes.
        for (int idx = 0; idx < ChunkSectionData.BLOCKS; idx++) {
            final int colliderId = cellCollider[idx];
            if (colliderId >= 0) {
                continue; // air or already merged
            }
            final int id = -colliderId;
            final JoltVoxelColliderData entry = this.colliderRegistry.get(id - 1);
            if (entry == null || entry.isFluid) {
                continue;
            }
            final int bx = idx & 15;
            final int bz = (idx >> 4) & 15;
            final int by = idx >> 8;
            final ObjectArrayList<JoltVoxelColliderData.VoxelBox> boxes = entry.boxes;
            final int size = boxes.size();
            final Object[] elements = boxes.elements();
            for (int i = 0; i < size; i++) {
                final JoltVoxelColliderData.VoxelBox box = (JoltVoxelColliderData.VoxelBox) elements[i];
                leafShapes.add(entry.shape(i));
                offsets.add(bx + (box.minX + box.maxX) * 0.5f,
                        by + (box.minY + box.maxY) * 0.5f,
                        bz + (box.minZ + box.maxZ) * 0.5f);
                children.add(new Child(baseX + bx, baseY + by, baseZ + bz, id, null));
            }
        }

        final long key = ChunkSectionData.packSectionPos(chunk.cx, chunk.cy, chunk.cz);
        if (anyFluid) {
            this.fluidSections.add(key);
        } else {
            this.fluidSections.remove(key);
        }
        chunk.special = special;

        final int leafCount = children.size();
        if (leafCount == 0) {
            chunk.shape = null;
            chunk.children = NO_CHILDREN;
            chunk.leafBits = 0;
            return;
        }
        if (leafCount == 1) {
            // A single leaf cannot carry an offset on its own: wrap it so the
            // body can stay at the section corner.
            final float[] o = offsets.array();
            mutableVec3.set(o[0], o[1], o[2]);
            final MutableCompoundShape single = new MutableCompoundShape();
            single.addShape(mutableVec3, IDENTITY_QUAT, leafShapes.get(0));
            chunk.shape = single;
            chunk.children = children.toArray(NO_CHILDREN);
            chunk.leafBits = 0;
            return;
        }
        final float[] o = offsets.array();
        for (int i = 0; i < leafCount; i++) {
            settings.addShape(o[i * 3], o[i * 3 + 1], o[i * 3 + 2], leafShapes.get(i));
        }
        final ShapeResult result = settings.create();
        if (result.hasError()) {
            Sable.LOGGER.error("[SableJolt] terrain section ({},{},{}) shape failed: {}", chunk.cx, chunk.cy, chunk.cz, result.getError());
            chunk.shape = null;
            chunk.children = NO_CHILDREN;
            chunk.leafBits = 0;
            return;
        }
        chunk.shape = result.get();
        chunk.children = children.toArray(NO_CHILDREN);
        chunk.leafBits = subShapeBits(leafCount);
    }

    /**
     * Growable float triple list (terrain leaf offsets).
     */
    private static final class FloatArrayCollector {
        private float[] data = new float[96];
        private int size;

        void add(final float x, final float y, final float z) {
            if (this.size + 3 > this.data.length) {
                this.data = Arrays.copyOf(this.data, this.data.length * 2);
            }
            this.data[this.size++] = x;
            this.data[this.size++] = y;
            this.data[this.size++] = z;
        }

        float[] array() {
            return this.data;
        }
    }

    /**
     * Creates, updates or removes the static body of a global chunk so that it
     * matches the chunk's current shape and data.
     */
    private void refreshGlobalChunkBody(final GlobalChunk chunk) {
        // Staleness guard: a chunk that was removed can linger in
        // dirtyGlobalChunks. Its body is already destroyed — touching it again
        // would double-free in native Jolt.
        if (this.globalChunks.get(ChunkSectionData.packSectionPos(chunk.cx, chunk.cy, chunk.cz)) != chunk) {
            return;
        }

        this.buildGlobalChunkShape(chunk);
        this.ensureGlobalChunkBody(chunk);
    }

    /**
     * Creates or updates the static body for the chunk's current shape (built
     * by {@link #buildGlobalChunkShape}), or removes it when the shape is empty.
     */
    private void ensureGlobalChunkBody(final GlobalChunk chunk) {
        if (chunk.children.length == 0 || chunk.shape == null) {
            this.destroyGlobalChunkBody(chunk);
            this.pendingTerrainBodies.remove(chunk);
            return;
        }

        if (chunk.joltId == 0) {
            // Keep room for sub-levels: terrain yields first when the world fills up.
            if (this.system.getNumBodies() >= this.config.maxBodies - this.terrainReserve()) {
                this.pendingTerrainBodies.add(chunk);
                this.warnCapacity("terrain section (" + chunk.cx + "," + chunk.cy + "," + chunk.cz + ") is waiting for a free body");
                return;
            }
            final BodyCreationSettings bcs = new BodyCreationSettings(chunk.shape,
                    chunk.cx << 4, chunk.cy << 4, chunk.cz << 4, 0, 0, 0, 1,
                    EMotionType.Static,
                    chunk.special ? LAYER_STATIC_SPECIAL : LAYER_STATIC
            );
            if (this.rapierMode) {
                bcs.setFriction(FRICTION_TERRAIN_RAPIER);
            }
            final Body body;
            try {
                body = this.bi.createBody(bcs);
            } catch (final IllegalStateException e) {
                this.pendingTerrainBodies.add(chunk);
                this.warnCapacity("terrain section (" + chunk.cx + "," + chunk.cy + "," + chunk.cz + ") is waiting for a free body");
                return;
            }
            body.setUserData(-1L);
            this.bi.addBody(body, EActivation.DontActivate);
            chunk.joltId = body.getId();
            chunk.body = body;
            chunk.layerSpecial = chunk.special;
            this.globalChunksByBodyId.put(chunk.joltId, chunk);
            this.pendingTerrainBodies.remove(chunk);
        } else {
            this.bi.setShape(chunk.joltId, chunk.shape, false, EActivation.DontActivate);
            if (chunk.layerSpecial != chunk.special) {
                this.bi.setObjectLayer(chunk.joltId, chunk.special ? LAYER_STATIC_SPECIAL : LAYER_STATIC);
                chunk.layerSpecial = chunk.special;
            }
        }
    }

    private int terrainReserve() {
        return Math.max(1024, this.config.maxBodies / 16);
    }

    private void retryPendingTerrain() {
        if (this.pendingTerrainBodies.isEmpty()) {
            return;
        }
        final int limit = this.config.maxBodies - this.terrainReserve();
        for (final GlobalChunk chunk : new ObjectArrayList<>(this.pendingTerrainBodies)) {
            if (this.system.getNumBodies() >= limit) {
                return;
            }
            // The shape is current (data edits rebuild it via dirtyGlobalChunks).
            if (this.globalChunks.get(ChunkSectionData.packSectionPos(chunk.cx, chunk.cy, chunk.cz)) == chunk) {
                this.ensureGlobalChunkBody(chunk);
            } else {
                this.pendingTerrainBodies.remove(chunk);
            }
        }
    }

    private long lastCapacityWarnNanos;

    private void warnCapacity(final String what) {
        final long now = System.nanoTime();
        if (now - this.lastCapacityWarnNanos > 30_000_000_000L) {
            this.lastCapacityWarnNanos = now;
            Sable.LOGGER.warn("[SableJolt] Physics body limit reached ({} bodies, max_bodies={}): {}. "
                            + "Raise max_bodies in config/sable_jolt_physics.properties.",
                    this.system.getNumBodies(), this.config.maxBodies, what);
        }
    }

    //endregion

    //region Kinematic contraptions

    /**
     * Creates a kinematic contraption. Contraptions mounted on a sub-level
     * become part of the sub-level's compound (Rapier attaches their collider
     * to the mount's rigid body): they push and support their mount and never
     * collide with it. World contraptions get their own kinematic body.
     */
    public void createKinematicContraption(final int mountId, final int id, final Vector3dc pos, final Quaterniondc rot) {
        if (this.disposed) {
            return;
        }
        final SableBody previous = this.bodies.get(id);
        if (previous != null) {
            this.destroyBody(previous);
        }
        final SableBody sb;
        if (mountId == -1) {
            sb = this.createBody(SableBody.Kind.CONTRAPTION, id, pos, rot, LAYER_MOVING);
        } else {
            sb = new SableBody(SableBody.Kind.CONTRAPTION, id);
            this.bodies.put(id, sb);
            final SableBody mount = this.bodies.get(mountId);
            if (mount != null && mount.kind == SableBody.Kind.SUB_LEVEL) {
                mount.mountedContraptions.add(sb);
                sb.embeddedInMount = true;
                mount.parentDirty = true;
                mount.wakeOnAssemble = true;
                this.markDirty(mount);
            }
        }
        sb.mountId = mountId;
        sb.relPos.set(pos);
        sb.relRot.set(rot);
        this.contraptionList.add(sb);
    }

    private void attachWaitingContraptions(final SableBody mount) {
        for (final SableBody c : this.contraptionList) {
            if (c.mountId == mount.runtimeId && !c.embeddedInMount) {
                mount.mountedContraptions.add(c);
                c.embeddedInMount = true;
                mount.parentDirty = true;
                mount.wakeOnAssemble = true;
                this.markDirty(mount);
            }
        }
    }

    private void detachFromMount(final SableBody contraption) {
        if (!contraption.embeddedInMount) {
            return;
        }
        contraption.embeddedInMount = false;
        final SableBody mount = this.bodies.get(contraption.mountId);
        if (mount != null && mount.mountedContraptions.remove(contraption)) {
            mount.parentDirty = true;
            mount.wakeOnAssemble = true;
            this.markDirty(mount);
        }
    }

    /**
     * Section keys of global chunks that contain at least one fluid block; buoyancy
     * is skipped entirely when empty.
     */
    private final LongOpenHashSet fluidSections = new LongOpenHashSet();

    public void removeKinematicContraption(final int id) {
        final SableBody sb = this.bodies.get(id);
        if (sb != null && sb.kind == SableBody.Kind.CONTRAPTION) {
            this.destroyBody(sb);
        }
    }

    /**
     * Uploads a kinematic contraption's pose: its position in the mount's block
     * space (world space when unmounted), orientation and velocities in its own
     * frame. All arguments are copied immediately; no allocations are performed.
     */
    public void setKinematicContraptionTransform(final int id, final Vector3dc centerOfMass, final Vector3dc pos,
                                                 final Quaterniondc rot, final Vector3dc linVel, final Vector3dc angVel) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null || sb.kind != SableBody.Kind.CONTRAPTION) {
            return;
        }
        if (!(Double.isFinite(pos.x()) && Double.isFinite(pos.y()) && Double.isFinite(pos.z())
                && Double.isFinite(rot.x()) && Double.isFinite(rot.y()) && Double.isFinite(rot.z()) && Double.isFinite(rot.w()))) {
            return;
        }
        if (centerOfMass != null && Double.isFinite(centerOfMass.x()) && Double.isFinite(centerOfMass.y()) && Double.isFinite(centerOfMass.z())
                && (centerOfMass.x() != sb.centerOfMass.x || centerOfMass.y() != sb.centerOfMass.y || centerOfMass.z() != sb.centerOfMass.z)) {
            sb.centerOfMass.set(centerOfMass);
            sb.parentDirty = true;
            this.markDirty(sb);
        }
        final boolean moved = !sb.relPos.equals(pos.x(), pos.y(), pos.z())
                || sb.relRot.x != rot.x() || sb.relRot.y != rot.y() || sb.relRot.z != rot.z() || sb.relRot.w != rot.w();
        sb.relPos.set(pos);
        sb.relRot.set(rot);
        sb.linVel.set(linVel);
        sb.angVel.set(angVel);
        if (moved) {
            this.markMountDirty(sb);
        }
    }

    public void addKinematicContraptionChunkSection(final int id, final int x, final int y, final int z, final int[] data) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null || sb.kind != SableBody.Kind.CONTRAPTION) {
            return;
        }
        final ChunkSectionData section = new ChunkSectionData();
        System.arraycopy(data, 0, section.array(), 0, ChunkSectionData.BLOCKS);
        final long key = ChunkSectionData.packSectionPos(x, y, z);
        sb.chunks.put(key, section);
        sb.dirtySections.add(key);
        this.markDirty(sb);
    }

    //endregion

    //region Simulation

    /**
     * Runs once per game tick before the substeps: the Rapier buoyancy model
     * computes its forces here and applies them in every substep (Rapier's
     * forces persist across the tick's substeps).
     */
    public void prePhysicsTick() {
        if (this.disposed) {
            return;
        }
        if (this.rapierMode) {
            this.computeTickBuoyancy();
        }
    }

    private long lastUpdateErrorNanos;

    /**
     * Steps the simulation by one substep.
     */
    public void step(final double timeStep) {
        if (this.disposed) {
            return;
        }
        final long stepStartNanos = System.nanoTime();
        this.flushDirty();
        if (this.rapierMode) {
            this.applyTickBuoyancy();
        } else {
            this.computeBuoyancy();
        }
        this.updateContraptionMotion((float) timeStep);
        // small scenes: a single-threaded job system avoids the wake/sync overhead
        // of the thread pool, which dominates when the solver has almost no work.
        // A single body with a huge compound is NOT a small scene: its narrow phase
        // dominates the step, so heavy bodies force the multithreaded job system.
        // The flag is maintained by refreshBodySnapshot (one tick stale at most).
        final boolean heavyBody = this.snapshotHeavyBody;
        final JobSystem jobs = this.bodyList.size() >= MULTITHREAD_BODY_THRESHOLD || heavyBody
                ? this.jobSystem : this.singleThreadJobSystem;

        // Adaptive solver depth (Jolt mode): a huge compound body resting on the
        // ground forms a contact island with thousands of points; halving the
        // iteration depth there roughly halves the solver cost. Exception: while
        // a creative drag is active the full depth is kept.
        final boolean reducedSolver = !this.rapierMode && heavyBody && this.creativeDragBodies.isEmpty();
        if (reducedSolver != this.reducedSolverApplied) {
            final PhysicsSettings physics = this.system.getPhysicsSettings();
            physics.setNumVelocitySteps(reducedSolver ? Math.max(5, this.configuredVelocitySteps / 2) : this.configuredVelocitySteps);
            physics.setNumPositionSteps(reducedSolver ? Math.max(2, this.configuredPositionSteps / 2) : this.configuredPositionSteps);
            this.system.setPhysicsSettings(physics);
            this.reducedSolverApplied = reducedSolver;
        }

        final int errors = this.system.update((float) timeStep, 1, this.tempAllocator, jobs);
        if (errors != EPhysicsUpdateError.None) {
            this.warnUpdateErrors(errors);
        }
        // refreshBodySnapshot also applies the huge-body tunneling guard inline
        // (velocities are already in registers there) instead of a separate pass.
        this.refreshBodySnapshot();
        if (JoltDebugLogging.STAFF) {
            this.staffWorldDump();
        }
        // Failsafe (Jolt mode): a huge compound landing with deep penetration can
        // collapse the solver into an ever-growing contact island. If the step
        // overran its budget, put oversized slow bodies to sleep so the world can
        // recover instead of freezing every subsequent tick.
        final long updateNanos = System.nanoTime() - stepStartNanos;
        if (!this.rapierMode && updateNanos > 250_000_000L) {
            for (final SableBody sb : this.bodyList) {
                if (sb.leafCount >= HEAVY_BODY_CHILDREN && sb.body.isActive()
                        && !this.creativeDragBodies.contains(sb.joltId)) {
                    if (sb.snapVx * sb.snapVx + sb.snapVy * sb.snapVy + sb.snapVz * sb.snapVz < 0.5f) {
                        this.bi.deactivateBody(sb.joltId);
                    }
                }
            }
        }
        this.pushDraggedOutOfPenetration();
    }

    private void warnUpdateErrors(final int errors) {
        final long now = System.nanoTime();
        if (now - this.lastUpdateErrorNanos < 30_000_000_000L) {
            return;
        }
        this.lastUpdateErrorNanos = now;
        final StringBuilder what = new StringBuilder();
        if ((errors & EPhysicsUpdateError.ManifoldCacheFull) != 0 || (errors & EPhysicsUpdateError.ContactConstraintsFull) != 0) {
            what.append(" max_contact_constraints=").append(this.config.maxContactConstraints);
        }
        if ((errors & EPhysicsUpdateError.BodyPairCacheFull) != 0) {
            what.append(" max_body_pairs=").append(this.config.maxBodyPairs);
        }
        Sable.LOGGER.warn("[SableJolt] Physics capacity exceeded (error flags {}), some contacts were skipped this step;"
                + " raise{} in config/sable_jolt_physics.properties", errors, what);
    }

    /**
     * Allowed embed depth (m) of a dragged body before the push-out engages;
     * contact slop below this is normal solver behavior.
     */
    private static final double DRAG_PUSH_SLOP = 0.12;
    /**
     * Max push-out distance (m) per step for a dragged body.
     */
    private static final double DRAG_PUSH_MAX = 1.5;

    /**
     * Nudges creatively dragged bodies out of deep penetration. Rotating or
     * dragging a huge body can sweep its parts meters into the terrain within
     * a single tick — faster than the solver's per-step recovery — which would
     * otherwise ratchet the body underground. The snapshot fields are updated
     * together with the body so later reads stay consistent.
     */
    private void pushDraggedOutOfPenetration() {
        if (this.creativeDragBodies.isEmpty()) {
            return;
        }
        for (final int joltId : this.creativeDragBodies) {
            final SableBody sb = this.bodyByJoltId(joltId);
            if (sb == null || sb.dragPenDepth <= DRAG_PUSH_SLOP) {
                continue;
            }
            final double push = Math.min(sb.dragPenDepth - DRAG_PUSH_SLOP, DRAG_PUSH_MAX);
            sb.snapX += sb.dragPenNx * push;
            sb.snapY += sb.dragPenNy * push;
            sb.snapZ += sb.dragPenNz * push;
            sb.snapComX = sb.snapX;
            sb.snapComY = sb.snapY;
            sb.snapComZ = sb.snapZ;
            this.bi.setPositionAndRotation(joltId, sb.snapX, sb.snapY, sb.snapZ,
                    sb.snapQx, sb.snapQy, sb.snapQz, sb.snapQw, EActivation.DontActivate);
        }
    }

    /**
     * Whether any tracked body currently holds a huge compound; maintained by
     * {@link #refreshBodySnapshot} so the pre-step job-system choice needs no
     * dedicated scan (one tick stale at most — a pure performance heuristic).
     */
    private boolean snapshotHeavyBody;

    /**
     * Native list of the awake bodies, filled by Jolt every step.
     */
    private final BodyIdVector activeIds = new BodyIdVector();
    /**
     * Bodies read by the previous refresh because they were awake: the ones
     * that fell asleep during this step are read one last time.
     */
    private ObjectArrayList<SableBody> lastAwake = new ObjectArrayList<>();
    private ObjectArrayList<SableBody> currentAwake = new ObjectArrayList<>();
    /**
     * Bodies whose snapshot must be re-read (created, or velocity changed by an impulse).
     */
    private final ObjectArrayList<SableBody> staleSnapshots = new ObjectArrayList<>();
    private final ObjectArrayList<SableBody> extraSnapshots = new ObjectArrayList<>();
    private java.nio.LongBuffer snapshotUser;

    /**
     * Marks a body's snapshot stale: reads fall back to the native body until
     * the next refresh re-reads it.
     */
    private void invalidateSnapshot(final SableBody sb) {
        sb.snapValid = false;
        if (!sb.snapQueued) {
            sb.snapQueued = true;
            this.staleSnapshots.add(sb);
        }
    }

    /**
     * Reads the pose and velocity of the bodies that can have moved this step:
     * the awake ones (Jolt's active list, read in place), the ones that fell
     * asleep during the step and stale ones. Sleeping bodies keep their
     * snapshot, so a world of resting sub-levels costs nothing here; the old
     * read of every body cost ~12 ms per substep at 10k bodies. Reads use the
     * lock-free body interface: nothing else touches bodies between steps.
     * Cached values are served by {@link #getPose}, the velocity getters and
     * the buoyancy passes.
     */
    private void refreshBodySnapshot() {
        final long gen = ++this.snapshotGen;
        final ObjectArrayList<SableBody> awake = this.currentAwake;
        awake.clear();
        boolean heavyBody = false;

        this.system.getActiveBodies(com.github.stephengold.joltjni.enumerate.EBodyType.RigidBody, this.activeIds);
        final int activeCount = this.activeIds.size();
        if (activeCount > 0) {
            final BodyIdArray ids = this.activeIds.data();
            this.ensureSnapshotBuffers(activeCount);
            this.batchNoLock.getUserData(ids, this.snapshotUser);
            this.readSnapshotBatch(ids, false);
            for (int idx = 0; idx < activeCount; idx++) {
                final long user = this.snapshotUser.get(idx);
                if (user < 0L || user > Integer.MAX_VALUE) {
                    continue; // rope points
                }
                final SableBody sb = this.bodies.get((int) user);
                if (sb == null || sb.body == null) {
                    continue;
                }
                sb.snapActive = true;
                heavyBody |= this.applySnapshot(sb, idx, gen);
                awake.add(sb);
            }
        }

        // Bodies that fell asleep this step, and stale snapshots.
        final ObjectArrayList<SableBody> extras = this.extraSnapshots;
        extras.clear();
        for (final SableBody sb : this.lastAwake) {
            if (sb.snapGen != gen && sb.body != null && this.bodies.get(sb.runtimeId) == sb) {
                sb.snapGen = gen;
                extras.add(sb);
            }
        }
        for (final SableBody sb : this.staleSnapshots) {
            sb.snapQueued = false;
            if (sb.snapGen != gen && sb.body != null && this.bodies.get(sb.runtimeId) == sb) {
                sb.snapGen = gen;
                extras.add(sb);
            }
        }
        this.staleSnapshots.clear();
        if (!extras.isEmpty()) {
            final int n = extras.size();
            if (this.snapshotIds == null || this.snapshotIds.length() < n) {
                final int capacity = Math.max(64, n + (n >> 1));
                this.snapshotIds = new BodyIdArray(capacity);
                // Every slot always holds a valid id: unused ones the static ground.
                for (int i = 0; i < capacity; i++) {
                    this.snapshotIds.set(i, this.groundBodyId);
                }
                this.snapshotIdsFilled = 0;
            }
            this.ensureSnapshotBuffers(this.snapshotIds.length());
            for (int i = 0; i < n; i++) {
                this.snapshotIds.set(i, extras.get(i).joltId);
            }
            for (int i = n; i < this.snapshotIdsFilled; i++) {
                this.snapshotIds.set(i, this.groundBodyId);
            }
            this.snapshotIdsFilled = n;
            this.readSnapshotBatch(this.snapshotIds, true);
            for (int i = 0; i < n; i++) {
                final SableBody sb = extras.get(i);
                sb.snapActive = this.snapshotActive.get(i) != 0;
                heavyBody |= this.applySnapshot(sb, i, gen);
                if (sb.snapActive) {
                    awake.add(sb);
                }
            }
        }

        // Contact routing for the next step (only bodies that changed this step).
        for (final SableBody sb : awake) {
            this.classifyLayer(sb);
        }
        for (final SableBody sb : extras) {
            if (!sb.snapActive) {
                this.classifyLayer(sb);
            }
        }

        this.currentAwake = this.lastAwake;
        this.lastAwake = awake;
        this.snapshotHeavyBody = heavyBody;
    }

    /**
     * Leading slots of {@link #snapshotIds} holding body ids (the rest is ground padding).
     */
    private int snapshotIdsFilled;

    private void setBodyLayer(final SableBody sb, final int layer) {
        if (sb.body == null || sb.layer == layer) {
            return;
        }
        this.bi.setObjectLayer(sb.joltId, layer);
        sb.layer = layer;
    }

    /**
     * Report layer for bodies whose contacts need Java (special blocks,
     * mounted contraptions, creative drag) or that move fast enough for an
     * impact report (linear speed plus spin times the body radius).
     */
    private void classifyLayer(final SableBody sb) {
        if (sb.body == null || sb.layer == LAYER_ROPE) {
            return;
        }
        final int desired;
        if (sb.layerSpecial || this.creativeDragBodies.contains(sb.joltId)) {
            desired = LAYER_MOVING_REPORT;
        } else if (!sb.snapActive) {
            desired = LAYER_MOVING;
        } else {
            final float v = (float) Math.sqrt(sb.snapVx * sb.snapVx + sb.snapVy * sb.snapVy + sb.snapVz * sb.snapVz);
            final float w = (float) Math.sqrt(sb.snapAx * sb.snapAx + sb.snapAy * sb.snapAy + sb.snapAz * sb.snapAz);
            final float bound = v + w * sb.boundRadius;
            // Hysteresis: a layer change re-files the body in the broad phase,
            // so bodies hovering around the threshold must not switch every step.
            if (sb.layer == LAYER_MOVING_REPORT) {
                desired = bound >= REPORT_SPEED * 0.6f ? LAYER_MOVING_REPORT : LAYER_MOVING;
            } else {
                desired = bound >= REPORT_SPEED ? LAYER_MOVING_REPORT : LAYER_MOVING;
            }
        }
        this.setBodyLayer(sb, desired);
    }

    private void readSnapshotBatch(final ConstBodyIdArray ids, final boolean withActive) {
        // Center of mass positions skip the per-body shape lookup of getPositions;
        // for these bodies both are the same point.
        this.batchNoLock.getCenterOfMassPositions(ids, this.snapshotPos);
        this.batchNoLock.getRotations(ids, this.snapshotRot);
        this.batchNoLock.getLinearVelocities(ids, this.snapshotLin);
        this.batchNoLock.getAngularVelocities(ids, this.snapshotAng);
        if (withActive) {
            this.batchNoLock.areActive(ids, this.snapshotActive);
        }
    }

    /**
     * Stores one batch-read entry into a body's snapshot (with the poison
     * watchdog and the Jolt-mode tunneling guard). Returns whether the body
     * holds a huge compound.
     */
    private boolean applySnapshot(final SableBody sb, final int idx, final long gen) {
        sb.snapGen = gen;
        sb.snapValid = true;
        final double px = this.snapshotPos.get(idx * 3);
        final double py = this.snapshotPos.get(idx * 3 + 1);
        final double pz = this.snapshotPos.get(idx * 3 + 2);
        // Every compound here keeps its center of mass at the shape origin
        // (bodies are positioned at their center of mass), so the body
        // position is the center of mass position.
        final double cx = px;
        final double cy = py;
        final double cz = pz;
        final float qx = this.snapshotRot.get(idx * 4);
        final float qy = this.snapshotRot.get(idx * 4 + 1);
        final float qz = this.snapshotRot.get(idx * 4 + 2);
        final float qw = this.snapshotRot.get(idx * 4 + 3);
        float vx = this.snapshotLin.get(idx * 3);
        float vy = this.snapshotLin.get(idx * 3 + 1);
        float vz = this.snapshotLin.get(idx * 3 + 2);
        final float ax = this.snapshotAng.get(idx * 3);
        final float ay = this.snapshotAng.get(idx * 3 + 1);
        final float az = this.snapshotAng.get(idx * 3 + 2);
        final boolean heavy = sb.leafCount >= HEAVY_BODY_CHILDREN;
        // Poison watchdog: non-finite state (NaN/Inf position, rotation or
        // velocity) makes the NEXT native update read garbage and die with
        // EXCEPTION_ACCESS_VIOLATION inside joltjni.dll. Quarantine the body
        // instead: restore the last finite snapshot, zero the velocities
        // and put it to sleep so the solver never sees the poison. The
        // cached fields keep serving the last good values downstream.
        if (!(Double.isFinite(px) && Double.isFinite(py) && Double.isFinite(pz)
                && Double.isFinite(cx) && Double.isFinite(cy) && Double.isFinite(cz)
                && Float.isFinite(qx) && Float.isFinite(qy) && Float.isFinite(qz) && Float.isFinite(qw)
                && Float.isFinite(vx) && Float.isFinite(vy) && Float.isFinite(vz)
                && Float.isFinite(ax) && Float.isFinite(ay) && Float.isFinite(az))) {
            this.quarantinePoisonedBody(sb);
            sb.snapVx = sb.snapVy = sb.snapVz = 0.0f;
            sb.snapAx = sb.snapAy = sb.snapAz = 0.0f;
            sb.snapActive = false;
            return heavy;
        }
        sb.snapX = px;
        sb.snapY = py;
        sb.snapZ = pz;
        sb.snapComX = cx;
        sb.snapComY = cy;
        sb.snapComZ = cz;
        sb.snapQx = qx;
        sb.snapQy = qy;
        sb.snapQz = qz;
        sb.snapQw = qw;
        // Tunneling guard for huge bodies (no CCD) in Jolt mode: cap speed
        // below one block per tick. Rapier-compatible mode keeps every speed.
        if (heavy && !this.rapierMode && sb.kind == SableBody.Kind.SUB_LEVEL) {
            final float speedSq = vx * vx + vy * vy + vz * vz;
            if (speedSq > 144.0f) {
                final float scale = 12.0f / (float) Math.sqrt(speedSq);
                vx *= scale;
                vy *= scale;
                vz *= scale;
                this.bi.setLinearVelocity(sb.joltId, vx, vy, vz);
            }
        }
        sb.snapVx = vx;
        sb.snapVy = vy;
        sb.snapVz = vz;
        sb.snapAx = ax;
        sb.snapAy = ay;
        sb.snapAz = az;
        return heavy && sb.snapActive;
    }

    private long lastPoisonLogNanos;

    /**
     * Quarantines a body whose native state read back non-finite: restores the
     * last finite pose (always valid: initialized at creation and teleport),
     * zeroes velocities and deactivates it so the next {@code system.update}
     * never ingests the poison. Rate-limited logging.
     */
    private void quarantinePoisonedBody(final SableBody sb) {
        final long now = System.nanoTime();
        if (now - this.lastPoisonLogNanos > 1_000_000_000L) {
            this.lastPoisonLogNanos = now;
            Sable.LOGGER.error("[SableJolt] body {} (jolt {}) read back non-finite state; quarantined at last good pose",
                    sb.runtimeId, sb.joltId);
        }
        try {
            this.bi.setPositionAndRotation(sb.joltId, sb.snapX, sb.snapY, sb.snapZ,
                    sb.snapQx, sb.snapQy, sb.snapQz, sb.snapQw, EActivation.DontActivate);
            this.bi.setLinearAndAngularVelocity(sb.joltId, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            this.bi.deactivateBody(sb.joltId);
        } catch (final Throwable ignored) {
            // best effort: the body is already unusable
        }
    }

    private void ensureSnapshotBuffers(final int n) {
        if (this.snapshotPos != null && this.snapshotPos.capacity() >= n * 3) {
            return;
        }
        final int cap = Math.max(64, n);
        this.snapshotPos = ByteBuffer.allocateDirect(cap * 3 * 8).order(ByteOrder.nativeOrder()).asDoubleBuffer();
        this.snapshotRot = ByteBuffer.allocateDirect(cap * 4 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        this.snapshotLin = ByteBuffer.allocateDirect(cap * 3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        this.snapshotAng = ByteBuffer.allocateDirect(cap * 3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        this.snapshotActive = ByteBuffer.allocateDirect(cap).order(ByteOrder.nativeOrder());
        this.snapshotUser = ByteBuffer.allocateDirect(cap * 8).order(ByteOrder.nativeOrder()).asLongBuffer();
    }

    /**
     * Splits per-body work into slices for the worker pool. The calling thread
     * processes the first slice and the method returns after all slices finish.
     */
    private void runParallel(final List<SableBody> bodies, final java.util.function.Consumer<SableBody> action) {
        runParallelIndices(bodies.size(), i -> action.accept(bodies.get(i)));
    }

    /**
     * Runs {@code action} for every index in [0, n) on the shared worker pool;
     * the calling thread takes the first slice. Returns after all finished.
     */
    private static void runParallelIndices(final int n, final IntConsumer action) {
        final ExecutorService pool = sharedWorkers();
        final int workers = Math.min(sharedWorkerCount + 1, n);
        if (workers <= 1) {
            for (int i = 0; i < n; i++) {
                action.accept(i);
            }
            return;
        }

        // Slices are collected first so the latch matches the number of tasks
        // actually submitted; otherwise an empty trailing slice would leave the
        // caller waiting on counts that never happen (permanent server stall).
        final int sliceSize = (n + workers - 1) / workers;
        final IntArrayList starts = new IntArrayList(workers);
        for (int w = 1; w < workers; w++) {
            final int from = w * sliceSize;
            if (from >= n) {
                break;
            }
            starts.add(from);
        }

        final CountDownLatch latch = new CountDownLatch(starts.size());
        for (int s = 0; s < starts.size(); s++) {
            final int from = starts.getInt(s);
            final int to = Math.min(n, from + sliceSize);
            pool.execute(() -> {
                try {
                    for (int i = from; i < to; i++) {
                        action.accept(i);
                    }
                } catch (final Throwable t) {
                    Sable.LOGGER.error("[SableJolt] parallel task failed", t);
                } finally {
                    latch.countDown();
                }
            });
        }
        final int headEnd = Math.min(sliceSize, n);
        for (int i = 0; i < headEnd; i++) {
            action.accept(i);
        }
        boolean interrupted = false;
        while (true) {
            try {
                latch.await();
                break;
            } catch (final InterruptedException e) {
                // Never return before the workers are done: they still touch
                // shared state.
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Whether the reduced solver iteration depth is currently applied (heavy-body
     * scenes only; restored when the heavy body goes away).
     */
    private boolean reducedSolverApplied;

    private long lastWorldDump;

    private void staffWorldDump() {
        final long now = System.currentTimeMillis();
        if (now - this.lastWorldDump < 2000) {
            return;
        }
        this.lastWorldDump = now;

        final StringBuilder sb = new StringBuilder("[SableJolt:staff] world: globalChunks=").append(this.globalChunks.size())
                .append(" fluidSections=").append(this.fluidSections.size())
                .append(" allChunks=").append(this.allChunks.size())
                .append(" bodies=").append(this.bodies.size());
        for (final SableBody body : this.bodyList) {
            sb.append(" | #").append(body.runtimeId).append("/").append(body.kind)
                    .append(" pos=(").append((float) body.snapComX).append(",").append((float) body.snapComY).append(",").append((float) body.snapComZ).append(")")
                    .append(" leaves=").append(body.leafCount)
                    .append(" bounds=(").append(body.minX).append(",").append(body.minY).append(",").append(body.minZ)
                    .append(")..(").append(body.maxX).append(",").append(body.maxY).append(",").append(body.maxZ).append(")")
                    .append(" com=(").append((float) body.centerOfMass.x).append(",").append((float) body.centerOfMass.y).append(",").append((float) body.centerOfMass.z).append(")")
                    .append(" active=").append(body.body.isActive());
        }
        Sable.LOGGER.info(sb.toString());
    }

    /**
     * Releases everything. Safe to call more than once; afterwards every
     * operation is a no-op (handles held by other mods may still call in).
     */
    public void dispose() {
        if (this.disposed) {
            return;
        }
        this.disposed = true;
        try {
            this.system.setContactListener(null);
            this.system.removeAllConstraints();
            this.system.destroyAllBodies();
        } catch (final Throwable t) {
            Sable.LOGGER.error("[SableJolt] failed to clear the physics world", t);
        }
        this.bodies.clear();
        this.bodiesByJoltId.clear();
        this.bodyList.clear();
        this.contraptionList.clear();
        this.pendingCreation.clear();
        this.globalChunks.clear();
        this.globalChunksByBodyId.clear();
        this.allChunks.clear();
        this.sectionOwners.clear();
        this.joints.clear();
        this.ropes.clear();
        synchronized (this.dirtyBodies) {
            this.dirtyBodies.clear();
        }
        this.dirtyGlobalChunks.clear();
        this.pendingTerrainBodies.clear();
        this.fluidSections.clear();
        this.angularFollowPrev.clear();
        this.angularCachedOmega.clear();
        this.creativeDragBodies.clear();
        this.creativeRestorePending.clear();
        this.ropePointBodies.clear();
        this.callbackResults = new Long2ObjectOpenHashMap<>();
        synchronized (this.pendingCallbacksLock) {
            this.pendingCallbacks.clear();
        }
        synchronized (this.reportedLock) {
            this.reportedCollisions.clear();
        }
        // The native listener holds a global reference to its Java object; it
        // must be freed explicitly or the whole scene stays reachable forever.
        try {
            this.system.close();
            this.contactListener.close();
            this.tempAllocator.close();
            this.singleThreadJobSystem.close();
        } catch (final Throwable t) {
            Sable.LOGGER.error("[SableJolt] failed to release native physics objects", t);
        }
    }

    public boolean isDisposed() {
        return this.disposed;
    }

    public int nextRuntimeId() {
        return this.nextRuntimeId.getAndIncrement();
    }

    //endregion

    //region Buoyancy

    /**
     * Fluid lookup with a one-section cache (per call site).
     */
    private final class FluidProbe {
        private long cachedKey = Long.MIN_VALUE;
        @Nullable
        private ChunkSectionData cached;

        /**
         * Returns the fluid level of a world block: 0 = none, 1..8 = level,
         * 9 = full (source / level not uploaded).
         */
        int fluidAt(final int x, final int y, final int z) {
            final long key = ChunkSectionData.packSectionPos(x >> 4, y >> 4, z >> 4);
            if (key != this.cachedKey) {
                this.cachedKey = key;
                if (!JoltPhysicsScene.this.fluidSections.contains(key)) {
                    this.cached = null;
                } else {
                    final GlobalChunk chunk = JoltPhysicsScene.this.globalChunks.get(key);
                    this.cached = chunk == null ? null : chunk.data;
                }
            }
            final ChunkSectionData data = this.cached;
            if (data == null) {
                return 0;
            }
            final int packed = data.get(x & 15, y & 15, z & 15);
            final int colliderId = ChunkSectionData.colliderIdOf(packed);
            if (colliderId == 0) {
                return 0;
            }
            final JoltVoxelColliderData entry = JoltPhysicsScene.this.colliderRegistry.get(colliderId - 1);
            if (entry == null || !entry.isFluid) {
                return 0;
            }
            final int level = ChunkSectionData.fluidLevelOf(packed);
            return level > 0 ? level : 9;
        }

        /**
         * Height of the fluid surface inside a world block (0 = no fluid): a
         * fluid block under another fluid block is full, otherwise its level
         * sets the surface (a source block is 8/9 high, as in Minecraft).
         */
        double fluidHeight(final int x, final int y, final int z) {
            final int level = this.fluidAt(x, y, z);
            if (level == 0) {
                return 0.0;
            }
            if (this.fluidAt(x, y + 1, z) != 0) {
                return 1.0;
            }
            return level >= 9 ? 1.0 : level / 9.0;
        }
    }

    private boolean isAwake(final SableBody sb) {
        return sb.snapValid ? sb.snapActive : sb.body.isActive();
    }

    private ObjectArrayList<SableBody> collectBuoyancyCandidates() {
        final ObjectArrayList<SableBody> active = new ObjectArrayList<>();
        for (final SableBody sb : this.bodyList) {
            if (sb.kind == SableBody.Kind.SUB_LEVEL && sb.buoyancyCount > 0
                    && !this.creativeDragBodies.contains(sb.joltId) && this.isAwake(sb)) {
                active.add(sb);
            }
        }
        return active;
    }

    /**
     * Rapier buoyancy (buoyancy.rs), once per game tick: every solid block of a
     * sub-level is an axis-aligned cube (eight quarter-size samples for small
     * bodies) whose overlap with world fluid blocks yields a constant upward
     * force (10.5 per submerged volume, scaled by the block's volume property)
     * and a drag of -1.7 * v * volume, both at the sample point.
     */
    private void computeTickBuoyancy() {
        for (final SableBody sb : this.bodyList) {
            sb.hasTickBuoyancy = false;
        }
        if (this.fluidSections.isEmpty()) {
            return;
        }
        final ObjectArrayList<SableBody> active = this.collectBuoyancyCandidates();
        if (active.isEmpty()) {
            return;
        }
        if (active.size() >= 12) {
            this.runParallel(active, this::rapierBuoyancyForBody);
        } else {
            for (final SableBody sb : active) {
                this.rapierBuoyancyForBody(sb);
            }
        }
    }

    private void applyTickBuoyancy() {
        for (final SableBody sb : this.bodyList) {
            // Forces on a sleeping body would accumulate until it wakes up.
            if (!sb.hasTickBuoyancy || sb.body == null || this.creativeDragBodies.contains(sb.joltId) || !this.isAwake(sb)) {
                continue;
            }
            sb.body.addForce((float) sb.tickFx, (float) sb.tickFy, (float) sb.tickFz);
            sb.body.addTorque((float) sb.tickTx, (float) sb.tickTy, (float) sb.tickTz);
        }
    }

    private void rapierBuoyancyForBody(final SableBody sb) {
        final boolean fresh = sb.snapValid;
        final double comX, comY, comZ;
        final float qx, qy, qz, qw;
        final double lvx, lvy, lvz, avx, avy, avz;
        if (fresh) {
            comX = sb.snapComX;
            comY = sb.snapComY;
            comZ = sb.snapComZ;
            qx = sb.snapQx;
            qy = sb.snapQy;
            qz = sb.snapQz;
            qw = sb.snapQw;
            lvx = sb.snapVx;
            lvy = sb.snapVy;
            lvz = sb.snapVz;
            avx = sb.snapAx;
            avy = sb.snapAy;
            avz = sb.snapAz;
        } else {
            final Body body = sb.body;
            final RVec3 comPos = body.getCenterOfMassPosition();
            comX = comPos.xx();
            comY = comPos.yy();
            comZ = comPos.zz();
            final Quat rot = body.getRotation();
            qx = rot.getX();
            qy = rot.getY();
            qz = rot.getZ();
            qw = rot.getW();
            final Vec3 lin = body.getLinearVelocity();
            final Vec3 ang = body.getAngularVelocity();
            lvx = lin.getX();
            lvy = lin.getY();
            lvz = lin.getZ();
            avx = ang.getX();
            avy = ang.getY();
            avz = ang.getZ();
        }

        // "complex" (small) bodies use eight quarter-size samples per block.
        final boolean complex = (sb.maxX - sb.minX) + (sb.maxY - sb.minY) + (sb.maxZ - sb.minZ) < 10;
        final double half = complex ? 0.25 : 0.5;
        final int samples = complex ? 8 : 1;

        final FluidProbe probe = new FluidProbe();
        double fx = 0.0, fy = 0.0, fz = 0.0;
        double tx = 0.0, ty = 0.0, tz = 0.0;
        boolean any = false;

        final double baseX = sb.centerOfMass.x;
        final double baseY = sb.centerOfMass.y;
        final double baseZ = sb.centerOfMass.z;
        final int[] blocks = sb.buoyancyBlocks;
        final float[] volumes = sb.buoyancyVolumes;
        final int count = sb.buoyancyCount;
        for (int b = 0; b < count; b++) {
            final double cx = blocks[b * 3] + 0.5 - baseX;
            final double cy = blocks[b * 3 + 1] + 0.5 - baseY;
            final double cz = blocks[b * 3 + 2] + 0.5 - baseZ;
            final double floatStrength = volumes[b];
            for (int s = 0; s < samples; s++) {
                double lx = cx, ly = cy, lz = cz;
                if (complex) {
                    lx += ((s & 1) * 2 - 1) * 0.25;
                    ly += (((s >> 1) & 1) * 2 - 1) * 0.25;
                    lz += (((s >> 2) & 1) * 2 - 1) * 0.25;
                }
                // world sample point = com + R * local
                final double tX = 2.0 * (qy * lz - qz * ly);
                final double tY = 2.0 * (qz * lx - qx * lz);
                final double tZ = 2.0 * (qx * ly - qy * lx);
                final double rx = lx + qw * tX + (qy * tZ - qz * tY);
                final double ry = ly + qw * tY + (qz * tX - qx * tZ);
                final double rz = lz + qw * tZ + (qx * tY - qy * tX);
                final double px = comX + rx;
                final double py = comY + ry;
                final double pz = comZ + rz;

                final int x0 = floor(px - half), x1 = floor(px + half);
                final int y0 = floor(py - half), y1 = floor(py + half);
                final int z0 = floor(pz - half), z1 = floor(pz + half);
                double submerged = 0.0;
                for (int wx = x0; wx <= x1; wx++) {
                    final double ox = Math.min(px + half, wx + 1.0) - Math.max(px - half, wx);
                    if (ox <= 0.0) {
                        continue;
                    }
                    for (int wy = y0; wy <= y1; wy++) {
                        final double oy = Math.min(py + half, wy + 1.0) - Math.max(py - half, wy);
                        if (oy <= 0.0) {
                            continue;
                        }
                        for (int wz = z0; wz <= z1; wz++) {
                            final double oz = Math.min(pz + half, wz + 1.0) - Math.max(pz - half, wz);
                            if (oz <= 0.0) {
                                continue;
                            }
                            if (probe.fluidAt(wx, wy, wz) != 0) {
                                submerged += ox * oy * oz;
                            }
                        }
                    }
                }
                if (submerged <= 0.0) {
                    continue;
                }
                any = true;
                // velocity at the sample point
                final double vx = lvx + avy * rz - avz * ry;
                final double vy = lvy + avz * rx - avx * rz;
                final double vz = lvz + avx * ry - avy * rx;
                final double sfx = -vx * FLUID_DRAG * submerged;
                final double sfy = -vy * FLUID_DRAG * submerged + RAPIER_FLOAT_FORCE * submerged * floatStrength;
                final double sfz = -vz * FLUID_DRAG * submerged;
                fx += sfx;
                fy += sfy;
                fz += sfz;
                tx += ry * sfz - rz * sfy;
                ty += rz * sfx - rx * sfz;
                tz += rx * sfy - ry * sfx;
            }
        }

        if (any && Double.isFinite(fx) && Double.isFinite(fy) && Double.isFinite(fz)
                && Double.isFinite(tx) && Double.isFinite(ty) && Double.isFinite(tz)) {
            sb.tickFx = fx;
            sb.tickFy = fy;
            sb.tickFz = fz;
            sb.tickTx = tx;
            sb.tickTy = ty;
            sb.tickTz = tz;
            sb.hasTickBuoyancy = true;
        }
    }

    private void computeBuoyancy() {
        if (this.fluidSections.isEmpty()) {
            return;
        }
        final ObjectArrayList<SableBody> active = this.collectBuoyancyCandidates();
        if (active.isEmpty()) {
            return;
        }

        if (active.size() >= 12) {
            this.runParallel(active, this::buoyancyForBody);
        } else {
            for (final SableBody sb : active) {
                this.buoyancyForBody(sb);
            }
        }
    }

    /**
     * Jolt buoyancy model: per block, the submerged height of the rotated cube
     * against the fluid surface (fluid levels respected), drag at the submerged
     * midpoint and one gravity-proportional float force at the submerged
     * centroid. Very large bodies are subsampled with volume compensation.
     */
    private void buoyancyForBody(final SableBody sb) {
        final Body body = sb.body;

        // Reads come from the batch snapshot when fresh (one tick stale is
        // irrelevant for bulk fluid forces) and fall back to direct JNI reads
        // otherwise.
        final boolean fresh = sb.snapValid;
        final double comX, comY, comZ;
        final float qx, qy, qz, qw;
        final float lvx, lvy, lvz, avx, avy, avz;
        if (fresh) {
            comX = sb.snapComX;
            comY = sb.snapComY;
            comZ = sb.snapComZ;
            qx = sb.snapQx;
            qy = sb.snapQy;
            qz = sb.snapQz;
            qw = sb.snapQw;
            lvx = sb.snapVx;
            lvy = sb.snapVy;
            lvz = sb.snapVz;
            avx = sb.snapAx;
            avy = sb.snapAy;
            avz = sb.snapAz;
        } else {
            final RVec3 comPos = body.getCenterOfMassPosition();
            comX = comPos.xx();
            comY = comPos.yy();
            comZ = comPos.zz();
            final Quat rot = body.getRotation();
            qx = rot.getX();
            qy = rot.getY();
            qz = rot.getZ();
            qw = rot.getW();
            final Vec3 lin = body.getLinearVelocity();
            final Vec3 ang = body.getAngularVelocity();
            lvx = lin.getX();
            lvy = lin.getY();
            lvz = lin.getZ();
            avx = ang.getX();
            avy = ang.getY();
            avz = ang.getZ();
        }

        // Rotated unit axes give the world-space Y extent of every child cube:
        // an extent that stays correct (and smooth) for any body orientation.
        Vector3f r = this.rotateInto(1.0f, 0.0f, 0.0f, qx, qy, qz, qw);
        final double xAxY = r.y;
        r = this.rotateInto(0.0f, 1.0f, 0.0f, qx, qy, qz, qw);
        final double yAxY = r.y;
        r = this.rotateInto(0.0f, 0.0f, 1.0f, qx, qy, qz, qw);
        final double halfY = 0.5 * (Math.abs(xAxY) + Math.abs(yAxY) + Math.abs(r.y));

        // Accumulated force + torque: summing F and (p - com) x F over all
        // submerged samples in doubles and applying once is exactly equivalent
        // to one JNI call per sample (linearity).
        double accFx = 0.0, accFy = 0.0, accFz = 0.0;
        double accTx = 0.0, accTy = 0.0, accTz = 0.0;

        // Accumulated submerged volume and its centroid: the float force is applied
        // at the centroid of the submerged part, so a tilted body gets a smooth
        // righting torque and settles level.
        double floatVolume = 0.0;
        double centroidX = 0.0, centroidY = 0.0, centroidZ = 0.0;

        final FluidProbe probe = new FluidProbe();

        final double comBaseX = sb.centerOfMass.x;
        final double comBaseY = sb.centerOfMass.y;
        final double comBaseZ = sb.centerOfMass.z;

        // One sample per unique solid block (multi-box blocks such as stairs
        // count once). Very large bodies are subsampled with volume
        // compensation: fluid forces are a bulk effect.
        final int[] blocks = sb.buoyancyBlocks;
        final float[] volumes = sb.buoyancyVolumes;
        final int blockCount = sb.buoyancyCount;
        final int stride = Math.max(1, blockCount / BUOYANCY_MAX_CHILDREN);
        for (int bi = 0; bi < blockCount; bi += stride) {
            final double sampleScale = stride;
            final float lpx = (float) (blocks[bi * 3] + 0.5 - comBaseX);
            final float lpy = (float) (blocks[bi * 3 + 1] + 0.5 - comBaseY);
            final float lpz = (float) (blocks[bi * 3 + 2] + 0.5 - comBaseZ);

            // Inline rotateInto: same math, no ThreadLocal.get / Vector3f per block.
            final float tX = 2.0f * (qy * lpz - qz * lpy);
            final float tY = 2.0f * (qz * lpx - qx * lpz);
            final float tZ = 2.0f * (qx * lpy - qy * lpx);
            final double wx = comX + lpx + qw * tX + (qy * tZ - qz * tY);
            final double wy = comY + lpy + qw * tY + (qz * tX - qx * tZ);
            final double wz = comZ + lpz + qw * tZ + (qx * tY - qy * tX);

            final int wbx = floor(wx);
            final int wby = floor(wy);
            final int wbz = floor(wz);
            // Fluid at the block's center, or right below it: a block floating
            // high still dips its lower half into the water underneath.
            int fluidY = wby;
            double fluidHeight = probe.fluidHeight(wbx, wby, wbz);
            if (fluidHeight <= 0.0) {
                fluidY = wby - 1;
                fluidHeight = probe.fluidHeight(wbx, fluidY, wbz);
                if (fluidHeight <= 0.0) {
                    continue;
                }
            }
            final float mult = volumes[bi];

            // Vertical span of the rotated unit cube vs the fluid surface: the
            // submerged height is smooth in rotation. For a centrally-symmetric
            // cube the horizontal centroid of the submerged part is always its
            // center — no spurious yaw torque.
            final double fluidTop = fluidY + fluidHeight;
            final double minY = wy - halfY;
            final double maxY = wy + halfY;
            final double submerged = Math.min(maxY, fluidTop) - minY;
            if (submerged <= 0.0) {
                continue;
            }
            final double volume = Math.min(1.0, submerged / (halfY * 2.0)) * mult * sampleScale;
            final double midY = Math.min(maxY, fluidTop) - submerged * 0.5;

            // drag: F = -v * 1.7 * volume at the submerged midpoint — damps both
            // linear motion and rotation while submerged.
            final double rx = wx - comX, ry = midY - comY, rz = wz - comZ;
            final double vx = lvx + avy * rz - avz * ry;
            final double vy = lvy + avz * rx - avx * rz;
            final double vz = lvz + avx * ry - avy * rx;
            final double fdx = -vx * FLUID_DRAG * volume;
            final double fdy = -vy * FLUID_DRAG * volume;
            final double fdz = -vz * FLUID_DRAG * volume;
            accFx += fdx;
            accFy += fdy;
            accFz += fdz;
            accTx += ry * fdz - rz * fdy;
            accTy += rz * fdx - rx * fdz;
            accTz += rx * fdy - ry * fdx;

            // accumulate submerged volume for the single centroid float force
            floatVolume += volume;
            centroidX += wx * volume;
            centroidY += midY * volume;
            centroidZ += wz * volume;
        }

        if (floatVolume > 0.0) {
            // Float applied at the centroid of the submerged volume: zero torque
            // when level, smooth righting torque when tilted — stable flotation.
            final double fby = -this.gravityY * FLUID_DENSITY * floatVolume;
            final double pbx = centroidX / floatVolume;
            final double pbz = centroidZ / floatVolume;
            accFy += fby;
            // Buoyancy force is vertical (0, fby, 0): r x F = (-rz*fby, 0, rx*fby).
            final double brx = pbx - comX, brz = pbz - comZ;
            accTx += -brz * fby;
            accTz += brx * fby;
        }

        if (!(Double.isFinite(accFx) && Double.isFinite(accFy) && Double.isFinite(accFz)
                && Double.isFinite(accTx) && Double.isFinite(accTy) && Double.isFinite(accTz))) {
            return;
        }
        if (accFx != 0.0 || accFy != 0.0 || accFz != 0.0) {
            body.addForce((float) accFx, (float) accFy, (float) accFz);
            if (accTx != 0.0 || accTy != 0.0 || accTz != 0.0) {
                body.addTorque((float) accTx, (float) accTy, (float) accTz);
            }
        }
    }

    private static int floor(final double v) {
        return (int) Math.floor(v);
    }
    //endregion

    //region Contraption motion

    // Reusable objects for updateContraptionMotion (server thread only).
    private final RVec3 tmpMovePos = new RVec3();
    private final Quat tmpMoveRot = new Quat();

    /**
     * Drives the kinematic bodies of world contraptions to their target pose.
     * Jolt derives the body velocity from the motion, so contacts see the
     * contraption's real velocity. Mounted contraptions move with their mount's
     * compound instead (see {@link #assembleParent}).
     */
    private void updateContraptionMotion(final float dt) {
        for (final SableBody sb : this.contraptionList) {
            if (sb.body == null) {
                continue;
            }
            this.tmpMovePos.set(sb.relPos.x, sb.relPos.y, sb.relPos.z);
            this.tmpMoveRot.set((float) sb.relRot.x, (float) sb.relRot.y, (float) sb.relRot.z, (float) sb.relRot.w);
            this.bi.moveKinematic(sb.joltId, this.tmpMovePos, this.tmpMoveRot, dt);
        }
    }

    //endregion

    //region Collision reporting

    /**
     * Reused output buffer of {@link #clearCollisions} (server thread only).
     */
    private final double[] collisionOut = new double[100 * 15];
    private int collisionCount;

    /**
     * Reads & clears all reported collisions.
     * Each collision is formatted as:
     * [body_a, body_b, force_amount, local_normal_a, local_normal_b, local_point_a, local_point_b]
     */
    public double[] clearCollisions() {
        final int max = 100;
        synchronized (this.reportedLock) {
            if (this.reportedCollisions.size() > max) {
                this.reportedCollisions.subList(max, this.reportedCollisions.size()).clear();
            }
            this.collisionCount = this.reportedCollisions.size();
            int i = 0;
            for (final double[] rec : this.reportedCollisions) {
                System.arraycopy(rec, 0, this.collisionOut, i, 15);
                i += 15;
            }
            this.reportedCollisions.clear();
        }
        return this.collisionOut;
    }

    /**
     * Number of valid 15-double records in the last {@link #clearCollisions} result.
     */
    public int lastCollisionCount() {
        return this.collisionCount;
    }

    void reportCollision(final double[] rec) {
        synchronized (this.reportedLock) {
            if (this.reportedCollisions.size() < 100) {
                this.reportedCollisions.add(rec);
            }
        }
    }

    JoltVoxelColliderData.Registry registry() {
        return this.colliderRegistry;
    }

    GlobalChunk globalChunkByBodyId(final int joltId) {
        return this.globalChunksByBodyId.get(joltId);
    }

    //endregion

    //region Block contact callbacks

    /**
     * A block contact callback recorded during the step; invoked afterwards on
     * the server thread (callbacks read and edit the world, which must never
     * happen on a Jolt worker thread or inside the physics update).
     */
    private record PendingCallback(JoltVoxelColliderData entry, long blockKey, int x, int y, int z,
                                   int otherX, int otherY, int otherZ, boolean hasOther,
                                   double hitX, double hitY, double hitZ, double impactVelocity) {
    }

    /**
     * Callback result: tangent motion in the block's frame and whether the
     * contact is removed. Applied to the block's contacts in the next step.
     */
    private record CallbackResult(float tanX, float tanY, float tanZ, boolean remove) {
    }

    private final Object pendingCallbacksLock = new Object();
    private ObjectArrayList<PendingCallback> pendingCallbacks = new ObjectArrayList<>();
    /**
     * Results of the previous step's callbacks, keyed by block position (plot
     * and world coordinates never overlap). Replaced wholesale between steps;
     * read by contact callbacks during the update.
     */
    private volatile Long2ObjectOpenHashMap<CallbackResult> callbackResults = new Long2ObjectOpenHashMap<>();

    private void queueCallback(final PendingCallback callback) {
        synchronized (this.pendingCallbacksLock) {
            if (this.pendingCallbacks.size() < 4096) {
                this.pendingCallbacks.add(callback);
            }
        }
    }

    private long lastCallbackErrorNanos;

    /**
     * Invokes the block contact callbacks recorded during the last step. Must be
     * called on the server thread between steps (still inside Sable's physics
     * tick, as the callbacks expect).
     */
    public void processDeferredCallbacks() {
        final ObjectArrayList<PendingCallback> list;
        synchronized (this.pendingCallbacksLock) {
            if (this.pendingCallbacks.isEmpty()) {
                if (!this.callbackResults.isEmpty()) {
                    this.callbackResults = new Long2ObjectOpenHashMap<>();
                }
                return;
            }
            list = this.pendingCallbacks;
            this.pendingCallbacks = new ObjectArrayList<>();
        }
        final Long2ObjectOpenHashMap<CallbackResult> results = new Long2ObjectOpenHashMap<>();
        final LongOpenHashSet seen = new LongOpenHashSet();
        for (final PendingCallback p : list) {
            if (this.disposed) {
                return;
            }
            if (!seen.add(p.blockKey())) {
                continue; // one invocation per block per step
            }
            try {
                final double[] result = p.entry().contactEvents.onCollision(
                        p.x(), p.y(), p.z(), p.otherX(), p.otherY(), p.otherZ(),
                        p.hitX(), p.hitY(), p.hitZ(), p.impactVelocity(), p.hasOther());
                if (result != null && result.length >= 4) {
                    final boolean remove = result[3] > 0.0;
                    if (remove || result[0] != 0.0 || result[1] != 0.0 || result[2] != 0.0) {
                        results.put(p.blockKey(), new CallbackResult((float) result[0], (float) result[1], (float) result[2], remove));
                    }
                }
            } catch (final Throwable t) {
                final long now = System.nanoTime();
                if (now - this.lastCallbackErrorNanos > 5_000_000_000L) {
                    this.lastCallbackErrorNanos = now;
                    Sable.LOGGER.error("Block contact callback failed", t);
                }
            }
        }
        this.callbackResults = results;
    }

    private static long blockKey(final int x, final int y, final int z) {
        return (((long) x & 0x3FFFFFFL) << 38) | (((long) z & 0x3FFFFFFL) << 12) | ((long) y & 0xFFFL);
    }

    //endregion

    //region Constraints

    /**
     * Whether a constraint side can be built: the static world (no body), or a
     * body with a live native body. Native constraint creation with a missing
     * body would crash.
     */
    public boolean canConstrain(@Nullable final dev.ryanhcode.sable.api.physics.PhysicsPipelineBody body, @Nullable final SableBody sb) {
        if (this.disposed) {
            return false;
        }
        return body == null || (sb != null && sb.body != null && sb.joltId != 0);
    }

    public TwoBodyConstraint createConstraint(final TwoBodyConstraintSettings settings, final int joltIdA, final int joltIdB) {
        final TwoBodyConstraint constraint = this.bi.createConstraint(settings, joltIdA, joltIdB);
        this.system.addConstraint(constraint);
        return constraint;
    }

    public void removeConstraint(final Constraint constraint) {
        if (constraint != null && !this.disposed) {
            this.system.removeConstraint(constraint);
        }
    }

    public long registerJoint(final JointRecord record) {
        record.id = this.nextJointId.getAndIncrement();
        this.joints.put(record.id, record);
        this.countAttachment(record.joltIdA, 1);
        this.countAttachment(record.joltIdB, 1);
        return record.id;
    }

    @Nullable
    public JointRecord joint(final long id) {
        return this.joints.get(id);
    }

    public void removeJoint(final long id) {
        final JointRecord rec = this.joints.remove(id);
        if (rec == null) {
            return;
        }
        this.countAttachment(rec.joltIdA, -1);
        this.countAttachment(rec.joltIdB, -1);
        if (rec.constraint != null) {
            this.removeConstraint(rec.constraint);
            rec.constraint = null;
        }
    }

    private void countAttachment(final int joltId, final int delta) {
        final SableBody sb = this.bodiesByJoltId.get(joltId);
        if (sb != null) {
            sb.attachmentCount = Math.max(0, sb.attachmentCount + delta);
        }
    }

    /**
     * Keeps joints anchored at the same block-space point when a body's center
     * of mass moves (Rapier re-derives anchors from the current center of mass
     * every tick; Jolt stores them relative to it).
     */
    private void notifyComShift(final int joltId, final float dx, final float dy, final float dz) {
        final Vec3 delta = this.tmpVec.get();
        delta.set(dx, dy, dz);
        for (final JointRecord rec : this.joints.values()) {
            if (rec.constraint == null || (rec.joltIdA != joltId && rec.joltIdB != joltId)) {
                continue;
            }
            rec.constraint.notifyShapeChanged(joltId, delta);
            // The servo math of six-DOF handles reads the anchors from the settings.
            if (rec.genericSettings instanceof final SixDofConstraintSettings settings) {
                if (rec.joltIdA == joltId) {
                    final RVec3 p = settings.getPosition1();
                    settings.setPosition1(new RVec3(p.xx() - dx, p.yy() - dy, p.zz() - dz));
                }
                if (rec.joltIdB == joltId) {
                    final RVec3 p = settings.getPosition2();
                    settings.setPosition2(new RVec3(p.xx() - dx, p.yy() - dy, p.zz() - dz));
                }
            }
        }
        for (final RopeStrand strand : this.ropes.values()) {
            if (strand.start != null && strand.start.mountJoltId == joltId && strand.start.constraint != null) {
                strand.start.constraint.notifyShapeChanged(joltId, delta);
            }
            if (strand.end != null && strand.end.mountJoltId == joltId && strand.end.constraint != null) {
                strand.end.constraint.notifyShapeChanged(joltId, delta);
            }
        }
    }

    //endregion

    //region Ropes

    private int createRopePoint(final double x, final double y, final double z, final double pointRadius) {
        final float radius = (float) Math.max(pointRadius, 0.01);
        final BodyCreationSettings bcs = new BodyCreationSettings(
                this.sharedBoxShape(radius, radius, radius),
                new RVec3(x, y, z), Quat.sIdentity(), EMotionType.Dynamic, LAYER_ROPE);
        bcs.setLinearDamping((float) (this.universalDrag + 6.0));
        bcs.setAngularDamping((float) this.universalDrag);
        bcs.setInertiaMultiplier(0.0f);
        // Rapier averages frictions; Jolt multiplies them under a square root. 0.214
        // reproduces Rapier's rope (0.15) contacts with sub-levels and terrain.
        bcs.setFriction(this.rapierMode ? 0.214f : FRICTION_ROPE);
        final MassProperties mp = new MassProperties();
        mp.setMass(0.35f);
        bcs.setOverrideMassProperties(EOverrideMassProperties.CalculateInertia);
        bcs.setMassPropertiesOverride(mp);

        final Body body;
        try {
            body = this.bi.createBody(bcs);
        } catch (final IllegalStateException e) {
            this.warnCapacity("rope point was not created");
            return -1;
        }
        body.setUserData(-1L);
        this.bi.addBody(body, EActivation.Activate);
        this.ropePointBodies.put(body.getId(), body);
        return body.getId();
    }

    private final Int2ObjectOpenHashMap<Body> ropePointBodies = new Int2ObjectOpenHashMap<>();

    private void destroyRopePoint(final int bodyId) {
        if (this.ropePointBodies.remove(bodyId) == null) {
            return;
        }
        this.bi.removeBody(bodyId);
        this.bi.destroyBody(bodyId);
    }

    private DistanceConstraint createRopeJoint(final int joltIdA, final int joltIdB, final double length) {
        final DistanceConstraintSettings settings = new DistanceConstraintSettings();
        settings.setPoint1(RVec3.sZero());
        settings.setPoint2(RVec3.sZero());
        settings.setMinDistance(0.0f);
        settings.setMaxDistance((float) length);
        final SpringSettings spring = settings.getLimitsSpringSettings();
        spring.setMode(ESpringMode.FrequencyAndDamping);
        spring.setFrequency(30.0f);
        spring.setDamping(1.0f);
        return (DistanceConstraint) this.createConstraint(settings, joltIdA, joltIdB);
    }

    public long createRope(final double pointRadius, final double firstJointLength, final double[] points, final int pointCount) {
        final long id = this.nextRopeId.getAndIncrement();
        if (this.disposed) {
            return id;
        }
        final RopeStrand strand = new RopeStrand();
        strand.pointRadius = pointRadius;
        strand.firstJointLength = firstJointLength;

        for (int i = 0; i < pointCount; i++) {
            final int point = this.createRopePoint(points[i * 3], points[i * 3 + 1], points[i * 3 + 2], pointRadius);
            if (point != -1) {
                strand.points.add(point);
            }
        }
        for (int i = 0; i < strand.points.size() - 1; i++) {
            final double length = i == 0 ? firstJointLength : 1.0;
            strand.joints.add(this.createRopeJoint(strand.points.getInt(i), strand.points.getInt(i + 1), length));
        }

        this.ropes.put(id, strand);
        return id;
    }

    public void removeRope(final long ropeId) {
        final RopeStrand strand = this.ropes.remove(ropeId);
        if (strand == null) {
            return;
        }
        if (strand.start != null) {
            this.removeRopeAttachment(strand.start);
        }
        if (strand.end != null) {
            this.removeRopeAttachment(strand.end);
        }
        for (final DistanceConstraint joint : strand.joints) {
            this.removeConstraint(joint);
        }
        for (final int point : strand.points) {
            this.destroyRopePoint(point);
        }
    }

    private void removeRopeAttachment(final RopeAttachment attachment) {
        this.removeConstraint(attachment.constraint);
        attachment.constraint = null;
        this.countAttachment(attachment.mountJoltId, -1);
    }

    public double[] queryRope(final long ropeId) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null) {
            return new double[0];
        }
        final double[] out = new double[strand.points.size() * 3];
        final RVec3 p = this.tmpCom.get();
        for (int i = 0; i < strand.points.size(); i++) {
            final Body body = this.ropePointBodies.get(strand.points.getInt(i));
            if (body == null) {
                continue;
            }
            body.getPosition(p);
            out[i * 3] = p.xx();
            out[i * 3 + 1] = p.yy();
            out[i * 3 + 2] = p.zz();
        }
        return out;
    }

    public void setRopeFirstSegmentLength(final long ropeId, final double length) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null || strand.joints.isEmpty()) {
            return;
        }
        strand.firstJointLength = length;
        strand.joints.get(0).setDistance(0.0f, (float) length);
    }

    public void addRopePointAtStart(final long ropeId, final double x, final double y, final double z) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null) {
            return;
        }
        final int bodyId = this.createRopePoint(x, y, z, strand.pointRadius);
        if (bodyId == -1) {
            return;
        }
        if (!strand.joints.isEmpty()) {
            strand.joints.get(0).setDistance(0.0f, 1.0f);
        }
        if (!strand.points.isEmpty()) {
            strand.joints.add(0, this.createRopeJoint(bodyId, strand.points.getInt(0), strand.firstJointLength));
        }
        strand.points.add(0, bodyId);

        if (strand.start != null) {
            this.removeRopeAttachment(strand.start);
            strand.start = null;
        }
    }

    public void removeRopePointAtStart(final long ropeId) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null || strand.points.isEmpty()) {
            return;
        }
        if (strand.start != null) {
            this.removeRopeAttachment(strand.start);
            strand.start = null;
        }
        final int point = strand.points.removeInt(0);
        if (!strand.joints.isEmpty()) {
            this.removeConstraint(strand.joints.remove(0));
        }
        this.destroyRopePoint(point);

        if (!strand.joints.isEmpty()) {
            strand.joints.get(0).setDistance(0.0f, (float) strand.firstJointLength);
        }
    }

    public void wakeUpRope(final long ropeId) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null) {
            return;
        }
        for (final int point : strand.points) {
            this.bi.activateBody(point);
        }
    }

    public void setRopeAttachment(final long ropeId, final int subLevelId, final double x, final double y, final double z, final boolean end) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null || strand.points.isEmpty()) {
            return;
        }

        final int ropeBody = end ? strand.points.getInt(strand.points.size() - 1) : strand.points.getInt(0);
        final SableBody mountSb = subLevelId != -1 ? this.bodies.get(subLevelId) : null;
        if (subLevelId != -1 && (mountSb == null || mountSb.body == null)) {
            return;
        }
        final int mountBody = mountSb != null ? mountSb.joltId : this.groundBodyId;

        // Block-space anchor relative to the mount's center of mass (Rapier:
        // pos - center_of_mass); later COM shifts are applied by notifyComShift.
        final double comX = mountSb != null ? mountSb.centerOfMass.x : 0.0;
        final double comY = mountSb != null ? mountSb.centerOfMass.y : 0.0;
        final double comZ = mountSb != null ? mountSb.centerOfMass.z : 0.0;

        final PointConstraintSettings settings = new PointConstraintSettings();
        settings.setSpace(com.github.stephengold.joltjni.enumerate.EConstraintSpace.LocalToBodyCom);
        settings.setPoint1(new RVec3(x - comX, y - comY, z - comZ));
        settings.setPoint2(RVec3.sZero());
        final Constraint constraint = this.createConstraint(settings, mountBody, ropeBody);

        final RopeAttachment attachment = new RopeAttachment();
        attachment.mountJoltId = mountBody;
        attachment.x = x;
        attachment.y = y;
        attachment.z = z;
        attachment.constraint = constraint;
        this.countAttachment(mountBody, 1);

        if (end) {
            if (strand.end != null) {
                this.removeRopeAttachment(strand.end);
            }
            strand.end = attachment;
        } else {
            if (strand.start != null) {
                this.removeRopeAttachment(strand.start);
            }
            strand.start = attachment;
        }
    }

    //endregion

    //region Config

    public void configSolverIterations(final int solverIterations, final int pgsIterations) {
        if (this.disposed) {
            return;
        }
        // clamped to the minimums that keep dragged bodies from sinking into walls
        this.configuredVelocitySteps = Math.max(10, solverIterations);
        this.configuredPositionSteps = Math.max(4, pgsIterations);
        final var settings = this.system.getPhysicsSettings();
        settings.setNumVelocitySteps(this.reducedSolverApplied ? Math.max(5, this.configuredVelocitySteps / 2) : this.configuredVelocitySteps);
        settings.setNumPositionSteps(this.reducedSolverApplied ? Math.max(2, this.configuredPositionSteps / 2) : this.configuredPositionSteps);
        this.system.setPhysicsSettings(settings);
    }

    //endregion

    private static int unpackChunkX(final long key) {
        return (int) (key >> 42);
    }

    private static int unpackChunkY(final long key) {
        return (int) ((key << 44) >> 44);
    }

    private static int unpackChunkZ(final long key) {
        return (int) ((key << 22) >> 42);
    }

    /**
     * Per-thread cache of native body addresses to Java wrappers for the contact
     * listener. A {@code Body} wrapper is stateless (every method re-reads through
     * the stored address) and the native address is stable for the body's lifetime,
     * so the cache is exact; each fresh wrapper otherwise costs 3 heap objects
     * per contact callback. Manifold/settings addresses are callback-scoped
     * temporaries and must NOT be cached.
     */
    private static final ThreadLocal<it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap<Body>> BODY_WRAPPER_CACHE =
            ThreadLocal.withInitial(it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap::new);

    private static Body wrapContactBody(final long bodyVa) {
        final var cache = BODY_WRAPPER_CACHE.get();
        Body wrapper = cache.get(bodyVa);
        if (wrapper == null) {
            wrapper = new Body(bodyVa);
            cache.put(bodyVa, wrapper);
            if (cache.size() > 4096) {
                // Bound entries churned by short-lived bodies (rope points);
                // repopulates lazily and stays correct (address-keyed).
                cache.clear();
            }
        }
        return wrapper;
    }

    private long lastListenerErrorNanos;

    private void logListenerFailure(final Throwable t) {
        final long now = System.nanoTime();
        if (now - this.lastListenerErrorNanos > 5_000_000_000L) {
            this.lastListenerErrorNanos = now;
            Sable.LOGGER.error("[SableJolt] contact listener failed", t);
        }
    }

    /**
     * The scene contact listener: reports collisions, records block contact
     * callbacks, applies friction / restitution overrides and the surface
     * velocities of mounted contraptions. Runs on Jolt worker threads: it only
     * reads scene state and never throws into native code.
     */
    private final class SceneContactListener extends FilteredContactListener {

        @Override
        public void onContactAdded(final long body1Va, final long body2Va, final long manifoldVa, final long settingsVa) {
            this.handle(body1Va, body2Va, manifoldVa, settingsVa);
        }

        @Override
        public void onContactPersisted(final long body1Va, final long body2Va, final long manifoldVa, final long settingsVa) {
            this.handle(body1Va, body2Va, manifoldVa, settingsVa);
        }

        private void handle(final long body1Va, final long body2Va, final long manifoldVa, final long settingsVa) {
            // A Java exception must never propagate into the native solver thread.
            try {
                this.handleUnchecked(body1Va, body2Va, manifoldVa, settingsVa);
            } catch (final Throwable t) {
                JoltPhysicsScene.this.logListenerFailure(t);
            }
        }

        private void handleUnchecked(final long body1Va, final long body2Va, final long manifoldVa, final long settingsVa) {
            final JoltPhysicsScene scene = JoltPhysicsScene.this;
            if (scene.disposed) {
                return;
            }
            // Bodies are long-lived: wrappers come from the cache (no allocation).
            // Manifold/settings are callback-scoped and must wrap fresh.
            final Body body1 = JoltPhysicsScene.wrapContactBody(body1Va);
            final Body body2 = JoltPhysicsScene.wrapContactBody(body2Va);
            final int id1 = body1.getId();
            final int id2 = body2.getId();

            final SableBody sb1 = scene.bodiesByJoltId.get(id1);
            final SableBody sb2 = scene.bodiesByJoltId.get(id2);
            final GlobalChunk gc1 = sb1 == null ? scene.globalChunksByBodyId.get(id1) : null;
            final GlobalChunk gc2 = sb2 == null ? scene.globalChunksByBodyId.get(id2) : null;

            if (sb1 == null && sb2 == null && gc1 == null && gc2 == null) {
                return;
            }

            final ContactManifold manifold = new ContactManifold(manifoldVa);
            final ContactSettings settings = new ContactSettings(settingsVa);

            // Deepest-penetration tracking for creatively dragged bodies: the
            // end-of-step push-out keeps the wand from grinding them into the
            // terrain faster than the solver can recover.
            if (sb1 != null && scene.creativeDragBodies.contains(sb1.joltId)) {
                this.accumulateDragPenetration(sb1, body1, manifold, true);
            }
            if (sb2 != null && scene.creativeDragBodies.contains(sb2.joltId)) {
                this.accumulateDragPenetration(sb2, body2, manifold, false);
            }

            final Child c1 = sb1 != null ? sb1.childAt(manifold.getSubShapeId1()) : gc1 != null ? gc1.childAt(manifold.getSubShapeId1()) : null;
            final Child c2 = sb2 != null ? sb2.childAt(manifold.getSubShapeId2()) : gc2 != null ? gc2.childAt(manifold.getSubShapeId2()) : null;

            final JoltVoxelColliderData entry1 = c1 != null ? scene.colliderRegistry.get(c1.colliderId - 1) : null;
            final JoltVoxelColliderData entry2 = c2 != null ? scene.colliderRegistry.get(c2.colliderId - 1) : null;

            // friction / restitution overrides (block-level)
            final float mult1 = entry1 == null ? 1.0f : entry1.frictionMultiplier;
            final float mult2 = entry2 == null ? 1.0f : entry2.frictionMultiplier;
            if (scene.rapierMode) {
                // Rapier: average of the collider frictions times both multipliers.
                final float f1 = frictionOf(sb1, gc1, c1);
                final float f2 = frictionOf(sb2, gc2, c2);
                settings.setCombinedFriction((f1 + f2) * 0.5f * mult1 * mult2);
            } else if (mult1 != 1.0f || mult2 != 1.0f) {
                settings.setCombinedFriction(settings.getCombinedFriction() * mult1 * mult2);
            }
            final float rest1 = entry1 == null ? 0.0f : entry1.restitution;
            final float rest2 = entry2 == null ? 0.0f : entry2.restitution;
            if (rest1 > 0.0f || rest2 > 0.0f) {
                settings.setCombinedRestitution(Math.max(rest1, rest2));
            }

            // Mounted contraption blocks move relative to their mount's body:
            // their motion becomes contact surface velocity (Rapier's fake velocities).
            this.applyContraptionSurfaceVelocity(c1, sb1, settings, true);
            this.applyContraptionSurfaceVelocity(c2, sb2, settings, false);

            final boolean callback1 = entry1 != null && entry1.contactEvents != null && !isContraptionBlock(c1);
            final boolean callback2 = entry2 != null && entry2.contactEvents != null && !isContraptionBlock(c2);
            final boolean report = c1 != null || c2 != null;
            if (!callback1 && !callback2 && !report) {
                return;
            }

            // Approximate contact point: the block of the callback side (or of
            // body 1) moved half a block towards the other body along the normal.
            final Vec3 n = manifold.getWorldSpaceNormal();
            final double nx = n.getX(), ny = n.getY(), nz = n.getZ();

            if (callback1) {
                this.recordCallback(entry1, c1, c2, sb1, gc2 != null, nx, ny, nz, sb1, sb2, true, settings);
            }
            if (callback2) {
                this.recordCallback(entry2, c2, c1, sb2, gc1 != null, -nx, -ny, -nz, sb1, sb2, false, settings);
            }

            if (report) {
                this.report(sb1, sb2, c1, c2, body1, body2, nx, ny, nz);
            }
        }

        private static boolean isContraptionBlock(@Nullable final Child c) {
            return c != null && c.owner != null && c.owner.kind == SableBody.Kind.CONTRAPTION;
        }

        private static float frictionOf(@Nullable final SableBody sb, @Nullable final GlobalChunk gc, @Nullable final Child c) {
            if (gc != null) {
                return FRICTION_TERRAIN_RAPIER;
            }
            if (sb == null) {
                return FRICTION_ROPE;
            }
            if (isContraptionBlock(c) || sb.kind == SableBody.Kind.CONTRAPTION) {
                return FRICTION_CONTRAPTION;
            }
            return sb.kind == SableBody.Kind.BOX ? FRICTION_BOX : FRICTION_SUB_LEVEL;
        }

        /**
         * Records the deepest contact penetration of a dragged body this step.
         * The manifold normal moves body 2 out of collision, so it is flipped
         * for body 1; the result is additionally verified against the
         * contact-to-COM direction so a wrong convention can never push the
         * body deeper.
         */
        private void accumulateDragPenetration(final SableBody dragged, final Body self,
                                               final ContactManifold manifold, final boolean isFirst) {
            final float depth = manifold.getPenetrationDepth();
            if (depth <= 0.0f || depth <= dragged.dragPenDepth) {
                return;
            }
            final Vec3 n = manifold.getWorldSpaceNormal();
            double nx = n.getX(), ny = n.getY(), nz = n.getZ();
            if (isFirst) {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }
            final RVec3 com = self.getCenterOfMassPosition();
            final RVec3 base = manifold.getBaseOffset();
            final double ox = com.xx() - base.xx(), oy = com.yy() - base.yy(), oz = com.zz() - base.zz();
            if (nx * ox + ny * oy + nz * oz < 0.0) {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }
            dragged.dragPenDepth = depth;
            dragged.dragPenNx = nx;
            dragged.dragPenNy = ny;
            dragged.dragPenNz = nz;
        }

        /**
         * World-space velocity of a body (snapshot of the last step) at a point.
         */
        private static void velocityAt(@Nullable final SableBody sb, final double px, final double py, final double pz,
                                       final double[] out, final int offset) {
            if (sb == null || sb.body == null) {
                out[offset] = out[offset + 1] = out[offset + 2] = 0.0;
                return;
            }
            final double rx = px - sb.snapComX, ry = py - sb.snapComY, rz = pz - sb.snapComZ;
            out[offset] = sb.snapVx + sb.snapAy * rz - sb.snapAz * ry;
            out[offset + 1] = sb.snapVy + sb.snapAz * rx - sb.snapAx * rz;
            out[offset + 2] = sb.snapVz + sb.snapAx * ry - sb.snapAy * rx;
        }

        private final ThreadLocal<double[]> velScratch = ThreadLocal.withInitial(() -> new double[6]);

        /**
         * Records a block contact callback for the server thread and applies the
         * result of the previous step's invocation for this block.
         *
         * @param dirX direction from the block towards the other body (world)
         */
        private void recordCallback(final JoltVoxelColliderData entry, final Child c, @Nullable final Child other,
                                    @Nullable final SableBody self, final boolean otherIsTerrain,
                                    final double dirX, final double dirY, final double dirZ,
                                    @Nullable final SableBody sb1, @Nullable final SableBody sb2,
                                    final boolean isFirst, final ContactSettings settings) {
            final JoltPhysicsScene scene = JoltPhysicsScene.this;
            final long key = blockKey(c.bx, c.by, c.bz);

            // Apply last step's result (tangent motion in the block's frame).
            final CallbackResult result = scene.callbackResults.get(key);
            if (result != null) {
                if (result.remove()) {
                    settings.setIsSensor(true);
                }
                if (result.tanX() != 0.0f || result.tanY() != 0.0f || result.tanZ() != 0.0f) {
                    float tx = result.tanX(), ty = result.tanY(), tz = result.tanZ();
                    if (self != null) {
                        final Vector3f r = scene.rotateInto(tx, ty, tz, self.snapQx, self.snapQy, self.snapQz, self.snapQw);
                        tx = r.x;
                        ty = r.y;
                        tz = r.z;
                    }
                    // remove the normal component (Rapier projects onto the tangent plane)
                    final float dot = (float) (tx * dirX + ty * dirY + tz * dirZ);
                    tx -= (float) (dirX * dot);
                    ty -= (float) (dirY * dot);
                    tz -= (float) (dirZ * dot);
                    final float sign = isFirst ? -1.0f : 1.0f;
                    final Vec3 existing = settings.getRelativeLinearSurfaceVelocity();
                    final Vec3 combined = scene.tmpVec.get();
                    combined.set(existing.getX() + sign * tx, existing.getY() + sign * ty, existing.getZ() + sign * tz);
                    settings.setRelativeLinearSurfaceVelocity(combined);
                }
            }

            // Contact point estimate in world space.
            double cxw, cyw, czw;
            if (self != null) {
                final float lx = (float) (c.bx + 0.5 - self.centerOfMass.x);
                final float ly = (float) (c.by + 0.5 - self.centerOfMass.y);
                final float lz = (float) (c.bz + 0.5 - self.centerOfMass.z);
                final Vector3f r = scene.rotateInto(lx, ly, lz, self.snapQx, self.snapQy, self.snapQz, self.snapQw);
                cxw = self.snapComX + r.x;
                cyw = self.snapComY + r.y;
                czw = self.snapComZ + r.z;
            } else {
                cxw = c.bx + 0.5;
                cyw = c.by + 0.5;
                czw = c.bz + 0.5;
            }
            final double px = cxw + dirX * 0.5;
            final double py = cyw + dirY * 0.5;
            final double pz = czw + dirZ * 0.5;

            // Impact velocity: relative normal speed at the contact point.
            final double[] v = this.velScratch.get();
            velocityAt(sb1, px, py, pz, v, 0);
            velocityAt(sb2, px, py, pz, v, 3);
            final double impact = Math.abs((v[0] - v[3]) * dirX + (v[1] - v[4]) * dirY + (v[2] - v[5]) * dirZ);

            // Hit position in the block's own frame (plot coordinates for sub-levels).
            double hx = px, hy = py, hz = pz;
            if (self != null) {
                final Vector3f r = scene.rotateInto((float) (px - self.snapComX), (float) (py - self.snapComY), (float) (pz - self.snapComZ),
                        -self.snapQx, -self.snapQy, -self.snapQz, self.snapQw);
                hx = self.centerOfMass.x + r.x;
                hy = self.centerOfMass.y + r.y;
                hz = self.centerOfMass.z + r.z;
            }

            int ox = 0, oy = 0, oz = 0;
            final boolean hasOther = other != null;
            if (other != null) {
                if (otherIsTerrain) {
                    // merged terrain slabs: derive the touched block from the point
                    ox = floor(px + dirX * 0.25);
                    oy = floor(py + dirY * 0.25);
                    oz = floor(pz + dirZ * 0.25);
                } else {
                    ox = other.bx;
                    oy = other.by;
                    oz = other.bz;
                }
            }
            scene.queueCallback(new PendingCallback(entry, key, c.bx, c.by, c.bz, ox, oy, oz, hasOther, hx, hy, hz, impact));
        }

        /**
         * Surface velocity of a mounted contraption block (its motion relative
         * to the mount body), evaluated at the block center.
         */
        private void applyContraptionSurfaceVelocity(@Nullable final Child c, @Nullable final SableBody body,
                                                     final ContactSettings settings, final boolean isFirst) {
            if (c == null || body == null || c.owner == null || c.owner == body || c.owner.kind != SableBody.Kind.CONTRAPTION) {
                return;
            }
            final SableBody contraption = c.owner;
            if (contraption.linVel.lengthSquared() == 0.0 && contraption.angVel.lengthSquared() == 0.0) {
                return;
            }
            // local velocity at the block: v + w x r (contraption frame)
            final double rx = c.bx + 0.5 - contraption.centerOfMass.x;
            final double ry = c.by + 0.5 - contraption.centerOfMass.y;
            final double rz = c.bz + 0.5 - contraption.centerOfMass.z;
            final Vector3d w = contraption.angVel;
            final double lvx = contraption.linVel.x + w.y * rz - w.z * ry;
            final double lvy = contraption.linVel.y + w.z * rx - w.x * rz;
            final double lvz = contraption.linVel.z + w.x * ry - w.y * rx;
            // contraption frame -> mount block space -> world
            final Quaterniond rel = contraption.relRot;
            final Vector3d mountLocal = new Vector3d(lvx, lvy, lvz);
            rel.transform(mountLocal);
            final Vector3f r = JoltPhysicsScene.this.rotateInto((float) mountLocal.x, (float) mountLocal.y, (float) mountLocal.z,
                    body.snapQx, body.snapQy, body.snapQz, body.snapQw);

            // relative surface velocity: body2 relative to body1
            final float sign = isFirst ? -1.0f : 1.0f;
            final Vec3 existing = settings.getRelativeLinearSurfaceVelocity();
            final Vec3 combined = JoltPhysicsScene.this.tmpVec.get();
            combined.set(existing.getX() + sign * r.x,
                    existing.getY() + sign * r.y,
                    existing.getZ() + sign * r.z);
            settings.setRelativeLinearSurfaceVelocity(combined);
        }

        private void report(@Nullable final SableBody sb1, @Nullable final SableBody sb2, @Nullable final Child c1, @Nullable final Child c2,
                            final Body body1, final Body body2, final double nx, final double ny, final double nz) {
            // No collision damage while god-dragging.
            if ((sb1 != null && JoltPhysicsScene.this.creativeDragBodies.contains(sb1.joltId))
                    || (sb2 != null && JoltPhysicsScene.this.creativeDragBodies.contains(sb2.joltId))) {
                return;
            }
            final JoltPhysicsScene scene = JoltPhysicsScene.this;

            // contact point approximated by the involved block center (body-local, relative to the COM)
            double px1 = 0, py1 = 0, pz1 = 0;
            double px2 = 0, py2 = 0, pz2 = 0;
            double wx1, wy1, wz1;
            if (c1 != null && sb1 != null && c1.owner == sb1) {
                px1 = c1.bx + 0.5 - sb1.centerOfMass.x;
                py1 = c1.by + 0.5 - sb1.centerOfMass.y;
                pz1 = c1.bz + 0.5 - sb1.centerOfMass.z;
                final Vector3f r = scene.rotateInto((float) px1, (float) py1, (float) pz1, sb1.snapQx, sb1.snapQy, sb1.snapQz, sb1.snapQw);
                wx1 = sb1.snapComX + r.x;
                wy1 = sb1.snapComY + r.y;
                wz1 = sb1.snapComZ + r.z;
            } else if (sb1 != null) {
                wx1 = sb1.snapComX;
                wy1 = sb1.snapComY;
                wz1 = sb1.snapComZ;
            } else {
                return; // body 1 is always the moving one
            }
            if (c2 != null && sb2 != null && c2.owner == sb2) {
                px2 = c2.bx + 0.5 - sb2.centerOfMass.x;
                py2 = c2.by + 0.5 - sb2.centerOfMass.y;
                pz2 = c2.bz + 0.5 - sb2.centerOfMass.z;
            }

            final double[] v = this.velScratch.get();
            velocityAt(sb1, wx1, wy1, wz1, v, 0);
            velocityAt(sb2, wx1, wy1, wz1, v, 3);
            final double rvx = v[0] - v[3], rvy = v[1] - v[4], rvz = v[2] - v[5];
            // Cheap reject: below this relative speed no record can pass the
            // pipeline's 25 * minMass force threshold.
            if (rvx * rvx + rvy * rvy + rvz * rvz < 1.5625) {
                return;
            }
            final double relSpeed = Math.abs(rvx * nx + rvy * ny + rvz * nz);
            final float invM1 = body1.isDynamic() ? body1.getMotionProperties().getInverseMass() : 0.0f;
            final float invM2 = body2.isDynamic() ? body2.getMotionProperties().getInverseMass() : 0.0f;
            final double effMass = (invM1 + invM2) > 1.0e-9f ? 1.0 / (invM1 + invM2) : 0.0;
            // force estimate: normal speed * effective mass * tick rate
            final double force = relSpeed * effMass * 20.0;
            final double m1 = invM1 > 0.0f ? 1.0 / invM1 : Double.MAX_VALUE;
            final double m2 = invM2 > 0.0f ? 1.0 / invM2 : Double.MAX_VALUE;
            if (!(force > 25.0 * Math.min(m1, m2))) {
                return;
            }
            final int id1 = sb1.kind == SableBody.Kind.SUB_LEVEL ? sb1.runtimeId : -1;
            final int id2 = sb2 != null && sb2.kind == SableBody.Kind.SUB_LEVEL ? sb2.runtimeId : -1;
            final double[] rec = new double[15];
            rec[0] = id1;
            rec[1] = id2;
            rec[2] = force;
            rec[3] = nx;
            rec[4] = ny;
            rec[5] = nz;
            rec[6] = -nx;
            rec[7] = -ny;
            rec[8] = -nz;
            rec[9] = px1;
            rec[10] = py1;
            rec[11] = pz1;
            rec[12] = px2;
            rec[13] = py2;
            rec[14] = pz2;

            scene.reportCollision(rec);
        }
    }
}
