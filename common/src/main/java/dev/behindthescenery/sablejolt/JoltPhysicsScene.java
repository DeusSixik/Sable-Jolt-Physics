package dev.behindthescenery.sablejolt;

import com.github.stephengold.joltjni.*;
import com.github.stephengold.joltjni.enumerate.EActivation;
import com.github.stephengold.joltjni.enumerate.EAxis;
import com.github.stephengold.joltjni.enumerate.EMotionQuality;
import com.github.stephengold.joltjni.enumerate.EMotionType;
import com.github.stephengold.joltjni.enumerate.EMotorState;
import com.github.stephengold.joltjni.enumerate.EOverrideMassProperties;
import com.github.stephengold.joltjni.enumerate.ESpringMode;
import dev.behindthescenery.sablejolt.collider.JoltVoxelColliderData;
import dev.ryanhcode.sable.Sable;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.longs.LongArrayList;
import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.Nullable;
import org.joml.*;

import java.lang.Math;
import java.lang.Runtime;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A Jolt physics scene: the direct equivalent of the native rapier {@code PhysicsScene}.
 * <p>
 * Every sub-level is a dynamic body whose compound shape is rebuilt from voxel
 * chunk data; the global world is stored as one static body per chunk section.
 */
@ApiStatus.Internal
public final class JoltPhysicsScene {
    public static final int LAYER_MOVING = 0;
    public static final int LAYER_STATIC = 1;
    public static final int LAYER_ROPE = 2;

    /**
     * Bitmask of all DOFs, used when overriding mass properties.
     */
    private static final int ALLOWED_DOFS_ALL = 0x3F;

    /**
     * Density of fluids in per-block mass units. Buoyancy is gravity-proportional
     * (F = density * submergedVolume * |g|), so a body floats when its average
     * block mass is below this value and sinks when it is above.
     */
    private static final double FLUID_DENSITY = 2.0;

    private final PhysicsSystem system;
    private final BodyInterface bi;

    public BodyInterface getBodyInterface() {
        return this.bi;
    }
    private final TempAllocator tempAllocator;
    private final JobSystem jobSystem;
    private final JobSystem singleThreadJobSystem;
    private static final int WORKER_THREADS = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);
    private final java.util.concurrent.ExecutorService bodyWorkers = java.util.concurrent.Executors.newFixedThreadPool(
            WORKER_THREADS, r -> {
                final Thread t = new Thread(r, "sable-jolt-worker");
                t.setDaemon(true);
                return t;
            });
    private static final int MULTITHREAD_BODY_THRESHOLD = 24;

    /**
     * A body with at least this many compound children makes the native narrow
     * phase dominate the step; such scenes use the multithreaded job system
     * regardless of the body count.
     */
    private static final int HEAVY_BODY_CHILDREN = 2048;

    /**
     * Buoyancy iterates compound children per tick; above this count the loop is
     * subsampled (with volume compensation) so a 30k-child body costs the same
     * as a 4k-child one.
     */
    private static final int BUOYANCY_MAX_CHILDREN = 4096;
    private final ContactListener contactListener;

    public final JoltVoxelColliderData.Registry colliderRegistry = new JoltVoxelColliderData.Registry();

    /**
     * Set by the pipeline; used to lazily (re)build collision boxes of entries whose
     * first computation yielded no boxes.
     */
    private volatile dev.behindthescenery.sablejolt.collider.JoltVoxelColliderBakery colliderBakery;

    public void attachColliderBakery(final dev.behindthescenery.sablejolt.collider.JoltVoxelColliderBakery bakery) {
        this.colliderBakery = bakery;
    }

    private final Int2ObjectOpenHashMap<SableBody> bodies = new Int2ObjectOpenHashMap<>();
    private final Int2ObjectOpenHashMap<SableBody> bodiesByJoltId = new Int2ObjectOpenHashMap<>();
    private final Long2ObjectOpenHashMap<GlobalChunk> globalChunks = new Long2ObjectOpenHashMap<>();
    private final Int2ObjectOpenHashMap<GlobalChunk> globalChunksByBodyId = new Int2ObjectOpenHashMap<>();
    private final Long2ObjectOpenHashMap<ChunkSectionData> allChunks = new Long2ObjectOpenHashMap<>();
    private final Long2ObjectOpenHashMap<JointRecord> joints = new Long2ObjectOpenHashMap<>();
    private final Long2ObjectOpenHashMap<RopeStrand> ropes = new Long2ObjectOpenHashMap<>();

    private final Int2ObjectOpenHashMap<org.joml.Quaterniond> angularFollowPrev = new Int2ObjectOpenHashMap<>();
    private final Int2ObjectOpenHashMap<org.joml.Vector3d> angularCachedOmega = new Int2ObjectOpenHashMap<>();

    public org.joml.Quaterniond getAngularFollowPrev(final int joltId) {
        return this.angularFollowPrev.get(joltId);
    }

    public void setAngularFollowPrev(final int joltId, final org.joml.Quaterniond q) {
        this.angularFollowPrev.computeIfAbsent(joltId, k -> new org.joml.Quaterniond()).set(q);
    }

    public org.joml.Vector3d getCachedAngularOmega(final int joltId) {
        return this.angularCachedOmega.get(joltId);
    }

    public void setCachedAngularOmega(final int joltId, final org.joml.Vector3d omega) {
        this.angularCachedOmega.computeIfAbsent(joltId, k -> new org.joml.Vector3d()).set(omega);
    }

    public void clearAngularServo(final int joltId) {
        this.angularFollowPrev.remove(joltId);
        this.angularCachedOmega.remove(joltId);
    }

    private final AtomicInteger nextRuntimeId = new AtomicInteger();
    private final AtomicLong nextJointId = new AtomicLong(1);
    private final AtomicLong nextRopeId = new AtomicLong(1);
    private final AtomicInteger nextGroupId = new AtomicInteger(1);

    private final ArrayList<double[]> reportedCollisions = new ArrayList<>();

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

    private GroupFilterTable mountFilter;
    private GroupFilterTableRef mountFilterRef;
    private int groundBodyId;

    final double gravityX;
    final double gravityY;
    final double gravityZ;
    final double universalDrag;

    JoltPhysicsScene(final double gravityX, final double gravityY, final double gravityZ, final double universalDrag) {
        JoltNative.ensureInitialized();

        this.gravityX = gravityX;
        this.gravityY = gravityY;
        this.gravityZ = gravityZ;
        this.universalDrag = universalDrag;

        this.mountFilter = new GroupFilterTable(16);
        this.mountFilterRef = this.mountFilter.toRef();

        final int numObjLayers = 3;
        final int numBpLayers = 2;

        final ObjectLayerPairFilterTable ovoFilter = new ObjectLayerPairFilterTable(numObjLayers);
        ovoFilter.enableCollision(LAYER_MOVING, LAYER_MOVING);
        ovoFilter.enableCollision(LAYER_MOVING, LAYER_STATIC);
        ovoFilter.enableCollision(LAYER_MOVING, LAYER_ROPE);
        ovoFilter.enableCollision(LAYER_STATIC, LAYER_ROPE);

        final BroadPhaseLayerInterfaceTable layerMap = new BroadPhaseLayerInterfaceTable(numObjLayers, numBpLayers);
        layerMap.mapObjectToBroadPhaseLayer(LAYER_MOVING, 0);
        layerMap.mapObjectToBroadPhaseLayer(LAYER_ROPE, 0);
        layerMap.mapObjectToBroadPhaseLayer(LAYER_STATIC, 1);

        final ObjectVsBroadPhaseLayerFilterTable ovbFilter = new ObjectVsBroadPhaseLayerFilterTable(layerMap, numBpLayers, ovoFilter, numObjLayers);

        this.system = new PhysicsSystem();
        this.system.init(100_000, 0, 262_144, 65_536, layerMap, ovbFilter, ovoFilter);
        this.system.setGravity((float) gravityX, (float) gravityY, (float) gravityZ);

        // Tighter contact margins than the Jolt defaults: dragged bodies must not
        // sink into walls, and penetration is corrected aggressively along normals.
        final PhysicsSettings physics = this.system.getPhysicsSettings();
        physics.setPenetrationSlop(0.005f);
        physics.setSpeculativeContactDistance(0.02f);
        physics.setNumVelocitySteps(10);
        physics.setNumPositionSteps(4);
        // Big resting structures form huge contact islands; the splitter parallelizes
        // and bounds their solve instead of one monolithic island iteration.
        physics.setUseLargeIslandSplitter(true);
        physics.setUseManifoldReduction(true);
        this.system.setPhysicsSettings(physics);

        this.bi = this.system.getBodyInterface();
        this.tempAllocator = new TempAllocatorMalloc();
        this.jobSystem = new JobSystemThreadPool(Jolt.cMaxPhysicsJobs, Jolt.cMaxPhysicsBarriers, Math.max(1, Runtime.getRuntime().availableProcessors() - 1));
        this.singleThreadJobSystem = new JobSystemSingleThreaded(Jolt.cMaxPhysicsJobs);

        this.mountFilterRef = this.mountFilter.toRef();
        this.contactListener = new SceneContactListener();
        this.system.setContactListener(this.contactListener);

        this.createGround();
    }

    private Body groundBody;

    private void createGround() {
        // The ground body MUST sit at the world origin: constraint anchors on the
        // world are measured from its center of mass, and a shifted origin would
        // add a phantom offset to every motor target.
        final BodyCreationSettings bcs = new BodyCreationSettings(
                new com.github.stephengold.joltjni.BoxShape(new Vec3(0.01f, 0.01f, 0.01f), 0.001f),
                RVec3.sZero(), Quat.sIdentity(), EMotionType.Static, LAYER_STATIC);
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

    public static final class Child {
        public int bx;
        public int by;
        public int bz;
        public int colliderId;

        public Child(final int bx, final int by, final int bz, final int colliderId) {
            this.bx = bx;
            this.by = by;
            this.bz = bz;
            this.colliderId = colliderId;
        }
    }

    /**
     * Bookkeeping for a single rigid body managed by this scene.
     */
    public static final class SableBody {
        public enum Kind {SUB_LEVEL, BOX, CONTRAPTION}

        public final Kind kind;
        public final int runtimeId;
        public int joltId;

        /**
         * The Java wrapper of the native body, kept alive for the body's lifetime.
         * {@code new Body(long)} takes a virtual address, not a body ID, so the
         * wrapper from {@code createBody} must be retained instead of reconstructed.
         */
        public Body body;

        public MutableCompoundShape shape;
        public final ObjectArrayList<Child> children = new ObjectArrayList<>();

        public final Long2ObjectOpenHashMap<ChunkSectionData> chunks = new Long2ObjectOpenHashMap<>();

        public int minX;
        public int minY;
        public int minZ;
        public int maxX;
        public int maxY;
        public int maxZ;
        public boolean hasBounds;

        public final Vector3d centerOfMass = new Vector3d();

        /**
         * Full sub-shape ID → leaf child. The body shape is a two-level compound
         * (per-section static compounds under one parent), so the raw sub-shape ID
         * from contacts is NOT the child index; this map, rebuilt with the shape,
         * is the only mapping.
         */
        public final Int2ObjectOpenHashMap<Child> childById = new Int2ObjectOpenHashMap<>();

        public         int mountId = -1;
        public final Vector3d relPos = new Vector3d();
        public final Quaterniond relRot = new Quaterniond(1.0, 0.0, 0.0, 0.0);
        public final Vector3d linVel = new Vector3d();
        public final Vector3d angVel = new Vector3d();
        public int groupSubId = -1;

        // Dedup state of the last shape rebuild: if none of these inputs changed,
        // the rebuild is skipped (mass-stat updates arrive every tick).
        long rebuiltDataVersion = -1;
        boolean rebuiltHasBounds;
        int rebuiltMinX;
        int rebuiltMinY;
        int rebuiltMinZ;
        int rebuiltMaxX;
        int rebuiltMaxY;
        int rebuiltMaxZ;
        double rebuiltComX = Double.NaN;
        double rebuiltComY;
        double rebuiltComZ;
        int rebuiltOwnChunks = -1;

        /**
         * Nano time of the last completed shape rebuild. rebuildShape is throttled
         * against this: a schematic spawn fires onStatsChanged per placed block,
         * and without throttling the full compound rebuild runs once per block
         * (quadratic in the structure size, tens of seconds of server freeze).
         */
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

        @Nullable
        public Child childAt(final int subShapeId) {
            return this.childById.get(subShapeId);
        }
    }

    public static final class GlobalChunk {
        final int cx;
        final int cy;
        final int cz;
        final ChunkSectionData data;
        int joltId;

        /**
         * The Java wrapper of the native body; must stay referenced for the body's
         * whole lifetime, otherwise the jolt-jni cleaner frees the native body while
         * the physics system still uses it.
         */
        Body body;

        com.github.stephengold.joltjni.readonly.ConstShape shape;
        final ArrayList<Child> children = new ArrayList<>();

        /**
         * Full sub-shape ID → leaf child (see SableBody.childById); the chunk shape
         * is a static compound whose IDs are hierarchical, not plain indices.
         */
        final Int2ObjectOpenHashMap<Child> childById = new Int2ObjectOpenHashMap<>();

        GlobalChunk(final int cx, final int cy, final int cz, final ChunkSectionData data) {
            this.cx = cx;
            this.cy = cy;
            this.cz = cz;
            this.data = data;
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
        double comX;
        double comY;
        double comZ;
        public Constraint constraint;
    }

    public     static final class RopeStrand {
        final ArrayList<Integer> points = new ArrayList<>();
        final ArrayList<Body> pointBodies = new ArrayList<>();
        final ArrayList<DistanceConstraint> joints = new ArrayList<>();
        double pointRadius;
        double firstJointLength;
        @Nullable RopeAttachment start;
        @Nullable RopeAttachment end;
    }

    //endregion

    //region Body management

    private SableBody createBody(final SableBody.Kind kind, final int runtimeId, final Vector3dc pos, final Quaterniondc rot, final int layer) {
        final MutableCompoundShape shape = new MutableCompoundShape();
        final BodyCreationSettings bcs = new BodyCreationSettings(
                shape,
                new RVec3(pos.x(), pos.y(), pos.z()),
                new Quat((float) rot.x(), (float) rot.y(), (float) rot.z(), (float) rot.w()),
                kind == SableBody.Kind.CONTRAPTION ? EMotionType.Kinematic : EMotionType.Dynamic,
                layer);
        bcs.setLinearDamping((float) this.universalDrag);
        bcs.setAngularDamping((float) this.universalDrag);
        bcs.setMotionQuality(EMotionQuality.LinearCast);
        bcs.setFriction(0.525f);
        bcs.setRestitution(0.0f);
        bcs.setApplyGyroscopicForce(true);
        bcs.setOverrideMassProperties(EOverrideMassProperties.CalculateMassAndInertia);

        final Body body = this.bi.createBody(bcs);
        body.setUserData(runtimeId);
        this.bi.addBody(body, EActivation.DontActivate);

        final SableBody sb = new SableBody(kind, runtimeId);
        sb.joltId = body.getId();
        sb.body = body;
        sb.shape = shape;
        this.bodies.put(runtimeId, sb);
        this.bodiesByJoltId.put(sb.joltId, sb);
        return sb;
    }

    private void destroyBody(final SableBody sb) {
        this.bodies.remove(sb.runtimeId);
        this.bodiesByJoltId.remove(sb.joltId);
        this.dirtyBodies.remove(sb);
        if (sb.mountId != -1) {
            final SableBody mount = this.bodies.get(sb.mountId);
            if (mount != null) {
                mount.mountId = -1;
            }
        }
        this.bi.removeBody(sb.joltId);
        this.bi.destroyBody(sb.joltId);
    }

    public void createSubLevel(final int id, final Vector3dc pos, final Quaterniondc rot) {
        this.createBody(SableBody.Kind.SUB_LEVEL, id, pos, rot, LAYER_MOVING);
    }

    public void removeSubLevel(final int id) {
        final SableBody sb = this.bodies.get(id);
        if (sb != null) {
            this.destroyBody(sb);
        }
    }

    public void createBox(final int id, final double mass, final double hx, final double hy, final double hz, final Vector3dc pos, final Quaterniondc rot) {
        final SableBody sb = this.createBody(SableBody.Kind.BOX, id, pos, rot, LAYER_MOVING);

        final BodyCreationSettings bcs = new BodyCreationSettings(
                new com.github.stephengold.joltjni.BoxShape(new Vec3((float) hx, (float) hy, (float) hz), 0.025f),
                new RVec3(pos.x(), pos.y(), pos.z()),
                new Quat((float) rot.x(), (float) rot.y(), (float) rot.z(), (float) rot.w()),
                EMotionType.Dynamic, LAYER_MOVING);
        bcs.setMotionQuality(EMotionQuality.LinearCast);
        bcs.setFriction(0.45f);
        bcs.setApplyGyroscopicForce(true);
        final MassProperties mp = new MassProperties();
        mp.setMass((float) mass);
        bcs.setOverrideMassProperties(EOverrideMassProperties.CalculateInertia);
        bcs.setMassPropertiesOverride(mp);

        final Body body = this.bi.createBody(bcs);
        body.setUserData(id);
        this.bi.addBody(body, EActivation.DontActivate);

        this.bodiesByJoltId.remove(sb.joltId);
        this.bi.removeBody(sb.joltId);
        this.bi.destroyBody(sb.joltId);

        sb.joltId = body.getId();
        sb.body = body;
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
        final Body body = sb.body;
        final RVec3 pos = body.getPosition();
        final Quat rot = body.getRotation();
        store.p1 = pos.xx();
        store.p2 = pos.yy();
        store.p3 = pos.zz();
        store.p4 = rot.getX();
        store.p5 = rot.getY();
        store.p6 = rot.getZ();
        store.p7 = rot.getW();
    }

    /**
     * Updates the center of mass bookkeeping of a body and rebuilds its voxel shape offsets.
     */
    public void setCenterOfMass(final int id, final double x, final double y, final double z) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        sb.centerOfMass.set(x, y, z);
        // Deferred: a schematic spawn fires this per placed block; a synchronous
        // full compound rebuild per block is quadratic in the structure size.
        this.markDirty(sb);
    }

    public void setLocalBounds(final int id, final int minX, final int minY, final int minZ, final int maxX, final int maxY, final int maxZ) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        sb.minX = minX;
        sb.minY = minY;
        sb.minZ = minZ;
        sb.maxX = maxX;
        sb.maxY = maxY;
        sb.maxZ = maxZ;
        sb.hasBounds = true;
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

        final Body body = sb.body;
        final var motion = body.getMotionProperties();
        if (motion == null) {
            return;
        }

        final MassProperties mp = new MassProperties();
        mp.setMass((float) mass);
        // Mat44's varargs constructor expects 16 floats in column-major order
        mp.setInertia(new Mat44(
                (float) inertiaTensor.m00(), (float) inertiaTensor.m10(), (float) inertiaTensor.m20(), 0.0f,
                (float) inertiaTensor.m01(), (float) inertiaTensor.m11(), (float) inertiaTensor.m21(), 0.0f,
                (float) inertiaTensor.m02(), (float) inertiaTensor.m12(), (float) inertiaTensor.m22(), 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f
        ));
        motion.setMassProperties(ALLOWED_DOFS_ALL, mp);
    }

    public void teleportObject(final int id, final double x, final double y, final double z, final double i, final double j, final double k, final double r) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        this.bi.setPositionAndRotation(sb.joltId, new RVec3(x, y, z),
                new Quat((float) i, (float) j, (float) k, (float) r), EActivation.Activate);
    }

    public void wakeUpObject(final int id) {
        final SableBody sb = this.bodies.get(id);
        if (sb != null) {
            this.bi.activateBody(sb.joltId);
        }
    }

    public void addLinearAngularVelocities(final int id, final double lx, final double ly, final double lz,
                                           final double ax, final double ay, final double az, final boolean wakeUp) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        final Body body = sb.body;
        if (!wakeUp && !body.isActive()) {
            return;
        }
        final Vec3 lin = body.getLinearVelocity();
        final Vec3 ang = body.getAngularVelocity();
        this.bi.setLinearAndAngularVelocity(sb.joltId,
                new Vec3((float) (lin.getX() + lx), (float) (lin.getY() + ly), (float) (lin.getZ() + lz)),
                new Vec3((float) (ang.getX() + ax), (float) (ang.getY() + ay), (float) (ang.getZ() + az)));
        if (wakeUp) {
            this.bi.activateBody(sb.joltId);
        }
    }

    public void getLinearVelocity(final int id, final PoseCache store) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        final Vec3 v = sb.body.getLinearVelocity();
        store.p1 = v.getX();
        store.p2 = v.getY();
        store.p3 = v.getZ();
    }

    public void getAngularVelocity(final int id, final PoseCache store) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        final Vec3 v = sb.body.getAngularVelocity();
        store.p1 = v.getX();
        store.p2 = v.getY();
        store.p3 = v.getZ();
    }

    /**
     * Applies an impulse at a local offset (relative to the center of mass), replicating the rapier semantics.
     */
    public void applyForce(final int id, final double x, final double y, final double z,
                           final double fx, final double fy, final double fz, final boolean wakeUp) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        final Body body = sb.body;
        if (!wakeUp && !body.isActive()) {
            return;
        }

        final Quat rot = body.getRotation();
        final float qx = rot.getX();
        final float qy = rot.getY();
        final float qz = rot.getZ();
        final float qw = rot.getW();

        final Vec3 impulse = rotate((float) fx, (float) fy, (float) fz, qx, qy, qz, qw);
        final Vec3 offset = rotate((float) x, (float) y, (float) z, qx, qy, qz, qw);
        final RVec3 com = body.getCenterOfMassPosition();

        // Jolt does not wake bodies on AddImpulse; activation is required separately
        body.addImpulse(impulse, new RVec3(com.xx() + offset.getX(), com.yy() + offset.getY(), com.zz() + offset.getZ()));
        if (wakeUp) {
            this.bi.activateBody(sb.joltId);
        }
    }

    public void applyForceAndTorque(final int id, final double fx, final double fy, final double fz,
                                    final double tx, final double ty, final double tz, final boolean wakeUp) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        final Body body = sb.body;
        if (!wakeUp && !body.isActive()) {
            return;
        }

        final Quat rot = body.getRotation();
        final float rotX = rot.getX();
        final float rotY = rot.getY();
        final float rotZ = rot.getZ();
        final float rotW = rot.getW();

        final Vec3 impulse = rotate((float) fx, (float) fy, (float) fz, rotX, rotY, rotZ, rotW);
        body.addImpulse(impulse.getX(), impulse.getY(), impulse.getZ());
        body.addAngularImpulse(rotate((float) tx, (float) ty, (float) tz, rotX, rotY, rotZ, rotW));
        if (wakeUp) {
            this.bi.activateBody(sb.joltId);
        }
    }

    private static Vec3 rotate(final float vx, final float vy, final float vz,
                               final Quat q) {
        return rotate(vx, vy, vz, q.getX(), q.getY(), q.getZ(), q.getW());
    }

    private static Vec3 rotate(final float vx, final float vy, final float vz,
                               final float qx, final float qy, final float qz, final float qw) {
        // t = 2 * cross(q.xyz, v)
        final float tx = 2.0f * (qy * vz - qz * vy);
        final float ty = 2.0f * (qz * vx - qx * vz);
        final float tz = 2.0f * (qx * vy - qy * vx);
        // v + w*t + cross(q.xyz, t)
        return new Vec3(
                vx + qw * tx + (qy * tz - qz * ty),
                vy + qw * ty + (qz * tx - qx * tz),
                vz + qw * tz + (qx * ty - qy * tx));
    }

    /**
     * Converts a world-space point into the body-local frame (relative to the
     * center of mass, rotated by the body rotation) as required by Jolt's
     * {@code EConstraintSpace.LocalToBodyCom}. For the static ground the world
     * point is used as-is.
     */
    public RVec3 toLocalAnchor(@Nullable final SableBody sb, final double x, final double y, final double z) {
        if (sb == null) {
            return new RVec3(x, y, z);
        }
        final Vec3 offset = rotate((float) (x - sb.centerOfMass.x), (float) (y - sb.centerOfMass.y), (float) (z - sb.centerOfMass.z),
                sb.body.getRotation().conjugated());
        return new RVec3(offset.getX(), offset.getY(), offset.getZ());
    }

    //endregion

    //region Voxel geometry

    private boolean isSolidBlock(final int packed, final JoltVoxelColliderData entry) {
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
        // lazily (re)build the boxes: the first computation may have run before the
        // block existed at its target position, yielding a registered-but-empty entry
        if (!entry.hasBoxes()) {
            entry.ensureBoxes(this.colliderBakery);
        }
        return entry.hasBoxes();
    }

    /**
     * Minimal interval between two full compound rebuilds of one body. Mass-stat
     * storms (schematic spawns fire one {@code onStatsChanged} per placed block)
     * coalesce into a couple of rebuilds per second instead of one per block.
     */
    private static final long SHAPE_REBUILD_THROTTLE_NANOS = 500_000_000L;

    private void rebuildShape(final SableBody sb) {
        // Dedup: mass-stat updates arrive every tick with the same inputs; a rebuild
        // is only needed when the com, bounds, chunk set or chunk data changed.
        if (sb.rebuiltDataVersion == this.chunkDataVersion
                && sb.rebuiltHasBounds == sb.hasBounds
                && sb.rebuiltMinX == sb.minX && sb.rebuiltMinY == sb.minY && sb.rebuiltMinZ == sb.minZ
                && sb.rebuiltMaxX == sb.maxX && sb.rebuiltMaxY == sb.maxY && sb.rebuiltMaxZ == sb.maxZ
                && sb.rebuiltComX == sb.centerOfMass.x && sb.rebuiltComY == sb.centerOfMass.y && sb.rebuiltComZ == sb.centerOfMass.z
                && sb.rebuiltOwnChunks == sb.chunks.size()) {
            return;
        }

        // Throttle: during a mass-stat storm every flush would rebuild the whole
        // compound; cap the rate and re-mark for a later flush instead.
        final long now = System.nanoTime();
        if (sb.lastShapeRebuildNanos != 0 && now - sb.lastShapeRebuildNanos < SHAPE_REBUILD_THROTTLE_NANOS) {
            this.markDirty(sb);
            return;
        }
        sb.lastShapeRebuildNanos = now;

        final MutableCompoundShape shape = new MutableCompoundShape();
        sb.children.clear();
        sb.childById.clear();

        // Two-level compound: per-section static compounds (BVH-backed narrow
        // phase) under one small parent. A flat mutable compound forces Jolt to
        // brute-force every child per query, which made big bodies dominate the
        // simulation step; sections keep the per-section cost logarithmic.
        final LongArrayList sectionKeys = new LongArrayList();
        if (sb.kind == SableBody.Kind.CONTRAPTION) {
            // Contraptions own a dedicated chunk store (they have no plot java-side).
            for (final var entry : sb.chunks.long2ObjectEntrySet()) {
                sectionKeys.add(entry.getLongKey());
            }
        } else if (sb.kind == SableBody.Kind.SUB_LEVEL && sb.hasBounds) {
            // Sub-levels build from the SHARED chunk store (plot sections are uploaded
            // without an owner id), filtered by the body bounds - like the rapier
            // octree rebuild from main_level_chunks.
            final int chunkMinX = (sb.minX >> 4) - 1;
            final int chunkMinY = (sb.minY >> 4) - 1;
            final int chunkMinZ = (sb.minZ >> 4) - 1;
            final int chunkMaxX = (sb.maxX >> 4) + 1;
            final int chunkMaxY = (sb.maxY >> 4) + 1;
            final int chunkMaxZ = (sb.maxZ >> 4) + 1;

            for (final var entry : this.allChunks.long2ObjectEntrySet()) {
                final long key = entry.getLongKey();
                final int cx = unpackChunkX(key);
                final int cy = unpackChunkY(key);
                final int cz = unpackChunkZ(key);
                if (cx < chunkMinX || cx > chunkMaxX || cy < chunkMinY || cy > chunkMaxY || cz < chunkMinZ || cz > chunkMaxZ) {
                    continue;
                }
                sectionKeys.add(key);
            }
        }
        final long[] keys = sectionKeys.toLongArray();
        java.util.Arrays.sort(keys);

        int sectionOrdinal = 0;
        int sectionsBuilt = 0;
        try {
            for (final long key : keys) {
                final ChunkSectionData data = sb.kind == SableBody.Kind.CONTRAPTION
                        ? sb.chunks.get(key) : this.allChunks.get(key);
                if (data == null) {
                    continue;
                }
                final StaticCompoundShapeSettings sectionSettings = new StaticCompoundShapeSettings();
                final ArrayList<Child> sectionChildren = new ArrayList<>();
                // Section COM (com-relative space): Jolt rebases section children onto
                // the section center of mass when the static compound is created, so
                // the section must be placed at exactly that offset inside the parent.
                final Object[] sectionBuild = this.appendSectionBlocks(sb, sectionSettings, sectionChildren,
                        unpackChunkX(key), unpackChunkY(key), unpackChunkZ(key), data,
                        sb.kind == SableBody.Kind.CONTRAPTION);
                if (sectionBuild == null) {
                    // empty section (all air / no leaf boxes) — not an error
                    continue;
                }
                final double[] sectionCom = (double[]) sectionBuild[0];
                final Vec3 singleOffset = (Vec3) sectionBuild[1];
                final com.github.stephengold.joltjni.readonly.ConstShape singleLeaf =
                        (com.github.stephengold.joltjni.readonly.ConstShape) sectionBuild[2];
                if (sectionChildren.isEmpty()) {
                    continue;
                }

                final SubShapeIdCreator prefix;
                if (singleLeaf != null) {
                    // Jolt's CompoundShapeSettings.create() returns the single child
                    // shape itself (dropping its offset) — add the leaf directly.
                    shape.addShape(singleOffset, Quat.sIdentity(), singleLeaf);
                    prefix = shape.getSubShapeIdFromIndex(sectionOrdinal++, new SubShapeIdCreator());
                    final Child child = sectionChildren.get(0);
                    sb.children.add(child);
                    sb.childById.put(prefix.getId(), child);
                    sectionsBuilt++;
                    continue;
                }

                final CompoundShape sectionShape =
                        (CompoundShape) ((com.github.stephengold.joltjni.ShapeRefC) sectionSettings.create().get()).getPtr();
                shape.addShape(new Vec3((float) sectionCom[0], (float) sectionCom[1], (float) sectionCom[2]),
                        Quat.sIdentity(), sectionShape);

                // Full sub-shape ID → child: the parent contributes its child index,
                // the section appends its internal path. Recorded at build time so the
                // contact listener never has to decode the hierarchical ID.
                prefix = shape.getSubShapeIdFromIndex(sectionOrdinal++, new SubShapeIdCreator());
                for (int i = 0; i < sectionChildren.size(); i++) {
                    final int id = sectionShape.getSubShapeIdFromIndex(i, prefix).getId();
                    final Child child = sectionChildren.get(i);
                    sb.children.add(child);
                    sb.childById.put(id, child);
                }
                sectionsBuilt++;
            }
        } catch (final Throwable t) {
            Sable.LOGGER.error("[SableJolt] rebuild body {}: section build FAILED (sectionsBuilt={} children={})",
                    sb.runtimeId, sectionsBuilt, sb.children.size(), t);
            sb.childById.clear();
            sb.children.clear();
            return;
        }

        if (JoltDebugLogging.HANDLE && sectionsBuilt > 1) {
            Sable.LOGGER.info("[SableJolt:handle] rebuild body {}: sections={} children={} idMap={}",
                    sb.runtimeId, sectionsBuilt, sb.children.size(), sb.childById.size());
        }

        sb.shape = shape;
        this.bi.setShape(sb.joltId, shape, false, EActivation.DontActivate);

        // CCD (LinearCast) sweeps the ENTIRE compound against the world every step;
        // for huge bodies that costs seconds per tick while they fall. Huge bodies
        // move slowly in practice, so switch them to discrete motion.
        final boolean huge = sb.children.size() >= HEAVY_BODY_CHILDREN;
        if (huge && this.bi.getMotionQuality(sb.joltId) != EMotionQuality.Discrete) {
            this.bi.setMotionQuality(sb.joltId, EMotionQuality.Discrete);
        }

        sb.rebuiltDataVersion = this.chunkDataVersion;
        sb.rebuiltHasBounds = sb.hasBounds;
        sb.rebuiltMinX = sb.minX;
        sb.rebuiltMinY = sb.minY;
        sb.rebuiltMinZ = sb.minZ;
        sb.rebuiltMaxX = sb.maxX;
        sb.rebuiltMaxY = sb.maxY;
        sb.rebuiltMaxZ = sb.maxZ;
        sb.rebuiltComX = sb.centerOfMass.x;
        sb.rebuiltComY = sb.centerOfMass.y;
        sb.rebuiltComZ = sb.centerOfMass.z;
        sb.rebuiltOwnChunks = sb.chunks.size();

        if (JoltDebugLogging.STAFF) {
            int sectionsInWindow = 0;
            if (sb.kind == SableBody.Kind.SUB_LEVEL && sb.hasBounds) {
                final int cMinX = (sb.minX >> 4) - 2, cMaxX = (sb.maxX >> 4) + 2;
                final int cMinY = (sb.minY >> 4) - 2, cMaxY = (sb.maxY >> 4) + 2;
                final int cMinZ = (sb.minZ >> 4) - 2, cMaxZ = (sb.maxZ >> 4) + 2;
                for (final var entry : this.allChunks.long2ObjectEntrySet()) {
                    final long key = entry.getLongKey();
                    final int cx = unpackChunkX(key);
                    final int cy = unpackChunkY(key);
                    final int cz = unpackChunkZ(key);
                    if (cx >= cMinX && cx <= cMaxX && cy >= cMinY && cy <= cMaxY && cz >= cMinZ && cz <= cMaxZ) {
                        sectionsInWindow++;
                    }
                }
            }
            Sable.LOGGER.info("[SableJolt] rebuild body {}: kind={} ownChunks={} children={} hasBounds={} bounds=({},{},{})..({},{},{}) com=({},{},{}) sectionsInWindow={}",
                    sb.runtimeId, sb.kind, sb.chunks.size(), sb.children.size(), sb.hasBounds,
                    sb.minX, sb.minY, sb.minZ, sb.maxX, sb.maxY, sb.maxZ,
                    sb.centerOfMass.x, sb.centerOfMass.y, sb.centerOfMass.z, sectionsInWindow);
        }
    }

    /**
     * Builds one section's leaf boxes into {@code settings} and returns:
     * [0] the section's volume-weighted center of mass in the body-COM-relative
     * space (children were added at COM-relative offsets; the caller rebases the
     * whole section onto this point), [1] the single leaf's rebased offset and
     * [2] the single leaf shape — both non-null exactly when the section holds a
     * single leaf (Jolt's {@code create()} returns that leaf directly, dropping
     * its offset, so the caller must add it to the parent itself). Returns
     * {@code null} when nothing was added.
     */
    @Nullable
    private Object[] appendSectionBlocks(final SableBody sb, final StaticCompoundShapeSettings settings, final ArrayList<Child> outChildren,
                                         final int cx, final int cy, final int cz,
                                         final ChunkSectionData data, final boolean ignoreBounds) {
        final int blockMinX = cx << 4;
        final int blockMinY = cy << 4;
        final int blockMinZ = cz << 4;

        // First pass: collect leaf boxes with their COM-relative offsets and volumes
        // so the section COM is known before anything is added to the settings.
        final ArrayList<Vec3> offsets = new ArrayList<>();
        final ArrayList<com.github.stephengold.joltjni.readonly.ConstShape> shapes = new ArrayList<>();
        final Vector3d translation = new Vector3d();
        final Vector3d half = new Vector3d();
        double volSum = 0.0;
        double comX = 0.0, comY = 0.0, comZ = 0.0;

        String rejectLog = null;
        final int[] blocks = data.array();

        // NOTE: children stay 1:1 with compound leaf sub-shapes — the ID map built
        // by the caller relies on the leaf order matching this traversal.
        for (int by = 0; by < 16; by++) {
            final int worldY = blockMinY + by;
            final int rowBase = (by << 8);
            for (int bz = 0; bz < 16; bz++) {
                final int worldZ = blockMinZ + bz;
                final int cellBase = rowBase + (bz << 4);
                for (int bx = 0; bx < 16; bx++) {
                    final int packed = blocks[cellBase + bx];
                    final int colliderId = ChunkSectionData.colliderIdOf(packed);
                    final JoltVoxelColliderData entry = colliderId == 0 ? null : this.colliderRegistry.get(colliderId - 1);
                    final boolean solid = isSolidBlock(packed, entry);
                    final int worldX = blockMinX + bx;

                    if (JoltDebugLogging.STAFF && colliderId > 0 && (!solid || (!ignoreBounds && !sb.contains(worldX, worldY, worldZ)))) {
                        final String reason = !solid
                                ? "state=" + ChunkSectionData.voxelStateOf(packed) + (entry == null ? " entry=null" : " boxes=" + entry.boxes.size())
                                : "outOfBounds";
                        rejectLog = appendReject(rejectLog, worldX, worldY, worldZ, reason);
                    }

                    if (!solid) {
                        continue;
                    }
                    if (!ignoreBounds && !sb.contains(worldX, worldY, worldZ)) {
                        continue;
                    }

                    final List<float[]> secBoxes = this.colliderRegistry.get(colliderId - 1).boxes;
                    for (int i = 0; i < secBoxes.size(); i++) {
                        final float[] box = secBoxes.get(i);
                        JoltVoxelColliderData.boxCenter(box, translation);
                        JoltVoxelColliderData.boxHalfExtent(box, half);
                        final double volume = (half.x() * 2.0) * (half.y() * 2.0) * (half.z() * 2.0);
                        final Vec3 off = new Vec3(
                                (float) (worldX + translation.x - sb.centerOfMass.x),
                                (float) (worldY + translation.y - sb.centerOfMass.y),
                                (float) (worldZ + translation.z - sb.centerOfMass.z));
                        offsets.add(off);
                        shapes.add(this.colliderRegistry.get(colliderId - 1).shape(i));
                        outChildren.add(new Child(worldX, worldY, worldZ, colliderId));
                        volSum += volume;
                        comX += off.getX() * volume;
                        comY += off.getY() * volume;
                        comZ += off.getZ() * volume;
                    }
                }
            }
        }

        if (JoltDebugLogging.STAFF && rejectLog != null) {
            Sable.LOGGER.info("[SableJolt] rebuild body {}: rejected: {}", sb.runtimeId, rejectLog);
        }
        if (offsets.isEmpty() || volSum <= 0.0) {
            return null;
        }

        comX /= volSum;
        comY /= volSum;
        comZ /= volSum;

        // Second pass: emit boxes rebased onto the section COM.
        for (int i = 0; i < offsets.size(); i++) {
            settings.addShape(new Vec3(
                    (float) (offsets.get(i).getX() - comX),
                    (float) (offsets.get(i).getY() - comY),
                    (float) (offsets.get(i).getZ() - comZ)), Quat.sIdentity(), shapes.get(i));
        }
        final double[] com = new double[]{comX, comY, comZ};
        if (offsets.size() == 1) {
            return new Object[]{com,
                    new Vec3((float) (offsets.get(0).getX() - comX),
                            (float) (offsets.get(0).getY() - comY),
                            (float) (offsets.get(0).getZ() - comZ)),
                    shapes.get(0)};
        }
        return new Object[]{com, null, null};
    }

    private static String appendReject(final String log, final int x, final int y, final int z, final String reason) {
        final String entry = "block(" + x + "," + y + "," + z + ") " + reason;
        return log == null ? entry : log + "; " + entry;
    }

    //endregion

    //region Chunks

    public void addChunk(final int x, final int y, final int z, final int[] data, final boolean global, final int ownerId) {
        final ChunkSectionData section = new ChunkSectionData();
        System.arraycopy(data, 0, section.array(), 0, ChunkSectionData.BLOCKS);

        final long key = ChunkSectionData.packSectionPos(x, y, z);
        this.allChunks.put(key, section);
        this.chunkDataVersion++;

        if (global) {
            final GlobalChunk chunk = new GlobalChunk(x, y, z, section);
            final GlobalChunk existing = this.globalChunks.put(key, chunk);
            if (existing != null && existing.joltId != 0) {
                this.globalChunksByBodyId.remove(existing.joltId);
                this.bi.removeBody(existing.joltId);
                this.bi.destroyBody(existing.joltId);
                existing.joltId = 0;
            }
            this.refreshGlobalChunkBody(chunk);
            if (JoltDebugLogging.STAFF) {
                Sable.LOGGER.info("[SableJolt] addChunk GLOBAL at ({},{},{}) children={} totalGlobal={}",
                        x, y, z, chunk.children.size(), this.globalChunks.size());
            }
        } else if (ownerId != -1) {
            // Section explicitly claimed by a body (kinematic contraption upload).
            final SableBody sb = this.bodies.get(ownerId);
            if (sb != null) {
                sb.chunks.put(key, section);
                this.markDirty(sb);
            }
            if (JoltDebugLogging.STAFF) {
                Sable.LOGGER.info("[SableJolt] addChunk OWNED at ({},{},{}) owner={} bodyFound={} children={}",
                        x, y, z, ownerId, sb != null, sb != null ? sb.children.size() : -1);
            }
        } else {
            if (JoltDebugLogging.STAFF) {
                Sable.LOGGER.info("[SableJolt] addChunk PLOT (no owner) at ({},{},{}) solids={}",
                        x, y, z, countSolid(section));
            }
            // Plot section uploaded without an owner: rebuild every sub-level whose
            // bounds cover it, mirroring the rapier shared chunk store behavior.
            for (final SableBody sb : this.bodies.values()) {
                if (sb.kind != SableBody.Kind.SUB_LEVEL || !sb.hasBounds) {
                    continue;
                }
                final int cx = x, cy = y, cz = z;
                final boolean intersects = (cx << 4) <= sb.maxX && ((cx << 4) + 15) >= sb.minX
                        && (cy << 4) <= sb.maxY && ((cy << 4) + 15) >= sb.minY
                        && (cz << 4) <= sb.maxZ && ((cz << 4) + 15) >= sb.minZ;
                if (intersects) {
                    this.markDirty(sb);
                }
            }
        }
    }

    private static int countSolid(final ChunkSectionData section) {
        int n = 0;
        for (int i = 0; i < ChunkSectionData.BLOCKS; i++) {
            if (ChunkSectionData.colliderIdOf(section.array()[i]) > 0) {
                n++;
            }
        }
        return n;
    }

    /**
     * Marks a body's shape for rebuild; the rebuild itself is deferred to the next
     * simulation step so bursts of block edits cost a single rebuild.
     */
    private void markDirty(final SableBody sb) {
        // synchronized: rebuildShape runs on worker threads and may re-mark a body
        // when a rebuild is throttled, while the server thread flushes the queue.
        synchronized (this.dirtyBodies) {
            this.dirtyBodies.add(sb);
        }
    }

    private void flushDirty() {
        if (!this.dirtyBodies.isEmpty()) {
            final ObjectArrayList<SableBody> list;
            synchronized (this.dirtyBodies) {
                list = new ObjectArrayList<>(this.dirtyBodies);
                // removed before processing: a throttled rebuildShape re-marks its
                // body for the next flush, and that must survive this flush
                this.dirtyBodies.removeAll(list);
            }
            if (list.size() >= 8) {
                this.runParallel(list, sb -> {
                    try {
                        this.rebuildShape(sb);
                    } catch (final Throwable t) {
                        Sable.LOGGER.error("[SableJolt] failed to rebuild body {}", sb.runtimeId, t);
                    }
                });
            } else {
                for (final SableBody sb : list) {
                    try {
                        this.rebuildShape(sb);
                    } catch (final Throwable t) {
                        Sable.LOGGER.error("[SableJolt] failed to rebuild body {}", sb.runtimeId, t);
                    }
                }
            }
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

    public boolean hasChunk(final int sectionX, final int sectionY, final int sectionZ) {
        return this.allChunks.containsKey(ChunkSectionData.packSectionPos(sectionX, sectionY, sectionZ));
    }

    public void removeChunk(final int x, final int y, final int z, final boolean global) {
        final long key = ChunkSectionData.packSectionPos(x, y, z);
        this.allChunks.remove(key);
        this.chunkDataVersion++;

        if (global) {
            final GlobalChunk chunk = this.globalChunks.remove(key);
            this.fluidSections.remove(key);
            if (chunk != null && chunk.joltId != 0) {
                this.globalChunksByBodyId.remove(chunk.joltId);
                this.bi.removeBody(chunk.joltId);
                this.bi.destroyBody(chunk.joltId);
                chunk.joltId = 0;
            }
            // a removed chunk must not be touched by a pending flush
            this.dirtyGlobalChunks.remove(chunk);
        }
    }

    public void changeBlock(final int x, final int y, final int z, final int newState) {
        final long key = ChunkSectionData.packSectionPos(x >> 4, y >> 4, z >> 4);
        final ChunkSectionData data = this.allChunks.get(key);
        if (data == null) {
            return;
        }
        data.set(x & 15, y & 15, z & 15, newState);
        this.chunkDataVersion++;

        boolean any = false;
        for (final SableBody sb : this.bodies.values()) {
            if (sb.kind == SableBody.Kind.BOX || !sb.contains(x, y, z)) {
                continue;
            }
            // batched: the actual rebuild runs once per simulation step
            this.markDirty(sb);
            any = true;
            break;
        }

        if (!any) {
            final GlobalChunk chunk = this.globalChunks.get(key);
            if (chunk != null) {
                this.dirtyGlobalChunks.add(chunk);
            }
        }
    }

    /**
     * Rebuilds the static chunk body's shape from the section data. Static terrain
     * uses a quadtree-backed static compound shape, which is dramatically faster in
     * the narrow phase than a mutable linear compound.
     */
    private void buildGlobalChunkShape(final GlobalChunk chunk) {
        final com.github.stephengold.joltjni.StaticCompoundShapeSettings settings = new com.github.stephengold.joltjni.StaticCompoundShapeSettings();
        chunk.children.clear();
        chunk.childById.clear();

        final int baseX = chunk.cx << 4;
        final int baseY = chunk.cy << 4;
        final int baseZ = chunk.cz << 4;
        final Vector3d translation = new Vector3d();
        boolean anyFluid = false;
        for (int bx = 0; bx < 16; bx++) {
            for (int by = 0; by < 16; by++) {
                for (int bz = 0; bz < 16; bz++) {
                    final int packed = chunk.data.get(bx, by, bz);
                    final int colliderId = ChunkSectionData.colliderIdOf(packed);
                    final JoltVoxelColliderData entry = colliderId == 0 ? null : this.colliderRegistry.get(colliderId - 1);
                    if (entry == null) {
                        continue;
                    }
                    // fluids are marked for buoyancy even though they never collide
                    anyFluid |= entry.isFluid;
                    if (!isSolidBlock(packed, entry)) {
                        continue;
                    }
                    final List<float[]> boxes = entry.boxes;
                    for (int i = 0; i < boxes.size(); i++) {
                        final float[] box = boxes.get(i);
                        JoltVoxelColliderData.boxCenter(box, translation);
                        settings.addShape(
                                new Vec3((float) (baseX + bx + translation.x),
                                        (float) (baseY + by + translation.y),
                                        (float) (baseZ + bz + translation.z)),
                                Quat.sIdentity(),
                                entry.shape(i));
                        chunk.children.add(new Child(baseX + bx, baseY + by, baseZ + bz, colliderId));
                    }
                }
            }
        }

        if (anyFluid) {
            this.fluidSections.add(ChunkSectionData.packSectionPos(chunk.cx, chunk.cy, chunk.cz));
        } else {
            this.fluidSections.remove(ChunkSectionData.packSectionPos(chunk.cx, chunk.cy, chunk.cz));
        }

        if (chunk.children.isEmpty()) {
            chunk.shape = null;
            return;
        }
        chunk.shape = settings.create().get();

        // Full sub-shape ID → child (static compound IDs are hierarchical). A
        // single-child chunk's shape IS the leaf itself; its whole-shape contact
        // sub-shape ID is the empty ID (0).
        if (chunk.children.size() == 1) {
            chunk.childById.put(0, chunk.children.get(0));
            return;
        }
        final CompoundShape compound =
                (CompoundShape) ((com.github.stephengold.joltjni.ShapeRefC) chunk.shape).getPtr();
        final SubShapeIdCreator parent = new SubShapeIdCreator();
        for (int i = 0; i < chunk.children.size(); i++) {
            chunk.childById.put(compound.getSubShapeIdFromIndex(i, parent).getId(), chunk.children.get(i));
        }
    }

    /**
     * Creates, updates or removes the static body of a global chunk so that it
     * matches the chunk's current shape and data.
     */
    private void refreshGlobalChunkBody(final GlobalChunk chunk) {
        // Staleness guard: a chunk that was replaced (addChunk) or removed
        // (removeChunk) can linger in dirtyGlobalChunks. Its body is already
        // destroyed — touching it again would double-free in native Jolt.
        if (this.globalChunks.get(ChunkSectionData.packSectionPos(chunk.cx, chunk.cy, chunk.cz)) != chunk) {
            return;
        }

        this.buildGlobalChunkShape(chunk);

        if (chunk.children.isEmpty()) {
            if (chunk.joltId != 0) {
                this.globalChunksByBodyId.remove(chunk.joltId);
                this.bi.removeBody(chunk.joltId);
                this.bi.destroyBody(chunk.joltId);
                chunk.joltId = 0;
            }
            return;
        }

        if (chunk.joltId == 0) {
            final BodyCreationSettings bcs = new BodyCreationSettings(chunk.shape, RVec3.sZero(), Quat.sIdentity(), EMotionType.Static, LAYER_STATIC);
            final Body body = this.bi.createBody(bcs);
            body.setUserData(-1L);
            this.bi.addBody(body, EActivation.DontActivate);
            chunk.joltId = body.getId();
            chunk.body = body;
            this.globalChunksByBodyId.put(chunk.joltId, chunk);
        } else {
            this.bi.setShape(chunk.joltId, chunk.shape, false, EActivation.DontActivate);
        }
    }

    /**
     * Checks whether the given world block position holds a global fluid block.
     */
    /**
     * @return the fluid fill amount (0 = no fluid, 1..9) of the global fluid
     * block at the given world block position; 9 means a full block.
     */
    int globalFluidLevelAt(final int x, final int y, final int z) {
        final long sectionKey = ChunkSectionData.packSectionPos(x >> 4, y >> 4, z >> 4);
        if (!this.fluidSections.contains(sectionKey)) {
            return 0;
        }
        final GlobalChunk chunk = this.globalChunks.get(sectionKey);
        if (chunk == null) {
            return 0;
        }
        final int colliderId = chunk.data.colliderId(x & 15, y & 15, z & 15);
        if (colliderId == 0) {
            return 0;
        }
        final JoltVoxelColliderData entry = this.colliderRegistry.get(colliderId - 1);
        if (entry == null || !entry.isFluid) {
            return 0;
        }
        final int level = ChunkSectionData.fluidLevelOf(chunk.data.get(x & 15, y & 15, z & 15));
        return level > 0 ? level : 9;
    }

    //endregion

    //region Kinematic contraptions

    public void createKinematicContraption(final int mountId, final int id, final Vector3dc pos, final Quaterniondc rot) {
        final SableBody sb = this.createBody(SableBody.Kind.CONTRAPTION, id, pos, rot, LAYER_MOVING);
        sb.mountId = mountId;
        sb.relPos.set(pos);
        sb.relRot.set(rot);

        if (mountId != -1) {
            final SableBody mount = this.bodies.get(mountId);
            if (mount != null) {
                this.assignMountGroups(mount, sb);
            }
        }
    }

    private void assignMountGroups(final SableBody mount, final SableBody contraption) {
        final int groupId;
        if (mount.groupSubId != -1) {
            groupId = this.mountGroupIds.get(mount.runtimeId);
        } else {
            mount.groupSubId = 0;
            groupId = this.nextGroupId.getAndIncrement();
            this.mountGroupIds.put(mount.runtimeId, groupId);
        final Body mountBody = mount.body;
            mountBody.setCollisionGroup(new CollisionGroup(this.mountFilterRef, groupId, 0));
        }

        int sub = 1;
        for (final SableBody other : this.bodies.values()) {
            if (other.kind == SableBody.Kind.CONTRAPTION && other.mountId == mount.runtimeId && other.groupSubId > 0) {
                sub = Math.max(sub, other.groupSubId + 1);
            }
        }

        contraption.groupSubId = sub;
        this.mountFilter.disableCollision(0, sub);
        for (final SableBody other : this.bodies.values()) {
            if (other != contraption && other.kind == SableBody.Kind.CONTRAPTION && other.mountId == mount.runtimeId) {
                this.mountFilter.disableCollision(sub, other.groupSubId);
            }
        }

        final Body body = contraption.body;
        body.setCollisionGroup(new CollisionGroup(this.mountFilterRef, groupId, sub));
    }

    private final it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap mountGroupIds = new it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap();

    /**
     * Section keys of global chunks that contain at least one fluid block; buoyancy
     * is skipped entirely when empty.
     */
    private final it.unimi.dsi.fastutil.longs.LongOpenHashSet fluidSections = new it.unimi.dsi.fastutil.longs.LongOpenHashSet();

    /**
     * Bumped whenever any chunk section data changes; used to invalidate the
     * per-body shape rebuild dedup.
     */
    private long chunkDataVersion;

    /**
     * Reusable jolt-jni wrappers for the per-substep hot paths (server thread only):
     * avoids native allocations and cleaner churn.
     */
    private JobSystem activeJobSystem;

    public void removeKinematicContraption(final int id) {
        final SableBody sb = this.bodies.get(id);
        if (sb != null) {
            this.destroyBody(sb);
        }
    }

    /**
     * Uploads a kinematic contraption's pose directly from the caller's JOML
     * vectors. All arguments are read-only and copied into the body state
     * immediately; no allocations are performed.
     */
    public void setKinematicContraptionTransform(final int id, final Vector3dc centerOfMass, final Vector3dc pos,
                                                 final Quaterniondc rot, final Vector3dc linVel, final Vector3dc angVel) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        sb.centerOfMass.set(centerOfMass);
        sb.relPos.set(pos);
        sb.relRot.set(rot);
        sb.linVel.set(linVel);
        sb.angVel.set(angVel);
        this.markDirty(sb);
    }

    public void addKinematicContraptionChunkSection(final int id, final int x, final int y, final int z, final int[] data) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        final ChunkSectionData section = new ChunkSectionData();
        System.arraycopy(data, 0, section.array(), 0, ChunkSectionData.BLOCKS);
        sb.chunks.put(ChunkSectionData.packSectionPos(x, y, z), section);
        this.markDirty(sb);
    }

    //endregion

    //region Simulation

    /**
     * Steps the simulation by one substep. Buoyancy is recomputed every substep,
     * matching the cumulative effect of the rapier implementation.
     */
    public void step(final double timeStep) {
        final long stepStartNanos = System.nanoTime();
        this.flushDirty();
        this.tickRopeAttachments();
        this.computeBuoyancy();
        this.updateContraptionMotion((float) timeStep);
        // small scenes: a single-threaded job system avoids the wake/sync overhead
        // of the thread pool, which dominates when the solver has almost no work.
        // A single body with a huge compound is NOT a small scene: its narrow phase
        // dominates the step, so heavy bodies force the multithreaded job system.
        boolean heavyBody = false;
        for (final SableBody sb : this.bodies.values()) {
            if (sb.children.size() >= HEAVY_BODY_CHILDREN) {
                heavyBody = true;
                break;
            }
        }
        final JobSystem jobs = this.bodies.size() >= MULTITHREAD_BODY_THRESHOLD || heavyBody
                ? this.jobSystem : this.singleThreadJobSystem;

        // Adaptive solver depth: the default 10 velocity / 4 position iterations are
        // tuned for small scenes. A huge compound body resting on the ground forms a
        // contact island with thousands of points; halving the iteration depth there
        // roughly halves the solver cost with no visible stability change.
        if (heavyBody != this.reducedSolverApplied) {
            final PhysicsSettings physics = this.system.getPhysicsSettings();
            physics.setNumVelocitySteps(heavyBody ? 5 : 10);
            physics.setNumPositionSteps(heavyBody ? 2 : 4);
            this.system.setPhysicsSettings(physics);
            this.reducedSolverApplied = heavyBody;
        }

        this.system.update((float) timeStep, 1, this.tempAllocator, jobs);
        if (JoltDebugLogging.STAFF) {
            this.staffWorldDump();
        }
        // Failsafe: a huge compound landing with deep penetration can collapse the
        // solver into an ever-growing contact island. If the step overran its
        // budget, put oversized bodies to sleep so the world can recover instead
        // of freezing every subsequent tick.
        final long updateNanos = System.nanoTime() - stepStartNanos;
        if (updateNanos > 250_000_000L) {            for (final SableBody sb : this.bodies.values()) {
                if (sb.children.size() >= HEAVY_BODY_CHILDREN && sb.body.isActive()) {
                    this.bi.activateBody(sb.joltId); // refresh internal bounds bookkeeping
                    if (sb.body.getLinearVelocity().lengthSq() < 0.5f) {
                        this.bi.deactivateBody(sb.joltId);
                    }
                }
            }
        }
    }

    private long stepStartNanos;

    /**
     * Splits per-body work into slices for the worker pool. The calling thread
     * processes the first slice and the method returns after all slices finish.
     */
    private void runParallel(final List<SableBody> bodies, final java.util.function.Consumer<SableBody> action) {
        final int workers = Math.min(WORKER_THREADS, bodies.size());
        if (workers <= 1) {
            for (final SableBody sb : bodies) {
                action.accept(sb);
            }
            return;
        }

        // Slices are collected first so the latch matches the number of tasks
        // actually submitted; otherwise an empty trailing slice would leave the
        // caller waiting on counts that never happen (permanent server stall).
        final int sliceSize = (bodies.size() + workers - 1) / workers;
        final java.util.List<int[]> slices = new java.util.ArrayList<>(workers - 1);
        for (int w = 1; w < workers; w++) {
            final int from = w * sliceSize;
            final int to = Math.min(bodies.size(), from + sliceSize);
            if (from >= to) {
                break;
            }
            slices.add(new int[]{from, to});
        }
        if (slices.isEmpty()) {
            for (final SableBody sb : bodies) {
                action.accept(sb);
            }
            return;
        }

        final java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(slices.size());
        for (final int[] slice : slices) {
            final List<SableBody> part = bodies.subList(slice[0], slice[1]);
            this.bodyWorkers.submit(() -> {
                try {
                    for (final SableBody sb : part) {
                        action.accept(sb);
                    }
                } catch (final Throwable t) {
                    Sable.LOGGER.error("[SableJolt] parallel body task failed", t);
                } finally {
                    latch.countDown();
                }
            });
        }
        final int headEnd = Math.min(sliceSize, bodies.size());
        for (final SableBody sb : bodies.subList(0, headEnd)) {
            action.accept(sb);
        }
        try {
            latch.await();
        } catch (final InterruptedException e) {
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
        for (final SableBody body : this.bodies.values()) {
            final RVec3 p = body.body.getCenterOfMassPosition();
            sb.append(" | #").append(body.runtimeId).append("/").append(body.kind)
                    .append(" pos=(").append((float) p.xx()).append(",").append((float) p.yy()).append(",").append((float) p.zz()).append(")")
                    .append(" children=").append(body.children.size())
                    .append(" bounds=(").append(body.minX).append(",").append(body.minY).append(",").append(body.minZ)
                    .append(")..(").append(body.maxX).append(",").append(body.maxY).append(",").append(body.maxZ).append(")")
                    .append(" com=(").append((float) body.centerOfMass.x).append(",").append((float) body.centerOfMass.y).append(",").append((float) body.centerOfMass.z).append(")")
                    .append(" bounds=").append(body.hasBounds)
                    .append(" active=").append(body.body.isActive());
        }
        Sable.LOGGER.info(sb.toString());
    }

    public void dispose() {
        this.system.removeAllConstraints();
        this.system.destroyAllBodies();
        this.bodies.clear();
        this.bodiesByJoltId.clear();
        this.globalChunks.clear();
        this.globalChunksByBodyId.clear();
        this.allChunks.clear();
        this.joints.clear();
        this.ropes.clear();
        this.dirtyBodies.clear();
        this.dirtyGlobalChunks.clear();
        this.fluidSections.clear();
        this.angularFollowPrev.clear();
        this.angularCachedOmega.clear();
        this.ropePointBodies.clear();
        this.mountGroupIds.clear();
    }

    public int nextRuntimeId() {
        return this.nextRuntimeId.getAndIncrement();
    }

    //endregion

    //region Buoyancy (ported from buoyancy.rs)

    private void computeBuoyancy() {
        if (this.fluidSections.isEmpty()) {
            return;
        }

        final ArrayList<SableBody> active = new ArrayList<>();
        for (final SableBody sb : this.bodies.values()) {
            if (sb.kind == SableBody.Kind.SUB_LEVEL && sb.hasBounds && !sb.children.isEmpty()) {
                final Body body = sb.body;
                if (body.isActive()) {
                    active.add(sb);
                }
            }
        }
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

    private void buoyancyForBody(final SableBody sb) {
        final Body body = sb.body;

        final RVec3 comPos = body.getCenterOfMassPosition();
        final double comX = comPos.xx(), comY = comPos.yy(), comZ = comPos.zz();
        final Quat rot = body.getRotation();
        final Vec3 lin = body.getLinearVelocity();
        final Vec3 ang = body.getAngularVelocity();
        final float lvx = lin.getX(), lvy = lin.getY(), lvz = lin.getZ();
        final float avx = ang.getX(), avy = ang.getY(), avz = ang.getZ();

        final RVec3 tmpPoint = this.tmpPoint.get();
        final Vec3 tmpForce = this.tmpForce.get();

        // Rotated unit axes give the world-space Y extent of every child cube:
        // an extent that stays correct (and smooth) for any body orientation.
        final Vec3 ex = rotate(1.0f, 0.0f, 0.0f, rot);
        final Vec3 ey = rotate(0.0f, 1.0f, 0.0f, rot);
        final Vec3 ez = rotate(0.0f, 0.0f, 1.0f, rot);
        final double halfY = 0.5 * (Math.abs(ex.getY()) + Math.abs(ey.getY()) + Math.abs(ez.getY()));

        // Accumulated submerged volume and its centroid: the float force is applied
        // at the centroid of the submerged part, so a tilted body gets a smooth
        // righting torque (center of buoyancy shifts toward the deep end) and
        // settles level, instead of keeping a frozen tilt or jittering.
        double floatVolume = 0.0;
        double centroidX = 0.0, centroidY = 0.0, centroidZ = 0.0;

        // Iterate the body's own collider blocks (already rebuilt and bounds-filtered
        // in sb.children) instead of rescanning chunk section data. Very large
        // bodies are subsampled with volume compensation: fluid forces are a bulk
        // effect, so a strided sample represents the whole body accurately while
        // keeping the per-tick cost bounded.
        final int childCount = sb.children.size();
        final int stride = Math.max(1, childCount / BUOYANCY_MAX_CHILDREN);
        for (int ci = 0; ci < childCount; ci += stride) {
            final Child c = sb.children.get(ci);
            final double sampleScale = stride;
            final double lpx = c.bx + 0.5 - sb.centerOfMass.x;
            final double lpy = c.by + 0.5 - sb.centerOfMass.y;
            final double lpz = c.bz + 0.5 - sb.centerOfMass.z;

            final Vec3 worldOffset = rotate((float) lpx, (float) lpy, (float) lpz, rot);
            final double wx = comX + worldOffset.getX();
            final double wy = comY + worldOffset.getY();
            final double wz = comZ + worldOffset.getZ();

            final int wbx = floor(wx);
            final int wby = floor(wy);
            final int wbz = floor(wz);
            final int fluidLevel = this.globalFluidLevelAt(wbx, wby, wbz);
            if (fluidLevel <= 0) {
                continue;
            }

            final JoltVoxelColliderData entry = this.colliderRegistry.get(c.colliderId - 1);
            final float mult = entry == null ? 1.0f : entry.volume;

            // Vertical span of the rotated unit cube vs the fluid surface: the
            // submerged height is smooth in rotation, so tilted small bodies get
            // stable forces instead of axis-aligned overlap artifacts that spin
            // them. For a centrally-symmetric cube the horizontal centroid of the
            // submerged part is always its center — no spurious yaw torque.
            final double fluidTop = wby + fluidLevel * (1.0 / 9.0);
            final double minY = wy - halfY;
            final double maxY = wy + halfY;
            final double submerged = Math.min(maxY, fluidTop) - minY;
            if (submerged <= 0.0) {
                continue;
            }
            final double volume = Math.min(1.0, submerged / (halfY * 2.0)) * mult * sampleScale;
            final double midY = Math.min(maxY, fluidTop) - submerged * 0.5;

            // drag: F = -v * 1.7 * volume at the submerged midpoint — damps both
            // linear motion and rotation while submerged
            final double rx = wx - comX, ry = midY - comY, rz = wz - comZ;
            final double vx = lvx + avy * rz - avz * ry;
            final double vy = lvy + avz * rx - avx * rz;
            final double vz = lvz + avx * ry - avy * rx;
            tmpPoint.set(wx, midY, wz);
            tmpForce.set((float) (-vx * 1.7 * volume), (float) (-vy * 1.7 * volume), (float) (-vz * 1.7 * volume));
            body.addForce(tmpForce, tmpPoint);

            // accumulate submerged volume for the single centroid float force
            floatVolume += volume;
            centroidX += wx * volume;
            centroidY += midY * volume;
            centroidZ += wz * volume;
        }

        if (floatVolume > 0.0) {
            // Float applied at the centroid of the submerged volume: zero torque
            // when level, smooth righting torque when tilted — stable flotation.
            tmpPoint.set((float) (centroidX / floatVolume),
                    (float) (centroidY / floatVolume),
                    (float) (centroidZ / floatVolume));
            final float buoyancyK = (float) (-this.gravityY * FLUID_DENSITY);
            tmpForce.set(0.0f, (float) (buoyancyK * floatVolume), 0.0f);
            body.addForce(tmpForce, tmpPoint);
        }
    }

    private static int floor(final double v) {
        return (int) Math.floor(v);
    }

    // Reusable wrappers for the buoyancy hot path; ThreadLocal because the buoyancy
    // work is distributed across worker threads.
    private final ThreadLocal<Vec3> tmpForce = ThreadLocal.withInitial(Vec3::new);
    private final ThreadLocal<RVec3> tmpPoint = ThreadLocal.withInitial(RVec3::new);
    //endregion

    //region Contraption motion

    private void updateContraptionMotion(final float dt) {
        for (final SableBody sb : this.bodies.values()) {
            if (sb.kind != SableBody.Kind.CONTRAPTION) {
                continue;
            }

            Quaterniondc parentRot = null;
            double px = 0.0, py = 0.0, pz = 0.0;
            if (sb.mountId != -1) {
                final SableBody mount = this.bodies.get(sb.mountId);
                if (mount != null) {
                    final Body mountBody = mount.body;
                    final RVec3 mp = mountBody.getCenterOfMassPosition();
                    final Quat mr = mountBody.getRotation();
                    px = mp.xx();
                    py = mp.yy();
                    pz = mp.zz();
                    parentRot = new Quaterniond(mr.getX(), mr.getY(), mr.getZ(), mr.getW());
                }
            }

            double wx;
            double wy;
            double wz;
            Quaterniond wrot;
            if (parentRot != null) {
                final org.joml.Vector3d off = parentRot.transform(new org.joml.Vector3d(sb.relPos.x, sb.relPos.y, sb.relPos.z));
                wx = px + off.x;
                wy = py + off.y;
                wz = pz + off.z;
                wrot = new Quaterniond(parentRot).mul(sb.relRot);
            } else {
                wx = sb.relPos.x;
                wy = sb.relPos.y;
                wz = sb.relPos.z;
                wrot = new Quaterniond(sb.relRot);
            }

            final org.joml.Vector3d worldLin = wrot.transform(new org.joml.Vector3d(sb.linVel));
            final org.joml.Vector3d worldAng = wrot.transform(new org.joml.Vector3d(sb.angVel));

            this.bi.moveKinematic(sb.joltId, new RVec3(wx, wy, wz),
                    new Quat((float) wrot.x, (float) wrot.y, (float) wrot.z, (float) wrot.w), dt);
            this.bi.setLinearAndAngularVelocity(sb.joltId,
                    new Vec3((float) worldLin.x, (float) worldLin.y, (float) worldLin.z),
                    new Vec3((float) worldAng.x, (float) worldAng.y, (float) worldAng.z));
        }
    }

    //endregion

    //region Collision reporting

    /**
     * Reads & clears all reported collisions.
     * Each collision is formatted as:
     * [body_a, body_b, force_amount, local_normal_a, local_normal_b, local_point_a, local_point_b]
     */
    /**
     * Reads & clears all reported collisions.
     * Each collision is formatted as:
     * [body_a, body_b, force_amount, local_normal_a, local_normal_b, local_point_a, local_point_b]
     */
    public double[] clearCollisions() {
        final int max = 100;
        final double[] arr;
        synchronized (this.reportedLock) {
            if (this.reportedCollisions.size() > max) {
                this.reportedCollisions.subList(max, this.reportedCollisions.size()).clear();
            }
            arr = new double[this.reportedCollisions.size() * 15];
            int i = 0;
            for (final double[] rec : this.reportedCollisions) {
                System.arraycopy(rec, 0, arr, i, 15);
                i += 15;
            }
            this.reportedCollisions.clear();
        }
        return arr;
    }

    void reportCollision(final double[] rec) {
        synchronized (this.reportedLock) {
            this.reportedCollisions.add(rec);
        }
    }

    JoltVoxelColliderData.Registry registry() {
        return this.colliderRegistry;
    }

    GlobalChunk globalChunkByBodyId(final int joltId) {
        return this.globalChunksByBodyId.get(joltId);
    }

    //endregion

    //region Constraints

    public TwoBodyConstraint createConstraint(final TwoBodyConstraintSettings settings, final int joltIdA, final int joltIdB) {
        final TwoBodyConstraint constraint = this.bi.createConstraint(settings, joltIdA, joltIdB);
        this.system.addConstraint(constraint);
        return constraint;
    }

    public void removeConstraint(final Constraint constraint) {
        this.system.removeConstraint(constraint);
    }

    public long registerJoint(final JointRecord record) {
        record.id = this.nextJointId.getAndIncrement();
        this.joints.put(record.id, record);
        return record.id;
    }

    @Nullable
    public JointRecord joint(final long id) {
        return this.joints.get(id);
    }

    public void removeJoint(final long id) {
        final JointRecord rec = this.joints.remove(id);
        if (rec != null && rec.constraint != null) {
            this.system.removeConstraint(rec.constraint);
            rec.constraint = null;
        }
    }

    //endregion

    //region Ropes

    private int createRopePoint(final double x, final double y, final double z, final double pointRadius) {
        final BodyCreationSettings bcs = new BodyCreationSettings(
                new com.github.stephengold.joltjni.BoxShape(new Vec3((float) pointRadius, (float) pointRadius, (float) pointRadius), 0.02f),
                new RVec3(x, y, z), Quat.sIdentity(), EMotionType.Dynamic, LAYER_ROPE);
        bcs.setLinearDamping((float) (this.universalDrag + 6.0));
        bcs.setAngularDamping((float) this.universalDrag);
        bcs.setInertiaMultiplier(0.0f);
        bcs.setFriction(0.15f);
        final MassProperties mp = new MassProperties();
        mp.setMass(0.35f);
        bcs.setOverrideMassProperties(EOverrideMassProperties.CalculateInertia);
        bcs.setMassPropertiesOverride(mp);

        final Body body = this.bi.createBody(bcs);
        body.setUserData(-1L);
        this.bi.addBody(body, EActivation.Activate);
        this.ropePointBodies.put(body.getId(), body);
        return body.getId();
    }

    private final Int2ObjectOpenHashMap<Body> ropePointBodies = new Int2ObjectOpenHashMap<>();

    private void destroyRopePoint(final int bodyId) {
        this.ropePointBodies.remove(bodyId);
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
        final RopeStrand strand = new RopeStrand();
        strand.pointRadius = pointRadius;
        strand.firstJointLength = firstJointLength;

        for (int i = 0; i < pointCount; i++) {
            strand.points.add(this.createRopePoint(points[i * 3], points[i * 3 + 1], points[i * 3 + 2], pointRadius));
        }
        for (int i = 0; i < strand.points.size() - 1; i++) {
            final double length = i == 0 ? firstJointLength : 1.0;
            strand.joints.add(this.createRopeJoint(strand.points.get(i), strand.points.get(i + 1), length));
        }

        final long id = this.nextRopeId.getAndIncrement();
        this.ropes.put(id, strand);
        return id;
    }

    public void removeRope(final long ropeId) {
        final RopeStrand strand = this.ropes.remove(ropeId);
        if (strand == null) {
            return;
        }
        if (strand.start != null) {
            this.removeConstraint(strand.start.constraint);
        }
        if (strand.end != null) {
            this.removeConstraint(strand.end.constraint);
        }
        for (final DistanceConstraint joint : strand.joints) {
            this.removeConstraint(joint);
        }
        for (final int point : strand.points) {
            this.destroyRopePoint(point);
        }
    }

    public double[] queryRope(final long ropeId) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null) {
            return new double[0];
        }
        final double[] out = new double[strand.points.size() * 3];
        for (int i = 0; i < strand.points.size(); i++) {
            final RVec3 p = this.bi.getPosition(strand.points.get(i));
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
        if (!strand.joints.isEmpty()) {
            strand.joints.get(0).setDistance(0.0f, 1.0f);
        }
        final int bodyId = this.createRopePoint(x, y, z, strand.pointRadius);
        strand.joints.add(0, this.createRopeJoint(bodyId, strand.points.get(0), strand.firstJointLength));
        strand.points.add(0, bodyId);

        if (strand.start != null) {
            this.removeConstraint(strand.start.constraint);
            strand.start = null;
        }
    }

    public void removeRopePointAtStart(final long ropeId) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null || strand.points.isEmpty()) {
            return;
        }
        final int point = strand.points.remove(0);
        if (!strand.joints.isEmpty()) {
            this.removeConstraint(strand.joints.remove(0));
        }
        this.destroyRopePoint(point);

        if (!strand.joints.isEmpty()) {
            strand.joints.get(0).setDistance(0.0f, (float) strand.firstJointLength);
        }
        if (strand.start != null) {
            this.removeConstraint(strand.start.constraint);
            strand.start = null;
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

        final int ropeBody = end ? strand.points.get(strand.points.size() - 1) : strand.points.get(0);
        final SableBody mountSb = subLevelId != -1 ? this.bodies.get(subLevelId) : null;
        final int mountBody = mountSb != null ? mountSb.joltId : this.groundBodyId;

        double comX = 0.0, comY = 0.0, comZ = 0.0;
        if (mountSb != null) {
            final RVec3 com = mountSb.body.getCenterOfMassPosition();
            comX = com.xx();
            comY = com.yy();
            comZ = com.zz();
        }

        final PointConstraintSettings settings = new PointConstraintSettings();
        settings.setPoint1(new RVec3(x - comX, y - comY, z - comZ));
        settings.setPoint2(RVec3.sZero());
        final Constraint constraint = this.createConstraint(settings, mountBody, ropeBody);

        final RopeAttachment attachment = new RopeAttachment();
        attachment.mountJoltId = mountBody;
        attachment.x = x;
        attachment.y = y;
        attachment.z = z;
        attachment.comX = comX;
        attachment.comY = comY;
        attachment.comZ = comZ;
        attachment.constraint = constraint;

        if (end) {
            if (strand.end != null) {
                this.removeConstraint(strand.end.constraint);
            }
            strand.end = attachment;
        } else {
            if (strand.start != null) {
                this.removeConstraint(strand.start.constraint);
            }
            strand.start = attachment;
        }
    }

    /**
     * Re-anchors rope attachments whose mount center of mass has shifted.
     */
    void tickRopeAttachments() {
        for (final RopeStrand strand : this.ropes.values()) {
            this.retargAttachment(strand, true);
            this.retargAttachment(strand, false);
        }
    }

    private void retargAttachment(final RopeStrand strand, final boolean end) {
        final RopeAttachment attachment = end ? strand.end : strand.start;
        if (attachment == null || attachment.mountJoltId == this.groundBodyId) {
            return;
        }

        final SableBody mount = this.bodyByJoltId(attachment.mountJoltId);
        if (mount == null) {
            return;
        }
        if (attachment.comX == mount.centerOfMass.x && attachment.comY == mount.centerOfMass.y && attachment.comZ == mount.centerOfMass.z) {
            return;
        }
        attachment.comX = mount.centerOfMass.x;
        attachment.comY = mount.centerOfMass.y;
        attachment.comZ = mount.centerOfMass.z;

        final int ropeBody = end ? strand.points.get(strand.points.size() - 1) : strand.points.get(0);

        this.removeConstraint(attachment.constraint);
        final PointConstraintSettings settings = new PointConstraintSettings();
        settings.setPoint1(new RVec3(attachment.x - mount.centerOfMass.x, attachment.y - mount.centerOfMass.y, attachment.z - mount.centerOfMass.z));
        settings.setPoint2(RVec3.sZero());
        attachment.constraint = this.createConstraint(settings, attachment.mountJoltId, ropeBody);
    }

    //endregion

    //region Config

    public void configSolverIterations(final int solverIterations, final int pgsIterations) {
        final var settings = this.system.getPhysicsSettings();
        // clamped to the minimums that keep dragged bodies from sinking into walls
        settings.setNumVelocitySteps(Math.max(10, solverIterations));
        settings.setNumPositionSteps(Math.max(4, pgsIterations));
        this.system.setPhysicsSettings(settings);
    }

    //endregion

    private static int unpackChunkX(final long key) {
        return (int) (key >> 42);
    }

    private static int unpackChunkY(final long key) {
        return (int) (key & 0xFFFFFL);
    }

    private static int unpackChunkZ(final long key) {
        return (int) ((key << 22) >> 42);
    }

    /**
     * The scene contact listener: reports collisions, applies block contact
     * callbacks, friction / restitution overrides, and kinematic surface velocities.
     */
    private final class SceneContactListener extends CustomContactListener {
        private final Vector3d tmpA = new Vector3d();
        private final Vector3d tmpB = new Vector3d();

        @Override
        public void onContactAdded(final long body1Va, final long body2Va, final long manifoldVa, final long settingsVa) {
            this.handle(body1Va, body2Va, manifoldVa, settingsVa);
        }

        @Override
        public void onContactPersisted(final long body1Va, final long body2Va, final long manifoldVa, final long settingsVa) {
            this.handle(body1Va, body2Va, manifoldVa, settingsVa);
        }

        private void handle(final long body1Va, final long body2Va, final long manifoldVa, final long settingsVa) {
            final Body body1 = new Body(body1Va);
            final Body body2 = new Body(body2Va);
            final ContactManifold manifold = new ContactManifold(manifoldVa);
            final ContactSettings settings = new ContactSettings(settingsVa);

            final SableBody sb1 = JoltPhysicsScene.this.bodyByJoltId(body1.getId());
            final SableBody sb2 = JoltPhysicsScene.this.bodyByJoltId(body2.getId());
            final GlobalChunk gc1 = sb1 == null ? JoltPhysicsScene.this.globalChunkByBodyId(body1.getId()) : null;
            final GlobalChunk gc2 = sb2 == null ? JoltPhysicsScene.this.globalChunkByBodyId(body2.getId()) : null;

            if (sb1 == null && sb2 == null && gc1 == null && gc2 == null) {
                return;
            }

            final int subShape1 = manifold.getSubShapeId1();
            final int subShape2 = manifold.getSubShapeId2();

            final Child c1 = sb1 != null ? sb1.childAt(subShape1) : gc1 != null ? childOf(gc1, subShape1) : null;
            final Child c2 = sb2 != null ? sb2.childAt(subShape2) : gc2 != null ? childOf(gc2, subShape2) : null;

            final JoltVoxelColliderData entry1 = c1 != null ? JoltPhysicsScene.this.colliderRegistry.get(c1.colliderId - 1) : null;
            final JoltVoxelColliderData entry2 = c2 != null ? JoltPhysicsScene.this.colliderRegistry.get(c2.colliderId - 1) : null;

            // friction / restitution overrides (block-level, approximating per-contact hooks)
            if (entry1 != null || entry2 != null) {
                final float mult1 = entry1 == null ? 1.0f : entry1.frictionMultiplier;
                final float mult2 = entry2 == null ? 1.0f : entry2.frictionMultiplier;
                if (mult1 != 1.0f || mult2 != 1.0f) {
                    settings.setCombinedFriction(body1.getFriction() * mult1 * mult2);
                }
                final float rest1 = entry1 == null ? 0.0f : entry1.restitution;
                final float rest2 = entry2 == null ? 0.0f : entry2.restitution;
                if (rest1 > 0.0f || rest2 > 0.0f) {
                    settings.setCombinedRestitution(Math.max(rest1, rest2));
                }
            }

            // block contact callbacks
            if (entry1 != null && entry1.contactEvents != null && gc2 == null) {
                this.invokeCallback(entry1, c1, c2, body1, manifold, settings, true);
            }
            if (entry2 != null && entry2.contactEvents != null && gc1 == null) {
                this.invokeCallback(entry2, c2, c1, body2, manifold, settings, false);
            }

            // kinematic contraption surface velocity
            this.applySurfaceVelocity(sb1, body1, manifold, settings, true);
            this.applySurfaceVelocity(sb2, body2, manifold, settings, false);

            // collision reporting
            if (c1 != null || c2 != null) {
                this.report(sb1, sb2, c1, c2, entry1, entry2, body1, body2, manifold);
            }
        }

        private static Child childOf(final GlobalChunk chunk, final int subShapeId) {
            return chunk.childById.get(subShapeId);
        }

        private void invokeCallback(final JoltVoxelColliderData entry, final Child c, final Child other,
                                    final Body body, final ContactManifold manifold, final ContactSettings settings,
                                    final boolean isFirst) {
            final Vec3 normal = manifold.getWorldSpaceNormal();
            final Vec3 tangent;
            try {
                final Vector3d point = new Vector3d(c.bx + 0.5, c.by + 0.5, c.bz + 0.5);
                final int ox = other != null ? other.bx : 0;
                final int oy = other != null ? other.by : 0;
                final int oz = other != null ? other.bz : 0;

                final double[] result = entry.contactEvents.onCollision(
                        c.bx, c.by, c.bz, ox, oy, oz,
                        point.x, point.y, point.z,
                        manifold.getPenetrationDepth(),
                        other != null);
                if (result == null || result.length < 4) {
                    return;
                }
                final Quat rot = body.getRotation();
                tangent = rotate((float) result[0], (float) result[1], (float) result[2], rot);
                final boolean remove = result[3] > 0.0;
                if (remove) {
                    settings.setIsSensor(true);
                }
            } catch (final Throwable t) {
                Sable.LOGGER.error("Block contact callback failed", t);
                return;
            }

            // relative surface velocity: body2 relative to body1
            final float sign = isFirst ? -1.0f : 1.0f;
            final Vec3 existing = settings.getRelativeLinearSurfaceVelocity();
            settings.setRelativeLinearSurfaceVelocity(new Vec3(
                    existing.getX() + sign * tangent.getX(),
                    existing.getY() + sign * tangent.getY(),
                    existing.getZ() + sign * tangent.getZ()));
        }

        private void applySurfaceVelocity(final SableBody sb, final Body body, final ContactManifold manifold, final ContactSettings settings, final boolean isFirst) {
            if (sb == null || sb.kind != SableBody.Kind.CONTRAPTION || (sb.linVel.lengthSquared() == 0 && sb.angVel.lengthSquared() == 0)) {
                return;
            }
            final Quat rot = body.getRotation();
            final Vec3 lin = rotate((float) sb.linVel.x, (float) sb.linVel.y, (float) sb.linVel.z, rot);
            final Vec3 ang = rotate((float) sb.angVel.x, (float) sb.angVel.y, (float) sb.angVel.z, rot);

            // approximate surface velocity at the body center; direction relative to body 1 / 2
            final float sign = isFirst ? -1.0f : 1.0f;
            final Vec3 existing = settings.getRelativeLinearSurfaceVelocity();
            settings.setRelativeLinearSurfaceVelocity(new Vec3(
                    existing.getX() + sign * lin.getX(),
                    existing.getY() + sign * lin.getY(),
                    existing.getZ() + sign * lin.getZ()));

            if (ang.lengthSq() > 1.0e-9f) {
                final Vec3 existingAng = settings.getRelativeAngularSurfaceVelocity();
                settings.setRelativeAngularSurfaceVelocity(new Vec3(
                        existingAng.getX() + sign * ang.getX(),
                        existingAng.getY() + sign * ang.getY(),
                        existingAng.getZ() + sign * ang.getZ()));
            }
        }

        private void report(final SableBody sb1, final SableBody sb2, final Child c1, final Child c2,
                            final JoltVoxelColliderData entry1, final JoltVoxelColliderData entry2,
                            final Body body1, final Body body2, final ContactManifold manifold) {
            final int id1 = sb1 != null ? sb1.runtimeId : -1;
            final int id2 = sb2 != null ? sb2.runtimeId : -1;

            // force estimate: normal speed * effective mass * tick rate
            final double[] rec = new double[15];
            rec[0] = id1;
            rec[1] = id2;

            final RVec3 com1 = body1.getCenterOfMassPosition();
            final RVec3 com2 = body2.getCenterOfMassPosition();
            final double com1x = com1.xx(), com1y = com1.yy(), com1z = com1.zz();
            final double com2x = com2.xx(), com2y = com2.yy(), com2z = com2.zz();

            final Vec3 n = manifold.getWorldSpaceNormal();
            final double nx = n.getX(), ny = n.getY(), nz = n.getZ();

            // contact point approximated by the involved block center
            double px1 = 0, py1 = 0, pz1 = 0;
            double px2 = 0, py2 = 0, pz2 = 0;
            if (c1 != null) {
                if (sb1 != null) {
                    px1 = c1.bx + 0.5 - sb1.centerOfMass.x;
                    py1 = c1.by + 0.5 - sb1.centerOfMass.y;
                    pz1 = c1.bz + 0.5 - sb1.centerOfMass.z;
                } else {
                    // global chunk: block coords are world-absolute; express them
                    // relative to the (static, unrotated) body position so the
                    // com1 + rotate() reconstruction below stays exact
                    px1 = c1.bx + 0.5 - com1x;
                    py1 = c1.by + 0.5 - com1y;
                    pz1 = c1.bz + 0.5 - com1z;
                }
            }
            if (c2 != null) {
                if (sb2 != null) {
                    px2 = c2.bx + 0.5 - sb2.centerOfMass.x;
                    py2 = c2.by + 0.5 - sb2.centerOfMass.y;
                    pz2 = c2.bz + 0.5 - sb2.centerOfMass.z;
                } else {
                    px2 = c2.bx + 0.5 - com2x;
                    py2 = c2.by + 0.5 - com2y;
                    pz2 = c2.bz + 0.5 - com2z;
                }
            }

            final Quat r1 = body1.getRotation();
            final Quat r2 = body2.getRotation();
            final Vec3 worldP1 = rotate((float) px1, (float) py1, (float) pz1, r1);
            final Vec3 worldP2 = rotate((float) px2, (float) py2, (float) pz2, r2);

            final double p1x = com1x + worldP1.getX(), p1y = com1y + worldP1.getY(), p1z = com1z + worldP1.getZ();
            final double p2x = com2x + worldP2.getX(), p2y = com2y + worldP2.getY(), p2z = com2z + worldP2.getZ();

            final Vec3 v1 = body1.getLinearVelocity();
            final Vec3 w1 = body1.getAngularVelocity();
            final Vec3 v2 = body2.getLinearVelocity();
            final Vec3 w2 = body2.getAngularVelocity();

            final double r1x = p1x - com1x, r1y = p1y - com1y, r1z = p1z - com1z;
            final double vp1x = v1.getX() + w1.getY() * r1z - w1.getZ() * r1y;
            final double vp1y = v1.getY() + w1.getZ() * r1x - w1.getX() * r1z;
            final double vp1z = v1.getZ() + w1.getX() * r1y - w1.getY() * r1x;

            final double r2x = p2x - com2x, r2y = p2y - com2y, r2z = p2z - com2z;
            final double vp2x = v2.getX() + w2.getY() * r2z - w2.getZ() * r2y;
            final double vp2y = v2.getY() + w2.getZ() * r2x - w2.getX() * r2z;
            final double vp2z = v2.getZ() + w2.getX() * r2y - w2.getY() * r2x;

            final double relSpeed = Math.abs((vp1x - vp2x) * nx + (vp1y - vp2y) * ny + (vp1z - vp2z) * nz);
            final float invM1 = body1.isDynamic() ? body1.getMotionProperties().getInverseMass() : 0.0f;
            final float invM2 = body2.isDynamic() ? body2.getMotionProperties().getInverseMass() : 0.0f;
            final double effMass = (invM1 + invM2) > 1.0e-9f ? 1.0 / (invM1 + invM2) : 0.0;
            final double force = relSpeed * effMass * 20.0;

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

            JoltPhysicsScene.this.reportCollision(rec);
        }
    }
}
