package dev.behindthescenery.sablejolt;

import com.github.stephengold.joltjni.Body;
import com.github.stephengold.joltjni.BodyCreationSettings;
import com.github.stephengold.joltjni.BodyInterface;
import com.github.stephengold.joltjni.Constraint;
import com.github.stephengold.joltjni.ContactListener;
import com.github.stephengold.joltjni.ContactManifold;
import com.github.stephengold.joltjni.ContactSettings;
import com.github.stephengold.joltjni.CustomContactListener;
import com.github.stephengold.joltjni.DistanceConstraint;
import com.github.stephengold.joltjni.DistanceConstraintSettings;
import com.github.stephengold.joltjni.FixedConstraintSettings;
import com.github.stephengold.joltjni.HingeConstraint;
import com.github.stephengold.joltjni.HingeConstraintSettings;
import com.github.stephengold.joltjni.JobSystem;
import com.github.stephengold.joltjni.JobSystemThreadPool;
import com.github.stephengold.joltjni.Jolt;
import com.github.stephengold.joltjni.Mat44;
import com.github.stephengold.joltjni.MassProperties;
import com.github.stephengold.joltjni.MotorSettings;
import com.github.stephengold.joltjni.MutableCompoundShape;
import com.github.stephengold.joltjni.PhysicsSystem;
import com.github.stephengold.joltjni.PointConstraintSettings;
import com.github.stephengold.joltjni.Quat;
import com.github.stephengold.joltjni.RVec3;
import com.github.stephengold.joltjni.SixDofConstraint;
import com.github.stephengold.joltjni.SixDofConstraintSettings;
import com.github.stephengold.joltjni.SpringSettings;
import com.github.stephengold.joltjni.TempAllocator;
import com.github.stephengold.joltjni.TempAllocatorMalloc;
import com.github.stephengold.joltjni.TwoBodyConstraint;
import com.github.stephengold.joltjni.TwoBodyConstraintSettings;
import com.github.stephengold.joltjni.Vec3;
import com.github.stephengold.joltjni.BroadPhaseLayerInterfaceTable;
import com.github.stephengold.joltjni.ObjectLayerPairFilterTable;
import com.github.stephengold.joltjni.ObjectVsBroadPhaseLayerFilterTable;
import com.github.stephengold.joltjni.CollisionGroup;
import com.github.stephengold.joltjni.GroupFilterTable;
import com.github.stephengold.joltjni.GroupFilterTableRef;
import com.github.stephengold.joltjni.BodyIdVector;
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
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.Nullable;
import org.joml.Quaterniond;
import org.joml.Quaterniondc;
import org.joml.Vector3d;
import org.joml.Vector3dc;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
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

    private final PhysicsSystem system;
    private final BodyInterface bi;

    public BodyInterface getBodyInterface() {
        return this.bi;
    }
    private final TempAllocator tempAllocator;
    private final JobSystem jobSystem;
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

        this.bi = this.system.getBodyInterface();
        this.tempAllocator = new TempAllocatorMalloc();
        this.jobSystem = new JobSystemThreadPool(Jolt.cMaxPhysicsJobs, Jolt.cMaxPhysicsBarriers, Math.max(1, Runtime.getRuntime().availableProcessors() - 1));

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
        public final ArrayList<Child> children = new ArrayList<>();

        public final Long2ObjectOpenHashMap<ChunkSectionData> chunks = new Long2ObjectOpenHashMap<>();

        public int minX;
        public int minY;
        public int minZ;
        public int maxX;
        public int maxY;
        public int maxZ;
        public boolean hasBounds;

        public final Vector3d centerOfMass = new Vector3d();

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
        public Child childAt(final int shapeIndex) {
            if (shapeIndex < 0 || shapeIndex >= this.children.size()) {
                return null;
            }
            return this.children.get(shapeIndex);
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

    private SableBody createBody(final SableBody.Kind kind, final int runtimeId, final double[] pose, final int layer) {
        final MutableCompoundShape shape = new MutableCompoundShape();
        final BodyCreationSettings bcs = new BodyCreationSettings(
                shape,
                new RVec3(pose[0], pose[1], pose[2]),
                new Quat((float) pose[3], (float) pose[4], (float) pose[5], (float) pose[6]),
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

    public void createSubLevel(final int id, final double[] pose) {
        this.createBody(SableBody.Kind.SUB_LEVEL, id, pose, LAYER_MOVING);
    }

    public void removeSubLevel(final int id) {
        final SableBody sb = this.bodies.get(id);
        if (sb != null) {
            this.destroyBody(sb);
        }
    }

    public void createBox(final int id, final double mass, final double hx, final double hy, final double hz, final double[] pose) {
        final SableBody sb = this.createBody(SableBody.Kind.BOX, id, pose, LAYER_MOVING);

        final BodyCreationSettings bcs = new BodyCreationSettings(
                new com.github.stephengold.joltjni.BoxShape(new Vec3((float) hx, (float) hy, (float) hz), 0.025f),
                new RVec3(pose[0], pose[1], pose[2]),
                new Quat((float) pose[3], (float) pose[4], (float) pose[5], (float) pose[6]),
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

    public void getPose(final int id, final double[] store) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        final Body body = sb.body;
        final RVec3 pos = body.getPosition();
        final Quat rot = body.getRotation();
        store[0] = pos.xx();
        store[1] = pos.yy();
        store[2] = pos.zz();
        store[3] = rot.getX();
        store[4] = rot.getY();
        store[5] = rot.getZ();
        store[6] = rot.getW();
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
        this.rebuildShape(sb);
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
        this.rebuildShape(sb);
    }

    /**
     * Overrides the mass, center of mass, and inertia tensor of a body.
     */
    public void setMassProperties(final int id, final double mass, final double[] centerOfMass, final double[] inertiaTensor) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }

        final Body body = sb.body;
        final var motion = body.getMotionProperties();
        if (motion == null) {
            return;
        }

        final float[] m = new float[16];
        m[0] = (float) inertiaTensor[0];
        m[1] = (float) inertiaTensor[3];
        m[2] = (float) inertiaTensor[6];
        m[4] = (float) inertiaTensor[1];
        m[5] = (float) inertiaTensor[4];
        m[6] = (float) inertiaTensor[7];
        m[8] = (float) inertiaTensor[2];
        m[9] = (float) inertiaTensor[5];
        m[10] = (float) inertiaTensor[8];
        m[15] = 1.0f;

        final MassProperties mp = new MassProperties();
        mp.setMass((float) mass);
        mp.setInertia(new Mat44(m));
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

    public void getLinearVelocity(final int id, final double[] store) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        final Vec3 v = sb.body.getLinearVelocity();
        store[0] = v.getX();
        store[1] = v.getY();
        store[2] = v.getZ();
    }

    public void getAngularVelocity(final int id, final double[] store) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        final Vec3 v = sb.body.getAngularVelocity();
        store[0] = v.getX();
        store[1] = v.getY();
        store[2] = v.getZ();
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
        final Vec3 impulse = rotate(new Vec3((float) fx, (float) fy, (float) fz), rot);
        final Vec3 offset = rotate(new Vec3((float) x, (float) y, (float) z), rot);
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
        body.addImpulse(rotate(new Vec3((float) fx, (float) fy, (float) fz), rot));
        body.addAngularImpulse(rotate(new Vec3((float) tx, (float) ty, (float) tz), rot));
        if (wakeUp) {
            this.bi.activateBody(sb.joltId);
        }
    }

    private static Vec3 rotate(final Vec3 v, final Quat q) {
        final float qx = q.getX(), qy = q.getY(), qz = q.getZ(), qw = q.getW();
        final float vx = v.getX(), vy = v.getY(), vz = v.getZ();
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
        final Vec3 offset = rotate(new Vec3((float) (x - sb.centerOfMass.x), (float) (y - sb.centerOfMass.y), (float) (z - sb.centerOfMass.z)),
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
        // lazily (re)build the boxes: the first computation may have run before the
        // block existed at its target position, yielding a registered-but-empty entry
        if (!entry.hasBoxes()) {
            entry.ensureBoxes(this.colliderBakery);
        }
        return entry.hasBoxes();
    }

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

        final MutableCompoundShape shape = new MutableCompoundShape();
        sb.children.clear();

        if (sb.kind == SableBody.Kind.CONTRAPTION) {
            // Contraptions own a dedicated chunk store (they have no plot java-side).
            for (final var entry : sb.chunks.long2ObjectEntrySet()) {
                final long key = entry.getLongKey();
                this.appendChunkBlocks(sb, shape, unpackChunkX(key), unpackChunkY(key), unpackChunkZ(key), entry.getValue(), true);
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
                this.appendChunkBlocks(sb, shape, cx, cy, cz, entry.getValue(), false);
            }
        }

        sb.shape = shape;
        this.bi.setShape(sb.joltId, shape, false, EActivation.DontActivate);

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

    private void appendChunkBlocks(final SableBody sb, final MutableCompoundShape shape, final int cx, final int cy, final int cz,
                                   final ChunkSectionData data, final boolean ignoreBounds) {
        final int blockMinX = cx << 4;
        final int blockMinY = cy << 4;
        final int blockMinZ = cz << 4;

        String rejectLog = null;
        final Vector3d translation = new Vector3d();
        for (int bx = 0; bx < 16; bx++) {
            for (int by = 0; by < 16; by++) {
                for (int bz = 0; bz < 16; bz++) {
                    final int packed = data.get(bx, by, bz);
                    final int colliderId = ChunkSectionData.colliderIdOf(packed);
                    final int voxelState = ChunkSectionData.voxelStateOf(packed);
                    final JoltVoxelColliderData entry = this.colliderRegistry.get(colliderId - 1);
                    final boolean solid = isSolidBlock(packed, entry);

                    if (JoltDebugLogging.STAFF && colliderId > 0 && (!solid || !sb.contains(blockMinX + bx, blockMinY + by, blockMinZ + bz))) {
                        final String reason = !solid
                                ? "state=" + voxelState + (entry == null ? " entry=null" : " boxes=" + entry.boxes.size())
                                : "outOfBounds";
                        rejectLog = (rejectLog == null ? "" : rejectLog + "; ")
                                + "block(" + (blockMinX + bx) + "," + (blockMinY + by) + "," + (blockMinZ + bz) + ") " + reason;
                    }

                    if (!solid) {
                        continue;
                    }
                    final int worldX = blockMinX + bx;
                    final int worldY = blockMinY + by;
                    final int worldZ = blockMinZ + bz;
                    if (!ignoreBounds && !sb.contains(worldX, worldY, worldZ)) {
                        continue;
                    }
                    this.appendBlock(sb, shape, worldX, worldY, worldZ, colliderId, translation);
                }
            }
        }

        if (JoltDebugLogging.STAFF && rejectLog != null) {
            Sable.LOGGER.info("[SableJolt] rebuild body {}: rejected: {}", sb.runtimeId, rejectLog);
        }
    }

    private void appendBlock(final SableBody sb, final MutableCompoundShape shape, final int bx, final int by, final int bz,
                             final int colliderId, final Vector3d translation) {
        final JoltVoxelColliderData entry = this.colliderRegistry.get(colliderId - 1);
        final List<float[]> boxes = entry.boxes;
        for (int i = 0; i < boxes.size(); i++) {
            final float[] box = boxes.get(i);
            JoltVoxelColliderData.boxCenter(box, translation);
            shape.addShape(
                    new Vec3((float) (bx + translation.x - sb.centerOfMass.x),
                            (float) (by + translation.y - sb.centerOfMass.y),
                            (float) (bz + translation.z - sb.centerOfMass.z)),
                    Quat.sIdentity(),
                    entry.shape(i));
            sb.children.add(new Child(bx, by, bz, colliderId));
        }
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
                this.rebuildShape(sb);
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
        this.dirtyBodies.add(sb);
    }

    private void flushDirty() {
        if (!this.dirtyBodies.isEmpty()) {
            for (final SableBody sb : this.dirtyBodies) {
                try {
                    this.rebuildShape(sb);
                } catch (final Throwable t) {
                    Sable.LOGGER.error("[SableJolt] failed to rebuild body {}", sb.runtimeId, t);
                }
            }
            this.dirtyBodies.clear();
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
            if (chunk != null && chunk.joltId != 0) {
                this.globalChunksByBodyId.remove(chunk.joltId);
                this.bi.removeBody(chunk.joltId);
                this.bi.destroyBody(chunk.joltId);
            }
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

        final int baseX = chunk.cx << 4;
        final int baseY = chunk.cy << 4;
        final int baseZ = chunk.cz << 4;
        final Vector3d translation = new Vector3d();
        for (int bx = 0; bx < 16; bx++) {
            for (int by = 0; by < 16; by++) {
                for (int bz = 0; bz < 16; bz++) {
                    final int packed = chunk.data.get(bx, by, bz);
                    final int colliderId = ChunkSectionData.colliderIdOf(packed);
                    if (!isSolidBlock(packed, this.colliderRegistry.get(colliderId - 1))) {
                        continue;
                    }
                    final JoltVoxelColliderData entry = this.colliderRegistry.get(colliderId - 1);
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

        if (chunk.children.isEmpty()) {
            chunk.shape = null;
            return;
        }
        chunk.shape = settings.create().get();
    }

    /**
     * Creates, updates or removes the static body of a global chunk so that it
     * matches the chunk's current shape and data.
     */
    private void refreshGlobalChunkBody(final GlobalChunk chunk) {
        this.buildGlobalChunkShape(chunk);

        if (chunk.children.isEmpty()) {
        if (chunk.joltId != 0) {
            this.globalChunksByBodyId.remove(chunk.joltId);
            this.bi.removeBody(chunk.joltId);
            this.bi.destroyBody(chunk.joltId);
            chunk.body = null;
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
    boolean isGlobalFluid(final int x, final int y, final int z) {
        final GlobalChunk chunk = this.globalChunks.get(ChunkSectionData.packSectionPos(x >> 4, y >> 4, z >> 4));
        if (chunk == null) {
            return false;
        }
        final int colliderId = chunk.data.colliderId(x & 15, y & 15, z & 15);
        if (colliderId == 0) {
            return false;
        }
        final JoltVoxelColliderData entry = this.colliderRegistry.get(colliderId - 1);
        return entry != null && entry.isFluid;
    }

    //endregion

    //region Kinematic contraptions

    public void createKinematicContraption(final int mountId, final int id, final double[] pose) {
        final SableBody sb = this.createBody(SableBody.Kind.CONTRAPTION, id, pose, LAYER_MOVING);
        sb.mountId = mountId;
        sb.relPos.set(pose[0], pose[1], pose[2]);
        sb.relRot.set(pose[3], pose[4], pose[5], pose[6]);

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
     * Bumped whenever any chunk section data changes; used to invalidate the
     * per-body shape rebuild dedup.
     */
    private long chunkDataVersion;

    public void removeKinematicContraption(final int id) {
        final SableBody sb = this.bodies.get(id);
        if (sb != null) {
            this.destroyBody(sb);
        }
    }

    public void setKinematicContraptionTransform(final int id, final double[] centerOfMass, final double[] pose, final double[] velocities) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        sb.centerOfMass.set(centerOfMass[0], centerOfMass[1], centerOfMass[2]);
        sb.relPos.set(pose[0], pose[1], pose[2]);
        sb.relRot.set(pose[3], pose[4], pose[5], pose[6]);
        sb.linVel.set(velocities[0], velocities[1], velocities[2]);
        sb.angVel.set(velocities[3], velocities[4], velocities[5]);
        this.rebuildShape(sb);
    }

    public void addKinematicContraptionChunkSection(final int id, final int x, final int y, final int z, final int[] data) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        final ChunkSectionData section = new ChunkSectionData();
        System.arraycopy(data, 0, section.array(), 0, ChunkSectionData.BLOCKS);
        sb.chunks.put(ChunkSectionData.packSectionPos(x, y, z), section);
        this.rebuildShape(sb);
    }

    //endregion

    //region Simulation

    /**
     * Steps the simulation by one substep. Buoyancy is recomputed every substep,
     * matching the cumulative effect of the rapier implementation.
     */
    public void step(final double timeStep) {
        this.flushDirty();
        this.tickRopeAttachments();
        this.computeBuoyancy();
        this.updateContraptionMotion((float) timeStep);
        this.system.update((float) timeStep, 1, this.tempAllocator, this.jobSystem);
        if (JoltDebugLogging.STAFF) {
            this.staffWorldDump();
        }
    }

    private long lastWorldDump;

    private void staffWorldDump() {
        final long now = System.currentTimeMillis();
        if (now - this.lastWorldDump < 2000) {
            return;
        }
        this.lastWorldDump = now;

        final StringBuilder sb = new StringBuilder("[SableJolt:staff] world: globalChunks=").append(this.globalChunks.size())
                .append(" allChunks=").append(this.allChunks.size())
                .append(" bodies=").append(this.bodies.size());
        for (final SableBody body : this.bodies.values()) {
            final RVec3 p = body.body.getCenterOfMassPosition();
            sb.append(" | #").append(body.runtimeId).append("/").append(body.kind)
                    .append(" pos=(").append((float) p.xx()).append(",").append((float) p.yy()).append(",").append((float) p.zz()).append(")")
                    .append(" children=").append(body.children.size())
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
        this.angularFollowPrev.clear();
        this.angularCachedOmega.clear();
    }

    public int nextRuntimeId() {
        return this.nextRuntimeId.getAndIncrement();
    }

    //endregion

    //region Buoyancy (ported from buoyancy.rs)

    private void computeBuoyancy() {
        if (this.globalChunks.isEmpty()) {
            return;
        }

        for (final SableBody sb : this.bodies.values()) {
            if (sb.kind != SableBody.Kind.SUB_LEVEL || !sb.hasBounds || sb.children.isEmpty()) {
                continue;
            }

            final Body body = sb.body;
            if (!body.isActive()) {
                continue;
            }

            final RVec3 comPos = body.getCenterOfMassPosition();
            final double comX = comPos.xx(), comY = comPos.yy(), comZ = comPos.zz();
            final Quat rot = body.getRotation();
            final Vec3 lin = body.getLinearVelocity();
            final Vec3 ang = body.getAngularVelocity();
            final float lvx = lin.getX(), lvy = lin.getY(), lvz = lin.getZ();
            final float avx = ang.getX(), avy = ang.getY(), avz = ang.getZ();

            final int sizeSum = (sb.maxX - sb.minX) + (sb.maxY - sb.minY) + (sb.maxZ - sb.minZ);
            final boolean complex = sizeSum < 10;

            // Iterate the body's own collider blocks (already rebuilt and bounds-filtered
            // in sb.children) instead of rescanning chunk section data.
            for (final Child c : sb.children) {
                final double lpx = c.bx + 0.5 - sb.centerOfMass.x;
                final double lpy = c.by + 0.5 - sb.centerOfMass.y;
                final double lpz = c.bz + 0.5 - sb.centerOfMass.z;

                final Vec3 local = new Vec3((float) lpx, (float) lpy, (float) lpz);
                final Vec3 worldOffset = rotate(local, rot);
                final double wx = comX + worldOffset.getX();
                final double wy = comY + worldOffset.getY();
                final double wz = comZ + worldOffset.getZ();

                final int wbx = floor(wx);
                final int wby = floor(wy);
                final int wbz = floor(wz);
                if (!this.isGlobalFluid(wbx, wby, wbz)) {
                    continue;
                }

                final JoltVoxelColliderData entry = this.colliderRegistry.get(c.colliderId - 1);

                if (complex) {
                    for (int i = 0; i < 8; i++) {
                        final double ox = ((i & 1) * 2 - 1) * 0.25;
                        final double oy = (((i >> 1) & 1) * 2 - 1) * 0.25;
                        final double oz = (((i >> 2) & 1) * 2 - 1) * 0.25;
                        this.buoyancySample(body, sb, wbx, wby, wbz,
                                wx + ox, wy + oy, wz + oz, 0.25,
                                lvx, lvy, lvz, avx, avy, avz, comX, comY, comZ,
                                entry == null ? 1.0f : entry.volume);
                    }
                } else {
                    this.buoyancySample(body, sb, wbx, wby, wbz,
                            wx, wy, wz, 0.5,
                            lvx, lvy, lvz, avx, avy, avz, comX, comY, comZ,
                            entry == null ? 1.0f : entry.volume);
                }
            }
        }
    }

    private static int floor(final double v) {
        return (int) Math.floor(v);
    }

    private void buoyancySample(final Body body, final SableBody sb, final int wbx, final int wby, final int wbz,
                                final double px, final double py, final double pz, final double half,
                                final float lvx, final float lvy, final float lvz,
                                final float avx, final float avy, final float avz,
                                final double comX, final double comY, final double comZ,
                                final float fluidVolume) {
        // overlap volume between the cube around the sample point and the unit cube of the fluid block
        final double oxMin = Math.max(px - half, wbx);
        final double oyMin = Math.max(py - half, wby);
        final double ozMin = Math.max(pz - half, wbz);
        final double oxMax = Math.min(px + half, wbx + 1.0);
        final double oyMax = Math.min(py + half, wby + 1.0);
        final double ozMax = Math.min(pz + half, wbz + 1.0);
        final double volume = Math.max(0.0, oxMax - oxMin) * Math.max(0.0, oyMax - oyMin) * Math.max(0.0, ozMax - ozMin);
        if (volume <= 0.0) {
            return;
        }

        final RVec3 point = new RVec3(px, py, pz);

        // drag: F = -v * 1.7 * volume
        final double rx = px - comX, ry = py - comY, rz = pz - comZ;
        final double vx = lvx + avy * rz - avz * ry;
        final double vy = lvy + avz * rx - avx * rz;
        final double vz = lvz + avx * ry - avy * rx;
        body.addForce(new Vec3((float) (-vx * 1.7 * volume), (float) (-vy * 1.7 * volume), (float) (-vz * 1.7 * volume)), point);

        // float: F = (0, 10.5 * volume * fluidVolume, 0)
        body.addForce(new Vec3(0.0f, (float) (10.5 * volume * fluidVolume), 0.0f), point);
    }

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
    public double[] clearCollisions() {
        final int max = 100;
        if (this.reportedCollisions.size() > max) {
            this.reportedCollisions.subList(max, this.reportedCollisions.size()).clear();
        }

        final double[] arr = new double[this.reportedCollisions.size() * 15];
        int i = 0;
        for (final double[] rec : this.reportedCollisions) {
            System.arraycopy(rec, 0, arr, i, 15);
            i += 15;
        }
        this.reportedCollisions.clear();
        return arr;
    }

    void reportCollision(final double[] rec) {
        this.reportedCollisions.add(rec);
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
        settings.setNumVelocitySteps(Math.max(1, solverIterations));
        settings.setNumPositionSteps(Math.max(1, pgsIterations));
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

        private static Child childOf(final GlobalChunk chunk, final int shapeIndex) {
            if (shapeIndex < 0 || shapeIndex >= chunk.children.size()) {
                return null;
            }
            return chunk.children.get(shapeIndex);
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
                final Vec3 localTangent = new Vec3((float) result[0], (float) result[1], (float) result[2]);
                final Quat rot = body.getRotation();
                tangent = rotate(localTangent, rot);
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
            final Vec3 lin = rotate(new Vec3((float) sb.linVel.x, (float) sb.linVel.y, (float) sb.linVel.z), rot);
            final Vec3 ang = rotate(new Vec3((float) sb.angVel.x, (float) sb.angVel.y, (float) sb.angVel.z), rot);

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
                final SableBody sb = sb1;
                px1 = c1.bx + 0.5 - sb.centerOfMass.x;
                py1 = c1.by + 0.5 - sb.centerOfMass.y;
                pz1 = c1.bz + 0.5 - sb.centerOfMass.z;
            }
            if (c2 != null) {
                final SableBody sb = sb2;
                px2 = c2.bx + 0.5 - sb.centerOfMass.x;
                py2 = c2.by + 0.5 - sb.centerOfMass.y;
                pz2 = c2.bz + 0.5 - sb.centerOfMass.z;
            }

            final Quat r1 = body1.getRotation();
            final Quat r2 = body2.getRotation();
            final Vec3 worldP1 = rotate(new Vec3((float) px1, (float) py1, (float) pz1), r1);
            final Vec3 worldP2 = rotate(new Vec3((float) px2, (float) py2, (float) pz2), r2);

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
