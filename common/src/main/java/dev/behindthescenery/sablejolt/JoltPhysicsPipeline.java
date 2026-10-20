package dev.behindthescenery.sablejolt;

import dev.behindthescenery.sablejolt.box.JoltBoxHandle;
import dev.behindthescenery.sablejolt.collider.JoltVoxelColliderBakery;
import dev.behindthescenery.sablejolt.collider.JoltVoxelColliderData;
import dev.behindthescenery.sablejolt.constraint.fixed.JoltFixedConstraintHandle;
import dev.behindthescenery.sablejolt.constraint.free.JoltFreeConstraintHandle;
import dev.behindthescenery.sablejolt.constraint.generic.JoltGenericConstraintHandle;
import dev.behindthescenery.sablejolt.constraint.rotary.JoltRotaryConstraintHandle;
import dev.behindthescenery.sablejolt.rope.JoltRopeHandle;
import dev.ryanhcode.sable.Sable;
import dev.ryanhcode.sable.api.physics.PhysicsPipeline;
import dev.ryanhcode.sable.api.physics.PhysicsPipelineBody;
import dev.ryanhcode.sable.api.physics.constraint.*;
import dev.ryanhcode.sable.api.physics.mass.MassData;
import dev.ryanhcode.sable.api.physics.mass.MassTracker;
import dev.ryanhcode.sable.api.physics.object.box.BoxHandle;
import dev.ryanhcode.sable.api.physics.object.box.BoxPhysicsObject;
import dev.ryanhcode.sable.api.physics.object.rope.RopeHandle;
import dev.ryanhcode.sable.api.physics.object.rope.RopePhysicsObject;
import dev.ryanhcode.sable.api.sublevel.KinematicContraption;
import dev.ryanhcode.sable.api.sublevel.ServerSubLevelContainer;
import dev.ryanhcode.sable.api.sublevel.SubLevelContainer;
import dev.ryanhcode.sable.companion.math.BoundingBox3dc;
import dev.ryanhcode.sable.companion.math.BoundingBox3i;
import dev.ryanhcode.sable.companion.math.BoundingBox3ic;
import dev.ryanhcode.sable.companion.math.Pose3d;
import dev.ryanhcode.sable.companion.math.Pose3dc;
import dev.ryanhcode.sable.physics.chunk.VoxelNeighborhoodState;
import dev.ryanhcode.sable.physics.config.PhysicsConfigData;
import dev.ryanhcode.sable.sublevel.ServerSubLevel;
import dev.ryanhcode.sable.sublevel.SubLevel;
import dev.ryanhcode.sable.sublevel.plot.LevelPlot;
import dev.ryanhcode.sable.sublevel.system.SubLevelPhysicsSystem;
import dev.ryanhcode.sable.util.LevelAccelerator;
import dev.ryanhcode.sable.util.SableMathUtils;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2LongMap;
import it.unimi.dsi.fastutil.longs.Long2LongOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.*;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.SectionPos;
import net.minecraft.core.particles.BlockParticleOption;
import net.minecraft.core.particles.ParticleTypes;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.sounds.SoundSource;
import net.minecraft.util.Mth;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.SoundType;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.LevelChunk;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.material.FluidState;
import net.minecraft.world.level.material.Fluids;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.joml.Matrix3dc;
import org.joml.Quaterniond;
import org.joml.Quaterniondc;
import org.joml.Vector3d;
import org.joml.Vector3dc;

/**
 * Implementation of {@link PhysicsPipeline} for the Jolt physics engine.
 * <p>
 * Sable persists sub-levels (plot blocks, pose, velocities) itself, in a format
 * independent of the physics engine; this pipeline keeps the exact conventions
 * of the Rapier pipeline (pose = center of mass position + orientation,
 * world-space velocities, impulses in the body frame), so worlds move between
 * the two engines in both directions without converting anything.
 */
public class JoltPhysicsPipeline implements PhysicsPipeline {

    private static final Direction[] DIRECTIONS = Direction.values();

    /**
     * Distance threshold for uploading sub-contraptions to the physics pipeline
     */
    private static final double DISTANCE_THRESHOLD = 1e-7;

    /**
     * Angle threshold for uploading sub-contraptions to the physics pipeline
     */
    private static final double ANGULAR_THRESHOLD = 1e-7;

    /**
     * Ticks a terrain section uploaded ahead of Sable's chunk tickets stays when
     * Sable does not take it over.
     */
    private static final long PROVISIONAL_SECTION_TICKS = 100L;

    private final ServerLevel level;
    private final LevelAccelerator accelerator;
    // Open hash map: per-collision get() calls must not linearly scan.
    private final Int2ObjectMap<ServerSubLevel> activeSubLevels = new Int2ObjectOpenHashMap<>();
    private final Object2ObjectMap<KinematicContraption, TrackedKinematicContraption> activeContraptions = new Object2ObjectOpenHashMap<>();
    private final Long2LongOpenHashMap recentCollisions = new Long2LongOpenHashMap();
    private final ReferenceList<PhysicsPipelineBody> queuedWakeUps = new ReferenceArrayList<>();
    private final JoltPhysicsScene.PoseCache poseCache;
    /**
     * Sub-levels added since the last step whose terrain is uploaded before
     * they first simulate.
     */
    private final ObjectArrayList<ServerSubLevel> pendingTerrainWarmup = new ObjectArrayList<>();
    /**
     * Terrain sections uploaded ahead of Sable's chunk tickets → expiry tick.
     * Sable adopts them (handleChunkSectionAddition) or they expire.
     */
    private final Long2LongOpenHashMap provisionalSections = new Long2LongOpenHashMap();
    private JoltPhysicsScene scene;
    private JoltVoxelColliderBakery colliderBakery;

    public JoltPhysicsPipeline(final ServerLevel level) {
        this.level = level;
        this.accelerator = new LoadedChunksAccelerator(level);
        this.poseCache = new JoltPhysicsScene.PoseCache();
        // First-time puts must report "absent": the effect code checks
        // put(...) != -1, and the fastutil default return is 0 (without this,
        // every first occurrence would look like a repeat and be skipped).
        this.recentCollisions.defaultReturnValue(-1L);
    }

    /**
     * Packs a voxel collider ID and neighborhood state into an integer the pipeline will re-interpret as a block-state.
     */
    private static int packBlockState(final VoxelNeighborhoodState state, final int colliderID, final int fluidLevel) {
        return ((int) state.byteRepresentation() & 0xFF) | ((fluidLevel & 0xF) << 8) | (colliderID << 16);
    }

    private JoltPhysicsScene scene() {
        if (this.scene == null) {
            throw new IllegalStateException("Physics scene is not initialized");
        }
        return this.scene;
    }

    private JoltVoxelColliderBakery bakery() {
        if (this.colliderBakery == null) {
            throw new IllegalStateException("Physics scene is not initialized");
        }
        return this.colliderBakery;
    }

    /**
     * Initializes the physics pipeline.
     */
    @Override
    public void init(@Nullable final Vector3dc gravity, final double universalDrag) {
        final double gx = gravity != null ? gravity.x() : 0.0;
        final double gy = gravity != null ? gravity.y() : -9.81;
        final double gz = gravity != null ? gravity.z() : 0.0;
        this.scene = new JoltPhysicsScene(gx, gy, gz, universalDrag);
        this.colliderBakery = new JoltVoxelColliderBakery(this.level, this.scene.colliderRegistry);
        this.scene.attachColliderBakery(this.colliderBakery);
    }

    /**
     * Disposes all resources used by the physics pipeline.
     */
    @Override
    public void dispose() {
        // The disposed scene stays referenced: Create contraptions and handles
        // of other mods may still call in while the world closes, and every
        // scene operation is a no-op after disposal.
        if (this.scene != null) {
            this.scene.dispose();
        }
        this.activeSubLevels.clear();
        this.activeContraptions.clear();
        this.queuedWakeUps.clear();
        this.pendingTerrainWarmup.clear();
        this.provisionalSections.clear();
    }

    /**
     * Runs once before the physics substeps.
     */
    @Override
    public void prePhysicsTicks() {
        final JoltPhysicsScene scene = this.scene;
        if (scene == null) {
            return;
        }
        if (!this.pendingTerrainWarmup.isEmpty()) {
            this.warmUpTerrain();
        }
        scene.prePhysicsTick();
    }

    /**
     * Runs a physics substep with a time step of {@code 1.0 / 20.0 / substeps} seconds.
     */
    @Override
    public void physicsTick(final double timeStep) {
        final JoltPhysicsScene scene = this.scene;
        if (scene == null) {
            return;
        }
        this.updateContraptionPoses();
        scene.step(timeStep);
        // Block contact callbacks read and edit the world: they run here, on the
        // server thread, after the native update (never on a solver thread).
        scene.processDeferredCallbacks();

        if (!this.queuedWakeUps.isEmpty()) {
            for (int i = 0, n = this.queuedWakeUps.size(); i < n; i++) {
                final PhysicsPipelineBody queuedWakeUp = this.queuedWakeUps.get(i);
                if (queuedWakeUp.isRemoved()) {
                    continue;
                }
                scene.wakeUpObject(queuedWakeUp.getRuntimeId());
            }
            this.queuedWakeUps.clear();
        }
    }

    /**
     * Called after all physics substeps have been run, to finalize the physics tick.
     */
    @Override
    public void postPhysicsTicks() {
        if (this.scene == null) {
            return;
        }
        this.processCollisionEffects();
        this.expireProvisionalSections();
    }

    /**
     * Runs a tick to update any separate sub-level tracking / logic, even if physics is currently paused
     */
    @Override
    public void tick() {
        this.accelerator.clearCache();
    }

    /**
     * Adds a {@link SubLevel} to the physics pipeline.
     */
    @Override
    public void add(final ServerSubLevel subLevel, final Pose3dc pose) {
        this.assertBodyValid(subLevel);
        final JoltPhysicsScene scene = this.scene;
        if (scene == null) {
            return;
        }
        final Vector3dc pos = pose.position();
        final Quaterniondc rot = pose.orientation();

        final int id = subLevel.getRuntimeId();
        scene.createSubLevel(id, pos, rot);
        this.activeSubLevels.put(id, subLevel);

        // Terrain under the sub-level is uploaded before its first step, from
        // already loaded chunks only (Sable's chunk tickets take over next tick).
        this.pendingTerrainWarmup.add(subLevel);

        subLevel.updateMergedMassData(1.0f);
        final Vector3dc centerOfMass = subLevel.getMassTracker().getCenterOfMass();

        if (centerOfMass != null) {
            subLevel.logicalPose().rotationPoint().set(centerOfMass);
            this.onStatsChanged(subLevel);
        }
    }

    /**
     * Uploads the world sections a newly added sub-level overlaps (its world
     * bounds plus one block, like Sable's chunk tickets) that are not in the
     * scene yet. Only already loaded chunks are read: a synchronous chunk load
     * here can generate terrain on the server thread and re-enter Sable.
     */
    private void warmUpTerrain() {
        final ServerSubLevelContainer container = ServerSubLevelContainer.getContainer(this.level);
        final JoltPhysicsScene scene = this.scene;
        if (container == null || scene == null) {
            this.pendingTerrainWarmup.clear();
            return;
        }
        final long expiry = this.level.getGameTime() + PROVISIONAL_SECTION_TICKS;
        final int minSectionY = this.level.getMinSection();
        final int maxSectionY = this.level.getMaxSection() - 1;
        for (final ServerSubLevel subLevel : this.pendingTerrainWarmup) {
            if (subLevel.isRemoved()) {
                continue;
            }
            final BoundingBox3dc bounds = subLevel.boundingBox();
            if (bounds == null || !Double.isFinite(bounds.minX()) || !Double.isFinite(bounds.maxX())) {
                continue;
            }
            final int minX = Mth.floor(bounds.minX() - 1.0) >> 4, maxX = Mth.floor(bounds.maxX() + 1.0) >> 4;
            final int minY = Math.max(minSectionY, Mth.floor(bounds.minY() - 1.0) >> 4);
            final int maxY = Math.min(maxSectionY, Mth.floor(bounds.maxY() + 1.0) >> 4);
            final int minZ = Mth.floor(bounds.minZ() - 1.0) >> 4, maxZ = Mth.floor(bounds.maxZ() + 1.0) >> 4;
            // Degenerate (or absurdly large) bounds: leave it to Sable's tickets.
            if ((long) (maxX - minX + 1) * (maxY - minY + 1) * (maxZ - minZ + 1) > 4096) {
                continue;
            }
            for (int cx = minX; cx <= maxX; cx++) {
                for (int cz = minZ; cz <= maxZ; cz++) {
                    if (container.inBounds(cx, cz)) {
                        continue;
                    }
                    final LevelChunk chunk = this.level.getChunkSource().getChunkNow(cx, cz);
                    if (chunk == null) {
                        continue;
                    }
                    for (int cy = minY; cy <= maxY; cy++) {
                        if (scene.hasChunk(cx, cy, cz)) {
                            continue;
                        }
                        final int index = this.level.getSectionIndexFromSectionY(cy);
                        if (index < 0 || index >= chunk.getSectionsCount()) {
                            continue;
                        }
                        this.uploadSection(chunk.getSection(index), chunk, cx, cy, cz, true, -1);
                        this.provisionalSections.put(ChunkSectionData.packSectionPos(cx, cy, cz), expiry);
                    }
                }
            }
        }
        this.pendingTerrainWarmup.clear();
    }

    private void expireProvisionalSections() {
        if (this.provisionalSections.isEmpty() || (this.level.getGameTime() % 20L) != 0L) {
            return;
        }
        final long now = this.level.getGameTime();
        final ObjectIterator<Long2LongMap.Entry> it = this.provisionalSections.long2LongEntrySet().iterator();
        while (it.hasNext()) {
            final Long2LongMap.Entry entry = it.next();
            if (entry.getLongValue() > now) {
                continue;
            }
            final long key = entry.getLongKey();
            it.remove();
            this.scene().removeChunk(unpackX(key), unpackY(key), unpackZ(key), true);
        }
    }

    private static int unpackX(final long key) {
        return (int) (key >> 42);
    }

    private static int unpackY(final long key) {
        return (int) ((key << 44) >> 44);
    }

    private static int unpackZ(final long key) {
        return (int) ((key << 22) >> 42);
    }

    /**
     * Removes a {@link SubLevel} from the physics pipeline.
     */
    @Override
    public void remove(final ServerSubLevel subLevel) {
        final int id = subLevel.getRuntimeId();
        this.activeSubLevels.remove(id);
        this.pendingTerrainWarmup.remove(subLevel);
        if (this.scene != null) {
            this.scene.removeSubLevel(id);
        }
    }

    /**
     * Adds a kinematic contraption to the scene
     */
    @Override
    public void add(final KinematicContraption contraption) {
        if (this.activeContraptions.containsKey(contraption)) {
            throw new IllegalStateException("Contraption " + contraption + " is already present in pipeline");
        }
        final JoltPhysicsScene scene = this.scene;
        if (scene == null) {
            return;
        }
        final JoltVoxelColliderBakery bakery = this.bakery();

        final int id = this.getNextRuntimeID();
        this.activeContraptions.put(contraption, new TrackedKinematicContraption(new Vector3d(), new Quaterniond(), new Vector3d(), new Vector3d(), id));

        // The mount is fixed at creation, like the rapier pipeline.
        final SubLevel mountSubLevel = Sable.HELPER.getContaining(this.level, contraption.sable$getPosition());
        final int mountId = mountSubLevel instanceof final ServerSubLevel serverMount ? serverMount.getRuntimeId() : -1;

        final BoundingBox3i localBounds = new BoundingBox3i();
        contraption.sable$getLocalBounds(localBounds);

        final Vector3dc pos = contraption.sable$getPosition();
        final Quaterniond rot = contraption.sable$getOrientation();

        scene.createKinematicContraption(mountId, id, pos, rot);

        final Long2ObjectMap<int[]> chunks = new Long2ObjectOpenHashMap<>();

        final BlockPos.MutableBlockPos blockPos = new BlockPos.MutableBlockPos();
        final Object2IntOpenHashMap<BlockState> resolveCache = newResolveCache();
        // Sections repeat across the iteration order; bypass the map lookup
        // while the key stays put.
        long lastSectionKey = Long.MIN_VALUE;
        int[] lastChunk = null;
        for (int x = localBounds.minX(); x <= localBounds.maxX(); x++) {
            for (int z = localBounds.minZ(); z <= localBounds.maxZ(); z++) {
                for (int y = localBounds.minY(); y <= localBounds.maxY(); y++) {
                    final BlockState blockState = contraption.sable$blockGetter().getBlockState(blockPos.set(x, y, z));

                    if (blockState.isAir()) continue;

                    final long sectionKey = SectionPos.asLong(x >> 4, y >> 4, z >> 4);
                    final int[] chunk;
                    if (sectionKey == lastSectionKey) {
                        chunk = lastChunk;
                    } else {
                        chunk = chunks.computeIfAbsent(sectionKey, longPos -> new int[LevelChunkSection.SECTION_SIZE]);
                        lastSectionKey = sectionKey;
                        lastChunk = chunk;
                    }

                    final int index = (x & 15) + ((z & 15) << 4) + ((y & 15) << 8);
                    final int resolved = resolveVoxel(resolveCache, bakery, blockState, true);
                    chunk[index] = packBlockState(VoxelNeighborhoodState.CORNER, resolved >>> 4, resolved & 0xF);
                }
            }
        }

        if (contraption.sable$shouldCollide()) {
            for (final Long2ObjectMap.Entry<int[]> entry : chunks.long2ObjectEntrySet()) {
                final SectionPos sectionPos = SectionPos.of(entry.getLongKey());
                scene.addKinematicContraptionChunkSection(id, sectionPos.x(), sectionPos.y(), sectionPos.z(), entry.getValue());
            }
        }

        this.updateContraptionPose(contraption, 1.0f);
        scene.setLocalBounds(id, localBounds.minX(), localBounds.minY(), localBounds.minZ(), localBounds.maxX(), localBounds.maxY(), localBounds.maxZ());
    }

    private static Object2IntOpenHashMap<BlockState> newResolveCache() {
        final Object2IntOpenHashMap<BlockState> cache = new Object2IntOpenHashMap<>(16);
        cache.defaultReturnValue(-1);
        return cache;
    }

    /**
     * Resolves the per-state part of a voxel word (collider id + fluid level),
     * caching by {@code BlockState}: a section holds thousands of voxels but
     * only a handful of distinct states. Returns
     * {@code (colliderValue << 4) | fluidLevel}.
     *
     * @param solid whether the voxel is solid for collision; a solid block whose
     *              collider has no boxes yet gets them (re)computed here, on the
     *              server thread, never during shape rebuilds
     */
    private static int resolveVoxel(final Object2IntOpenHashMap<BlockState> cache,
                                    final JoltVoxelColliderBakery bakery, final BlockState blockState, final boolean solid) {
        int cached = cache.getInt(blockState);
        if (cached == -1) {
            final JoltVoxelColliderData data = bakery.getPhysicsDataForBlock(blockState);
            if (solid && data != null && !data.isFluid && !data.hasBoxes()) {
                data.ensureBoxes(bakery);
            }
            final int colliderValue = data == null ? 0 : data.colliderHandle + 1;
            cached = (colliderValue << 4) | (blockState.getFluidState().getAmount() & 0xF);
            cache.put(blockState, cached);
        }
        return cached;
    }

    private static boolean isSolidState(final VoxelNeighborhoodState state) {
        return state == VoxelNeighborhoodState.FACE || state == VoxelNeighborhoodState.EDGE || state == VoxelNeighborhoodState.CORNER;
    }

    /**
     * Removes a kinematic contraption from the scene
     */
    @Override
    public void remove(final KinematicContraption contraption) {
        final TrackedKinematicContraption removed = this.activeContraptions.remove(contraption);

        if (removed == null || this.scene == null) {
            return;
        }

        this.scene.removeKinematicContraption(removed.id());
    }

    /**
     * Queries the physics pipeline for the current pose of a {@link SubLevel}.
     */
    @Override
    public Pose3d readPose(final ServerSubLevel subLevel, final Pose3d dest) {
        this.assertBodyValid(subLevel);
        final JoltPhysicsScene scene = this.scene;
        if (scene == null || scene.body(subLevel.getRuntimeId()) == null) {
            // Not simulated (yet): report Sable's own pose unchanged.
            dest.position().set(subLevel.logicalPose().position());
            dest.orientation().set(subLevel.logicalPose().orientation());
            return dest;
        }
        scene.getPose(subLevel.getRuntimeId(), this.poseCache);

        dest.position().set(this.poseCache.p1, this.poseCache.p2, this.poseCache.p3);
        dest.orientation().set(this.poseCache.p4, this.poseCache.p5, this.poseCache.p6, this.poseCache.p7);

        return dest;
    }

    /**
     * Adds a rope to the physics pipeline
     */
    @Override
    public RopeHandle addRope(final RopePhysicsObject rope) {
        return JoltRopeHandle.create(this.scene(), rope.getCollisionRadius(), rope.getPoints());
    }

    /**
     * Adds a box to the physics pipeline
     */
    @Override
    public BoxHandle addBox(final BoxPhysicsObject box) {
        return JoltBoxHandle.create(this.scene(), box.getPose(), box.getHalfExtents(), box.getMass());
    }

    /**
     * Handles the addition of a chunk section to the physics context
     */
    @Override
    public void handleChunkSectionAddition(final LevelChunkSection section, final int x, final int y, final int z, final boolean uploadDataIfGlobal) {
        if (this.scene == null) {
            return;
        }
        this.accelerator.clearCache();
        final LevelChunk chunk = section.hasOnlyAir() ? null : this.accelerator.getChunk(x, z);

        final SubLevelContainer container = SubLevelContainer.getContainer(this.level);
        final LevelPlot plot = container == null ? null : container.getPlot(x, z);
        final boolean global = plot == null;
        int ownerId = -1;
        if (plot != null && plot.getSubLevel() instanceof final ServerSubLevel owner) {
            ownerId = owner.getRuntimeId();
        }
        if (global) {
            // Sable now tracks this section with a ticket and will remove it.
            this.provisionalSections.remove(ChunkSectionData.packSectionPos(x, y, z));
        }
        this.uploadSection(section, chunk, x, y, z, global, ownerId);
    }

    /**
     * Converts a level section into voxel words and stores it in the scene.
     */
    private void uploadSection(final LevelChunkSection section, @Nullable final LevelChunk chunk,
                               final int x, final int y, final int z, final boolean global, final int ownerId) {
        final int[] array = new int[LevelChunkSection.SECTION_SIZE];

        if (!section.hasOnlyAir() && chunk != null) {
            final SectionPos sectionPos = SectionPos.of(x, y, z);
            final int baseX = sectionPos.minBlockX();
            final int baseY = sectionPos.minBlockY();
            final int baseZ = sectionPos.minBlockZ();
            final BlockPos.MutableBlockPos globalPos = new BlockPos.MutableBlockPos();
            // Distinct block states per section are a handful; the collider +
            // fluid resolve is cached per state instead of per voxel.
            final Object2IntOpenHashMap<BlockState> resolveCache = newResolveCache();
            final JoltVoxelColliderBakery bakery = this.bakery();

            for (int bx = 0; bx < 16; bx++) {
                for (int bz = 0; bz < 16; bz++) {
                    for (int by = 0; by < 16; by++) {
                        // Section-local read; air needs no neighborhood scan.
                        final BlockState blockState = section.getBlockState(bx, by, bz);
                        if (blockState.isAir()) {
                            continue;
                        }
                        globalPos.set(bx + baseX, by + baseY, bz + baseZ);
                        final VoxelNeighborhoodState state = VoxelNeighborhoodState.getState(this.accelerator, globalPos, chunk);

                        final int index = bx + (bz << 4) + (by << 8);
                        final int resolved = resolveVoxel(resolveCache, bakery, blockState, isSolidState(state));
                        array[index] = packBlockState(state, resolved >>> 4, resolved & 0xF);
                    }
                }
            }
        }

        this.scene().addChunk(x, y, z, array, global, ownerId);
    }

    /**
     * Handles the removal of a chunk section from the physics context
     */
    @Override
    public void handleChunkSectionRemoval(final int x, final int y, final int z) {
        if (this.scene == null) {
            return;
        }
        final SubLevelContainer container = SubLevelContainer.getContainer(this.level);
        this.provisionalSections.remove(ChunkSectionData.packSectionPos(x, y, z));
        this.scene.removeChunk(x, y, z, container == null || !container.inBounds(x, z));
    }

    /**
     * Handles the change of a block (from oldState to newState) in a chunk at
     * chunk-relative position x, y, z. The block and its six neighbors (whose
     * neighborhood state may change) are sent to the scene in one batch, each
     * tagged with the sub-level owning its plot (or -1 for the world).
     */
    @Override
    public void handleBlockChange(final SectionPos sectionPos, final LevelChunkSection chunk,
                                  final int localX, final int localY, final int localZ,
                                  final BlockState oldState, final BlockState newState) {
        final JoltPhysicsScene scene = this.scene;
        if (scene == null) {
            return;
        }

        final int worldX = (sectionPos.x() << 4) | (localX & 0xF);
        final int worldY = (sectionPos.y() << 4) | (localY & 0xF);
        final int worldZ = (sectionPos.z() << 4) | (localZ & 0xF);

        final JoltVoxelColliderBakery bakery = this.bakery();
        final LevelAccelerator accelerator = this.accelerator;
        final SubLevelContainer container = SubLevelContainer.getContainer(this.level);

        final BlockPos.MutableBlockPos mpos = new BlockPos.MutableBlockPos();
        final Object2IntOpenHashMap<BlockState> resolveCache = newResolveCache();
        // One batched scene call for the 6 neighbors plus the block itself.
        final int[] batchX = new int[7];
        final int[] batchY = new int[7];
        final int[] batchZ = new int[7];
        final int[] batchPacked = new int[7];
        final int[] batchOwner = new int[7];

        int n = 0;
        for (final Direction dir : DIRECTIONS) {
            final int nx = worldX + dir.getStepX();
            final int ny = worldY + dir.getStepY();
            final int nz = worldZ + dir.getStepZ();
            mpos.set(nx, ny, nz);

            // Never loads chunks: unloaded neighbors read as air.
            final BlockState neighborState = accelerator.getBlockState(mpos);
            final VoxelNeighborhoodState state = neighborState.isAir()
                    ? VoxelNeighborhoodState.EMPTY
                    : VoxelNeighborhoodState.getState(accelerator, mpos, null);

            final int resolved = neighborState.isAir() ? 0 : resolveVoxel(resolveCache, bakery, neighborState, isSolidState(state));
            batchX[n] = nx;
            batchY[n] = ny;
            batchZ[n] = nz;
            batchPacked[n] = neighborState.isAir() ? 0 : packBlockState(state, resolved >>> 4, resolved & 0xF);
            batchOwner[n] = ownerOf(container, nx >> 4, nz >> 4);
            n++;
        }

        mpos.set(worldX, worldY, worldZ);
        if (newState.isAir()) {
            batchPacked[n] = 0;
        } else {
            final VoxelNeighborhoodState selfState = VoxelNeighborhoodState.getState(accelerator, mpos, null);
            final int selfResolved = resolveVoxel(resolveCache, bakery, newState, isSolidState(selfState));
            batchPacked[n] = packBlockState(selfState, selfResolved >>> 4, selfResolved & 0xF);
        }
        batchX[n] = worldX;
        batchY[n] = worldY;
        batchZ[n] = worldZ;
        batchOwner[n] = ownerOf(container, worldX >> 4, worldZ >> 4);
        n++;
        scene.changeBlocks(batchX, batchY, batchZ, batchPacked, batchOwner, n);
    }

    private static int ownerOf(@Nullable final SubLevelContainer container, final int chunkX, final int chunkZ) {
        if (container == null || !container.inBounds(chunkX, chunkZ)) {
            return -1;
        }
        final LevelPlot plot = container.getPlot(chunkX, chunkZ);
        return plot != null && plot.getSubLevel() instanceof final ServerSubLevel subLevel ? subLevel.getRuntimeId() : -1;
    }

    @Override
    public void onStatsChanged(@NotNull final ServerSubLevel subLevel) {
        this.assertBodyValid(subLevel);
        final JoltPhysicsScene scene = this.scene;
        if (scene == null) {
            return;
        }

        final BoundingBox3ic plotBounds = subLevel.getPlot().getBoundingBox();
        final int id = subLevel.getRuntimeId();

        final Vector3dc centerOfMass = subLevel.getMassTracker().getCenterOfMass();
        if (centerOfMass != null) {
            scene.setCenterOfMass(id, centerOfMass.x(), centerOfMass.y(), centerOfMass.z());
            this.setMassPropertiesFrom(id, subLevel.getMassTracker());
        }

        scene.setLocalBounds(id, plotBounds.minX(), plotBounds.minY(), plotBounds.minZ(), plotBounds.maxX(), plotBounds.maxY(), plotBounds.maxZ());
    }

    /**
     * Teleports the physics body of a sub-level to a given position.
     */
    @Override
    public void teleport(final PhysicsPipelineBody body, final Vector3dc position, final Quaterniondc orientation) {
        this.assertBodyValid(body);

        if (this.scene != null) {
            this.scene.teleportObject(body.getRuntimeId(), position.x(), position.y(), position.z(), orientation.x(), orientation.y(), orientation.z(), orientation.w());
        }
        if (body instanceof final ServerSubLevel subLevel) {
            subLevel.logicalPose().position().set(position);
            subLevel.logicalPose().orientation().set(orientation);
        }
    }

    /**
     * Adds a force at a given world position to a sub-level containing the position
     */
    @Override
    public void applyImpulse(final PhysicsPipelineBody body, final Vector3dc position, final Vector3dc force) {
        this.assertBodyValid(body);
        final Vector3dc centerOfMass = body.getMassTracker().getCenterOfMass();
        if (this.scene == null || centerOfMass == null) {
            return;
        }
        this.scene.applyForce(body.getRuntimeId(), position.x() - centerOfMass.x(), position.y() - centerOfMass.y(), position.z() - centerOfMass.z(), force.x(), force.y(), force.z(), true);
    }

    /**
     * Adds a local force and torque
     */
    @Override
    public void applyLinearAndAngularImpulse(final PhysicsPipelineBody body, final Vector3dc force, final Vector3dc torque, final boolean wakeUp) {
        this.assertBodyValid(body);
        if (this.scene != null) {
            this.scene.applyForceAndTorque(body.getRuntimeId(), force.x(), force.y(), force.z(), torque.x(), torque.y(), torque.z(), wakeUp);
        }
    }

    /**
     * Adds linear and angular velocities to a sub-level
     */
    @Override
    public void addLinearAndAngularVelocity(final PhysicsPipelineBody body, final Vector3dc linearVelocity, final Vector3dc angularVelocity) {
        this.assertBodyValid(body);
        if (this.scene != null) {
            this.scene.addLinearAngularVelocities(body.getRuntimeId(), linearVelocity.x(), linearVelocity.y(), linearVelocity.z(), angularVelocity.x(), angularVelocity.y(), angularVelocity.z(), true);
        }
    }

    @Override
    public Vector3d getLinearVelocity(final PhysicsPipelineBody body, final Vector3d dest) {
        this.assertBodyValid(body);
        if (this.scene == null || this.scene.body(body.getRuntimeId()) == null) {
            return dest.zero();
        }
        this.scene.getLinearVelocity(body.getRuntimeId(), this.poseCache);
        return dest.set(this.poseCache.p1, this.poseCache.p2, this.poseCache.p3);
    }

    @Override
    public Vector3d getAngularVelocity(final PhysicsPipelineBody body, final Vector3d dest) {
        this.assertBodyValid(body);
        if (this.scene == null || this.scene.body(body.getRuntimeId()) == null) {
            return dest.zero();
        }
        this.scene.getAngularVelocity(body.getRuntimeId(), this.poseCache);
        return dest.set(this.poseCache.p1, this.poseCache.p2, this.poseCache.p3);
    }

    /**
     * "Wakes up" a sub-level, indicating environmental or other changes have occurred that should resume physics for idled or sleeping sub-levels.
     */
    @Override
    public void wakeUp(final PhysicsPipelineBody body) {
        this.assertBodyValid(body);
        if (this.scene == null) {
            return;
        }

        if (!SubLevelPhysicsSystem.IN_PHYSICS_STEP) {
            this.scene.wakeUpObject(body.getRuntimeId());
        } else {
            this.queuedWakeUps.add(body);
        }
    }

    /**
     * Adds a constraint to the engine, returning its handle
     */
    @SuppressWarnings("unchecked")
    @Override
    @Nullable
    public <T extends PhysicsConstraintHandle> T addConstraint(@Nullable final PhysicsPipelineBody bodyA, @Nullable final PhysicsPipelineBody bodyB, @NotNull final PhysicsConstraintConfiguration<T> configuration) {
        if (bodyA == null && bodyB == null) {
            throw new IllegalArgumentException("Cannot add a constraint between the static world and static world");
        }

        if (bodyA == bodyB) {
            throw new IllegalArgumentException("Cannot add a constraint between a body and itself");
        }

        try {
            configuration.validate(ServerSubLevelContainer.getContainer(this.level), bodyA, bodyB);
        } catch (final Exception e) {
            if (JoltDebugLogging.HANDLE) {
                Sable.LOGGER.error("[SableJolt:handle] addConstraint validation failed: A={} B={} cfg={}",
                        bodyA, bodyB, configuration, e);
            }
            throw new IllegalArgumentException("Constraint validation failed", e);
        }

        final JoltPhysicsScene scene = this.scene;
        if (scene == null) {
            return null;
        }

        // Sable's constraint contract: each anchor is in the block space of its
        // body (the plot of a sub-level, world coordinates for the static world),
        // as in the rapier pipeline. isZeroAnchor marks the physics-staff
        // sentinel (pos1 = ZERO + per-tick render-frame motor targets).
        final T constraint = switch (configuration) {
            case final RotaryConstraintConfiguration config ->
                    (T) JoltRotaryConstraintHandle.create(scene, bodyA, bodyB, config);
            case final FixedConstraintConfiguration config ->
                    (T) JoltFixedConstraintHandle.create(scene, bodyA, bodyB, config);
            case final FreeConstraintConfiguration config ->
                    (T) JoltFreeConstraintHandle.create(scene, bodyA, bodyB, config, isZeroAnchor(config.pos1()));
            case final GenericConstraintConfiguration config ->
                    (T) JoltGenericConstraintHandle.create(scene, bodyA, bodyB, config);
        };

        if (!constraint.isValid()) {
            if (JoltDebugLogging.HANDLE) {
                Sable.LOGGER.error("[SableJolt:handle] addConstraint produced an invalid handle: A={} B={} cfg={}",
                        bodyA, bodyB, configuration);
            }
            return null;
        }

        if (JoltDebugLogging.HANDLE && configuration instanceof final FreeConstraintConfiguration config) {
            Sable.LOGGER.info("[SableJolt:handle] addConstraint ok: A={} B={} pos1=({}) pos2=({})",
                    bodyA, bodyB, config.pos1(), config.pos2());
        }

        return constraint;
    }

    /**
     * The physics staff passes the ZERO sentinel as pos1 and drives the motors
     * with per-tick render-frame targets; that selects the render-frame servo.
     */
    private static boolean isZeroAnchor(final Vector3dc p) {
        return p.x() == 0.0 && p.y() == 0.0 && p.z() == 0.0;
    }

    /**
     * Updates the config of the physics engine from a data object
     */
    @Override
    public void updateConfigFrom(final PhysicsConfigData data) {
        if (this.scene != null) {
            this.scene.configSolverIterations(data.solverIterations, data.pgsIterations);
        }
    }

    /**
     * @return the next runtime ID for a collider / sub-level
     */
    @Override
    public int getNextRuntimeID() {
        return this.scene != null ? this.scene.nextRuntimeId() : this.fallbackRuntimeId++;
    }

    /**
     * Ids handed out before init (never expected); far from the scene's range.
     */
    private int fallbackRuntimeId = 1 << 30;

    private void setMassPropertiesFrom(final int id, final MassData massTracker) {
        final Matrix3dc inertiaTensor = massTracker.getInertiaTensor();
        final double mass = massTracker.getMass();

        this.scene().setMassProperties(id, mass, inertiaTensor);
    }

    private void assertBodyValid(final PhysicsPipelineBody body) {
        if (body.isRemoved()) {
            throw new RuntimeException("Body has been removed");
        }
    }

    // Reusable scratch for contraption pose uploads (server thread only).
    private final Vector3d tmpContraptionPos = new Vector3d();
    private final Vector3d tmpLinVel = new Vector3d();
    private final Vector3d tmpAngVel = new Vector3d();
    private final Quaterniond tmpQuat = new Quaterniond();
    private final Vector3d tmpCollisionPoint = new Vector3d();

    private void updateContraptionPoses() {
        if (this.activeContraptions.isEmpty()) {
            return;
        }
        final SubLevelPhysicsSystem system = SubLevelPhysicsSystem.get(this.level);
        final double partialPhysicsTick = system != null ? system.getPartialPhysicsTick() : 1.0;

        for (final KinematicContraption contraption : this.activeContraptions.keySet()) {
            this.updateContraptionPose(contraption, partialPhysicsTick);
        }
    }

    /**
     * Uploads a contraption's pose at the end of this substep: its position in
     * its mount's block space (world space when unmounted), orientation and
     * velocities in its own frame (used as contact surface velocities).
     */
    private void updateContraptionPose(final KinematicContraption contraption, final double partialPhysicsTick) {
        final TrackedKinematicContraption trackedContraption = this.activeContraptions.get(contraption);
        if (trackedContraption == null || this.scene == null) {
            return;
        }

        final Vector3dc lastPosition = contraption.sable$getPosition(partialPhysicsTick - 1.0f);
        final Quaterniond lastOrientation = contraption.sable$getOrientation(partialPhysicsTick - 1.0f);

        final Vector3d pos = this.tmpContraptionPos.set(contraption.sable$getPosition(partialPhysicsTick));
        final Quaterniondc rot = contraption.sable$getOrientation(partialPhysicsTick);

        final Vector3d linVel = this.tmpLinVel.set(pos).sub(lastPosition);
        final Vector3d angVel = SableMathUtils.getAngularVelocity(lastOrientation, rot, this.tmpAngVel);

        linVel.mul(20.0);
        angVel.mul(20.0);
        rot.transformInverse(linVel);
        rot.transformInverse(angVel);

        final double rotationDelta = this.tmpQuat.set(rot).div(trackedContraption.lastUploadedOrientation()).angle();
        if (
                pos.distanceSquared(trackedContraption.lastUploadedPosition()) > DISTANCE_THRESHOLD * DISTANCE_THRESHOLD ||
                        linVel.distanceSquared(trackedContraption.lastUploadedLinVel()) > DISTANCE_THRESHOLD * DISTANCE_THRESHOLD ||
                        angVel.distanceSquared(trackedContraption.lastUploadedAngVel()) > DISTANCE_THRESHOLD * DISTANCE_THRESHOLD ||
                        rotationDelta > ANGULAR_THRESHOLD * ANGULAR_THRESHOLD
        ) {
            final MassTracker massTracker = contraption.sable$getMassTracker();
            final Vector3dc centerOfMass = massTracker.getCenterOfMass();

            this.scene.setKinematicContraptionTransform(trackedContraption.id(), centerOfMass, pos, rot, linVel, angVel);

            trackedContraption.lastUploadedPosition().set(pos);
            trackedContraption.lastUploadedLinVel().set(linVel);
            trackedContraption.lastUploadedAngVel().set(angVel);
            trackedContraption.lastUploadedOrientation().set(rot);
        }
    }

    private static double cachedMass(final Int2DoubleOpenHashMap cache, final int id,
                                     @Nullable final ServerSubLevel subLevel) {
        if (subLevel == null) {
            return Double.MAX_VALUE;
        }
        double mass = cache.get(id);
        if (Double.isNaN(mass)) {
            mass = subLevel.getMassTracker().getMass();
            cache.put(id, mass);
        }
        return mass;
    }

    private void processCollisionEffects() {
        final double[] collisions = this.scene().clearCollisions();
        final int collisionRecords = this.scene().lastCollisionCount();
        // Common case fast path: nothing reported and nothing debounced.
        if (collisionRecords == 0 && this.recentCollisions.isEmpty()) {
            return;
        }

        final ObjectIterator<Long2LongMap.Entry> recentCollectionsIterator =
                this.recentCollisions.long2LongEntrySet().iterator();

        final long gameTime = this.level.getGameTime();
        while (recentCollectionsIterator.hasNext()) {
            final Long2LongMap.Entry entry = recentCollectionsIterator.next();

            if (gameTime - entry.getLongValue() > 2) {
                recentCollectionsIterator.remove();
            }
        }

        final BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
        final Vector3d globalPoint = this.tmpCollisionPoint;
        // Masses repeat across records of the same pair; resolve once per id.
        final Int2DoubleOpenHashMap massCache = new Int2DoubleOpenHashMap();
        massCache.defaultReturnValue(Double.NaN);

        for (int i = 0; i < collisionRecords; i++) {
            final int base = i * 15;
            // record layout: [idA, idB, force, normalA(3), normalB(3), pointA(3), pointB(3)];
            // normals and point B are unused downstream (same as the rapier pipeline)
            final int idA = (int) collisions[base];
            final int idB = (int) collisions[base + 1];
            final double forceAmount = collisions[base + 2];
            final double pax = collisions[base + 9], pay = collisions[base + 10], paz = collisions[base + 11];
            final double pbx = collisions[base + 12], pby = collisions[base + 13], pbz = collisions[base + 14];

            final ServerSubLevel subLevelA = this.activeSubLevels.get(idA);
            final ServerSubLevel subLevelB = this.activeSubLevels.get(idB);

            final double minMass = Math.min(cachedMass(massCache, idA, subLevelA), cachedMass(massCache, idB, subLevelB));
            if (forceAmount <= 25.0 * minMass) {
                continue;
            }

            BlockState state = Blocks.STONE.defaultBlockState();

            if (subLevelA != null) {
                final Pose3d pose = subLevelA.logicalPose();

                final double ppx = pax + pose.rotationPoint().x;
                final double ppy = pay + pose.rotationPoint().y;
                final double ppz = paz + pose.rotationPoint().z;

                final int cornerPosX = Mth.floor(ppx + 0.5);
                final int cornerPosY = Mth.floor(ppy + 0.5);
                final int cornerPosZ = Mth.floor(ppz + 0.5);

                final long cornerPosLong = BlockPos.asLong(cornerPosX, cornerPosY, cornerPosZ);

                final long exists = this.recentCollisions.put(cornerPosLong, gameTime);

                if (exists != -1) {
                    continue;
                }
            }

            if (subLevelB != null) {
                final Pose3d pose = subLevelB.logicalPose();

                final double ppx = pbx + pose.rotationPoint().x;
                final double ppy = pby + pose.rotationPoint().y;
                final double ppz = pbz + pose.rotationPoint().z;

                final int cornerPosX = Mth.floor(ppx + 0.5);
                final int cornerPosY = Mth.floor(ppy + 0.5);
                final int cornerPosZ = Mth.floor(ppz + 0.5);

                final long cornerPosLong = BlockPos.asLong(cornerPosX, cornerPosY, cornerPosZ);

                pos.set(ppx, ppy, ppz);

                final long exists = this.recentCollisions.put(cornerPosLong, gameTime);

                if (exists != -1) {
                    continue;
                }

                state = this.accelerator.getBlockState(pos);
            }

            globalPoint.set(pax, pay, paz);
            if (subLevelA != null) {
                final Pose3d pose = subLevelA.logicalPose();
                pose.orientation().transform(globalPoint).add(pose.position());
            }

            final double px = globalPoint.x;
            final double py = globalPoint.y;
            final double pz = globalPoint.z;

            this.level.sendParticles(new BlockParticleOption(ParticleTypes.BLOCK, state), px, py, pz, 2, 0.0, 0.0, 0.0, 0.1);

            final float volumeScale = 0.4f;
            final SoundType soundType = state.getSoundType();

            this.level.playSound(null, px, py, pz, soundType.getStepSound(), SoundSource.BLOCKS, 0.2f * volumeScale, (float) (0.6 - 0.2 + Math.random() * 0.4));
            this.level.playSound(null, px, py, pz, soundType.getHitSound(), SoundSource.BLOCKS, 0.2f * volumeScale, (float) (Math.random() * 0.4));
            this.level.playSound(null, px, py, pz, soundType.getPlaceSound(), SoundSource.BLOCKS, 0.2f * volumeScale, (float) (0.5 - 0.2 + Math.random() * 0.4));
        }
    }

    private record TrackedKinematicContraption(Vector3d lastUploadedPosition, Quaterniond lastUploadedOrientation,
                                               Vector3d lastUploadedLinVel, Vector3d lastUploadedAngVel, int id) {
    }

    /**
     * A level accelerator that never loads chunks: blocks of unloaded world
     * chunks read as air. Neighborhood scans at the edge of the loaded area
     * otherwise load (or generate) chunks synchronously on the server thread,
     * which stalls the tick and can re-enter the pipeline through world
     * generation. Plot chunks are always served (Sable keeps them in memory).
     */
    private static final class LoadedChunksAccelerator extends LevelAccelerator {
        private static final BlockState AIR = Blocks.AIR.defaultBlockState();

        private final ServerLevel serverLevel;

        LoadedChunksAccelerator(final ServerLevel level) {
            super(level);
            this.serverLevel = level;
        }

        private boolean available(final BlockPos pos) {
            final int chunkX = pos.getX() >> 4;
            final int chunkZ = pos.getZ() >> 4;
            final SubLevelContainer container = SubLevelContainer.getContainer(this.serverLevel);
            if (container != null && container.inBounds(chunkX, chunkZ)) {
                return true;
            }
            return this.serverLevel.getChunkSource().getChunkNow(chunkX, chunkZ) != null;
        }

        @Override
        public BlockState getBlockState(final BlockPos pos) {
            if (this.isOutsideBuildHeight(pos) || !this.available(pos)) {
                return AIR;
            }
            return super.getBlockState(pos);
        }

        @Override
        public FluidState getFluidState(final BlockPos pos) {
            if (this.isOutsideBuildHeight(pos) || !this.available(pos)) {
                return Fluids.EMPTY.defaultFluidState();
            }
            return super.getFluidState(pos);
        }
    }
}
