package dev.behindthescenery.sablejolt;

import dev.behindthescenery.sablejolt.constraint.fixed.JoltFixedConstraintHandle;
import dev.behindthescenery.sablejolt.constraint.free.JoltFreeConstraintHandle;
import dev.behindthescenery.sablejolt.constraint.generic.JoltGenericConstraintHandle;
import dev.behindthescenery.sablejolt.constraint.rotary.JoltRotaryConstraintHandle;
import dev.behindthescenery.sablejolt.rope.JoltRopeHandle;
import dev.behindthescenery.sablejolt.box.JoltBoxHandle;
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
import dev.ryanhcode.sable.companion.math.BoundingBox3i;
import dev.ryanhcode.sable.companion.math.BoundingBox3ic;
import dev.ryanhcode.sable.companion.math.JOMLConversion;
import dev.ryanhcode.sable.companion.math.Pose3d;
import dev.ryanhcode.sable.companion.math.Pose3dc;
import dev.behindthescenery.sablejolt.collider.JoltVoxelColliderBakery;
import dev.behindthescenery.sablejolt.collider.JoltVoxelColliderData;
import dev.ryanhcode.sable.physics.chunk.VoxelNeighborhoodState;
import dev.ryanhcode.sable.physics.config.PhysicsConfigData;
import dev.ryanhcode.sable.sublevel.ServerSubLevel;
import dev.ryanhcode.sable.sublevel.SubLevel;
import dev.ryanhcode.sable.sublevel.plot.LevelPlot;
import dev.ryanhcode.sable.sublevel.system.SubLevelPhysicsSystem;
import dev.ryanhcode.sable.util.LevelAccelerator;
import dev.ryanhcode.sable.util.SableMathUtils;
import it.unimi.dsi.fastutil.ints.Int2ObjectArrayMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
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
import net.minecraft.world.level.chunk.BulkSectionAccess;
import net.minecraft.world.level.chunk.LevelChunk;
import net.minecraft.world.level.chunk.LevelChunkSection;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.joml.Matrix3dc;
import org.joml.Quaterniond;
import org.joml.Quaterniondc;
import org.joml.Vector3d;
import org.joml.Vector3dc;

import java.util.Objects;

/**
 * Implementation of {@link PhysicsPipeline} for the Jolt physics engine.
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

    private final ServerLevel level;
    private final LevelAccelerator accelerator;
    private final Int2ObjectMap<ServerSubLevel> activeSubLevels = new Int2ObjectArrayMap<>();
    private final Object2ObjectMap<KinematicContraption, TrackedKinematicContraption> activeContraptions = new Object2ObjectOpenHashMap<>();
    private final Long2LongOpenHashMap recentCollisions = new Long2LongOpenHashMap();
    private final ReferenceList<PhysicsPipelineBody> queuedWakeUps = new ReferenceArrayList<>();
    private final JoltPhysicsScene.PoseCache poseCache;
    private JoltPhysicsScene scene;
    private JoltVoxelColliderBakery colliderBakery;

    public JoltPhysicsPipeline(final ServerLevel level) {
        this.level = level;
        this.accelerator = new LevelAccelerator(level);
        this.poseCache = new JoltPhysicsScene.PoseCache();
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
        this.scene = new JoltPhysicsScene(gravity.x(), gravity.y(), gravity.z(), universalDrag);
        this.colliderBakery = new JoltVoxelColliderBakery(this.level, this.scene.colliderRegistry);
        this.scene.attachColliderBakery(this.colliderBakery);
    }

    /**
     * Disposes all resources used by the physics pipeline.
     */
    @Override
    public void dispose() {
        if (this.scene != null) {
            this.scene.dispose();
            this.scene = null;
            this.colliderBakery = null;
        }
    }

    /**
     * Runs once before the physics substeps.
     */
    @Override
    public void prePhysicsTicks() {
    }

    /**
     * Runs a physics substep with a time step of {@code 1.0 / 20.0 / substeps} seconds.
     */
    @Override
    public void physicsTick(final double timeStep) {
        this.updateContraptionPoses();
        final JoltPhysicsScene scene = this.scene();
        scene.step(timeStep);

        for (final PhysicsPipelineBody queuedWakeUp : this.queuedWakeUps) {
            if (queuedWakeUp.isRemoved()) {
                continue;
            }
            scene.wakeUpObject(queuedWakeUp.getRuntimeId());
        }
        this.queuedWakeUps.clear();
    }

    /**
     * Called after all physics substeps have been run, to finalize the physics tick.
     */
    @Override
    public void postPhysicsTicks() {
        this.processCollisionEffects();
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
        final Vector3dc pos = pose.position();
        final Quaterniondc rot = pose.orientation();

        final int id = subLevel.getRuntimeId();
        this.scene().createSubLevel(id, pos, rot);

        // Guarantee terrain collision at the assembly point even if Sable's chunk
        // tickets have not uploaded the world sections there yet.
        this.uploadWorldSectionsAround(pos);

        subLevel.updateMergedMassData(1.0f);
        final Vector3dc centerOfMass = subLevel.getMassTracker().getCenterOfMass();

        if (centerOfMass != null) {
            subLevel.logicalPose().rotationPoint().set(centerOfMass);
            this.onStatsChanged(subLevel);
        }

        this.activeSubLevels.put(id, subLevel);
    }

    /**
     * Uploads world terrain sections in a 3x3x3 section cube around the assembly
     * point into the physics scene (skipping plot sections).
     */
    private void uploadWorldSectionsAround(final Vector3dc pos) {
        final ServerSubLevelContainer container = ServerSubLevelContainer.getContainer(this.level);
        if (container == null) {
            return;
        }

        final BlockPos center = BlockPos.containing(pos.x(), pos.y(), pos.z());
        final SectionPos centerSection = SectionPos.of(center);
        final int minSectionY = this.level.getMinSection();
        final int maxSectionY = this.level.getMaxSection() - 1;

        // The Mojang mechanism for caching Sections in order to get BlockState as quickly as possible.
        // It would be possible to get the elements directly from the PalettedContainer, but then there is a
        // high chance of breaking the comp with some kind of mod.
        try (BulkSectionAccess sectionAccess = new BulkSectionAccess(this.level)) {
            final BlockPos.MutableBlockPos globalPos = new BlockPos.MutableBlockPos();
            for (int x = centerSection.x() - 1; x <= centerSection.x() + 1; x++) {
                for (int z = centerSection.z() - 1; z <= centerSection.z() + 1; z++) {
                    if (container.getPlot(x, z) != null) {
                        continue;
                    }
                    for (int y = Math.max(centerSection.y() - 1, minSectionY); y <= Math.min(centerSection.y() + 1, maxSectionY); y++) {
                        final int[] array = new int[LevelChunkSection.SECTION_SIZE];

                        boolean anySolid = false;
                        for (int bx = 0; bx < 16; bx++) {
                            globalPos.setX(bx + (x << 4));
                            for (int bz = 0; bz < 16; bz++) {
                                globalPos.setZ(bz + (z << 4));
                                for (int by = 0; by < 16; by++) {
                                    globalPos.setY(by + (y << 4));
                                    final BlockState blockState = sectionAccess.getBlockState(globalPos);
                                    if (blockState.isAir()) {
                                        continue;
                                    }
                                    anySolid = true;

                                    final VoxelNeighborhoodState state = VoxelNeighborhoodState.getState(this.accelerator, globalPos, null);
                                    final JoltVoxelColliderData colliderData = this.bakery().getPhysicsDataForBlock(blockState);

                                    final int index = bx + (bz << 4) + (by << 8);
                                    final int colliderValue = colliderData == null ? 0 : this.colliderHandleOf(colliderData) + 1;
                                    array[index] = packBlockState(state, colliderValue, blockState.getFluidState().getAmount());
                                }
                            }
                        }

                        if (anySolid) {
                            this.scene().addChunk(x, y, z, array, true, -1);
                        }
                    }
                }
            }
        }
    }

    /**
     * Removes a {@link SubLevel} from the physics pipeline.
     */
    @Override
    public void remove(final ServerSubLevel subLevel) {
        final int id = subLevel.getRuntimeId();
        this.scene().removeSubLevel(id);
        this.activeSubLevels.remove(id);
    }

    /**
     * Adds a kinematic contraption to the scene
     */
    @Override
    public void add(final KinematicContraption contraption) {
        if (this.activeContraptions.containsKey(contraption)) {
            throw new IllegalStateException("Contraption " + contraption + " is already present in pipeline");
        }

        final JoltPhysicsScene scene = this.scene();
        final JoltVoxelColliderBakery bakery = this.bakery();

        final int id = this.getNextRuntimeID();
        this.activeContraptions.put(contraption, new TrackedKinematicContraption(new Vector3d(), new Quaterniond(), new Vector3d(), new Vector3d(), id));

        final SubLevel mountSubLevel = Sable.HELPER.getContaining(this.level, contraption.sable$getPosition());
        final int mountId = mountSubLevel != null ? ((ServerSubLevel) mountSubLevel).getRuntimeId() : -1;

        final BoundingBox3i localBounds = new BoundingBox3i();
        contraption.sable$getLocalBounds(localBounds);

        final Vector3dc pos = contraption.sable$getPosition();
        final Quaterniond rot = contraption.sable$getOrientation();

        scene.createKinematicContraption(mountId, id, pos, rot);

        /*
        record UploadingContraptionChunk(int[] data) {
        }
        */
        // UploadingContraptionChunk
        final Long2ObjectMap<int[]> chunks = new Long2ObjectOpenHashMap<>();

        final BlockPos.MutableBlockPos blockPos = new BlockPos.MutableBlockPos();
        for (int x = localBounds.minX(); x <= localBounds.maxX(); x++) {
            for (int z = localBounds.minZ(); z <= localBounds.maxZ(); z++) {
                for (int y = localBounds.minY(); y <= localBounds.maxY(); y++) {
                    final BlockState blockState = contraption.sable$blockGetter().getBlockState(blockPos.set(x, y, z));

                    if (blockState.isAir()) continue;

                    final int[] chunk = chunks.computeIfAbsent(
                            SectionPos.asLong(x >> 4, y >> 4, z >> 4),
                            longPos -> new int[LevelChunkSection.SECTION_SIZE]
                    );

                    final VoxelNeighborhoodState state = VoxelNeighborhoodState.CORNER;
                    final JoltVoxelColliderData colliderData = bakery.getPhysicsDataForBlock(blockState);

                    final int index = (x & 15) + ((z & 15) << 4) + ((y & 15) << 8);

                    final int colliderValue = colliderData == null ? 0 : this.colliderHandleOf(colliderData) + 1;
                    chunk[index] = packBlockState(state, colliderValue, blockState.getFluidState().getAmount());
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

    private int colliderHandleOf(final JoltVoxelColliderData data) {
        return this.scene().colliderRegistry.indexOf(data);
    }

    /**
     * Removes a kinematic contraption from the scene
     */
    @Override
    public void remove(final KinematicContraption contraption) {
        final TrackedKinematicContraption removed = this.activeContraptions.remove(contraption);

        if (removed == null) {
            return;
        }

        this.scene().removeKinematicContraption(removed.id());
    }

    /**
     * Queries the physics pipeline for the current pose of a {@link SubLevel}.
     */
    @Override
    public Pose3d readPose(final ServerSubLevel subLevel, final Pose3d dest) {
        this.assertBodyValid(subLevel);
        this.scene().getPose(subLevel.getRuntimeId(), this.poseCache);


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
        this.accelerator.clearCache();

        final int[] array = new int[LevelChunkSection.SECTION_SIZE];

        final SectionPos sectionPos = SectionPos.of(x, y, z);

        if (!section.hasOnlyAir()) {
            final LevelChunk chunk = this.accelerator.getChunk(x, z);

            for (int bx = 0; bx < 16; bx++) {
                for (int bz = 0; bz < 16; bz++) {
                    for (int by = 0; by < 16; by++) {
                        final BlockPos globalPos = new BlockPos(bx, by, bz).offset(sectionPos.minBlockX(), sectionPos.minBlockY(), sectionPos.minBlockZ());
                        final VoxelNeighborhoodState state = VoxelNeighborhoodState.getState(this.accelerator, globalPos, chunk);
                        final BlockState blockState = this.accelerator.getBlockState(globalPos);
                        final JoltVoxelColliderData colliderData = this.bakery().getPhysicsDataForBlock(blockState);

                        final int index = bx + (bz << 4) + (by << 8);

                        final int colliderValue = colliderData == null ? 0 : this.colliderHandleOf(colliderData) + 1;
                        array[index] = packBlockState(state, colliderValue, blockState.getFluidState().getAmount());
                    }
                }
            }
        }

        final LevelPlot plot = SubLevelContainer.getContainer(this.level).getPlot(x, z);
        final boolean global = plot == null;
        int id = -1;

        if (plot != null && uploadDataIfGlobal) id = ((ServerSubLevel) plot.getSubLevel()).getRuntimeId();
        this.scene().addChunk(x, y, z, array, global, id);
    }

    /**
     * Handles the removal of a chunk section from the physics context
     */
    @Override
    public void handleChunkSectionRemoval(final int x, final int y, final int z) {
        this.scene().removeChunk(x, y, z, !SubLevelContainer.getContainer(this.level).inBounds(x, z));
    }

    /**
     * Handles the change of a block (from oldState to newState) in a chunk at chunk-relative position x, y, z.
     */
    /**
     * Re-entrancy guard for {@link #handleBlockChange}: our chunk loads can run
     * queued generation tasks whose setBlock calls re-enter this hook, which
     * previously recursed without bound (world-enter stall).
     */
    private static final ThreadLocal<Boolean> HANDLING_BLOCK_CHANGE = ThreadLocal.withInitial(() -> Boolean.FALSE);

    /**
     * Handles the change of a block (from oldState to newState) in a chunk at chunk-relative position x, y, z.
     */
    @Override
    public void handleBlockChange(final SectionPos sectionPos, final LevelChunkSection chunk,
                                  final int localX, final int localY, final int localZ,
                                  final BlockState oldState, final BlockState newState) {

        final int secX = sectionPos.x();
        final int secY = sectionPos.y();
        final int secZ = sectionPos.z();

        final int worldX = (secX << 4) | (localX & 0xF);
        final int worldY = (secY << 4) | (localY & 0xF);
        final int worldZ = (secZ << 4) | (localZ & 0xF);

        // Re-entrancy guard: ensureWorldSection below synchronously loads chunks on
        // this thread; a loaded chunk may run queued generation tasks whose setBlock
        // fires this hook again. On re-entry forward only the single block update —
        // no chunk loads, no sweeps; the affected sections are re-uploaded wholesale
        // by Sable afterwards.
        if (HANDLING_BLOCK_CHANGE.get()) {
            final BlockPos.MutableBlockPos mpos = new BlockPos.MutableBlockPos(worldX, worldY, worldZ);
            final VoxelNeighborhoodState state = VoxelNeighborhoodState.getState(this.accelerator, mpos, null);
            final JoltVoxelColliderData colliderData = this.bakery().getPhysicsDataForBlock(newState);

            final int colliderValue = colliderData == null ? 0 : this.colliderHandleOf(colliderData) + 1;
            this.scene().changeBlock(worldX, worldY, worldZ,
                    packBlockState(state, colliderValue, newState.getFluidState().getAmount()));
            return;
        }

        HANDLING_BLOCK_CHANGE.set(Boolean.TRUE);
        try {
            // Self-heal: if Sable believes a world section is already uploaded but we lost
            // it (e.g., it was removed while out of physics range and never re-added),
            // re-read it from the live level so block edits keep colliding.
            this.ensureWorldSection(secX, secY, secZ);
            if (localX == 0)  this.ensureWorldSection(secX - 1, secY, secZ);
            else if (localX == 15) this.ensureWorldSection(secX + 1, secY, secZ);
            if (localY == 0)  this.ensureWorldSection(secX, secY - 1, secZ);
            else if (localY == 15) this.ensureWorldSection(secX, secY + 1, secZ);
            if (localZ == 0)  this.ensureWorldSection(secX, secY, secZ - 1);
            else if (localZ == 15) this.ensureWorldSection(secX, secY, secZ + 1);

            final var scene = this.scene();
            final var bakery = this.bakery();
            final var level = this.level;
            final var accelerator = this.accelerator;

            final BlockPos.MutableBlockPos mpos = new BlockPos.MutableBlockPos();

            for (final Direction dir : DIRECTIONS) {
                final int nx = worldX + dir.getStepX();
                final int ny = worldY + dir.getStepY();
                final int nz = worldZ + dir.getStepZ();
                mpos.set(nx, ny, nz);

                final BlockState neighborState = level.getBlockState(mpos);
                final VoxelNeighborhoodState state = VoxelNeighborhoodState.getState(accelerator, mpos, null);
                final JoltVoxelColliderData colliderData = bakery.getPhysicsDataForBlock(neighborState);

                final int colliderValue = colliderData == null ? 0 : this.colliderHandleOf(colliderData) + 1;
                scene.changeBlock(nx, ny, nz, packBlockState(state, colliderValue, neighborState.getFluidState().getAmount()));
            }

            mpos.set(worldX, worldY, worldZ);
            final VoxelNeighborhoodState selfState = VoxelNeighborhoodState.getState(accelerator, mpos, null);
            final JoltVoxelColliderData selfColliderData = bakery.getPhysicsDataForBlock(newState);

            final int selfColliderValue = selfColliderData == null ? 0 : this.colliderHandleOf(selfColliderData) + 1;
            scene.changeBlock(worldX, worldY, worldZ, packBlockState(selfState, selfColliderValue, newState.getFluidState().getAmount()));
        } finally {
            HANDLING_BLOCK_CHANGE.set(Boolean.FALSE);
        }
    }

    /**
     * Re-reads a world section from the live level into the physics scene if it is a
     * global (non-plot) loaded section that the scene currently lacks.
     */
    private void ensureWorldSection(final int sx, final int sy, final int sz) {
        if (this.scene().hasChunk(sx, sy, sz)) {
            return;
        }

        final ServerLevel level = this.level;
        final ServerSubLevelContainer container = ServerSubLevelContainer.getContainer(level);
        if (container != null && container.getPlot(sx, sz) != null) {
            return;
        }

        final SectionPos sectionPos = SectionPos.of(sx, sy, sz);
        if (!level.hasChunkAt(sectionPos.center())) {
            return;
        }

        final int[] array = new int[LevelChunkSection.SECTION_SIZE];
        boolean anySolid = false;

        final BlockPos.MutableBlockPos globalPos = new BlockPos.MutableBlockPos();
        final JoltVoxelColliderBakery bakery = this.bakery();

        // The Mojang mechanism for caching Sections in order to get BlockState as quickly as possible.
        // It would be possible to get the elements directly from the PalettedContainer, but then there is a
        // high chance of breaking the comp with some kind of mod.
        try (BulkSectionAccess sectionAccess = new BulkSectionAccess(level)) {
            for (int bx = 0; bx < 16; bx++) {
                globalPos.setX(bx + sectionPos.minBlockX());
                for (int bz = 0; bz < 16; bz++) {
                    globalPos.setZ(bz + sectionPos.minBlockZ());
                    for (int by = 0; by < 16; by++) {
                        globalPos.setY(by + sectionPos.minBlockY());
                        final BlockState blockState = sectionAccess.getBlockState(globalPos);
                        if (blockState.isAir()) {
                            continue;
                        }
                        anySolid = true;

                        final VoxelNeighborhoodState state = VoxelNeighborhoodState.getState(this.accelerator, globalPos, null);
                        final JoltVoxelColliderData colliderData = bakery.getPhysicsDataForBlock(blockState);

                        final int index = bx + (bz << 4) + (by << 8);
                        final int colliderValue = colliderData == null ? 0 : this.colliderHandleOf(colliderData) + 1;
                        array[index] = packBlockState(state, colliderValue, blockState.getFluidState().getAmount());
                    }
                }
            }
        }

        if (anySolid) {
            this.scene().addChunk(sx, sy, sz, array, true, -1);
        }
    }

    @Override
    public void onStatsChanged(@NotNull final ServerSubLevel subLevel) {
        this.assertBodyValid(subLevel);

        final BoundingBox3ic plotBounds = subLevel.getPlot().getBoundingBox();
        final int id = subLevel.getRuntimeId();
        final JoltPhysicsScene scene = this.scene();

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

        this.scene().teleportObject(body.getRuntimeId(), position.x(), position.y(), position.z(), orientation.x(), orientation.y(), orientation.z(), orientation.w());
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
        this.scene().applyForce(body.getRuntimeId(), position.x() - centerOfMass.x(), position.y() - centerOfMass.y(), position.z() - centerOfMass.z(), force.x(), force.y(), force.z(), true);
    }

    /**
     * Adds a local force and torque
     */
    @Override
    public void applyLinearAndAngularImpulse(final PhysicsPipelineBody body, final Vector3dc force, final Vector3dc torque, final boolean wakeUp) {
        this.assertBodyValid(body);
        this.scene().applyForceAndTorque(body.getRuntimeId(), force.x(), force.y(), force.z(), torque.x(), torque.y(), torque.z(), wakeUp);
    }

    /**
     * Adds linear and angular velocities to a sub-level
     */
    @Override
    public void addLinearAndAngularVelocity(final PhysicsPipelineBody body, final Vector3dc linearVelocity, final Vector3dc angularVelocity) {
        this.assertBodyValid(body);
        this.scene().addLinearAngularVelocities(body.getRuntimeId(), linearVelocity.x(), linearVelocity.y(), linearVelocity.z(), angularVelocity.x(), angularVelocity.y(), angularVelocity.z(), true);
    }

    @Override
    public Vector3d getLinearVelocity(final PhysicsPipelineBody body, final Vector3d dest) {
        this.assertBodyValid(body);
        this.scene().getLinearVelocity(body.getRuntimeId(), this.poseCache);
        return dest.set(this.poseCache.p1, this.poseCache.p2, this.poseCache.p3);
    }

    @Override
    public Vector3d getAngularVelocity(final PhysicsPipelineBody body, final Vector3d dest) {
        this.assertBodyValid(body);
        this.scene().getAngularVelocity(body.getRuntimeId(), this.poseCache);
        return dest.set(this.poseCache.p1, this.poseCache.p2, this.poseCache.p3);
    }

    /**
     * "Wakes up" a sub-level, indicating environmental or other changes have occurred that should resume physics for idled or sleeping sub-levels.
     */
    @Override
    public void wakeUp(final PhysicsPipelineBody body) {
        this.assertBodyValid(body);

        if (!SubLevelPhysicsSystem.IN_PHYSICS_STEP) {
            this.scene().wakeUpObject(body.getRuntimeId());
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

        // Sable's constraint contract: pos1 (world/bodyA side) is in the render
        // frame next to the player, pos2 (bodyB side) is in the plot-global frame.
        // The Jolt scene lives in the render frame, so pos1 is used as-is and pos2
        // is projected into the render frame inside the handle (see
        // JoltFreeConstraintHandle.create). isZeroAnchor marks the physics-staff
        // sentinel (pos1 = ZERO + per-tick render-frame motor targets).
        final T constraint = switch (configuration) {
            case final RotaryConstraintConfiguration config ->
                    (T) JoltRotaryConstraintHandle.create(this.scene(), bodyA, bodyB, config);
            case final FixedConstraintConfiguration config ->
                    (T) JoltFixedConstraintHandle.create(this.scene(), bodyA, bodyB, config);
            case final FreeConstraintConfiguration config ->
                    (T) JoltFreeConstraintHandle.create(this.scene(), bodyA, bodyB, config, isZeroAnchor(config.pos1()));
            case final GenericConstraintConfiguration config ->
                    (T) JoltGenericConstraintHandle.create(this.scene(), bodyA, bodyB, config);
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
        this.scene().configSolverIterations(data.solverIterations, data.pgsIterations);
    }

    /**
     * @return the next runtime ID for a collider / sub-level
     */
    @Override
    public int getNextRuntimeID() {
        return this.scene().nextRuntimeId();
    }

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

    private void updateContraptionPoses() {
        final SubLevelPhysicsSystem system = SubLevelPhysicsSystem.require(this.level);
        final double partialPhysicsTick = system.getPartialPhysicsTick();

        for (final KinematicContraption contraption : this.activeContraptions.keySet()) {
            this.updateContraptionPose(contraption, partialPhysicsTick);
        }
    }

    private void updateContraptionPose(final KinematicContraption contraption, final double partialPhysicsTick) {
        final TrackedKinematicContraption trackedContraption = this.activeContraptions.get(contraption);

        final SubLevel mountSubLevel = Sable.HELPER.getContaining(this.level, contraption.sable$getPosition());
        final Vector3dc parentCenterOfMass = mountSubLevel != null ? ((ServerSubLevel) mountSubLevel).getMassTracker().getCenterOfMass() : JOMLConversion.ZERO;

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

        pos.sub(parentCenterOfMass);

        final double rotationDelta = this.tmpQuat.set(rot).div(trackedContraption.lastUploadedOrientation()).angle();
        if (
                pos.distanceSquared(trackedContraption.lastUploadedPosition()) > DISTANCE_THRESHOLD * DISTANCE_THRESHOLD ||
                        linVel.distanceSquared(trackedContraption.lastUploadedLinVel()) > DISTANCE_THRESHOLD * DISTANCE_THRESHOLD ||
                        angVel.distanceSquared(trackedContraption.lastUploadedAngVel()) > DISTANCE_THRESHOLD * DISTANCE_THRESHOLD ||
                        rotationDelta > ANGULAR_THRESHOLD * ANGULAR_THRESHOLD
        ) {
            final MassTracker massTracker = contraption.sable$getMassTracker();
            final Vector3dc centerOfMass = massTracker.getCenterOfMass();

            this.scene().setKinematicContraptionTransform(trackedContraption.id(), centerOfMass, pos, rot, linVel, angVel);

            trackedContraption.lastUploadedPosition().set(pos);
            trackedContraption.lastUploadedLinVel().set(linVel);
            trackedContraption.lastUploadedAngVel().set(angVel);
            trackedContraption.lastUploadedOrientation().set(rot);
        }
    }

    private void processCollisionEffects() {
        final ObjectIterator<Long2LongMap.Entry> recentCollectionsIterator =
                this.recentCollisions.long2LongEntrySet().iterator();

        // Hand unwrap virtual invoke
        final long gameTime = this.level.getGameTime();
        while (recentCollectionsIterator.hasNext()) {
            final Long2LongMap.Entry entry = recentCollectionsIterator.next();

            if (gameTime - entry.getLongValue() > 2) {
                recentCollectionsIterator.remove();
            }
        }

        final double[] collisions = this.scene().clearCollisions();

        final BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
        final Vector3d globalPoint = new Vector3d();

        for (int i = 0; i < collisions.length / 15; i++) {
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

            final double minMass = Math.min(subLevelA != null ? subLevelA.getMassTracker().getMass() : Double.MAX_VALUE, subLevelB != null ? subLevelB.getMassTracker().getMass() : Double.MAX_VALUE);
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

                final long exists = this.recentCollisions.put(cornerPosLong, this.level.getGameTime());

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

                final long exists = this.recentCollisions.put(cornerPosLong, this.level.getGameTime());

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
}
