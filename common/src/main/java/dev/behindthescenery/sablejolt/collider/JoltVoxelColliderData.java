package dev.behindthescenery.sablejolt.collider;

import com.github.stephengold.joltjni.Vec3;
import com.github.stephengold.joltjni.readonly.ConstShape;
import dev.ryanhcode.sable.api.block.BlockSubLevelCollisionShape;
import dev.ryanhcode.sable.api.physics.callback.BlockSubLevelCollisionCallback;
import dev.ryanhcode.sable.api.physics.collider.VoxelColliderData;
import net.minecraft.world.level.block.state.BlockState;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.Nullable;
import org.joml.Vector3d;
import org.joml.Vector3dc;

import java.util.ArrayList;
import java.util.List;

/**
 * The physics data of a single block state, kept entirely on the Java side.
 * Collision box shapes are built lazily and shared between all bodies.
 */
@ApiStatus.Internal
public final class JoltVoxelColliderData implements VoxelColliderData {
    /**
     * The boxes of this collider, formatted [minX, minY, minZ, maxX, maxY, maxZ] within 0-1 block space.
     */
    public final List<float[]> boxes = new ArrayList<>(4);
    public final boolean isFluid;
    public final float frictionMultiplier;
    public final float volume;
    public final float restitution;
    @Nullable public final BlockSubLevelCollisionCallback contactEvents;

    /**
     * The block state this data was built for; used to (re)build boxes lazily.
     */
    @Nullable public final BlockState sourceState;

    /**
     * Lazily built shared shapes, one per box index, reused by every body.
     */
    private volatile ConstShape[] shapes;

    JoltVoxelColliderData(final double frictionMultiplier, final double volume, final double restitution,
                          final boolean isFluid, @Nullable final BlockSubLevelCollisionCallback contactEvents,
                          @Nullable final BlockState sourceState) {
        this.frictionMultiplier = (float) frictionMultiplier;
        this.volume = (float) volume;
        this.restitution = (float) restitution;
        this.isFluid = isFluid;
        this.contactEvents = contactEvents;
        this.sourceState = sourceState;
    }

    public boolean needsSpecialContacts() {
        return this.contactEvents != null || this.frictionMultiplier != 1.0f || this.restitution != 0.0f;
    }

    public boolean hasBoxes() {
        return !this.boxes.isEmpty();
    }

    /**
     * Ensures the collision boxes have been computed. The box computation can run
     * before the block actually exists at its plot/world position, producing an
     * empty (but registered) entry; it is retried on demand here.
     */
    public synchronized void ensureBoxes(@Nullable final JoltVoxelColliderBakery owner) {
        if (this.isFluid || !this.boxes.isEmpty() || this.sourceState == null || owner == null) {
            return;
        }
        owner.buildBoxesInto(this, this.sourceState);
    }

    /**
     * Returns (building if necessary) the shared box shape for the given box index.
     */
    public ConstShape shape(final int boxIndex) {
        ConstShape[] arr = this.shapes;
        if (arr == null || arr.length <= boxIndex || arr[boxIndex] == null) {
            synchronized (this) {
                arr = this.shapes;
                if (arr == null) {
                    arr = new ConstShape[Math.max(this.boxes.size(), 1)];
                    this.shapes = arr;
                }
                if (boxIndex < arr.length && arr[boxIndex] == null) {
                    final float[] b = this.boxes.get(boxIndex);
                    final float hx = Math.max((b[3] - b[0]) * 0.5f, 0.0001f);
                    final float hy = Math.max((b[4] - b[1]) * 0.5f, 0.0001f);
                    final float hz = Math.max((b[5] - b[2]) * 0.5f, 0.0001f);
                    arr[boxIndex] = new com.github.stephengold.joltjni.BoxShape(new Vec3(hx, hy, hz), 0.025f);
                }
            }
        }
        return arr[boxIndex];
    }

    @Override
    public void addBox(final Vector3dc min, final Vector3dc max) {
        this.boxes.add(new float[]{
                (float) min.x(), (float) min.y(), (float) min.z(),
                (float) max.x(), (float) max.y(), (float) max.z()
        });
        this.shapes = null;
    }

    @Override
    public void clearBoxes() {
        this.boxes.clear();
        this.shapes = null;
    }

    /**
     * Global registry of voxel collider entries, mirroring the native {@code VoxelColliderMap}.
     */
    @ApiStatus.Internal
    public static final class Registry {
        private final List<JoltVoxelColliderData> entries = new ArrayList<>();

        public synchronized JoltVoxelColliderData create(final double frictionMultiplier, final double volume,
                                                         final double restitution, final boolean isFluid,
                                                         @Nullable final BlockSubLevelCollisionCallback contactEvents,
                                                         @Nullable final BlockState sourceState) {
            final JoltVoxelColliderData data = new JoltVoxelColliderData(frictionMultiplier, volume, restitution, isFluid, contactEvents, sourceState);
            this.entries.add(data);
            return data;
        }

        public synchronized JoltVoxelColliderData get(final int handle) {
            if (handle < 0 || handle >= this.entries.size()) {
                return null;
            }
            return this.entries.get(handle);
        }

        public synchronized int indexOf(final JoltVoxelColliderData data) {
            return this.entries.indexOf(data);
        }

        public synchronized void clear() {
            this.entries.clear();
        }
    }

    /**
     * Builds the shared shapes of every known entry. Used after bulk edits.
     */
    public static Vector3d boxCenter(final float[] box, final Vector3d dest) {
        return dest.set((box[0] + box[3]) * 0.5, (box[1] + box[4]) * 0.5, (box[2] + box[5]) * 0.5);
    }

    public static Vector3d boxHalfExtent(final float[] box, final Vector3d dest) {
        return dest.set((box[3] - box[0]) * 0.5, (box[4] - box[1]) * 0.5, (box[5] - box[2]) * 0.5);
    }
}
