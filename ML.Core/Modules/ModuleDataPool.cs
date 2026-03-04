using System.Collections.Concurrent;

namespace ML.Core.Modules;

public sealed class ModuleDataPool(Func<IModuleSnapshot> snapshotGetter, Func<IModuleGradients> gradientGetter)
{
    private readonly ConcurrentStack<IModuleGradients> gradientCache = [];
    private readonly ConcurrentStack<IModuleSnapshot> snaphotCache = [];

    public int UnusedItems => gradientCache.Count;

    public ModuleDataPool(IModule module)
    : this(() => module.CreateSnapshot(), () => module.CreateGradients())
    {

    }

    public RentedSnapshotsMarker RentSnapshot()
    {
        var rented = snaphotCache.TryPop(out var snapshots) ? snapshots : snapshotGetter();
        return new(this, rented);
    }

    public IModuleGradients RentGradients()
    {
        if (gradientCache.TryPop(out var gradients))
        {
            return gradients;
        }

        return gradientGetter();
    }


    public void Return(IModuleGradients gradients)
    {
        Debug.Assert(!gradientCache.Contains(gradients));
        gradients.Reset();
        gradientCache.Push(gradients);
    }

    public void Return(IModuleSnapshot snapshots)
    {
        Debug.Assert(!snaphotCache.Contains(snapshots));
        // snapshots are always overriden, so no reset
        snaphotCache.Push(snapshots);
    }

    public void Clear()
    {
        snaphotCache.Clear();
        gradientCache.Clear();
    }

    public readonly ref struct RentedSnapshotsMarker(ModuleDataPool pool, IModuleSnapshot snapshot)
    {
        public IModuleSnapshot Snapshot { get; } = snapshot;

        public readonly void Dispose()
        {
            pool.Return(Snapshot);
        }
    }
}