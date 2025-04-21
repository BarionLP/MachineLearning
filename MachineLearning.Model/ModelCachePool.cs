using System.Collections.Concurrent;
using System.Diagnostics;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model;

public sealed class ModelCachePool(Func<ImmutableArray<IGradients>> gradientGetter, Func<ImmutableArray<ILayerSnapshot>> snapshotGetter)
{
    private readonly ConcurrentBag<ImmutableArray<IGradients>> gradientCache = [];
    private readonly ConcurrentBag<ImmutableArray<ILayerSnapshot>> snaphotCache = [];

    public int UnusedItems => gradientCache.Count;

    public ModelCachePool(ImmutableArray<ILayer> layers)
    : this(() => [.. layers.Select(static l => l.CreateGradientAccumulator())], () => [.. layers.Select(static l => l.CreateSnapshot())])
    {

    }

    public ImmutableArray<IGradients> RentGradients()
    {
        if (gradientCache.TryTake(out var gradients))
        {
            return gradients;
        }

        return gradientGetter();
    }

    public RentedSnapshotsMarker RentSnapshots(out ImmutableArray<ILayerSnapshot> rented)
    {
        rented = snaphotCache.TryTake(out var snapshots) ? snapshots : snapshotGetter();
        return new(this, rented);
    }

    public void Return(ImmutableArray<IGradients> gradients)
    {
        Debug.Assert(!gradientCache.Contains(gradients));
        foreach (var gradient in gradients)
        {
            gradient.Reset();
        }
        gradientCache.Add(gradients);
    }

    public void Return(ImmutableArray<ILayerSnapshot> snapshots)
    {
        // snapshots are always overriden, so no reset
        Debug.Assert(!snaphotCache.Contains(snapshots));
        snaphotCache.Add(snapshots);
    }

    public void Clear()
    {
        snaphotCache.Clear();
        gradientCache.Clear();
    }

    public ref struct RentedSnapshotsMarker(ModelCachePool pool, ImmutableArray<ILayerSnapshot> rented)
    {
        public readonly void Dispose()
        {
            pool.Return(rented);
        }
    }
}