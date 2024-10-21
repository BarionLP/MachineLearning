using System.Collections.Concurrent;
using MachineLearning.Model.Embedding;

namespace MachineLearning.Model.Layer.Snapshot;

public interface ILayerSnapshot;

public static class LayerSnapshots
{
    public static ILayerSnapshot Empty { get; } = new EmptySnapshot();
    private static readonly ConcurrentDictionary<ILayer, ConcurrentQueue<ILayerSnapshot>> _registry = new();
    static volatile int created = 0;
    static volatile int away = 0;
    public static ILayerSnapshot Get(ILayer layer)
    {
        var queue = _registry.GetOrAdd(layer, static (layer) => []);

        if (queue.TryDequeue(out var snapshot))
        {
            //Console.WriteLine("Reused Layer");
            away++;
            return snapshot;
        }

        away++;
        return Create(layer);
    }

    public static void Return(ILayer layer, ILayerSnapshot snapshot)
    {
        //Console.WriteLine("Returned Layer");
        away--;
        _registry[layer].Enqueue(snapshot);
    }

    public static T Is<T>(ILayerSnapshot snapshot)
    {
        if (snapshot is T t)
        {
            return t;
        }
        throw new InvalidOperationException($"LayerSnapshot {snapshot} did not match expected Type {typeof(T).Name}");
    }


    public static ILayerSnapshot Create(ILayer layer)
    {
        //Console.WriteLine("Created new Snapshot");
        created++;
        return layer switch
        {
            SimpleLayer simpleLayer => new Simple(simpleLayer.InputNodeCount, simpleLayer.OutputNodeCount),
            StringEmbeddingLayer => new Embedding(),
            TokenOutputLayer or IEmbedder<string, char> or IEmbedder<double[], int> => Empty,
            _ => throw new NotImplementedException($"No snapshot for {layer} found"),
        };
    }

    public static void Validate()
    {
        var inBag = _registry.Sum(p => p.Value.Count);
        ArgumentOutOfRangeException.ThrowIfNotEqual(inBag, created);
        ArgumentOutOfRangeException.ThrowIfNotEqual(away, 0);
    }

    public static void Clear(IEnumerable<ILayer> layers) => layers.Consume(Clear);
    public static void Clear(ILayer layer)
    {
        if (_registry.TryGetValue(layer, out var queue))
        {
            created -= queue.Count;
            queue.Clear();
        }
    }

    public sealed class Simple(int inputNodes, int outputNodes) : ILayerSnapshot
    {
        public readonly Vector LastRawInput = Vector.Create(inputNodes);
        public readonly Vector LastWeightedInput = Vector.Create(outputNodes);
        public readonly Vector LastActivatedWeights = Vector.Create(outputNodes);
        public readonly Matrix WeightGradients = Matrix.Create(outputNodes, inputNodes);
    }

    public sealed class Embedding : ILayerSnapshot
    {
        public string LastInput { get; set; } = string.Empty;
        //public Matrix Gradients { get; set; } = Matrix.Create(contextSize, contextSize);
        //public Vector LastOutput { get; } = Vector.Create(contextSize * embeddingSize);
    }

    internal sealed class EmptySnapshot : ILayerSnapshot;
}