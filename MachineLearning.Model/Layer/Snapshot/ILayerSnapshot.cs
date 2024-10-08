using System.Collections.Concurrent;
using MachineLearning.Model.Embedding;

namespace MachineLearning.Model.Layer.Snapshot;

public interface ILayerSnapshot;

public static class LayerSnapshots
{
    private static readonly ConcurrentDictionary<ILayer, ConcurrentQueue<ILayerSnapshot>> _registry = [];
    public static ILayerSnapshot Get(ILayer layer)
    {
        var queue = _registry.GetOrAdd(layer, static (layer) => []);

        if (queue.TryDequeue(out var snapshot))
        {
            return snapshot;
        }

        return Create(layer);
    }

    public static void Return(ILayer layer, ILayerSnapshot snapshot)
    {
        _registry[layer].Enqueue(snapshot);
    }

    public static T Is<T>(ILayerSnapshot snapshot)
    {
        if (snapshot is T t)
        {
            return t;
        }
        throw new InvalidOperationException();
    }


    public static ILayerSnapshot Create(ILayer layer) => layer switch
    {
        SimpleLayer simpleLayer => new Simple(simpleLayer.InputNodeCount, simpleLayer.OutputNodeCount),
        StringEmbeddingLayer => new Embedding(),
        TokenOutputLayer or IEmbedder<string, char> or IEmbedder<double[], int> => EmptySnapshot.Empty,
        _ => throw new NotImplementedException($"No snapshot for {layer} found"),
    };

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

    public sealed class EmptySnapshot : ILayerSnapshot
    {
        public static readonly EmptySnapshot Empty = new();
    }
}