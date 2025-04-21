using System.Diagnostics;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model.Layer;

public sealed class TokenOutputLayer(int tokenCount, bool weightedRandom, Random? random = null) : IUnembeddingLayer<int>
{
    public int TokenCount { get; } = tokenCount;
    public bool WeightedRandom { get; } = weightedRandom;
    public Random Random { get; } = random ?? Random.Shared;

    public int InputNodeCount => TokenCount;
    public long WeightCount => 0;

    public (int output, Weight confidence) Process(Vector input)
    {
        var (result, confidence, _) = Process(input, default!);
        return (result, confidence);
    }

    public (int output, Weight confidence, Vector weights) Process(Vector input, ILayerSnapshot snapshot)
    {
        Debug.Assert(input.Count == TokenCount);

        var index = WeightedRandom ? GetWeightedRandomIndex(input, Random) : input.MaximumIndex();
        return (index, input[index], input);
    }

    private static int GetWeightedRandomIndex(Vector weights, Random random)
    {
        var value = random.NextDouble();
        for (int i = 0; i < weights.Count; i++)
        {
            value -= weights[i];
            if (value < 0)
                return i;
        }
        return weights.Count - 1;
    }

    public ILayerSnapshot CreateSnapshot() => ILayerSnapshot.Empty;
    public IGradients CreateGradientAccumulator() => IGradients.Empty;
}
