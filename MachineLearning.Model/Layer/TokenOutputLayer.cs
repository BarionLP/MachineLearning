using System.Diagnostics;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model.Layer;

public sealed class TokenOutputLayer(string tokens, bool weightedRandom, Random? random = null) : IUnembeddingLayer<char>
{
    public string Tokens { get; } = tokens;
    public bool WeightedRandom { get; } = weightedRandom;
    public Random Random { get; } = random ?? Random.Shared;

    public int InputNodeCount => Tokens.Length;
    public long ParameterCount => 0;

    public (char, Weight) Forward(Vector input)
    {
        var (result, index, weights) = Forward(input, default!);
        return (result, weights[index]);
    }

    private static int GetWeightedRandomIndex(Vector weights, Random random)
    {
        var value = random.NextDouble();
        for (int i = 0; i < weights.Count; i++)
        {
            value -= weights[i];
            if (value < 0) return i;
        }
        return weights.Count - 1;
    }

    public (char output, int index, Vector weights) Forward(Vector input, ILayerSnapshot snapshot)
    {
        Debug.Assert(input.Count == Tokens.Length);

        // temperature adjustments
        if (WeightedRandom)
        {
            // cannot work on self
            // input.PointwiseLogToSelf();
            // input.DivideToSelf(temperature);
            // input.SoftMaxToSelf();
        }

        var index = WeightedRandom ? GetWeightedRandomIndex(input, Random) : input.MaximumIndex();
        return (Tokens[index], index, input);
    }
}
