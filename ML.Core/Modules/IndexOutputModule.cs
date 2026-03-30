using ML.Core.Attributes;

namespace ML.Core.Modules;

[GeneratedModule(IncludeSerializer: true)]
public sealed partial class IndexOutputModule(int tokenCount, bool weightedRandom, Random? random = null) : IOutputModule<Vector, int, EmptyModuleData, EmptyModuleData>
{
    [Property] public int TokenCount { get; } = tokenCount;
    [Property] public bool WeightedRandom { get; } = weightedRandom;
    public Random Random { get; } = random ?? Random.Shared;

    public (int Output, float Confidence, Vector Weights) Forward(Vector input, EmptyModuleData snapshot)
    {
        Debug.Assert(input.Count == TokenCount);

        var index = WeightedRandom ? GetWeightedRandomIndex(input, Random) : input.MaximumIndex();
        return (index, input[index], input);
    }

    public Vector Backward(Vector outputGradient, EmptyModuleData snapshot, EmptyModuleData gradients) => outputGradient;

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
}