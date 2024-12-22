using MachineLearning.Model.Initialization;

namespace MachineLearning.Model.Layer.Initialization;

/// <summary>
/// suited for ReLU activations
/// </summary>
public sealed class HeInitializer(Random? random = null) : IInitializer<FeedForwardLayer>
{
    public static HeInitializer Instance { get; } = new HeInitializer();
    public Random Random { get; } = random ?? Random.Shared;

    public void Initialize(FeedForwardLayer layer)
    {
        var inputCount = layer.Weights.ColumnCount;
        var standardDeviation = MathF.Sqrt(2.0f / inputCount);

        layer.Weights.MapToSelf(v => InitializationHelper.RandomInNormalDistribution(Random, 0, standardDeviation));
        layer.Biases.MapToSelf(v => InitializationHelper.RandomInNormalDistribution(Random, 0, 0.1f));
    }
}
