using MachineLearning.Model.Initialization;

namespace MachineLearning.Model.Layer.Initialization;

/// <summary>
/// suited for ReLU activations
/// </summary>
public sealed class HeInitializer(Random? random = null) : ILayerInitializer
{
    public static HeInitializer Instance { get; } = new HeInitializer();
    public Random Random { get; } = random ?? Random.Shared;

    public void Initialize(Matrix weights, Vector biases)
    {
        var inputCount = weights.ColumnCount;
        var standartDeviation = Math.Sqrt(2.0 / inputCount);

        weights.MapToSelf(v => InitializationHelper.RandomInNormalDistribution(Random, 0, standartDeviation));
        biases.MapInPlace(v => InitializationHelper.RandomInNormalDistribution(Random, 0, 0.1));
    }
}
