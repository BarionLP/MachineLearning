using MachineLearning.Model.Initialization;

namespace MachineLearning.Model.Layer.Initialization;

/// <summary>
/// initializes randomly from a normal distribution.<br/>
/// not recommended
/// </summary>
public sealed class RandomInitializer(Random? random = null) : ILayerInitializer
{
    public Random Random { get; } = random ?? Random.Shared;

    public void Initialize(Matrix weights, Vector biases)
    {
        var sqrtInputNodeCount = Math.Sqrt(weights.ColumnCount);

        weights.MapToSelf(v => InitializationHelper.RandomInNormalDistribution(Random, 0, 1) / sqrtInputNodeCount);
        biases.MapInPlace(v => InitializationHelper.RandomInNormalDistribution(Random, 0, 0.1));
    }
}
