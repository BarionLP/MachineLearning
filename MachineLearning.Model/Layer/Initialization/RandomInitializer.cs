using MachineLearning.Model.Initialization;

namespace MachineLearning.Model.Layer.Initialization;

/// <summary>
/// initializes randomly from a normal distribution.<br/>
/// not recommended
/// </summary>
public sealed class RandomInitializer(Random? random = null) : IInitializer<FeedForwardLayer>
{
    public Random Random { get; } = random ?? Random.Shared;

    public void Initialize(FeedForwardLayer layer)
    {
        var sqrtInputNodeCount = MathF.Sqrt(layer.Weights.ColumnCount);

        layer.Weights.MapToSelf(v => InitializationHelper.RandomInNormalDistribution(Random, 0f, 1f) / sqrtInputNodeCount);
        layer.Biases.MapToSelf(v => InitializationHelper.RandomInNormalDistribution(Random, 0f, 0.1f));
    }
}
