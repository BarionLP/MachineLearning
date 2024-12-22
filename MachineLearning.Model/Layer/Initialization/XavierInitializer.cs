using MachineLearning.Model.Initialization;

namespace MachineLearning.Model.Layer.Initialization;

/// <summary>
/// suited for sigmoid, tanh and softmax activations
/// </summary>
public sealed class XavierInitializer(Random? random = null) : IInitializer<FeedForwardLayer>
{
    public static XavierInitializer Instance { get; } = new();
    public Random Random { get; } = random ?? Random.Shared;

    public void Initialize(FeedForwardLayer layer)
    {
        var inputCount = layer.Weights.ColumnCount;
        var outputCount = layer.Biases.Count;
        var standardDeviation = MathF.Sqrt(2.0f / (inputCount + outputCount));

        layer.Weights.MapToSelf(v => InitializationHelper.RandomInNormalDistribution(Random, 0, standardDeviation));
        layer.Biases.MapToSelf(v => InitializationHelper.RandomInNormalDistribution(Random, 0, 0.1f));
    }
}
