namespace MachineLearning.Model.Layer.Initialization;

/// <summary>
/// suited for sigmoid, tanh and softmax activations
/// </summary>
public sealed class XavierInitializer(Random? random = null) : ILayerInitializer
{
    public Random Random { get; } = random ?? Random.Shared;

    public void Initialize(Matrix weights, Vector biases)
    {
        var inputCount = weights.ColumnCount;
        var outputCount = biases.Count;
        var standartDeviation = Math.Sqrt(2.0 / (inputCount + outputCount));

        weights.MapInPlace(v => LayerInitializationHelper.RandomInNormalDistribution(Random, 0, standartDeviation));
        biases.MapInPlace(v => LayerInitializationHelper.RandomInNormalDistribution(Random, 0, 0.1));
    }
}
