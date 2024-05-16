namespace MachineLearning.Model.Layer.Initialization;

/// <summary>
/// initializes from a normal distribution.<br/>
/// not recommended
/// </summary>
public sealed class RandomInitializer(Random? random = null) : ILayerInitializer<double>
{
    public Random Random { get; } = random ?? Random.Shared;

    public void Initialize(Matrix<double> weights, Vector<double> biases)
    {
        var sqrtInputNodeCount = Math.Sqrt(weights.ColumnCount);

        weights.MapInplace(v => LayerInitializationHelper.RandomInNormalDistribution(Random, 0, 1) / sqrtInputNodeCount);
        biases.MapInplace(v => LayerInitializationHelper.RandomInNormalDistribution(Random, 0, 0.1));
    }
}
