namespace MachineLearning.Model.Layer.Initialization;

/// <summary>
/// initializes from a normal distribution.<br/>
/// not recommended
/// </summary>
public sealed class RandomInitializer(Random? random = null) : ILayerInitializer<double>
{
    public Random Random { get; } = random ?? Random.Shared;

    public void Initialize(double[,] weights, double[] biases)
    {
        var sqrtInputNodeCount = Math.Sqrt(weights.GetLength(0));

        for (int outputNodeIndex = 0; outputNodeIndex < biases.Length; outputNodeIndex++)
        {
            for (int inputNodeIndex = 0; inputNodeIndex < weights.GetLength(0); inputNodeIndex++)
            {
                weights[inputNodeIndex, outputNodeIndex] = LayerInitializationHelper.RandomInNormalDistribution(Random, 0, 1) / sqrtInputNodeCount;
            }
            biases[outputNodeIndex] = LayerInitializationHelper.RandomInNormalDistribution(Random, 0, 0.1);
        }
    }
}
