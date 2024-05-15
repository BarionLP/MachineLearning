namespace MachineLearning.Model.Layer.Initialization;

/// <summary>
/// suited for ReLU activations
/// </summary>
public sealed class HeInitializer(Random? random = null) : ILayerInitializer<double>
{
    public Random Random { get; } = random ?? Random.Shared;

    public void Initialize(double[,] weights, double[] biases)
    {
        var inputCount = weights.GetLength(0);
        var standartDeviation = Math.Sqrt(2.0 / inputCount);

        for (int outputNodeIndex = 0; outputNodeIndex < biases.Length; outputNodeIndex++)
        {
            for (int inputNodeIndex = 0; inputNodeIndex < inputCount; inputNodeIndex++)
            {
                weights[inputNodeIndex, outputNodeIndex] = LayerInitializationHelper.RandomInNormalDistribution(Random, 0, standartDeviation);
            }
            biases[outputNodeIndex] = LayerInitializationHelper.RandomInNormalDistribution(Random, 0, 0.1);
        }
    }
}
