namespace MachineLearning.Model.Layer.Initialization;

/// <summary>
/// suited for sigmoid, tanh and softmax activations
/// </summary>
public sealed class XavierInitializer(Random? random = null) : ILayerInitializer<double>
{
    public Random Random { get; } = random ?? Random.Shared;

    public void Initialize(double[,] weights, double[] biases)
    {
        var inputCount = weights.GetLength(0);
        var outputCount = biases.Length;
        var stddev = Math.Sqrt(2.0 / (inputCount + outputCount));

        for (int outputNodeIndex = 0; outputNodeIndex < inputCount; outputNodeIndex++)
        {
            for (int inputNodeIndex = 0; inputNodeIndex < outputCount; inputNodeIndex++)
            {
                weights[inputNodeIndex, outputNodeIndex] = LayerInitializationHelper.RandomInNormalDistribution(Random, 0, stddev);
            }
            biases[outputNodeIndex] = LayerInitializationHelper.RandomInNormalDistribution(Random, 0, 0.1);
        }
    }
}
