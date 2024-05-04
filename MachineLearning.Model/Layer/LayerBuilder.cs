using MachineLearning.Domain.Activation;

namespace MachineLearning.Model.Layer;

public sealed class LayerBuilder<TLayer>(int inputNodeCount, int outputNodeCount) where TLayer : ILayer<Number>
{
    public Number[,] Weights { get; } = new Number[inputNodeCount, outputNodeCount];
    public Number[] Biases { get; } = new Number[outputNodeCount];
    public IActivationMethod<Number> ActivationMethod { get; set; } = SigmoidActivation.Instance;

    public LayerBuilder<TLayer> SetActivationMethod(IActivationMethod<Number> activationMethod)
    {
        ActivationMethod = activationMethod;
        return this;
    }

    public LayerBuilder<TLayer> Initialize(int defaultValue) => Initialize(defaultValue, defaultValue);
    public LayerBuilder<TLayer> Initialize(int weightDefault, int biasDefault)
    {
        foreach (int outputNodeIndex in ..Biases.Length)
        {
            foreach (int inputNodeIndex in ..Weights.GetLength(0))
            {
                Weights[inputNodeIndex, outputNodeIndex] = weightDefault;
            }
            Biases[outputNodeIndex] = biasDefault;
        }
        return this;
    }

    public LayerBuilder<TLayer> InitializeRandom(Random? random = null)
    {
        var sqrtInputNodeCount = Math.Sqrt(Weights.GetLength(0));
        random ??= Random.Shared;
        foreach (int outputNodeIndex in ..Biases.Length)
        {
            foreach (int inputNodeIndex in ..Weights.GetLength(0))
            {
                Weights[inputNodeIndex, outputNodeIndex] = RandomInNormalDistribution(random, 0, 1) / sqrtInputNodeCount;
            }
            Biases[outputNodeIndex] = RandomInNormalDistribution(random, 0, 0.1);
        }
        return this;

        static Number RandomInNormalDistribution(Random random, Number mean, Number standardDeviation)
        {
            var x1 = 1 - random.NextDouble();
            var x2 = 1 - random.NextDouble();

            var y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
            return y1 * standardDeviation + mean;
        }
    }

    public TLayer Build() => (TLayer)TLayer.Create(Weights, Biases, ActivationMethod);
}
