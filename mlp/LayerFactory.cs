using MachineLearning.Model.Activation;
using MachineLearning.Model.Layer.Initialization;

namespace ML.MultiLayerPerceptron;

public sealed class LayerFactory(int inputNodeCount, int outputNodeCount)
{
    public int OutputNodeCount { get; } = outputNodeCount;
    public int InputNodeCount { get; } = inputNodeCount;
    public IActivationFunction ActivationFunction { get; set; } = SigmoidActivation.Instance;
    public IInitializer<PerceptronLayer> Initializer { get; set; } = NoInitializer<PerceptronLayer>.Instance;

    public LayerFactory SetActivationFunction(IActivationFunction activationMethod)
    {
        ActivationFunction = activationMethod;
        return this;
    }

    public LayerFactory SetInitializer(IInitializer<PerceptronLayer> initializer)
    {
        Initializer = initializer;
        return this;
    }

    public PerceptronLayer Create()
    {
        var layer = new PerceptronLayer(ActivationFunction, InputNodeCount, OutputNodeCount);
        Initializer.Initialize(layer);
        return layer;
    }
}
