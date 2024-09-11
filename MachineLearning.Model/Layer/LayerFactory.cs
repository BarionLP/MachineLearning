using MachineLearning.Model.Activation;
using MachineLearning.Model.Layer.Initialization;

namespace MachineLearning.Model.Layer;

public sealed class LayerFactory(int inputNodeCount, int outputNodeCount)
{
    public int OutputNodeCount { get; } = outputNodeCount;
    public int InputNodeCount { get; } = inputNodeCount;
    public IActivationFunction ActivationFunction { get; set; } = SigmoidActivation.Instance;
    public ILayerInitializer<SimpleLayer> Initializer { get; set; } = ZeroInitializer<SimpleLayer>.Instance;

    public LayerFactory SetActivationFunction(IActivationFunction activationMethod)
    {
        ActivationFunction = activationMethod;
        return this;
    }

    public LayerFactory Initialize(ILayerInitializer<SimpleLayer> initializer)
    {
        Initializer = initializer;
        return this;
    }

    public SimpleLayer Create()
    {
        var layer = new SimpleLayer(Matrix.Create(OutputNodeCount, InputNodeCount), Vector.Create(OutputNodeCount), ActivationFunction);
        Initializer.Initialize(layer);
        return layer;
    }
}
