using MachineLearning.Model.Activation;
using MachineLearning.Model.Layer.Initialization;

namespace MachineLearning.Model.Layer;

public sealed class LayerFactory(int inputNodeCount, int outputNodeCount)
{
    public int OutputNodeCount { get; } = outputNodeCount;
    public int InputNodeCount { get; } = inputNodeCount;
    public IActivationFunction ActivationFunction { get; set; } = SigmoidActivation.Instance;
    public IInitializer<FeedForwardLayer> Initializer { get; set; } = NoInitializer<FeedForwardLayer>.Instance;

    public LayerFactory SetActivationFunction(IActivationFunction activationMethod)
    {
        ActivationFunction = activationMethod;
        return this;
    }

    public LayerFactory SetInitializer(IInitializer<FeedForwardLayer> initializer)
    {
        Initializer = initializer;
        return this;
    }

    public FeedForwardLayer Create()
    {
        var layer = new FeedForwardLayer(Matrix.Create(OutputNodeCount, InputNodeCount), Vector.Create(OutputNodeCount), ActivationFunction);
        Initializer.Initialize(layer);
        return layer;
    }
}
