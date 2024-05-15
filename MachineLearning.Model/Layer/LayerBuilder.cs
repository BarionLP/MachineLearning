using MachineLearning.Domain.Activation;
using MachineLearning.Model.Layer.Initialization;

namespace MachineLearning.Model.Layer;

public sealed class LayerBuilder<TLayer>(int inputNodeCount, int outputNodeCount) where TLayer : ILayer<Number>
{
    public int OutputNodeCount { get; } = outputNodeCount;
    public int InputNodeCount { get; } = inputNodeCount;
    public Number[,] Weights { get; } = new Number[inputNodeCount, outputNodeCount];
    public Number[] Biases { get; } = new Number[outputNodeCount];
    public IActivationMethod<Number> ActivationMethod { get; set; } = SigmoidActivation.Instance;

    public LayerBuilder<TLayer> SetActivationMethod(IActivationMethod<Number> activationMethod) {
        ActivationMethod = activationMethod;
        return this;
    }

    public LayerBuilder<TLayer> Initialize(ILayerInitializer<double> initializer)
    {
        initializer.Initialize(Weights, Biases);
        return this;
    }

    public TLayer Build() => (TLayer)TLayer.Create(Weights.Copy(), Biases.Copy(), ActivationMethod);
}
