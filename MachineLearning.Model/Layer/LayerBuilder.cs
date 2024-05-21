using MachineLearning.Domain.Activation;
using MachineLearning.Model.Layer.Initialization;

namespace MachineLearning.Model.Layer;

public sealed class LayerBuilder<TLayer>(int inputNodeCount, int outputNodeCount) where TLayer : ILayer
{
    public int OutputNodeCount { get; } = outputNodeCount;
    public int InputNodeCount { get; } = inputNodeCount;
    public Matrix Weights { get; } = Matrix.Create(outputNodeCount, inputNodeCount);
    public Vector Biases { get; } = Vector.Create(outputNodeCount);
    public IActivationMethod ActivationMethod { get; set; } = SigmoidActivation.Instance;

    public LayerBuilder<TLayer> SetActivationMethod(IActivationMethod activationMethod) {
        ActivationMethod = activationMethod;
        return this;
    }

    public LayerBuilder<TLayer> Initialize(ILayerInitializer initializer)
    {
        initializer.Initialize(Weights, Biases);
        return this;
    }

    public TLayer Build() => (TLayer)TLayer.Create(Weights.Copy(), Biases.Copy(), ActivationMethod);
}
