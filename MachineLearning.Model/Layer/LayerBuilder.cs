using MachineLearning.Domain.Activation;
using MachineLearning.Model.Layer.Initialization;

namespace MachineLearning.Model.Layer;

public sealed class LayerBuilder<TLayer>(int inputNodeCount, int outputNodeCount) where TLayer : ILayer<double>
{
    public int OutputNodeCount { get; } = outputNodeCount;
    public int InputNodeCount { get; } = inputNodeCount;
    public Matrix<double> Weights { get; } = Matrix.Build.Dense(outputNodeCount, inputNodeCount);
    public Vector<double> Biases { get; } = Vector.Build.Dense(outputNodeCount);
    public IActivationMethod<double> ActivationMethod { get; set; } = SigmoidActivation.Instance;

    public LayerBuilder<TLayer> SetActivationMethod(IActivationMethod<double> activationMethod) {
        ActivationMethod = activationMethod;
        return this;
    }

    public LayerBuilder<TLayer> Initialize(ILayerInitializer<double> initializer)
    {
        initializer.Initialize(Weights, Biases);
        return this;
    }

    public TLayer Build() => (TLayer)TLayer.Create(Weights.Clone(), Biases.Clone(), ActivationMethod);
}
