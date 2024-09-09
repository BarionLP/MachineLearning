using MachineLearning.Model.Activation;
using MachineLearning.Model.Layer.Initialization;

namespace MachineLearning.Model.Layer;

public sealed class LayerBuilder(int inputNodeCount, int outputNodeCount)
{
    public int OutputNodeCount { get; } = outputNodeCount;
    public int InputNodeCount { get; } = inputNodeCount;
    public Matrix Weights { get; } = Matrix.Create(outputNodeCount, inputNodeCount);
    public Vector Biases { get; } = Vector.Create(outputNodeCount);
    public IActivationMethod ActivationMethod { get; set; } = SigmoidActivation.Instance;

    public LayerBuilder SetActivationMethod(IActivationMethod activationMethod)
    {
        ActivationMethod = activationMethod;
        return this;
    }

    public LayerBuilder Initialize(ILayerInitializer initializer)
    {
        initializer.Initialize(Weights, Biases);
        return this;
    }

    public SimpleLayer Build() => new (Weights.Copy(), Biases.Copy(), ActivationMethod);
}
