using MachineLearning.Domain.Activation;

namespace MachineLearning.Model.Layer;

public interface ILayer 
{
    public int InputNodeCount { get; }
    public int OutputNodeCount { get; }
    public Matrix Weights { get; }
    public Vector Biases { get; }
    public IActivationMethod ActivationFunction { get; }

    public Vector Forward(Vector input);

    public virtual static ILayer Create(Matrix weights, Vector biases, IActivationMethod activationMethod) => throw new NotImplementedException();
}
