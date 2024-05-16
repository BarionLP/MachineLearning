using MachineLearning.Domain.Activation;

namespace MachineLearning.Model.Layer;

public interface ILayer<TWeights> where TWeights : struct, IEquatable<TWeights>, IFormattable
{
    public int InputNodeCount { get; }
    public int OutputNodeCount { get; }
    public Matrix<TWeights> Weights { get; }
    public Vector<TWeights> Biases { get; }
    public IActivationMethod<TWeights> ActivationFunction { get; }

    public Vector<TWeights> Forward(Vector<TWeights> input);

    public virtual static ILayer<TWeights> Create(Matrix<TWeights> weights, Vector<TWeights> biases, IActivationMethod<TWeights> activationMethod) => throw new NotImplementedException();
}
