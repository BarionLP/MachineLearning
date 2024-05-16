using MachineLearning.Domain.Activation;

namespace MachineLearning.Model.Layer;

public sealed class SimpleLayer(Matrix<double> Weights, Vector<double> Biases, IActivationMethod<double> Activation) : ILayer<double>
{
    public int InputNodeCount { get; } = Weights.ColumnCount;
    public int OutputNodeCount { get; } = Biases.Count;
    public Matrix<double> Weights { get; } = Weights; // output * input!!
    public Vector<double> Biases { get; } = Biases;

    public IActivationMethod<double> ActivationFunction { get; } = Activation;

    public Vector<double> Forward(Vector<double> input)
    {
        var weightedSum = Weights * input;
        var output = weightedSum + Biases;

        return ActivationFunction.Activate(output); //TODO: operate on output directly instead of creating a copy
    }

    public static ILayer<double> Create(Matrix<double> weights, Vector<double> biases, IActivationMethod<double> activationMethod) => new SimpleLayer(weights, biases, activationMethod);
}
