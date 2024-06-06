using MachineLearning.Domain.Activation;

namespace MachineLearning.Model.Layer;

public sealed class SimpleLayer(Matrix Weights, Vector Biases, IActivationMethod Activation)
{
    public int InputNodeCount { get; } = Weights.ColumnCount;
    public int OutputNodeCount { get; } = Biases.Count;
    public Matrix Weights { get; } = Weights; // output * input!!
    public Vector Biases { get; } = Biases;

    public IActivationMethod ActivationFunction { get; } = Activation;

    public Vector Forward(Vector input)
    {
        // TODO: can i just operate on input?
        var result = Weights.Multiply(input);
        result.AddInPlace(Biases);
        ActivationFunction.Activate(result, result);

        return result;
    }
}
