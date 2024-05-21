using MachineLearning.Domain.Activation;

namespace MachineLearning.Model.Layer;

public sealed class SimpleLayer(Matrix Weights, Vector Biases, IActivationMethod Activation) : ILayer
{
    public int InputNodeCount { get; } = Weights.ColumnCount;
    public int OutputNodeCount { get; } = Biases.Count;
    public Matrix Weights { get; } = Weights; // output * input!!
    public Vector Biases { get; } = Biases;

    public IActivationMethod ActivationFunction { get; } = Activation;

    public Vector Forward(Vector input)
    {
        // // TODO: can i just operate on input?
        Weights.Multiply(input, input);
        input.AddInPlace(Biases);
        ActivationFunction.Activate(input, input);

        return input;
    }

    public static ILayer Create(Matrix weights, Vector biases, IActivationMethod activationMethod) => new SimpleLayer(weights, biases, activationMethod);
}
