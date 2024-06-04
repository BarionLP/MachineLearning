using MachineLearning.Domain.Activation;

namespace MachineLearning.Model.Layer;

public sealed class RecordingLayer(Matrix Weights, Vector Biases, IActivationMethod ActivationMethod) : ILayer
{
    public int InputNodeCount { get; } = Weights.ColumnCount;
    public int OutputNodeCount { get; } = Biases.Count;
    public Matrix Weights { get; } = Weights;
    public Vector Biases { get; } = Biases;

    public IActivationMethod ActivationFunction { get; } = ActivationMethod;

    public Vector LastRawInput = Vector.Empty;
    public Vector LastWeightedInput = Vector.Create(Biases.Count);
    public Vector LastActivatedWeights = Vector.Create(Biases.Count);


    public Vector Forward(Vector input)
    {
        LastRawInput = input;

        Weights.Multiply(input, LastWeightedInput);
        LastWeightedInput.AddInPlace(Biases);

        ActivationFunction.Activate(LastWeightedInput, LastActivatedWeights);

        return LastActivatedWeights;
    }

    public static ILayer Create(Matrix weights, Vector biases, IActivationMethod activationMethod) => new RecordingLayer(weights, biases, activationMethod);
}