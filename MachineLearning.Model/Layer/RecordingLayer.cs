using MachineLearning.Domain.Activation;

namespace MachineLearning.Model.Layer;

public sealed class RecordingLayer(Matrix<double> Weights, Vector<double> Biases, IActivationMethod<double> ActivationMethod) : ILayer<double>
{
    public int InputNodeCount { get; } = Weights.ColumnCount;
    public int OutputNodeCount { get; } = Biases.Count;
    public Matrix<double> Weights { get; } = Weights;
    public Vector<double> Biases { get; } = Biases;

    public IActivationMethod<double> ActivationFunction { get; } = ActivationMethod;

    public Vector<double> LastRawInput = Vector.Build.Dense(0);
    public Vector<double> LastWeightedInput = Vector.Build.Dense(0); // OutputNodeCount
    public Vector<double> LastActivatedWeights = Vector.Build.Dense(0);


    public Vector<double> Forward(Vector<double> input)
    {
        LastRawInput = input;
        LastWeightedInput = (Weights * input) + Biases;

        LastActivatedWeights = ActivationFunction.Activate(LastWeightedInput);

        return LastActivatedWeights;
    }

    public static ILayer<double> Create(Matrix<double> weights, Vector<double> biases, IActivationMethod<double> activationMethod) => new RecordingLayer(weights, biases, activationMethod);
}