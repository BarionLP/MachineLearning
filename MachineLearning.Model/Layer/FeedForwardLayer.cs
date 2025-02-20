using MachineLearning.Model.Activation;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model.Layer;

public sealed class FeedForwardLayer(Matrix Weights, Vector Biases, IActivationFunction Activation) : ILayer
{
    public int InputNodeCount { get; } = Weights.ColumnCount;
    public int OutputNodeCount { get; } = Biases.Count;
    public Matrix Weights { get; } = Weights; // output * input!!
    public Vector Biases { get; } = Biases;

    public IActivationFunction ActivationFunction { get; } = Activation;

    public Vector Forward(Vector input)
    {
        var result = Weights.Multiply(input);
        result.AddToSelf(Biases);
        ActivationFunction.ActivateTo(result, result);

        return result;
    }

    public Vector Forward(Vector input, LayerSnapshots.Simple snapshot)
    {
        input.CopyTo(snapshot.LastRawInput);
        Weights.MultiplyTo(input, snapshot.LastWeightedInput);
        snapshot.LastWeightedInput.AddToSelf(Biases);

        ActivationFunction.ActivateTo(snapshot.LastWeightedInput, snapshot.LastActivatedWeights);

        return snapshot.LastActivatedWeights;
    }

    public long ParameterCount => Biases.Count + Weights.FlatCount;
    public ILayerSnapshot CreateSnapshot() => new LayerSnapshots.Simple(InputNodeCount, OutputNodeCount);
}
