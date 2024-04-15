using Simple.Network.Activation;

namespace Simple.Network.Layer;

public sealed class SimpleLayer(Number[,] Weights, Number[] Biases, IActivation Activation) : ILayer<Number> {
    public int InputNodeCount { get; } = Biases.Length;
    public int OutputNodeCount { get; } = Weights.GetLength(0);
    public Number[,] Weights { get; } = Weights;
    public Number[] Biases { get; }  = Biases;
    public IActivation ActivationMethod { get; } = Activation;

    public Number[] Process(Number[] input) {
        var weighted = new Number[OutputNodeCount];

        //for each output node sum up the products of each input node times the weight assigned to that connection, finally add the bias of the output node
        foreach(int outputNodeIndex in ..OutputNodeCount) {
            weighted[outputNodeIndex] = Biases[outputNodeIndex];
            foreach(int inputNodeIndex in ..InputNodeCount) {
                weighted[outputNodeIndex] += input[inputNodeIndex] * Weights[inputNodeIndex, outputNodeIndex];
            }
        }

        return ActivationMethod.Activate(weighted); //TODO: operate on weighted directly instead of creating a copy
    }

    public static SimpleLayer Of(RecordingLayer parent) {
        return new(parent.Weights, parent.Biases, parent.ActivationMethod);
    }
}
