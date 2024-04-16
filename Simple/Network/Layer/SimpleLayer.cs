using Simple.Network.Activation;

namespace Simple.Network.Layer;

public sealed class SimpleLayer(Number[,] Weights, Number[] Biases, IActivationMethod Activation) : ILayer<Number> {
    public int InputNodeCount { get; } = Weights.GetLength(0);
    public int OutputNodeCount { get; } = Biases.Length;
    public Number[,] Weights { get; } = Weights;
    public Number[] Biases { get; } = Biases;

    public IActivationMethod ActivationMethod { get; } = Activation;

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
}
