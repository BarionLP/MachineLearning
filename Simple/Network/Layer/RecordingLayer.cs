using Simple.Network.Activation;

namespace Simple.Network.Layer;

public sealed class RecordingLayer(Number[,] Weights, Number[] Biases, IActivationMethod<Number> ActivationMethod) : ILayer<Number> {
    public int InputNodeCount { get; } = Weights.GetLength(0);
    public int OutputNodeCount { get; } = Biases.Length;
    public Number[,] Weights { get; } = Weights;
    public Number[] Biases { get; } = Biases;

    public IActivationMethod<Number> ActivationMethod { get; } = ActivationMethod;

    public Number[] LastRawInput = [];
    public readonly Number[] LastWeightedInput = new Number[Biases.Length]; // OutputNodeCount
    public Number[] LastActivatedWeights = [];


    public Number[] Process(Number[] input){
        LastRawInput = input;

        //for each output node sum up the products of each input node times the weight assigned to that connection, finally add the bias of the output node
        foreach(int outputNodeIndex in ..OutputNodeCount){
            LastWeightedInput[outputNodeIndex] = Biases[outputNodeIndex];
            foreach (int inputNodeIndex in ..InputNodeCount){
                LastWeightedInput[outputNodeIndex] += input[inputNodeIndex] * Weights[inputNodeIndex, outputNodeIndex];
            }
        }

        LastActivatedWeights = ActivationMethod.Activate(LastWeightedInput);

        return LastActivatedWeights;
    }

    public static ILayer<Number> Create(Number[,] weights, Number[] biases, IActivationMethod<Number> activationMethod) => new RecordingLayer(weights, biases, activationMethod);
}