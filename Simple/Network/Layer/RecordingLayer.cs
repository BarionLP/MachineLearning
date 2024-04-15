using Simple.Network.Activation;

namespace Simple.Network.Layer;

public sealed class RecordingLayer : ILayer<Number> {
    public int InputNodeCount { get; }
    public int OutputNodeCount { get; }
    public Number[,] Weights { get; }
    public Number[] Biases { get; }

    public Number[] WeightedInput { get; }
    public Number[] ActivatedWeights = [];
    public Number[] LastInput = [];

    public IActivation ActivationMethod { get; init; } = SigmoidActivation.Instance;

    public RecordingLayer(int inputNodeCount, int outputNodeCount){
        InputNodeCount = inputNodeCount;
        OutputNodeCount = outputNodeCount;
        Weights = new Number[inputNodeCount, outputNodeCount];
        Biases = new Number[outputNodeCount];

        WeightedInput = new Number[outputNodeCount];

        InitializeRandomWeights();
    }

    public Number[] Process(Number[] input){
        LastInput = input;

        //for each output node sum up the products of each input node times the weight assigned to that connection, finally add the bias of the output node
        foreach(int outputNodeIndex in ..OutputNodeCount){
            WeightedInput[outputNodeIndex] = Biases[outputNodeIndex];
            foreach (int inputNodeIndex in ..InputNodeCount){
                WeightedInput[outputNodeIndex] += input[inputNodeIndex] * Weights[inputNodeIndex, outputNodeIndex];
            }
        }

        ActivatedWeights = ActivationMethod.Activate(WeightedInput);

        return ActivatedWeights;
    }

    private void InitializeRandomWeights(){
        foreach (int outputNodeIndex in ..OutputNodeCount){
            foreach (int inputNodeIndex in ..InputNodeCount){
                Weights[inputNodeIndex, outputNodeIndex] = RandomInNormalDistribution(Random.Shared, 0, 1) / Math.Sqrt(InputNodeCount);
            }
            Biases[outputNodeIndex] = RandomInNormalDistribution(Random.Shared, 0, 0.1);
        }

        static Number RandomInNormalDistribution(Random rng, Number mean, Number standardDeviation){
            var x1 = 1 - rng.NextDouble();
            var x2 = 1 - rng.NextDouble();

            var y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
            return y1 * standardDeviation + mean;
        }
    }
}