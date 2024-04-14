using Simple.Activation;

namespace Simple;

public sealed class Layer {
    internal readonly int _inputNodeCount;
    internal readonly int _outputNodeCount;
    internal readonly Number[,] _weights;
    internal readonly Number[] _biases;

    internal readonly Number[] _weightedInput;
    internal Number[] _activationWeights = [];
    internal Number[] _input = [];

    internal readonly IActivation _activation = SigmoidActivation.Instance;

    public Layer(int inputNodeCount, int outputNodeCount) {
        _inputNodeCount = inputNodeCount;
        _outputNodeCount = outputNodeCount;
        _weights = new Number[inputNodeCount, outputNodeCount];
        _biases = new Number[outputNodeCount];

        _weightedInput = new Number[outputNodeCount];

        InitializeRandomWeights();
    }

    public Number[] Process(Number[] input) {
        _input = input;

        foreach(int outputNodeIndex in .._outputNodeCount) {
            var weightedNode = _biases[outputNodeIndex];
            foreach(int inputNodeIndex in .._inputNodeCount) {
                weightedNode += input[inputNodeIndex] * _weights[inputNodeIndex, outputNodeIndex];
            }
            _weightedInput[outputNodeIndex] = weightedNode;
        }

        _activationWeights = _activation.Function(_weightedInput); 

        return _activationWeights;
    }

    private void InitializeRandomWeights() {
        foreach(int outputNodeIndex in .._outputNodeCount) {
            foreach(int inputNodeIndex in .._inputNodeCount) {
                _weights[inputNodeIndex, outputNodeIndex] = RandomInNormalDistribution(Random.Shared, 0, 1) / Math.Sqrt(_inputNodeCount);
            }
            _biases[outputNodeIndex] = RandomInNormalDistribution(Random.Shared, 0, 0.1);
        }

        static Number RandomInNormalDistribution(Random rng, Number mean, Number standardDeviation) {
            var x1 = 1 - rng.NextDouble();
            var x2 = 1 - rng.NextDouble();

            var y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
            return y1 * standardDeviation + mean;
        }
    }
}

public sealed class Node {
    public Number Bias;
    public Number[] Weights = [];
}