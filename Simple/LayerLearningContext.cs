namespace Simple;

public sealed class LayerLearningContext {
    internal readonly Layer _layer;
    private readonly Number[,] _costGradientWeights;
    private readonly Number[] _costGradientBias;

    public LayerLearningContext(Layer layer) {
        _layer = layer;
        _costGradientWeights = new Number[_layer._inputNodeCount, _layer._outputNodeCount];
        _costGradientBias = new Number[_layer._outputNodeCount];
    }

    public void UpdateGradients(Number[] nodeValues) {
        foreach(int outputNodeIndex in .._layer._outputNodeCount) {
            foreach(int inputNodeIndex in .._layer._inputNodeCount) {
                // partial derivative cost / weight of current connection
                var derivativeCostWrtWeight = _layer._input[inputNodeIndex] * nodeValues[outputNodeIndex];
                _costGradientWeights[inputNodeIndex, outputNodeIndex] += derivativeCostWrtWeight;
            }

            // derivative cost / bias (bias' = 1)
            var derivativeCostWrtBias = 1 * nodeValues[outputNodeIndex];
            _costGradientBias[outputNodeIndex] += derivativeCostWrtBias;
        }
    }

    public void ApplyGradients(Number learnRate) {
        foreach(int outputNodeIndex in .._layer._outputNodeCount) {
            _layer._biases[outputNodeIndex] -= _costGradientBias[outputNodeIndex] * learnRate;
            foreach(int inputNodeIndex in .._layer._inputNodeCount) {
                _layer._weights[inputNodeIndex, outputNodeIndex] -= _costGradientWeights[inputNodeIndex, outputNodeIndex] * learnRate; //was assignment in video! mistake?
            }
        }
    }

    public void ResetGradients() {
        foreach(var outputNodeIndex in .._layer._outputNodeCount) {
            _costGradientBias[outputNodeIndex] = 0;
            foreach(var inputNodeIndex in .._layer._inputNodeCount) {
                _costGradientWeights[inputNodeIndex, outputNodeIndex] = 0;
            }
        }
    }

    public Number[] CalculateOutputLayerNodeValues(Number[] expected) {
        var nodeValues = new Number[expected.Length];

        foreach(int i in ..expected.Length) {
            var costDerivative = NodeHelper.NodeCostDerivative(_layer._activationWeights[i], expected[i]);
            var activationDerivative = _layer._activation.Derivative(_layer._weightedInput[i]);
            nodeValues[i] = costDerivative * activationDerivative;
        }

        return nodeValues;
    }

    public Number[] CalculateHiddenLayerNodeValues(Layer oldLayer, Number[] oldNodeValues) {
        var newNodeValues = new Number[_layer._outputNodeCount];

        foreach(int newNodeIndex in ..newNodeValues.Length) {
            var newNodeValue = 0d;
            
            foreach(int oldNodeIndex in ..oldNodeValues.Length) {
                // Partial derivative of the weighted input with respect to the input
                var weightedInputDerivative = oldLayer._weights[newNodeIndex, oldNodeIndex];
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }
            
            newNodeValue *= _layer._activation.Derivative(_layer._weightedInput[newNodeIndex]);
            newNodeValues[newNodeIndex] = newNodeValue;
        }

        return newNodeValues;
    }
}
