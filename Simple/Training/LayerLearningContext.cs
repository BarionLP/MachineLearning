using Simple.Network.Layer;
using Simple.Training.Cost;

namespace Simple.Training;

internal sealed class LayerLearningContext {
    public RecordingLayer Layer { get; }
    public readonly Number[,] CostGradientWeights;
    public readonly Number[] CostGradientBiases;
    public ICostFunction CostFunction { get; init; } = MeanSquaredErrorCost.Instance;

    public LayerLearningContext(RecordingLayer layer) {
        Layer = layer;
        CostGradientWeights = new Number[Layer.InputNodeCount, Layer.OutputNodeCount];
        CostGradientBiases = new Number[Layer.OutputNodeCount];
    }

    public void UpdateGradients(Number[] nodeValues) {
        foreach(int outputNodeIndex in ..Layer.OutputNodeCount) {
            foreach(int inputNodeIndex in ..Layer.InputNodeCount) {
                // partial derivative cost with respect to weight of current connection
                var derivativeCostWrtWeight = Layer.LastRawInput[inputNodeIndex] * nodeValues[outputNodeIndex];
                CostGradientWeights[inputNodeIndex, outputNodeIndex] += derivativeCostWrtWeight;
            }

            // derivative cost with respect to bias (bias' = 1)
            var derivativeCostWrtBias = 1 * nodeValues[outputNodeIndex];
            CostGradientBiases[outputNodeIndex] += derivativeCostWrtBias;
        }
    }

    public void ApplyGradients(Number learnRate) {
        foreach(int outputNodeIndex in ..Layer.OutputNodeCount) {
            Layer.Biases[outputNodeIndex] -= CostGradientBiases[outputNodeIndex] * learnRate;
            foreach(int inputNodeIndex in ..Layer.InputNodeCount) {
                Layer.Weights[inputNodeIndex, outputNodeIndex] -= CostGradientWeights[inputNodeIndex, outputNodeIndex] * learnRate;
            }
        }
    }

    public void ResetGradients() {
        foreach(var outputNodeIndex in ..Layer.OutputNodeCount) {
            CostGradientBiases[outputNodeIndex] = 0;
            foreach(var inputNodeIndex in ..Layer.InputNodeCount) {
                CostGradientWeights[inputNodeIndex, outputNodeIndex] = 0;
            }
        }
    }

    public Number[] CalculateOutputLayerNodeValues(Number[] expected) {
        var nodeValues = new Number[expected.Length];

        var activationDerivatives = Layer.ActivationMethod.Derivative(Layer.LastWeightedInput); // can i derive in-place?
        foreach(int i in ..expected.Length) {
            var costDerivative = CostFunction.Derivative(Layer.LastActivatedWeights[i], expected[i]);
            nodeValues[i] = costDerivative * activationDerivatives[i];
        }

        return nodeValues;
    }

    public Number[] CalculateHiddenLayerNodeValues(RecordingLayer oldLayer, Number[] oldNodeValues) {
        var newNodeValues = new Number[Layer.OutputNodeCount];
        var derivatives = Layer.ActivationMethod.Derivative(Layer.LastWeightedInput);

        foreach(int newNodeIndex in ..newNodeValues.Length) {
            var newNodeValue = 0d;

            foreach(int oldNodeIndex in ..oldNodeValues.Length) {
                // Partial derivative of the weighted input with respect to the input
                var weightedInputDerivative = oldLayer.Weights[newNodeIndex, oldNodeIndex];
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }

            newNodeValue *= derivatives[newNodeIndex];
            newNodeValues[newNodeIndex] = newNodeValue;
        }

        return newNodeValues;
    }
}
