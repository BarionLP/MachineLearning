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
                // partial derivative cost / weight of current connection
                var derivativeCostWrtWeight = Layer.LastInput[inputNodeIndex] * nodeValues[outputNodeIndex];
                CostGradientWeights[inputNodeIndex, outputNodeIndex] += derivativeCostWrtWeight;
            }

            // derivative cost / bias (bias' = 1)
            var derivativeCostWrtBias = 1 * nodeValues[outputNodeIndex];
            CostGradientBiases[outputNodeIndex] += derivativeCostWrtBias;
        }
    }

    public void ApplyGradients(Number learnRate) {
        foreach(int outputNodeIndex in ..Layer.OutputNodeCount) {
            Layer.Biases[outputNodeIndex] -= CostGradientBiases[outputNodeIndex] * learnRate;
            foreach(int inputNodeIndex in ..Layer.InputNodeCount) {
                Layer.Weights[inputNodeIndex, outputNodeIndex] -= CostGradientWeights[inputNodeIndex, outputNodeIndex] * learnRate; //was assignment in video! mistake?
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

        foreach(int i in ..expected.Length) {
            var costDerivative = CostFunction.Derivative(Layer.ActivatedWeights[i], expected[i]);
            var activationDerivative = Layer.ActivationMethod.Derivative(Layer.WeightedInput[i]);
            nodeValues[i] = costDerivative * activationDerivative;
        }

        return nodeValues;
    }

    public Number[] CalculateHiddenLayerNodeValues(RecordingLayer oldLayer, Number[] oldNodeValues) {
        var newNodeValues = new Number[Layer.OutputNodeCount];

        foreach(int newNodeIndex in ..newNodeValues.Length) {
            var newNodeValue = 0d;

            foreach(int oldNodeIndex in ..oldNodeValues.Length) {
                // Partial derivative of the weighted input with respect to the input
                var weightedInputDerivative = oldLayer.Weights[newNodeIndex, oldNodeIndex];
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }

            newNodeValue *= Layer.ActivationMethod.Derivative(Layer.WeightedInput[newNodeIndex]);
            newNodeValues[newNodeIndex] = newNodeValue;
        }

        return newNodeValues;
    }
}
