using Simple.Network.Layer;
using Simple.Training.Cost;

namespace Simple.Training.Optimization.Layer;

public sealed class GDMomentumLayerOptimizer: ILayerOptimizer<Number> {
    public RecordingLayer Layer { get; }
    public readonly Number[,] CostGradientWeights;
    public readonly Number[] CostGradientBiases;
    public readonly Number[,] WeightVelocities;
    public readonly Number[] BiasVelocities;
    public ICostFunction CostFunction { get; }

    public GDMomentumLayerOptimizer(RecordingLayer layer, ICostFunction costFunction) {
        Layer = layer;
        CostFunction = costFunction;
        CostGradientWeights = new Number[Layer.InputNodeCount, Layer.OutputNodeCount];
        CostGradientBiases = new Number[Layer.OutputNodeCount];
        WeightVelocities = new Number[Layer.InputNodeCount, Layer.OutputNodeCount];
        BiasVelocities = new Number[Layer.OutputNodeCount];
    }

    public void Update(Number[] nodeValues){
        foreach (int outputNodeIndex in ..Layer.OutputNodeCount) {
            foreach (int inputNodeIndex in ..Layer.InputNodeCount) {
                // partial derivative cost with respect to weight of current connection
                var derivativeCostWrtWeight = Layer.LastRawInput[inputNodeIndex] * nodeValues[outputNodeIndex];
                CostGradientWeights[inputNodeIndex, outputNodeIndex] += derivativeCostWrtWeight;
            }

            // derivative cost with respect to bias (bias' = 1)
            var derivativeCostWrtBias = 1 * nodeValues[outputNodeIndex];
            CostGradientBiases[outputNodeIndex] += derivativeCostWrtBias;
        }
    }

    public void Apply(GDMomentumOptimizer optimizer, int dataCounter) {
        var averagedLearningRate = optimizer.CurrentLearningRate/dataCounter;
        var weightDecay = 1 - optimizer.Regularization * averagedLearningRate; //used against overfitting

        foreach (int outputNodeIndex in ..Layer.OutputNodeCount) {
            var biasVelocity = BiasVelocities[outputNodeIndex] * optimizer.Momentum - CostGradientBiases[outputNodeIndex] * averagedLearningRate;
            BiasVelocities[outputNodeIndex] = biasVelocity;
            Layer.Biases[outputNodeIndex] += biasVelocity;
            CostGradientBiases[outputNodeIndex] = 0;
            //Layer.Biases[outputNodeIndex] -= CostGradientBiases[outputNodeIndex] * learnRate; //old

            foreach (int inputNodeIndex in ..Layer.InputNodeCount) {
                var weight = Layer.Weights[inputNodeIndex, outputNodeIndex];
                var weightVelocity = WeightVelocities[inputNodeIndex, outputNodeIndex] * optimizer.Momentum - CostGradientWeights[inputNodeIndex, outputNodeIndex] * averagedLearningRate;
                WeightVelocities[inputNodeIndex, outputNodeIndex] = weightVelocity;
                Layer.Weights[inputNodeIndex, outputNodeIndex] = weight * weightDecay + weightVelocity;
                //Layer.Weights[inputNodeIndex, outputNodeIndex] -= CostGradientWeights[inputNodeIndex, outputNodeIndex] * learnRate; //old
            }
        }
    }

    public void Reset() {
        foreach (var outputNodeIndex in ..Layer.OutputNodeCount) {
            CostGradientBiases[outputNodeIndex] = 0;
            foreach (var inputNodeIndex in ..Layer.InputNodeCount) {
                CostGradientWeights[inputNodeIndex, outputNodeIndex] = 0;
            }
        }
    }

    public Number[] CalculateOutputLayerNodeValues(Number[] expected) {
        var nodeValues = new Number[expected.Length];

        var activationDerivatives = Layer.ActivationMethod.Derivative(Layer.LastWeightedInput); // can i derive in-place?
        foreach (int i in ..expected.Length) {
            // Evaluate partial derivatives for current node: cost/activation & activation/weightedInput
            var costDerivative = CostFunction.Derivative(Layer.LastActivatedWeights[i], expected[i]);
            nodeValues[i] = costDerivative * activationDerivatives[i];
        }

        return nodeValues;
    }

    public Number[] CalculateHiddenLayerNodeValues(RecordingLayer oldLayer, Number[] oldNodeValues) {
        var newNodeValues = new Number[Layer.OutputNodeCount];
        var derivatives = Layer.ActivationMethod.Derivative(Layer.LastWeightedInput);

        foreach (int newNodeIndex in ..newNodeValues.Length) {
            var newNodeValue = 0d;

            foreach (int oldNodeIndex in ..oldNodeValues.Length) {
                // Partial derivative of the weighted input with respect to the input
                var weightedInputDerivative = oldLayer.Weights[newNodeIndex, oldNodeIndex];
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }

            newNodeValue *= derivatives[newNodeIndex];
            newNodeValues[newNodeIndex] = newNodeValue;
        }

        return newNodeValues;
    }

    void ILayerOptimizer<double>.Apply(IOptimizerConfig<double> optimizer, int dataCounter) => Apply((GDMomentumOptimizer)optimizer, dataCounter);
}