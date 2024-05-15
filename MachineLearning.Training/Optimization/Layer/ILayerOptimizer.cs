using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;
using System.Numerics;

namespace MachineLearning.Training.Optimization.Layer;

public interface ILayerOptimizer<TData>
{
    public RecordingLayer Layer { get; }
    public ICostFunction CostFunction { get; }
    public void Update(TData[] nodeValues);
    public void Apply(int dataCounter);
    public void GradientCostReset();
    public void FullReset();

    public double[] CalculateOutputLayerNodeValues(double[] expected)
    {
        var nodeValues = new double[expected.Length];

        var activationDerivatives = Layer.ActivationMethod.Derivative(Layer.LastWeightedInput);
        foreach (int i in ..expected.Length)
        {
            // Evaluate partial derivatives for current node: cost/activation & activation/weightedInput
            var costDerivative = CostFunction.Derivative(Layer.LastActivatedWeights[i], expected[i]);
            nodeValues[i] = costDerivative * activationDerivatives[i];
        }

        return nodeValues;
    }

    public double[] CalculateHiddenLayerNodeValues(RecordingLayer oldLayer, double[] oldNodeValues)
    {
        var newNodeValues = new double[Layer.OutputNodeCount];
        var derivatives = Layer.ActivationMethod.Derivative(Layer.LastWeightedInput);

        foreach (int newNodeIndex in ..newNodeValues.Length)
        {
            var newNodeValue = 0d;

            foreach (int oldNodeIndex in ..oldNodeValues.Length)
            {
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
