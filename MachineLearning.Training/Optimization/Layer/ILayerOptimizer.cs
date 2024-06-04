using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.Layer;

public interface ILayerOptimizer
{
    public RecordingLayer Layer { get; }
    public ICostFunction CostFunction { get; }
    public void Update(Vector nodeValues);
    public void Apply(int dataCounter);
    public void GradientCostReset();
    public void FullReset();

    public Vector ComputeOutputLayerErrors(Vector expected)
    {
        var activationDerivatives = Layer.ActivationFunction.Derivative(Layer.LastWeightedInput);
        var costDerivatives = CostFunction.Derivative(Layer.LastActivatedWeights, expected);
        #if DEBUG
        if(activationDerivatives.AsSpan().Contains(double.NaN))
        {
            Console.WriteLine();
        }
        if(costDerivatives.AsSpan().Contains(double.NaN))
        {
            Console.WriteLine();
        }
        #endif
        costDerivatives.PointwiseMultiplyInPlace(activationDerivatives);

        return costDerivatives;
    }

    public Vector ComputeHiddenLayerErrors(RecordingLayer nextLayer, Vector nextErrors)
    {
        var activationDerivatives = Layer.ActivationFunction.Derivative(Layer.LastWeightedInput);
        var weightedInputDerivatives = nextErrors.Multiply(nextLayer.Weights); // other option? cannot be simded
        weightedInputDerivatives.PointwiseMultiplyInPlace(activationDerivatives);

        return weightedInputDerivatives; // contains now the error values (weightedInputDerivatives*activationDerivatives)
    }
}
