using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public interface ILayerOptimizer
{
    public SimpleLayer Layer { get; }
    public ICostFunction CostFunction { get; }
    public void Update(Vector nodeValues, LayerSnapshot snapshot);
    public void Apply(int dataCounter);
    public void GradientCostReset();
    public void FullReset();

    public Vector ComputeOutputLayerErrors(Vector expected, LayerSnapshot snapshot)
    {
        var activationDerivatives = Layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);
        var costDerivatives = CostFunction.Derivative(snapshot.LastActivatedWeights, expected);
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

    public Vector ComputeHiddenLayerErrors(SimpleLayer nextLayer, Vector nextErrors, LayerSnapshot snapshot)
    {
        var activationDerivatives = Layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);
        var weightedInputDerivatives = nextErrors.Multiply(nextLayer.Weights); // other option? cannot be simded
        weightedInputDerivatives.PointwiseMultiplyInPlace(activationDerivatives);

        return weightedInputDerivatives; // contains now the error values (weightedInputDerivatives*activationDerivatives)
    }
}
