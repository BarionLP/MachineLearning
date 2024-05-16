using MachineLearning.Domain;
using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.Layer;

public interface ILayerOptimizer<TWeight> where TWeight : struct, IEquatable<TWeight>, IFormattable
{
    public RecordingLayer Layer { get; }
    public ICostFunction CostFunction { get; }
    public void Update(Vector<TWeight> nodeValues);
    public void Apply(int dataCounter);
    public void GradientCostReset();
    public void FullReset();

    public Vector<double> ComputeOutputLayerErrors(Vector<double> expected)
    {
        var activationDerivatives = Layer.ActivationFunction.Derivative(Layer.LastWeightedInput);
        var costDerivatives = CostFunction.Derivative(Layer.LastActivatedWeights, expected);
        costDerivatives.PointwiseMultiplyInPlace(activationDerivatives);

        return costDerivatives;
    }

    public Vector<double> ComputeHiddenLayerErrors(RecordingLayer nextLayer, Vector<double> nextErrors)
    {
        var activationDerivatives = Layer.ActivationFunction.Derivative(Layer.LastWeightedInput);
        var weightedInputDerivatives = nextErrors * nextLayer.Weights; // TransposeThisAndMultiply ??
        weightedInputDerivatives.PointwiseMultiplyInPlace(activationDerivatives);

        return weightedInputDerivatives; // contains now the error values (weightedInputDerivatives*activationDerivatives)
    }
}
