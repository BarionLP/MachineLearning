using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public sealed class EmptyLayerOptimizer : ILayerOptimizer
{
    public static EmptyLayerOptimizer Instance { get; } = new();
    public ICostFunction CostFunction => null!;

    public void Apply(int dataCounter) { }
    public void FullReset() { }
    public void GradientCostReset() { }
    public void Update(Vector nodeValues, ILayerSnapshot snapshot) { }
}