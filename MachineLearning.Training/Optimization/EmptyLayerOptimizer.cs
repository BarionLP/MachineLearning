using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Training.Optimization;

public sealed class EmptyLayerOptimizer : ILayerOptimizer
{
    public static EmptyLayerOptimizer Instance { get; } = new();

    public void Apply(IGradients gradients) { }
    public void FullReset() { }
    public void GradientCostReset() { }
    public void Update(Vector nodeValues, ILayerSnapshot snapshot, IGradients gradients) { }
}