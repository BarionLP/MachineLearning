using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Training.Optimization;

public sealed class EmptyLayerOptimizer : ILayerOptimizer
{
    public static EmptyLayerOptimizer Instance { get; } = new();

    public void Apply(IGradients gradients) { }
    public void FullReset() { }
    public void GradientCostReset() { }
    public Vector Update(Vector outputGradient, ILayerSnapshot snapshot, IGradients gradients) => outputGradient;
}